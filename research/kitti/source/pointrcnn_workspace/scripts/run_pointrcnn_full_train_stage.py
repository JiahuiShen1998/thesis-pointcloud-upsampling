#!/usr/bin/env python3
"""Run one deterministic full-split PointRCNN adaptation stage."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
SEED = 20260823


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("rpn", "rcnn", "rcnn_offline"), required=True)
    parser.add_argument("--lidar-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-ckpt", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ckpt-save-interval", type=int)
    parser.add_argument("--resume-ckpt", type=Path)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rcnn-training-roi-dir", type=Path)
    parser.add_argument("--rcnn-training-feature-dir", type=Path)
    return parser.parse_args()


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    lidar_dir = args.lidar_dir.resolve()
    split_file = args.split_file.resolve()
    output_dir = args.output_dir.resolve()
    init_ckpt = args.init_ckpt.resolve()
    resume_ckpt = args.resume_ckpt.resolve() if args.resume_ckpt else None
    expected_ids = read_split(split_file)
    ckpt_save_interval = args.ckpt_save_interval or args.epochs

    if ckpt_save_interval <= 0:
        raise ValueError("--ckpt-save-interval must be positive")

    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError(f"split contains duplicate frame IDs: {split_file}")
    actual_ids = {path.stem for path in lidar_dir.glob("*.bin")}
    expected_set = set(expected_ids)
    if actual_ids != expected_set:
        missing = sorted(expected_set - actual_ids)[:10]
        extra = sorted(actual_ids - expected_set)[:10]
        raise RuntimeError(
            f"LiDAR/split mismatch: expected={len(expected_set)} actual={len(actual_ids)} "
            f"missing={missing} extra={extra}"
        )
    if not init_ckpt.is_file():
        raise FileNotFoundError(init_ckpt)
    if resume_ckpt is not None and not resume_ckpt.is_file():
        raise FileNotFoundError(resume_ckpt)
    if args.mode == "rcnn_offline":
        if args.rcnn_training_roi_dir is None or args.rcnn_training_feature_dir is None:
            raise ValueError("offline RCNN training requires ROI and feature directories")
        if not args.rcnn_training_roi_dir.is_dir():
            raise FileNotFoundError(args.rcnn_training_roi_dir)
        if not args.rcnn_training_feature_dir.is_dir():
            raise FileNotFoundError(args.rcnn_training_feature_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "mode": args.mode,
        "lidar_dir": str(lidar_dir),
        "split_file": str(split_file),
        "frame_count": len(expected_ids),
        "output_dir": str(output_dir),
        "init_ckpt": str(init_ckpt),
        "epochs": args.epochs,
        "ckpt_save_interval": ckpt_save_interval,
        "resume_ckpt": str(resume_ckpt) if resume_ckpt else None,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "rcnn_training_roi_dir": (
            str(args.rcnn_training_roi_dir.resolve()) if args.rcnn_training_roi_dir else None
        ),
        "rcnn_training_feature_dir": (
            str(args.rcnn_training_feature_dir.resolve()) if args.rcnn_training_feature_dir else None
        ),
        "seed": SEED,
        "gt_database_augmentation": False,
        "optimization": {
            "name": "adam_onecycle",
            "max_learning_rate": 0.0002,
            "initial_learning_rate": 0.00002,
            "final_learning_rate": 0.000000002,
            "adam_betas": [0.9, 0.99],
            "weight_decay": 0.001,
            "momentums": [0.95, 0.85],
            "division_factor": 10.0,
            "pct_start": 0.4,
            "gradient_norm_clip": 1.0,
            "lr_warmup": False,
        },
        "batch_norm_schedule": {
            "initial_momentum": 0.1,
            "decay": 0.5,
            "minimum_momentum": 0.01,
            "decay_epoch_steps": [2],
        },
        "rpn_loc_xz_fine": False,
    }
    (output_dir / "adaptation_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )

    lock_path = Path("/tmp") / f"pointrcnn_full_{output_dir.parent.name}_{args.mode}.lock"
    lock_file = lock_path.open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    sys.path.insert(0, str(TOOLS_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))
    from lib.config import cfg
    import lib.config as config_module
    from lib.datasets.kitti_dataset import KittiDataset
    from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset

    original_dataset_init = KittiDataset.__init__

    def dataset_init(self, root_dir, split="train"):
        original_dataset_init(self, root_dir=root_dir, split=split)
        if split == "train":
            self.image_idx_list = list(expected_ids)
            self.num_sample = len(self.image_idx_list)
            self.lidar_dir = str(lidar_dir)

    KittiDataset.__init__ = dataset_init

    original_rcnn_dataset_init = KittiRCNNDataset.__init__

    def rcnn_dataset_init(self, *positional, **keywords):
        keywords["gt_database_dir"] = None
        original_rcnn_dataset_init(self, *positional, **keywords)

    KittiRCNNDataset.__init__ = rcnn_dataset_init

    original_cfg_from_file = config_module.cfg_from_file

    def cfg_from_file(path):
        original_cfg_from_file(path)
        cfg.GT_AUG_ENABLED = False
        cfg.RPN.LOC_XZ_FINE = False
        cfg.TRAIN.SPLIT = "train"
        cfg.TRAIN.VAL_SPLIT = "patch_causal_pilot256"
        cfg.TEST.SPLIT = "patch_causal_pilot256"
        cfg.TRAIN.LR = 0.0002
        cfg.TRAIN.LR_CLIP = 0.000001
        cfg.TRAIN.DECAY_STEP_LIST = [2]
        cfg.TRAIN.LR_WARMUP = False
        cfg.TRAIN.BN_DECAY_STEP_LIST = [2]
        cfg.TRAIN.OPTIMIZER = "adam_onecycle"

    config_module.cfg_from_file = cfg_from_file

    sys.argv = [
        "train_rcnn.py",
        "--cfg_file",
        "cfgs/default.yaml",
        "--train_mode",
        args.mode,
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--workers",
        str(args.workers),
        "--ckpt_save_interval",
        str(ckpt_save_interval),
        "--output_dir",
        str(output_dir),
    ]
    if resume_ckpt is None:
        sys.argv.extend(["--rpn_ckpt", str(init_ckpt)])
    else:
        # A resumed checkpoint already contains the stage weights and optimizer
        # state.  Passing --rpn_ckpt as well would reload the initialization
        # after the resume and silently discard the resumed model weights.
        sys.argv.extend(["--ckpt", str(resume_ckpt)])
    if args.mode == "rcnn_offline":
        sys.argv.extend(
            [
                "--rcnn_training_roi_dir",
                str(args.rcnn_training_roi_dir.resolve()),
                "--rcnn_training_feature_dir",
                str(args.rcnn_training_feature_dir.resolve()),
            ]
        )
    os.chdir(TOOLS_DIR)
    runpy.run_path("train_rcnn.py", run_name="__main__")


if __name__ == "__main__":
    main()
