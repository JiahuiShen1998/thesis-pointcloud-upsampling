#!/usr/bin/env python3
"""Adapt KITTI CenterPoint on a full-split LiDAR input variant."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET = PROJECT_ROOT / "external/OpenPCDet"
TOOLS = OPENPCDET / "tools"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-ckpt", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--ckpt-save-interval", type=int, default=None)
    parser.add_argument("--max-ckpt-save-num", type=int, default=30)
    args = parser.parse_args()

    lidar_dir = args.lidar_dir.resolve()
    split_file = args.split_file.resolve()
    output_dir = args.output_dir.resolve()
    init_ckpt = args.init_ckpt.resolve()
    ckpt_save_interval = args.ckpt_save_interval or args.epochs
    if ckpt_save_interval < 1:
        raise ValueError("ckpt-save-interval must be positive")
    if args.max_ckpt_save_num < 1:
        raise ValueError("max-ckpt-save-num must be positive")
    frame_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    expected = set(frame_ids)
    actual = {path.stem for path in lidar_dir.glob("*.bin")}
    if len(frame_ids) != len(expected):
        raise RuntimeError(f"duplicate frame IDs in {split_file}")
    if actual != expected:
        raise RuntimeError(
            f"LiDAR/split mismatch: expected={len(expected)} actual={len(actual)} "
            f"missing={sorted(expected - actual)[:10]} extra={sorted(actual - expected)[:10]}"
        )
    if not init_ckpt.is_file():
        raise FileNotFoundError(init_ckpt)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_link = OPENPCDET / "output/kitti_models/centerpoint" / args.tag
    output_link.parent.mkdir(parents=True, exist_ok=True)
    if output_link.is_symlink():
        if output_link.resolve() != output_dir:
            raise RuntimeError(f"output link points elsewhere: {output_link}")
    elif output_link.exists():
        raise RuntimeError(f"refusing to replace existing output path: {output_link}")
    else:
        output_link.symlink_to(output_dir, target_is_directory=True)

    protocol = {
        "tag": args.tag,
        "lidar_dir": str(lidar_dir),
        "split_file": str(split_file),
        "frame_count": len(frame_ids),
        "init_ckpt": str(init_ckpt),
        "epochs": args.epochs,
        "workers": args.workers,
        "learning_rate": args.learning_rate,
        "ckpt_save_interval": ckpt_save_interval,
        "max_ckpt_save_num": args.max_ckpt_save_num,
        "gt_sampling": False,
        "seed": 666,
    }
    (output_dir / "adaptation_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )

    lock_file = (Path("/tmp") / f"centerpoint_full_{args.tag}.lock").open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(TOOLS))
    sys.path.insert(0, str(OPENPCDET))

    from pcdet.config import cfg
    import pcdet.config as config_module
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    import test as test_module

    def get_lidar(self, idx):
        path = lidar_dir / f"{idx}.bin"
        if not path.is_file():
            raise FileNotFoundError(path)
        return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)

    KittiDataset.get_lidar = get_lidar
    original_cfg_loader = config_module.cfg_from_yaml_file

    def cfg_from_yaml_file(config_file, config):
        result = original_cfg_loader(config_file, config)
        config.DATA_CONFIG.DATA_AUGMENTOR.DISABLE_AUG_LIST = ["gt_sampling"]
        config.OPTIMIZATION.LR = args.learning_rate
        return result

    config_module.cfg_from_yaml_file = cfg_from_yaml_file
    test_module.repeat_eval_ckpt = lambda *positional, **keywords: None

    sys.argv = [
        "train.py",
        "--cfg_file",
        "cfgs/kitti_models/centerpoint.yaml",
        "--batch_size",
        "2",
        "--epochs",
        str(args.epochs),
        "--workers",
        str(args.workers),
        "--fix_random_seed",
        "--extra_tag",
        args.tag,
        "--pretrained_model",
        str(init_ckpt),
        "--ckpt_save_interval",
        str(ckpt_save_interval),
        "--max_ckpt_save_num",
        str(args.max_ckpt_save_num),
        "--num_epochs_to_eval",
        "0",
        "--wo_gpu_stat",
    ]
    os.chdir(TOOLS)
    runpy.run_path("train.py", run_name="__main__")
    checkpoint = output_dir / "ckpt" / f"checkpoint_epoch_{args.epochs}.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"CenterPoint checkpoint was not created: {checkpoint}")
    print(f"CENTERPOINT_FULL_ADAPTATION_PASS checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
