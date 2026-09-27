#!/usr/bin/env python3
"""Export RPN proposals and features without importing the legacy evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
import _init_path  # noqa: F401,E402

from lib.config import cfg, cfg_from_file
from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset
from lib.net.point_rcnn import PointRCNN
from tools.train_utils import train_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--rpn-ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch-label", default="selected")
    parser.add_argument("--workers", type=int, default=0)
    return parser.parse_args()


def save_roi_file(path: Path, boxes: np.ndarray, scores: np.ndarray) -> None:
    with path.open("w", encoding="ascii") as stream:
        for box, score in zip(boxes, scores):
            x, y, z, h, w, length, ry = box.tolist()
            stream.write(
                f"Car -1 -1 0 0 0 0 0 {h:.4f} {w:.4f} {length:.4f} "
                f"{x:.4f} {y:.4f} {z:.4f} {ry:.4f} {float(score):.6f}\n"
            )


def main() -> None:
    args = parse_args()
    lidar_dir = args.lidar_dir.resolve()
    split_file = args.split_file.resolve()
    rpn_ckpt = args.rpn_ckpt.resolve()
    output_dir = args.output_dir.resolve()
    frame_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]

    actual_ids = {path.stem for path in lidar_dir.glob("*.bin")}
    if actual_ids != set(frame_ids):
        raise RuntimeError(
            f"LiDAR/split mismatch: expected={len(frame_ids)} actual={len(actual_ids)}"
        )
    if not rpn_ckpt.is_file():
        raise FileNotFoundError(rpn_ckpt)

    if not args.epoch_label or "/" in args.epoch_label or ".." in args.epoch_label:
        raise ValueError("--epoch-label must be a simple non-empty path component")
    feature_dir = output_dir / f"eval/epoch_{args.epoch_label}/train/features"
    roi_dir = output_dir / f"eval/epoch_{args.epoch_label}/train/detections/data"
    feature_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "export_protocol.json").write_text(
        json.dumps(
            {
                "lidar_dir": str(lidar_dir),
                "split_file": str(split_file),
                "frame_count": len(frame_ids),
                "rpn_ckpt": str(rpn_ckpt),
                "epoch_label": args.epoch_label,
                "point_count": 16384,
                "seed": 1024,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    np.random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)
    cfg_from_file(str(TOOLS_DIR / "cfgs/default.yaml"))
    cfg.RPN.ENABLED = True
    cfg.RCNN.ENABLED = False
    cfg.RPN.NUM_POINTS = 16384
    cfg.RPN.LOC_XZ_FINE = False

    dataset = KittiRCNNDataset(
        root_dir=str(PROJECT_ROOT / "data"),
        npoints=cfg.RPN.NUM_POINTS,
        split="train",
        mode="EVAL",
        random_select=True,
        classes=cfg.CLASSES,
        logger=_Logger(),
    )
    dataset.image_idx_list = list(frame_ids)
    dataset.sample_id_list = [int(frame_id) for frame_id in frame_ids]
    dataset.num_sample = len(frame_ids)
    dataset.lidar_dir = str(lidar_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=dataset.collate_batch,
    )

    model = PointRCNN(num_classes=dataset.num_class, use_xyz=True, mode="TEST").cuda()
    train_utils.load_checkpoint(model, filename=str(rpn_ckpt))
    model.eval()

    with torch.no_grad():
        for index, data in enumerate(loader, start=1):
            sample_id = int(data["sample_id"][0])
            pts_features = data["pts_features"][0]
            pts_input = torch.from_numpy(data["pts_input"]).cuda(non_blocking=True).float()
            result = model({"pts_input": pts_input})

            rpn_scores_raw = result["rpn_cls"][:, :, 0]
            seg_result = (torch.sigmoid(rpn_scores_raw) > cfg.RPN.SCORE_THRESH).float()
            backbone_xyz = result["backbone_xyz"]
            backbone_features = result["backbone_features"]
            rois, roi_scores_raw = model.rpn.proposal_layer(
                rpn_scores_raw, result["rpn_reg"], backbone_xyz
            )

            stem = f"{sample_id:06d}"
            np.save(feature_dir / f"{stem}.npy", backbone_features[0].cpu().numpy().T)
            np.save(feature_dir / f"{stem}_xyz.npy", backbone_xyz[0].cpu().numpy())
            np.save(feature_dir / f"{stem}_seg.npy", seg_result[0].cpu().numpy())
            np.save(feature_dir / f"{stem}_rawscore.npy", rpn_scores_raw[0].cpu().numpy())
            np.save(feature_dir / f"{stem}_intensity.npy", pts_features[:, 0])
            save_roi_file(
                roi_dir / f"{stem}.txt",
                rois[0].cpu().numpy(),
                roi_scores_raw[0].cpu().numpy(),
            )
            if index == 1 or index % 128 == 0 or index == len(frame_ids):
                print(f"RPN_EXPORT {index}/{len(frame_ids)} {stem}", flush=True)

    feature_count = len(list(feature_dir.glob("*_xyz.npy")))
    roi_count = len(list(roi_dir.glob("*.txt")))
    if feature_count != len(frame_ids) or roi_count != len(frame_ids):
        raise RuntimeError(
            f"incomplete RPN export: features={feature_count} rois={roi_count} "
            f"expected={len(frame_ids)}"
        )
    (output_dir / "export_complete.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "frames": len(frame_ids),
                "rois": roi_count,
                "rpn_ckpt": str(rpn_ckpt),
                "epoch_label": args.epoch_label,
                "feature_dir": str(feature_dir),
                "roi_dir": str(roi_dir),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


class _Logger:
    def info(self, message: str) -> None:
        print(message, flush=True)

    def warning(self, message: str) -> None:
        print(message, flush=True)


if __name__ == "__main__":
    main()
