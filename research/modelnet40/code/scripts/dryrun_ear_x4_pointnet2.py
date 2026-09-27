#!/usr/bin/env python3
"""CPU dry-run for EAR ×4 PointNet++ experiments (Line A 4096 / Line B 2048)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from modelnet_npy_dataloader import ModelNetNPYDataset  # noqa: E402

EXPECTED_COUNTS = {"train": 9843, "test": 2468}


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run EAR ×4 PointNet++ dataloader.")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--num-point", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    data_root = (PROJECT_ROOT / args.data_root).resolve() if not args.data_root.is_absolute() else args.data_root.resolve()
    if not data_root.is_dir():
        print(f"ERROR: data root missing: {data_root}", file=sys.stderr)
        return 1

    sample_npy = next((data_root / "train").rglob("*.npy"))
    raw = np.load(sample_npy)

    print(f"experiment: {args.variant}")
    print(f"dataset path: {data_root}")
    print(f"sample npy: {sample_npy}")
    print(f"raw npy shape: {raw.shape}")
    print(f"num_point: {args.num_point}")
    print(f"allow_resample: false")

    issues: list[str] = []
    for split in ("train", "test"):
        ds = ModelNetNPYDataset(
            data_root,
            split=split,
            num_points=args.num_point,
            normalize=True,
            allow_resample=False,
        )
        count = len(ds)
        expected = EXPECTED_COUNTS[split]
        print(f"{split} count: {count} (expected {expected})")
        if count != expected:
            issues.append(f"{split} count {count} != {expected}")

    train_ds = ModelNetNPYDataset(
        data_root, split="train", num_points=args.num_point, allow_resample=False
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    points, labels = next(iter(loader))

    print(f"batch shape (B x N x 3): {tuple(points.shape)}")
    print(f"batch shape (B x 3 x N): ({points.shape[0]}, 3, {points.shape[1]})")
    print(f"point count per sample: {points.shape[1]}")
    print(f"label shape: {tuple(labels.shape)}")
    print(f"num_classes: {len(train_ds.class_to_idx)}")
    print(f"finite points: {torch.isfinite(points).all().item()}")
    print(f"NaN/Inf in batch: {not torch.isfinite(points).all().item()}")

    if points.shape[1] != args.num_point:
        issues.append(f"batch points {points.shape[1]} != num_point {args.num_point}")
    if len(train_ds.class_to_idx) != 40:
        issues.append(f"num_classes {len(train_ds.class_to_idx)} != 40")
    if not torch.isfinite(points).all():
        issues.append("batch contains NaN/Inf")

    if issues:
        print("Dry-run FAILED:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    print("Dry-run PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
