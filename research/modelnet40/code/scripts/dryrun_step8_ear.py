#!/usr/bin/env python3
"""Step 8 dry-run: verify Downsampled50+EAR dataloader with native 1024 points."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 8 dry-run: Downsampled50+EAR PointNet++ loader.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_ear",
    )
    parser.add_argument("--variant", type=str, default="downsampled50_ear_pointnet2")
    parser.add_argument("--num-point", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--allow-resample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Must be false for EAR: do not silently crop/pad.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"ERROR: data root missing: {data_root}", file=sys.stderr)
        return 1

    is_symlink = data_root.is_symlink()
    print(f"variant: {args.variant}")
    print(f"dataset path: {data_root}")
    print(f"is_symlink: {is_symlink}")
    if is_symlink:
        print(f"symlink target: {data_root.resolve()}")

    sample_npy = next((data_root / "train").rglob("*.npy"))
    raw = np.load(sample_npy)
    print(f"sample npy: {sample_npy}")
    print(f"raw npy shape: {raw.shape}")

    train_dataset = ModelNetNPYDataset(
        data_root,
        split="train",
        num_points=args.num_point,
        normalize=True,
        allow_resample=args.allow_resample,
        resample_seed=42,
    )
    test_dataset = ModelNetNPYDataset(
        data_root,
        split="test",
        num_points=args.num_point,
        normalize=True,
        allow_resample=args.allow_resample,
        resample_seed=42,
    )

    print(f"train count: {len(train_dataset)}")
    print(f"test count: {len(test_dataset)}")
    print(f"num classes: {len(train_dataset.class_to_idx)}")
    print(f"num_point (config): {args.num_point}")
    print(f"allow_resample: {args.allow_resample}")

    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    points, labels = next(iter(loader))

    print(f"batch shape (points): {tuple(points.shape)}")
    print(f"point count per sample: {points.shape[1]}")
    print(f"label shape: {tuple(labels.shape)}")
    print(f"finite points: {torch.isfinite(points).all().item()}")

    # Verify no silent resampling: loader output must match raw on-disk count
    on_disk_counts = set()
    for i in range(min(20, len(train_dataset))):
        npy_path = train_dataset.samples[i][0]
        on_disk_counts.add(np.load(npy_path).shape[0])
    loader_counts = {points[i].shape[0] for i in range(points.shape[0])}
    print(f"on_disk point counts (first 20 samples): {sorted(on_disk_counts)}")
    print(f"loader output point counts (batch): {sorted(loader_counts)}")

    if on_disk_counts != {args.num_point}:
        print(f"WARNING: on-disk counts {on_disk_counts} != num_point {args.num_point}", file=sys.stderr)
        if not args.allow_resample:
            print("ERROR: allow_resample=false but counts differ", file=sys.stderr)
            return 1

    if points.shape[1] != args.num_point:
        print(f"ERROR: batch point dim {points.shape[1]} != num_point {args.num_point}", file=sys.stderr)
        return 1

    print("Dry-run PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
