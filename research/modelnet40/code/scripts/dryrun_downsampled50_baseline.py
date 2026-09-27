#!/usr/bin/env python3
"""CPU dry-run for Downsampled50 baseline: verify 512 -> 1024 loader path."""

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
    parser = argparse.ArgumentParser(description="Dry-run Downsampled50 -> PointNet++ loader.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "modelnet40_downsampled50",
    )
    parser.add_argument("--variant", type=str, default="downsampled50_baseline")
    parser.add_argument("--num-point", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        print(f"ERROR: data root missing: {data_root}", file=sys.stderr)
        return 1

    sample_npy = next((data_root / "train").rglob("*.npy"))
    raw = np.load(sample_npy)
    print(f"variant: {args.variant}")
    print(f"data_root: {data_root}")
    print(f"sample npy: {sample_npy}")
    print(f"raw npy shape: {raw.shape}")

    dataset = ModelNetNPYDataset(
        data_root,
        split="train",
        num_points=args.num_point,
        normalize=True,
        allow_resample=True,
        resample_seed=42,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    points, labels = next(iter(loader))

    print(f"loader output shape: {tuple(points.shape)}")
    print(f"label shape: {tuple(labels.shape)}")
    print(f"num classes in label map: {len(dataset.class_to_idx)}")
    print(f"finite points: {torch.isfinite(points).all().item()}")
    print(f"dataset length (train): {len(dataset)}")

    test_dataset = ModelNetNPYDataset(data_root, split="test", num_points=args.num_point)
    print(f"dataset length (test): {len(test_dataset)}")

    expected_output = PROJECT_ROOT / "outputs" / args.variant
    expected_log = PROJECT_ROOT / "logs" / f"{args.variant}.log"
    print(f"expected output dir: {expected_output}")
    print(f"expected log file: {expected_log}")
    print("Dry-run PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
