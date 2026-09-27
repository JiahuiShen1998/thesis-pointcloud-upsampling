#!/usr/bin/env python3
"""CPU dry-run for Line B main baseline: native 512 pts, no resample."""

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
    parser = argparse.ArgumentParser(description="Dry-run native512 Line B baseline loader.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "modelnet40_downsampled50",
    )
    parser.add_argument("--variant", type=str, default="downsampled50_native512_pointnet2")
    parser.add_argument("--num-point", type=int, default=512)
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
        allow_resample=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    points, labels = next(iter(loader))

    print(f"loader output shape: {tuple(points.shape)}")
    print(f"label shape: {tuple(labels.shape)}")
    print(f"num classes in label map: {len(dataset.class_to_idx)}")
    print(f"finite points: {torch.isfinite(points).all().item()}")
    print(f"allow_resample: False")
    print(f"dataset length (train): {len(dataset)}")

    test_dataset = ModelNetNPYDataset(
        data_root, split="test", num_points=args.num_point, allow_resample=False
    )
    print(f"dataset length (test): {len(test_dataset)}")

    if points.shape[1] != 512:
        print(f"ERROR: expected 512 points, got {points.shape[1]}", file=sys.stderr)
        return 1

    print("Dry-run PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
