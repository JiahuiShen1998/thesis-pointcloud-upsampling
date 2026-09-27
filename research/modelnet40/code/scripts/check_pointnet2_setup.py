#!/usr/bin/env python3
"""Smoke-test PointNet++ setup: imports, dataloader, one forward pass."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PN2_ROOT = PROJECT_ROOT / "external" / "Pointnet_Pointnet2_pytorch"
sys.path.insert(0, str(PN2_ROOT))
sys.path.insert(0, str(PN2_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from modelnet_npy_dataloader import ModelNetNPYDataset  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "modelnet40_original",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--use-cpu", action="store_true")
    args = parser.parse_args()

    if not args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PointNet++ repo: {PN2_ROOT}")
    assert PN2_ROOT.is_dir(), "PointNet++ repo missing"

    dataset = ModelNetNPYDataset(args.data_root, split="train", num_points=1024)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    points, labels = next(iter(loader))
    print(f"Batch shape: points={tuple(points.shape)} labels={tuple(labels.shape)}")

    model_module = importlib.import_module("pointnet2_cls_ssg")
    model = model_module.get_model(40, normal_channel=False)
    if not args.use_cpu and torch.cuda.is_available():
        model = model.cuda()
        points = points.cuda()
        labels = labels.cuda()

    points = points.transpose(2, 1)
    pred, _ = model(points)
    print(f"Forward OK: pred shape={tuple(pred.shape)}")
    print("PointNet++ setup smoke test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
