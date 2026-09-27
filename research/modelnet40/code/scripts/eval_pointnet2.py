#!/usr/bin/env python3
"""Evaluate a trained PointNet++ checkpoint on ModelNet40 .npy data."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PN2_ROOT = PROJECT_ROOT / "external" / "Pointnet_Pointnet2_pytorch"
sys.path.insert(0, str(PN2_ROOT))
sys.path.insert(0, str(PN2_ROOT / "models"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from modelnet_npy_dataloader import ModelNetNPYDataset  # noqa: E402
from train_pointnet2 import evaluate, write_result_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PointNet++ ModelNet40 evaluation")
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--model", type=str, default="pointnet2_cls_ssg")
    parser.add_argument("--num-category", type=int, default=40)
    parser.add_argument("--num-point", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or (PROJECT_ROOT / "outputs" / args.variant)
    log_file = args.log_file or (PROJECT_ROOT / "logs" / f"{args.variant}_eval.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("eval_pointnet2")

    if not args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_dataset = ModelNetNPYDataset(args.data_root, split="test", num_points=args.num_point)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_module = importlib.import_module(args.model)
    classifier = model_module.get_model(args.num_category, normal_channel=False)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(args.checkpoint, map_location="cpu" if args.use_cpu else None)
    classifier.load_state_dict(checkpoint["model_state_dict"])
    best_epoch = int(checkpoint.get("epoch", -1))

    overall_acc, class_acc, per_class = evaluate(
        classifier,
        test_loader,
        args.num_category,
        args.use_cpu,
    )
    logger.info("Eval overall=%.4f class=%.4f", overall_acc, class_acc)

    idx_to_class_path = args.data_root / "metadata" / "idx_to_class.json"
    with open(idx_to_class_path, encoding="utf-8") as handle:
        idx_to_class = json.load(handle)

    write_result_files(
        output_dir=output_dir,
        variant=args.variant,
        overall_accuracy=overall_acc,
        class_accuracy=class_acc,
        per_class=per_class,
        idx_to_class=idx_to_class,
        checkpoint_path=args.checkpoint,
        log_path=log_file,
        best_epoch=best_epoch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
