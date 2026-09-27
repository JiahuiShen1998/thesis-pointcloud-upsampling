#!/usr/bin/env python3
"""Train PointNet++ classifier on preprocessed ModelNet40 .npy datasets."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
import random
import shutil
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

import provider  # noqa: E402
from modelnet_npy_dataloader import ModelNetNPYDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("PointNet++ ModelNet40 training")
    parser.add_argument("--variant", type=str, default="original_baseline")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "modelnet40_original",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--model", type=str, default="pointnet2_cls_ssg")
    parser.add_argument("--num-category", type=int, default=40)
    parser.add_argument("--num-point", type=int, default=1024)
    parser.add_argument(
        "--allow-resample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If false, reject samples whose on-disk count != num_point (no silent crop/pad).",
    )
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--decay-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--use-cpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=0, help="0 = full epoch")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 = full eval")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inplace_relu(module: torch.nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("ReLU") != -1:
        module.inplace = True


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    num_class: int,
    use_cpu: bool,
    max_batches: int = 0,
) -> tuple[float, float, dict[int, float]]:
    model.eval()
    mean_correct: list[float] = []
    class_acc = np.zeros((num_class, 3))

    with torch.no_grad():
        for batch_idx, (points, target) in enumerate(tqdm(loader, total=len(loader), desc="eval")):
            if not use_cpu:
                points, target = points.cuda(), target.cuda()
            points = points.transpose(2, 1)
            pred, _ = model(points)
            pred_choice = pred.data.max(1)[1]

            for cat in np.unique(target.cpu()):
                cat_int = int(cat)
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat_int, 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[cat_int, 1] += 1

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break

    per_class: dict[int, float] = {}
    for idx in range(num_class):
        if class_acc[idx, 1] > 0:
            per_class[idx] = float(class_acc[idx, 0] / class_acc[idx, 1])

    valid = class_acc[:, 1] > 0
    class_accuracy = float(np.mean(class_acc[valid, 0] / class_acc[valid, 1])) if valid.any() else 0.0
    overall_accuracy = float(np.mean(mean_correct)) if mean_correct else 0.0
    return overall_accuracy, class_accuracy, per_class


def write_result_files(
    output_dir: Path,
    variant: str,
    overall_accuracy: float,
    class_accuracy: float,
    per_class: dict[int, float],
    idx_to_class: dict[str, str],
    checkpoint_path: Path,
    log_path: Path,
    best_epoch: int,
    reports_dir: Path | None = None,
) -> None:
    out_reports = reports_dir or (PROJECT_ROOT / "reports")
    out_reports.mkdir(parents=True, exist_ok=True)

    csv_path = out_reports / f"{variant}_result.csv"
    md_path = out_reports / f"{variant}_result.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["variant", variant])
        writer.writerow(["overall_accuracy", f"{overall_accuracy:.6f}"])
        writer.writerow(["class_accuracy", f"{class_accuracy:.6f}"])
        writer.writerow(["best_epoch", best_epoch])
        writer.writerow(["checkpoint_path", str(checkpoint_path)])
        writer.writerow(["log_path", str(log_path)])
        writer.writerow([])
        writer.writerow(["class_idx", "class_name", "accuracy"])
        for idx in sorted(per_class):
            writer.writerow([idx, idx_to_class.get(str(idx), idx_to_class.get(idx, "?")), f"{per_class[idx]:.6f}"])

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        f"# PointNet++ Result — {variant}",
        "",
        f"- Generated at: {now}",
        f"- Overall accuracy: **{overall_accuracy * 100:.2f}%**",
        f"- Class accuracy: **{class_accuracy * 100:.2f}%**",
        f"- Best epoch: {best_epoch}",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Log: `{log_path}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir or (PROJECT_ROOT / "outputs" / args.variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir = output_dir / "logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    log_file = args.log_file or (PROJECT_ROOT / "logs" / f"{args.variant}.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("train_pointnet2")

    if not args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    logger.info("Args: %s", args)
    logger.info("PointNet++ repo: %s", PN2_ROOT)
    logger.info("Data root: %s", args.data_root)
    logger.info("num_point=%d allow_resample=%s", args.num_point, args.allow_resample)

    train_dataset = ModelNetNPYDataset(
        args.data_root,
        split="train",
        num_points=args.num_point,
        allow_resample=args.allow_resample,
    )
    test_dataset = ModelNetNPYDataset(
        args.data_root,
        split="test",
        num_points=args.num_point,
        allow_resample=args.allow_resample,
    )
    sample_pts, _ = train_dataset[0]
    logger.info("First train sample shape after loader: %s", tuple(sample_pts.shape))
    logger.info("Dataset sizes: train=%d test=%d", len(train_dataset), len(test_dataset))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_module = importlib.import_module(args.model)
    shutil.copy(PN2_ROOT / "models" / f"{args.model}.py", output_dir / f"{args.model}.py")
    shutil.copy(PN2_ROOT / "models" / "pointnet2_utils.py", output_dir / "pointnet2_utils.py")

    classifier = model_module.get_model(args.num_category, normal_channel=False)
    criterion = model_module.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    best_overall = 0.0
    best_class = 0.0
    best_epoch = 0
    best_per_class: dict[int, float] = {}
    final_overall = 0.0
    final_class = 0.0
    final_epoch = 0

    for epoch in range(args.epoch):
        classifier.train()
        scheduler.step()
        mean_correct: list[float] = []

        train_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"train e{epoch+1}")
        for batch_idx, (points, target) in train_iter:
            optimizer.zero_grad()
            points_np = points.data.numpy()
            points_np = provider.random_point_dropout(points_np)
            points_np[:, :, 0:3] = provider.random_scale_point_cloud(points_np[:, :, 0:3])
            points_np[:, :, 0:3] = provider.shift_point_cloud(points_np[:, :, 0:3])
            points = torch.Tensor(points_np).transpose(2, 1)

            if epoch == 0 and batch_idx == 0:
                logger.info(
                    "First train batch tensor shape before device move: %s (batch=%d, channels=%d, points=%d)",
                    tuple(points.shape),
                    points.shape[0],
                    points.shape[1],
                    points.shape[2],
                )

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            if epoch == 0 and batch_idx == 0:
                logger.info("First train batch loss: %.6f", float(loss.item()))
            loss.backward()
            optimizer.step()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

            if args.max_train_batches > 0 and batch_idx + 1 >= args.max_train_batches:
                break

        train_acc = float(np.mean(mean_correct)) if mean_correct else 0.0
        overall_acc, class_acc, per_class = evaluate(
            classifier,
            test_loader,
            args.num_category,
            args.use_cpu,
            max_batches=args.max_eval_batches,
        )
        final_overall = overall_acc
        final_class = class_acc
        final_epoch = epoch + 1
        logger.info(
            "Epoch %d/%d train_acc=%.4f test_overall=%.4f test_class=%.4f",
            epoch + 1,
            args.epoch,
            train_acc,
            overall_acc,
            class_acc,
        )

        if overall_acc >= best_overall:
            best_overall = overall_acc
            best_class = class_acc
            best_epoch = epoch + 1
            best_per_class = per_class
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": best_epoch,
                    "instance_acc": overall_acc,
                    "class_acc": class_acc,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "variant": args.variant,
                    "seed": args.seed,
                },
                checkpoint_path,
            )
            logger.info("Saved checkpoint to %s", checkpoint_path)

    idx_to_class_path = args.data_root / "metadata" / "idx_to_class.json"
    with open(idx_to_class_path, encoding="utf-8") as handle:
        idx_to_class = json.load(handle)

    write_result_files(
        output_dir=output_dir,
        variant=args.variant,
        overall_accuracy=best_overall,
        class_accuracy=best_class,
        per_class=best_per_class,
        idx_to_class=idx_to_class,
        checkpoint_path=checkpoint_dir / "best_model.pth",
        log_path=log_file,
        best_epoch=best_epoch,
        reports_dir=args.reports_dir,
    )

    summary = {
        "variant": args.variant,
        "overall_accuracy": best_overall,
        "class_accuracy": best_class,
        "best_epoch": best_epoch,
        "checkpoint_path": str(checkpoint_dir / "best_model.pth"),
        "final_test_overall_accuracy": final_overall,
        "final_test_class_accuracy": final_class,
        "final_epoch": final_epoch,
        "data_root": str(args.data_root),
        "num_point": args.num_point,
        "allow_resample": args.allow_resample,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Training finished. best_overall=%.4f best_class=%.4f", best_overall, best_class)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
