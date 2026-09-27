#!/usr/bin/env python3
"""Validate ModelNet40 Downsampled50 dataset against original 1024-point data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
DEFAULT_ORIGINAL = PROJECT_ROOT / "datasets" / "modelnet40_original"
DEFAULT_DOWNSAMPLED = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "step5_downsampled50_check.md"

EXPECTED_TRAIN = 9843
EXPECTED_TEST = 2468
EXPECTED_TOTAL = EXPECTED_TRAIN + EXPECTED_TEST
EXPECTED_CLASSES = 40
ORIGINAL_SHAPE = (1024, 3)
DOWNSAMPLED_SHAPE = (512, 3)
SPLITS = ("train", "test")
GLOBAL_SEED = 42


def per_file_seed(global_seed: int, split: str, class_name: str, shape_id: str) -> int:
    token = f"{global_seed}:{split}:{class_name}:{shape_id}".encode("utf-8")
    digest = hashlib.md5(token).hexdigest()
    return global_seed + (int(digest[:8], 16) % 1_000_000)


def collect_npy_files(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.rglob("*.npy"))


def count_per_class(root: Path, split: str) -> Counter:
    counts: Counter = Counter()
    split_dir = root / split
    if not split_dir.is_dir():
        return counts
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        counts[class_dir.name] = len(list(class_dir.glob("*.npy")))
    return counts


def point_is_in_original(point: np.ndarray, original: np.ndarray, atol: float = 1e-5) -> bool:
    diff = np.abs(original - point[None, :])
    return bool(np.any(np.all(diff <= atol, axis=1)))


def verify_subset_membership(
    original_path: Path,
    downsampled_path: Path,
    downsampled_root: Path,
    split: str,
    class_name: str,
    shape_id: str,
    seed: int,
) -> dict:
    original = np.load(original_path)
    downsampled = np.load(downsampled_path)
    rng = np.random.default_rng(seed)
    expected_indices = np.sort(rng.choice(original.shape[0], size=downsampled.shape[0], replace=False))
    expected = original[expected_indices]

    exact_match = np.allclose(expected, downsampled, atol=1e-6, rtol=0)
    subset_ok = all(point_is_in_original(downsampled[i], original) for i in range(downsampled.shape[0]))
    return {
        "path": str(downsampled_path.relative_to(downsampled_root)),
        "exact_match_expected_indices": exact_match,
        "subset_of_original": subset_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ModelNet40 Downsampled50 dataset.")
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--downsampled-root", type=Path, default=DEFAULT_DOWNSAMPLED)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()

    original_root = args.original_root.resolve()
    downsampled_root = args.downsampled_root.resolve()

    orig_train = len(collect_npy_files(original_root, "train"))
    orig_test = len(collect_npy_files(original_root, "test"))
    down_train = len(collect_npy_files(downsampled_root, "train"))
    down_test = len(collect_npy_files(downsampled_root, "test"))

    class_to_idx_path = downsampled_root / "metadata" / "class_to_idx.json"
    num_classes = 0
    if class_to_idx_path.is_file():
        with open(class_to_idx_path, encoding="utf-8") as handle:
            num_classes = len(json.load(handle))

    orig_class_counts = {split: count_per_class(original_root, split) for split in SPLITS}
    down_class_counts = {split: count_per_class(downsampled_root, split) for split in SPLITS}

    shape_bad: list[str] = []
    nan_bad: list[str] = []
    inf_bad: list[str] = []
    all_down = collect_npy_files(downsampled_root, "train") + collect_npy_files(downsampled_root, "test")
    for path in all_down:
        arr = np.load(path)
        if arr.shape != DOWNSAMPLED_SHAPE:
            shape_bad.append(str(path))
        if np.isnan(arr).any():
            nan_bad.append(str(path))
        if np.isinf(arr).any():
            inf_bad.append(str(path))

    class_count_mismatch: list[str] = []
    for split in SPLITS:
        for class_name in set(orig_class_counts[split]) | set(down_class_counts[split]):
            o = orig_class_counts[split].get(class_name, 0)
            d = down_class_counts[split].get(class_name, 0)
            if o != d:
                class_count_mismatch.append(f"{split}/{class_name}: original={o}, downsampled={d}")

    rng = random.Random(args.seed)
    sample_paths = rng.sample(all_down, min(args.sample_count, len(all_down))) if all_down else []
    subset_checks: list[dict] = []
    for down_path in sample_paths:
        rel = down_path.relative_to(downsampled_root)
        split = rel.parts[0]
        class_name = rel.parts[1]
        shape_id = down_path.stem
        orig_path = original_root / split / class_name / f"{shape_id}.npy"
        seed = per_file_seed(args.seed, split, class_name, shape_id)
        subset_checks.append(
            verify_subset_membership(orig_path, down_path, downsampled_root, split, class_name, shape_id, seed)
        )

    checks = {
        "train_count_match_original": down_train == orig_train == EXPECTED_TRAIN,
        "test_count_match_original": down_test == orig_test == EXPECTED_TEST,
        "total_count_ok": (down_train + down_test) == EXPECTED_TOTAL,
        "class_count_ok": num_classes == EXPECTED_CLASSES,
        "per_class_counts_match": len(class_count_mismatch) == 0,
        "shape_ok": len(shape_bad) == 0,
        "nan_ok": len(nan_bad) == 0,
        "inf_ok": len(inf_bad) == 0,
        "subset_membership_ok": all(row["subset_of_original"] for row in subset_checks),
        "deterministic_repro_ok": all(row["exact_match_expected_indices"] for row in subset_checks),
    }
    all_ok = all(checks.values())

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# Step 5 — ModelNet40 Downsampled50 Check",
        "",
        f"- Generated at: {now}",
        f"- Original root: `{original_root}`",
        f"- Downsampled root: `{downsampled_root}`",
        f"- Seed: {args.seed}",
        "",
        "## Counts",
        "",
        f"- Original train/test: {orig_train} / {orig_test}",
        f"- Downsampled train/test: {down_train} / {down_test}",
        f"- Classes: {num_classes} (expected {EXPECTED_CLASSES})",
        "",
        "## Validation checks",
        "",
        "| check | result |",
        "| --- | --- |",
    ]
    for name, ok in checks.items():
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")

    if subset_checks:
        lines.extend(["", "## Random subset verification", ""])
        lines.extend(
            [
                "| file | subset_of_original | exact_repro |",
                "| --- | --- | --- |",
            ]
        )
        for row in subset_checks:
            lines.append(
                f"| `{row['path']}` | {'yes' if row['subset_of_original'] else 'no'} | "
                f"{'yes' if row['exact_match_expected_indices'] else 'no'} |"
            )

    if class_count_mismatch:
        lines.extend(["", "## Class count mismatches", ""])
        for item in class_count_mismatch[:20]:
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"**{'PASS' if all_ok else 'FAIL'}** — "
            f"{'ready for Step 6 after Step 4 completes' if all_ok else 'fix issues before Step 6'}.",
            "",
        ]
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Check report: {args.report_path}")
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
