#!/usr/bin/env python3
"""Validate processed ModelNet40 original dataset before PointNet++ training."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_DATA_ROOT = (
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/"
    "datasets/modelnet40_original"
)
DEFAULT_REPORT_PATH = (
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/"
    "reports/step2_modelnet40_original_check.md"
)
EXPECTED_CLASSES = 40
EXPECTED_TRAIN = 9843
EXPECTED_TEST = 2468
EXPECTED_TOTAL = EXPECTED_TRAIN + EXPECTED_TEST
EXPECTED_SHAPE = (1024, 3)
SPLITS = ("train", "test")


def collect_npy_files(data_root: Path, split: str) -> list[Path]:
    split_dir = data_root / split
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.rglob("*.npy"))


def inspect_file(path: Path) -> dict:
    arr = np.load(path)
    radii = np.linalg.norm(arr, axis=1) if arr.ndim == 2 and arr.shape[1] >= 3 else np.array([])
    return {
        "path": str(path),
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "max_radius": float(radii.max()) if radii.size else None,
        "min_radius": float(radii.min()) if radii.size else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check processed ModelNet40 original dataset.")
    parser.add_argument("--data-root", type=Path, default=Path(DEFAULT_DATA_ROOT))
    parser.add_argument("--report-path", type=Path, default=Path(DEFAULT_REPORT_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    report_path = args.report_path.resolve()

    if not data_root.is_dir():
        print(f"ERROR: data root not found: {data_root}", file=sys.stderr)
        return 1

    split_counts = {split: len(collect_npy_files(data_root, split)) for split in SPLITS}
    total = split_counts["train"] + split_counts["test"]

    class_to_idx_path = data_root / "metadata" / "class_to_idx.json"
    num_classes = 0
    if class_to_idx_path.is_file():
        with open(class_to_idx_path, encoding="utf-8") as handle:
            num_classes = len(json.load(handle))
    else:
        classes = set()
        for split in SPLITS:
            split_dir = data_root / split
            if split_dir.is_dir():
                classes.update(p.name for p in split_dir.iterdir() if p.is_dir())
        num_classes = len(classes)

    per_class_counts: dict[str, Counter] = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            per_class_counts[split][class_dir.name] = len(list(class_dir.glob("*.npy")))

    all_files = collect_npy_files(data_root, "train") + collect_npy_files(data_root, "test")
    rng = random.Random(args.seed)
    sample_files = rng.sample(all_files, min(args.sample_count, len(all_files))) if all_files else []

    shape_bad: list[str] = []
    nan_bad: list[str] = []
    inf_bad: list[str] = []
    radius_bad: list[str] = []
    sample_rows: list[dict] = []

    for path in all_files:
        info = inspect_file(path)
        if info["shape"] != EXPECTED_SHAPE:
            shape_bad.append(info["path"])
        if info["has_nan"]:
            nan_bad.append(info["path"])
        if info["has_inf"]:
            inf_bad.append(info["path"])
        if info["max_radius"] is not None and info["max_radius"] > 1.000001:
            radius_bad.append(info["path"])

    for path in sample_files:
        sample_rows.append(inspect_file(path))

    checks = {
        "train_count_ok": split_counts["train"] == EXPECTED_TRAIN,
        "test_count_ok": split_counts["test"] == EXPECTED_TEST,
        "total_count_ok": total == EXPECTED_TOTAL,
        "class_count_ok": num_classes == EXPECTED_CLASSES,
        "shape_ok": len(shape_bad) == 0,
        "nan_ok": len(nan_bad) == 0,
        "inf_ok": len(inf_bad) == 0,
        "unit_sphere_ok": len(radius_bad) == 0,
    }
    all_ok = all(checks.values())

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# Step 2 — ModelNet40 Original Dataset Check",
        "",
        f"- Generated at: {now}",
        f"- Data root: `{data_root}`",
        "",
        "## Counts",
        "",
        f"- Train `.npy` files: **{split_counts['train']}** (expected {EXPECTED_TRAIN})",
        f"- Test `.npy` files: **{split_counts['test']}** (expected {EXPECTED_TEST})",
        f"- Total: **{total}** (expected {EXPECTED_TOTAL})",
        f"- Classes: **{num_classes}** (expected {EXPECTED_CLASSES})",
        "",
        "## Validation checks",
        "",
        "| check | result |",
        "| --- | --- |",
    ]
    for name, ok in checks.items():
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")

    lines.extend(["", "## Random sample inspection", ""])
    if sample_rows:
        lines.extend(["| file | shape | max_radius | nan | inf |", "| --- | --- | ---: | --- | --- |"])
        for row in sample_rows:
            rel = Path(row["path"]).relative_to(data_root)
            lines.append(
                f"| `{rel}` | {row['shape']} | {row['max_radius']:.6f} | "
                f"{'yes' if row['has_nan'] else 'no'} | {'yes' if row['has_inf'] else 'no'} |"
            )
    else:
        lines.append("_No `.npy` files found._")

    for split in SPLITS:
        lines.extend(["", f"## {split} per-class counts", "", "| class | count |", "| --- | ---: |"])
        for class_name in sorted(per_class_counts[split]):
            lines.append(f"| {class_name} | {per_class_counts[split][class_name]} |")

    if shape_bad:
        lines.extend(["", "## Shape mismatches (first 10)", ""])
        for item in shape_bad[:10]:
            lines.append(f"- `{item}`")

    if radius_bad:
        lines.extend(["", "## Unit-sphere violations (first 10)", ""])
        for item in radius_bad[:10]:
            lines.append(f"- `{item}`")

    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"**{'PASS' if all_ok else 'FAIL'}** — dataset is "
            f"{'ready for PointNet++ Step 3' if all_ok else 'NOT ready; inspect failures above'}.",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Check report written to: {report_path}")
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
