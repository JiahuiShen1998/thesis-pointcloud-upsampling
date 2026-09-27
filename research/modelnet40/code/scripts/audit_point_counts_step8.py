#!/usr/bin/env python3
"""Step 8: audit native point counts for baseline and EAR datasets."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")

DATASETS: list[tuple[str, Path]] = [
    ("original_baseline", PROJECT_ROOT / "datasets" / "modelnet40_original"),
    ("downsampled50_baseline", PROJECT_ROOT / "datasets" / "modelnet40_downsampled50"),
    ("downsampled50_ear", PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear"),
]


def audit_dataset(name: str, data_root: Path) -> dict:
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"{name}: missing train/test under {data_root}")

    split_stats: dict[str, dict] = {}
    all_point_counts: list[int] = []
    all_shapes: Counter[tuple[int, ...]] = Counter()
    nan_inf_files: list[str] = []
    train_count = 0
    test_count = 0

    for split, split_dir in [("train", train_dir), ("test", test_dir)]:
        point_counts: list[int] = []
        for npy_path in sorted(split_dir.rglob("*.npy")):
            arr = np.load(npy_path)
            shape = tuple(arr.shape)
            all_shapes[shape] += 1
            if arr.ndim == 2 and arr.shape[1] == 3:
                point_counts.append(int(arr.shape[0]))
                all_point_counts.append(int(arr.shape[0]))
            if not np.isfinite(arr).all():
                nan_inf_files.append(str(npy_path))
            if split == "train":
                train_count += 1
            else:
                test_count += 1

        split_stats[split] = {
            "count": len(point_counts),
            "min": min(point_counts) if point_counts else 0,
            "max": max(point_counts) if point_counts else 0,
            "mean": float(np.mean(point_counts)) if point_counts else 0.0,
        }

    shape_summary = ", ".join(f"{k}:{v}" for k, v in sorted(all_shapes.items(), key=lambda x: -x[1]))

    return {
        "dataset_name": name,
        "data_root": str(data_root.resolve()),
        "train_count": train_count,
        "test_count": test_count,
        "min_points": min(all_point_counts) if all_point_counts else 0,
        "max_points": max(all_point_counts) if all_point_counts else 0,
        "mean_points": float(np.mean(all_point_counts)) if all_point_counts else 0.0,
        "sample_shape": shape_summary,
        "nan_inf_exists": len(nan_inf_files) > 0,
        "nan_inf_count": len(nan_inf_files),
        "train_min": split_stats["train"]["min"],
        "train_max": split_stats["train"]["max"],
        "train_mean": split_stats["train"]["mean"],
        "test_min": split_stats["test"]["min"],
        "test_max": split_stats["test"]["max"],
        "test_mean": split_stats["test"]["mean"],
    }


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "dataset_name",
        "data_root",
        "train_count",
        "test_count",
        "min_points",
        "max_points",
        "mean_points",
        "sample_shape",
        "nan_inf_exists",
        "nan_inf_count",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def write_md(rows: list[dict], path: Path) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# Step 8 — Point Count Audit",
        "",
        f"- Generated at: {now}",
        "",
        "## Purpose",
        "",
        "Record **native on-disk point counts** for each dataset variant.",
        "This experiment does **not** enforce equal point counts across variants.",
        "",
        "| dataset | train | test | min | max | mean | sample shape | NaN/Inf |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        nan_flag = "YES" if row["nan_inf_exists"] else "no"
        lines.append(
            f"| {row['dataset_name']} | {row['train_count']} | {row['test_count']} "
            f"| {row['min_points']} | {row['max_points']} | {row['mean_points']:.1f} "
            f"| {row['sample_shape']} | {nan_flag} ({row['nan_inf_count']}) |"
        )

    lines.extend(
        [
            "",
            "## Per-Dataset Detail",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['dataset_name']}",
                "",
                f"- Path: `{row['data_root']}`",
                f"- Train: min={row['train_min']}, max={row['train_max']}, mean={row['train_mean']:.1f}",
                f"- Test: min={row['test_min']}, max={row['test_max']}, mean={row['test_mean']:.1f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Experiment Point-Count Policy (Step 8)",
            "",
            "- `original_baseline`: native 1024 points → `num_points=1024`",
            "- `downsampled50_baseline`: native 512 points → `num_points=512` (no upsample to match EAR)",
            "- `downsampled50_ear`: native 1024 points (EAR upsampled) → `num_points=1024`",
            "- Do **not** crop EAR output back to 512.",
            "- Do **not** pad downsampled50 baseline to 1024 for this comparison.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows: list[dict] = []
    for name, root in DATASETS:
        if not root.exists():
            print(f"ERROR: dataset root missing: {root}", file=sys.stderr)
            return 1
        print(f"Auditing {name} ...")
        rows.append(audit_dataset(name, root))

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "step8_point_count_audit.csv"
    md_path = reports_dir / "step8_point_count_audit.md"
    write_csv(rows, csv_path)
    write_md(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
