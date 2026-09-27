#!/usr/bin/env python3
"""Validate EAR full Line B outputs and run PointNet++ DataLoader dry-run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "step7b_ear_full_lineB_report.md"

EXPECTED_TRAIN = 9843
EXPECTED_TEST = 2468
EXPECTED_TOTAL = EXPECTED_TRAIN + EXPECTED_TEST
EXPECTED_CLASSES = 40
TARGET_SHAPE = (1024, 3)
SPLITS = ("train", "test")


def collect_npy_files(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.rglob("*.npy"))


def check_outputs(output_root: Path, input_root: Path) -> dict:
    issues: list[str] = []
    files = [p for split in SPLITS for p in collect_npy_files(output_root, split)]
    split_counts = Counter(p.parent.parent.name for p in files)
    class_counts = Counter(p.parent.name for p in files)

    if split_counts.get("train", 0) != EXPECTED_TRAIN:
        issues.append(f"train count {split_counts.get('train', 0)} != {EXPECTED_TRAIN}")
    if split_counts.get("test", 0) != EXPECTED_TEST:
        issues.append(f"test count {split_counts.get('test', 0)} != {EXPECTED_TEST}")
    if len(files) != EXPECTED_TOTAL:
        issues.append(f"total count {len(files)} != {EXPECTED_TOTAL}")
    if len(class_counts) != EXPECTED_CLASSES:
        issues.append(f"class dir count {len(class_counts)} != {EXPECTED_CLASSES}")

    shape_bad = 0
    nan_inf = 0
    for npy_path in files:
        arr = np.load(npy_path)
        if arr.shape != TARGET_SHAPE:
            shape_bad += 1
            if shape_bad <= 5:
                issues.append(f"{npy_path}: shape {arr.shape}")
        if not np.isfinite(arr).all():
            nan_inf += 1
            if nan_inf <= 5:
                issues.append(f"{npy_path}: NaN/Inf")

    class_to_idx_path = output_root / "metadata" / "class_to_idx.json"
    if not class_to_idx_path.is_file():
        issues.append(f"Missing {class_to_idx_path}")
    else:
        with open(class_to_idx_path, encoding="utf-8") as handle:
            class_to_idx = json.load(handle)
        if len(class_to_idx) != EXPECTED_CLASSES:
            issues.append(f"class_to_idx has {len(class_to_idx)} classes")

    for split in SPLITS:
        input_split = input_root / split
        for class_dir in sorted(p for p in input_split.iterdir() if p.is_dir()):
            for input_npy in sorted(class_dir.glob("*.npy")):
                rel = input_npy.relative_to(input_root)
                out_npy = output_root / rel
                if not out_npy.is_file():
                    issues.append(f"Missing output for {rel}")
                    if len(issues) > 100:
                        break

    return {
        "file_count": len(files),
        "split_counts": dict(split_counts),
        "class_count": len(class_counts),
        "shape_bad": shape_bad,
        "nan_inf": nan_inf,
        "issues": issues,
        "ok": not issues,
    }


def dataloader_dry_run(data_root: Path, batch_size: int = 24) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from modelnet_npy_dataloader import ModelNetNPYDataset

    results = {}
    for split in SPLITS:
        dataset = ModelNetNPYDataset(
            data_root=data_root,
            split=split,
            num_points=1024,
            normalize=True,
            allow_resample=False,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        batch_points, batch_labels = next(iter(loader))
        results[split] = {
            "num_samples": len(dataset),
            "batch_points_shape": tuple(batch_points.shape),
            "batch_labels_shape": tuple(batch_labels.shape),
            "finite": bool(torch.isfinite(batch_points).all()),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Check EAR full Line B outputs.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--update-report", action="store_true", default=True)
    args = parser.parse_args()

    check = check_outputs(args.output_root, args.input_root)
    dataloader_results = {}
    dataloader_error = ""
    if check["ok"]:
        try:
            dataloader_results = dataloader_dry_run(args.output_root)
        except Exception as exc:
            dataloader_error = str(exc)

    all_ok = check["ok"] and dataloader_results and not dataloader_error
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    if args.update_report:
        lines = [
            "# Step 7b — EAR Full Line B Upsampling Report",
            "",
            f"- Generated at: {now}",
            f"- Validation status: **{'PASS' if all_ok else 'FAIL'}**",
            "",
            "## Output Counts",
            "",
            f"- Train: {check['split_counts'].get('train', 0)} (expected {EXPECTED_TRAIN})",
            f"- Test: {check['split_counts'].get('test', 0)} (expected {EXPECTED_TEST})",
            f"- Total: {check['file_count']} (expected {EXPECTED_TOTAL})",
            f"- Class dirs: {check['class_count']} (expected {EXPECTED_CLASSES})",
            "",
            "## Quality Checks",
            "",
            f"- Shape violations: {check['shape_bad']}",
            f"- NaN/Inf files: {check['nan_inf']}",
            "",
            "## DataLoader Dry-Run",
            "",
        ]
        if dataloader_results:
            for split, info in dataloader_results.items():
                lines.append(
                    f"- `{split}`: samples={info['num_samples']}, "
                    f"batch_points={info['batch_points_shape']}, "
                    f"batch_labels={info['batch_labels_shape']}, finite={info['finite']}"
                )
        elif dataloader_error:
            lines.append(f"- ERROR: {dataloader_error}")
        else:
            lines.append("- Skipped (output check failed)")

        if check["issues"]:
            lines.extend(["", "## Issues", ""])
            for issue in check["issues"][:50]:
                lines.append(f"- {issue}")
            if len(check["issues"]) > 50:
                lines.append(f"- ... and {len(check['issues']) - 50} more")

        lines.extend(
            [
                "",
                "## Step 8 Readiness",
                "",
                f"- Ready for Downsampled50+EAR → PointNet++ training: **{'Yes' if all_ok else 'No'}**",
                "",
                f"- Output root: `{args.output_root}`",
                f"- Audit CSV: `{PROJECT_ROOT / 'reports/step7b_ear_full_lineB_audit.csv'}`",
            ]
        )
        args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Validation: {'PASS' if all_ok else 'FAIL'}")
    print(f"Counts: {check['split_counts']}, total={check['file_count']}")
    if check["issues"]:
        print(f"Issues: {len(check['issues'])}")
        for issue in check["issues"][:10]:
            print(f"  - {issue}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
