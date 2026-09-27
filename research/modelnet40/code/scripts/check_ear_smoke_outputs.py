#!/usr/bin/env python3
"""Validate Step 7a EAR smoke outputs and run PointNet++ DataLoader dry-run."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "step7_smoke_sample_manifest.csv"
DEFAULT_AUDIT = PROJECT_ROOT / "reports" / "step7a_ear_smoke_audit.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "step7a_ear_smoke_report.md"

OUT_ORIGINAL = PROJECT_ROOT / "datasets" / "modelnet40_original_up" / "ear_smoke"
OUT_DOWN = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear_smoke"

EXPECTED_PER_VARIANT = 80
TARGET_SHAPE = (1024, 3)
SPLITS = ("train", "test")


def collect_npy_files(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.rglob("*.npy"))


def check_variant(
    output_root: Path,
    manifest_rows: list[dict],
    source_variant: str,
    smoke_key: str,
) -> dict:
    issues: list[str] = []
    files = [p for split in SPLITS for p in collect_npy_files(output_root, split)]
    file_count = len(files)

    if file_count != EXPECTED_PER_VARIANT:
        issues.append(f"{source_variant}: expected {EXPECTED_PER_VARIANT} npy files, found {file_count}")

    shape_bad = 0
    nan_inf = 0
    label_mismatch = 0
    class_mismatch = 0

    expected_by_output: dict[str, dict] = {}
    for row in manifest_rows:
        smoke_path = Path(row[smoke_key])
        rel = smoke_path.relative_to(smoke_path.parents[2])  # split/class/file
        out_path = output_root / rel
        expected_by_output[str(out_path)] = row

    for npy_path in files:
        arr = np.load(npy_path)
        if arr.shape != TARGET_SHAPE:
            shape_bad += 1
            issues.append(f"{npy_path}: shape {arr.shape} != {TARGET_SHAPE}")
        if not np.isfinite(arr).all():
            nan_inf += 1
            issues.append(f"{npy_path}: contains NaN/Inf")

        key = str(npy_path)
        if key not in expected_by_output:
            issues.append(f"{npy_path}: no matching manifest row")
            continue

        row = expected_by_output[key]
        if npy_path.parent.name != row["class_name"]:
            class_mismatch += 1
            issues.append(f"{npy_path}: class dir mismatch")
        if npy_path.parent.parent.name != row["split"]:
            issues.append(f"{npy_path}: split mismatch")

    for key, row in expected_by_output.items():
        out_path = Path(key)
        if not out_path.is_file():
            issues.append(f"Missing output for manifest sample: {out_path}")
            continue
        audit_label = int(row["label"])
        class_to_idx_path = output_root / "metadata" / "class_to_idx.json"
        if class_to_idx_path.is_file():
            with open(class_to_idx_path, encoding="utf-8") as handle:
                class_to_idx = json.load(handle)
            if class_to_idx.get(row["class_name"]) != audit_label:
                label_mismatch += 1
                issues.append(f"{out_path}: label mismatch vs class_to_idx")

    split_counts = Counter(p.parent.parent.name for p in files)
    class_counts = Counter(p.parent.name for p in files)

    return {
        "source_variant": source_variant,
        "output_root": str(output_root),
        "file_count": file_count,
        "expected_count": EXPECTED_PER_VARIANT,
        "shape_bad": shape_bad,
        "nan_inf": nan_inf,
        "label_mismatch": label_mismatch,
        "class_mismatch": class_mismatch,
        "split_counts": dict(split_counts),
        "class_count": len(class_counts),
        "issues": issues,
        "ok": not issues,
    }


def dataloader_dry_run(data_root: Path, batch_size: int = 8) -> dict:
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
            "dtype_points": str(batch_points.dtype),
            "dtype_labels": str(batch_labels.dtype),
            "finite": bool(torch.isfinite(batch_points).all()),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Check EAR smoke outputs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    with open(args.manifest, newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))

    audit_rows: list[dict] = []
    if args.audit.is_file():
        with open(args.audit, newline="", encoding="utf-8") as handle:
            audit_rows = list(csv.DictReader(handle))

    original_check = check_variant(
        OUT_ORIGINAL, manifest_rows, "original", "smoke_original_path"
    )
    down_check = check_variant(
        OUT_DOWN, manifest_rows, "downsampled50", "smoke_downsampled50_path"
    )

    dataloader_results = {}
    dataloader_errors: list[str] = []
    if original_check["ok"]:
        try:
            dataloader_results["original"] = dataloader_dry_run(OUT_ORIGINAL)
        except Exception as exc:
            dataloader_errors.append(f"original DataLoader failed: {exc}")
    else:
        dataloader_errors.append("original outputs invalid; skipped DataLoader dry-run")

    if down_check["ok"]:
        try:
            dataloader_results["downsampled50"] = dataloader_dry_run(OUT_DOWN)
        except Exception as exc:
            dataloader_errors.append(f"downsampled50 DataLoader failed: {exc}")
    else:
        dataloader_errors.append("downsampled50 outputs invalid; skipped DataLoader dry-run")

    audit_success = {
        "original": sum(1 for r in audit_rows if r.get("source_variant") == "original" and r.get("status") == "success"),
        "downsampled50": sum(
            1 for r in audit_rows if r.get("source_variant") == "downsampled50" and r.get("status") == "success"
        ),
        "failed": sum(1 for r in audit_rows if r.get("status") != "success"),
    }

    all_ok = original_check["ok"] and down_check["ok"] and not dataloader_errors
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Step 7a — EAR Smoke Test Report",
        "",
        f"- Generated at: {now}",
        f"- Status: **{'PASS' if all_ok else 'FAIL'}**",
        "",
        "## EAR Configuration",
        "",
        "- EAR code path: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/EAR/ear_upsampling.py`",
        "- Input format: ModelNet40 `.npy` `(N, 3)` wrapped as `(N, 4)` with zero intensity for `ear_upsample_kitti`",
        "- Output format: `.npy` `(1024, 3)` after EAR + resample normalization",
        "- GPU required: **No** (numpy + scikit-learn only)",
        "- Checkpoint required: **No** (deterministic edge-aware resampling)",
        "- Smoke executed: **Yes** (CPU on login node)",
        "",
        "## Smoke Results",
        "",
        f"- Original success count (audit): {audit_success['original']}/{EXPECTED_PER_VARIANT}",
        f"- Downsampled50 success count (audit): {audit_success['downsampled50']}/{EXPECTED_PER_VARIANT}",
        f"- Failed samples (audit): {audit_success['failed']}",
        "",
        "### Original Output Check",
        "",
        f"- Output root: `{OUT_ORIGINAL}`",
        f"- File count: {original_check['file_count']} (expected {EXPECTED_PER_VARIANT})",
        f"- Shape violations: {original_check['shape_bad']}",
        f"- NaN/Inf files: {original_check['nan_inf']}",
        f"- Split counts: {original_check['split_counts']}",
        f"- Class dirs present: {original_check['class_count']}",
        "",
        "### Downsampled50 Output Check",
        "",
        f"- Output root: `{OUT_DOWN}`",
        f"- File count: {down_check['file_count']} (expected {EXPECTED_PER_VARIANT})",
        f"- Shape violations: {down_check['shape_bad']}",
        f"- NaN/Inf files: {down_check['nan_inf']}",
        f"- Split counts: {down_check['split_counts']}",
        f"- Class dirs present: {down_check['class_count']}",
        "",
        "## DataLoader Dry-Run",
        "",
    ]

    if dataloader_results:
        for variant, split_results in dataloader_results.items():
            lines.append(f"### {variant}")
            lines.append("")
            for split, info in split_results.items():
                lines.append(
                    f"- `{split}`: samples={info['num_samples']}, "
                    f"batch_points={info['batch_points_shape']}, "
                    f"batch_labels={info['batch_labels_shape']}, finite={info['finite']}"
                )
            lines.append("")
    if dataloader_errors:
        lines.append("### DataLoader Errors")
        lines.append("")
        for err in dataloader_errors:
            lines.append(f"- {err}")
        lines.append("")

    all_issues = original_check["issues"] + down_check["issues"] + dataloader_errors
    if all_issues:
        lines.extend(["## Issues", ""])
        for issue in all_issues[:50]:
            lines.append(f"- {issue}")
        if len(all_issues) > 50:
            lines.append(f"- ... and {len(all_issues) - 50} more")
        lines.append("")

    lines.extend(
        [
            "## Readiness",
            "",
            f"- Can proceed to EAR full upsampling: **{'Yes' if all_ok else 'No — fix smoke failures first'}**",
            "- Can proceed to PDANS smoke test: **Yes** (independent method; PDANS needs GPU)",
            "",
            "## Full EAR Upsampling Estimate",
            "",
            "- Hardware: CPU sufficient (no GPU required)",
            "- Dataset size: 9843 train + 2468 test = 12311 samples per line",
            "- Total full runs: 2 lines × 12311 ≈ 24622 EAR inferences",
            "- Per-sample CPU time (smoke avg): see audit `ear_raw_points` / logs",
            "- Recommendation: submit CPU batch job via `sbatch` on standard partition, or parallel chunk scripts",
            "",
            "## Artifacts",
            "",
            f"- Audit CSV: `{args.audit}`",
            f"- Raw EAR outputs: `{PROJECT_ROOT / 'outputs/ear_smoke/raw'}`",
            f"- Log: `{PROJECT_ROOT / 'logs/ear_smoke/run_ear_smoke.log'}`",
        ]
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Report written: {args.report}")
    print(f"Original: {original_check['file_count']} files, ok={original_check['ok']}")
    print(f"Downsampled50: {down_check['file_count']} files, ok={down_check['ok']}")
    print(f"Overall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
