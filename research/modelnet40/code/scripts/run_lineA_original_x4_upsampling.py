#!/usr/bin/env python3
"""Organize Line A (Original + Upsampling ×4) from legacy outputs or run upsampling."""

from __future__ import annotations

import argparse
import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    GLOBAL_SEED,
    LEGACY_LINEA_UP_ROOT,
    MAIN_METHODS,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    legacy_method_subdir,
    lineA_paths,
    protocol_counts,
    reports_dir,
)
from strict_normalize_point_count import normalize_tree, write_audit

METHOD_LABELS = {
    "ear": "EAR",
    "pu_net": "PU-Net",
    "pu_gcn": "PU-GCN",
    "pdans": "PDANS",
}


def symlink_or_copy_tree(src: Path, dst: Path, use_symlink: bool = True) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copytree(src, dst)


def organize_method(method: str, four_n: int, use_symlink: bool = True) -> dict:
    m = method
    legacy_sub = legacy_method_subdir(m)
    legacy_root = LEGACY_LINEA_UP_ROOT / legacy_sub
    paths = lineA_paths(m)
    raw_root = paths["raw"]
    strict_root = paths["strict_4N"]

    status = "missing"
    train_count = test_count = 0
    dominant_shape = None

    if legacy_root.is_dir():
        train_count = sum(1 for _ in (legacy_root / "train").rglob("*.npy")) if (legacy_root / "train").is_dir() else 0
        test_count = sum(1 for _ in (legacy_root / "test").rglob("*.npy")) if (legacy_root / "test").is_dir() else 0
        sample = next(legacy_root.rglob("*.npy"), None)
        if sample:
            dominant_shape = tuple(np.load(sample).shape)
        if train_count == EXPECTED_TRAIN and test_count == EXPECTED_TEST and dominant_shape == (four_n, 3):
            symlink_or_copy_tree(legacy_root, raw_root, use_symlink)
            rows = normalize_tree(raw_root, strict_root, four_n, f"lineA_{m}_strict")
            status = "reused_legacy"
        elif legacy_root.is_dir() and train_count + test_count > 0:
            symlink_or_copy_tree(legacy_root, raw_root, use_symlink)
            normalize_tree(raw_root, strict_root, four_n, f"lineA_{m}_strict")
            status = "reused_partial"
        else:
            status = "legacy_incomplete"
    else:
        status = "not_generated"

    return {
        "method": METHOD_LABELS.get(m, m),
        "method_key": m,
        "legacy_path": str(legacy_root),
        "raw_path": str(raw_root),
        "strict_path": str(strict_root),
        "train_count": train_count,
        "test_count": test_count,
        "expected_output": four_n,
        "dominant_shape": dominant_shape,
        "status": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-symlink", action="store_true", default=True)
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = parser.parse_args()
    use_symlink = not args.copy

    n = detect_original_point_count()
    counts = protocol_counts(n)
    four_n = counts["four_N"]

    # Baseline: symlink original -> pointnet2 input path later; here ensure lineA baseline ref
    baseline_dst = PROJECT_ROOT / "pointnet2_inputs" / "lineA_original_baseline"
    baseline_dst.parent.mkdir(parents=True, exist_ok=True)
    if not baseline_dst.exists():
        baseline_dst.symlink_to(ORIGINAL_ROOT.resolve())

    rows = [organize_method(m, four_n, use_symlink) for m in MAIN_METHODS]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rep = reports_dir()
    csv_path = rep / "modelnet40_lineA_original_up_count_audit.csv"
    md_path = rep / "modelnet40_lineA_original_up_count_audit.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Line A — Original + Upsampling ×4 Count Audit",
        "",
        f"- Generated at: {ts}",
        f"- Input N: {n}",
        f"- Expected output 4N: {four_n}",
        "",
        "| method | train | test | shape | status |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['method']} | {r['train_count']} | {r['test_count']} | {r['dominant_shape']} | {r['status']} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    for r in rows:
        print(f"{r['method']}: {r['status']} train={r['train_count']} test={r['test_count']} shape={r['dominant_shape']}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
