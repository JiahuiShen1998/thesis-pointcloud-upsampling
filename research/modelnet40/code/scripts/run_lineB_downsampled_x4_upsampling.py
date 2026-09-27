#!/usr/bin/env python3
"""Run or organize Line B (Downsampled ×4 + Upsampling ×4) outputs."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    LEGACY_LINEB_UP_ROOT,
    MAIN_METHODS,
    PROJECT_ROOT,
    detect_original_point_count,
    legacy_method_subdir,
    lineB_paths,
    protocol_counts,
    reports_dir,
)
from run_lineA_original_x4_upsampling import organize_method as _unused  # noqa: F401
from strict_normalize_point_count import normalize_tree

METHOD_LABELS = {"ear": "EAR", "pu_net": "PU-Net", "pu_gcn": "PU-GCN", "pdans": "PDANS"}


def check_lineB_method(method: str, n: int) -> dict:
    m = method
    legacy_sub = legacy_method_subdir(m)
    legacy_root = LEGACY_LINEB_UP_ROOT / legacy_sub
    paths = lineB_paths(m)
    raw_root = paths["raw"]
    strict_root = paths["strict_N"]

    # Legacy Line B was 512→2048 — NOT valid
    status = "pending_generation"
    train_count = test_count = 0
    dominant_shape = None

    if raw_root.is_dir():
        train_count = sum(1 for _ in (raw_root / "train").rglob("*.npy")) if (raw_root / "train").is_dir() else 0
        test_count = sum(1 for _ in (raw_root / "test").rglob("*.npy")) if (raw_root / "test").is_dir() else 0
        sample = next(raw_root.rglob("*.npy"), None)
        if sample:
            dominant_shape = tuple(np.load(sample).shape)
        if train_count == EXPECTED_TRAIN and test_count == EXPECTED_TEST and dominant_shape == (n, 3):
            normalize_tree(raw_root, strict_root, n, f"lineB_{m}_strict")
            status = "complete"
        elif train_count + test_count > 0:
            status = "partial"
    elif legacy_root.is_dir():
        sample = next(legacy_root.rglob("*.npy"), None)
        if sample:
            dominant_shape = tuple(np.load(sample).shape)
        if dominant_shape == (n * 2, 3):
            status = "legacy_wrong_protocol_512_to_2048"

    return {
        "method": METHOD_LABELS.get(m, m),
        "method_key": m,
        "input_points": n // 4,
        "expected_output": n,
        "raw_path": str(raw_root),
        "strict_path": str(strict_root),
        "train_count": train_count,
        "test_count": test_count,
        "dominant_shape": dominant_shape,
        "status": status,
    }


def submit_upsampling_jobs(methods: list[str], dry_run: bool = True) -> list[str]:
    """Submit new-protocol Line B upsampling jobs (256→1024)."""
    notes: list[str] = []
    submit_script = PROJECT_ROOT / "scripts" / "submit_lineB_downsampled_x4_method.sh"
    for method in methods:
        if method == "pu_gcn":
            notes.append("PU-GCN: pending (skipped)")
            continue
        cmd = ["bash", str(submit_script), method]
        if dry_run:
            notes.append(f"DRY-RUN: {' '.join(cmd)}")
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            notes.append(result.stdout.strip() or result.stderr.strip())
            if result.returncode != 0:
                notes.append(f"FAILED ({method}): exit {result.returncode}")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit-jobs", action="store_true")
    args = parser.parse_args()

    n = detect_original_point_count()
    down_n = n // 4

    baseline_dst = PROJECT_ROOT / "pointnet2_inputs" / "lineB_downsampled_x4_baseline"
    baseline_dst.parent.mkdir(parents=True, exist_ok=True)
    if not baseline_dst.exists() and DOWNSAMPLED_X4_ROOT.is_dir():
        baseline_dst.symlink_to(DOWNSAMPLED_X4_ROOT.resolve())

    rows = [check_lineB_method(m, n) for m in MAIN_METHODS]

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rep = reports_dir()
    csv_path = rep / "modelnet40_lineB_downsampled_x4_up_count_audit.csv"
    md_path = rep / "modelnet40_lineB_downsampled_x4_up_count_audit.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    job_notes = submit_upsampling_jobs(MAIN_METHODS) if args.submit_jobs else []

    md_lines = [
        "# Line B — Downsampled ×4 + Upsampling Count Audit",
        "",
        f"- Generated at: {ts}",
        f"- Downsampled input N/4: {down_n}",
        f"- Expected upsampling output N: {n}",
        "",
        "**Note:** Legacy `downsampled50_up/*_x4` (512→2048) is NOT valid for this protocol.",
        "",
        "| method | train | test | shape | status |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['method']} | {r['train_count']} | {r['test_count']} | {r['dominant_shape']} | {r['status']} |"
        )
    if job_notes:
        md_lines += ["", "## Job submission notes", ""] + [f"- {n}" for n in job_notes]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    for r in rows:
        print(f"{r['method']}: {r['status']}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
