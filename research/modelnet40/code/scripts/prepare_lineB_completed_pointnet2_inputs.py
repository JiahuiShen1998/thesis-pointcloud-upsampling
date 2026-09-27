#!/usr/bin/env python3
"""Prepare PointNet++ inputs for Line B completed methods (EAR, PDANS) only."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    PROJECT_ROOT,
    detect_original_point_count,
    lineB_paths,
    pointnet2_input_paths,
    reports_dir,
)

COMPLETED_METHODS = ("ear", "pdans")


def count_split(root: Path) -> dict[str, int]:
    return {
        "train": sum(1 for _ in (root / "train").rglob("*.npy")) if (root / "train").is_dir() else 0,
        "test": sum(1 for _ in (root / "test").rglob("*.npy")) if (root / "test").is_dir() else 0,
    }


def ensure_symlink(src: Path, dst: Path) -> str:
    if not src.is_dir():
        return "missing_source"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return "exists"
        return "exists_other"
    dst.symlink_to(src.resolve())
    return "linked"


def main() -> int:
    n = detect_original_point_count()
    down_n = n // 4
    paths = pointnet2_input_paths()
    rows = []

    # Baseline 256
    bl_src = DOWNSAMPLED_X4_ROOT
    bl_dst = paths["lineB_baseline"]
    st = ensure_symlink(bl_src, bl_dst)
    c = count_split(bl_src)
    rows.append({
        "branch": "baseline",
        "method": "downsampled_x4",
        "source_path": str(bl_src),
        "pointnet2_input_path": str(bl_dst),
        "train_samples": c["train"],
        "test_samples": c["test"],
        "point_count": down_n,
        "status": st,
        "dataloader_resamples_internally": "no — num_point must match 256",
    })

    for method in COMPLETED_METHODS:
        strict = lineB_paths(method)["strict_N"]
        dst = paths["lineB_up_root"] / method
        st = ensure_symlink(strict, dst)
        c = count_split(strict)
        rows.append({
            "branch": "upsampling",
            "method": method,
            "source_path": str(strict),
            "pointnet2_input_path": str(dst),
            "train_samples": c["train"],
            "test_samples": c["test"],
            "point_count": n,
            "status": st,
            "dataloader_resamples_internally": "no — strict_N already exact 1024",
        })

    rep = reports_dir()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    csv_path = rep / "modelnet40_lineB_pointnet2_input_audit_completed_methods.csv"
    md_path = rep / "modelnet40_lineB_pointnet2_input_audit_completed_methods.md"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Line B PointNet++ Input Audit — Completed Methods",
        "",
        f"- Generated at: {ts}",
        f"- Expected train/test: {EXPECTED_TRAIN} / {EXPECTED_TEST}",
        "",
        "| branch | method | point_count | train | test | status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['branch']} | {r['method']} | {r['point_count']} | {r['train_samples']} | {r['test_samples']} | {r['status']} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
