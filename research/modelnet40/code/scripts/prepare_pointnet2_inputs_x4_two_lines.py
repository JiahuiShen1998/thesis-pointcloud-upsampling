#!/usr/bin/env python3
"""Prepare PointNet++ inputs for both experiment lines."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    MAIN_METHODS,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    lineA_paths,
    lineB_paths,
    pointnet2_input_paths,
    protocol_counts,
    reports_dir,
)


def ensure_symlink(src: Path, dst: Path) -> str:
    if not src.is_dir():
        return "missing_source"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    dst.symlink_to(src.resolve())
    return "linked"


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    n = detect_original_point_count()
    counts = protocol_counts(n)
    paths = pointnet2_input_paths()
    rows = []

    # Line A baseline
    status = ensure_symlink(ORIGINAL_ROOT, paths["lineA_baseline"])
    rows.append({
        "line": "A", "branch": "baseline", "method": "original",
        "input_path": str(paths["lineA_baseline"]),
        "source_path": str(ORIGINAL_ROOT),
        "point_count": n, "status": status,
    })

    # Line A upsampling
    for method in MAIN_METHODS:
        strict = lineA_paths(method)["strict_4N"]
        dst = paths["lineA_up_root"] / method
        status = ensure_symlink(strict, dst)
        rows.append({
            "line": "A", "branch": "upsampling", "method": method,
            "input_path": str(dst), "source_path": str(strict),
            "point_count": counts["four_N"], "status": status,
        })

    # Line B baseline
    status = ensure_symlink(DOWNSAMPLED_X4_ROOT, paths["lineB_baseline"])
    rows.append({
        "line": "B", "branch": "baseline", "method": "downsampled_x4",
        "input_path": str(paths["lineB_baseline"]),
        "source_path": str(DOWNSAMPLED_X4_ROOT),
        "point_count": counts["N_div_4"], "status": status,
    })

    # Line B upsampling
    for method in MAIN_METHODS:
        strict = lineB_paths(method)["strict_N"]
        dst = paths["lineB_up_root"] / method
        status = ensure_symlink(strict, dst) if strict.is_dir() else "pending"
        rows.append({
            "line": "B", "branch": "upsampling", "method": method,
            "input_path": str(dst), "source_path": str(strict),
            "point_count": n, "status": status,
        })

    rep = reports_dir()
    csv_path = rep / "modelnet40_pointnet2_inputs_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md = rep / "modelnet40_pointnet2_inputs_report.md"
    md.write_text(
        "\n".join([
            "# PointNet++ Inputs Manifest",
            "",
            f"- Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Dataloader note",
            "",
            "PointNet++ `ModelNetNPYDataset` uses `num_point` and `allow_resample`.",
            "When `allow_resample=false`, samples must match `num_point` exactly on disk.",
            "Set `num_point` per variant: N, 4N, N/4 as listed below.",
            "",
            f"| line | branch | method | points | status | path |",
            f"| --- | --- | --- | ---: | --- | --- |",
        ] + [
            f"| {r['line']} | {r['branch']} | {r['method']} | {r['point_count']} | {r['status']} | `{r['input_path']}` |"
            for r in rows
        ]),
        encoding="utf-8",
    )
    print(f"Prepared {len(rows)} input mappings -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
