#!/usr/bin/env python3
"""Prepare PointNet++ inputs for PU-GCN two-line protocol outputs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    PROJECT_ROOT,
    detect_original_point_count,
    lineA_paths,
    lineB_paths,
    pointnet2_input_paths,
    reports_dir,
)

EXPECTED_TOTAL = EXPECTED_TRAIN + EXPECTED_TEST


def ensure_symlink(src: Path, dst: Path) -> tuple[str, bool]:
    if not src.is_dir():
        return "missing_source", False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        valid = dst.resolve() == src.resolve()
        return ("linked" if valid else "broken_symlink"), valid
    if dst.exists():
        return "exists_other", dst.resolve() == src.resolve()
    dst.symlink_to(src.resolve())
    return "linked", True


def count_npy(root: Path) -> int:
    if not root.is_dir() and not root.is_symlink():
        return 0
    return sum(1 for _ in root.rglob("*.npy"))


def audit_one(line: str, method: str, strict: Path, dst: Path, exp_pts: int) -> dict:
    st, valid = ensure_symlink(strict, dst)
    bad = nan_c = inf_c = 0
    for npy in strict.rglob("*.npy"):
        arr = np.load(npy)
        if arr.shape != (exp_pts, 3):
            bad += 1
        if np.isnan(arr).any():
            nan_c += 1
        if np.isinf(arr).any():
            inf_c += 1

    src_count = count_npy(strict)
    dst_count = count_npy(dst)
    link_type = "symlink" if dst.is_symlink() else ("copy" if dst.exists() else "missing")

    status = "PASS"
    if st in ("missing_source", "broken_symlink", "exists_other"):
        status = "FAIL"
    elif src_count != EXPECTED_TOTAL or dst_count != EXPECTED_TOTAL:
        status = "FAIL"
    elif bad or nan_c or inf_c:
        status = "FAIL"

    return {
        "line": line,
        "method": method,
        "source_path": str(strict),
        "pointnet2_input_path": str(dst),
        "expected_point_count": exp_pts,
        "source_file_count": src_count,
        "target_file_count": dst_count,
        "link_type": link_type,
        "symlink_valid": valid if link_type == "symlink" else "",
        "bad_shape_count": bad,
        "nan_count": nan_c,
        "inf_count": inf_c,
        "status": status,
    }


def main() -> int:
    n = detect_original_point_count()
    four_n = n * 4
    p2 = pointnet2_input_paths()

    rows = [
        audit_one(
            "lineA",
            "pu_gcn",
            lineA_paths("pu_gcn")["strict_4N"],
            p2["lineA_up_root"] / "pu_gcn",
            four_n,
        ),
        audit_one(
            "lineB",
            "pu_gcn",
            lineB_paths("pu_gcn")["strict_N"],
            p2["lineB_up_root"] / "pu_gcn",
            n,
        ),
    ]

    rep = reports_dir()
    csv_path = rep / "modelnet40_pointnet2_input_audit.csv"
    md_path = rep / "modelnet40_pointnet2_input_audit.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Merge with existing rows if present (keep non-pu_gcn entries)
    existing: list[dict] = []
    if csv_path.is_file():
        with open(csv_path, newline="", encoding="utf-8") as handle:
            existing = [r for r in csv.DictReader(handle) if r.get("method") != "pu_gcn"]
    all_rows = existing + rows

    fields = list(all_rows[0].keys()) if all_rows else list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    overall = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    md_lines = [
        "# PointNet++ Input Audit",
        "",
        f"- Updated at: {ts}",
        f"- PU-GCN entries: **{overall}**",
        "",
        "| line | method | source_count | target_count | points | link | status |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['line']} | {r['method']} | {r['source_file_count']} | {r['target_file_count']} | "
            f"{r['expected_point_count']} | {r['link_type']} | {r['status']} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"PU-GCN pointnet2 input audit: {overall}")
    print(f"Wrote {csv_path}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
