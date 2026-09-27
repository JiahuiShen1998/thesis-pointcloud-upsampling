#!/usr/bin/env python3
"""Final count audit for ModelNet40 PU-GCN Line A / Line B strict outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pugcn_paths import resolve_project_root  # noqa: E402

EXPECTED_TOTAL = 12311


def audit_line(project_root: Path, line: str) -> list[dict]:
    if line == "lineA":
        strict_root = project_root / "datasets" / "lineA_original_up" / "strict_4N" / "pu_gcn"
        raw_root = project_root / "datasets" / "lineA_original_up" / "raw" / "pu_gcn"
        expected_shape = (4096, 3)
    else:
        strict_root = project_root / "datasets" / "lineB_downsampled_x4_up" / "strict_N" / "pu_gcn"
        raw_root = project_root / "datasets" / "lineB_downsampled_x4_up" / "raw" / "pu_gcn"
        expected_shape = (1024, 3)

    rows = []
    for split in ("train", "test"):
        split_dir = strict_root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            strict_files = sorted(class_dir.glob("*.npy"))
            raw_files = sorted((raw_root / split / class_dir.name).glob("*.npy")) if (raw_root / split / class_dir.name).is_dir() else []
            shapes = []
            nan_count = inf_count = 0
            for sf in strict_files:
                arr = np.load(sf)
                shapes.append(arr.shape[0])
                nan_count += int(not np.isfinite(arr).all())
                inf_count += int(np.isinf(arr).any())
            exact = sum(1 for s in strict_files if np.load(s).shape == expected_shape)
            status = "PASS" if (
                len(strict_files) > 0
                and exact == len(strict_files)
                and nan_count == 0
                and inf_count == 0
            ) else "FAIL"
            rows.append(
                {
                    "method": "PU-GCN",
                    "line": line,
                    "split": split,
                    "class": class_dir.name,
                    "raw_count": len(raw_files),
                    "strict_count": len(strict_files),
                    "expected_count": EXPECTED_TOTAL,
                    "min_points": min(shapes) if shapes else "",
                    "max_points": max(shapes) if shapes else "",
                    "exact_point_count_samples": exact,
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "missing_count": max(0, EXPECTED_TOTAL - sum(r["strict_count"] for r in rows) - len(strict_files)),
                    "status": status,
                }
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args()
    project_root = resolve_project_root(Path(args.project_root) if args.project_root else None)
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    rows = audit_line(project_root, "lineA") + audit_line(project_root, "lineB")
    csv_path = reports / "modelnet40_pugcn_final_count_audit.csv"
    md_path = reports / "modelnet40_pugcn_final_count_audit.md"
    fields = list(rows[0].keys()) if rows else [
        "method", "line", "split", "class", "raw_count", "strict_count",
        "expected_count", "min_points", "max_points", "exact_point_count_samples",
        "nan_count", "inf_count", "missing_count", "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    total_a = sum(r["strict_count"] for r in rows if r["line"] == "lineA")
    total_b = sum(r["strict_count"] for r in rows if r["line"] == "lineB")
    pass_a = total_a == EXPECTED_TOTAL and all(r["status"] == "PASS" for r in rows if r["line"] == "lineA")
    pass_b = total_b == EXPECTED_TOTAL and all(r["status"] == "PASS" for r in rows if r["line"] == "lineB")

    md = [
        "# ModelNet40 PU-GCN Final Count Audit",
        "",
        f"- Line A strict count: `{total_a}` / `{EXPECTED_TOTAL}` -> **{'PASS' if pass_a else 'FAIL'}**",
        f"- Line B strict count: `{total_b}` / `{EXPECTED_TOTAL}` -> **{'PASS' if pass_b else 'FAIL'}**",
        "",
        f"- CSV: `{csv_path}`",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0 if pass_a and pass_b else 1


if __name__ == "__main__":
    raise SystemExit(main())
