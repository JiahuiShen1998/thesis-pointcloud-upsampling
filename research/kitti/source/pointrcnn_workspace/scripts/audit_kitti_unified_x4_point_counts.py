#!/usr/bin/env python3
"""Audit KITTI unified x4 frame-level point counts."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "kitti_unified_x4_current_methods_no_detector"
TRAINING = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
DEFAULT_ORIGINAL = TRAINING / "velodyne_original_val"
DEFAULT_DOWNSAMPLED = RESULT_ROOT / "downsampled_x4" / "velodyne_downsampled_x4_val"
DEFAULT_AUDIT = RESULT_ROOT / "audits" / "frame_level_point_count_audit.csv"
METHOD_DIRS = ["pu_net", "pu_gcn", "pdans", "pu_edgeformer"]


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def point_count(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size % 16 != 0:
        return None
    return path.stat().st_size // 16


def add_row(rows: list[dict[str, object]], line: str, method: str, frame_id: str, path: Path, expected: int) -> None:
    observed = point_count(path)
    rows.append(
        {
            "line": line,
            "method": method,
            "frame_id": frame_id,
            "path": str(path),
            "expected_points": int(expected),
            "observed_points": "" if observed is None else int(observed),
            "exists": path.exists(),
            "file_size_mod16": "" if not path.exists() else path.stat().st_size % 16,
            "status": "PASS" if observed == expected else "FAIL",
        }
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["line", "method", "frame_id", "path", "expected_points", "observed_points", "exists", "file_size_mod16", "status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--original-dir", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--downsampled-x4-dir", type=Path, default=DEFAULT_DOWNSAMPLED)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--methods", nargs="*", default=METHOD_DIRS)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for frame_id in read_split(args.split):
        original = args.original_dir / f"{frame_id}.bin"
        downsampled = args.downsampled_x4_dir / f"{frame_id}.bin"
        n_points = point_count(original)
        if n_points is None:
            add_row(rows, "original_baseline", "baseline", frame_id, original, -1)
            continue
        m_points = n_points // 4
        add_row(rows, "original_baseline", "baseline", frame_id, original, n_points)
        add_row(rows, "downsampled_x4_baseline", "baseline", frame_id, downsampled, m_points)
        for method in args.methods:
            add_row(
                rows,
                "line_a_original_x4_up",
                method,
                frame_id,
                args.result_root / "line_a_original_x4_up" / method / "final_bin" / f"{frame_id}.bin",
                4 * n_points,
            )
            add_row(
                rows,
                "line_b_downsampled_x4_up",
                method,
                frame_id,
                args.result_root / "line_b_downsampled_x4_up" / method / "final_bin" / f"{frame_id}.bin",
                4 * m_points,
            )

    write_csv(args.audit_csv, rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    print(f"POINT_COUNT_AUDIT_ROWS={len(rows)}")
    print(f"POINT_COUNT_AUDIT_PASS={len(rows) - len(failures)}")
    print(f"POINT_COUNT_AUDIT_FAIL={len(failures)}")
    print(f"AUDIT_CSV={args.audit_csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
