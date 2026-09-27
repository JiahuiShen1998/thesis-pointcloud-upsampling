#!/usr/bin/env python3
"""Independently verify strict x4 KITTI XYZI outputs for a frame split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-file", type=Path, required=True)
    parser.add_argument("--observed-dir", type=Path, required=True)
    parser.add_argument("--predicted-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def read_xyzi(path: Path) -> tuple[np.ndarray | None, str | None]:
    if not path.is_file():
        return None, "missing"
    byte_count = path.stat().st_size
    if byte_count == 0:
        return None, "empty"
    if byte_count % 16:
        return None, f"size_not_xyzi:{byte_count}"
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        return None, f"float_count_not_xyzi:{values.size}"
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        return points, "nan_or_inf"
    return points, None


def main() -> int:
    args = parse_args()
    frames = [
        line.strip()
        for line in args.frames_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError(f"invalid frame list: {args.frames_file}")

    rows: list[dict[str, object]] = []
    for frame in frames:
        observed, observed_error = read_xyzi(args.observed_dir / f"{frame}.bin")
        predicted, predicted_error = read_xyzi(args.predicted_dir / f"{frame}.bin")
        observed_points = None if observed is None else int(len(observed))
        predicted_points = None if predicted is None else int(len(predicted))
        exact_x4 = bool(
            observed_error is None
            and predicted_error is None
            and predicted_points == 4 * observed_points
        )
        errors = [
            item
            for item in (
                f"observed:{observed_error}" if observed_error else None,
                f"predicted:{predicted_error}" if predicted_error else None,
                None
                if exact_x4 or observed_error or predicted_error
                else f"ratio:{predicted_points}/{observed_points}",
            )
            if item is not None
        ]
        rows.append(
            {
                "frame_id": frame,
                "observed_points": observed_points,
                "predicted_points": predicted_points,
                "expected_points": (
                    None if observed_points is None else 4 * observed_points
                ),
                "exact_x4": exact_x4,
                "finite": observed_error is None and predicted_error is None,
                "status": "PASS" if exact_x4 and not errors else "FAIL",
                "error": ";".join(errors),
            }
        )

    failures = [row for row in rows if row["status"] != "PASS"]
    summary = {
        "status": "PASS" if not failures else "FAIL",
        "frames_expected": len(frames),
        "frames_checked": len(rows),
        "frames_pass": len(rows) - len(failures),
        "frames_fail": len(failures),
        "observed_dir": str(args.observed_dir.resolve()),
        "predicted_dir": str(args.predicted_dir.resolve()),
        "total_observed_points": sum(
            int(row["observed_points"] or 0) for row in rows
        ),
        "total_predicted_points": sum(
            int(row["predicted_points"] or 0) for row in rows
        ),
        "failures": failures,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(
        f"{summary['status']} frames={len(rows)} pass={summary['frames_pass']} "
        f"fail={summary['frames_fail']}",
        flush=True,
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
