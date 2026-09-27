#!/usr/bin/env python3
"""Create KITTI val downsampled-x4 input with M=floor(N/4) per frame."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_original_val"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "kitti_unified_x4_current_methods_no_detector"
    / "downsampled_x4"
    / "velodyne_downsampled_x4_val"
)
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "kitti_unified_x4_current_methods_no_detector"
    / "manifests"
    / "downsampled_x4_manifest.csv"
)


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_kitti_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(f"{path} is not KITTI XYZI float32")
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN or Inf")
    return points


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "frame_id",
        "source_path",
        "output_path",
        "original_point_count_N",
        "downsampled_point_count_M",
        "expected_M_floor_N_div_4",
        "count_rule_pass",
        "seed",
        "status",
        "failure_reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frames = read_split(args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures = 0

    for frame_id in frames:
        source = args.input_dir / f"{frame_id}.bin"
        output = args.output_dir / f"{frame_id}.bin"
        row: dict[str, object] = {
            "frame_id": frame_id,
            "source_path": str(source),
            "output_path": str(output),
            "seed": args.seed,
            "status": "STARTED",
        }
        try:
            if output.exists() and not args.overwrite:
                points_existing = read_kitti_bin(output)
                source_points = read_kitti_bin(source)
                expected = source_points.shape[0] // 4
                row.update(
                    {
                        "original_point_count_N": int(source_points.shape[0]),
                        "downsampled_point_count_M": int(points_existing.shape[0]),
                        "expected_M_floor_N_div_4": int(expected),
                        "count_rule_pass": bool(points_existing.shape[0] == expected),
                        "status": "SKIP_EXISTING_PASS" if points_existing.shape[0] == expected else "FAIL_EXISTING_COUNT",
                    }
                )
                if points_existing.shape[0] != expected:
                    failures += 1
                rows.append(row)
                continue

            points = read_kitti_bin(source)
            n_points = points.shape[0]
            target = n_points // 4
            if target <= 0:
                raise ValueError(f"source has too few points for floor(N/4): N={n_points}")
            rng = np.random.default_rng(args.seed + int(frame_id))
            indices = rng.choice(n_points, size=target, replace=False)
            indices.sort()
            downsampled = points[indices].astype(np.float32, copy=False)
            downsampled.tofile(output)
            row.update(
                {
                    "original_point_count_N": int(n_points),
                    "downsampled_point_count_M": int(downsampled.shape[0]),
                    "expected_M_floor_N_div_4": int(target),
                    "count_rule_pass": bool(downsampled.shape[0] == target),
                    "status": "PASS",
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures += 1
            row["status"] = "FAIL"
            row["failure_reason"] = str(exc)
        rows.append(row)

    write_manifest(args.manifest, rows)
    passed = sum(1 for row in rows if str(row.get("status", "")).endswith("PASS") or row.get("status") == "PASS")
    print(f"DOWNSAMPLED_X4_ROWS={len(rows)}")
    print(f"DOWNSAMPLED_X4_PASS={passed}")
    print(f"DOWNSAMPLED_X4_FAIL={failures}")
    print(f"MANIFEST={args.manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
