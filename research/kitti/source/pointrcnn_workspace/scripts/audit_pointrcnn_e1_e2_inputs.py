#!/usr/bin/env python3
"""Independent count and sampled multiset-preservation audit for E1/E2 inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = REPO / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
VAL = REPO / "data/KITTI/ImageSets/val.txt"
ORIGINAL = REPO / "data/KITTI/object/training/velodyne_original_val"
DOWNSAMPLED = (
    REPO
    / "results/kitti_unified_x4_current_methods_no_detector"
    / "downsampled_x4/velodyne_downsampled_x4_val"
)


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"invalid KITTI binary: {path}")
    return values.reshape(-1, 4)


def rows_as_bytes(points: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(points)
    return contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).reshape(-1)


def multiset_contains(container: np.ndarray, required: np.ndarray) -> bool:
    container_values, container_counts = np.unique(rows_as_bytes(container), return_counts=True)
    required_values, required_counts = np.unique(rows_as_bytes(required), return_counts=True)
    positions = np.searchsorted(container_values, required_values)
    in_bounds = positions < container_values.size
    if not in_bounds.all():
        return False
    return bool(
        np.array_equal(container_values[positions], required_values)
        and np.all(container_counts[positions] >= required_counts)
    )


def sampled_frames(frames: list[str], count: int) -> list[str]:
    if count >= len(frames):
        return frames
    positions = np.linspace(0, len(frames) - 1, num=count, dtype=np.int64)
    return [frames[int(position)] for position in positions]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--sample-frames", type=int, default=32)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    frames = [line.strip() for line in VAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample = sampled_frames(frames, args.sample_frames)
    failures: list[str] = []
    e1_results = []
    e2_results = []

    e1_root = workspace / "inputs/e1_default_16384"
    for variant_dir in sorted(path for path in e1_root.iterdir() if path.is_dir()):
        observed_dir = ORIGINAL if variant_dir.name.startswith("original_x4_") else DOWNSAMPLED
        files = list(variant_dir.glob("*.bin"))
        if len(files) != len(frames):
            failures.append(f"{variant_dir.name}: E1 file count {len(files)} != {len(frames)}")
        checked = 0
        for frame in sample:
            observed = read_bin(observed_dir / f"{frame}.bin")
            output = read_bin(variant_dir / f"{frame}.bin")
            if output.shape[0] != 4 * observed.shape[0]:
                failures.append(f"{variant_dir.name}/{frame}: output is not exact x4")
            if not multiset_contains(output, observed):
                failures.append(f"{variant_dir.name}/{frame}: observed XYZI multiset not preserved")
            checked += 1
        e1_results.append(
            {
                "variant": variant_dir.name,
                "full_file_count": len(files),
                "sampled_frames_checked": checked,
                "exact_x4_and_observed_multiset_preserved": not any(
                    item.startswith(f"{variant_dir.name}/") for item in failures
                ),
            }
        )

    e2_root = workspace / "inputs/e2_canonical_16384"
    for variant_dir in sorted(path for path in e2_root.iterdir() if path.is_dir()):
        files = list(variant_dir.glob("*.bin"))
        bad_size = [path.name for path in files if path.stat().st_size != 16384 * 4 * 4]
        if len(files) != len(frames):
            failures.append(f"{variant_dir.name}: E2 file count {len(files)} != {len(frames)}")
        if bad_size:
            failures.append(f"{variant_dir.name}: {len(bad_size)} E2 files are not exactly 16384x4 float32")
        e2_results.append(
            {
                "variant": variant_dir.name,
                "full_file_count": len(files),
                "all_files_exactly_16384_points": not bad_size,
            }
        )

    payload = {
        "status": "PASS" if not failures else "FAIL",
        "val_frame_count": len(frames),
        "sampled_frame_ids": sample,
        "e1": e1_results,
        "e2": e2_results,
        "failures": failures,
    }
    report = workspace / "reports/input_audit.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
