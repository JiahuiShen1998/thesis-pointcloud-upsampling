#!/usr/bin/env python3
"""Evaluate native-point-count TULIP outputs with the frozen CenterPoint.

TULIP is deliberately kept outside the exact-4N E1 matrix: its KITTI exports
contain a native, range-image-derived number of points rather than exactly 4N.
This wrapper reuses the isolated-overlay and audit machinery from the exact-4N
runner while writing to a separate result root.
"""

from __future__ import annotations

from pathlib import Path

import run_centerpoint_exact4n_matrix as matrix


REPO = Path(__file__).resolve().parents[1]

matrix.RESULT_ROOT = REPO / "results/centerpoint_tulip_native_extended_20260729"
matrix.VARIANTS = (
    matrix.Variant(
        "tulip_original_native",
        "line_a_extended",
        "tulip",
        REPO / "data/KITTI/object/training/tulip_original_up_bin",
        "tulip_native_point_count",
    ),
    matrix.Variant(
        "tulip_downsampled_native",
        "line_b_extended",
        "tulip",
        REPO / "data/KITTI/object/training/tulip_downsampled_up_bin",
        "tulip_native_point_count",
    ),
)


if __name__ == "__main__":
    raise SystemExit(matrix.main())
