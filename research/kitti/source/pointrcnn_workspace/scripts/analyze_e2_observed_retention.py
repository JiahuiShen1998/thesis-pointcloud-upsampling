#!/usr/bin/env python3
"""Measure how many observed KITTI points survive in E2 detector inputs.

E1 stores N observed + 3N generated points, but PointRCNN consumes only 16384.
This audit reports an upper bound on exact observed-point retention in the
saved E2 inputs, plus observed 0.1 m voxel coverage.  Exact generated points
that happen to equal an observed XYZI row are conservatively attributed to the
observed set, so the reported observed fraction is an upper bound.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402


WORKSPACE = REPO / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
E2_INPUTS = WORKSPACE / "inputs/e2_canonical_16384"
E2_MANIFESTS = WORKSPACE / "manifests/e2_canonical_16384"
METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
SPECS = {
    "A": {
        "observed": prep.ORIGINAL,
        "baseline": "original_baseline",
        "variants": tuple(f"original_x4_{method}" for method in METHODS),
    },
    "B": {
        "observed": prep.DOWNSAMPLED,
        "baseline": "downsampled_x4_baseline",
        "variants": tuple(f"downsampled_x4_{method}" for method in METHODS),
    },
}


def row_keys(points: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(points, dtype=np.float32)
    dtype = np.dtype((np.void, values.dtype.itemsize * values.shape[1]))
    return values.view(dtype).ravel()


def multiset_intersection_count(reference: np.ndarray, selected: np.ndarray) -> int:
    ref_key, ref_count = np.unique(row_keys(reference), return_counts=True)
    sel_key, sel_count = np.unique(row_keys(selected), return_counts=True)
    _, ref_idx, sel_idx = np.intersect1d(
        ref_key, sel_key, assume_unique=True, return_indices=True
    )
    return int(np.minimum(ref_count[ref_idx], sel_count[sel_idx]).sum())


def unique_voxel_keys(rect_xyz: np.ndarray, size: float = 0.1) -> np.ndarray:
    quantized = np.ascontiguousarray(np.floor(rect_xyz / size).astype(np.int32))
    dtype = np.dtype((np.void, quantized.dtype.itemsize * quantized.shape[1]))
    return np.unique(quantized.view(dtype).ravel())


def voxel_coverage(reference_rect: np.ndarray, selected_rect: np.ndarray) -> float:
    reference = unique_voxel_keys(reference_rect)
    selected = unique_voxel_keys(selected_rect)
    overlap = np.intersect1d(reference, selected, assume_unique=True).size
    return float(overlap / reference.size) if reference.size else float("nan")


def load_manifest(variant: str) -> dict[str, dict[str, str]]:
    path = E2_MANIFESTS / f"{variant}.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["frame_id"]: row for row in csv.DictReader(handle)}


def choose_frames(count: int) -> list[str]:
    frames = prep.read_frames()
    if count >= len(frames):
        return frames
    positions = np.linspace(0, len(frames) - 1, count, dtype=np.int64)
    return [frames[int(index)] for index in np.unique(positions)]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-frames", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "reports/e2_observed_retention_v1",
    )
    args = parser.parse_args()

    frames = choose_frames(args.sample_frames)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_frame: list[dict] = []

    for line, spec in SPECS.items():
        variants = (spec["baseline"],) + spec["variants"]
        manifests = {variant: load_manifest(variant) for variant in variants}
        observed_dir = Path(spec["observed"])

        for position, frame in enumerate(frames, start=1):
            observed = prep.read_bin(observed_dir / f"{frame}.bin")
            observed_mask, observed_rect_all = prep.fov_valid_mask(observed, frame)
            observed_valid = observed[observed_mask]
            observed_rect = observed_rect_all[observed_mask]

            for variant in variants:
                selected = prep.read_bin(E2_INPUTS / variant / f"{frame}.bin")
                selected_mask, selected_rect_all = prep.fov_valid_mask(selected, frame)
                if not bool(selected_mask.all()):
                    raise ValueError(f"{line}/{variant}/{frame}: E2 contains invalid points")
                exact_observed = multiset_intersection_count(observed_valid, selected)
                source_fov = int(manifests[variant][frame]["fov_valid_points"])
                baseline = variant == spec["baseline"]
                per_frame.append(
                    {
                        "line": line,
                        "variant": variant,
                        "kind": "baseline" if baseline else "upsampled",
                        "frame_id": frame,
                        "observed_fov_points": int(observed_valid.shape[0]),
                        "e1_source_fov_points": source_fov,
                        "e1_observed_fraction_in_valid_4n": (
                            1.0 if baseline else float(observed_valid.shape[0] / source_fov)
                        ),
                        "e2_selected_points": int(selected.shape[0]),
                        "e2_exact_observed_points_upper": exact_observed,
                        "e2_exact_observed_fraction_upper": float(exact_observed / selected.shape[0]),
                        "e2_non_observed_fraction_lower": float(1.0 - exact_observed / selected.shape[0]),
                        "e2_observed_point_retention_upper": float(
                            exact_observed / observed_valid.shape[0]
                        ),
                        "e2_observed_voxel_0p1_coverage": voxel_coverage(
                            observed_rect, selected_rect_all
                        ),
                        "fill_policy": manifests[variant][frame]["fill_policy"],
                    }
                )
            if position == 1 or position % 32 == 0 or position == len(frames):
                print(f"observed retention {line}: {position}/{len(frames)} frames", flush=True)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in per_frame:
        grouped[(row["line"], row["variant"], row["kind"])].append(row)

    summary: list[dict] = []
    metrics = (
        "observed_fov_points",
        "e1_source_fov_points",
        "e1_observed_fraction_in_valid_4n",
        "e2_exact_observed_points_upper",
        "e2_exact_observed_fraction_upper",
        "e2_non_observed_fraction_lower",
        "e2_observed_point_retention_upper",
        "e2_observed_voxel_0p1_coverage",
    )
    for (line, variant, kind), rows in sorted(grouped.items()):
        output = {"line": line, "variant": variant, "kind": kind, "frames": len(rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            output[f"{metric}_median"] = quantile(values, 0.5)
            output[f"{metric}_p10"] = quantile(values, 0.1)
            output[f"{metric}_p90"] = quantile(values, 0.9)
        summary.append(output)

    write_csv(output_dir / "e2_observed_retention_per_frame.csv", per_frame)
    write_csv(output_dir / "e2_observed_retention_summary.csv", summary)
    print(f"E2_OBSERVED_RETENTION_PASS output={output_dir} frames={len(frames)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
