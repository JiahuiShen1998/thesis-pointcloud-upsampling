#!/usr/bin/env python3
"""Corrected E2 observed-content and unique-observation retention audit."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

import analyze_e2_observed_retention as base


def membership_count(reference: np.ndarray, selected: np.ndarray) -> int:
    """Count selected rows equal to any observed row, including repeat fills."""
    reference_key = np.unique(base.row_keys(reference))
    return int(np.isin(base.row_keys(selected), reference_key, assume_unique=False).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-frames", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base.WORKSPACE / "reports/e2_observed_retention_v2",
    )
    args = parser.parse_args()

    frames = base.choose_frames(args.sample_frames)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_frame: list[dict] = []

    for line, spec in base.SPECS.items():
        variants = (spec["baseline"],) + spec["variants"]
        manifests = {variant: base.load_manifest(variant) for variant in variants}
        observed_dir = Path(spec["observed"])

        for position, frame in enumerate(frames, start=1):
            observed = base.prep.read_bin(observed_dir / f"{frame}.bin")
            observed_mask, observed_rect_all = base.prep.fov_valid_mask(observed, frame)
            observed_valid = observed[observed_mask]
            observed_rect = observed_rect_all[observed_mask]

            for variant in variants:
                selected = base.prep.read_bin(base.E2_INPUTS / variant / f"{frame}.bin")
                selected_mask, selected_rect_all = base.prep.fov_valid_mask(selected, frame)
                if not bool(selected_mask.all()):
                    raise ValueError(f"{line}/{variant}/{frame}: E2 contains invalid points")

                exact_observed_rows = membership_count(observed_valid, selected)
                unique_observed_retained = base.multiset_intersection_count(
                    observed_valid, selected
                )
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
                        "e2_exact_observed_rows_upper": exact_observed_rows,
                        "e2_exact_observed_fraction_upper": float(
                            exact_observed_rows / selected.shape[0]
                        ),
                        "e2_non_observed_fraction_lower": float(
                            1.0 - exact_observed_rows / selected.shape[0]
                        ),
                        "e2_unique_observed_points_retained_upper": unique_observed_retained,
                        "e2_observed_point_retention_upper": float(
                            unique_observed_retained / observed_valid.shape[0]
                        ),
                        "e2_observed_voxel_0p1_coverage": base.voxel_coverage(
                            observed_rect, selected_rect_all
                        ),
                        "fill_policy": manifests[variant][frame]["fill_policy"],
                    }
                )
            if position == 1 or position % 32 == 0 or position == len(frames):
                print(f"observed retention v2 {line}: {position}/{len(frames)} frames", flush=True)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in per_frame:
        grouped[(row["line"], row["variant"], row["kind"])].append(row)

    summary: list[dict] = []
    metrics = (
        "observed_fov_points",
        "e1_source_fov_points",
        "e1_observed_fraction_in_valid_4n",
        "e2_exact_observed_rows_upper",
        "e2_exact_observed_fraction_upper",
        "e2_non_observed_fraction_lower",
        "e2_unique_observed_points_retained_upper",
        "e2_observed_point_retention_upper",
        "e2_observed_voxel_0p1_coverage",
    )
    for (line, variant, kind), rows in sorted(grouped.items()):
        output = {"line": line, "variant": variant, "kind": kind, "frames": len(rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            output[f"{metric}_median"] = base.quantile(values, 0.5)
            output[f"{metric}_p10"] = base.quantile(values, 0.1)
            output[f"{metric}_p90"] = base.quantile(values, 0.9)
        summary.append(output)

    base.write_csv(output_dir / "e2_observed_retention_per_frame.csv", per_frame)
    base.write_csv(output_dir / "e2_observed_retention_summary.csv", summary)
    print(f"E2_OBSERVED_RETENTION_V2_PASS output={output_dir} frames={len(frames)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
