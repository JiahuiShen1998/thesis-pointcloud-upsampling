#!/usr/bin/env python3
"""Prepare V4 measured-voxel anchored PDANS supplements.

V4 is a single-factor revision of Full V3: a candidate must occupy an exact
CenterPoint voxel that already contains at least one observed LiDAR point and
has remaining capacity below the native five-points-per-voxel limit.  All
other proposal, geometry, confidence, budget, seed and observed-first rules
remain frozen.  No GT is read.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import prepare_detector_aware_pdans_inputs as v1  # noqa: E402
import prepare_detector_aware_pdans_v2_inputs as v2  # noqa: E402
import prepare_detector_aware_pdans_v3_inputs as v3  # noqa: E402
import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402


CAUSAL = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
DEFAULT_WORKSPACE = REPO / "results/detector_aware_pdans_v4_20260805"
DEFAULT_SPLIT = CAUSAL / "splits/pilot256.txt"
DEFAULT_OBSERVED = REPO / "data/KITTI/object/training/velodyne_original_val"
DEFAULT_PDANS = CAUSAL / "surface_candidate256_v1/pdans_surface_cover_exact_pr1_c32/line_a_original_x4_up/final_bin"
DEFAULT_POINT = REPO / "results/detector_aware_pdans_v3_20260805/internal_proposals/pointrcnn_rpn_256/merged_kitti"
DEFAULT_CENTER = REPO / "results/detector_aware_pdans_v3_20260805/internal_proposals/centerpoint_prenms_256/npz"
VARIANTS = ("v4_existing_voxel_pointrcnn", "v4_existing_voxel_centerpoint")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_existing_voxel(
    frame: str,
    candidate_xyz: np.ndarray,
    geometry: dict[str, np.ndarray],
    voxels: dict[str, np.ndarray],
    proposal_score: np.ndarray,
    observed_count: int,
) -> tuple[np.ndarray, dict]:
    ratio = v1.adaptive_ratio(geometry, "full")
    target = int(round(observed_count * ratio))
    eligible = (
        (geometry["confidence"] >= 0.35)
        & geometry["absolute_support"]
        & (geometry["nearest_m"] <= v2.MAX_NEAREST_REAL_M)
        & (voxels["observed_points_in_voxel"] > 0)
        & (voxels["observed_points_in_voxel"] < v2.MAX_POINTS_PER_VOXEL)
        & (voxels["retention_capacity"] > 0)
        & (proposal_score > 0)
    )
    proximity = np.exp(-np.square(geometry["nearest_m"] / 0.10))
    remaining = np.clip(
        (v2.MAX_POINTS_PER_VOXEL - voxels["observed_points_in_voxel"])
        / v2.MAX_POINTS_PER_VOXEL,
        0.0, 1.0,
    )
    priority = geometry["utility"] * proposal_score * proximity * (0.5 + 0.5 * remaining)
    selected = v2.exact_voxel_select(
        voxels["candidate_coord"], priority, eligible,
        voxels["retention_capacity"], target,
        v2.stable_seed("v4_existing_voxel", frame),
    )
    return selected, {
        "requested_ratio": ratio, "requested_generated": target,
        "eligible_candidates": int(np.count_nonzero(eligible)),
        "selected_generated": int(len(selected)),
        "realized_ratio": len(selected) / max(observed_count, 1),
        "selected_nearest_real_mean_m": float(geometry["nearest_m"][selected].mean()) if len(selected) else 0.0,
        "selected_nearest_real_p90_m": float(np.quantile(geometry["nearest_m"][selected], 0.90)) if len(selected) else 0.0,
        "selected_confidence_mean": float(geometry["confidence"][selected].mean()) if len(selected) else 0.0,
        "selected_inside_proposal_fraction": float(np.mean(proposal_score[selected] > 0)) if len(selected) else 0.0,
        "selected_existing_observed_voxel_fraction": float(np.mean(voxels["observed_points_in_voxel"][selected] > 0)) if len(selected) else 0.0,
        "selected_generated_only_voxel_fraction": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--observed", type=Path, default=DEFAULT_OBSERVED)
    parser.add_argument("--pdans", type=Path, default=DEFAULT_PDANS)
    parser.add_argument("--point-proposals", type=Path, default=DEFAULT_POINT)
    parser.add_argument("--center-proposals", type=Path, default=DEFAULT_CENTER)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()

    frames = v1.read_frames(args.split_file.resolve())
    if args.limit:
        frames = frames[: args.limit]
    workspace = args.workspace.resolve()
    tag = "20" if args.limit else "256"
    input_root = workspace / ("inputs20" if args.limit else "inputs")
    rows = []
    for position, frame in enumerate(frames, start=1):
        observed = prep.read_bin(args.observed / f"{frame}.bin")
        candidates = prep.read_bin(args.pdans / f"{frame}.bin")
        observed_mask, unused = prep.fov_valid_mask(observed, frame)
        candidate_mask, candidate_rect = prep.fov_valid_mask(candidates, frame)
        observed_valid = observed[observed_mask]
        candidate_valid = candidates[candidate_mask]
        rect_valid = candidate_rect[candidate_mask]
        geometry = v1.geometry_features(observed_valid[:, :3].astype(np.float64),
                                        candidate_valid[:, :3].astype(np.float64),
                                        rect_valid[:, 2].astype(np.float64))
        voxel = v2.voxel_features(observed_valid[:, :3], candidate_valid[:, :3])
        proposal_sets = {
            "pointrcnn": v3.point_proposals(frame, args.point_proposals),
            "centerpoint": v3.center_proposals(frame, args.center_proposals),
        }
        for source, output_name in (("pointrcnn", VARIANTS[0]), ("centerpoint", VARIANTS[1])):
            scores = v2.proposal_scores(candidate_valid[:, :3].astype(np.float64), proposal_sets[source])
            selected, stats = select_existing_voxel(
                frame, candidate_valid[:, :3], geometry, voxel, scores, len(observed_valid)
            )
            final = np.concatenate((observed, candidate_valid[selected]), axis=0)
            if not np.isfinite(final).all() or not np.array_equal(final[: len(observed)], observed):
                raise RuntimeError(f"{frame}/{output_name}: invariant failed")
            if not args.geometry_only:
                output = input_root / output_name / f"{frame}.bin"
                output.parent.mkdir(parents=True, exist_ok=True)
                final.astype(np.float32, copy=False).tofile(output)
            rows.append({
                "frame_id": frame, "variant": output_name, "proposal_source": source,
                "internal_proposals": len(proposal_sets[source]),
                "observed_total_preserved": len(observed), "observed_valid": len(observed_valid),
                "candidate_valid": len(candidate_valid), "output_total": len(final), **stats,
                "status": "PASS",
            })
        if position == 1 or position % 10 == 0 or position == len(frames):
            print(f"PDANS_V4_PREP {position}/{len(frames)} frame={frame}", flush=True)

    manifest_root = workspace / f"manifests{tag}"
    write_csv(manifest_root / "per_frame.csv", rows)
    aggregates = []
    gate = True
    for variant in VARIANTS:
        selected_rows = [row for row in rows if row["variant"] == variant]
        nonempty_rows = [row for row in selected_rows if row["selected_generated"] > 0]
        item = {
            "variant": variant, "frames": len(selected_rows),
            "frames_with_safe_baseline_fallback": len(selected_rows) - len(nonempty_rows),
            "safe_baseline_fallback_fraction": (len(selected_rows) - len(nonempty_rows)) / len(selected_rows),
            "eligible_candidates_min": min(row["eligible_candidates"] for row in selected_rows),
            "eligible_candidates_median": float(np.median([row["eligible_candidates"] for row in selected_rows])),
            "selected_generated_min": min(row["selected_generated"] for row in selected_rows),
            "selected_generated_median": float(np.median([row["selected_generated"] for row in selected_rows])),
            "realized_ratio_median": float(np.median([row["realized_ratio"] for row in selected_rows])),
            "nearest_real_p90_max": max(row["selected_nearest_real_p90_m"] for row in selected_rows),
            "existing_observed_voxel_fraction_min_nonempty": min(row["selected_existing_observed_voxel_fraction"] for row in nonempty_rows),
            "generated_only_voxel_fraction_max": max(row["selected_generated_only_voxel_fraction"] for row in selected_rows),
        }
        item["gate_pass"] = bool(
            item["safe_baseline_fallback_fraction"] <= 0.05
            and item["selected_generated_median"] > 0
            and item["nearest_real_p90_max"] <= v2.MAX_NEAREST_REAL_M + 1e-9
            and item["existing_observed_voxel_fraction_min_nonempty"] == 1.0
            and item["generated_only_voxel_fraction_max"] == 0.0
        )
        gate &= item["gate_pass"]
        aggregates.append(item)
    write_csv(manifest_root / "aggregate.csv", aggregates)
    protocol = {
        "experiment": f"detector_aware_pdans_v4_existing_measured_voxel_{len(frames)}frame_gate",
        "status": "PASS" if gate else "FAIL", "frames": len(frames),
        "uses_gt_labels_boxes_or_ap": False, "all_observed_points_preserved": True,
        "safe_baseline_fallback_policy": "if no candidate passes, keep the observed-only frame; at most 5% frames",
        "single_change_vs_v3_full": "require exact 0.05x0.05x0.1m CenterPoint voxel to contain observed point and have capacity below 5",
        "internal_proposals_geometry_confidence_budget_seed_frozen": True,
        "variants": aggregates,
    }
    (workspace / f"protocol{tag}.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2), flush=True)
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
