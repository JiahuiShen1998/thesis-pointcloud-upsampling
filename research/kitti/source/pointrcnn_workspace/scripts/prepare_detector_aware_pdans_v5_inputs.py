#!/usr/bin/env python3
"""Prepare V5 predicted-class adaptive CenterPoint input.

V5 freezes V4's measured-voxel anchor and all geometry/budget rules, but spends
the CenterPoint supplement budget only inside internal pre-NMS candidates whose
predicted class is Car.  Labels are detector predictions, never GT labels.
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
import prepare_detector_aware_pdans_v4_inputs as v4  # noqa: E402
import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402


CAUSAL = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
DEFAULT_WORKSPACE = REPO / "results/detector_aware_pdans_v5_20260805"
DEFAULT_SPLIT = CAUSAL / "splits/pilot256.txt"
DEFAULT_OBSERVED = REPO / "data/KITTI/object/training/velodyne_original_val"
DEFAULT_PDANS = CAUSAL / "surface_candidate256_v1/pdans_surface_cover_exact_pr1_c32/line_a_original_x4_up/final_bin"
DEFAULT_PROPOSALS = REPO / "results/detector_aware_pdans_v3_20260805/internal_proposals/centerpoint_prenms_256/npz"
VARIANT = "v5_car_predicted_existing_voxel_centerpoint"


def car_proposals(frame: str, root: Path) -> list[dict]:
    payload = np.load(root / f"{frame}.npz")
    output = []
    for box, score, label in zip(payload["boxes_lidar"], payload["scores"], payload["labels"]):
        if int(label) != 1 or float(score) < 0.10:
            continue
        values = np.asarray(box, dtype=np.float64)
        output.append({"center": values[:3], "dims": values[3:6],
                       "heading": float(values[6]), "score": float(score)})
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--observed", type=Path, default=DEFAULT_OBSERVED)
    parser.add_argument("--pdans", type=Path, default=DEFAULT_PDANS)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    frames = v1.read_frames(args.split_file.resolve())
    if args.limit:
        frames = frames[: args.limit]
    workspace = args.workspace.resolve()
    tag = "20" if args.limit else "256"
    output_root = workspace / ("inputs20" if args.limit else "inputs") / VARIANT
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
        proposals = car_proposals(frame, args.proposals)
        scores = v2.proposal_scores(candidate_valid[:, :3].astype(np.float64), proposals)
        selected, stats = v4.select_existing_voxel(
            frame, candidate_valid[:, :3], geometry, voxel, scores, len(observed_valid)
        )
        final = np.concatenate((observed, candidate_valid[selected]), axis=0)
        if not np.isfinite(final).all() or not np.array_equal(final[: len(observed)], observed):
            raise RuntimeError(f"{frame}: invariant failed")
        if not args.geometry_only:
            output_root.mkdir(parents=True, exist_ok=True)
            final.astype(np.float32, copy=False).tofile(output_root / f"{frame}.bin")
        rows.append({
            "frame_id": frame, "variant": VARIANT, "predicted_car_proposals": len(proposals),
            "observed_total_preserved": len(observed), "observed_valid": len(observed_valid),
            "candidate_valid": len(candidate_valid), "output_total": len(final), **stats,
            "status": "PASS",
        })
        if position == 1 or position % 10 == 0 or position == len(frames):
            print(f"PDANS_V5_PREP {position}/{len(frames)} frame={frame}", flush=True)
    manifest = workspace / f"manifests{tag}"
    write_csv(manifest / "per_frame.csv", rows)
    nonempty = [row for row in rows if row["selected_generated"] > 0]
    aggregate = {
        "variant": VARIANT, "frames": len(rows),
        "frames_with_safe_baseline_fallback": len(rows) - len(nonempty),
        "safe_baseline_fallback_fraction": (len(rows) - len(nonempty)) / len(rows),
        "predicted_car_proposals_median": float(np.median([row["predicted_car_proposals"] for row in rows])),
        "selected_generated_median": float(np.median([row["selected_generated"] for row in rows])),
        "realized_ratio_median": float(np.median([row["realized_ratio"] for row in rows])),
        "nearest_real_p90_max": max(row["selected_nearest_real_p90_m"] for row in rows),
        "existing_observed_voxel_fraction_min_nonempty": min(row["selected_existing_observed_voxel_fraction"] for row in nonempty),
        "generated_only_voxel_fraction_max": max(row["selected_generated_only_voxel_fraction"] for row in rows),
    }
    aggregate["gate_pass"] = bool(
        # A frame without a predicted Car is intentionally observed-only in
        # this class-adaptive ablation. Require useful coverage on at least
        # half the frames; fallback itself is the safe intended behavior.
        aggregate["safe_baseline_fallback_fraction"] <= 0.50
        and aggregate["selected_generated_median"] > 0
        and aggregate["nearest_real_p90_max"] <= v2.MAX_NEAREST_REAL_M + 1e-9
        and aggregate["existing_observed_voxel_fraction_min_nonempty"] == 1.0
        and aggregate["generated_only_voxel_fraction_max"] == 0.0
    )
    write_csv(manifest / "aggregate.csv", [aggregate])
    protocol = {
        "experiment": f"pdans_v5_predicted_car_existing_voxel_{len(frames)}frame_gate",
        "status": "PASS" if aggregate["gate_pass"] else "FAIL", "frames": len(frames),
        "uses_gt_labels_boxes_or_ap": False, "all_observed_points_preserved": True,
        "safe_fallback_policy": "no predicted Car or no anchored candidate => observed-only; require >=50% nonempty frames",
        "single_change_vs_v4_centerpoint": "only predicted class label 1 (Car) pre-NMS candidates",
        "measured_voxel_anchor_geometry_budget_seed_frozen": True,
        "variant": aggregate,
    }
    (workspace / f"protocol{tag}.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2), flush=True)
    return 0 if aggregate["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
