#!/usr/bin/env python3
"""Prepare GT-free PDANS V3 inputs gated by true detector internals.

PointRCNN uses pre-RCNN RPN proposals.  CenterPoint uses decoded heatmap top-K
boxes before NMS.  Every observed LiDAR point is preserved and all geometric,
voxel, candidate-pool, budget, intensity and seed rules remain frozen from V2.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import prepare_detector_aware_pdans_inputs as v1  # noqa: E402
import prepare_detector_aware_pdans_v2_inputs as v2  # noqa: E402
import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402


CAUSAL = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
DEFAULT_OBSERVED = REPO / "data/KITTI/object/training/velodyne_original_val"
DEFAULT_PDANS = CAUSAL / "surface_candidate256_v1/pdans_surface_cover_exact_pr1_c32/line_a_original_x4_up/final_bin"
DEFAULT_SPLIT = CAUSAL / "splits/pilot256.txt"
DEFAULT_WORKSPACE = REPO / "results/detector_aware_pdans_v3_20260805"
DEFAULT_POINT = DEFAULT_WORKSPACE / "internal_proposals/pointrcnn_rpn_20/merged_kitti"
DEFAULT_CENTER = DEFAULT_WORKSPACE / "internal_proposals/centerpoint_prenms_20/npz"
POINT_RPN_PROBABILITY_THRESHOLD = 0.30
CENTER_HEATMAP_THRESHOLD = 0.10
VARIANTS = (
    "v3_internal_proposal_pointrcnn",
    "v3_internal_full_pointrcnn",
    "v3_internal_proposal_centerpoint",
    "v3_internal_full_centerpoint",
)


def detection_helpers():
    path = REPO / "tools/generate_pointrcnn_detection_box_visualization_v1.py"
    spec = importlib.util.spec_from_file_location("v3_detection_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


det = detection_helpers()


def point_proposals(frame: str, proposal_dir: Path) -> list[dict]:
    calib = det.parse_calib(REPO / "data/KITTI/object/training/calib" / f"{frame}.txt")
    parsed = det.parse_kitti_boxes(proposal_dir / f"{frame}.txt", calib, "rpn")
    output = []
    for box in parsed:
        raw_logit = float(box["score"] or -100.0)
        probability = 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, raw_logit))))
        if probability < POINT_RPN_PROBABILITY_THRESHOLD:
            continue
        corners = np.asarray(box["corners_lidar"], dtype=np.float64)
        base = corners[:4]
        center_xy = base[:, :2].mean(axis=0)
        edge_a = base[1, :2] - base[0, :2]
        edge_b = base[3, :2] - base[0, :2]
        output.append({
            "center": np.asarray([center_xy[0], center_xy[1], 0.5 * (corners[:, 2].min() + corners[:, 2].max())]),
            "dims": np.asarray([np.linalg.norm(edge_a), np.linalg.norm(edge_b), corners[:, 2].max() - corners[:, 2].min()]),
            "heading": float(np.arctan2(edge_a[1], edge_a[0])), "score": probability,
        })
    return output


def center_proposals(frame: str, proposal_dir: Path) -> list[dict]:
    payload = np.load(proposal_dir / f"{frame}.npz")
    output = []
    for box, score in zip(payload["boxes_lidar"], payload["scores"]):
        if float(score) < CENTER_HEATMAP_THRESHOLD:
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
    parser.add_argument("--point-proposals", type=Path, default=DEFAULT_POINT)
    parser.add_argument("--center-proposals", type=Path, default=DEFAULT_CENTER)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()

    frames = v1.read_frames(args.split_file.resolve())
    if args.limit:
        frames = frames[: args.limit]
    workspace = args.workspace.resolve()
    artifact_tag = "20" if args.limit else "256"
    input_root = workspace / ("inputs20" if args.limit else "inputs")
    rows: list[dict] = []
    mapping = (
        ("v3_internal_proposal_pointrcnn", "proposal_only", "pointrcnn"),
        ("v3_internal_full_pointrcnn", "full", "pointrcnn"),
        ("v3_internal_proposal_centerpoint", "proposal_only", "centerpoint"),
        ("v3_internal_full_centerpoint", "full", "centerpoint"),
    )
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
        voxels = v2.voxel_features(observed_valid[:, :3], candidate_valid[:, :3])
        point_boxes = point_proposals(frame, args.point_proposals)
        center_boxes = center_proposals(frame, args.center_proposals)
        point_score = v2.proposal_scores(candidate_valid[:, :3].astype(np.float64), point_boxes)
        center_score = v2.proposal_scores(candidate_valid[:, :3].astype(np.float64), center_boxes)
        for output_name, mode, source in mapping:
            scores = point_score if source == "pointrcnn" else center_score
            selected, stats = v2.select_variant(mode, frame, candidate_valid[:, :3], geometry,
                                                voxels, scores, len(observed_valid))
            final = np.concatenate((observed, candidate_valid[selected]), axis=0)
            if not np.isfinite(final).all() or not np.array_equal(final[: len(observed)], observed):
                raise RuntimeError(f"{frame}/{output_name}: finite or observed-prefix invariant failed")
            if not args.geometry_only:
                target = input_root / output_name / f"{frame}.bin"
                target.parent.mkdir(parents=True, exist_ok=True)
                final.astype(np.float32, copy=False).tofile(target)
            rows.append({
                "frame_id": frame, "variant": output_name, "selector_mode": mode,
                "proposal_source": source, "internal_proposals": len(point_boxes) if source == "pointrcnn" else len(center_boxes),
                "observed_total_preserved": len(observed), "observed_valid": len(observed_valid),
                "candidate_valid": len(candidate_valid), "output_total": len(final), **stats,
                "status": "PASS",
            })
        if position == 1 or position % 10 == 0 or position == len(frames):
            print(f"PDANS_V3_PREP {position}/{len(frames)} frame={frame}", flush=True)

    manifest_root = workspace / f"manifests{artifact_tag}"
    write_csv(manifest_root / "per_frame.csv", rows)
    aggregates = []
    gate_pass = True
    for variant in VARIANTS:
        chosen = [row for row in rows if row["variant"] == variant]
        aggregate = {
            "variant": variant, "frames": len(chosen),
            "internal_proposals_min": min(row["internal_proposals"] for row in chosen),
            "internal_proposals_median": float(np.median([row["internal_proposals"] for row in chosen])),
            "eligible_candidates_median": float(np.median([row["eligible_candidates"] for row in chosen])),
            "selected_generated_median": float(np.median([row["selected_generated"] for row in chosen])),
            "realized_ratio_median": float(np.median([row["realized_ratio"] for row in chosen])),
            "nearest_real_mean_median": float(np.median([row["selected_nearest_real_mean_m"] for row in chosen])),
            "nearest_real_p90_max": float(max(row["selected_nearest_real_p90_m"] for row in chosen)),
            "inside_proposal_fraction_median": float(np.median([row["selected_inside_proposal_fraction"] for row in chosen])),
        }
        # Proposal-only is the deliberate ablation without V2's strict
        # 0.20 m voxel/NN gate; it is bounded by the frozen absolute-support
        # guard (0.75 m).  Full V3 must satisfy the stricter 0.20 m rule.
        nearest_limit = v2.MAX_NEAREST_REAL_M if "_full_" in variant else 0.75
        aggregate["nearest_gate_m"] = nearest_limit
        aggregate["gate_pass"] = bool(
            aggregate["internal_proposals_min"] > 0
            and aggregate["eligible_candidates_median"] > 0
            and aggregate["selected_generated_median"] > 0
            and aggregate["nearest_real_p90_max"] <= nearest_limit + 1e-9
        )
        gate_pass &= aggregate["gate_pass"]
        aggregates.append(aggregate)
    write_csv(manifest_root / "aggregate.csv", aggregates)
    protocol = {
        "experiment": f"detector_aware_pdans_v3_internal_proposals_{len(frames)}frame_gate",
        "status": "PASS" if gate_pass else "FAIL", "frames": len(frames),
        "uses_gt_labels_boxes_or_ap": False,
        "point_stage": "true RPN proposals before RCNN; raw logit sigmoid >= 0.30",
        "center_stage": "decoded heatmap top-K before NMS; native score >= 0.10",
        "all_observed_points_preserved": True,
        "v2_geometry_voxel_budget_and_seed_rules_frozen": True,
        "variants": aggregates,
    }
    (workspace / f"protocol{artifact_tag}.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2), flush=True)
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
