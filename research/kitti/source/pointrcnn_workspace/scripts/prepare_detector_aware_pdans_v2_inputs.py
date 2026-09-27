#!/usr/bin/env python3
"""Prepare pre-registered voxel/proposal-aware PDANS V2 inputs.

The selector is GT-free.  It preserves every observed LiDAR point, reuses the
frozen repaired surface-c32 PDANS candidate pool, and emits three controlled
variants per detector: exact-voxel support only, frozen baseline-proposal only,
and their intersection (Full V2).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import prepare_detector_aware_pdans_inputs as v1  # noqa: E402
import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402


CAUSAL = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
DEFAULT_OBSERVED = REPO / "data/KITTI/object/training/velodyne_original_val"
DEFAULT_PDANS = CAUSAL / "surface_candidate256_v1/pdans_surface_cover_exact_pr1_c32/line_a_original_x4_up/final_bin"
DEFAULT_SPLIT = CAUSAL / "splits/pilot256.txt"
DEFAULT_WORKSPACE = REPO / "results/detector_aware_pdans_v2_20260805"
POINT_BASELINE = REPO / "results/pointrcnn_split_region_object_preserving256_v1_20260804/detector/original_baseline/merged_predictions"
CENTER_BASELINE = CAUSAL / "detectors/centerpoint/original_baseline"
POINT_RANGE = np.asarray([0.0, -40.0, -3.0, 70.4, 40.0, 1.0], dtype=np.float64)
VOXEL_SIZE = np.asarray([0.05, 0.05, 0.1], dtype=np.float64)
MAX_POINTS_PER_VOXEL = 5
PROPOSAL_SCORE = 0.25
PROPOSAL_EXPAND_XY_M = 0.50
PROPOSAL_EXPAND_Z_M = 0.30
MAX_VOXEL_NEIGHBOR_STEPS = 2.0
MIN_VOXEL_NEIGHBORS = 2
MAX_NEAREST_REAL_M = 0.20
VARIANTS = (
    "v2_voxel_only",
    "v2_proposal_pointrcnn",
    "v2_full_pointrcnn",
    "v2_proposal_centerpoint",
    "v2_full_centerpoint",
)


def load_detection_helpers():
    path = REPO / "tools/generate_pointrcnn_detection_box_visualization_v1.py"
    spec = importlib.util.spec_from_file_location("v2_detection_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


det = load_detection_helpers()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def center_result() -> Path:
    matches = sorted(CENTER_BASELINE.rglob("result.pkl"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one baseline CenterPoint result, got {matches}")
    return matches[0]


def load_center_proposals() -> dict[str, list[dict]]:
    import pickle
    with center_result().open("rb") as handle:
        payload = pickle.load(handle)
    output: dict[str, list[dict]] = {}
    for item in payload:
        proposals = []
        for name, score, box in zip(item["name"], item["score"], item["boxes_lidar"]):
            if str(name) not in ("Car", "Pedestrian", "Cyclist") or float(score) < PROPOSAL_SCORE:
                continue
            values = np.asarray(box, dtype=np.float64)
            proposals.append({"center": values[:3], "dims": values[3:6], "heading": float(values[6]), "score": float(score)})
        output[str(item["frame_id"])] = proposals
    return output


def point_proposals(frame: str) -> list[dict]:
    calib = det.parse_calib(REPO / "data/KITTI/object/training/calib" / f"{frame}.txt")
    parsed = det.parse_kitti_boxes(POINT_BASELINE / f"{frame}.txt", calib, "baseline")
    output = []
    for box in parsed:
        if box["class"] != "Car" or float(box["score"] or 0.0) < PROPOSAL_SCORE:
            continue
        corners = np.asarray(box["corners_lidar"], dtype=np.float64)
        base = corners[:4]
        center_xy = base[:, :2].mean(axis=0)
        edge_a = base[1, :2] - base[0, :2]
        edge_b = base[3, :2] - base[0, :2]
        dims = np.asarray([np.linalg.norm(edge_a), np.linalg.norm(edge_b), corners[:, 2].max() - corners[:, 2].min()])
        heading = float(np.arctan2(edge_a[1], edge_a[0]))
        output.append({
            "center": np.asarray([center_xy[0], center_xy[1], (corners[:, 2].min() + corners[:, 2].max()) * 0.5]),
            "dims": dims, "heading": heading, "score": float(box["score"]),
        })
    return output


def proposal_scores(xyz: np.ndarray, proposals: list[dict]) -> np.ndarray:
    scores = np.zeros(len(xyz), dtype=np.float64)
    for proposal in proposals:
        center = proposal["center"]
        dims = proposal["dims"].copy()
        dims[:2] += 2.0 * PROPOSAL_EXPAND_XY_M
        dims[2] += 2.0 * PROPOSAL_EXPAND_Z_M
        c, s = np.cos(proposal["heading"]), np.sin(proposal["heading"])
        delta = xyz - center
        local_x = c * delta[:, 0] + s * delta[:, 1]
        local_y = -s * delta[:, 0] + c * delta[:, 1]
        inside = (np.abs(local_x) <= dims[0] * 0.5) & (np.abs(local_y) <= dims[1] * 0.5) & (np.abs(delta[:, 2]) <= dims[2] * 0.5)
        scores[inside] = np.maximum(scores[inside], float(proposal["score"]))
    return scores


def voxel_coordinates(xyz: np.ndarray) -> np.ndarray:
    return np.floor((xyz.astype(np.float64) - POINT_RANGE[:3]) / VOXEL_SIZE).astype(np.int64)


def voxel_features(observed_xyz: np.ndarray, candidate_xyz: np.ndarray) -> dict[str, np.ndarray]:
    observed_coord = voxel_coordinates(observed_xyz)
    candidate_coord = voxel_coordinates(candidate_xyz)
    unique_observed, observed_counts = np.unique(observed_coord, axis=0, return_counts=True)
    tree = cKDTree(unique_observed.astype(np.float64))
    nearest_steps = tree.query(candidate_coord.astype(np.float64), k=1, p=np.inf, workers=-1)[0]
    neighbor_counts = tree.query_ball_point(
        candidate_coord.astype(np.float64), r=MAX_VOXEL_NEIGHBOR_STEPS,
        p=np.inf, return_length=True, workers=-1,
    ).astype(np.int64)
    count_map = {tuple(int(value) for value in key): int(count) for key, count in zip(unique_observed, observed_counts)}
    observed_in_voxel = np.asarray([count_map.get(tuple(int(value) for value in key), 0) for key in candidate_coord], dtype=np.int64)
    capacity = np.where(observed_in_voxel > 0, np.maximum(0, MAX_POINTS_PER_VOXEL - observed_in_voxel), 1)
    return {
        "candidate_coord": candidate_coord,
        "nearest_voxel_steps": nearest_steps,
        "neighbor_observed_voxels": neighbor_counts,
        "observed_points_in_voxel": observed_in_voxel,
        "retention_capacity": capacity,
    }


def exact_voxel_select(
    candidate_coord: np.ndarray,
    priority: np.ndarray,
    eligible: np.ndarray,
    capacity: np.ndarray,
    target: int,
    seed: int,
) -> np.ndarray:
    candidates = np.flatnonzero(eligible & (capacity > 0))
    if target <= 0 or not len(candidates):
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = candidates[np.argsort(-(priority[candidates] + rng.random(len(candidates)) * 1e-9), kind="stable")]
    used: dict[tuple[int, int, int], int] = {}
    selected: list[int] = []
    for index in order:
        key = tuple(int(value) for value in candidate_coord[index])
        limit = int(capacity[index])
        if used.get(key, 0) >= limit:
            continue
        used[key] = used.get(key, 0) + 1
        selected.append(int(index))
        if len(selected) >= target:
            break
    return np.asarray(selected, dtype=np.int64)


def select_variant(
    variant: str,
    frame: str,
    xyz: np.ndarray,
    geometry: dict[str, np.ndarray],
    voxels: dict[str, np.ndarray],
    proposal_score: np.ndarray,
    observed_count: int,
) -> tuple[np.ndarray, dict]:
    ratio = v1.adaptive_ratio(geometry, "full")
    target = int(round(observed_count * ratio))
    base = (geometry["confidence"] >= 0.35) & geometry["absolute_support"]
    voxel_gate = (
        base
        & (geometry["nearest_m"] <= MAX_NEAREST_REAL_M)
        & (voxels["nearest_voxel_steps"] <= MAX_VOXEL_NEIGHBOR_STEPS)
        & (voxels["neighbor_observed_voxels"] >= MIN_VOXEL_NEIGHBORS)
        & (voxels["retention_capacity"] > 0)
    )
    proposal_gate = base & (proposal_score >= PROPOSAL_SCORE)
    support = np.clip(voxels["neighbor_observed_voxels"] / 6.0, 0.0, 1.0)
    proximity = np.exp(-np.square(geometry["nearest_m"] / 0.16))
    voxel_priority = geometry["utility"] * (0.35 + 0.65 * support) * proximity
    proposal_priority = geometry["utility"] * proposal_score
    if variant == "voxel_only":
        selected = exact_voxel_select(voxels["candidate_coord"], voxel_priority, voxel_gate, voxels["retention_capacity"], target, stable_seed(frame, variant))
        eligible = voxel_gate
    elif variant == "proposal_only":
        selected = v1.select_with_voxel_cap(xyz, proposal_priority, proposal_gate, target, stable_seed(frame, variant))
        eligible = proposal_gate
    elif variant == "full":
        eligible = voxel_gate & proposal_gate
        selected = exact_voxel_select(voxels["candidate_coord"], voxel_priority * proposal_score, eligible, voxels["retention_capacity"], target, stable_seed(frame, variant))
    else:
        raise ValueError(variant)
    generated_only = voxels["observed_points_in_voxel"][selected] == 0
    return selected, {
        "requested_ratio": ratio,
        "requested_generated": target,
        "eligible_candidates": int(np.count_nonzero(eligible)),
        "selected_generated": int(len(selected)),
        "realized_ratio": len(selected) / max(observed_count, 1),
        "selected_nearest_real_mean_m": float(geometry["nearest_m"][selected].mean()) if len(selected) else 0.0,
        "selected_nearest_real_p90_m": float(np.quantile(geometry["nearest_m"][selected], 0.90)) if len(selected) else 0.0,
        "selected_confidence_mean": float(geometry["confidence"][selected].mean()) if len(selected) else 0.0,
        "selected_generated_only_voxel_fraction": float(generated_only.mean()) if len(selected) else 0.0,
        "selected_neighbor_observed_voxels_mean": float(voxels["neighbor_observed_voxels"][selected].mean()) if len(selected) else 0.0,
        "selected_inside_proposal_fraction": float(np.mean(proposal_score[selected] >= PROPOSAL_SCORE)) if len(selected) else 0.0,
    }


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
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    frames = [line.strip() for line in args.split_file.read_text().splitlines() if line.strip()]
    if args.limit:
        frames = frames[: args.limit]
    workspace = args.workspace.resolve()
    center = load_center_proposals()
    rows: list[dict] = []
    mapping = (
        ("v2_voxel_only", "voxel_only", "none"),
        ("v2_proposal_pointrcnn", "proposal_only", "pointrcnn"),
        ("v2_full_pointrcnn", "full", "pointrcnn"),
        ("v2_proposal_centerpoint", "proposal_only", "centerpoint"),
        ("v2_full_centerpoint", "full", "centerpoint"),
    )
    for position, frame in enumerate(frames, start=1):
        observed = prep.read_bin(args.observed / f"{frame}.bin")
        candidates = prep.read_bin(args.pdans / f"{frame}.bin")
        observed_mask, _ = prep.fov_valid_mask(observed, frame)
        candidate_mask, candidate_rect = prep.fov_valid_mask(candidates, frame)
        observed_valid = observed[observed_mask]
        candidate_valid = candidates[candidate_mask]
        candidate_rect_valid = candidate_rect[candidate_mask]
        geometry = v1.geometry_features(observed_valid[:, :3].astype(np.float64), candidate_valid[:, :3].astype(np.float64), candidate_rect_valid[:, 2].astype(np.float64))
        voxels = voxel_features(observed_valid[:, :3], candidate_valid[:, :3])
        point_score = proposal_scores(candidate_valid[:, :3].astype(np.float64), point_proposals(frame))
        center_score = proposal_scores(candidate_valid[:, :3].astype(np.float64), center.get(frame, []))
        for output_name, mode, proposal_source in mapping:
            scores = point_score if proposal_source == "pointrcnn" else center_score if proposal_source == "centerpoint" else np.zeros(len(candidate_valid))
            selected, stats = select_variant(mode, frame, candidate_valid[:, :3], geometry, voxels, scores, len(observed_valid))
            final = np.concatenate((observed, candidate_valid[selected]), axis=0)
            if not np.isfinite(final).all() or not np.array_equal(final[: len(observed)], observed):
                raise RuntimeError(f"{frame}/{output_name}: finite/prefix invariant failed")
            if not args.geometry_only:
                output = workspace / "inputs" / output_name / f"{frame}.bin"
                output.parent.mkdir(parents=True, exist_ok=True)
                final.astype(np.float32, copy=False).tofile(output)
            rows.append({
                "frame_id": frame, "variant": output_name, "selector_mode": mode,
                "proposal_source": proposal_source, "observed_total_preserved": len(observed),
                "observed_valid": len(observed_valid), "candidate_valid": len(candidate_valid),
                "output_total": len(final), **stats, "status": "PASS",
            })
        if position == 1 or position % 16 == 0 or position == len(frames):
            print(f"PDANS_V2_PREP {position}/{len(frames)} frame={frame}", flush=True)
    write_csv(workspace / "manifests/per_frame.csv", rows)
    aggregates = []
    for output_name in VARIANTS:
        chosen = [row for row in rows if row["variant"] == output_name]
        aggregates.append({
            "variant": output_name, "frames": len(chosen),
            "selected_generated_median": float(np.median([row["selected_generated"] for row in chosen])),
            "realized_ratio_median": float(np.median([row["realized_ratio"] for row in chosen])),
            "eligible_candidates_median": float(np.median([row["eligible_candidates"] for row in chosen])),
            "nearest_real_mean_median": float(np.median([row["selected_nearest_real_mean_m"] for row in chosen])),
            "generated_only_voxel_fraction_median": float(np.median([row["selected_generated_only_voxel_fraction"] for row in chosen])),
            "inside_proposal_fraction_median": float(np.median([row["selected_inside_proposal_fraction"] for row in chosen])),
            "neighbor_observed_voxels_mean_median": float(np.median([row["selected_neighbor_observed_voxels_mean"] for row in chosen])),
        })
    write_csv(workspace / "manifests/aggregate.csv", aggregates)
    protocol = {
        "experiment": "detector_aware_pdans_v2", "status": "PASS", "frames": len(frames),
        "geometry_only": args.geometry_only, "uses_gt_labels_or_boxes": False,
        "proposal_source": "frozen baseline detector predictions; two-pass GT-free inference",
        "proposal_score_threshold": PROPOSAL_SCORE, "proposal_expansion_xy_m": PROPOSAL_EXPAND_XY_M,
        "proposal_expansion_z_m": PROPOSAL_EXPAND_Z_M, "centerpoint_voxel_size_m": VOXEL_SIZE.tolist(),
        "max_points_per_voxel": MAX_POINTS_PER_VOXEL, "max_nearest_real_m": MAX_NEAREST_REAL_M,
        "voxel_neighbor_radius_steps": MAX_VOXEL_NEIGHBOR_STEPS, "min_neighbor_observed_voxels": MIN_VOXEL_NEIGHBORS,
        "all_observed_points_preserved": True, "candidate_source": str(args.pdans.resolve()),
        "variants": aggregates,
    }
    (workspace / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
