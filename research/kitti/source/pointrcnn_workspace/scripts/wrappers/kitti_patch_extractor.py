#!/usr/bin/env python3
"""Deterministic KITTI patch extraction for unified upsampling protocol probes."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np


def load_kitti_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(f"{path} is not KITTI XYZI: value count is not divisible by 4")
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN or Inf")
    return points


def deterministic_spatial_order(points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3]
    mins = xyz.min(axis=0)
    span = np.maximum(xyz.max(axis=0) - mins, 1e-6)
    bins = np.floor((xyz - mins) / span * 32.0).astype(np.int32)
    bins = np.clip(bins, 0, 31)
    original = np.arange(points.shape[0], dtype=np.int64)
    return np.lexsort((original, xyz[:, 2], bins[:, 1], bins[:, 0]))


def extract_spatial_chunks(
    points: np.ndarray,
    patch_input_points: int,
    max_patches: int | None,
    min_tail_real_points: int = 32,
) -> tuple[list[np.ndarray], list[dict]]:
    order = deterministic_spatial_order(points)
    patch_count = int(math.ceil(points.shape[0] / float(patch_input_points)))
    if max_patches is not None:
        patch_count = min(patch_count, max_patches)

    patches: list[np.ndarray] = []
    metadata: list[dict] = []
    for patch_id in range(patch_count):
        start = patch_id * patch_input_points
        end = min(start + patch_input_points, order.shape[0])
        idx = order[start:end]
        real_idx = idx.copy()
        real_count = int(real_idx.shape[0])
        pad_count = 0
        tail_support_used = False
        neighbor_support_count = 0
        support_repeat_count = 0
        if idx.shape[0] < patch_input_points:
            pad_count = patch_input_points - idx.shape[0]
            if idx.shape[0] == 0:
                idx = order[:patch_input_points]
            elif real_count < min_tail_real_points:
                # PDANS cannot consume a patch made from one repeated point.
                # Keep every real tail point, then add unchanged real points
                # from this frame in deterministic distance/index order.
                centroid = points[real_idx, :3].astype(np.float64).mean(axis=0)
                delta = points[:, :3].astype(np.float64) - centroid
                dist2 = np.einsum("ij,ij->i", delta, delta)
                original = np.arange(points.shape[0], dtype=np.int64)
                ranked = np.lexsort((original, dist2))
                is_tail = np.zeros(points.shape[0], dtype=bool)
                is_tail[real_idx] = True
                neighbors = ranked[~is_tail[ranked]][:pad_count]
                idx = np.concatenate((real_idx, neighbors))
                neighbor_support_count = int(neighbors.shape[0])
                if idx.shape[0] < patch_input_points:
                    support_repeat_count = patch_input_points - idx.shape[0]
                    idx = np.resize(idx, patch_input_points)
                tail_support_used = True
            else:
                reps = np.resize(idx, patch_input_points)
                idx = reps
        patch = points[idx, :3].astype(np.float32, copy=True)
        patches.append(patch)
        metadata.append(
            {
                "patch_id": patch_id,
                "file": f"patch_{patch_id:06d}.npy",
                "shape": [int(patch.shape[0]), int(patch.shape[1])],
                "original_indices": idx.astype(int).tolist(),
                "unique_original_index_count": int(np.unique(idx).shape[0]),
                "pad_count": int(pad_count),
                "source_order_start": int(start),
                "source_order_end_exclusive": int(end),
                "tail_support_rule": "degenerate_tail_neighbor_support_v1" if tail_support_used else None,
                "original_tail_real_count": real_count,
                "patch_input_points": int(patch_input_points),
                "neighbor_support_count": neighbor_support_count,
                "support_repeat_count": int(support_repeat_count),
                "support_source": "same_frame_real_input_points" if tail_support_used else None,
                "no_coordinate_modification": True,
                "no_intensity_modification": True,
                "no_jitter": True,
                "no_interpolation": True,
                "no_final_output_padding": True,
                "deterministic_sort": "distance_then_original_index" if tail_support_used else None,
            }
        )
    return patches, metadata


def _patch_geometry(points_xyz: np.ndarray, center_xyz: np.ndarray) -> dict:
    """Return inexpensive, deterministic locality diagnostics for one patch."""
    xyz64 = points_xyz.astype(np.float64, copy=False)
    xyz_span = xyz64.max(axis=0) - xyz64.min(axis=0)
    center_delta = xyz64 - center_xyz.astype(np.float64, copy=False)
    center_dist2 = np.einsum("ij,ij->i", center_delta, center_delta)
    return {
        "xy_bbox_diagonal_m": float(np.linalg.norm(xyz_span[:2])),
        "xyz_bbox_diagonal_m": float(np.linalg.norm(xyz_span)),
        "center_furthest_distance_m": float(np.sqrt(center_dist2.max(initial=0.0))),
    }


def extract_fps_knn_local(
    points: np.ndarray,
    patch_input_points: int,
    max_patches: int | None,
    seed: int,
    patch_num_ratio: int,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Extract overlapping local patches using FPS centers and exact kNN.

    PU-GCN and PU-EdgeFormer use a native patch_num_ratio default of 3 for
    whole-cloud inference.  The overlap is necessary on strongly non-uniform
    LiDAR scans: a ceil(N/K) budget either misses much of the frame or forces
    capacity-balanced patches to become non-local.
    """
    point_count = int(points.shape[0])
    patch_count = int(math.ceil(point_count / float(patch_input_points))) * patch_num_ratio
    if max_patches is not None:
        patch_count = min(patch_count, max_patches)
    if patch_count <= 0:
        return [], [], {
            "covered_original_point_count": 0,
            "coverage_fraction": 0.0,
            "source_membership_count": 0,
            "unique_source_membership_count": 0,
        }

    xyz = points[:, :3].astype(np.float64, copy=False)
    original = np.arange(point_count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    first_center = int(rng.integers(0, point_count))
    centers: list[int] = []
    min_dist2 = np.full(point_count, np.inf, dtype=np.float64)
    next_center = first_center
    for _ in range(patch_count):
        centers.append(next_center)
        delta = xyz - xyz[next_center]
        dist2 = np.einsum("ij,ij->i", delta, delta)
        np.minimum(min_dist2, dist2, out=min_dist2)
        next_center = int(np.argmax(min_dist2))

    patches: list[np.ndarray] = []
    metadata: list[dict] = []
    membership_count = np.zeros(point_count, dtype=np.int32)
    for patch_id, center_idx in enumerate(centers):
        delta = xyz - xyz[center_idx]
        dist2 = np.einsum("ij,ij->i", delta, delta)
        unique_take = min(point_count, patch_input_points)
        if unique_take == point_count:
            idx = original.copy()
        else:
            idx = np.argpartition(dist2, unique_take - 1)[:unique_take].astype(np.int64)
        idx = idx[np.lexsort((idx, dist2[idx]))]
        unique_count = int(idx.shape[0])
        support_repeat_count = 0
        if idx.shape[0] < patch_input_points:
            support_repeat_count = patch_input_points - idx.shape[0]
            idx = np.resize(idx, patch_input_points)

        np.add.at(membership_count, np.unique(idx), 1)
        patch = points[idx, :3].astype(np.float32, copy=True)
        geometry = _patch_geometry(patch, points[center_idx, :3])
        patches.append(patch)
        metadata.append(
            {
                "patch_id": patch_id,
                "file": f"patch_{patch_id:06d}.npy",
                "shape": [int(patch.shape[0]), int(patch.shape[1])],
                "center_original_index": center_idx,
                "center_xyz": points[center_idx, :3].astype(float).tolist(),
                "original_indices": idx.astype(int).tolist(),
                "unique_original_index_count": int(np.unique(idx).shape[0]),
                "support_repeat_count": int(support_repeat_count),
                "knn_unique_support_count": unique_count,
                "knn_k": int(patch_input_points),
                "knn_furthest_distance_m": float(np.sqrt(dist2[idx[:unique_count]].max(initial=0.0))),
                "no_coordinate_modification": True,
                "no_intensity_modification": True,
                "no_jitter": True,
                "no_interpolation": True,
                "no_final_output_padding": True,
                "deterministic_sort": "distance_then_original_index",
                **geometry,
            }
        )

    covered = int(np.count_nonzero(membership_count))
    membership_total = int(sum(len(meta["original_indices"]) for meta in metadata))
    summary = {
        "fps_center_original_indices": centers,
        "covered_original_point_count": covered,
        "coverage_fraction": float(covered / point_count),
        "source_membership_count": membership_total,
        "unique_source_membership_count": covered,
        "membership_count_min": int(membership_count.min()),
        "membership_count_median": float(np.median(membership_count)),
        "membership_count_max": int(membership_count.max()),
        "first_center_rule": "seeded_uniform_original_index_then_standard_fps",
        "coverage_policy": "overlapping_exact_knn",
        "patch_num_ratio": int(patch_num_ratio),
        "fps_distance_space": "xyz_euclidean_meters",
        "knn_distance_space": "xyz_euclidean_meters",
    }
    return patches, metadata, summary


def extract_fps_ball_local(
    points: np.ndarray,
    patch_input_points: int,
    max_patches: int | None,
    seed: int,
    patch_num_ratio: int,
    ball_radius_m: float,
    min_ball_points: int,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Extract radius-capped local patches around density-eligible FPS centers."""
    from scipy.spatial import cKDTree  # noqa: PLC0415

    point_count = int(points.shape[0])
    patch_count = int(math.ceil(point_count / float(patch_input_points))) * patch_num_ratio
    if max_patches is not None:
        patch_count = min(patch_count, max_patches)
    xyz = points[:, :3].astype(np.float64, copy=False)
    original = np.arange(point_count, dtype=np.int64)
    tree = cKDTree(xyz)
    support_counts = np.asarray(
        tree.query_ball_point(xyz, r=ball_radius_m, return_length=True, workers=1),
        dtype=np.int32,
    )
    eligible = original[support_counts >= min_ball_points]
    if eligible.size == 0:
        raise ValueError(
            f"no FPS center has at least {min_ball_points} points within {ball_radius_m} m"
        )

    rng = np.random.default_rng(seed)
    eligible_xyz = xyz[eligible]
    first_pos = int(rng.integers(0, eligible.shape[0]))
    center_positions: list[int] = []
    min_dist2 = np.full(eligible.shape[0], np.inf, dtype=np.float64)
    next_pos = first_pos
    for _ in range(patch_count):
        center_positions.append(next_pos)
        delta = eligible_xyz - eligible_xyz[next_pos]
        dist2 = np.einsum("ij,ij->i", delta, delta)
        np.minimum(min_dist2, dist2, out=min_dist2)
        next_pos = int(np.argmax(min_dist2))
    centers = eligible[np.asarray(center_positions, dtype=np.int64)].astype(np.int64).tolist()

    patches: list[np.ndarray] = []
    metadata: list[dict] = []
    membership_count = np.zeros(point_count, dtype=np.int32)
    for patch_id, center_idx in enumerate(centers):
        neighbor_list = tree.query_ball_point(xyz[center_idx], r=ball_radius_m, workers=1)
        idx = np.asarray(neighbor_list, dtype=np.int64)
        delta = xyz[idx] - xyz[center_idx]
        dist2_local = np.einsum("ij,ij->i", delta, delta)
        idx = idx[np.lexsort((idx, dist2_local))]
        if idx.shape[0] > patch_input_points:
            idx = idx[:patch_input_points]
        unique_count = int(idx.shape[0])
        support_repeat_count = 0
        if idx.shape[0] < patch_input_points:
            support_repeat_count = patch_input_points - idx.shape[0]
            idx = np.resize(idx, patch_input_points)
        np.add.at(membership_count, np.unique(idx), 1)
        patch = points[idx, :3].astype(np.float32, copy=True)
        geometry = _patch_geometry(patch, points[center_idx, :3])
        patches.append(patch)
        metadata.append(
            {
                "patch_id": patch_id,
                "file": f"patch_{patch_id:06d}.npy",
                "shape": [int(patch.shape[0]), int(patch.shape[1])],
                "center_original_index": int(center_idx),
                "center_xyz": points[center_idx, :3].astype(float).tolist(),
                "original_indices": idx.astype(int).tolist(),
                "unique_original_index_count": int(np.unique(idx).shape[0]),
                "support_repeat_count": int(support_repeat_count),
                "ball_unique_support_count": unique_count,
                "ball_radius_m": float(ball_radius_m),
                "center_full_ball_support_count": int(support_counts[center_idx]),
                "no_coordinate_modification": True,
                "no_intensity_modification": True,
                "no_jitter": True,
                "no_interpolation": True,
                "no_final_output_padding": True,
                "deterministic_sort": "distance_then_original_index",
                **geometry,
            }
        )

    covered = int(np.count_nonzero(membership_count))
    summary = {
        "fps_center_original_indices": centers,
        "covered_original_point_count": covered,
        "coverage_fraction": float(covered / point_count),
        "source_membership_count": int(sum(len(meta["original_indices"]) for meta in metadata)),
        "unique_source_membership_count": covered,
        "membership_count_min": int(membership_count.min()),
        "membership_count_median": float(np.median(membership_count)),
        "membership_count_max": int(membership_count.max()),
        "first_center_rule": "seeded_uniform_density_eligible_then_standard_fps",
        "coverage_policy": "overlapping_radius_capped_ball_query_with_repeat_fill",
        "patch_num_ratio": int(patch_num_ratio),
        "ball_radius_m": float(ball_radius_m),
        "min_ball_points": int(min_ball_points),
        "eligible_center_point_count": int(eligible.shape[0]),
        "eligible_center_fraction": float(eligible.shape[0] / point_count),
        "fps_distance_space": "xyz_euclidean_meters",
        "ball_distance_space": "xyz_euclidean_meters",
    }
    return patches, metadata, summary


def extract_fps_ball_cover(
    points: np.ndarray,
    patch_input_points: int,
    max_patches: int | None,
    seed: int,
    patch_num_ratio: int,
    ball_radius_m: float,
    min_ball_points: int,
    cover_min_points: int,
    cover_radius_m: float,
    supplement_knn_floor: int = 0,
) -> tuple[list[np.ndarray], list[dict], dict]:
    """Radius-capped FPS patches plus local centers for every uncovered source point.

    The primary budget is identical to ``fps_ball_local_v1``.  Supplemental
    centers are selected by farthest distance among currently uncovered input
    points.  When ``supplement_knn_floor`` is positive, a sparse supplemental
    ball is deterministically augmented with nearest neighbors until that many
    unique source points are present.
    """
    from scipy.spatial import cKDTree  # noqa: PLC0415

    patches, metadata, summary = extract_fps_ball_local(
        points=points,
        patch_input_points=patch_input_points,
        max_patches=max_patches,
        seed=seed,
        patch_num_ratio=patch_num_ratio,
        ball_radius_m=ball_radius_m,
        min_ball_points=min_ball_points,
    )
    primary_patch_count = len(patches)
    for item in metadata:
        item["center_reason"] = "density_eligible_fps_primary"

    point_count = int(points.shape[0])
    xyz = points[:, :3].astype(np.float64, copy=False)
    tree = cKDTree(xyz)
    cover_support_counts = np.asarray(
        tree.query_ball_point(xyz, r=cover_radius_m, return_length=True, workers=1),
        dtype=np.int32,
    )
    cover_eligible = cover_support_counts >= cover_min_points
    membership_count = np.zeros(point_count, dtype=np.int32)
    for item in metadata:
        np.add.at(
            membership_count,
            np.unique(np.asarray(item["original_indices"], dtype=np.int64)),
            1,
        )

    min_center_dist2 = np.full(point_count, np.inf, dtype=np.float64)
    centers = [int(value) for value in summary["fps_center_original_indices"]]
    for center_idx in centers:
        delta = xyz - xyz[center_idx]
        dist2 = np.einsum("ij,ij->i", delta, delta)
        np.minimum(min_center_dist2, dist2, out=min_center_dist2)

    while np.any((membership_count == 0) & cover_eligible):
        if max_patches is not None and len(patches) >= max_patches:
            break
        score = np.where(
            (membership_count == 0) & cover_eligible,
            min_center_dist2,
            -1.0,
        )
        center_idx = int(np.argmax(score))
        if membership_count[center_idx] != 0 or not cover_eligible[center_idx]:
            raise RuntimeError("coverage supplementation made no progress")
        neighbor_list = tree.query_ball_point(xyz[center_idx], r=cover_radius_m, workers=1)
        neighbor_idx = np.asarray(neighbor_list, dtype=np.int64)
        delta_local = xyz[neighbor_idx] - xyz[center_idx]
        dist2_local = np.einsum("ij,ij->i", delta_local, delta_local)
        ranked = neighbor_idx[np.lexsort((neighbor_idx, dist2_local))]
        uncovered_ranked = ranked[membership_count[ranked] == 0]
        covered_ranked = ranked[membership_count[ranked] != 0]
        unique_idx = np.concatenate((uncovered_ranked, covered_ranked))[:patch_input_points]
        ball_unique_count = int(unique_idx.shape[0])
        if unique_idx.shape[0] < supplement_knn_floor:
            query_k = min(supplement_knn_floor, point_count)
            _, knn_idx = tree.query(xyz[center_idx], k=query_k, workers=1)
            knn_idx = np.atleast_1d(knn_idx).astype(np.int64, copy=False)
            present = np.zeros(point_count, dtype=bool)
            present[unique_idx] = True
            missing_knn = knn_idx[~present[knn_idx]]
            unique_idx = np.concatenate((unique_idx, missing_knn))[:query_k]
        unique_count = int(unique_idx.shape[0])
        if unique_count == 0 or center_idx not in unique_idx:
            raise RuntimeError("supplemental ball patch does not contain its center")
        idx = unique_idx
        support_repeat_count = 0
        if idx.shape[0] < patch_input_points:
            support_repeat_count = patch_input_points - idx.shape[0]
            idx = np.resize(idx, patch_input_points)
        np.add.at(membership_count, unique_idx, 1)
        patch_id = len(patches)
        patch = points[idx, :3].astype(np.float32, copy=True)
        geometry = _patch_geometry(patch, points[center_idx, :3])
        patches.append(patch)
        metadata.append(
            {
                "patch_id": patch_id,
                "file": f"patch_{patch_id:06d}.npy",
                "shape": [int(patch.shape[0]), int(patch.shape[1])],
                "center_original_index": center_idx,
                "center_xyz": points[center_idx, :3].astype(float).tolist(),
                "center_reason": "uncovered_fps_supplement",
                "original_indices": idx.astype(int).tolist(),
                "unique_original_index_count": int(np.unique(idx).shape[0]),
                "support_repeat_count": int(support_repeat_count),
                "ball_unique_support_count": ball_unique_count,
                "supplement_knn_floor": int(supplement_knn_floor),
                "supplement_knn_added_count": int(unique_count - ball_unique_count),
                "ball_radius_m": float(cover_radius_m),
                "primary_ball_radius_m": float(ball_radius_m),
                "supplement_ball_radius_m": float(cover_radius_m),
                "center_full_ball_support_count": int(len(neighbor_list)),
                "no_coordinate_modification": True,
                "no_intensity_modification": True,
                "no_jitter": True,
                "no_interpolation": True,
                "no_final_output_padding": True,
                "deterministic_sort": (
                    "uncovered_ball_then_covered_ball_then_knn_distance"
                    if supplement_knn_floor
                    else "uncovered_first_then_distance_then_original_index"
                ),
                **geometry,
            }
        )
        centers.append(center_idx)
        delta_all = xyz - xyz[center_idx]
        dist2_all = np.einsum("ij,ij->i", delta_all, delta_all)
        np.minimum(min_center_dist2, dist2_all, out=min_center_dist2)

    covered = int(np.count_nonzero(membership_count))
    eligible_covered = int(np.count_nonzero((membership_count > 0) & cover_eligible))
    eligible_count = int(np.count_nonzero(cover_eligible))
    if eligible_covered != eligible_count:
        raise ValueError(
            "eligible source coverage was not reached: "
            f"{eligible_covered}/{eligible_count}; max_patches={max_patches}"
        )
    summary.update(
        {
            "fps_center_original_indices": centers,
            "primary_patch_count": int(primary_patch_count),
            "supplemental_patch_count": int(len(patches) - primary_patch_count),
            "covered_original_point_count": covered,
            "coverage_fraction": float(covered / point_count),
            "cover_min_points": int(cover_min_points),
            "cover_radius_m": float(cover_radius_m),
            "supplement_knn_floor": int(supplement_knn_floor),
            "cover_eligible_original_point_count": eligible_count,
            "cover_eligible_fraction": float(eligible_count / point_count),
            "cover_eligible_covered_point_count": eligible_covered,
            "cover_eligible_coverage_fraction": 1.0,
            "source_membership_count": int(
                sum(len(item["original_indices"]) for item in metadata)
            ),
            "unique_source_membership_count": covered,
            "membership_count_min": int(membership_count.min()),
            "membership_count_median": float(np.median(membership_count)),
            "membership_count_max": int(membership_count.max()),
            "coverage_policy": (
                "primary_ball_plus_uncovered_fps_ball_knn_floor_supplement"
                if supplement_knn_floor
                else "radius_capped_primary_plus_uncovered_fps_supplement_100pct_for_local_support_eligible"
            ),
            "supplement_center_policy": "farthest_uncovered_from_existing_centers",
            "supplement_expands_radius": bool(supplement_knn_floor),
        }
    )
    return patches, metadata, summary


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_bin", required=True)
    parser.add_argument("--output_patch_dir", required=True)
    parser.add_argument("--patch_input_points", type=int, required=True)
    parser.add_argument(
        "--patch_selection",
        choices=[
            "deterministic_spatial_chunk",
            "fps_knn_local_v1",
            "fps_ball_local_v1",
            "fps_ball_cover_v2",
            "fps_ball_cover_knn_v3",
        ],
        required=True,
    )
    parser.add_argument("--max_patches", type=int, default=None)
    parser.add_argument(
        "--patch_num_ratio",
        type=int,
        default=3,
        help="FPS+kNN overlap ratio; ignored by deterministic_spatial_chunk.",
    )
    parser.add_argument("--ball_radius_m", type=float, default=8.0)
    parser.add_argument("--min_ball_points", type=int, default=256)
    parser.add_argument("--cover_min_points", type=int, default=32)
    parser.add_argument("--cover_radius_m", type=float, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--line", required=True)
    parser.add_argument("--frame_id", required=True)
    parser.add_argument("--metadata_json", required=True)
    parser.add_argument("--min_tail_real_points", type=int, default=32)
    args = parser.parse_args()

    started = time.time()
    input_path = Path(args.input_bin).resolve()
    patch_dir = Path(args.output_patch_dir).resolve()
    metadata_path = Path(args.metadata_json).resolve()
    patch_dir.mkdir(parents=True, exist_ok=True)

    if args.patch_input_points <= 0:
        raise ValueError("--patch_input_points must be positive")
    if args.max_patches is not None and args.max_patches <= 0:
        raise ValueError("--max_patches must be positive when provided")
    if args.patch_num_ratio <= 0:
        raise ValueError("--patch_num_ratio must be positive")
    if args.ball_radius_m <= 0:
        raise ValueError("--ball_radius_m must be positive")
    if args.min_ball_points <= 0 or args.min_ball_points > args.patch_input_points:
        raise ValueError("--min_ball_points must be in [1, patch_input_points]")
    if args.cover_min_points <= 0 or args.cover_min_points > args.min_ball_points:
        raise ValueError("--cover_min_points must be in [1, min_ball_points]")
    if args.cover_radius_m is None:
        args.cover_radius_m = args.ball_radius_m
    if args.cover_radius_m < args.ball_radius_m:
        raise ValueError("--cover_radius_m must be >= ball_radius_m")

    points = load_kitti_bin(input_path)
    selection_summary: dict = {}
    if args.patch_selection == "deterministic_spatial_chunk":
        patches, patch_metadata = extract_spatial_chunks(
            points=points,
            patch_input_points=args.patch_input_points,
            max_patches=args.max_patches,
            min_tail_real_points=args.min_tail_real_points,
        )
    elif args.patch_selection == "fps_knn_local_v1":
        patches, patch_metadata, selection_summary = extract_fps_knn_local(
            points=points,
            patch_input_points=args.patch_input_points,
            max_patches=args.max_patches,
            seed=args.seed,
            patch_num_ratio=args.patch_num_ratio,
        )
    elif args.patch_selection == "fps_ball_local_v1":
        patches, patch_metadata, selection_summary = extract_fps_ball_local(
            points=points,
            patch_input_points=args.patch_input_points,
            max_patches=args.max_patches,
            seed=args.seed,
            patch_num_ratio=args.patch_num_ratio,
            ball_radius_m=args.ball_radius_m,
            min_ball_points=args.min_ball_points,
        )
    else:
        patches, patch_metadata, selection_summary = extract_fps_ball_cover(
            points=points,
            patch_input_points=args.patch_input_points,
            max_patches=args.max_patches,
            seed=args.seed,
            patch_num_ratio=args.patch_num_ratio,
            ball_radius_m=args.ball_radius_m,
            min_ball_points=args.min_ball_points,
            cover_min_points=args.cover_min_points,
            cover_radius_m=args.cover_radius_m,
            supplement_knn_floor=(
                args.min_ball_points
                if args.patch_selection == "fps_ball_cover_knn_v3"
                else 0
            ),
        )
    for meta in patch_metadata:
        if meta.get("tail_support_rule") is not None:
            meta["affected_frame_id"] = args.frame_id
            meta["affected_patch_id"] = meta["file"].removesuffix(".npy")
    for meta, patch in zip(patch_metadata, patches):
        np.save(patch_dir / meta["file"], patch)

    payload = {
        "input_bin": str(input_path),
        "input_point_count": int(points.shape[0]),
        "line": args.line,
        "frame_id": args.frame_id,
        "patch_input_points": int(args.patch_input_points),
        "patch_selection": args.patch_selection,
        "patch_protocol_id": f"{args.patch_selection}_k{args.patch_input_points}",
        "patch_overlap": (
            "none"
            if args.patch_selection == "deterministic_spatial_chunk"
            else "allowed_variable_budget_for_full_coverage"
            if args.patch_selection in ("fps_ball_cover_v2", "fps_ball_cover_knn_v3")
            else "allowed_fixed_budget"
        ),
        "seed": int(args.seed),
        "deterministic": True,
        "output_patch_dir": str(patch_dir),
        "patch_count": int(len(patches)),
        "patch_format": "npy float32 xyz",
        "tail_support_rule": "degenerate_tail_neighbor_support_v1",
        "min_tail_real_points": int(args.min_tail_real_points),
        "tail_support_affected_patches": [
            meta["file"] for meta in patch_metadata if meta.get("tail_support_rule") is not None
        ],
        "patches": patch_metadata,
        "runtime_patch_extraction_sec": float(time.time() - started),
        "status": "PASS",
        **selection_summary,
    }
    save_json(metadata_path, payload)
    print(json.dumps({k: payload[k] for k in ("status", "input_point_count", "patch_count", "runtime_patch_extraction_sec")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
