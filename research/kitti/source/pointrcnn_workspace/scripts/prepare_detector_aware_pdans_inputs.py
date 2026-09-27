#!/usr/bin/env python3
"""Prepare label-free detector-aware PDANS selection variants.

All original LiDAR measurements are preserved.  Repaired surface-c32 PDANS
points are optional candidates and never replace measurements.  Candidate
quality uses only test-time geometry: nearest measurement distance, local
density scale, local plane consistency, ground consistency, repeated local
support and empty-space rejection.  Selection additionally favors sparse and
far regions and uses a frame-adaptive generated-point budget.

The script also emits the requested ablations and controls.  It never reads GT
labels, GT boxes, detector predictions or AP values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402


CAUSAL = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
DEFAULT_OBSERVED = REPO / "data/KITTI/object/training/velodyne_original_val"
DEFAULT_PDANS = (
    CAUSAL
    / "surface_candidate256_v1/pdans_surface_cover_exact_pr1_c32"
    / "line_a_original_x4_up/final_bin"
)
DEFAULT_SPLIT = CAUSAL / "splits/pilot256.txt"
DEFAULT_WORKSPACE = REPO / "results/detector_aware_pdans_surface256_v1_20260804"
BASE_SEED = 20260804
VARIANTS = (
    "fixed_2p5_pdans",
    "matched_real_dynamic",
    "full",
    "no_confidence",
    "no_sparse",
    "no_adaptive",
    "nn_only",
    "density_only",
)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def read_frames(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def voxel_inverse(xyz: np.ndarray, size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.floor(xyz / size).astype(np.int64)
    unique, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return unique, inverse, counts


def geometry_features(
    observed_xyz: np.ndarray,
    candidate_xyz: np.ndarray,
    candidate_depth: np.ndarray,
) -> dict[str, np.ndarray]:
    if observed_xyz.shape[0] < 16:
        raise ValueError(f"too few valid observed points: {observed_xyz.shape[0]}")
    tree = cKDTree(observed_xyz)
    distances, indices = tree.query(candidate_xyz, k=16, workers=-1)
    nearest = distances[:, 0]
    local_radius = np.maximum(distances[:, 7], 0.03)

    neighborhoods = observed_xyz[indices]
    centers = neighborhoods.mean(axis=1)
    centered = neighborhoods - centers[:, None, :]
    covariance = np.einsum("nki,nkj->nij", centered, centered) / neighborhoods.shape[1]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, :, 0]
    plane_distance = np.abs(np.einsum("ni,ni->n", candidate_xyz - centers, normal))
    plane_scale = np.sqrt(np.maximum(eigenvalues.sum(axis=1), 1e-8))
    plane_residual = plane_distance / plane_scale
    planarity = np.clip(1.0 - eigenvalues[:, 0] / np.maximum(eigenvalues.sum(axis=1), 1e-8), 0, 1)

    nearest_score = np.exp(-np.square(nearest / (0.45 * local_radius + 0.04)))
    density_score = np.exp(-np.square(nearest / (1.25 * local_radius + 0.03)))
    plane_score = np.exp(-np.square(plane_residual / 0.20)) * planarity

    local_z = neighborhoods[:, :, 2]
    local_z_median = np.median(local_z, axis=1)
    local_z_spread = np.quantile(local_z, 0.75, axis=1) - np.quantile(local_z, 0.25, axis=1)
    ground_like = (local_z_median < -1.25) & (local_z_spread < 0.25)
    ground_delta = np.abs(candidate_xyz[:, 2] - local_z_median)
    ground_score = np.where(ground_like, np.exp(-np.square(ground_delta / 0.18)), 1.0)

    unused_keys, voxel_id, voxel_counts = voxel_inverse(candidate_xyz, 0.10)
    del unused_keys
    support_count = voxel_counts[voxel_id]
    support_score = np.clip(np.log1p(support_count) / math.log(5.0), 0.0, 1.0)

    empty_space_score = np.exp(-np.square(nearest / (1.8 * local_radius + 0.10)))
    empty_space_score *= np.exp(-np.power(nearest / 0.75, 4.0))
    novelty_score = np.clip(nearest / 0.04, 0.0, 1.0)
    confidence = (
        np.maximum(nearest_score, 1e-6)
        * np.maximum(density_score, 1e-6)
        * np.maximum(plane_score, 1e-6)
        * np.maximum(ground_score, 1e-6)
        * np.maximum(0.5 + 0.5 * support_score, 1e-6)
        * np.maximum(empty_space_score, 1e-6)
    ) ** (1.0 / 6.0)
    confidence *= 0.35 + 0.65 * novelty_score

    sparse_need = np.clip((local_radius - 0.15) / 0.65, 0.0, 1.0)
    depth_need = np.clip((candidate_depth - 20.0) / 50.0, 0.0, 1.0)
    utility = confidence * (0.30 + 0.70 * sparse_need) * (0.45 + 0.55 * depth_need)
    nn_only = nearest_score * density_score * empty_space_score * novelty_score
    density_only = (0.25 + 0.75 * sparse_need) * (0.45 + 0.55 * depth_need)
    return {
        "nearest_m": nearest,
        "local_radius_m": local_radius,
        "plane_residual": plane_residual,
        "ground_score": ground_score,
        "support_count": support_count.astype(np.float64),
        "confidence": confidence,
        "sparse_need": sparse_need,
        "depth_need": depth_need,
        "utility": utility,
        "nn_only": nn_only,
        "density_only": density_only,
        "absolute_support": nearest <= 0.75,
    }


def adaptive_ratio(features: dict[str, np.ndarray], mode: str) -> float:
    if mode == "no_sparse":
        need = float(np.quantile(features["depth_need"], 0.75))
    elif mode == "density_only":
        need = float(np.quantile(features["sparse_need"], 0.75))
    else:
        combined = features["sparse_need"] * (0.4 + 0.6 * features["depth_need"])
        need = float(np.quantile(combined, 0.75))
    return float(np.clip(0.02 + 0.10 * need, 0.02, 0.12))


def select_with_voxel_cap(
    xyz: np.ndarray,
    priority: np.ndarray,
    eligible: np.ndarray,
    target: int,
    seed: int,
    voxel_size: float = 0.10,
    voxel_cap: int = 2,
) -> np.ndarray:
    candidates = np.flatnonzero(eligible)
    if target <= 0 or candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    tie_break = rng.random(candidates.size) * 1e-9
    order = candidates[np.argsort(-(priority[candidates] + tie_break), kind="mergesort")]
    voxel_keys = np.floor(xyz / voxel_size).astype(np.int64)
    selected: list[int] = []
    used: dict[tuple[int, int, int], int] = {}
    deferred: list[int] = []
    for index in order:
        key = tuple(int(value) for value in voxel_keys[index])
        if used.get(key, 0) < voxel_cap:
            selected.append(int(index))
            used[key] = used.get(key, 0) + 1
            if len(selected) == target:
                break
        else:
            deferred.append(int(index))
    if len(selected) < target:
        selected.extend(deferred[: target - len(selected)])
    return np.asarray(selected, dtype=np.int64)


def choose_variant(
    variant: str,
    candidate_xyz: np.ndarray,
    features: dict[str, np.ndarray],
    observed_valid_count: int,
    frame: str,
) -> tuple[np.ndarray, dict]:
    confidence = features["confidence"]
    if variant == "fixed_2p5_pdans":
        ratio = 0.025
        priority = np.ones_like(confidence)
        eligible = np.ones(confidence.shape, dtype=bool)
    elif variant == "full":
        ratio = adaptive_ratio(features, "full")
        priority = features["utility"]
        eligible = (confidence >= 0.35) & features["absolute_support"]
    elif variant == "no_confidence":
        ratio = adaptive_ratio(features, "full")
        priority = features["density_only"]
        eligible = np.ones(confidence.shape, dtype=bool)
    elif variant == "no_sparse":
        ratio = adaptive_ratio(features, "no_sparse")
        priority = confidence * (0.45 + 0.55 * features["depth_need"])
        eligible = (confidence >= 0.35) & features["absolute_support"]
    elif variant == "no_adaptive":
        ratio = 0.025
        priority = features["utility"]
        eligible = (confidence >= 0.35) & features["absolute_support"]
    elif variant == "nn_only":
        ratio = adaptive_ratio(features, "full")
        priority = features["nn_only"] * (0.30 + 0.70 * features["sparse_need"])
        eligible = (features["nn_only"] >= 0.30) & features["absolute_support"]
    elif variant == "density_only":
        ratio = adaptive_ratio(features, "density_only")
        priority = 0.25 + 0.75 * features["sparse_need"]
        eligible = np.ones(confidence.shape, dtype=bool)
    else:
        raise ValueError(variant)
    target = int(round(observed_valid_count * ratio))
    selected = select_with_voxel_cap(
        candidate_xyz,
        priority,
        eligible,
        target,
        stable_seed(BASE_SEED, frame, variant),
    )
    return selected, {
        "requested_ratio": ratio,
        "requested_generated": target,
        "eligible_candidates": int(eligible.sum()),
        "selected_generated": int(selected.size),
        "realized_ratio_over_valid_observed": selected.size / observed_valid_count,
        "selected_confidence_mean": float(confidence[selected].mean()) if selected.size else 0.0,
        "selected_confidence_p10": float(np.quantile(confidence[selected], 0.10))
        if selected.size
        else 0.0,
        "selected_nearest_real_mean_m": float(features["nearest_m"][selected].mean())
        if selected.size
        else 0.0,
        "selected_nearest_real_p90_m": float(np.quantile(features["nearest_m"][selected], 0.90))
        if selected.size
        else 0.0,
        "selected_nearest_real_max_m": float(features["nearest_m"][selected].max())
        if selected.size
        else 0.0,
        "selected_sparse_need_mean": float(features["sparse_need"][selected].mean())
        if selected.size
        else 0.0,
        "selected_depth_need_mean": float(features["depth_need"][selected].mean())
        if selected.size
        else 0.0,
    }


def matched_real_fill(
    observed_valid: np.ndarray,
    target: int,
    frame: str,
) -> np.ndarray:
    if target <= 0:
        return np.empty((0, 4), dtype=np.float32)
    rng = np.random.default_rng(stable_seed(BASE_SEED, frame, "matched_real_dynamic"))
    indices = rng.choice(observed_valid.shape[0], size=target, replace=target > observed_valid.shape[0])
    return observed_valid[indices]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--observed", type=Path, default=DEFAULT_OBSERVED)
    parser.add_argument("--pdans", type=Path, default=DEFAULT_PDANS)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    observed_root = args.observed.resolve()
    pdans_root = args.pdans.resolve()
    frames = read_frames(args.split_file.resolve())
    selected_variants = tuple(args.variant or VARIANTS)
    rows: list[dict] = []
    outputs = {variant: workspace / "inputs" / variant for variant in selected_variants}
    for output in outputs.values():
        output.mkdir(parents=True, exist_ok=True)

    for position, frame in enumerate(frames, start=1):
        observed = prep.read_bin(observed_root / f"{frame}.bin")
        candidates = prep.read_bin(pdans_root / f"{frame}.bin")
        observed_mask, unused_observed_rect = prep.fov_valid_mask(observed, frame)
        candidate_mask, candidate_rect_all = prep.fov_valid_mask(candidates, frame)
        del unused_observed_rect
        observed_valid = observed[observed_mask]
        candidate_valid = candidates[candidate_mask]
        candidate_rect = candidate_rect_all[candidate_mask]
        if candidate_valid.shape[0] == 0:
            raise ValueError(f"{frame}: no valid PDANS candidates")

        features = geometry_features(
            observed_valid[:, :3].astype(np.float64),
            candidate_valid[:, :3].astype(np.float64),
            candidate_rect[:, 2].astype(np.float64),
        )
        full_selected: np.ndarray | None = None
        for variant in selected_variants:
            if variant == "matched_real_dynamic":
                if full_selected is None:
                    full_selected, full_stats = choose_variant(
                        "full",
                        candidate_valid[:, :3],
                        features,
                        observed_valid.shape[0],
                        frame,
                    )
                fill = matched_real_fill(observed_valid, int(full_selected.size), frame)
                final = np.concatenate((observed, fill), axis=0)
                stats = {
                    "requested_ratio": full_stats["requested_ratio"],
                    "requested_generated": 0,
                    "eligible_candidates": observed_valid.shape[0],
                    "selected_generated": 0,
                    "selected_matched_real": int(fill.shape[0]),
                    "realized_ratio_over_valid_observed": fill.shape[0] / observed_valid.shape[0],
                    "selected_confidence_mean": 1.0,
                    "selected_confidence_p10": 1.0,
                    "selected_nearest_real_mean_m": 0.0,
                    "selected_nearest_real_p90_m": 0.0,
                    "selected_nearest_real_max_m": 0.0,
                    "selected_sparse_need_mean": 0.0,
                    "selected_depth_need_mean": 0.0,
                }
            else:
                selected, stats = choose_variant(
                    variant,
                    candidate_valid[:, :3],
                    features,
                    observed_valid.shape[0],
                    frame,
                )
                if variant == "full":
                    full_selected = selected
                final = np.concatenate((observed, candidate_valid[selected]), axis=0)
                stats["selected_matched_real"] = 0
            if not np.isfinite(final).all():
                raise RuntimeError(f"{frame}/{variant}: NaN/Inf")
            final.astype(np.float32, copy=False).tofile(outputs[variant] / f"{frame}.bin")
            rows.append(
                {
                    "frame_id": frame,
                    "variant": variant,
                    "observed_total_preserved": int(observed.shape[0]),
                    "observed_valid": int(observed_valid.shape[0]),
                    "candidate_valid": int(candidate_valid.shape[0]),
                    "output_total": int(final.shape[0]),
                    **stats,
                    "uses_labels_boxes_detector_outputs_or_ap": False,
                    "status": "PASS",
                }
            )
        if position == 1 or position % 16 == 0 or position == len(frames):
            print(f"DETECTOR_AWARE_PDANS_PREP {position}/{len(frames)} frame={frame}", flush=True)

    write_csv(workspace / "manifests/per_frame.csv", rows)
    summaries = {}
    for variant in selected_variants:
        subset = [row for row in rows if row["variant"] == variant]
        summaries[variant] = {
            "frames": len(subset),
            "output_points_total": sum(int(row["output_total"]) for row in subset),
            "selected_generated_total": sum(int(row["selected_generated"]) for row in subset),
            "selected_matched_real_total": sum(int(row["selected_matched_real"]) for row in subset),
            "realized_ratio_median": float(
                np.median([float(row["realized_ratio_over_valid_observed"]) for row in subset])
            ),
        }
    protocol = {
        "experiment": "detector_aware_pdans_surface256_v1",
        "split_file": str(args.split_file.resolve()),
        "frames": len(frames),
        "frame_ids": frames,
        "observed_source": str(observed_root),
        "candidate_source": str(pdans_root),
        "candidate_patch": "surface-c32 repaired PDANS",
        "all_original_measurements_preserved": True,
        "candidate_points_only_supplement_measurements": True,
        "uses_labels_boxes_detector_outputs_or_ap": False,
        "confidence_components": [
            "nearest_measurement_distance",
            "local_density_scale",
            "local_plane_consistency",
            "ground_consistency",
            "multi_patch_local_support_proxy_from_local_candidate_consensus",
            "empty_space_rejection",
        ],
        "selection_components": ["sparse_local_support", "distance", "quality", "adaptive_ratio"],
        "variants": summaries,
        "status": "PASS",
    }
    (workspace / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
