#!/usr/bin/env python3
"""Quantify why exact-x4 upsampled KITTI clouds underperform detector baselines.

The analysis is detector-independent.  It recreates the exact E1 generated
subset, compares generated points with the original KITTI scan, measures
surface coverage inside Car GT boxes, audits effective voxel information, and
quantifies the locality of the actual 2048-point patch extraction protocol.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.utils import kitti_utils  # noqa: E402
from lib.utils.calibration import Calibration  # noqa: E402


SOURCE = REPO / "results/kitti_unified_x4_current_methods_no_detector"
WORKSPACE = REPO / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
TRAINING = REPO / "data/KITTI/object/training"
FULL = TRAINING / "velodyne_original_val"
DOWNSAMPLED = SOURCE / "downsampled_x4/velodyne_downsampled_x4_val"
VAL_SPLIT = REPO / "data/KITTI/ImageSets/val.txt"
METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
LINES = {
    "A": {
        "observed": FULL,
        "source_prefix": "line_a_original_x4_up",
        "variant_prefix": "original_x4",
    },
    "B": {
        "observed": DOWNSAMPLED,
        "source_prefix": "line_b_downsampled_x4_up",
        "variant_prefix": "downsampled_x4",
    },
}
BASE_SEED = 20260718
PATCH_POINTS = 2048

E0_MODERATE = {
    ("A", "pdans"): 60.72,
    ("A", "pu_gcn"): 53.81,
    ("A", "pu_edgeformer"): 39.53,
    ("A", "pu_net"): 6.72,
    ("B", "pdans"): 39.04,
    ("B", "pu_gcn"): 27.18,
    ("B", "pu_edgeformer"): 17.40,
    ("B", "pu_net"): 6.07,
}


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(x) for x in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"{path} is not Nx4 float32")
    return values.reshape(-1, 4)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_ap_moderate() -> dict[tuple[str, str, str], float]:
    path = WORKSPACE / "reports/e1_e2_live_ap_summary.csv"
    result: dict[tuple[str, str, str], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "PASS":
                continue
            variant = row["variant"]
            line = "B" if variant.startswith("downsampled") else "A"
            for method in METHODS:
                if variant.endswith(method):
                    result[(row["experiment"], line, method)] = float(row["3d_ap_moderate"])
    return result


def car_objects(frame: str):
    objects = kitti_utils.get_objects_from_label(str(TRAINING / "label_2" / f"{frame}.txt"))
    return [obj for obj in objects if obj.cls_type == "Car" and obj.level in (1, 2, 3)]


def choose_frames(count: int) -> list[str]:
    frames = [x.strip() for x in VAL_SPLIT.read_text(encoding="utf-8").splitlines() if x.strip()]
    candidates = [frame for frame in frames if car_objects(frame)]
    if count >= len(candidates):
        return candidates
    positions = np.linspace(0, len(candidates) - 1, count, dtype=np.int64)
    return [candidates[int(i)] for i in np.unique(positions)]


def recreate_generated(line: str, method: str, frame: str, observed_n: int) -> np.ndarray:
    cfg = LINES[line]
    predicted = read_bin(SOURCE / cfg["source_prefix"] / method / "final_bin" / f"{frame}.bin")
    expected = 4 * observed_n
    if predicted.shape[0] != expected:
        raise ValueError(f"{line}/{method}/{frame}: predicted {len(predicted)} != {expected}")
    variant = f"{cfg['variant_prefix']}_{method}"
    rng = np.random.default_rng(stable_seed(BASE_SEED, "e1", variant, frame))
    index = rng.choice(expected, size=3 * observed_n, replace=False)
    return predicted[index]


def deterministic_spatial_order(points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3]
    mins = xyz.min(axis=0)
    span = np.maximum(xyz.max(axis=0) - mins, 1e-6)
    bins = np.floor((xyz - mins) / span * 32.0).astype(np.int32)
    bins = np.clip(bins, 0, 31)
    original = np.arange(points.shape[0], dtype=np.int64)
    return np.lexsort((original, xyz[:, 2], bins[:, 1], bins[:, 0]))


def patch_metrics(points: np.ndarray) -> list[dict]:
    order = deterministic_spatial_order(points)
    rows = []
    for start in range(0, len(order), PATCH_POINTS):
        idx = order[start : start + PATCH_POINTS]
        if len(idx) < 32:
            continue
        xyz = points[idx, :3].astype(np.float64)
        center = xyz.mean(axis=0)
        radius = np.linalg.norm(xyz - center, axis=1)
        extent = xyz.max(axis=0) - xyz.min(axis=0)
        rows.append(
            {
                "real_points": len(idx),
                "radius_p50_m": float(np.quantile(radius, 0.50)),
                "radius_p90_m": float(np.quantile(radius, 0.90)),
                "radius_max_m": float(radius.max()),
                "xyz_bbox_diagonal_m": float(np.linalg.norm(extent)),
                "xy_bbox_diagonal_m": float(np.linalg.norm(extent[:2])),
            }
        )
    return rows


def box_mask(points_rect: np.ndarray, obj, extra: float = 0.0) -> np.ndarray:
    relative = points_rect - obj.pos.reshape(1, 3)
    c, s = math.cos(obj.ry), math.sin(obj.ry)
    rotation = np.asarray([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64)
    local = relative @ rotation.T
    return (
        (np.abs(local[:, 0]) <= obj.l / 2.0 + extra)
        & (np.abs(local[:, 2]) <= obj.w / 2.0 + extra)
        & (local[:, 1] >= -obj.h - extra)
        & (local[:, 1] <= extra)
    )


def voxel_keys(points: np.ndarray, size: float) -> np.ndarray:
    quantized = np.ascontiguousarray(np.floor(points[:, :3] / size).astype(np.int32))
    dtype = np.dtype((np.void, quantized.dtype.itemsize * quantized.shape[1]))
    return np.unique(quantized.view(dtype).ravel())


def finite_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else float("nan")


def summarize_values(values: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(x) for x in values if np.asarray(x).size]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)


def aggregate_manifest() -> list[dict]:
    rows = []
    manifest_root = WORKSPACE / "manifests/e2_canonical_16384"
    for path in sorted(manifest_root.glob("*.csv")):
        source, fov, voxel = [], [], []
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                source.append(int(row["source_points"]))
                fov.append(int(row["fov_valid_points"]))
                voxel.append(int(row["voxel_unique_points"]))
        source_a = np.asarray(source, dtype=np.float64)
        fov_a = np.asarray(fov, dtype=np.float64)
        voxel_a = np.asarray(voxel, dtype=np.float64)
        rows.append(
            {
                "variant": path.stem,
                "frames": len(source),
                "source_points_median": float(np.median(source_a)),
                "fov_valid_fraction_median": float(np.median(fov_a / source_a)),
                "voxel_unique_over_fov_median": float(np.median(voxel_a / fov_a)),
                "voxel_unique_over_source_median": float(np.median(voxel_a / source_a)),
                "voxel_unique_points_median": float(np.median(voxel_a)),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-frames", type=int, default=96)
    parser.add_argument("--generated-sample-per-frame", type=int, default=4096)
    parser.add_argument("--voxel-frames", type=int, default=24)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE / "reports/geometry_root_cause_v1",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = choose_frames(args.sample_frames)
    voxel_frame_set = set(frames[: min(args.voxel_frames, len(frames))])
    ap = read_ap_moderate()

    patch_rows: list[dict] = []
    voxel_rows: list[dict] = []
    object_rows: list[dict] = []
    scene_store: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {
        (line, method): defaultdict(list) for line in LINES for method in METHODS
    }

    for frame_pos, frame in enumerate(frames, start=1):
        reference = read_bin(FULL / f"{frame}.bin")
        reference_tree = cKDTree(reference[:, :3])
        calibration = Calibration(str(TRAINING / "calib" / f"{frame}.txt"))
        reference_rect = calibration.lidar_to_rect(reference[:, :3])
        objects = car_objects(frame)

        for line, line_cfg in LINES.items():
            observed = read_bin(line_cfg["observed"] / f"{frame}.bin")
            observed_tree = cKDTree(observed[:, :3])
            observed_rect = calibration.lidar_to_rect(observed[:, :3])

            for item in patch_metrics(observed):
                patch_rows.append({"line": line, "frame_id": frame, **item})

            reference_voxels = {}
            observed_voxels = {}
            if frame in voxel_frame_set:
                for size in (0.10, 0.20, 0.50):
                    reference_voxels[size] = voxel_keys(reference, size)
                    observed_voxels[size] = voxel_keys(observed, size)

            for method in METHODS:
                generated = recreate_generated(line, method, frame, len(observed))
                generated_rect = calibration.lidar_to_rect(generated[:, :3])
                store = scene_store[(line, method)]

                sample_n = min(args.generated_sample_per_frame, len(generated))
                sample_rng = np.random.default_rng(stable_seed("geometry", line, method, frame))
                sample_idx = sample_rng.choice(len(generated), size=sample_n, replace=False)
                sample = generated[sample_idx]

                d_ref, knn_ref = reference_tree.query(sample[:, :3], k=min(8, len(reference)), workers=-1)
                if d_ref.ndim == 1:
                    nearest_ref = d_ref
                    knn_ref = knn_ref[:, None]
                else:
                    nearest_ref = d_ref[:, 0]
                d_obs, nearest_obs_idx = observed_tree.query(sample[:, :3], k=1, workers=-1)
                store["generated_to_reference_m"].append(nearest_ref.astype(np.float32))
                store["generated_to_observed_m"].append(np.asarray(d_obs, dtype=np.float32))
                intensity_delta = np.abs(sample[:, 3] - observed[nearest_obs_idx, 3])
                store["nearest_observed_intensity_delta"].append(intensity_delta.astype(np.float32))

                neighbors = reference[np.asarray(knn_ref), :3].astype(np.float64)
                centers = neighbors.mean(axis=1)
                centered = neighbors - centers[:, None, :]
                covariance = np.einsum("nki,nkj->nij", centered, centered) / neighbors.shape[1]
                _, eigenvectors = np.linalg.eigh(covariance)
                normals = eigenvectors[:, :, 0]
                plane_residual = np.abs(np.einsum("ni,ni->n", sample[:, :3] - centers, normals))
                store["local_plane_residual_m"].append(plane_residual.astype(np.float32))

                if frame in voxel_frame_set:
                    for size in (0.10, 0.20, 0.50):
                        ref_key = reference_voxels[size]
                        obs_key = observed_voxels[size]
                        gen_key = voxel_keys(generated, size)
                        e1_key = np.union1d(obs_key, gen_key)
                        voxel_rows.append(
                            {
                                "line": line,
                                "method": method,
                                "frame_id": frame,
                                "voxel_size_m": size,
                                "reference_voxels": len(ref_key),
                                "observed_voxels": len(obs_key),
                                "generated_voxels": len(gen_key),
                                "e1_voxels": len(e1_key),
                                "observed_reference_recall": len(np.intersect1d(obs_key, ref_key)) / len(ref_key),
                                "e1_reference_recall": len(np.intersect1d(e1_key, ref_key)) / len(ref_key),
                                "generated_reference_precision": len(np.intersect1d(gen_key, ref_key)) / len(gen_key),
                                "e1_extra_voxel_fraction": len(np.setdiff1d(e1_key, ref_key)) / len(e1_key),
                            }
                        )

                for obj_idx, obj in enumerate(objects):
                    ref_inside_mask = box_mask(reference_rect, obj, 0.0)
                    if ref_inside_mask.sum() < 5:
                        continue
                    obs_inside_mask = box_mask(observed_rect, obj, 0.0)
                    gen_inside_mask = box_mask(generated_rect, obj, 0.0)
                    ref_large_mask = box_mask(reference_rect, obj, 0.5)
                    obs_large_mask = box_mask(observed_rect, obj, 0.5)
                    gen_large_mask = box_mask(generated_rect, obj, 0.5)
                    ref_shell = int(np.count_nonzero(ref_large_mask & ~ref_inside_mask))
                    gen_shell = int(np.count_nonzero(gen_large_mask & ~gen_inside_mask))

                    ref_object = reference_rect[ref_inside_mask]
                    obs_local = observed_rect[obs_large_mask]
                    gen_local = generated_rect[gen_large_mask]
                    if len(obs_local):
                        base_distance = cKDTree(obs_local).query(ref_object, k=1, workers=-1)[0]
                    else:
                        base_distance = np.full(len(ref_object), np.inf)
                    e1_local = np.concatenate((obs_local, gen_local), axis=0)
                    if len(e1_local):
                        e1_distance = cKDTree(e1_local).query(ref_object, k=1, workers=-1)[0]
                    else:
                        e1_distance = np.full(len(ref_object), np.inf)

                    gen_inside_xyz = generated[gen_inside_mask, :3]
                    if len(gen_inside_xyz):
                        gen_ref_distance = reference_tree.query(gen_inside_xyz, k=1, workers=-1)[0]
                    else:
                        gen_ref_distance = np.empty(0)

                    depth_bin = "0-20" if obj.pos[2] < 20 else "20-40" if obj.pos[2] < 40 else "40-70.4"
                    obs_inside = int(obs_inside_mask.sum())
                    gen_inside = int(gen_inside_mask.sum())
                    object_rows.append(
                        {
                            "line": line,
                            "method": method,
                            "frame_id": frame,
                            "object_index": obj_idx,
                            "difficulty": obj.level_str,
                            "depth_bin_m": depth_bin,
                            "object_depth_m": float(obj.pos[2]),
                            "reference_inside_points": int(ref_inside_mask.sum()),
                            "observed_inside_points": obs_inside,
                            "generated_inside_points": gen_inside,
                            "reference_shell_points_0p5m": ref_shell,
                            "generated_shell_points_0p5m": gen_shell,
                            "generated_vs_expected_3x_object_fill": gen_inside / max(3 * obs_inside, 1),
                            "generated_object_allocation_ratio": (gen_inside / len(generated))
                            / max(int(ref_inside_mask.sum()) / len(reference), 1e-12),
                            "generated_shell_to_inside_ratio": gen_shell / max(gen_inside, 1),
                            "reference_shell_to_inside_ratio": ref_shell / max(int(ref_inside_mask.sum()), 1),
                            "generated_inside_near_reference_0p25_fraction": float(np.mean(gen_ref_distance <= 0.25))
                            if len(gen_ref_distance)
                            else float("nan"),
                            "generated_inside_near_reference_0p50_fraction": float(np.mean(gen_ref_distance <= 0.50))
                            if len(gen_ref_distance)
                            else float("nan"),
                            "baseline_reference_coverage_0p20": float(np.mean(base_distance <= 0.20)),
                            "e1_reference_coverage_0p20": float(np.mean(e1_distance <= 0.20)),
                            "baseline_reference_coverage_0p50": float(np.mean(base_distance <= 0.50)),
                            "e1_reference_coverage_0p50": float(np.mean(e1_distance <= 0.50)),
                            "baseline_reference_distance_p90_m": finite_quantile(base_distance, 0.90),
                            "e1_reference_distance_p90_m": finite_quantile(e1_distance, 0.90),
                        }
                    )

        print(f"geometry analysis: {frame_pos}/{len(frames)} frames", flush=True)

    scene_rows = []
    for (line, method), store in scene_store.items():
        d_ref = summarize_values(store["generated_to_reference_m"])
        d_obs = summarize_values(store["generated_to_observed_m"])
        plane = summarize_values(store["local_plane_residual_m"])
        intensity = summarize_values(store["nearest_observed_intensity_delta"])
        relevant_objects = [r for r in object_rows if r["line"] == line and r["method"] == method]
        relevant_voxels = [r for r in voxel_rows if r["line"] == line and r["method"] == method and r["voxel_size_m"] == 0.20]
        e1_ap = ap[("E1", line, method)]
        e2_ap = ap[("E2", line, method)]
        baseline_ap = 82.26 if line == "A" else 65.75
        scene_rows.append(
            {
                "line": line,
                "method": method,
                "sample_frames": len(frames),
                "sampled_generated_points": len(d_ref),
                "generated_to_reference_p50_m": finite_quantile(d_ref, 0.50),
                "generated_to_reference_p90_m": finite_quantile(d_ref, 0.90),
                "generated_to_reference_p99_m": finite_quantile(d_ref, 0.99),
                "generated_farther_than_reference_0p25_fraction": float(np.mean(d_ref > 0.25)),
                "generated_farther_than_reference_0p50_fraction": float(np.mean(d_ref > 0.50)),
                "generated_farther_than_reference_1p0_fraction": float(np.mean(d_ref > 1.0)),
                "generated_to_observed_p50_m": finite_quantile(d_obs, 0.50),
                "generated_to_observed_p90_m": finite_quantile(d_obs, 0.90),
                "local_plane_residual_p50_m": finite_quantile(plane, 0.50),
                "local_plane_residual_p90_m": finite_quantile(plane, 0.90),
                "local_plane_residual_gt_0p25_fraction": float(np.mean(plane > 0.25)),
                "nearest_observed_intensity_exact_fraction": float(np.mean(intensity <= 1e-7)),
                "car_objects": len(relevant_objects),
                "generated_vs_expected_3x_object_fill_median": finite_quantile(
                    np.asarray([r["generated_vs_expected_3x_object_fill"] for r in relevant_objects]), 0.50
                ),
                "generated_object_allocation_ratio_median": finite_quantile(
                    np.asarray([r["generated_object_allocation_ratio"] for r in relevant_objects]), 0.50
                ),
                "generated_inside_near_reference_0p25_median": finite_quantile(
                    np.asarray([r["generated_inside_near_reference_0p25_fraction"] for r in relevant_objects]), 0.50
                ),
                "generated_shell_to_inside_median": finite_quantile(
                    np.asarray([r["generated_shell_to_inside_ratio"] for r in relevant_objects]), 0.50
                ),
                "reference_shell_to_inside_median": finite_quantile(
                    np.asarray([r["reference_shell_to_inside_ratio"] for r in relevant_objects]), 0.50
                ),
                "object_reference_coverage_0p20_baseline_median": finite_quantile(
                    np.asarray([r["baseline_reference_coverage_0p20"] for r in relevant_objects]), 0.50
                ),
                "object_reference_coverage_0p20_e1_median": finite_quantile(
                    np.asarray([r["e1_reference_coverage_0p20"] for r in relevant_objects]), 0.50
                ),
                "object_reference_coverage_0p50_baseline_median": finite_quantile(
                    np.asarray([r["baseline_reference_coverage_0p50"] for r in relevant_objects]), 0.50
                ),
                "object_reference_coverage_0p50_e1_median": finite_quantile(
                    np.asarray([r["e1_reference_coverage_0p50"] for r in relevant_objects]), 0.50
                ),
                "voxel_0p20_generated_reference_precision_median": finite_quantile(
                    np.asarray([r["generated_reference_precision"] for r in relevant_voxels]), 0.50
                ),
                "voxel_0p20_reference_recall_observed_median": finite_quantile(
                    np.asarray([r["observed_reference_recall"] for r in relevant_voxels]), 0.50
                ),
                "voxel_0p20_reference_recall_e1_median": finite_quantile(
                    np.asarray([r["e1_reference_recall"] for r in relevant_voxels]), 0.50
                ),
                "voxel_0p20_e1_extra_fraction_median": finite_quantile(
                    np.asarray([r["e1_extra_voxel_fraction"] for r in relevant_voxels]), 0.50
                ),
                "moderate_ap_e0": E0_MODERATE[(line, method)],
                "moderate_ap_e1": e1_ap,
                "moderate_ap_e2": e2_ap,
                "moderate_ap_e1_minus_e0": e1_ap - E0_MODERATE[(line, method)],
                "moderate_ap_e2_minus_e1": e2_ap - e1_ap,
                "moderate_ap_e2_minus_baseline": e2_ap - baseline_ap,
            }
        )

    patch_summary = []
    for line in LINES:
        subset = [row for row in patch_rows if row["line"] == line]
        patch_summary.append(
            {
                "line": line,
                "sample_frames": len(frames),
                "patches": len(subset),
                "patch_radius_p90_median": finite_quantile(
                    np.asarray([r["radius_p90_m"] for r in subset]), 0.50
                ),
                "patch_radius_p90_p90": finite_quantile(np.asarray([r["radius_p90_m"] for r in subset]), 0.90),
                "patch_radius_max_median": finite_quantile(
                    np.asarray([r["radius_max_m"] for r in subset]), 0.50
                ),
                "patch_radius_max_p90": finite_quantile(np.asarray([r["radius_max_m"] for r in subset]), 0.90),
                "patch_xy_diagonal_median": finite_quantile(
                    np.asarray([r["xy_bbox_diagonal_m"] for r in subset]), 0.50
                ),
                "patch_xy_diagonal_p90": finite_quantile(
                    np.asarray([r["xy_bbox_diagonal_m"] for r in subset]), 0.90
                ),
            }
        )

    depth_rows = []
    for line in LINES:
        for method in METHODS:
            for depth_bin in ("0-20", "20-40", "40-70.4"):
                subset = [
                    row
                    for row in object_rows
                    if row["line"] == line and row["method"] == method and row["depth_bin_m"] == depth_bin
                ]
                if not subset:
                    continue
                depth_rows.append(
                    {
                        "line": line,
                        "method": method,
                        "depth_bin_m": depth_bin,
                        "objects": len(subset),
                        "generated_vs_expected_3x_object_fill_median": finite_quantile(
                            np.asarray([r["generated_vs_expected_3x_object_fill"] for r in subset]), 0.50
                        ),
                        "generated_inside_near_reference_0p25_median": finite_quantile(
                            np.asarray([r["generated_inside_near_reference_0p25_fraction"] for r in subset]), 0.50
                        ),
                        "coverage_0p20_baseline_median": finite_quantile(
                            np.asarray([r["baseline_reference_coverage_0p20"] for r in subset]), 0.50
                        ),
                        "coverage_0p20_e1_median": finite_quantile(
                            np.asarray([r["e1_reference_coverage_0p20"] for r in subset]), 0.50
                        ),
                        "reference_distance_p90_baseline_median_m": finite_quantile(
                            np.asarray([r["baseline_reference_distance_p90_m"] for r in subset]), 0.50
                        ),
                        "reference_distance_p90_e1_median_m": finite_quantile(
                            np.asarray([r["e1_reference_distance_p90_m"] for r in subset]), 0.50
                        ),
                    }
                )

    manifest_rows = aggregate_manifest()
    write_csv(output_dir / "method_line_geometry_summary.csv", scene_rows)
    write_csv(output_dir / "object_depth_summary.csv", depth_rows)
    write_csv(output_dir / "object_level_metrics.csv", object_rows)
    write_csv(output_dir / "voxel_frame_metrics.csv", voxel_rows)
    write_csv(output_dir / "patch_locality_summary.csv", patch_summary)
    write_csv(output_dir / "patch_level_metrics.csv", patch_rows)
    write_csv(output_dir / "e2_manifest_effective_points_summary.csv", manifest_rows)
    payload = {
        "status": "PASS",
        "sample_frames": frames,
        "sample_frame_count": len(frames),
        "voxel_frame_count": len(voxel_frame_set),
        "generated_sample_per_frame": args.generated_sample_per_frame,
        "method_line_summary": scene_rows,
        "object_depth_summary": depth_rows,
        "patch_locality_summary": patch_summary,
        "e2_manifest_summary": manifest_rows,
    }
    (output_dir / "geometry_root_cause_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(f"GEOMETRY_ROOT_CAUSE_PASS output={output_dir} frames={len(frames)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
