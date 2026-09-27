#!/usr/bin/env python3
"""Unified single-frame point-cloud upsampling metrics for KITTI methods."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.density_metric_schema import DEFAULT_BEV_CELL_SIZE, DEFAULT_KITTI_LIDAR_RANGE, NA_VALUE, csv_fieldnames
from tools.evaluate_density_pointclouds import evaluate_pair

DEFAULT_METRICS_ROOT = PROJECT_ROOT / "results" / "pugcn_metrics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate upsampling quality for one or more KITTI .bin pairs.")
    parser.add_argument("--input-bin", type=Path, required=True)
    parser.add_argument("--output-bin", type=Path, required=True)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--method-name", type=str, default="PU-GCN")
    parser.add_argument("--experiment-name", type=str, default="pugcn_single_frame")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--cell-size", type=float, default=DEFAULT_BEV_CELL_SIZE)
    parser.add_argument("--duplicate-voxel-size", type=float, default=0.02)
    parser.add_argument("--near-duplicate-radius", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--write-visualizations", action="store_true")
    return parser.parse_args()


def load_kitti_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError("Invalid KITTI .bin shape: %s" % path)
    return raw.reshape(-1, 4)


def nn_output_to_input(output: np.ndarray, input_points: np.ndarray) -> Dict[str, object]:
    if len(output) == 0 or len(input_points) == 0:
        return {key: NA_VALUE for key in ["output_to_input_nn_min", "output_to_input_nn_mean", "output_to_input_nn_median", "output_to_input_nn_p95", "output_to_input_nn_max"]}
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(input_points[:, :3]).query(output[:, :3], k=1)
    except Exception:
        distances = []
        for point in output[:, :3]:
            diff = input_points[:, :3] - point
            distances.append(float(np.sqrt(np.min(np.sum(diff * diff, axis=1)))))
        distances = np.asarray(distances, dtype=np.float32)
    return {
        "output_to_input_nn_min": float(np.min(distances)),
        "output_to_input_nn_mean": float(np.mean(distances)),
        "output_to_input_nn_median": float(np.median(distances)),
        "output_to_input_nn_p95": float(np.percentile(distances, 95)),
        "output_to_input_nn_max": float(np.max(distances)),
    }


def self_nn_min_max(points: np.ndarray) -> Dict[str, object]:
    if len(points) < 2:
        return {"output_self_nn_min": NA_VALUE, "output_self_nn_max": NA_VALUE}
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(points[:, :3]).query(points[:, :3], k=2)
        nn = np.asarray(distances[:, 1], dtype=np.float32)
    except Exception:
        nn = []
        xyz = points[:, :3]
        for i, point in enumerate(xyz):
            diff = xyz - point
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            dist[i] = np.inf
            nn.append(float(np.min(dist)))
        nn = np.asarray(nn, dtype=np.float32)
    return {
        "output_self_nn_min": float(np.min(nn)),
        "output_self_nn_max": float(np.max(nn)),
    }


def invalid_stats(prefix: str, points: np.ndarray) -> Dict[str, object]:
    finite_xyz = np.isfinite(points[:, :3]).all(axis=1) if len(points) else np.asarray([], dtype=bool)
    return {
        f"{prefix}_nan_points": int(np.isnan(points).any(axis=1).sum()) if len(points) else 0,
        f"{prefix}_inf_points": int(np.isinf(points).any(axis=1).sum()) if len(points) else 0,
        f"{prefix}_invalid_xyz_points": int((~finite_xyz).sum()) if len(points) else 0,
    }


def kitti_range_stats(prefix: str, points: np.ndarray) -> Dict[str, object]:
    if len(points) == 0:
        return {
            f"{prefix}_out_of_range_points": 0,
            f"{prefix}_out_of_range_ratio": NA_VALUE,
        }
    xyz = points[:, :3]
    mask = (
        (xyz[:, 0] >= DEFAULT_KITTI_LIDAR_RANGE["x_min"])
        & (xyz[:, 0] <= DEFAULT_KITTI_LIDAR_RANGE["x_max"])
        & (xyz[:, 1] >= DEFAULT_KITTI_LIDAR_RANGE["y_min"])
        & (xyz[:, 1] <= DEFAULT_KITTI_LIDAR_RANGE["y_max"])
        & (xyz[:, 2] >= DEFAULT_KITTI_LIDAR_RANGE["z_min"])
        & (xyz[:, 2] <= DEFAULT_KITTI_LIDAR_RANGE["z_max"])
        & np.isfinite(xyz).all(axis=1)
    )
    out_count = int(np.sum(~mask))
    return {
        f"{prefix}_out_of_range_points": out_count,
        f"{prefix}_out_of_range_ratio": float(out_count / len(points)),
    }


def local_neighbor_stats(points: np.ndarray, radius: float = 0.5) -> Dict[str, object]:
    keys = ["local_neighbor_radius", "local_neighbor_count_mean", "local_neighbor_count_median", "local_neighbor_count_p95"]
    if len(points) == 0:
        return dict(zip(keys, [radius, NA_VALUE, NA_VALUE, NA_VALUE]))
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(points[:, :3])
        counts = np.asarray([len(ix) - 1 for ix in tree.query_ball_point(points[:, :3], r=radius)], dtype=np.float32)
    except Exception:
        counts = []
        xyz = points[:, :3]
        for point in xyz:
            dist2 = np.sum((xyz - point) ** 2, axis=1)
            counts.append(int(np.sum(dist2 <= radius * radius) - 1))
        counts = np.asarray(counts, dtype=np.float32)
    return {
        "local_neighbor_radius": radius,
        "local_neighbor_count_mean": float(np.mean(counts)),
        "local_neighbor_count_median": float(np.median(counts)),
        "local_neighbor_count_p95": float(np.percentile(counts, 95)),
    }


def duplicate_stats(points: np.ndarray, voxel_size: float, near_radius: float) -> Dict[str, object]:
    if len(points) == 0:
        return {
            "exact_duplicate_points": 0,
            "exact_duplicate_ratio": NA_VALUE,
            "duplicate_voxel_size": voxel_size,
            "duplicate_voxel_points": 0,
            "duplicate_voxel_ratio": NA_VALUE,
            "near_duplicate_radius": near_radius,
            "near_duplicate_points": 0,
            "near_duplicate_ratio": NA_VALUE,
        }
    _, exact_counts = np.unique(points[:, :3], axis=0, return_counts=True)
    exact_duplicate_points = int(np.sum(exact_counts[exact_counts > 1] - 1))
    if voxel_size > 0:
        coords = np.floor(points[:, :3] / voxel_size).astype(np.int64)
        _, counts = np.unique(coords, axis=0, return_counts=True)
        duplicate_voxel_points = int(np.sum(counts[counts > 1] - 1))
        duplicate_voxel_ratio = float(duplicate_voxel_points / len(points))
    else:
        duplicate_voxel_points = 0
        duplicate_voxel_ratio = NA_VALUE
    near_duplicate_points = 0
    if len(points) > 1 and near_radius > 0:
        try:
            from scipy.spatial import cKDTree

            distances, _ = cKDTree(points[:, :3]).query(points[:, :3], k=2, workers=-1)
        except TypeError:
            from scipy.spatial import cKDTree

            distances, _ = cKDTree(points[:, :3]).query(points[:, :3], k=2)
        except Exception:
            distances = None
        if distances is not None:
            near_duplicate_points = int(np.sum(distances[:, 1] <= near_radius))
    return {
        "exact_duplicate_points": exact_duplicate_points,
        "exact_duplicate_ratio": float(exact_duplicate_points / len(points)),
        "duplicate_voxel_size": voxel_size,
        "duplicate_voxel_points": duplicate_voxel_points,
        "duplicate_voxel_ratio": duplicate_voxel_ratio,
        "near_duplicate_radius": near_radius,
        "near_duplicate_points": near_duplicate_points,
        "near_duplicate_ratio": float(near_duplicate_points / len(points)),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra = []
    for row in rows:
        for key in row:
            if key not in csv_fieldnames() and key not in extra:
                extra.append(key)
    fieldnames = csv_fieldnames() + extra
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, NA_VALUE) for key in fieldnames})


def write_summary(path: Path, row: Dict[str, object]) -> None:
    lines = [
        "# PU-GCN Single-Frame Upsampling Metrics",
        "",
        f"- method_name: {row.get('method_name')}",
        f"- sample_id: {row.get('sample_id')}",
        f"- input_points: {row.get('input_points')}",
        f"- output_points: {row.get('output_points')}",
        f"- upsampling_ratio: {row.get('upsampling_ratio')}",
        f"- nn_mean_before: {row.get('nn_mean_before')}",
        f"- nn_mean_after: {row.get('nn_mean_after')}",
        f"- nn_median_before: {row.get('nn_median_before')}",
        f"- nn_median_after: {row.get('nn_median_after')}",
        f"- output_to_input_nn_mean: {row.get('output_to_input_nn_mean')}",
        f"- output_to_input_nn_median: {row.get('output_to_input_nn_median')}",
        f"- within_kitti_range_before: {row.get('within_kitti_range_before')}",
        f"- within_kitti_range_after: {row.get('within_kitti_range_after')}",
        f"- output_invalid_xyz_points: {row.get('output_invalid_xyz_points')}",
        f"- output_out_of_range_points: {row.get('output_out_of_range_points')}",
        f"- exact_duplicate_points: {row.get('exact_duplicate_points')}",
        f"- duplicate_voxel_ratio: {row.get('duplicate_voxel_ratio')}",
        f"- near_duplicate_ratio: {row.get('near_duplicate_ratio')}",
        "",
        "Metric names from the existing density schema are preserved where available; PU-GCN-specific nearest-neighbor and invalid-point fields are appended.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sample_id = args.sample_id or args.input_bin.stem
    reference = load_kitti_bin(args.input_bin)
    processed = load_kitti_bin(args.output_bin)
    viz_dir = args.output_root / "visualizations" / sample_id if args.write_visualizations else None
    row = evaluate_pair(
        sample_id=sample_id,
        reference=reference,
        processed=processed,
        experiment_name=args.experiment_name,
        method_name=args.method_name,
        reference_path=args.input_bin,
        processed_path=args.output_bin,
        transform_stage="pugcn_single_frame",
        intensity_strategy="nearest_original",
        processing_type="upsample",
        lidar_range=DEFAULT_KITTI_LIDAR_RANGE,
        cell_size=args.cell_size,
        sample_viz_dir=viz_dir,
        seed=args.seed,
    )
    row.update(invalid_stats("input", reference))
    row.update(invalid_stats("output", processed))
    row.update(kitti_range_stats("input", reference))
    row.update(kitti_range_stats("output", processed))
    row.update(nn_output_to_input(processed, reference))
    row.update(self_nn_min_max(processed))
    row.update({f"output_{k}": v for k, v in local_neighbor_stats(processed).items()})
    row.update(duplicate_stats(processed, args.duplicate_voxel_size, args.near_duplicate_radius))
    write_csv(args.output_root / "single_frame_metrics.csv", [row])
    write_summary(args.output_root / "single_frame_summary.md", row)
    print("Wrote upsampling metrics:", args.output_root)


if __name__ == "__main__":
    main()
