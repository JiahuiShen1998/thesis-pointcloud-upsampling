#!/usr/bin/env python3
"""Reusable density evaluation framework for KITTI point clouds.

This script standardizes the evaluation protocol for point-cloud-only methods
today and leaves detector metrics in the same table for future experiments.

Core design goals:
  - one metric schema for all methods,
  - one CSV column layout for sample-level and method-level records,
  - one visualization protocol,
  - detector columns kept in place as N/A until detector metrics are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.density_metric_schema import (
    ALL_COLUMNS,
    DEFAULT_BEV_CELL_SIZE,
    DEFAULT_KITTI_LIDAR_RANGE,
    DEFAULT_MAX_VISUALIZED_SAMPLES,
    NA_VALUE,
    SCHEMA_VERSION,
    VISUALIZATION_FILENAMES,
    blank_record,
    csv_fieldnames,
)

try:  # pragma: no cover - optional dependency
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional dependency fallback
    cKDTree = None


KITTI_BIN_COLUMNS = 4
PLOT_MAX_POINTS = 50_000
POINT_ROUND_DECIMALS = 6


@dataclass(frozen=True)
class MethodSpec:
    method_name: str
    input_subdir: str
    output_subdir: str
    transform_stage: str
    intensity_strategy: str = "preserved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate density methods with a unified schema")
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("data/KITTI/object/training"),
        help="KITTI training directory containing velodyne folders.",
    )
    parser.add_argument(
        "--reference-velodyne",
        type=Path,
        required=True,
        help="Reference KITTI velodyne folder, usually the original baseline input.",
    )
    parser.add_argument(
        "--processed-velodyne",
        type=Path,
        required=True,
        help="Processed velodyne folder produced by the selected method.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where the experiment record folders will be written.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Human-readable experiment name, e.g. EAR_original_upsampled.",
    )
    parser.add_argument(
        "--method-name",
        type=str,
        required=True,
        help="Method label used in tables and record folders, e.g. EAR or PU-Net.",
    )
    parser.add_argument(
        "--processing-type",
        type=str,
        required=True,
        choices=("original", "upsample", "downsample", "down_up"),
        help="High-level processing category for the comparison.",
    )
    parser.add_argument(
        "--detector-results-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory containing detector experiment folders. "
            "If present, the script looks for per-experiment evaluation/parsed_ap_results.csv."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="KITTI",
        help="Dataset name recorded in the output tables.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split recorded in the output tables.",
    )
    parser.add_argument(
        "--max-visualized-samples",
        type=int,
        default=DEFAULT_MAX_VISUALIZED_SAMPLES,
        help="How many representative samples to visualize per method.",
    )
    parser.add_argument(
        "--bev-cell-size",
        type=float,
        default=DEFAULT_BEV_CELL_SIZE,
        help="Grid size in meters for BEV density statistics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Random seed used for representative sample selection.",
    )
    return parser.parse_args()


def load_points(path: Path) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32)
    if points.size % KITTI_BIN_COLUMNS != 0:
        raise ValueError(f"Invalid KITTI bin file shape: {path}")
    return points.reshape(-1, KITTI_BIN_COLUMNS)


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, NA_VALUE) for k in csv_fieldnames()})


def _numeric(values: Iterable[object]) -> List[float]:
    out: List[float] = []
    for value in values:
        if value in (None, "", NA_VALUE):
            continue
        try:
            out.append(float(value))
        except Exception:
            continue
    return out


def _format_value(value: object) -> object:
    if value is None:
        return NA_VALUE
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if math.isnan(float(value)):
            return NA_VALUE
        return round(float(value), 6)
    return value


def _load_default_config(source_subdir: str) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "KITTI",
        "split": "val",
        "experiment_name": "density_pointcloud_evaluation",
        "source_subdir": source_subdir,
        "methods": [
            {
                "method_name": "original",
                "input_subdir": source_subdir,
                "output_subdir": source_subdir,
                "transform_stage": "identity",
                "intensity_strategy": "preserved",
            },
            {
                "method_name": "original_upsampled",
                "input_subdir": source_subdir,
                "output_subdir": "velodyne_upsampled",
                "transform_stage": "upsampled",
                "intensity_strategy": "knn_interpolated_or_local_average",
            },
            {
                "method_name": "downsampled",
                "input_subdir": source_subdir,
                "output_subdir": "velodyne_downsampled",
                "transform_stage": "downsampled",
                "intensity_strategy": "preserved",
            },
            {
                "method_name": "down_up",
                "input_subdir": source_subdir,
                "output_subdir": "velodyne_down_up",
                "transform_stage": "downsampled_then_upsampled",
                "intensity_strategy": "knn_interpolated_or_local_average",
            },
        ],
        "kitti_lidar_range": DEFAULT_KITTI_LIDAR_RANGE,
        "visualization": {
            "max_visualized_samples": DEFAULT_MAX_VISUALIZED_SAMPLES,
            "bev_cell_size": DEFAULT_BEV_CELL_SIZE,
        },
    }


def load_config(path: Optional[Path], source_subdir: str) -> Dict[str, object]:
    if path is None:
        return _load_default_config(source_subdir)
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("dataset", "KITTI")
    data.setdefault("split", "val")
    data.setdefault("source_subdir", source_subdir)
    data.setdefault("kitti_lidar_range", DEFAULT_KITTI_LIDAR_RANGE)
    data.setdefault(
        "visualization",
        {
            "max_visualized_samples": DEFAULT_MAX_VISUALIZED_SAMPLES,
            "bev_cell_size": DEFAULT_BEV_CELL_SIZE,
        },
    )
    return data


def list_bin_files(source_dir: Path) -> List[Path]:
    return sorted(source_dir.glob("*.bin"))


def load_detector_metrics(
    detector_results_root: Optional[Path],
    method_name: str,
    experiment_name: str,
) -> Dict[str, object]:
    if detector_results_root is None:
        return {}
    candidates = [
        detector_results_root / method_name / "evaluation" / "parsed_ap_results.csv",
        detector_results_root / experiment_name / "evaluation" / "parsed_ap_results.csv",
        detector_results_root / method_name / "parsed_ap_results.csv",
        detector_results_root / experiment_name / "parsed_ap_results.csv",
    ]
    parsed_path = next((p for p in candidates if p.exists()), None)
    if parsed_path is None:
        return {}

    metrics = {
        "detector_result_source": str(parsed_path.resolve()),
        "detector_bbox_ap_easy": NA_VALUE,
        "detector_bbox_ap_moderate": NA_VALUE,
        "detector_bbox_ap_hard": NA_VALUE,
        "detector_bev_ap_easy": NA_VALUE,
        "detector_bev_ap_moderate": NA_VALUE,
        "detector_bev_ap_hard": NA_VALUE,
        "detector_3d_ap_easy": NA_VALUE,
        "detector_3d_ap_moderate": NA_VALUE,
        "detector_3d_ap_hard": NA_VALUE,
    }

    with parsed_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    by_metric = {row["metric"].strip().lower(): row for row in rows if row.get("metric")}
    mapping = {
        "bbox": ("detector_bbox_ap_easy", "detector_bbox_ap_moderate", "detector_bbox_ap_hard"),
        "bev": ("detector_bev_ap_easy", "detector_bev_ap_moderate", "detector_bev_ap_hard"),
        "3d": ("detector_3d_ap_easy", "detector_3d_ap_moderate", "detector_3d_ap_hard"),
    }
    for metric_name, fields in mapping.items():
        row = by_metric.get(metric_name)
        if not row:
            continue
        for field, key in zip(fields, ("easy", "moderate", "hard")):
            metrics[field] = _format_value(row.get(key, NA_VALUE))
    return metrics


def point_hash_set(points: np.ndarray) -> set:
    rounded = np.round(points.astype(np.float64, copy=False), POINT_ROUND_DECIMALS)
    return {tuple(row.tolist()) for row in rounded}


def identify_added_points(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    before_hashes = point_hash_set(before)
    rounded_after = np.round(after.astype(np.float64, copy=False), POINT_ROUND_DECIMALS)
    mask = np.array([tuple(row.tolist()) not in before_hashes for row in rounded_after], dtype=bool)
    return mask


def compute_nn_stats(points: np.ndarray) -> Tuple[object, object, object]:
    if len(points) < 2:
        return NA_VALUE, NA_VALUE, NA_VALUE
    xyz = points[:, :3]
    if cKDTree is not None:
        tree = cKDTree(xyz)
        dists, _ = tree.query(xyz, k=2)
        nn = np.asarray(dists[:, 1], dtype=np.float64)
    else:  # pragma: no cover - fallback when scipy is unavailable
        nn = np.empty(len(xyz), dtype=np.float64)
        for i in range(len(xyz)):
            diff = xyz - xyz[i]
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            dist[i] = np.inf
            nn[i] = float(np.min(dist))
    return float(np.mean(nn)), float(np.median(nn)), float(np.std(nn))


def compute_bev_density_stats(points: np.ndarray, cell_size: float) -> Tuple[object, object, object, object, object]:
    if len(points) == 0:
        return NA_VALUE, NA_VALUE, NA_VALUE, NA_VALUE, NA_VALUE
    xy = points[:, :2]
    x_min = float(np.min(xy[:, 0]))
    x_max = float(np.max(xy[:, 0]))
    y_min = float(np.min(xy[:, 1]))
    y_max = float(np.max(xy[:, 1]))
    if x_max == x_min:
        x_max = x_min + cell_size
    if y_max == y_min:
        y_max = y_min + cell_size
    x_bins = max(1, int(math.ceil((x_max - x_min) / cell_size)))
    y_bins = max(1, int(math.ceil((y_max - y_min) / cell_size)))
    x_idx = np.clip(((xy[:, 0] - x_min) / cell_size).astype(np.int64), 0, x_bins - 1)
    y_idx = np.clip(((xy[:, 1] - y_min) / cell_size).astype(np.int64), 0, y_bins - 1)
    flat = x_idx * y_bins + y_idx
    counts = np.bincount(flat, minlength=x_bins * y_bins).astype(np.float64)
    occupied = counts[counts > 0]
    coverage = float(len(occupied) / len(counts)) if len(counts) else 0.0
    return (
        float(np.mean(occupied)) if len(occupied) else 0.0,
        float(np.median(occupied)) if len(occupied) else 0.0,
        float(np.std(occupied)) if len(occupied) else 0.0,
        int(len(occupied)),
        coverage,
    )


def compute_spatial_stats(points: np.ndarray, lidar_range: Dict[str, float]) -> Tuple[object, object, object, object, object, object, object, object, object, object, object, object, bool]:
    if len(points) == 0:
        return (NA_VALUE,) * 12 + (False,)
    xyz = points[:, :3]
    x_min = float(np.min(xyz[:, 0]))
    x_max = float(np.max(xyz[:, 0]))
    y_min = float(np.min(xyz[:, 1]))
    y_max = float(np.max(xyz[:, 1]))
    z_min = float(np.min(xyz[:, 2]))
    z_max = float(np.max(xyz[:, 2]))
    within = (
        np.isfinite(xyz).all()
        and x_min >= lidar_range["x_min"]
        and x_max <= lidar_range["x_max"]
        and y_min >= lidar_range["y_min"]
        and y_max <= lidar_range["y_max"]
        and z_min >= lidar_range["z_min"]
        and z_max <= lidar_range["z_max"]
    )
    return (
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        bool(within),
    )


def compute_intensity_stats(points: np.ndarray) -> Tuple[object, object, object, object]:
    if len(points) == 0:
        return NA_VALUE, NA_VALUE, NA_VALUE, NA_VALUE
    intensity = points[:, 3]
    return float(np.min(intensity)), float(np.max(intensity)), float(np.mean(intensity)), float(np.std(intensity))


def _downsample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if len(points) <= max_points:
        return points
    idx = rng.choice(len(points), size=max_points, replace=False)
    idx.sort()
    return points[idx]


def render_visualizations(
    before: np.ndarray,
    after: np.ndarray,
    output_dir: Path,
    sample_id: str,
    cell_size: float,
    seed: int,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - visualizations are optional
        return {key: NA_VALUE for key in VISUALIZATION_FILENAMES}

    rng = np.random.default_rng(seed)
    before_plot = _downsample_points(before, PLOT_MAX_POINTS, rng)
    after_plot = _downsample_points(after, PLOT_MAX_POINTS, rng)
    added_mask = identify_added_points(before, after)
    added = after[added_mask]
    added_plot = _downsample_points(added, PLOT_MAX_POINTS, rng) if len(added) else added

    xy_all = np.concatenate([before[:, :2], after[:, :2]], axis=0) if len(before) and len(after) else (
        before[:, :2] if len(before) else after[:, :2]
    )
    x_min, y_min = np.min(xy_all[:, 0]), np.min(xy_all[:, 1])
    x_max, y_max = np.max(xy_all[:, 0]), np.max(xy_all[:, 1])
    if len(added):
        zoom_center = np.median(added[:, :2], axis=0)
    else:
        zoom_center = np.median(xy_all, axis=0)
    zoom_half = max(8.0, min(20.0, 0.35 * max(x_max - x_min, y_max - y_min)))
    z_x0 = zoom_center[0] - zoom_half
    z_x1 = zoom_center[0] + zoom_half
    z_y0 = zoom_center[1] - zoom_half
    z_y1 = zoom_center[1] + zoom_half

    def _save(fig, filename: str) -> str:
        path = output_dir / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    # BEV comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    for ax, pts, title, color in [
        (axes[0], before_plot, "BEV before", "tab:gray"),
        (axes[1], after_plot, "BEV after", "tab:blue"),
    ]:
        ax.scatter(pts[:, 0], pts[:, 1], s=0.15, c=color, alpha=0.5, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)
    axes[2].scatter(before_plot[:, 0], before_plot[:, 1], s=0.15, c="tab:gray", alpha=0.3, linewidths=0)
    axes[2].scatter(after_plot[:, 0], after_plot[:, 1], s=0.15, c="tab:blue", alpha=0.35, linewidths=0)
    if len(added_plot):
        axes[2].scatter(added_plot[:, 0], added_plot[:, 1], s=0.25, c="tab:red", alpha=0.8, linewidths=0)
    axes[2].set_title("BEV overlay")
    axes[2].set_xlabel("x (m)")
    axes[2].set_ylabel("y (m)")
    axes[2].set_xlim(x_min, x_max)
    axes[2].set_ylim(y_min, y_max)
    axes[2].set_aspect("equal", adjustable="box")
    axes[3].scatter(after_plot[:, 0], after_plot[:, 1], s=0.15, c="tab:blue", alpha=0.3, linewidths=0)
    if len(added_plot):
        axes[3].scatter(added_plot[:, 0], added_plot[:, 1], s=0.4, c="tab:red", alpha=0.85, linewidths=0)
    axes[3].set_title("BEV zoom on added points")
    axes[3].set_xlabel("x (m)")
    axes[3].set_ylabel("y (m)")
    axes[3].set_xlim(z_x0, z_x1)
    axes[3].set_ylim(z_y0, z_y1)
    axes[3].set_aspect("equal", adjustable="box")
    bev_compare = _save(fig, VISUALIZATION_FILENAMES["bev_overlay_png"])

    # BEV single-view variants
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(before_plot[:, 0], before_plot[:, 1], s=0.15, c="tab:gray", alpha=0.55, linewidths=0)
    ax.set_title("BEV before")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    bev_before = _save(fig, VISUALIZATION_FILENAMES["bev_before_png"])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(after_plot[:, 0], after_plot[:, 1], s=0.15, c="tab:blue", alpha=0.55, linewidths=0)
    if len(added_plot):
        ax.scatter(added_plot[:, 0], added_plot[:, 1], s=0.25, c="tab:red", alpha=0.9, linewidths=0)
    ax.set_title("BEV after")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    bev_after = _save(fig, VISUALIZATION_FILENAMES["bev_after_png"])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(before_plot[:, 0], before_plot[:, 1], s=0.15, c="tab:gray", alpha=0.25, linewidths=0)
    ax.scatter(after_plot[:, 0], after_plot[:, 1], s=0.15, c="tab:blue", alpha=0.35, linewidths=0)
    if len(added_plot):
        ax.scatter(added_plot[:, 0], added_plot[:, 1], s=0.35, c="tab:red", alpha=0.9, linewidths=0)
    ax.set_title("BEV zoom")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(z_x0, z_x1)
    ax.set_ylim(z_y0, z_y1)
    ax.set_aspect("equal", adjustable="box")
    bev_zoom = _save(fig, VISUALIZATION_FILENAMES["bev_zoom_png"])

    # 3D cloud overlays
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(before_plot[:, 0], before_plot[:, 1], before_plot[:, 2], s=0.12, c="tab:gray", alpha=0.25)
    ax.scatter(after_plot[:, 0], after_plot[:, 1], after_plot[:, 2], s=0.12, c="tab:blue", alpha=0.25)
    if len(added_plot):
        ax.scatter(added_plot[:, 0], added_plot[:, 1], added_plot[:, 2], s=0.3, c="tab:red", alpha=0.9)
    ax.set_title("3D point cloud overlay")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    cloud_overlay = _save(fig, VISUALIZATION_FILENAMES["cloud_overlay_png"])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(before_plot[:, 0], before_plot[:, 1], before_plot[:, 2], s=0.12, c=before_plot[:, 2], cmap="viridis", alpha=0.35)
    ax.set_title("3D point cloud before")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    cloud_before = _save(fig, VISUALIZATION_FILENAMES["cloud_before_png"])

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(after_plot[:, 0], after_plot[:, 1], after_plot[:, 2], s=0.12, c=after_plot[:, 2], cmap="plasma", alpha=0.35)
    if len(added_plot):
        ax.scatter(added_plot[:, 0], added_plot[:, 1], added_plot[:, 2], s=0.3, c="tab:red", alpha=0.95)
    ax.set_title("3D point cloud after")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    cloud_after = _save(fig, VISUALIZATION_FILENAMES["cloud_after_png"])

    return {
        "bev_before_png": bev_before,
        "bev_after_png": bev_after,
        "bev_overlay_png": bev_compare,
        "bev_zoom_png": bev_zoom,
        "cloud_before_png": cloud_before,
        "cloud_after_png": cloud_after,
        "cloud_overlay_png": cloud_overlay,
    }


def evaluate_pair(
    sample_id: str,
    reference: np.ndarray,
    processed: np.ndarray,
    experiment_name: str,
    method_name: str,
    reference_path: Path,
    processed_path: Path,
    transform_stage: str,
    intensity_strategy: str,
    processing_type: str,
    lidar_range: Dict[str, float],
    cell_size: float,
    sample_viz_dir: Optional[Path],
    seed: int,
) -> Dict[str, object]:
    record = blank_record()
    record.update(
        {
            "schema_version": SCHEMA_VERSION,
            "scope": "sample",
            "dataset": "KITTI",
            "split": "val",
            "experiment_name": experiment_name,
            "method_name": method_name,
            "processing_type": processing_type,
            "sample_id": sample_id,
            "reference_velodyne": str(reference_path.resolve()),
            "processed_velodyne": str(processed_path.resolve()),
            "source_path": str(reference_path.resolve()),
            "input_path": str(reference_path.resolve()),
            "output_path": str(processed_path.resolve()),
            "transform_stage": transform_stage,
            "intensity_strategy": intensity_strategy,
            "format_ok": bool(reference.ndim == 2 and processed.ndim == 2 and reference.shape[1] == 4 and processed.shape[1] == 4),
            "coordinate_frame_preserved": bool(reference.ndim == 2 and processed.ndim == 2 and reference.shape[1] == 4 and processed.shape[1] == 4),
        }
    )
    before_shape = reference.shape[0]
    after_shape = processed.shape[0]
    record["input_points"] = int(before_shape)
    record["output_points"] = int(after_shape)
    record["added_points"] = int(max(after_shape - before_shape, 0))
    record["removed_points"] = int(max(before_shape - after_shape, 0))
    record["upsampling_ratio"] = float(after_shape / before_shape) if before_shape > 0 else NA_VALUE

    nn_before = compute_nn_stats(reference)
    nn_after = compute_nn_stats(processed)
    record["nn_mean_before"], record["nn_median_before"], record["nn_std_before"] = nn_before
    record["nn_mean_after"], record["nn_median_after"], record["nn_std_after"] = nn_after

    bev_before = compute_bev_density_stats(reference, cell_size)
    bev_after = compute_bev_density_stats(processed, cell_size)
    record["bev_density_mean_before"], record["bev_density_median_before"], record["bev_density_std_before"], record["bev_density_occupied_cells_before"], record["bev_density_coverage_before"] = bev_before
    record["bev_density_mean_after"], record["bev_density_median_after"], record["bev_density_std_after"], record["bev_density_occupied_cells_after"], record["bev_density_coverage_after"] = bev_after

    (
        record["x_min_before"],
        record["x_max_before"],
        record["y_min_before"],
        record["y_max_before"],
        record["z_min_before"],
        record["z_max_before"],
        record["x_min_after"],
        record["x_max_after"],
        record["y_min_after"],
        record["y_max_after"],
        record["z_min_after"],
        record["z_max_after"],
        record["within_kitti_range_after"],
    ) = (NA_VALUE,) * 12 + (False,)
    record["x_min_before"] = float(np.min(reference[:, 0])) if len(reference) else NA_VALUE
    record["x_max_before"] = float(np.max(reference[:, 0])) if len(reference) else NA_VALUE
    record["y_min_before"] = float(np.min(reference[:, 1])) if len(reference) else NA_VALUE
    record["y_max_before"] = float(np.max(reference[:, 1])) if len(reference) else NA_VALUE
    record["z_min_before"] = float(np.min(reference[:, 2])) if len(reference) else NA_VALUE
    record["z_max_before"] = float(np.max(reference[:, 2])) if len(reference) else NA_VALUE
    record["x_min_after"] = float(np.min(processed[:, 0])) if len(processed) else NA_VALUE
    record["x_max_after"] = float(np.max(processed[:, 0])) if len(processed) else NA_VALUE
    record["y_min_after"] = float(np.min(processed[:, 1])) if len(processed) else NA_VALUE
    record["y_max_after"] = float(np.max(processed[:, 1])) if len(processed) else NA_VALUE
    record["z_min_after"] = float(np.min(processed[:, 2])) if len(processed) else NA_VALUE
    record["z_max_after"] = float(np.max(processed[:, 2])) if len(processed) else NA_VALUE
    record["within_kitti_range_before"] = bool(_within_range(reference, lidar_range))
    record["within_kitti_range_after"] = bool(_within_range(processed, lidar_range))

    int_before = compute_intensity_stats(reference)
    int_after = compute_intensity_stats(processed)
    record["intensity_min_before"], record["intensity_max_before"], record["intensity_mean_before"], record["intensity_std_before"] = int_before
    record["intensity_min_after"], record["intensity_max_after"], record["intensity_mean_after"], record["intensity_std_after"] = int_after

    if sample_viz_dir is not None:
        viz = render_visualizations(reference, processed, sample_viz_dir, sample_id, cell_size=cell_size, seed=seed)
        record.update(viz)
    else:
        for key in VISUALIZATION_FILENAMES:
            record[key] = NA_VALUE

    record["summary_note"] = "sample-level metrics"
    return record


def _within_range(points: np.ndarray, lidar_range: Dict[str, float]) -> bool:
    if len(points) == 0:
        return False
    xyz = points[:, :3]
    finite = np.isfinite(xyz).all()
    if not finite:
        return False
    return bool(
        np.all(xyz[:, 0] >= lidar_range["x_min"])
        and np.all(xyz[:, 0] <= lidar_range["x_max"])
        and np.all(xyz[:, 1] >= lidar_range["y_min"])
        and np.all(xyz[:, 1] <= lidar_range["y_max"])
        and np.all(xyz[:, 2] >= lidar_range["z_min"])
        and np.all(xyz[:, 2] <= lidar_range["z_max"])
    )


def aggregate_rows(
    experiment_name: str,
    method_name: str,
    processing_type: str,
    reference_velodyne: Path,
    processed_velodyne: Path,
    rows: Sequence[Dict[str, object]],
    detector_metrics: Optional[Dict[str, object]] = None,
    summary_visual_dir: Optional[Path] = None,
    summary_note: str = "",
) -> Dict[str, object]:
    summary = blank_record()
    summary.update(
        {
            "schema_version": SCHEMA_VERSION,
            "scope": "method",
            "dataset": "KITTI",
            "split": "val",
            "experiment_name": experiment_name,
            "method_name": method_name,
            "processing_type": processing_type,
            "sample_id": "__aggregate__",
            "reference_velodyne": str(reference_velodyne.resolve()),
            "processed_velodyne": str(processed_velodyne.resolve()),
            "source_path": str(reference_velodyne.resolve()),
            "input_path": str(reference_velodyne.resolve()),
            "output_path": str(processed_velodyne.resolve()),
            "summary_note": summary_note or "method-level aggregate",
        }
    )
    if not rows:
        return summary

    numeric_fields = [
        "input_points",
        "output_points",
        "added_points",
        "removed_points",
        "upsampling_ratio",
        "nn_mean_before",
        "nn_mean_after",
        "nn_median_before",
        "nn_median_after",
        "nn_std_before",
        "nn_std_after",
        "bev_density_mean_before",
        "bev_density_mean_after",
        "bev_density_median_before",
        "bev_density_median_after",
        "bev_density_std_before",
        "bev_density_std_after",
        "bev_density_occupied_cells_before",
        "bev_density_occupied_cells_after",
        "bev_density_coverage_before",
        "bev_density_coverage_after",
        "intensity_min_before",
        "intensity_min_after",
        "intensity_max_before",
        "intensity_max_after",
        "intensity_mean_before",
        "intensity_mean_after",
        "intensity_std_before",
        "intensity_std_after",
    ]
    for field in numeric_fields:
        vals = _numeric(row.get(field) for row in rows)
        if not vals:
            continue
        if field in {
            "input_points",
            "output_points",
            "added_points",
            "removed_points",
            "bev_density_occupied_cells_before",
            "bev_density_occupied_cells_after",
        }:
            summary[field] = int(round(float(np.mean(vals))))
        elif field.startswith(("x_min", "y_min", "z_min")):
            summary[field] = float(np.min(vals))
        elif field.startswith(("x_max", "y_max", "z_max")):
            summary[field] = float(np.max(vals))
        elif field in {"within_kitti_range_before", "within_kitti_range_after", "format_ok", "coordinate_frame_preserved"}:
            summary[field] = bool(all(bool(v) for v in vals))
        else:
            summary[field] = float(np.mean(vals))

    summary["upsampling_ratio"] = float(np.mean(_numeric(row.get("upsampling_ratio") for row in rows))) if _numeric(
        row.get("upsampling_ratio") for row in rows
    ) else NA_VALUE

    summary["x_min_before"] = float(np.min(_numeric(row.get("x_min_before") for row in rows))) if _numeric(
        row.get("x_min_before") for row in rows
    ) else NA_VALUE
    summary["x_max_before"] = float(np.max(_numeric(row.get("x_max_before") for row in rows))) if _numeric(
        row.get("x_max_before") for row in rows
    ) else NA_VALUE
    summary["y_min_before"] = float(np.min(_numeric(row.get("y_min_before") for row in rows))) if _numeric(
        row.get("y_min_before") for row in rows
    ) else NA_VALUE
    summary["y_max_before"] = float(np.max(_numeric(row.get("y_max_before") for row in rows))) if _numeric(
        row.get("y_max_before") for row in rows
    ) else NA_VALUE
    summary["z_min_before"] = float(np.min(_numeric(row.get("z_min_before") for row in rows))) if _numeric(
        row.get("z_min_before") for row in rows
    ) else NA_VALUE
    summary["z_max_before"] = float(np.max(_numeric(row.get("z_max_before") for row in rows))) if _numeric(
        row.get("z_max_before") for row in rows
    ) else NA_VALUE
    summary["x_min_after"] = float(np.min(_numeric(row.get("x_min_after") for row in rows))) if _numeric(
        row.get("x_min_after") for row in rows
    ) else NA_VALUE
    summary["x_max_after"] = float(np.max(_numeric(row.get("x_max_after") for row in rows))) if _numeric(
        row.get("x_max_after") for row in rows
    ) else NA_VALUE
    summary["y_min_after"] = float(np.min(_numeric(row.get("y_min_after") for row in rows))) if _numeric(
        row.get("y_min_after") for row in rows
    ) else NA_VALUE
    summary["y_max_after"] = float(np.max(_numeric(row.get("y_max_after") for row in rows))) if _numeric(
        row.get("y_max_after") for row in rows
    ) else NA_VALUE
    summary["z_min_after"] = float(np.min(_numeric(row.get("z_min_after") for row in rows))) if _numeric(
        row.get("z_min_after") for row in rows
    ) else NA_VALUE
    summary["z_max_after"] = float(np.max(_numeric(row.get("z_max_after") for row in rows))) if _numeric(
        row.get("z_max_after") for row in rows
    ) else NA_VALUE
    summary["format_ok"] = bool(all(bool(row.get("format_ok", False)) for row in rows))
    summary["coordinate_frame_preserved"] = bool(all(bool(row.get("coordinate_frame_preserved", False)) for row in rows))
    summary["within_kitti_range_before"] = bool(all(bool(row.get("within_kitti_range_before", False)) for row in rows))
    summary["within_kitti_range_after"] = bool(all(bool(row.get("within_kitti_range_after", False)) for row in rows))

    if summary_visual_dir is not None:
        summary_visual_dir.mkdir(parents=True, exist_ok=True)
        summary["bev_before_png"] = str((summary_visual_dir / "point_count_hist.png").resolve())
        summary["bev_after_png"] = str((summary_visual_dir / "nn_distance_hist.png").resolve())
        summary["bev_overlay_png"] = str((summary_visual_dir / "bev_density_hist.png").resolve())
        summary["bev_zoom_png"] = str((summary_visual_dir / "summary_report.md").resolve())
        summary["cloud_before_png"] = summary["bev_before_png"]
        summary["cloud_after_png"] = summary["bev_after_png"]
        summary["cloud_overlay_png"] = summary["bev_overlay_png"]

    if detector_metrics:
        summary.update(detector_metrics)
    else:
        summary.setdefault("detector_result_source", NA_VALUE)
        summary.setdefault("detector_bbox_ap_easy", NA_VALUE)
        summary.setdefault("detector_bbox_ap_moderate", NA_VALUE)
        summary.setdefault("detector_bbox_ap_hard", NA_VALUE)
        summary.setdefault("detector_bev_ap_easy", NA_VALUE)
        summary.setdefault("detector_bev_ap_moderate", NA_VALUE)
        summary.setdefault("detector_bev_ap_hard", NA_VALUE)
        summary.setdefault("detector_3d_ap_easy", NA_VALUE)
        summary.setdefault("detector_3d_ap_moderate", NA_VALUE)
        summary.setdefault("detector_3d_ap_hard", NA_VALUE)
        summary.setdefault("detector_precision_easy", NA_VALUE)
        summary.setdefault("detector_precision_moderate", NA_VALUE)
        summary.setdefault("detector_precision_hard", NA_VALUE)
        summary.setdefault("detector_recall_easy", NA_VALUE)
        summary.setdefault("detector_recall_moderate", NA_VALUE)
        summary.setdefault("detector_recall_hard", NA_VALUE)
    return summary


def select_visualization_samples(rows: Sequence[Dict[str, object]], max_samples: int, seed: int) -> List[str]:
    if not rows or max_samples <= 0:
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("output_points", 0) or 0),
            float(row.get("input_points", 0) or 0),
            str(row.get("sample_id", "")),
        ),
    )
    if len(ranked) <= max_samples:
        return [str(row["sample_id"]) for row in ranked]
    idxs = np.linspace(0, len(ranked) - 1, num=max_samples, dtype=int)
    return [str(ranked[i]["sample_id"]) for i in idxs]


def write_histogram(path: Path, values: Sequence[float], title: str, xlabel: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional visualization support
        path.write_text("matplotlib unavailable\n", encoding="utf-8")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=40, color="steelblue", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_method_report(
    method_dir: Path,
    method_name: str,
    sample_rows: Sequence[Dict[str, object]],
    summary_row: Dict[str, object],
    config: Dict[str, object],
    detector_metrics: Dict[str, object],
) -> None:
    method_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [
        f"# {method_name}",
        "",
        f"- schema_version: {SCHEMA_VERSION}",
        f"- dataset: {config.get('dataset', 'KITTI')}",
        f"- split: {config.get('split', 'val')}",
        f"- scope: pointcloud-only with detector placeholders",
        f"- reference_velodyne: {config.get('reference_velodyne', NA_VALUE)}",
        f"- processed_velodyne: {config.get('processed_velodyne', NA_VALUE)}",
        f"- processing_type: {config.get('processing_type', NA_VALUE)}",
        f"- detector metrics available: {'yes' if detector_metrics else 'no'}",
        "",
        "## Summary",
        "",
        f"- input_points: {summary_row.get('input_points', NA_VALUE)}",
        f"- output_points: {summary_row.get('output_points', NA_VALUE)}",
        f"- added_points: {summary_row.get('added_points', NA_VALUE)}",
        f"- removed_points: {summary_row.get('removed_points', NA_VALUE)}",
        f"- upsampling_ratio: {summary_row.get('upsampling_ratio', NA_VALUE)}",
        f"- nn_mean_before: {summary_row.get('nn_mean_before', NA_VALUE)}",
        f"- nn_mean_after: {summary_row.get('nn_mean_after', NA_VALUE)}",
        f"- bev_density_mean_before: {summary_row.get('bev_density_mean_before', NA_VALUE)}",
        f"- bev_density_mean_after: {summary_row.get('bev_density_mean_after', NA_VALUE)}",
        f"- intensity_mean_before: {summary_row.get('intensity_mean_before', NA_VALUE)}",
        f"- intensity_mean_after: {summary_row.get('intensity_mean_after', NA_VALUE)}",
        "",
        "## Detector Columns",
        "",
        "- If detector metrics are not available, the detector columns remain `N/A` by design.",
    ]
    if detector_metrics:
        report_lines.extend(
            [
                "",
                "## Detector Metrics",
                "",
                f"- detector_bbox_ap_moderate: {summary_row.get('detector_bbox_ap_moderate', NA_VALUE)}",
                f"- detector_bev_ap_moderate: {summary_row.get('detector_bev_ap_moderate', NA_VALUE)}",
                f"- detector_3d_ap_moderate: {summary_row.get('detector_3d_ap_moderate', NA_VALUE)}",
            ]
        )
    (method_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def method_from_spec(spec: Dict[str, object]) -> MethodSpec:
    return MethodSpec(
        method_name=str(spec["method_name"]),
        input_subdir=str(spec["input_subdir"]),
        output_subdir=str(spec["output_subdir"]),
        transform_stage=str(spec.get("transform_stage", "unknown")),
        intensity_strategy=str(spec.get("intensity_strategy", "preserved")),
    )


def main() -> int:
    args = parse_args()
    training_dir = args.training_dir.resolve()
    reference_dir = args.reference_velodyne.resolve()
    processed_dir = args.processed_velodyne.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not reference_dir.exists():
        raise FileNotFoundError(f"Reference velodyne folder does not exist: {reference_dir}")
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed velodyne folder does not exist: {processed_dir}")

    config_dir = output_root / "config"
    preprocessing_dir = output_root / "preprocessing"
    evaluation_dir = output_root / "evaluation"
    visualization_dir = output_root / "visualization"
    summary_dir = output_root / "summary"
    git_dir = output_root / "git_record"
    notes_dir = output_root / "notes"
    for directory in (config_dir, preprocessing_dir, evaluation_dir, visualization_dir, summary_dir, git_dir, notes_dir):
        directory.mkdir(parents=True, exist_ok=True)

    experiment_config = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": args.experiment_name,
        "method_name": args.method_name,
        "processing_type": args.processing_type,
        "dataset": args.dataset,
        "split": args.split,
        "training_dir": str(training_dir),
        "reference_velodyne": str(reference_dir),
        "processed_velodyne": str(processed_dir),
        "output_root": str(output_root),
        "max_visualized_samples": args.max_visualized_samples,
        "bev_cell_size": args.bev_cell_size,
        "seed": args.seed,
        "detector_results_root": str(args.detector_results_root.resolve()) if args.detector_results_root else NA_VALUE,
    }
    (config_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (config_dir / "command_used.txt").write_text(
        " ".join(shlex.quote(part) for part in sys.argv) + "\n",
        encoding="utf-8",
    )
    (config_dir / "active_reference_velodyne_path.txt").write_text(str(reference_dir) + "\n", encoding="utf-8")
    (config_dir / "active_processed_velodyne_path.txt").write_text(str(processed_dir) + "\n", encoding="utf-8")

    ref_files = list_bin_files(reference_dir)
    proc_files = list_bin_files(processed_dir)
    proc_lookup = {p.name: p for p in proc_files}
    ref_names = [p.name for p in ref_files]
    proc_names = sorted(proc_lookup)
    if ref_names != proc_names:
        missing = sorted(set(ref_names) - set(proc_names))
        extra = sorted(set(proc_names) - set(ref_names))
        raise RuntimeError(
            "Reference and processed velodyne folders do not contain the same scans. "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    preprocessing_params = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": args.experiment_name,
        "method_name": args.method_name,
        "processing_type": args.processing_type,
        "reference_velodyne": str(reference_dir),
        "processed_velodyne": str(processed_dir),
        "note": "No point cloud generation is performed here. The framework only compares existing folders.",
    }
    (preprocessing_dir / "preprocessing_params.json").write_text(
        json.dumps(preprocessing_params, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (preprocessing_dir / "preprocessing_log.txt").write_text(
        "\n".join(
            [
                f"experiment_name={args.experiment_name}",
                f"method_name={args.method_name}",
                f"processing_type={args.processing_type}",
                f"reference_velodyne={reference_dir}",
                f"processed_velodyne={processed_dir}",
                f"num_scans={len(ref_files)}",
                "transform_note=method-agnostic comparison of already prepared folders",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (preprocessing_dir / "file_list.txt").write_text("\n".join(p.stem for p in ref_files) + "\n", encoding="utf-8")

    sample_rows: List[Dict[str, object]] = []
    for ref_path in ref_files:
        proc_path = proc_lookup[ref_path.name]
        reference = load_points(ref_path)
        processed = load_points(proc_path)
        sample_viz_dir = visualization_dir / "samples" / ref_path.stem
        sample_rows.append(
            evaluate_pair(
                sample_id=ref_path.stem,
                reference=reference,
                processed=processed,
                experiment_name=args.experiment_name,
                method_name=args.method_name,
                reference_path=ref_path,
                processed_path=proc_path,
                transform_stage=args.processing_type,
                intensity_strategy="external_processed_folder",
                processing_type=args.processing_type,
                lidar_range=DEFAULT_KITTI_LIDAR_RANGE,
                cell_size=float(args.bev_cell_size),
                sample_viz_dir=sample_viz_dir,
                seed=args.seed + int(ref_path.stem),
            )
        )

    preprocessing_rows = []
    for row in sample_rows:
        preprocessing_rows.append(
            {
                "sample_id": row.get("sample_id", NA_VALUE),
                "reference_points": row.get("input_points", NA_VALUE),
                "processed_points": row.get("output_points", NA_VALUE),
                "added_points": row.get("added_points", NA_VALUE),
                "removed_points": row.get("removed_points", NA_VALUE),
                "upsampling_ratio": row.get("upsampling_ratio", NA_VALUE),
                "nn_mean_before": row.get("nn_mean_before", NA_VALUE),
                "nn_mean_after": row.get("nn_mean_after", NA_VALUE),
                "bev_density_mean_before": row.get("bev_density_mean_before", NA_VALUE),
                "bev_density_mean_after": row.get("bev_density_mean_after", NA_VALUE),
                "intensity_mean_before": row.get("intensity_mean_before", NA_VALUE),
                "intensity_mean_after": row.get("intensity_mean_after", NA_VALUE),
            }
        )
    with (preprocessing_dir / "point_statistics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "reference_points",
                "processed_points",
                "added_points",
                "removed_points",
                "upsampling_ratio",
                "nn_mean_before",
                "nn_mean_after",
                "bev_density_mean_before",
                "bev_density_mean_after",
                "intensity_mean_before",
                "intensity_mean_after",
            ],
        )
        writer.writeheader()
        writer.writerows(preprocessing_rows)

    detector_metrics = load_detector_metrics(args.detector_results_root, args.method_name, args.experiment_name)
    summary_row = aggregate_rows(
        experiment_name=args.experiment_name,
        method_name=args.method_name,
        processing_type=args.processing_type,
        reference_velodyne=reference_dir,
        processed_velodyne=processed_dir,
        rows=sample_rows,
        detector_metrics=detector_metrics,
        summary_visual_dir=summary_dir,
        summary_note="method-agnostic aggregate comparison",
    )

    # Method-level summary visualizations.
    point_counts = [float(r.get("output_points", 0) or 0) for r in sample_rows]
    nn_before = [float(v) for v in _numeric(r.get("nn_mean_before") for r in sample_rows)]
    nn_after = [float(v) for v in _numeric(r.get("nn_mean_after") for r in sample_rows)]
    density_before = [float(v) for v in _numeric(r.get("bev_density_mean_before") for r in sample_rows)]
    density_after = [float(v) for v in _numeric(r.get("bev_density_mean_after") for r in sample_rows)]
    write_histogram(summary_dir / "point_count_hist.png", point_counts, f"{args.method_name}: output point count", "points")
    write_histogram(summary_dir / "nn_distance_hist.png", nn_before + nn_after, f"{args.method_name}: NN distance", "distance (m)")
    write_histogram(summary_dir / "bev_density_hist.png", density_before + density_after, f"{args.method_name}: BEV density", "occupied cell count")

    # Record outputs.
    save_csv(evaluation_dir / "sample_metrics.csv", sample_rows)
    save_csv(evaluation_dir / "method_summary.csv", [summary_row])
    save_csv(evaluation_dir / "all_methods_comparison.csv", [summary_row])

    detector_csv = evaluation_dir / "parsed_ap_results.csv"
    with detector_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "metric", "easy", "moderate", "hard"])
        for metric in ("bbox", "bev", "3d"):
            writer.writerow(
                [
                    args.experiment_name,
                    metric,
                    summary_row.get(f"detector_{metric}_ap_easy", NA_VALUE),
                    summary_row.get(f"detector_{metric}_ap_moderate", NA_VALUE),
                    summary_row.get(f"detector_{metric}_ap_hard", NA_VALUE),
                ]
            )

    (evaluation_dir / "result_dir.txt").write_text(str(output_root) + "\n", encoding="utf-8")
    (evaluation_dir / "eval_command.txt").write_text(
        " ".join(shlex.quote(part) for part in sys.argv) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Unified Density Evaluation Report",
        "",
        f"- schema_version: {SCHEMA_VERSION}",
        f"- experiment_name: {args.experiment_name}",
        f"- method_name: {args.method_name}",
        f"- processing_type: {args.processing_type}",
        f"- dataset: {args.dataset}",
        f"- split: {args.split}",
        f"- reference_velodyne: {reference_dir}",
        f"- processed_velodyne: {processed_dir}",
        "",
        "## Protocol",
        "",
        "- One schema for every method.",
        "- This run only compares already prepared point-cloud folders.",
        "- Detector AP columns remain in the table and are filled with `N/A` unless a detector result directory is supplied.",
        "",
        "## Summary",
        "",
        f"- input_points: {summary_row.get('input_points', NA_VALUE)}",
        f"- output_points: {summary_row.get('output_points', NA_VALUE)}",
        f"- added_points: {summary_row.get('added_points', NA_VALUE)}",
        f"- removed_points: {summary_row.get('removed_points', NA_VALUE)}",
        f"- upsampling_ratio: {summary_row.get('upsampling_ratio', NA_VALUE)}",
        f"- nn_mean_before: {summary_row.get('nn_mean_before', NA_VALUE)}",
        f"- nn_mean_after: {summary_row.get('nn_mean_after', NA_VALUE)}",
        f"- detector_bbox_ap_moderate: {summary_row.get('detector_bbox_ap_moderate', NA_VALUE)}",
        f"- detector_3d_ap_moderate: {summary_row.get('detector_3d_ap_moderate', NA_VALUE)}",
    ]
    (summary_dir / "comparison_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (notes_dir / "experiment_notes.md").write_text(
        "\n".join(
            [
                f"# {args.experiment_name}",
                "",
                f"- method_name: {args.method_name}",
                f"- processing_type: {args.processing_type}",
                "- The framework does not synthesize point clouds.",
                "- It compares one reference velodyne folder against one processed velodyne folder using a fixed schema.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Git and integrity records for auditability.
    git_commands = {
        "git_status.txt": ["git", "status", "--short"],
        "git_diff_name_only.txt": ["git", "diff", "--name-only"],
        "git_diff_stat.txt": ["git", "diff", "--stat"],
        "git_diff.patch": ["git", "diff"],
        "git_commit_hash.txt": ["git", "rev-parse", "HEAD"],
    }
    for filename, command in git_commands.items():
        result = subprocess.run(command, cwd=str(REPO_ROOT), check=True, capture_output=True, text=True)
        (git_dir / filename).write_text(result.stdout, encoding="utf-8")
    modified_text = (git_dir / "git_diff_name_only.txt").read_text(encoding="utf-8") + "\n-- git status --short --\n" + (git_dir / "git_status.txt").read_text(encoding="utf-8")
    (git_dir / "modified_files.txt").write_text(modified_text, encoding="utf-8")
    integrity_result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "check_pointrcnn_integrity.py")],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    (git_dir / "integrity_check_output.txt").write_text(integrity_result.stdout, encoding="utf-8")
    (git_dir / "integrity_check.txt").write_text(integrity_result.stdout, encoding="utf-8")

    print(f"Wrote unified density evaluation outputs to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
