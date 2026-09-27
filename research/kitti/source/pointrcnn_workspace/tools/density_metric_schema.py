#!/usr/bin/env python3
"""Canonical metric schema for density-baseline experiments.

This module defines one reusable schema for:
  - point-cloud only experiments,
  - detector-integrated experiments,
  - future ablations / sensitivity studies,
  - and cross-dataset runs.

The guiding rule is simple: keep the columns stable and fill unavailable
entries with ``N/A`` rather than changing the table shape per experiment.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List

SCHEMA_VERSION = "density_metric_schema_v1"
NA_VALUE = "N/A"

# Standard KITTI LiDAR bounds used by the density evaluation framework.
# These are intentionally configurable in the runner, but the defaults keep
# the protocol stable across point-cloud methods.
DEFAULT_KITTI_LIDAR_RANGE = {
    "x_min": 0.0,
    "x_max": 80.0,
    "y_min": -40.0,
    "y_max": 40.0,
    "z_min": -3.5,
    "z_max": 2.0,
}

DEFAULT_BEV_CELL_SIZE = 0.5
DEFAULT_MAX_VISUALIZED_SAMPLES = 12

SAMPLE_LEVEL_COLUMNS: List[str] = [
    "schema_version",
    "scope",
    "dataset",
    "split",
    "experiment_name",
    "method_name",
    "processing_type",
    "sample_id",
    "reference_velodyne",
    "processed_velodyne",
    "source_path",
    "input_path",
    "output_path",
    "transform_stage",
    "intensity_strategy",
    "format_ok",
    "coordinate_frame_preserved",
    "within_kitti_range_before",
    "within_kitti_range_after",
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
    "x_min_before",
    "x_min_after",
    "x_max_before",
    "x_max_after",
    "y_min_before",
    "y_min_after",
    "y_max_before",
    "y_max_after",
    "z_min_before",
    "z_min_after",
    "z_max_before",
    "z_max_after",
    "intensity_min_before",
    "intensity_min_after",
    "intensity_max_before",
    "intensity_max_after",
    "intensity_mean_before",
    "intensity_mean_after",
    "intensity_std_before",
    "intensity_std_after",
    "bev_before_png",
    "bev_after_png",
    "bev_overlay_png",
    "bev_zoom_png",
    "cloud_before_png",
    "cloud_after_png",
    "cloud_overlay_png",
    "summary_note",
]

METHOD_LEVEL_COLUMNS: List[str] = [
    "schema_version",
    "scope",
    "dataset",
    "split",
    "experiment_name",
    "method_name",
    "processing_type",
    "sample_id",
    "reference_velodyne",
    "processed_velodyne",
    "source_path",
    "input_path",
    "output_path",
    "transform_stage",
    "intensity_strategy",
    "format_ok",
    "coordinate_frame_preserved",
    "within_kitti_range_before",
    "within_kitti_range_after",
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
    "x_min_before",
    "x_min_after",
    "x_max_before",
    "x_max_after",
    "y_min_before",
    "y_min_after",
    "y_max_before",
    "y_max_after",
    "z_min_before",
    "z_min_after",
    "z_max_before",
    "z_max_after",
    "intensity_min_before",
    "intensity_min_after",
    "intensity_max_before",
    "intensity_max_after",
    "intensity_mean_before",
    "intensity_mean_after",
    "intensity_std_before",
    "intensity_std_after",
    "bev_before_png",
    "bev_after_png",
    "bev_overlay_png",
    "bev_zoom_png",
    "cloud_before_png",
    "cloud_after_png",
    "cloud_overlay_png",
    "summary_note",
    "detector_result_source",
    "detector_bbox_ap_easy",
    "detector_bbox_ap_moderate",
    "detector_bbox_ap_hard",
    "detector_bev_ap_easy",
    "detector_bev_ap_moderate",
    "detector_bev_ap_hard",
    "detector_3d_ap_easy",
    "detector_3d_ap_moderate",
    "detector_3d_ap_hard",
    "detector_precision_easy",
    "detector_precision_moderate",
    "detector_precision_hard",
    "detector_recall_easy",
    "detector_recall_moderate",
    "detector_recall_hard",
]

ALL_COLUMNS: List[str] = METHOD_LEVEL_COLUMNS

VISUALIZATION_FILENAMES = OrderedDict(
    [
        ("bev_before_png", "bev_before.png"),
        ("bev_after_png", "bev_after.png"),
        ("bev_overlay_png", "bev_overlay.png"),
        ("bev_zoom_png", "bev_zoom.png"),
        ("cloud_before_png", "cloud_before.png"),
        ("cloud_after_png", "cloud_after.png"),
        ("cloud_overlay_png", "cloud_overlay.png"),
    ]
)

def blank_record() -> Dict[str, str]:
    """Return a canonical empty record filled with N/A."""

    return {column: NA_VALUE for column in ALL_COLUMNS}


def csv_fieldnames() -> List[str]:
    return list(ALL_COLUMNS)
