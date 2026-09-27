#!/usr/bin/env python3
"""Measure point-count and CenterPoint-voxel effects for all available tracks."""

from __future__ import annotations

import csv
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
OPENPCDET = REPO / "external/OpenPCDet"
sys.path.insert(0, str(OPENPCDET))

from pcdet.datasets.kitti.kitti_dataset import KittiDataset  # noqa: E402
from pcdet.utils.calibration_kitti import Calibration  # noqa: E402


RESULT = REPO / "results/centerpoint_input_mechanism_analysis_20260729"
VAL_FILE = OPENPCDET / "data/kitti/ImageSets/val.txt"
VAL_INFO = OPENPCDET / "data/kitti/kitti_infos_val.pkl"
ORIGINAL = REPO / "data/KITTI/object/training/velodyne_original"
DOWNSAMPLED = (
    REPO
    / "results/kitti_unified_x4_current_methods_no_detector"
    / "downsampled_x4/velodyne_downsampled_x4_val"
)
E1 = (
    REPO
    / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
    / "inputs/e1_default_16384"
)
CALIB = REPO / "data/KITTI/object/training/calib"

POINT_RANGE = np.array([0.0, -40.0, -3.0, 70.4, 40.0, 1.0], dtype=np.float32)
VOXEL_SIZE = np.array([0.05, 0.05, 0.1], dtype=np.float32)
REFERENCE_VOXEL_SIZE = np.array([0.2, 0.2, 0.2], dtype=np.float32)
MAX_POINTS_PER_VOXEL = 5
MAX_VOXELS = 40_000
SAMPLE_FRAMES = 32


@dataclass(frozen=True)
class Source:
    name: str
    line: str
    method: str
    track: str
    path: Path
    line_input: Path
    contains_observed_rows: bool


SOURCES = (
    Source("original_baseline", "baseline_a", "none", "baseline", ORIGINAL, ORIGINAL, False),
    Source(
        "downsampled_x4_baseline",
        "baseline_b",
        "none",
        "baseline",
        DOWNSAMPLED,
        DOWNSAMPLED,
        False,
    ),
    *tuple(
        Source(
            f"original_x4_{method}",
            "line_a",
            method,
            "exact4n_e1",
            E1 / f"original_x4_{method}",
            ORIGINAL,
            True,
        )
        for method in ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
    ),
    *tuple(
        Source(
            f"downsampled_x4_{method}",
            "line_b",
            method,
            "exact4n_e1",
            E1 / f"downsampled_x4_{method}",
            DOWNSAMPLED,
            True,
        )
        for method in ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
    ),
    Source(
        "tulip_original_native",
        "line_a_extended",
        "tulip",
        "native_extended",
        REPO / "data/KITTI/object/training/tulip_original_up_bin",
        ORIGINAL,
        False,
    ),
    Source(
        "tulip_downsampled_native",
        "line_b_extended",
        "tulip",
        "native_extended",
        REPO / "data/KITTI/object/training/tulip_downsampled_up_bin",
        DOWNSAMPLED,
        False,
    ),
)


def read_points(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"malformed KITTI point file: {path}")
    return values.reshape(-1, 4)


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key_for(points: np.ndarray, voxel_size: np.ndarray) -> np.ndarray:
    coords = np.floor((points[:, :3] - POINT_RANGE[:3]) / voxel_size).astype(np.int64)
    grid = np.ceil((POINT_RANGE[3:] - POINT_RANGE[:3]) / voxel_size).astype(np.int64)
    return coords[:, 0] + grid[0] * (coords[:, 1] + grid[1] * coords[:, 2])


def fov_and_range(
    points: np.ndarray, calibration: Calibration, image_shape: np.ndarray
) -> np.ndarray:
    rect = calibration.lidar_to_rect(points[:, :3])
    fov = KittiDataset.get_fov_flag(rect, image_shape, calibration)
    selected = points[fov]
    xyz = selected[:, :3]
    inside = np.all(xyz >= POINT_RANGE[:3], axis=1) & np.all(
        xyz < POINT_RANGE[3:], axis=1
    )
    return selected[inside]


def split_observed_generated(
    cloud: np.ndarray, line_input: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Split by exact XYZI row membership; E1 does not promise row ordering."""
    row_dtype = np.dtype((np.void, cloud.dtype.itemsize * cloud.shape[1]))
    cloud_rows = np.ascontiguousarray(cloud).view(row_dtype).reshape(-1)
    input_rows = np.ascontiguousarray(line_input).view(row_dtype).reshape(-1)
    observed_mask = np.isin(cloud_rows, input_rows)
    input_rows_present = bool(np.all(np.isin(input_rows, cloud_rows)))
    return cloud[observed_mask], cloud[~observed_mask], input_rows_present


def point_count_rows(ids: list[str]) -> list[dict[str, object]]:
    rows = []
    for source in SOURCES:
        counts = np.array(
            [(source.path / f"{frame}.bin").stat().st_size // 16 for frame in ids],
            dtype=np.int64,
        )
        input_counts = np.array(
            [(source.line_input / f"{frame}.bin").stat().st_size // 16 for frame in ids],
            dtype=np.int64,
        )
        stats = percentile_summary(counts)
        ratios = counts / input_counts
        rows.append(
            {
                "variant": source.name,
                "line": source.line,
                "method": source.method,
                "track": source.track,
                "frames": len(ids),
                "points_min": int(stats["min"]),
                "points_p05": round(stats["p05"], 3),
                "points_median": round(stats["median"], 3),
                "points_mean": round(stats["mean"], 3),
                "points_p95": round(stats["p95"], 3),
                "points_max": int(stats["max"]),
                "ratio_to_line_input_median": round(float(np.median(ratios)), 6),
                "ratio_to_line_input_mean": round(float(ratios.mean()), 6),
                "exact_4x_frames": int(np.count_nonzero(counts == 4 * input_counts)),
            }
        )
    return rows


def sample_voxel_rows(
    ids: list[str], image_shapes: dict[str, np.ndarray]
) -> list[dict[str, object]]:
    chosen = [ids[i] for i in np.linspace(0, len(ids) - 1, SAMPLE_FRAMES, dtype=int)]
    rows: list[dict[str, object]] = []
    for frame in chosen:
        calibration = Calibration(CALIB / f"{frame}.txt")
        image_shape = image_shapes[frame]
        reference = fov_and_range(
            read_points(ORIGINAL / f"{frame}.bin"), calibration, image_shape
        )
        reference_keys = np.unique(key_for(reference, REFERENCE_VOXEL_SIZE))
        for source in SOURCES:
            cloud = read_points(source.path / f"{frame}.bin")
            input_rows_present: object = ""
            observed = np.empty((0, 4), dtype=np.float32)
            generated = np.empty((0, 4), dtype=np.float32)
            if source.contains_observed_rows:
                line_input = read_points(source.line_input / f"{frame}.bin")
                observed_all, generated_all, input_rows_present = (
                    split_observed_generated(cloud, line_input)
                )
                observed = fov_and_range(
                    observed_all, calibration, image_shape
                )
                generated = fov_and_range(
                    generated_all, calibration, image_shape
                )
            selected = fov_and_range(cloud, calibration, image_shape)
            voxel_keys, voxel_counts = np.unique(
                key_for(selected, VOXEL_SIZE), return_counts=True
            )
            observed_keys = (
                np.unique(key_for(observed, VOXEL_SIZE))
                if len(observed)
                else np.empty(0, dtype=np.int64)
            )
            generated_keys = (
                np.unique(key_for(generated, VOXEL_SIZE))
                if len(generated)
                else np.empty(0, dtype=np.int64)
            )
            shared_keys = np.intersect1d(
                observed_keys, generated_keys, assume_unique=True
            )
            generated_only = np.setdiff1d(
                generated_keys, observed_keys, assume_unique=True
            )
            cloud_reference_keys = np.unique(
                key_for(selected, REFERENCE_VOXEL_SIZE)
            )
            reference_overlap = np.intersect1d(
                cloud_reference_keys, reference_keys, assume_unique=True
            )
            generated_reference_keys = (
                np.unique(key_for(generated, REFERENCE_VOXEL_SIZE))
                if len(generated)
                else np.empty(0, dtype=np.int64)
            )
            generated_reference_overlap = np.intersect1d(
                generated_reference_keys, reference_keys, assume_unique=True
            )
            retained_slots = int(np.minimum(voxel_counts, MAX_POINTS_PER_VOXEL).sum())
            rows.append(
                {
                    "frame": frame,
                    "variant": source.name,
                    "line": source.line,
                    "method": source.method,
                    "track": source.track,
                    "raw_points": len(cloud),
                    "centerpoint_fov_range_points": len(selected),
                    "centerpoint_pre_cap_voxels": len(voxel_keys),
                    "centerpoint_post_cap_voxels": min(len(voxel_keys), MAX_VOXELS),
                    "voxel_cap_hit": len(voxel_keys) > MAX_VOXELS,
                    "voxel_cap_drop_fraction": max(
                        0.0, (len(voxel_keys) - MAX_VOXELS) / max(len(voxel_keys), 1)
                    ),
                    "points_per_voxel_mean": len(selected) / max(len(voxel_keys), 1),
                    "points_discarded_by_max5_fraction": (
                        len(selected) - retained_slots
                    )
                    / max(len(selected), 1),
                    "observed_input_rows_present": input_rows_present,
                    "observed_matched_points": len(observed),
                    "generated_unmatched_points": len(generated),
                    "observed_voxels": len(observed_keys),
                    "generated_voxels": len(generated_keys),
                    "generated_only_voxels": len(generated_only),
                    "mixed_observed_voxel_fraction": len(shared_keys)
                    / max(len(observed_keys), 1),
                    "reference_0p2_voxel_recall": len(reference_overlap)
                    / max(len(reference_keys), 1),
                    "cloud_0p2_extra_voxel_fraction": (
                        len(cloud_reference_keys) - len(reference_overlap)
                    )
                    / max(len(cloud_reference_keys), 1),
                    "generated_0p2_voxel_precision": (
                        len(generated_reference_overlap)
                        / max(len(generated_reference_keys), 1)
                        if len(generated_reference_keys)
                        else ""
                    ),
                }
            )
    return rows


def aggregate_voxel_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for source in SOURCES:
        chosen = [row for row in rows if row["variant"] == source.name]
        output.append(
            {
                "variant": source.name,
                "line": source.line,
                "method": source.method,
                "track": source.track,
                "sample_frames": len(chosen),
                "raw_points_median": round(
                    float(np.median([row["raw_points"] for row in chosen])), 3
                ),
                "fov_range_points_median": round(
                    float(
                        np.median(
                            [row["centerpoint_fov_range_points"] for row in chosen]
                        )
                    ),
                    3,
                ),
                "pre_cap_voxels_median": round(
                    float(
                        np.median(
                            [row["centerpoint_pre_cap_voxels"] for row in chosen]
                        )
                    ),
                    3,
                ),
                "cap_hit_frames": sum(bool(row["voxel_cap_hit"]) for row in chosen),
                "cap_drop_fraction_median": round(
                    float(
                        np.median(
                            [row["voxel_cap_drop_fraction"] for row in chosen]
                        )
                    ),
                    6,
                ),
                "points_per_voxel_mean_median": round(
                    float(
                        np.median([row["points_per_voxel_mean"] for row in chosen])
                    ),
                    6,
                ),
                "points_discarded_by_max5_fraction_median": round(
                    float(
                        np.median(
                            [
                                row["points_discarded_by_max5_fraction"]
                                for row in chosen
                            ]
                        )
                    ),
                    6,
                ),
                "mixed_observed_voxel_fraction_median": round(
                    float(
                        np.median(
                            [
                                row["mixed_observed_voxel_fraction"]
                                for row in chosen
                            ]
                        )
                    ),
                    6,
                ),
                "reference_0p2_voxel_recall_median": round(
                    float(
                        np.median(
                            [row["reference_0p2_voxel_recall"] for row in chosen]
                        )
                    ),
                    6,
                ),
                "cloud_0p2_extra_voxel_fraction_median": round(
                    float(
                        np.median(
                            [
                                row["cloud_0p2_extra_voxel_fraction"]
                                for row in chosen
                            ]
                        )
                    ),
                    6,
                ),
                "generated_0p2_voxel_precision_median": (
                    round(
                        float(
                            np.median(
                                [
                                    row["generated_0p2_voxel_precision"]
                                    for row in chosen
                                    if row["generated_0p2_voxel_precision"] != ""
                                ]
                            )
                        ),
                        6,
                    )
                    if source.contains_observed_rows
                    else ""
                ),
                "observed_input_rows_present_frames": (
                    sum(row["observed_input_rows_present"] is True for row in chosen)
                    if source.contains_observed_rows
                    else ""
                ),
            }
        )
    return output


def parse_detector_behavior() -> list[dict[str, object]]:
    roots = {
        "exact4n_e1": REPO / "results/centerpoint_exact4n_e1_20260729",
        "native_extended": REPO
        / "results/centerpoint_tulip_native_extended_20260729",
    }
    rows = []
    for source in SOURCES:
        root = roots.get(source.track)
        if root is None and source.track == "baseline":
            root = roots["exact4n_e1"]
        if root is None:
            continue
        run = root / source.name / "full"
        logs = sorted((run / "openpcdet_output").rglob("log_eval_*.txt"))
        result_files = sorted((run / "openpcdet_output").rglob("result.pkl"))
        stdout = run / "runner_stdout.log"
        if not logs or not result_files or not stdout.exists():
            continue
        log_text = logs[-1].read_text(errors="replace")
        stdout_text = stdout.read_text(errors="replace")
        behavior: dict[str, object] = {
            "variant": source.name,
            "line": source.line,
            "method": source.method,
            "track": source.track,
        }
        for threshold in ("0.3", "0.5", "0.7"):
            matches = re.findall(
                rf"recall_rcnn_{re.escape(threshold)}:\s*([0-9.]+)", stdout_text
            )
            behavior[f"recall_rcnn_{threshold}"] = (
                float(matches[-1]) if matches else ""
            )
        predicted = re.findall(
            r"Average predicted number of objects\([^)]+\):\s*([0-9.]+)",
            stdout_text,
        )
        behavior["predicted_objects_per_frame"] = (
            float(predicted[-1]) if predicted else ""
        )
        for class_name in ("Car", "Pedestrian", "Cyclist"):
            iou = (
                "0.70, 0.70, 0.70"
                if class_name == "Car"
                else "0.50, 0.50, 0.50"
            )
            matches = re.findall(
                rf"{class_name} AP_R40@{iou}:\s*"
                rf"bbox AP:[^\n]+\s*bev\s+AP:[^\n]+\s*"
                rf"3d\s+AP:([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)",
                log_text,
            )
            easy, moderate, hard = matches[-1]
            prefix = class_name.lower()
            behavior[f"{prefix}_3d_ap_r40_easy"] = float(easy)
            behavior[f"{prefix}_3d_ap_r40_moderate"] = float(moderate)
            behavior[f"{prefix}_3d_ap_r40_hard"] = float(hard)
        with result_files[-1].open("rb") as handle:
            predictions = pickle.load(handle)
        for class_name in ("Car", "Pedestrian", "Cyclist"):
            scores = np.concatenate(
                [
                    np.asarray(item["score"])[np.asarray(item["name"]) == class_name]
                    for item in predictions
                ]
            )
            prefix = class_name.lower()
            behavior[f"{prefix}_prediction_count"] = len(scores)
            behavior[f"{prefix}_score_median"] = (
                round(float(np.median(scores)), 6) if len(scores) else ""
            )
            behavior[f"{prefix}_score_ge_0p5_count"] = int(
                np.count_nonzero(scores >= 0.5)
            )
        rows.append(behavior)
    return rows


def main() -> int:
    ids = VAL_FILE.read_text().split()
    with VAL_INFO.open("rb") as handle:
        infos = pickle.load(handle)
    image_shapes = {
        str(info["point_cloud"]["lidar_idx"]): np.asarray(
            info["image"]["image_shape"]
        )
        for info in infos
    }
    count_rows = point_count_rows(ids)
    write_csv(RESULT / "point_count_summary.csv", count_rows)
    voxel_rows = sample_voxel_rows(ids, image_shapes)
    write_csv(RESULT / "centerpoint_voxel_sample_per_frame.csv", voxel_rows)
    write_csv(
        RESULT / "centerpoint_voxel_summary.csv", aggregate_voxel_rows(voxel_rows)
    )
    behavior = parse_detector_behavior()
    if behavior:
        write_csv(RESULT / "detector_behavior_summary.csv", behavior)
    print(f"Wrote analysis to {RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
