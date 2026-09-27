#!/usr/bin/env python3
"""QA post-processing variants for full-frame PU-GCN KITTI outputs."""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_BIN = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_original_val" / "000001.bin"
DEFAULT_RAW_100_BIN = PROJECT_ROOT / "results" / "pugcn_full_frame" / "000001" / "patches_100" / "000001_pugcn_full_frame.bin"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "pugcn_full_frame_postprocess" / "000001"
DEFAULT_THRESHOLDS = (0.01, 0.03, 0.05)
KITTI_COLUMNS = 4
DEFAULT_METRIC_RANGE = {
    "x_min": 0.0,
    "x_max": 80.0,
    "y_min": -40.0,
    "y_max": 40.0,
    "z_min": -3.5,
    "z_max": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create QA post-processed variants for PU-GCN full-frame outputs.")
    parser.add_argument("--original_bin", "--original-bin", dest="original_bin", type=Path, default=DEFAULT_ORIGINAL_BIN)
    parser.add_argument("--raw_bin", "--raw-bin", dest="raw_bin", type=Path, default=DEFAULT_RAW_100_BIN)
    parser.add_argument("--output_root", "--output-root", dest="output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frame_id", "--frame-id", dest="frame_id", type=str, default="000001")
    parser.add_argument("--voxel_thresholds", "--voxel-thresholds", dest="voxel_thresholds", nargs="*", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--variants", nargs="*", default=None, help="Optional subset of variant names to generate.")
    parser.add_argument("--local_neighbor_radius", "--local-neighbor-radius", dest="local_neighbor_radius", type=float, default=0.5)
    return parser.parse_args()


def load_kitti_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % KITTI_COLUMNS != 0:
        raise ValueError("Invalid KITTI .bin shape: %s" % path)
    return raw.reshape(-1, KITTI_COLUMNS)


def save_kitti_bin(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points.astype(np.float32).tofile(path)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def frame_bounds(original: np.ndarray) -> Dict[str, float]:
    xyz = original[:, :3]
    return {
        "x_min": 0.0,
        "x_max": float(np.max(xyz[:, 0])),
        "y_min": float(np.min(xyz[:, 1])),
        "y_max": float(np.max(xyz[:, 1])),
        "z_min": float(np.min(xyz[:, 2])),
        "z_max": float(np.max(xyz[:, 2])),
    }


def metric_range_mask(points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3]
    return (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= DEFAULT_METRIC_RANGE["x_min"])
        & (xyz[:, 0] <= DEFAULT_METRIC_RANGE["x_max"])
        & (xyz[:, 1] >= DEFAULT_METRIC_RANGE["y_min"])
        & (xyz[:, 1] <= DEFAULT_METRIC_RANGE["y_max"])
        & (xyz[:, 2] >= DEFAULT_METRIC_RANGE["z_min"])
        & (xyz[:, 2] <= DEFAULT_METRIC_RANGE["z_max"])
    )


def exact_dedup(points: np.ndarray) -> Tuple[np.ndarray, int]:
    if len(points) == 0:
        return points, 0
    _, keep_idx = np.unique(points[:, :3], axis=0, return_index=True)
    keep_idx.sort()
    return points[keep_idx], int(len(points) - len(keep_idx))


def voxel_dedup(points: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, int]:
    if len(points) == 0 or voxel_size <= 0:
        return points, 0
    coords = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, keep_idx = np.unique(coords, axis=0, return_index=True)
    keep_idx.sort()
    return points[keep_idx], int(len(points) - len(keep_idx))


def invalid_point_count(points: np.ndarray) -> int:
    if len(points) == 0:
        return 0
    return int((~np.isfinite(points[:, :3]).all(axis=1)).sum())


def filter_invalid(points: np.ndarray) -> Tuple[np.ndarray, int]:
    if len(points) == 0:
        return points, 0
    mask = np.isfinite(points[:, :3]).all(axis=1)
    return points[mask], int((~mask).sum())


def nearest_distances(query_xyz: np.ndarray, ref_xyz: np.ndarray) -> np.ndarray:
    if len(query_xyz) == 0 or len(ref_xyz) == 0:
        return np.zeros((0,), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(ref_xyz).query(query_xyz, k=1, workers=-1)
    except TypeError:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(ref_xyz).query(query_xyz, k=1)
    except Exception:
        distances = []
        for point in query_xyz:
            diff = ref_xyz - point
            distances.append(float(np.sqrt(np.min(np.sum(diff * diff, axis=1)))))
        distances = np.asarray(distances, dtype=np.float32)
    return np.asarray(distances, dtype=np.float32)


def self_nn_distances(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0,), dtype=np.float32)
    xyz = points[:, :3]
    try:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(xyz).query(xyz, k=2, workers=-1)
    except TypeError:
        from scipy.spatial import cKDTree

        distances, _ = cKDTree(xyz).query(xyz, k=2)
    except Exception:
        nearest = []
        for i, point in enumerate(xyz):
            diff = xyz - point
            dist = np.sqrt(np.sum(diff * diff, axis=1))
            dist[i] = np.inf
            nearest.append(float(np.min(dist)))
        return np.asarray(nearest, dtype=np.float32)
    return np.asarray(distances[:, 1], dtype=np.float32)


def local_neighbor_stats(points: np.ndarray, radius: float) -> Dict[str, object]:
    if len(points) == 0:
        return {
            "local_neighbor_radius": radius,
            "local_neighbor_count_mean": "N/A",
            "local_neighbor_count_median": "N/A",
            "local_neighbor_count_p95": "N/A",
        }
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


def duplicate_counts(points: np.ndarray, thresholds: Sequence[float]) -> Dict[str, object]:
    stats = {}
    if len(points) == 0:
        stats["exact_duplicate_points"] = 0
        stats["exact_duplicate_ratio"] = 0.0
        for threshold in thresholds:
            suffix = threshold_label(threshold)
            stats["duplicate_voxel_points_%s" % suffix] = 0
            stats["duplicate_voxel_ratio_%s" % suffix] = 0.0
            stats["near_duplicate_points_%s" % suffix] = 0
            stats["near_duplicate_ratio_%s" % suffix] = 0.0
        return stats

    _, exact_counts = np.unique(points[:, :3], axis=0, return_counts=True)
    exact_duplicate_points = int(np.sum(exact_counts[exact_counts > 1] - 1))
    stats["exact_duplicate_points"] = exact_duplicate_points
    stats["exact_duplicate_ratio"] = float(exact_duplicate_points / len(points))

    nn = self_nn_distances(points)
    for threshold in thresholds:
        suffix = threshold_label(threshold)
        if threshold > 0:
            coords = np.floor(points[:, :3] / threshold).astype(np.int64)
            _, voxel_counts = np.unique(coords, axis=0, return_counts=True)
            voxel_points = int(np.sum(voxel_counts[voxel_counts > 1] - 1))
        else:
            voxel_points = 0
        near_points = int(np.sum(nn <= threshold)) if len(nn) else 0
        stats["duplicate_voxel_points_%s" % suffix] = voxel_points
        stats["duplicate_voxel_ratio_%s" % suffix] = float(voxel_points / len(points))
        stats["near_duplicate_points_%s" % suffix] = near_points
        stats["near_duplicate_ratio_%s" % suffix] = float(near_points / len(points))
    return stats


def threshold_label(value: float) -> str:
    return "%03d" % int(round(value * 100))


def output_bounds(points: np.ndarray, prefix: str) -> Dict[str, object]:
    if len(points) == 0:
        return {
            "%s_x_min" % prefix: "N/A",
            "%s_x_max" % prefix: "N/A",
            "%s_y_min" % prefix: "N/A",
            "%s_y_max" % prefix: "N/A",
            "%s_z_min" % prefix: "N/A",
            "%s_z_max" % prefix: "N/A",
        }
    xyz = points[:, :3]
    return {
        "%s_x_min" % prefix: float(np.min(xyz[:, 0])),
        "%s_x_max" % prefix: float(np.max(xyz[:, 0])),
        "%s_y_min" % prefix: float(np.min(xyz[:, 1])),
        "%s_y_max" % prefix: float(np.max(xyz[:, 1])),
        "%s_z_min" % prefix: float(np.min(xyz[:, 2])),
        "%s_z_max" % prefix: float(np.max(xyz[:, 2])),
    }


def range_report(points: np.ndarray, orig_bounds: Dict[str, float]) -> Dict[str, object]:
    if len(points) == 0:
        return {
            "metric_out_of_range_points": 0,
            "metric_out_of_range_ratio": 0.0,
            "frame_bounds_out_of_range_points": 0,
            "frame_bounds_out_of_range_ratio": 0.0,
            "x_negative_points": 0,
            "x_gt_original_max_points": 0,
            "y_outside_original_bounds_points": 0,
            "z_outside_original_bounds_points": 0,
        }
    xyz = points[:, :3]
    metric_mask = metric_range_mask(points)
    x_negative = int(np.sum(xyz[:, 0] < 0))
    x_gt_original_max = int(np.sum(xyz[:, 0] > orig_bounds["x_max"]))
    y_outside = int(np.sum((xyz[:, 1] < orig_bounds["y_min"]) | (xyz[:, 1] > orig_bounds["y_max"])))
    z_outside = int(np.sum((xyz[:, 2] < orig_bounds["z_min"]) | (xyz[:, 2] > orig_bounds["z_max"])))
    frame_mask = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= 0.0)
        & (xyz[:, 0] <= orig_bounds["x_max"])
        & (xyz[:, 1] >= orig_bounds["y_min"])
        & (xyz[:, 1] <= orig_bounds["y_max"])
        & (xyz[:, 2] >= orig_bounds["z_min"])
        & (xyz[:, 2] <= orig_bounds["z_max"])
    )
    metric_bad = int(np.sum(~metric_mask))
    frame_bad = int(np.sum(~frame_mask))
    return {
        "metric_out_of_range_points": metric_bad,
        "metric_out_of_range_ratio": float(metric_bad / len(points)),
        "frame_bounds_out_of_range_points": frame_bad,
        "frame_bounds_out_of_range_ratio": float(frame_bad / len(points)),
        "x_negative_points": x_negative,
        "x_gt_original_max_points": x_gt_original_max,
        "y_outside_original_bounds_points": y_outside,
        "z_outside_original_bounds_points": z_outside,
    }


def nn_summary(distances: np.ndarray, prefix: str) -> Dict[str, object]:
    if len(distances) == 0:
        return {
            "%s_nn_min" % prefix: "N/A",
            "%s_nn_mean" % prefix: "N/A",
            "%s_nn_median" % prefix: "N/A",
            "%s_nn_max" % prefix: "N/A",
        }
    return {
        "%s_nn_min" % prefix: float(np.min(distances)),
        "%s_nn_mean" % prefix: float(np.mean(distances)),
        "%s_nn_median" % prefix: float(np.median(distances)),
        "%s_nn_max" % prefix: float(np.max(distances)),
    }


def transform_points(points: np.ndarray, filter_range: bool, dedup_mode: str, dedup_value: float) -> Tuple[np.ndarray, Dict[str, int]]:
    working = points.copy()
    stats = {
        "invalid_points_removed": 0,
        "filtered_out_of_range": 0,
        "exact_duplicates_removed": 0,
        "voxel_duplicates_removed": 0,
    }
    working, invalid_removed = filter_invalid(working)
    stats["invalid_points_removed"] = invalid_removed
    if filter_range and len(working):
        mask = metric_range_mask(working)
        stats["filtered_out_of_range"] = int(np.sum(~mask))
        working = working[mask]
    if dedup_mode == "exact":
        working, removed = exact_dedup(working)
        stats["exact_duplicates_removed"] = removed
    elif dedup_mode == "voxel":
        working, removed = voxel_dedup(working, dedup_value)
        stats["voxel_duplicates_removed"] = removed
    return working.astype(np.float32), stats


def variant_specs() -> List[Dict[str, object]]:
    return [
        {"name": "raw_100", "filter_range": False, "dedup_mode": "none", "dedup_value": 0.0},
        {"name": "filter_range", "filter_range": True, "dedup_mode": "none", "dedup_value": 0.0},
        {"name": "dedup_exact", "filter_range": False, "dedup_mode": "exact", "dedup_value": 0.0},
        {"name": "dedup_voxel_001", "filter_range": False, "dedup_mode": "voxel", "dedup_value": 0.01},
        {"name": "dedup_voxel_003", "filter_range": False, "dedup_mode": "voxel", "dedup_value": 0.03},
        {"name": "dedup_voxel_005", "filter_range": False, "dedup_mode": "voxel", "dedup_value": 0.05},
        {"name": "filter_range_dedup_exact", "filter_range": True, "dedup_mode": "exact", "dedup_value": 0.0},
        {"name": "filter_range_dedup_voxel_001", "filter_range": True, "dedup_mode": "voxel", "dedup_value": 0.01},
        {"name": "filter_range_dedup_voxel_003", "filter_range": True, "dedup_mode": "voxel", "dedup_value": 0.03},
        {"name": "filter_range_dedup_voxel_005", "filter_range": True, "dedup_mode": "voxel", "dedup_value": 0.05},
    ]


def select_variant_specs(names: Optional[Sequence[str]]) -> List[Dict[str, object]]:
    specs = variant_specs()
    if not names:
        return specs
    wanted = list(names)
    lookup = {str(spec["name"]): spec for spec in specs}
    missing = [name for name in wanted if name not in lookup]
    if missing:
        raise ValueError("Unknown variant(s): %s" % ", ".join(missing))
    return [lookup[name] for name in wanted]


def processed_bin_name(frame_id: str) -> str:
    return "%s_pugcn_full_frame.bin" % frame_id


def collect_stats(
    variant_name: str,
    points: np.ndarray,
    original: np.ndarray,
    orig_bounds: Dict[str, float],
    runtime_sec: float,
    transform_meta: Dict[str, object],
    thresholds: Sequence[float],
    output_bin: Path,
) -> Dict[str, object]:
    stats = {
        "variant_name": variant_name,
        "output_bin": str(output_bin.resolve()),
        "runtime_sec": float(runtime_sec),
        "input_points": int(len(original)),
        "output_points": int(len(points)),
        "effective_upsampling_ratio": float(len(points) / len(original)) if len(original) else 0.0,
        "invalid_xyz_points": invalid_point_count(points),
        "filter_range_enabled": bool(transform_meta["filter_range"]),
        "dedup_mode": transform_meta["dedup_mode"],
        "dedup_value_m": float(transform_meta["dedup_value"]),
        "postprocess_invalid_points_removed": int(transform_meta["postprocess_invalid_points_removed"]),
        "postprocess_filtered_out_of_range": int(transform_meta["postprocess_filtered_out_of_range"]),
        "postprocess_exact_duplicates_removed": int(transform_meta["postprocess_exact_duplicates_removed"]),
        "postprocess_voxel_duplicates_removed": int(transform_meta["postprocess_voxel_duplicates_removed"]),
    }
    stats.update(range_report(points, orig_bounds))
    stats.update(output_bounds(original, "input"))
    stats.update(output_bounds(points, "output"))
    stats.update(duplicate_counts(points, thresholds))
    stats.update(nn_summary(nearest_distances(points[:, :3], original[:, :3]), "output_to_input"))
    stats.update(nn_summary(self_nn_distances(points), "output_self"))
    stats.update(local_neighbor_stats(points, transform_meta["local_neighbor_radius"]))
    return stats


def markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[Tuple[str, str]]) -> List[str]:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append("{:.6f}".format(value).rstrip("0").rstrip("."))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def build_summary(
    output_root: Path,
    frame_id: str,
    rows: Sequence[Dict[str, object]],
    recommended_variant: Optional[str] = None,
) -> None:
    csv_fields = []
    for row in rows:
        for key in row.keys():
            if key not in csv_fields:
                csv_fields.append(key)
    write_csv(output_root / "postprocess_comparison_summary.csv", rows, csv_fields)

    summary_columns = [
        ("variant_name", "Variant"),
        ("output_points", "Output points"),
        ("effective_upsampling_ratio", "Eff. ratio"),
        ("metric_out_of_range_points", "Metric out-of-range"),
        ("exact_duplicate_points", "Exact dup"),
        ("near_duplicate_points_001", "Near dup 0.01"),
        ("near_duplicate_points_003", "Near dup 0.03"),
        ("near_duplicate_points_005", "Near dup 0.05"),
        ("runtime_sec", "Runtime sec"),
        ("visual_quality_note", "Visual quality"),
    ]
    lines = [
        "# PU-GCN Full-Frame Post-Processing Summary: KITTI %s" % frame_id,
        "",
        "This table compares the raw 20/50/100 full-frame reconstructions with QA post-processed variants derived from the raw 100-patch candidate.",
        "",
    ]
    lines.extend(markdown_table(rows, summary_columns))
    if recommended_variant:
        lines.extend(
            [
                "",
                "Recommended detector-facing candidate: `%s`." % recommended_variant,
            ]
        )
    (output_root / "postprocess_comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    original = load_kitti_bin(args.original_bin)
    raw = load_kitti_bin(args.raw_bin)
    orig_bounds = frame_bounds(original)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    thresholds = tuple(sorted(set(float(v) for v in args.voxel_thresholds)))
    variant_rows = []

    selected_specs = select_variant_specs(args.variants)
    for spec in selected_specs:
        start = time.time()
        processed, transform_stats = transform_points(raw, bool(spec["filter_range"]), str(spec["dedup_mode"]), float(spec["dedup_value"]))
        runtime_sec = time.time() - start
        variant_dir = output_root / str(spec["name"])
        output_bin = variant_dir / processed_bin_name(args.frame_id)
        save_kitti_bin(output_bin, processed)
        row = collect_stats(
            variant_name=str(spec["name"]),
            points=processed,
            original=original,
            orig_bounds=orig_bounds,
            runtime_sec=runtime_sec,
            transform_meta={
                "filter_range": bool(spec["filter_range"]),
                "dedup_mode": str(spec["dedup_mode"]),
                "dedup_value": float(spec["dedup_value"]),
                "postprocess_invalid_points_removed": transform_stats["invalid_points_removed"],
                "postprocess_filtered_out_of_range": transform_stats["filtered_out_of_range"],
                "postprocess_exact_duplicates_removed": transform_stats["exact_duplicates_removed"],
                "postprocess_voxel_duplicates_removed": transform_stats["voxel_duplicates_removed"],
                "local_neighbor_radius": float(args.local_neighbor_radius),
            },
            thresholds=thresholds,
            output_bin=output_bin,
        )
        save_kitti_bin(output_bin, processed)
        write_json(variant_dir / "postprocess_manifest.json", row)
        variant_rows.append(row)

    if variant_rows:
        write_csv(output_root / "postprocess_variant_metrics.csv", variant_rows, list(variant_rows[0].keys()))
    print("Wrote PU-GCN post-process variants to:", output_root)


if __name__ == "__main__":
    main()
