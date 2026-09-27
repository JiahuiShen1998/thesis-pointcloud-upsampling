#!/usr/bin/env python3
"""Audit how detector-aware supplements alter frozen CenterPoint voxels."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "results/detector_aware_pdans_surface256_v1_20260804/protocol.json"
INPUT_ROOT = REPO / "results/detector_aware_pdans_surface256_v1_20260804/inputs"
OBSERVED_ROOT = REPO / "data/KITTI/object/training/velodyne_original_val"
KITTI = REPO / "data/KITTI/object/training"
OUTPUT = REPO / "results/detector_aware_cross_detector_diagnosis256_v1_20260805"
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
POINT_RANGE = np.asarray([0.0, -40.0, -3.0, 70.4, 40.0, 1.0], dtype=np.float64)
VOXEL_SIZE = np.asarray([0.05, 0.05, 0.1], dtype=np.float64)
MAX_POINTS = 5
MAX_VOXELS = 40000


def load_helpers():
    path = REPO / "tools/generate_pointrcnn_detection_box_visualization_v1.py"
    spec = importlib.util.spec_from_file_location("voxel_diagnosis_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


det = load_helpers()


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"malformed point file {path}")
    return values.reshape(-1, 4)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def centerpoint_mask(points: np.ndarray, frame: str, calib: dict) -> np.ndarray:
    xyz1 = np.concatenate((points[:, :3].astype(np.float64), np.ones((len(points), 1))), axis=1)
    rect = xyz1 @ np.asarray(det.velo_to_rect_matrix(calib), dtype=np.float64).T
    projected = np.concatenate((rect[:, :3], np.ones((len(rect), 1))), axis=1) @ np.asarray(det.p2_matrix(calib), dtype=np.float64).T
    depth = projected[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, 1e-9, depth)
    uv = projected[:, :2] / safe[:, None]
    image_size = det.read_png_size(KITTI / "image_2" / f"{frame}.png")
    if image_size is None:
        raise FileNotFoundError(KITTI / "image_2" / f"{frame}.png")
    width, height = image_size
    fov = (depth > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    xyz = points[:, :3]
    in_range = np.all(xyz >= POINT_RANGE[:3], axis=1) & np.all(xyz < POINT_RANGE[3:], axis=1)
    return fov & in_range


def voxel_keys(points: np.ndarray) -> np.ndarray:
    coords = np.floor((points[:, :3].astype(np.float64) - POINT_RANGE[:3]) / VOXEL_SIZE).astype(np.int64)
    grid = np.ceil((POINT_RANGE[3:] - POINT_RANGE[:3]) / VOXEL_SIZE).astype(np.int64)
    return coords[:, 0] + grid[0] * (coords[:, 1] + grid[1] * coords[:, 2])


def retained_mask(keys: np.ndarray) -> np.ndarray:
    if len(keys) == 0:
        return np.zeros(0, dtype=bool)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.cumsum(np.r_[0, counts[:-1]])
    ranks_sorted = np.arange(len(keys)) - np.repeat(starts, counts)
    kept = np.zeros(len(keys), dtype=bool)
    kept[order] = ranks_sorted < MAX_POINTS
    return kept


def voxel_means(points: np.ndarray, keys: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected_keys = keys[keep]
    selected_points = points[keep].astype(np.float64)
    if not len(selected_keys):
        return np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float64)
    order = np.argsort(selected_keys, kind="stable")
    sorted_keys = selected_keys[order]
    sorted_points = selected_points[order]
    unique, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    sums = np.add.reduceat(sorted_points, starts, axis=0)
    return unique, sums / counts[:, None]


def points_in_gt(points: np.ndarray, gt_boxes: list[dict]) -> tuple[int, int, dict[str, int]]:
    if not len(points):
        return 0, 0, {}
    inside_any = np.zeros(len(points), dtype=bool)
    boundary_any = np.zeros(len(points), dtype=bool)
    class_counts: dict[str, int] = defaultdict(int)
    for box in gt_boxes:
        if box["class"] not in ("Car", "Pedestrian", "Cyclist"):
            continue
        corners = np.asarray(box["corners_lidar"], dtype=np.float64)
        base = corners[:4]
        center = base[:, :2].mean(axis=0)
        axis_a = base[1, :2] - base[0, :2]
        axis_b = base[3, :2] - base[0, :2]
        length_a = float(np.linalg.norm(axis_a))
        length_b = float(np.linalg.norm(axis_b))
        if length_a <= 1e-8 or length_b <= 1e-8:
            continue
        local = points[:, :2] - center
        coord_a = local @ (axis_a / length_a)
        coord_b = local @ (axis_b / length_b)
        z_min, z_max = float(corners[:, 2].min()), float(corners[:, 2].max())
        inside = (np.abs(coord_a) <= length_a / 2) & (np.abs(coord_b) <= length_b / 2) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        if np.any(inside):
            margins = np.column_stack((length_a / 2 - np.abs(coord_a), length_b / 2 - np.abs(coord_b), points[:, 2] - z_min, z_max - points[:, 2]))
            boundary = inside & (margins.min(axis=1) <= 0.20)
            inside_any |= inside
            boundary_any |= boundary
            class_counts[box["class"]] += int(np.count_nonzero(inside))
    return int(np.count_nonzero(inside_any)), int(np.count_nonzero(boundary_any)), dict(class_counts)


def analyze_cloud(frame: str, variant: str, observed: np.ndarray, cloud: np.ndarray, calib: dict, gt: list[dict]) -> dict:
    prefix_preserved = len(cloud) >= len(observed) and np.array_equal(cloud[: len(observed)], observed)
    if not prefix_preserved:
        raise RuntimeError(f"{frame}/{variant}: observed point prefix was not preserved")
    source_generated = np.arange(len(cloud)) >= len(observed)
    mask = centerpoint_mask(cloud, frame, calib)
    selected = cloud[mask]
    generated = source_generated[mask]
    keys = voxel_keys(selected)
    keep = retained_mask(keys)
    unique = np.unique(keys)
    observed_keys = np.unique(keys[~generated])
    generated_keys = np.unique(keys[generated])
    generated_only_keys = np.setdiff1d(generated_keys, observed_keys, assume_unique=True)
    mixed_keys = np.intersect1d(generated_keys, observed_keys, assume_unique=True)
    generated_count = int(np.count_nonzero(generated))
    retained_generated = int(np.count_nonzero(keep & generated))
    inside, boundary, class_counts = points_in_gt(selected[generated, :3].astype(np.float64), gt)
    return {
        "frame_id": frame,
        "variant": variant,
        "raw_points": len(cloud),
        "supplement_points_raw": len(cloud) - len(observed),
        "fov_range_points": len(selected),
        "active_voxels": len(unique),
        "voxel_cap_hit": len(unique) > MAX_VOXELS,
        "generated_fov_range_points": generated_count,
        "generated_voxels": len(generated_keys),
        "generated_only_voxels": len(generated_only_keys),
        "mixed_voxels": len(mixed_keys),
        "generated_points_in_observed_voxel_fraction": float(np.count_nonzero(generated & np.isin(keys, observed_keys))) / max(generated_count, 1),
        "generated_only_voxel_fraction": len(generated_only_keys) / max(len(generated_keys), 1),
        "generated_retained_after_max5": retained_generated,
        "generated_retention_fraction_max5": retained_generated / max(generated_count, 1),
        "generated_inside_gt": inside,
        "generated_inside_gt_fraction": inside / max(generated_count, 1),
        "generated_near_inside_boundary": boundary,
        "boundary_fraction_of_inside_gt": boundary / max(inside, 1),
        "generated_inside_car": class_counts.get("Car", 0),
        "generated_inside_pedestrian": class_counts.get("Pedestrian", 0),
        "generated_inside_cyclist": class_counts.get("Cyclist", 0),
        "_points": selected,
        "_keys": keys,
        "_keep": keep,
    }


def add_feature_shift(row: dict, baseline: dict) -> None:
    base_keys, base_means = voxel_means(baseline["_points"], baseline["_keys"], baseline["_keep"])
    keys, means = voxel_means(row["_points"], row["_keys"], row["_keep"])
    common, base_idx, new_idx = np.intersect1d(base_keys, keys, assume_unique=True, return_indices=True)
    delta = means[new_idx] - base_means[base_idx]
    xyz_shift = np.linalg.norm(delta[:, :3], axis=1) if len(delta) else np.empty(0)
    row["shared_voxels_with_baseline"] = len(common)
    row["new_active_voxels_vs_baseline"] = len(np.setdiff1d(keys, base_keys, assume_unique=True))
    row["lost_active_voxels_vs_baseline"] = len(np.setdiff1d(base_keys, keys, assume_unique=True))
    row["changed_shared_voxels"] = int(np.count_nonzero(np.any(np.abs(delta) > 1e-7, axis=1)))
    row["shared_voxel_xyz_mean_shift_mean_m"] = float(xyz_shift.mean()) if len(xyz_shift) else 0.0
    row["shared_voxel_xyz_mean_shift_p90_m"] = float(np.quantile(xyz_shift, 0.90)) if len(xyz_shift) else 0.0
    row["shared_voxel_intensity_mean_abs_shift"] = float(np.abs(delta[:, 3]).mean()) if len(delta) else 0.0


def aggregate(rows: list[dict]) -> list[dict]:
    fields = [
        "supplement_points_raw", "fov_range_points", "active_voxels", "generated_fov_range_points",
        "generated_voxels", "generated_only_voxels", "mixed_voxels",
        "generated_points_in_observed_voxel_fraction", "generated_only_voxel_fraction",
        "generated_retention_fraction_max5", "generated_inside_gt_fraction", "boundary_fraction_of_inside_gt",
        "new_active_voxels_vs_baseline", "changed_shared_voxels", "shared_voxel_xyz_mean_shift_mean_m",
        "shared_voxel_xyz_mean_shift_p90_m", "shared_voxel_intensity_mean_abs_shift",
    ]
    output = []
    for variant in VARIANTS:
        chosen = [row for row in rows if row["variant"] == variant]
        result: dict[str, object] = {"variant": variant, "frames": len(chosen), "voxel_cap_hit_frames": sum(bool(row["voxel_cap_hit"]) for row in chosen)}
        for field in fields:
            values = np.asarray([float(row[field]) for row in chosen], dtype=np.float64)
            result[f"{field}_mean"] = float(values.mean())
            result[f"{field}_median"] = float(np.median(values))
            result[f"{field}_p90"] = float(np.quantile(values, 0.90))
        output.append(result)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    frames = [str(value) for value in json.loads(PROTOCOL.read_text(encoding="utf-8"))["frame_ids"]]
    if args.limit:
        frames = frames[: args.limit]
    rows: list[dict] = []
    for position, frame in enumerate(frames, start=1):
        observed = read_bin(OBSERVED_ROOT / f"{frame}.bin")
        calib = det.parse_calib(KITTI / "calib" / f"{frame}.txt")
        gt = det.maybe_gt(frame, calib)
        base_points = observed[centerpoint_mask(observed, frame, calib)]
        base_keys = voxel_keys(base_points)
        baseline = {"_points": base_points, "_keys": base_keys, "_keep": retained_mask(base_keys)}
        for variant in VARIANTS:
            cloud = read_bin(INPUT_ROOT / variant / f"{frame}.bin")
            row = analyze_cloud(frame, variant, observed, cloud, calib, gt)
            add_feature_shift(row, baseline)
            rows.append({key: value for key, value in row.items() if not key.startswith("_")})
        if position == 1 or position % 16 == 0:
            print(f"voxels {position}/{len(frames)}", flush=True)
    output = args.output.resolve()
    write_csv(output / "voxels/per_frame_centerpoint_voxel_effects.csv", rows)
    write_csv(output / "voxels/aggregate_centerpoint_voxel_effects.csv", aggregate(rows))
    metadata = {
        "status": "PASS", "frames": len(frames), "variants": list(VARIANTS),
        "centerpoint_point_cloud_range": POINT_RANGE.tolist(), "centerpoint_voxel_size": VOXEL_SIZE.tolist(),
        "max_points_per_voxel": MAX_POINTS, "max_voxels_test": MAX_VOXELS,
        "observed_points_precede_supplements": True,
    }
    (output / "voxels/metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output / 'voxels'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
