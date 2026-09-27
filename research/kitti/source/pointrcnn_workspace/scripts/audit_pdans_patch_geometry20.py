#!/usr/bin/env python3
"""Audit old versus local-patch PDANS geometry on the fixed 20-frame split."""

from __future__ import annotations

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


ROOT = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
TRAINING = REPO / "data/KITTI/object/training"
OBSERVED = TRAINING / "velodyne_original_val"
FRAMES_FILE = ROOT / "splits/geometry20.txt"
OUTPUT = ROOT / "pdans20_cover_knn_v3/reports"
CONDITIONS = {
    "old_contiguous_patch": (
        REPO
        / "results/kitti_unified_x4_current_methods_no_detector"
        / "line_a_original_x4_up/pdans/final_bin"
    ),
    "local_cover_knn_v3": (
        ROOT
        / "pdans20_cover_knn_v3"
        / "pdans_cover_knn_v3_linea_r2_cr6_c32_k256"
        / "line_a_original_x4_up/final_bin"
    ),
}
CLASSES = ("Car", "Pedestrian", "Cyclist")


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 4:
        raise ValueError(f"invalid non-empty KITTI XYZI file: {path}")
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"NaN/Inf in {path}")
    return points


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def quantile(values, q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if array.size else float("nan")


def local_coordinates(points_rect: np.ndarray, obj) -> np.ndarray:
    relative = points_rect - obj.pos.reshape(1, 3)
    c, s = math.cos(obj.ry), math.sin(obj.ry)
    rotation = np.asarray(
        [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64
    )
    return relative @ rotation.T


def box_mask(points_rect: np.ndarray, obj, extra: float = 0.0) -> np.ndarray:
    local = local_coordinates(points_rect, obj)
    return (
        (np.abs(local[:, 0]) <= obj.l / 2.0 + extra)
        & (np.abs(local[:, 2]) <= obj.w / 2.0 + extra)
        & (local[:, 1] >= -obj.h - extra)
        & (local[:, 1] <= extra)
    )


def bev_mask(points_rect: np.ndarray, obj) -> np.ndarray:
    local = local_coordinates(points_rect, obj)
    return (np.abs(local[:, 0]) <= obj.l / 2.0) & (
        np.abs(local[:, 2]) <= obj.w / 2.0
    )


def road_plane(frame: str) -> np.ndarray:
    path = TRAINING / "planes" / f"{frame}.txt"
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for line in reversed(lines):
        values = np.asarray([float(item) for item in line.split()], dtype=np.float64)
        if values.size == 4 and np.linalg.norm(values[:3]) > 0:
            return values / np.linalg.norm(values[:3])
    raise ValueError(f"cannot parse road plane: {path}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    frames = [
        line.strip()
        for line in FRAMES_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(frames) != 20:
        raise ValueError(f"expected 20 frames, got {len(frames)}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scene_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []

    for frame_index, frame in enumerate(frames, start=1):
        observed = read_bin(OBSERVED / f"{frame}.bin")
        observed_tree = cKDTree(observed[:, :3])
        calibration = Calibration(str(TRAINING / "calib" / f"{frame}.txt"))
        observed_rect = calibration.lidar_to_rect(observed[:, :3])
        plane = road_plane(frame)
        objects = [
            obj
            for obj in kitti_utils.get_objects_from_label(
                str(TRAINING / "label_2" / f"{frame}.txt")
            )
            if obj.cls_type in CLASSES and obj.level in (1, 2, 3)
        ]

        for condition, directory in CONDITIONS.items():
            predicted = read_bin(directory / f"{frame}.bin")
            if len(predicted) != 4 * len(observed):
                raise ValueError(
                    f"{condition}/{frame}: {len(predicted)} is not strict 4N "
                    f"for observed {len(observed)}"
                )
            rng = np.random.default_rng(stable_seed("pdans_patch20", condition, frame))
            sample = predicted[
                rng.choice(len(predicted), size=min(16384, len(predicted)), replace=False)
            ]
            distance = observed_tree.query(sample[:, :3], k=1, workers=-1)[0]
            scene_rows.append(
                {
                    "frame_id": frame,
                    "condition": condition,
                    "observed_points": len(observed),
                    "predicted_points": len(predicted),
                    "nearest_real_p50_m": quantile(distance, 0.50),
                    "nearest_real_p90_m": quantile(distance, 0.90),
                    "nearest_real_p99_m": quantile(distance, 0.99),
                    "nearest_real_gt_0p25_fraction": float(np.mean(distance > 0.25)),
                    "nearest_real_gt_1p0_fraction": float(np.mean(distance > 1.0)),
                }
            )

            predicted_rect = calibration.lidar_to_rect(predicted[:, :3])
            distance_to_road = np.abs(predicted_rect @ plane[:3] + plane[3])
            for object_index, obj in enumerate(objects):
                observed_inside = box_mask(observed_rect, obj)
                if np.count_nonzero(observed_inside) < 3:
                    continue
                inside = box_mask(predicted_rect, obj)
                expanded = box_mask(predicted_rect, obj, extra=0.5)
                target_bev = bev_mask(predicted_rect, obj)
                inside_count = int(np.count_nonzero(inside))
                inside_distance = (
                    observed_tree.query(predicted[inside, :3], k=1, workers=-1)[0]
                    if inside_count
                    else np.empty(0, dtype=np.float64)
                )
                target_bev_count = int(np.count_nonzero(target_bev))
                ground_count = int(
                    np.count_nonzero(target_bev & (distance_to_road <= 0.20))
                )
                object_rows.append(
                    {
                        "frame_id": frame,
                        "condition": condition,
                        "object_index": object_index,
                        "class": obj.cls_type,
                        "difficulty": obj.level_str,
                        "observed_inside_points": int(np.count_nonzero(observed_inside)),
                        "predicted_inside_points": inside_count,
                        "predicted_inside_over_observed": inside_count
                        / max(int(np.count_nonzero(observed_inside)), 1),
                        "shell_0p5m_points": int(np.count_nonzero(expanded & ~inside)),
                        "inside_near_real_0p25_fraction": (
                            float(np.mean(inside_distance <= 0.25))
                            if inside_count
                            else float("nan")
                        ),
                        "target_bev_points": target_bev_count,
                        "ground_near_plane_points_in_target_bev": ground_count,
                        "ground_leakage_proxy_fraction": ground_count
                        / max(target_bev_count, 1),
                    }
                )
        print(f"AUDIT {frame_index}/20 {frame}", flush=True)

    scene_summary: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        rows = [row for row in scene_rows if row["condition"] == condition]
        scene_summary[condition] = {
            key: quantile([row[key] for row in rows], 0.50)
            for key in (
                "nearest_real_p50_m",
                "nearest_real_p90_m",
                "nearest_real_p99_m",
                "nearest_real_gt_0p25_fraction",
                "nearest_real_gt_1p0_fraction",
            )
        }

    object_summary: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in object_rows:
        grouped[(row["condition"], row["class"], row["difficulty"])].append(row)
    for (condition, class_name, difficulty), rows in sorted(grouped.items()):
        object_summary.append(
            {
                "condition": condition,
                "class": class_name,
                "difficulty": difficulty,
                "objects": len(rows),
                "predicted_inside_points_sum": sum(
                    int(row["predicted_inside_points"]) for row in rows
                ),
                "predicted_inside_over_observed_median": quantile(
                    [row["predicted_inside_over_observed"] for row in rows], 0.50
                ),
                "inside_near_real_0p25_fraction_median": quantile(
                    [row["inside_near_real_0p25_fraction"] for row in rows], 0.50
                ),
                "ground_leakage_proxy_median": quantile(
                    [row["ground_leakage_proxy_fraction"] for row in rows], 0.50
                ),
            }
        )

    old = scene_summary["old_contiguous_patch"]
    local = scene_summary["local_cover_knn_v3"]
    payload = {
        "status": "PASS",
        "frames": frames,
        "conditions": {key: str(value) for key, value in CONDITIONS.items()},
        "scene_summary_frame_medians": scene_summary,
        "local_minus_old": {key: local[key] - old[key] for key in old},
        "ground_leakage_proxy_definition": (
            "fraction of predicted points in each GT BEV footprint within 0.20m "
            "of the KITTI road plane"
        ),
    }
    write_csv(OUTPUT / "pdans_scene_geometry20.csv", scene_rows)
    write_csv(OUTPUT / "pdans_object_geometry20.csv", object_rows)
    write_csv(OUTPUT / "pdans_object_geometry20_aggregate.csv", object_summary)
    (OUTPUT / "pdans_geometry20_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
