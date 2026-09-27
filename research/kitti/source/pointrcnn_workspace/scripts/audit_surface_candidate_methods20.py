#!/usr/bin/env python3
"""Audit the frozen surface-patch candidate across all four upsamplers."""

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
OUTPUT = ROOT / "geometry20_surface_search_v1/reports/method_outputs"
LINE = "line_a_original_x4_up"
CLASSES = ("Car", "Pedestrian", "Cyclist")
CP_RANGE_MIN = np.asarray([0.0, -40.0, -3.0], dtype=np.float32)
CP_RANGE_MAX = np.asarray([70.4, 40.0, 1.0], dtype=np.float32)
CP_VOXEL_SIZE = np.asarray([0.05, 0.05, 0.1], dtype=np.float32)

OLD = REPO / "results/kitti_unified_x4_current_methods_no_detector" / LINE
CONDITIONS = {
    ("PU-GCN", "old_patch"): OLD / "pu_gcn/final_bin",
    ("PU-GCN", "surface_c32"): (
        ROOT / "pugcn20_surface_candidate_v1/pu_gcn_surface_cover_exact_pr1_c32" / LINE / "final_bin"
    ),
    ("PU-EdgeFormer", "old_patch"): OLD / "pu_edgeformer/final_bin",
    ("PU-EdgeFormer", "surface_c32"): (
        ROOT / "puedge20_surface_candidate_v1/pu_edgeformer_surface_cover_exact_pr1_c32" / LINE / "final_bin"
    ),
    ("PDANS", "old_patch"): OLD / "pdans/final_bin",
    ("PDANS", "surface_c32"): (
        ROOT / "pdans20_surface_candidate_v1/pdans_surface_cover_exact_pr1_c32" / LINE / "final_bin"
    ),
    # The PU-Net control uses the already-fixed normalization so this pair isolates patch effects.
    ("PU-Net-fixed", "old_patch"): (
        ROOT / "punet20_2x2/pu_net_B_fixed_oldpatch" / LINE / "final_bin"
    ),
    ("PU-Net-fixed", "surface_c32"): (
        ROOT / "punet20_surface_candidate_v1/pu_net_fixed_surface_cover_exact_pr1_c32" / LINE / "final_bin"
    ),
}


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


def box_mask(points_rect: np.ndarray, obj) -> np.ndarray:
    local = local_coordinates(points_rect, obj)
    return (
        (np.abs(local[:, 0]) <= obj.l / 2.0)
        & (np.abs(local[:, 2]) <= obj.w / 2.0)
        & (local[:, 1] >= -obj.h)
        & (local[:, 1] <= 0.0)
    )


def voxel_metrics(points: np.ndarray) -> dict[str, float | int]:
    xyz = points[:, :3]
    valid = np.all((xyz >= CP_RANGE_MIN) & (xyz < CP_RANGE_MAX), axis=1)
    selected = xyz[valid]
    if not len(selected):
        raise ValueError("no points in CenterPoint range")
    coordinates = np.floor((selected - CP_RANGE_MIN) / CP_VOXEL_SIZE).astype(np.int32)
    _, counts = np.unique(coordinates, axis=0, return_counts=True)
    descending = np.sort(counts)[::-1]
    top_count = max(1, int(np.ceil(0.10 * len(descending))))
    quantized_mm = np.rint(selected * 1000.0).astype(np.int32)
    return {
        "points_in_range": int(len(selected)),
        "occupied_voxels": int(len(counts)),
        "unique_1mm_fraction": float(len(np.unique(quantized_mm, axis=0)) / len(selected)),
        "points_per_voxel_p90": quantile(counts, 0.90),
        "top10pct_voxel_point_fraction": float(np.sum(descending[:top_count]) / len(selected)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    frames = [line.strip() for line in FRAMES_FILE.read_text().splitlines() if line.strip()]
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
        objects = [
            obj
            for obj in kitti_utils.get_objects_from_label(
                str(TRAINING / "label_2" / f"{frame}.txt")
            )
            if obj.cls_type in CLASSES and obj.level in (1, 2, 3)
        ]
        for (method, condition), directory in CONDITIONS.items():
            predicted = read_bin(directory / f"{frame}.bin")
            if len(predicted) != 4 * len(observed):
                raise ValueError(f"{method}/{condition}/{frame} is not strict 4N")
            rng = np.random.default_rng(stable_seed("surface20", method, condition, frame))
            sample = predicted[
                rng.choice(len(predicted), size=min(16384, len(predicted)), replace=False)
            ]
            distance = observed_tree.query(sample[:, :3], k=1, workers=1)[0]
            scene_rows.append(
                {
                    "frame_id": frame,
                    "method": method,
                    "condition": condition,
                    "predicted_points": len(predicted),
                    "nearest_real_p50_m": quantile(distance, 0.50),
                    "nearest_real_p90_m": quantile(distance, 0.90),
                    "nearest_real_p99_m": quantile(distance, 0.99),
                    "nearest_real_gt_0p25_fraction": float(np.mean(distance > 0.25)),
                    **voxel_metrics(predicted),
                }
            )
            predicted_rect = calibration.lidar_to_rect(predicted[:, :3])
            for object_index, obj in enumerate(objects):
                observed_count = int(np.count_nonzero(box_mask(observed_rect, obj)))
                if observed_count == 0:
                    continue
                inside_count = int(np.count_nonzero(box_mask(predicted_rect, obj)))
                object_rows.append(
                    {
                        "frame_id": frame,
                        "method": method,
                        "condition": condition,
                        "object_index": object_index,
                        "class": obj.cls_type,
                        "difficulty": obj.level_str,
                        "observed_inside_points": observed_count,
                        "inside_points": inside_count,
                        "inside_over_observed": inside_count / observed_count,
                    }
                )
        print(f"AUDIT {frame_index}/20 {frame}", flush=True)

    scene_summary: list[dict[str, object]] = []
    for method, condition in CONDITIONS:
        rows = [
            row
            for row in scene_rows
            if row["method"] == method and row["condition"] == condition
        ]
        entry: dict[str, object] = {"method": method, "condition": condition, "frames": len(rows)}
        for metric in (
            "nearest_real_p50_m",
            "nearest_real_p90_m",
            "nearest_real_p99_m",
            "nearest_real_gt_0p25_fraction",
            "occupied_voxels",
            "unique_1mm_fraction",
            "points_per_voxel_p90",
            "top10pct_voxel_point_fraction",
        ):
            entry[f"{metric}_frame_median"] = quantile([row[metric] for row in rows], 0.50)
        scene_summary.append(entry)

    object_summary: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in object_rows:
        grouped[(row["method"], row["condition"], row["class"], row["difficulty"])].append(row)
    for (method, condition, class_name, difficulty), rows in sorted(grouped.items()):
        object_summary.append(
            {
                "method": method,
                "condition": condition,
                "class": class_name,
                "difficulty": difficulty,
                "objects": len(rows),
                "inside_points_sum": sum(int(row["inside_points"]) for row in rows),
                "inside_over_observed_median": quantile(
                    [row["inside_over_observed"] for row in rows], 0.50
                ),
                "inside_over_observed_p10": quantile(
                    [row["inside_over_observed"] for row in rows], 0.10
                ),
            }
        )

    deltas: list[dict[str, object]] = []
    by_key = {(row["method"], row["condition"]): row for row in scene_summary}
    for method in sorted({method for method, _ in CONDITIONS}):
        old = by_key[(method, "old_patch")]
        new = by_key[(method, "surface_c32")]
        delta: dict[str, object] = {"method": method, "comparison": "surface_c32_minus_old_patch"}
        for key in old:
            if key.endswith("_frame_median"):
                delta[key] = float(new[key]) - float(old[key])
        deltas.append(delta)

    write_csv(OUTPUT / "scene20.csv", scene_rows)
    write_csv(OUTPUT / "scene20_summary.csv", scene_summary)
    write_csv(OUTPUT / "scene20_deltas.csv", deltas)
    write_csv(OUTPUT / "objects20.csv", object_rows)
    write_csv(OUTPUT / "objects20_summary.csv", object_summary)
    payload = {
        "status": "PASS",
        "frames": frames,
        "conditions": {f"{method}/{condition}": str(path) for (method, condition), path in CONDITIONS.items()},
        "scene_summary": scene_summary,
        "surface_minus_old": deltas,
        "note": "KITTI labels are used only for post-hoc audit, never for patch extraction.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "frames": len(frames), "conditions": len(CONDITIONS)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
