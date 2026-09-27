#!/usr/bin/env python3
"""Audit label-free patch candidates for object/surface completeness on geometry20.

KITTI labels are used only after extraction as a diagnostic.  They never affect
patch centers or memberships.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.utils import kitti_utils  # noqa: E402
from lib.utils.calibration import Calibration  # noqa: E402


ROOT = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
TRAINING = REPO / "data/KITTI/object/training"
SOURCE = TRAINING / "velodyne_original_val"
FRAMES = ROOT / "splits/geometry20.txt"
RUN_KIND = ROOT / "geometry20_surface_search_v1"
DEFAULT_OUTPUT = RUN_KIND / "reports"
DEFAULT_VARIANTS = (
    "surface_fps_knn_pr2",
    "surface_fps_knn_pr3",
    "surface_cover_exact_pr1_c32",
    "surface_cover_exact_pr1_c8",
)
CLASSES = ("Car", "Pedestrian", "Cyclist")


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 4:
        raise ValueError(f"invalid KITTI XYZI: {path}")
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"NaN/Inf: {path}")
    return points


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if len(array) else float("nan")


def local_coordinates(points_rect: np.ndarray, obj) -> np.ndarray:
    relative = points_rect - obj.pos.reshape(1, 3)
    c, s = math.cos(obj.ry), math.sin(obj.ry)
    rotation = np.asarray([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
    return relative @ rotation.T


def box_mask(points_rect: np.ndarray, obj) -> np.ndarray:
    local = local_coordinates(points_rect, obj)
    return (
        (np.abs(local[:, 0]) <= obj.l / 2.0)
        & (np.abs(local[:, 2]) <= obj.w / 2.0)
        & (local[:, 1] >= -obj.h)
        & (local[:, 1] <= 0.0)
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


def audit_variant(
    variant: str, frames: list[str]
) -> tuple[list[dict], list[dict], list[dict]]:
    base = RUN_KIND / variant / "line_a_original_x4_up"
    scene_rows: list[dict] = []
    patch_rows: list[dict] = []
    object_rows: list[dict] = []
    for position, frame in enumerate(frames, start=1):
        metadata_path = base / "manifests" / f"{frame}_patch_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        points = read_bin(SOURCE / f"{frame}.bin")
        calibration = Calibration(str(TRAINING / "calib" / f"{frame}.txt"))
        points_rect = calibration.lidar_to_rect(points[:, :3])
        plane = road_plane(frame)
        ground = np.abs(points_rect @ plane[:3] + plane[3]) <= 0.20
        objects = []
        for obj in kitti_utils.get_objects_from_label(
            str(TRAINING / "label_2" / f"{frame}.txt")
        ):
            if obj.cls_type not in CLASSES or obj.level not in (1, 2, 3):
                continue
            mask = box_mask(points_rect, obj)
            count = int(np.count_nonzero(mask))
            if count >= 3:
                objects.append((obj, mask, count))

        membership = np.zeros(len(points), dtype=np.int32)
        best = [None for _ in objects]
        total_support = 0
        total_repeats = 0
        multi_object_patches = 0
        for patch in metadata["patches"]:
            raw_indices = np.asarray(patch["original_indices"], dtype=np.int64)
            unique_indices = np.unique(raw_indices)
            membership[unique_indices] += 1
            total_support += len(raw_indices)
            total_repeats += len(raw_indices) - len(unique_indices)
            object_counts = [int(np.count_nonzero(mask[unique_indices])) for _, mask, _ in objects]
            objects_present = sum(count > 0 for count in object_counts)
            multi_object_patches += int(objects_present >= 2)
            labeled_union = np.zeros(len(unique_indices), dtype=bool)
            for _, mask, _ in objects:
                labeled_union |= mask[unique_indices]
            patch_rows.append(
                {
                    "variant": variant,
                    "frame_id": frame,
                    "patch_id": patch["patch_id"],
                    "unique_source_points": len(unique_indices),
                    "repeat_fraction": 1.0 - len(unique_indices) / len(raw_indices),
                    "xy_diameter_m": patch["xy_bbox_diagonal_m"],
                    "xyz_diameter_m": patch["xyz_bbox_diagonal_m"],
                    "ground_fraction": float(np.mean(ground[unique_indices])),
                    "labeled_target_fraction": float(np.mean(labeled_union)),
                    "labeled_objects_present": objects_present,
                }
            )
            for object_index, count in enumerate(object_counts):
                if count == 0:
                    continue
                obj, mask, source_count = objects[object_index]
                candidate = {
                    "covered_points": count,
                    "coverage_fraction": count / source_count,
                    "target_purity": count / len(unique_indices),
                    "ground_fraction": float(np.mean(ground[unique_indices])),
                    "other_target_fraction": float(
                        np.mean(labeled_union & ~mask[unique_indices])
                    ),
                    "xy_diameter_m": float(patch["xy_bbox_diagonal_m"]),
                    "patch_id": int(patch["patch_id"]),
                }
                previous = best[object_index]
                rank = (
                    candidate["coverage_fraction"],
                    candidate["target_purity"],
                    -candidate["ground_fraction"],
                    -candidate["xy_diameter_m"],
                )
                if previous is None or rank > previous[0]:
                    best[object_index] = (rank, candidate)

        covered = membership > 0
        xy = [float(item["xy_bbox_diagonal_m"]) for item in metadata["patches"]]
        unique_support = [
            int(item["unique_original_index_count"]) for item in metadata["patches"]
        ]
        scene_rows.append(
            {
                "variant": variant,
                "frame_id": frame,
                "source_points": len(points),
                "patches": len(metadata["patches"]),
                "coverage_fraction": float(np.mean(covered)),
                "membership_mean_covered": float(np.mean(membership[covered])),
                "membership_p90_covered": quantile(membership[covered], 0.90),
                "unique_support_p10": quantile(unique_support, 0.10),
                "unique_support_p50": quantile(unique_support, 0.50),
                "repeat_fraction": total_repeats / max(total_support, 1),
                "xy_diameter_p50_m": quantile(xy, 0.50),
                "xy_diameter_p90_m": quantile(xy, 0.90),
                "multi_object_patch_fraction": multi_object_patches
                / max(len(metadata["patches"]), 1),
            }
        )
        for object_index, ((obj, mask, source_count), selected) in enumerate(
            zip(objects, best)
        ):
            union_fraction = float(np.mean(covered[mask]))
            candidate = {} if selected is None else selected[1]
            object_rows.append(
                {
                    "variant": variant,
                    "frame_id": frame,
                    "object_index": object_index,
                    "class": obj.cls_type,
                    "difficulty": obj.level_str,
                    "source_points": source_count,
                    "union_coverage_fraction": union_fraction,
                    "best_patch_coverage_fraction": candidate.get(
                        "coverage_fraction", 0.0
                    ),
                    "best_patch_target_purity": candidate.get("target_purity", 0.0),
                    "best_patch_ground_fraction": candidate.get(
                        "ground_fraction", float("nan")
                    ),
                    "best_patch_other_target_fraction": candidate.get(
                        "other_target_fraction", float("nan")
                    ),
                    "best_patch_xy_diameter_m": candidate.get(
                        "xy_diameter_m", float("nan")
                    ),
                    "best_patch_id": candidate.get("patch_id", ""),
                }
            )
        if position == 1 or position % 5 == 0 or position == len(frames):
            print(f"AUDIT {variant} {position}/{len(frames)} {frame}", flush=True)
    return scene_rows, patch_rows, object_rows


def aggregate_scene(rows: list[dict]) -> list[dict]:
    result = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        result.append(
            {
                "variant": variant,
                "frames": len(selected),
                "patches": sum(int(row["patches"]) for row in selected),
                "coverage_p50": quantile(
                    [row["coverage_fraction"] for row in selected], 0.50
                ),
                "coverage_min": min(float(row["coverage_fraction"]) for row in selected),
                "membership_mean_p50": quantile(
                    [row["membership_mean_covered"] for row in selected], 0.50
                ),
                "unique_support_p10": quantile(
                    [row["unique_support_p10"] for row in selected], 0.10
                ),
                "repeat_fraction_p50": quantile(
                    [row["repeat_fraction"] for row in selected], 0.50
                ),
                "xy_diameter_p50_m": quantile(
                    [row["xy_diameter_p50_m"] for row in selected], 0.50
                ),
                "xy_diameter_p90_m": quantile(
                    [row["xy_diameter_p90_m"] for row in selected], 0.50
                ),
                "multi_object_patch_fraction_p50": quantile(
                    [row["multi_object_patch_fraction"] for row in selected], 0.50
                ),
            }
        )
    return result


def aggregate_objects(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["variant"]), str(row["class"]), str(row["difficulty"]))].append(row)
        grouped[(str(row["variant"]), "ALL", "ALL")].append(row)
    result = []
    for (variant, class_name, difficulty), selected in sorted(grouped.items()):
        best = np.asarray(
            [row["best_patch_coverage_fraction"] for row in selected], dtype=np.float64
        )
        result.append(
            {
                "variant": variant,
                "class": class_name,
                "difficulty": difficulty,
                "objects": len(selected),
                "union_coverage_p10": quantile(
                    [row["union_coverage_fraction"] for row in selected], 0.10
                ),
                "union_coverage_p50": quantile(
                    [row["union_coverage_fraction"] for row in selected], 0.50
                ),
                "best_patch_coverage_p10": quantile(best, 0.10),
                "best_patch_coverage_p50": quantile(best, 0.50),
                "best_patch_coverage_ge_0p5_fraction": float(np.mean(best >= 0.50)),
                "best_patch_coverage_ge_0p8_fraction": float(np.mean(best >= 0.80)),
                "best_patch_target_purity_p50": quantile(
                    [row["best_patch_target_purity"] for row in selected], 0.50
                ),
                "best_patch_ground_fraction_p50": quantile(
                    [row["best_patch_ground_fraction"] for row in selected], 0.50
                ),
                "best_patch_other_target_fraction_p50": quantile(
                    [row["best_patch_other_target_fraction"] for row in selected], 0.50
                ),
                "best_patch_xy_diameter_p50_m": quantile(
                    [row["best_patch_xy_diameter_m"] for row in selected], 0.50
                ),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frames = [line.strip() for line in FRAMES.read_text().splitlines() if line.strip()]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes: list[dict] = []
    patches: list[dict] = []
    objects: list[dict] = []
    for variant in args.variants:
        variant_scene, variant_patch, variant_object = audit_variant(variant, frames)
        scenes.extend(variant_scene)
        patches.extend(variant_patch)
        objects.extend(variant_object)
    scene_summary = aggregate_scene(scenes)
    object_summary = aggregate_objects(objects)
    write_csv(output / "scene_per_frame.csv", scenes)
    write_csv(output / "patch_per_patch.csv", patches)
    write_csv(output / "object_per_object.csv", objects)
    write_csv(output / "scene_summary.csv", scene_summary)
    write_csv(output / "object_summary.csv", object_summary)
    payload = {
        "status": "PASS",
        "frames": len(frames),
        "labels_used_for_extraction": False,
        "labels_used_for_posthoc_audit_only": True,
        "scene_summary": scene_summary,
        "object_summary": object_summary,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
