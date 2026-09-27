#!/usr/bin/env python3
"""Audit how patch membership covers KITTI target-box source points on geometry20."""

from __future__ import annotations

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
SOURCES = {
    "line_a_original_x4_up": TRAINING / "velodyne_original_val",
    "line_b_downsampled_x4_up": (
        REPO
        / "results/kitti_unified_x4_current_methods_no_detector"
        / "downsampled_x4/velodyne_downsampled_x4_val"
    ),
}
CONDITIONS = {
    ("line_a_original_x4_up", "local_v1"): (
        ROOT / "geometry20/patch_local_ball_r2_pr3_linea/line_a_original_x4_up/manifests"
    ),
    ("line_b_downsampled_x4_up", "local_v1"): (
        ROOT / "geometry20/patch_local_ball_r6_pr3/line_b_downsampled_x4_up/manifests"
    ),
    ("line_a_original_x4_up", "cover_v2"): (
        ROOT
        / "geometry20_cover_v2/pu_gcn_cover_v2_linea_r2_c256"
        / "line_a_original_x4_up/manifests"
    ),
    ("line_b_downsampled_x4_up", "cover_v2"): (
        ROOT
        / "geometry20_cover_v2/pu_gcn_cover_v2_lineb_r6_c256"
        / "line_b_downsampled_x4_up/manifests"
    ),
    ("line_a_original_x4_up", "cover_adaptive"): (
        ROOT
        / "geometry20_cover_adaptive/pu_gcn_cover_adaptive_linea_r2_cr6_c32"
        / "line_a_original_x4_up/manifests"
    ),
    ("line_b_downsampled_x4_up", "cover_adaptive"): (
        ROOT
        / "geometry20_cover_adaptive/pu_gcn_cover_adaptive_lineb_r6_cr12_c32"
        / "line_b_downsampled_x4_up/manifests"
    ),
    ("line_a_original_x4_up", "cover_adaptive_c256"): (
        ROOT
        / "geometry20_cover_adaptive_c256/pu_gcn_cover_adaptive_linea_r2_cr6_c256"
        / "line_a_original_x4_up/manifests"
    ),
    ("line_b_downsampled_x4_up", "cover_adaptive_c256"): (
        ROOT
        / "geometry20_cover_adaptive_c256/pu_gcn_cover_adaptive_lineb_r6_cr12_c256"
        / "line_b_downsampled_x4_up/manifests"
    ),
    ("line_a_original_x4_up", "cover_knn_v3"): (
        ROOT
        / "geometry20_cover_knn_v3/pu_gcn_cover_knn_v3_linea_r2_cr6_c32_k256"
        / "line_a_original_x4_up/manifests"
    ),
    ("line_b_downsampled_x4_up", "cover_knn_v3"): (
        ROOT
        / "geometry20_cover_knn_v3/pu_gcn_cover_knn_v3_lineb_r6_cr12_c32_k256"
        / "line_b_downsampled_x4_up/manifests"
    ),
}
CLASSES = ("Car", "Pedestrian", "Cyclist")


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN/Inf")
    return points


def box_mask(points_rect: np.ndarray, obj) -> np.ndarray:
    relative = points_rect - obj.pos.reshape(1, 3)
    c, s = math.cos(obj.ry), math.sin(obj.ry)
    rotation = np.asarray(
        [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]], dtype=np.float64
    )
    local = relative @ rotation.T
    return (
        (np.abs(local[:, 0]) <= obj.l / 2.0)
        & (np.abs(local[:, 2]) <= obj.w / 2.0)
        & (local[:, 1] >= -obj.h)
        & (local[:, 1] <= 0.0)
    )


def main() -> int:
    rows: list[dict[str, object]] = []
    for (line, condition), manifest_dir in CONDITIONS.items():
        metas = sorted(manifest_dir.glob("*_patch_metadata.json"))
        if len(metas) != 20:
            raise RuntimeError(f"expected 20 metadata files, found {len(metas)}: {manifest_dir}")
        for meta_path in metas:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            frame = str(meta["frame_id"])
            points = read_bin(SOURCES[line] / f"{frame}.bin")
            covered = np.zeros(len(points), dtype=bool)
            for patch in meta["patches"]:
                covered[np.asarray(patch["original_indices"], dtype=np.int64)] = True
            calibration = Calibration(str(TRAINING / "calib" / f"{frame}.txt"))
            points_rect = calibration.lidar_to_rect(points[:, :3])
            objects = kitti_utils.get_objects_from_label(
                str(TRAINING / "label_2" / f"{frame}.txt")
            )
            target_index = 0
            for obj in objects:
                if obj.cls_type not in CLASSES or obj.level not in (1, 2, 3):
                    continue
                inside = box_mask(points_rect, obj)
                source_count = int(np.count_nonzero(inside))
                if source_count == 0:
                    continue
                covered_count = int(np.count_nonzero(inside & covered))
                rows.append(
                    {
                        "line": line,
                        "condition": condition,
                        "frame_id": frame,
                        "target_index": target_index,
                        "class": obj.cls_type,
                        "difficulty": obj.level_str,
                        "source_inside_points": source_count,
                        "covered_inside_points": covered_count,
                        "target_source_coverage_fraction": covered_count / source_count,
                        "target_has_zero_covered_source_points": covered_count == 0,
                    }
                )
                target_index += 1

    output = ROOT / "geometry20_cover_knn_v3/reports"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "target_source_coverage_per_object.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["line"], row["condition"], row["class"])].append(row)
    aggregates: list[dict[str, object]] = []
    for (line, condition, class_name), selected in sorted(grouped.items()):
        fractions = np.asarray(
            [row["target_source_coverage_fraction"] for row in selected], dtype=np.float64
        )
        source_counts = np.asarray(
            [row["source_inside_points"] for row in selected], dtype=np.int64
        )
        zero_source_counts = source_counts[fractions == 0.0]
        aggregates.append(
            {
                "line": line,
                "condition": condition,
                "class": class_name,
                "objects": len(selected),
                "coverage_p10": float(np.quantile(fractions, 0.10)),
                "coverage_p50": float(np.quantile(fractions, 0.50)),
                "coverage_min": float(np.min(fractions)),
                "zero_covered_object_fraction": float(np.mean(fractions == 0.0)),
                "zero_covered_source_points_p50": (
                    float(np.quantile(zero_source_counts, 0.50))
                    if len(zero_source_counts)
                    else 0.0
                ),
                "zero_covered_source_points_max": (
                    int(np.max(zero_source_counts)) if len(zero_source_counts) else 0
                ),
                "fully_covered_object_fraction": float(np.mean(fractions == 1.0)),
            }
        )
    with (output / "target_source_coverage_aggregate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    print(json.dumps({"status": "PASS", "objects": len(rows), "aggregates": aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
