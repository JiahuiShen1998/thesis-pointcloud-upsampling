#!/usr/bin/env python3
"""Geometry and BEV audit for the 20-frame PU-Net normalization/patch 2x2."""

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

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
except ModuleNotFoundError:
    plt = None


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.utils import kitti_utils  # noqa: E402
from lib.utils.calibration import Calibration  # noqa: E402


ROOT = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
RUN = ROOT / "punet20_2x2"
LINE = "line_a_original_x4_up"
OBSERVED = REPO / "data/KITTI/object/training/velodyne_original_val"
TRAINING = REPO / "data/KITTI/object/training"
FRAMES_FILE = ROOT / "splits/geometry20.txt"
OUTPUT = RUN / "reports"
CONDITIONS = {
    "A_legacy_oldpatch": RUN / "pu_net_A_legacy_oldpatch" / LINE / "final_bin",
    "B_fixed_oldpatch": RUN / "pu_net_B_fixed_oldpatch" / LINE / "final_bin",
    "C_legacy_localpatch": RUN / "pu_net_C_legacy_cover_knn_v3" / LINE / "final_bin",
    "D_fixed_localpatch": RUN / "pu_net_D_fixed_cover_knn_v3" / LINE / "final_bin",
}
CLASSES = ("Car", "Pedestrian", "Cyclist")


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 4:
        raise ValueError(f"invalid KITTI bin: {path}")
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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = [line.strip() for line in FRAMES_FILE.read_text().splitlines() if line.strip()]
    if len(frames) != 20:
        raise ValueError(f"expected 20 frames, got {len(frames)}")
    scene_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    object_counts: dict[str, int] = {}

    for frame_index, frame in enumerate(frames, start=1):
        observed = read_bin(OBSERVED / f"{frame}.bin")
        tree = cKDTree(observed[:, :3])
        calibration = Calibration(str(TRAINING / "calib" / f"{frame}.txt"))
        observed_rect = calibration.lidar_to_rect(observed[:, :3])
        objects = [
            obj
            for obj in kitti_utils.get_objects_from_label(
                str(TRAINING / "label_2" / f"{frame}.txt")
            )
            if obj.cls_type in CLASSES and obj.level in (1, 2, 3)
        ]
        object_counts[frame] = len(objects)
        for condition, directory in CONDITIONS.items():
            predicted = read_bin(directory / f"{frame}.bin")
            if len(predicted) != 4 * len(observed):
                raise ValueError(f"{condition}/{frame} is not strict 4N")
            rng = np.random.default_rng(stable_seed("punet2x2", condition, frame))
            sample = predicted[
                rng.choice(len(predicted), size=min(16384, len(predicted)), replace=False)
            ]
            distances = tree.query(sample[:, :3], k=1, workers=-1)[0]
            radial = np.linalg.norm(sample[:, :3], axis=1)
            scene_rows.append(
                {
                    "frame_id": frame,
                    "condition": condition,
                    "observed_points": len(observed),
                    "predicted_points": len(predicted),
                    "nearest_real_p50_m": quantile(distances, 0.50),
                    "nearest_real_p90_m": quantile(distances, 0.90),
                    "nearest_real_p99_m": quantile(distances, 0.99),
                    "nearest_real_gt_0p25_fraction": float(np.mean(distances > 0.25)),
                    "nearest_real_gt_1p0_fraction": float(np.mean(distances > 1.0)),
                    "radial_p50_m": quantile(radial, 0.50),
                    "radial_p99_m": quantile(radial, 0.99),
                    "x_p01_m": quantile(sample[:, 0], 0.01),
                    "x_p99_m": quantile(sample[:, 0], 0.99),
                    "y_p01_m": quantile(sample[:, 1], 0.01),
                    "y_p99_m": quantile(sample[:, 1], 0.99),
                    "z_p01_m": quantile(sample[:, 2], 0.01),
                    "z_p99_m": quantile(sample[:, 2], 0.99),
                }
            )
            predicted_rect = calibration.lidar_to_rect(predicted[:, :3])
            for object_index, obj in enumerate(objects):
                real_inside = box_mask(observed_rect, obj)
                if not np.any(real_inside):
                    continue
                generated_inside = box_mask(predicted_rect, obj)
                object_rows.append(
                    {
                        "frame_id": frame,
                        "condition": condition,
                        "object_index": object_index,
                        "class": obj.cls_type,
                        "difficulty": obj.level_str,
                        "real_inside_points": int(np.count_nonzero(real_inside)),
                        "generated_inside_points": int(np.count_nonzero(generated_inside)),
                        "zero_generated_inside": not np.any(generated_inside),
                    }
                )
        print(f"AUDIT {frame_index}/20 {frame}", flush=True)

    aggregate: dict[str, dict[str, float]] = {}
    for condition in CONDITIONS:
        scene = [row for row in scene_rows if row["condition"] == condition]
        objects = [row for row in object_rows if row["condition"] == condition]
        aggregate[condition] = {
            "nearest_real_p50_frame_median_m": quantile(
                [row["nearest_real_p50_m"] for row in scene], 0.50
            ),
            "nearest_real_p90_frame_median_m": quantile(
                [row["nearest_real_p90_m"] for row in scene], 0.50
            ),
            "nearest_real_gt_0p25_fraction_frame_median": quantile(
                [row["nearest_real_gt_0p25_fraction"] for row in scene], 0.50
            ),
            "nearest_real_gt_1p0_fraction_frame_median": quantile(
                [row["nearest_real_gt_1p0_fraction"] for row in scene], 0.50
            ),
            "radial_p99_frame_median_m": quantile(
                [row["radial_p99_m"] for row in scene], 0.50
            ),
            "target_objects": len(objects),
            "zero_generated_target_fraction": float(
                np.mean([row["zero_generated_inside"] for row in objects])
            ),
            "generated_inside_points_median": quantile(
                [row["generated_inside_points"] for row in objects], 0.50
            ),
        }

    pairs = {
        "normalization_effect_oldpatch_B_minus_A": ("B_fixed_oldpatch", "A_legacy_oldpatch"),
        "normalization_effect_localpatch_D_minus_C": (
            "D_fixed_localpatch",
            "C_legacy_localpatch",
        ),
        "patch_effect_legacy_C_minus_A": ("C_legacy_localpatch", "A_legacy_oldpatch"),
        "patch_effect_fixed_D_minus_B": ("D_fixed_localpatch", "B_fixed_oldpatch"),
    }
    effects = {}
    effect_metrics = (
        "nearest_real_p50_frame_median_m",
        "nearest_real_p90_frame_median_m",
        "nearest_real_gt_0p25_fraction_frame_median",
        "zero_generated_target_fraction",
    )
    for name, (left, right) in pairs.items():
        effects[name] = {
            metric: aggregate[left][metric] - aggregate[right][metric]
            for metric in effect_metrics
        }

    write_csv(OUTPUT / "scene_geometry.csv", scene_rows)
    write_csv(OUTPUT / "object_coverage.csv", object_rows)

    visual_frame = max(frames, key=lambda item: (object_counts[item], item))
    figure_path = None
    if plt is not None:
        observed = read_bin(OBSERVED / f"{visual_frame}.bin")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
        for axis, (condition, directory) in zip(axes.flat, CONDITIONS.items()):
            predicted = read_bin(directory / f"{visual_frame}.bin")
            rng = np.random.default_rng(
                stable_seed("punet2x2_visual", condition, visual_frame)
            )
            observed_sample = observed[
                rng.choice(len(observed), size=min(20000, len(observed)), replace=False)
            ]
            predicted_sample = predicted[
                rng.choice(len(predicted), size=min(20000, len(predicted)), replace=False)
            ]
            axis.scatter(
                observed_sample[:, 0],
                observed_sample[:, 1],
                s=0.3,
                c="0.75",
                label="real",
            )
            axis.scatter(
                predicted_sample[:, 0],
                predicted_sample[:, 1],
                s=0.35,
                c="#d62728",
                alpha=0.55,
                label="PU-Net",
            )
            axis.set_title(condition)
            axis.set_xlim(-10, 85)
            axis.set_ylim(-45, 45)
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.15)
        axes[0, 0].legend(markerscale=8, loc="upper right")
        fig.supxlabel("LiDAR x (m)")
        fig.supylabel("LiDAR y (m)")
        fig.suptitle(f"PU-Net 2×2 BEV audit, frame {visual_frame}")
        fig.tight_layout()
        figure_path = OUTPUT / f"punet_2x2_bev_{visual_frame}.png"
        fig.savefig(figure_path, dpi=180)
        plt.close(fig)

    payload = {
        "status": "PASS",
        "frames": frames,
        "conditions": {key: str(value) for key, value in CONDITIONS.items()},
        "aggregate": aggregate,
        "effects": effects,
        "visual_frame": visual_frame,
        "visualization": str(figure_path) if figure_path else None,
    }
    (OUTPUT / "geometry_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
