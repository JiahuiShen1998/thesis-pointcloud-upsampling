#!/usr/bin/env python3
"""Compare old/local patch PU-GCN geometry on a fixed KITTI pilot subset."""

from __future__ import annotations

import argparse
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


TRAINING = REPO / "data/KITTI/object/training"
FULL = TRAINING / "velodyne_original_val"
DOWNSAMPLED = (
    REPO
    / "results/kitti_unified_x4_current_methods_no_detector"
    / "downsampled_x4/velodyne_downsampled_x4_val"
)
OLD_ROOT = REPO / "results/kitti_unified_x4_current_methods_no_detector"
NEW_ROOT = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/pilot256"
DEFAULT_PROTOCOL = (
    REPO
    / "results/centerpoint_exact4n_reconstructed_order_safe_20260729"
    / "pilot256_protocol.json"
)
DEFAULT_OUTPUT = (
    REPO
    / "results/kitti_patch_punet_causal_ablation_v1_20260731"
    / "geometry256/pu_gcn_old_vs_local"
)
CLASSES = ("Car", "Pedestrian", "Cyclist")


CONDITIONS = {
    ("line_a", "old_patch"): OLD_ROOT / "line_a_original_x4_up/pu_gcn/final_bin",
    ("line_a", "local_patch"): (
        NEW_ROOT / "pu_gcn_local_linea_r2/line_a_original_x4_up/final_bin"
    ),
    ("line_b", "old_patch"): OLD_ROOT / "line_b_downsampled_x4_up/pu_gcn/final_bin",
    ("line_b", "local_patch"): (
        NEW_ROOT / "pu_gcn_local_lineb_r6/line_b_downsampled_x4_up/final_bin"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-json", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--sample-frames", type=int, default=64)
    parser.add_argument("--points-per-frame", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(item) for item in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 4:
        raise ValueError(f"{path} is not non-empty float32 XYZI")
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN/Inf")
    return points


def choose_frames(protocol: Path, count: int) -> list[str]:
    frames = json.loads(protocol.read_text(encoding="utf-8"))["frame_ids"]
    if not 1 <= count <= len(frames):
        raise ValueError(f"--sample-frames must be in [1, {len(frames)}]")
    positions = np.linspace(0, len(frames) - 1, num=count, dtype=np.int64)
    return [str(frames[int(index)]) for index in np.unique(positions)]


def target_objects(frame: str):
    objects = kitti_utils.get_objects_from_label(
        str(TRAINING / "label_2" / f"{frame}.txt")
    )
    return [obj for obj in objects if obj.cls_type in CLASSES and obj.level in (1, 2, 3)]


def local_coordinates(points_rect: np.ndarray, obj) -> np.ndarray:
    relative = points_rect - obj.pos.reshape(1, 3)
    c, s = math.cos(obj.ry), math.sin(obj.ry)
    rotation = np.asarray(
        [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]],
        dtype=np.float64,
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


def bev_mask(points_rect: np.ndarray, obj, extra: float = 0.0) -> np.ndarray:
    local = local_coordinates(points_rect, obj)
    return (
        (np.abs(local[:, 0]) <= obj.l / 2.0 + extra)
        & (np.abs(local[:, 2]) <= obj.w / 2.0 + extra)
    )


def road_plane(frame: str) -> np.ndarray:
    lines = [
        line.strip()
        for line in (TRAINING / "planes" / f"{frame}.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for line in reversed(lines):
        values = np.asarray([float(item) for item in line.split()], dtype=np.float64)
        if values.size == 4:
            norm = np.linalg.norm(values[:3])
            if norm <= 0:
                break
            return values / norm
    raise ValueError(f"cannot parse road plane for {frame}")


def quantile(values: list[float] | np.ndarray, q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if array.size else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    frames = choose_frames(args.protocol_json.resolve(), args.sample_frames)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []

    for frame_index, frame in enumerate(frames, start=1):
        reference = read_bin(FULL / f"{frame}.bin")
        reference_tree = cKDTree(reference[:, :3])
        calibration = Calibration(str(TRAINING / "calib" / f"{frame}.txt"))
        reference_rect = calibration.lidar_to_rect(reference[:, :3])
        plane = road_plane(frame)
        objects = target_objects(frame)

        for line in ("line_a", "line_b"):
            observed = read_bin(
                (FULL if line == "line_a" else DOWNSAMPLED) / f"{frame}.bin"
            )
            observed_tree = cKDTree(observed[:, :3])
            for condition in ("old_patch", "local_patch"):
                predicted = read_bin(CONDITIONS[(line, condition)] / f"{frame}.bin")
                if predicted.shape[0] != 4 * observed.shape[0]:
                    raise ValueError(
                        f"{line}/{condition}/{frame}: {len(predicted)} is not strict 4N"
                    )
                sample_n = min(args.points_per_frame, len(predicted))
                rng = np.random.default_rng(
                    stable_seed("patch_causal_geometry", line, condition, frame)
                )
                sample_index = rng.choice(len(predicted), size=sample_n, replace=False)
                sample = predicted[sample_index]
                d_observed = observed_tree.query(sample[:, :3], k=1, workers=-1)[0]
                d_reference, neighbors = reference_tree.query(
                    sample[:, :3], k=min(8, len(reference)), workers=-1
                )
                if d_reference.ndim == 1:
                    nearest_reference = d_reference
                    neighbor_index = neighbors[:, None]
                else:
                    nearest_reference = d_reference[:, 0]
                    neighbor_index = neighbors
                local_neighbors = reference[neighbor_index, :3].astype(np.float64)
                centers = local_neighbors.mean(axis=1)
                centered = local_neighbors - centers[:, None, :]
                covariance = np.einsum("nki,nkj->nij", centered, centered) / local_neighbors.shape[1]
                _, eigenvectors = np.linalg.eigh(covariance)
                normals = eigenvectors[:, :, 0]
                plane_residual = np.abs(
                    np.einsum("ni,ni->n", sample[:, :3] - centers, normals)
                )
                scene_rows.append(
                    {
                        "frame_id": frame,
                        "line": line,
                        "condition": condition,
                        "observed_points": int(observed.shape[0]),
                        "predicted_points": int(predicted.shape[0]),
                        "sampled_points": int(sample_n),
                        "nearest_observed_p50_m": quantile(d_observed, 0.50),
                        "nearest_observed_p90_m": quantile(d_observed, 0.90),
                        "nearest_reference_p50_m": quantile(nearest_reference, 0.50),
                        "nearest_reference_p90_m": quantile(nearest_reference, 0.90),
                        "reference_distance_gt_0p25_fraction": float(
                            np.mean(nearest_reference > 0.25)
                        ),
                        "reference_distance_gt_0p50_fraction": float(
                            np.mean(nearest_reference > 0.50)
                        ),
                        "local_plane_residual_p50_m": quantile(plane_residual, 0.50),
                        "local_plane_residual_p90_m": quantile(plane_residual, 0.90),
                    }
                )

                predicted_rect = calibration.lidar_to_rect(predicted[:, :3])
                distance_to_road = np.abs(
                    predicted_rect @ plane[:3] + plane[3]
                )
                for object_index, obj in enumerate(objects):
                    ref_inside = box_mask(reference_rect, obj)
                    if np.count_nonzero(ref_inside) < 3:
                        continue
                    inside = box_mask(predicted_rect, obj)
                    expanded = box_mask(predicted_rect, obj, extra=0.5)
                    in_bev = bev_mask(predicted_rect, obj)
                    inside_count = int(np.count_nonzero(inside))
                    bev_count = int(np.count_nonzero(in_bev))
                    inside_points = predicted[inside, :3]
                    if inside_count:
                        inside_distance = reference_tree.query(
                            inside_points, k=1, workers=-1
                        )[0]
                    else:
                        inside_distance = np.empty(0, dtype=np.float64)
                    object_rows.append(
                        {
                            "frame_id": frame,
                            "line": line,
                            "condition": condition,
                            "class": obj.cls_type,
                            "difficulty": obj.level_str,
                            "object_index": object_index,
                            "object_depth_m": float(obj.pos[2]),
                            "reference_inside_points": int(np.count_nonzero(ref_inside)),
                            "predicted_inside_points": inside_count,
                            "predicted_inside_fraction_of_frame": inside_count / len(predicted),
                            "predicted_shell_0p5m_points": int(
                                np.count_nonzero(expanded & ~inside)
                            ),
                            "shell_to_inside_ratio": int(
                                np.count_nonzero(expanded & ~inside)
                            )
                            / max(inside_count, 1),
                            "inside_near_reference_0p25_fraction": (
                                float(np.mean(inside_distance <= 0.25))
                                if inside_count
                                else float("nan")
                            ),
                            "target_bev_points": bev_count,
                            "ground_near_plane_points_in_target_bev": int(
                                np.count_nonzero(in_bev & (distance_to_road <= 0.20))
                            ),
                            "ground_leakage_proxy_fraction": int(
                                np.count_nonzero(in_bev & (distance_to_road <= 0.20))
                            )
                            / max(bev_count, 1),
                        }
                    )
        print(f"GEOMETRY {frame_index}/{len(frames)} {frame}", flush=True)

    aggregate_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in object_rows:
        grouped[
            (row["line"], row["condition"], row["class"], row["difficulty"])
        ].append(row)
    for key, rows in sorted(grouped.items()):
        line, condition, class_name, difficulty = key
        aggregate_rows.append(
            {
                "line": line,
                "condition": condition,
                "class": class_name,
                "difficulty": difficulty,
                "objects": len(rows),
                "predicted_inside_points_sum": sum(
                    int(row["predicted_inside_points"]) for row in rows
                ),
                "inside_near_reference_0p25_median": quantile(
                    [row["inside_near_reference_0p25_fraction"] for row in rows], 0.50
                ),
                "shell_to_inside_ratio_median": quantile(
                    [row["shell_to_inside_ratio"] for row in rows], 0.50
                ),
                "ground_leakage_proxy_median": quantile(
                    [row["ground_leakage_proxy_fraction"] for row in rows], 0.50
                ),
            }
        )
    write_csv(output / "scene_geometry.csv", scene_rows)
    write_csv(output / "object_geometry.csv", object_rows)
    write_csv(output / "object_geometry_aggregate.csv", aggregate_rows)
    metadata = {
        "status": "PASS",
        "frames": frames,
        "sample_frames": len(frames),
        "points_per_frame": args.points_per_frame,
        "ground_leakage_proxy_definition": (
            "fraction of predicted points in each GT BEV footprint whose absolute "
            "distance to the KITTI road plane is <=0.20m"
        ),
        "conditions": {f"{line}/{condition}": str(path) for (line, condition), path in CONDITIONS.items()},
    }
    (output / "protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "frames": len(frames)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
