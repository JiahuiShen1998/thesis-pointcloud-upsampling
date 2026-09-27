#!/usr/bin/env python3
"""Explain opposite old/new PDANS patch effects in PointRCNN and CenterPoint.

This is a read-only audit of frozen detector predictions and point clouds.  It
matches predictions to KITTI ground truth, reports per-object state changes,
and measures density/voxel concentration on the fixed geometry-20 subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from lib.utils import kitti_utils  # noqa: E402
from lib.utils.calibration import Calibration  # noqa: E402
import analyze_dual_detector_frame_transitions as transitions  # noqa: E402


TRAINING = REPO / "data/KITTI/object/training"
OBSERVED = TRAINING / "velodyne_original_val"
PROTOCOL = (
    REPO
    / "results/centerpoint_exact4n_reconstructed_order_safe_20260729"
    / "pilot256_protocol.json"
)
GEOMETRY20 = (
    REPO
    / "results/kitti_patch_punet_causal_ablation_v1_20260731"
    / "splits/geometry20.txt"
)
OLD_POINTS = (
    REPO
    / "results/kitti_unified_x4_current_methods_no_detector"
    / "line_a_original_x4_up/pdans/final_bin"
)
NEW_POINTS = (
    REPO
    / "results/kitti_patch_punet_causal_ablation_v1_20260731"
    / "pdans256_cover_knn_v3"
    / "pdans_cover_knn_v3_linea_r2_cr6_c32_k256"
    / "line_a_original_x4_up/final_bin"
)
OLD_POINT_PRED = (
    REPO
    / "results/kitti_patch_punet_causal_ablation_v1_20260731"
    / "detectors/pointrcnn/pdans_oldpatch_pilot256"
    / "inference/eval/epoch_no_number/patch_causal_pilot256/final_result/data"
)
NEW_POINT_PRED = (
    REPO
    / "results/kitti_patch_punet_causal_ablation_v1_20260731"
    / "detectors/pointrcnn/pdans_cover_knn_v3_linea_r2_cr6_c32_k256"
    / "inference/eval/epoch_no_number/patch_causal_pilot256/final_result/data"
)
OLD_CENTER_PRED = (
    REPO
    / "results/centerpoint_exact4n_reconstructed_order_safe_20260729"
    / "runs/reconstructed_e1/original_x4_pdans/pilot256/openpcdet_output"
    / "eval/epoch_80/val/frozen_exact4n_observed_first/result.pkl"
)
NEW_CENTER_PRED = (
    REPO
    / "results/kitti_patch_punet_causal_ablation_v1_20260731"
    / "detectors/centerpoint/pdans_cover_knn_v3_linea_r2_cr6_c32_k256"
    / "openpcdet_output/eval/epoch_80/val/frozen_patch_causal_pilot256/result.pkl"
)
DEFAULT_OUTPUT = (
    REPO
    / "results/kitti_patch_punet_causal_ablation_v1_20260731"
    / "analysis/pdans_old_vs_cover_knn_v3/root_cause"
)
CLASSES = ("Car", "Pedestrian", "Cyclist")
CLASS_BY_DETECTOR = {"PointRCNN": ("Car",), "CenterPoint": CLASSES}
SCORE_THRESHOLDS = (0.1, 0.3, 0.5, 0.7)
CP_RANGE_MIN = np.asarray([0.0, -40.0, -3.0], dtype=np.float32)
CP_RANGE_MAX = np.asarray([70.4, 40.0, 1.0], dtype=np.float32)
CP_VOXEL_SIZE = np.asarray([0.05, 0.05, 0.1], dtype=np.float32)
E1_BASE_SEED = 20260718


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 4:
        raise ValueError(f"invalid KITTI XYZI: {path}")
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"NaN/Inf in {path}")
    return points


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def centerpoint_input(frame: str, observed: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    expected = 4 * len(observed)
    if len(candidate) != expected:
        raise ValueError(f"{frame}: strict x4 violation {len(candidate)} != {expected}")
    rng = np.random.default_rng(stable_seed(E1_BASE_SEED, "e1", "original_x4_pdans", frame))
    selected = rng.choice(expected, size=3 * len(observed), replace=False)
    return np.concatenate((observed, candidate[selected]), axis=0)


def png_shape(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"invalid PNG header: {path}")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return height, width


def pointrcnn_sample(
    points: np.ndarray,
    calibration: Calibration,
    image_shape: tuple[int, int],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Reproduce the frozen PointRCNN 16,384-point evaluation sampler."""
    rect = calibration.lidar_to_rect(points[:, :3])
    image_xy, depth = calibration.rect_to_img(rect)
    height, width = image_shape
    valid = (
        (image_xy[:, 0] >= 0)
        & (image_xy[:, 0] < width)
        & (image_xy[:, 1] >= 0)
        & (image_xy[:, 1] < height)
        & (depth >= 0)
        & (rect[:, 0] >= -40.0)
        & (rect[:, 0] <= 40.0)
        & (rect[:, 1] >= -1.0)
        & (rect[:, 1] <= 3.0)
        & (rect[:, 2] >= 0.0)
        & (rect[:, 2] <= 70.4)
    )
    valid_points = points[valid]
    valid_rect = rect[valid]
    npoints = 16384
    if len(valid_points) > npoints:
        far = np.where(valid_rect[:, 2] >= 40.0)[0]
        near = np.where(valid_rect[:, 2] < 40.0)[0]
        if len(far) >= npoints:
            choice = rng.choice(np.arange(len(valid_points)), npoints, replace=False)
        else:
            near_choice = rng.choice(near, npoints - len(far), replace=False)
            choice = np.concatenate((near_choice, far)) if len(far) else near_choice
        rng.shuffle(choice)
    else:
        choice = np.arange(len(valid_points), dtype=np.int32)
        if len(valid_points) < npoints:
            extra = rng.choice(
                choice,
                npoints - len(valid_points),
                replace=(npoints - len(valid_points)) > len(choice),
            )
            choice = np.concatenate((choice, extra))
        rng.shuffle(choice)
    return valid_points[choice]


def reproduce_pointrcnn_samples(
    frames: list[str], selected_frames: set[str]
) -> dict[tuple[str, str], np.ndarray]:
    old_rng = np.random.RandomState(1024)
    new_rng = np.random.RandomState(1024)
    selected: dict[tuple[str, str], np.ndarray] = {}
    for position, frame in enumerate(frames, start=1):
        calibration = Calibration(str(TRAINING / "calib" / f"{frame}.txt"))
        shape = png_shape(TRAINING / "image_2" / f"{frame}.png")
        old_sample = pointrcnn_sample(
            read_bin(OLD_POINTS / f"{frame}.bin"), calibration, shape, old_rng
        )
        new_sample = pointrcnn_sample(
            read_bin(NEW_POINTS / f"{frame}.bin"), calibration, shape, new_rng
        )
        if frame in selected_frames:
            selected[("old_pointrcnn_sample", frame)] = old_sample
            selected[("new_pointrcnn_sample", frame)] = new_sample
        if position == 1 or position % 64 == 0 or position == len(frames):
            print(f"POINT_SAMPLE {position}/{len(frames)} {frame}", flush=True)
    return selected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values, q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, q)) if len(array) else float("nan")


def prediction_stats(predictions: list[dict], class_name: str) -> dict[str, float | int]:
    scores = np.asarray(
        [float(box.get("score") or 0.0) for box in predictions if box["class"] == class_name],
        dtype=np.float64,
    )
    result: dict[str, float | int] = {
        "boxes": int(len(scores)),
        "score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
        "score_p50": percentile(scores, 0.50),
        "score_p90": percentile(scores, 0.90),
    }
    for threshold in SCORE_THRESHOLDS:
        result[f"score_ge_{threshold:.1f}"] = int(np.count_nonzero(scores >= threshold))
    return result


def transition_for_gt(old_match: dict | None, new_match: dict | None) -> str:
    if old_match and new_match:
        iou_delta = float(new_match["iou3d"] - old_match["iou3d"])
        score_delta = float(new_match.get("score") or 0.0) - float(old_match.get("score") or 0.0)
        if iou_delta <= -0.10:
            return "localization_degraded"
        if score_delta <= -0.10:
            return "confidence_degraded"
        return "maintained"
    if old_match:
        return "lost"
    if new_match:
        return "recovered"
    return "missed_both"


def detector_audit(frames: list[str]) -> tuple[list[dict], list[dict]]:
    old_center = transitions.centerpoint_annotations(OLD_CENTER_PRED)
    new_center = transitions.centerpoint_annotations(NEW_CENTER_PRED)
    per_gt: list[dict] = []
    prediction_rows: list[dict] = []
    for position, frame in enumerate(frames, start=1):
        calib = transitions.det.parse_calib(TRAINING / "calib" / f"{frame}.txt")
        gt_boxes = transitions.gt_for_frame(frame, calib)
        prediction_pairs = {
            "PointRCNN": (
                transitions.det.parse_kitti_boxes(OLD_POINT_PRED / f"{frame}.txt", calib, "old"),
                transitions.det.parse_kitti_boxes(NEW_POINT_PRED / f"{frame}.txt", calib, "new"),
            ),
            "CenterPoint": (
                transitions.centerpoint_boxes(old_center[frame], "old"),
                transitions.centerpoint_boxes(new_center[frame], "new"),
            ),
        }
        for detector, (old_predictions, new_predictions) in prediction_pairs.items():
            classes = CLASS_BY_DETECTOR[detector]
            old_audit = transitions.match_gt_predictions(gt_boxes, old_predictions, classes)
            new_audit = transitions.match_gt_predictions(gt_boxes, new_predictions, classes)
            for class_name in classes:
                old_stats = prediction_stats(old_predictions, class_name)
                new_stats = prediction_stats(new_predictions, class_name)
                prediction_rows.append(
                    {
                        "frame_id": frame,
                        "detector": detector,
                        "class": class_name,
                        **{f"old_{key}": value for key, value in old_stats.items()},
                        **{f"new_{key}": value for key, value in new_stats.items()},
                        "old_false_positive_boxes": sum(
                            old_audit["pred"][idx]["class"] == class_name
                            for idx in old_audit["false_positive_pred"]
                        ),
                        "new_false_positive_boxes": sum(
                            new_audit["pred"][idx]["class"] == class_name
                            for idx in new_audit["false_positive_pred"]
                        ),
                    }
                )
            old_by_gt = old_audit["match_by_gt"]
            new_by_gt = new_audit["match_by_gt"]
            for gt_index, gt_box in enumerate(old_audit["gt"]):
                old_match = old_by_gt.get(gt_index)
                new_match = new_by_gt.get(gt_index)
                per_gt.append(
                    {
                        "frame_id": frame,
                        "detector": detector,
                        "class": gt_box["class"],
                        "gt_index": gt_index,
                        "transition": transition_for_gt(old_match, new_match),
                        "old_score": "" if not old_match else old_match["score"],
                        "new_score": "" if not new_match else new_match["score"],
                        "old_3d_iou": "" if not old_match else old_match["iou3d"],
                        "new_3d_iou": "" if not new_match else new_match["iou3d"],
                    }
                )
        if position == 1 or position % 64 == 0 or position == len(frames):
            print(f"DETECTOR {position}/{len(frames)} {frame}", flush=True)
    return per_gt, prediction_rows


def detector_summary(per_gt: list[dict], prediction_rows: list[dict]) -> list[dict]:
    result = []
    keys = sorted({(row["detector"], row["class"]) for row in per_gt})
    for detector, class_name in keys:
        gt_rows = [
            row for row in per_gt if row["detector"] == detector and row["class"] == class_name
        ]
        pred_rows = [
            row
            for row in prediction_rows
            if row["detector"] == detector and row["class"] == class_name
        ]
        counts = Counter(str(row["transition"]) for row in gt_rows)
        old_tp = sum(counts[key] for key in ("maintained", "localization_degraded", "confidence_degraded", "lost"))
        new_tp = sum(counts[key] for key in ("maintained", "localization_degraded", "confidence_degraded", "recovered"))
        row: dict[str, object] = {
            "detector": detector,
            "class": class_name,
            "gt_objects": len(gt_rows),
            **{key: counts[key] for key in ("maintained", "localization_degraded", "confidence_degraded", "lost", "recovered", "missed_both")},
            "old_matched_gt": old_tp,
            "new_matched_gt": new_tp,
            "net_matched_gt": new_tp - old_tp,
            "old_false_positive_boxes": sum(int(item["old_false_positive_boxes"]) for item in pred_rows),
            "new_false_positive_boxes": sum(int(item["new_false_positive_boxes"]) for item in pred_rows),
        }
        for condition in ("old", "new"):
            for metric in ("boxes", "score_ge_0.1", "score_ge_0.3", "score_ge_0.5", "score_ge_0.7"):
                row[f"{condition}_{metric}"] = sum(int(item[f"{condition}_{metric}"]) for item in pred_rows)
        result.append(row)
    return result


def local_coordinates(points_rect: np.ndarray, obj) -> np.ndarray:
    relative = points_rect - obj.pos.reshape(1, 3)
    c, s = np.cos(obj.ry), np.sin(obj.ry)
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


def voxel_metrics(points: np.ndarray) -> dict[str, float | int]:
    xyz = points[:, :3]
    valid = np.all((xyz >= CP_RANGE_MIN) & (xyz < CP_RANGE_MAX), axis=1)
    selected = xyz[valid]
    if not len(selected):
        return {"points_in_range": 0, "occupied_voxels": 0}
    coordinates = np.floor((selected - CP_RANGE_MIN) / CP_VOXEL_SIZE).astype(np.int32)
    _, counts = np.unique(coordinates, axis=0, return_counts=True)
    descending = np.sort(counts)[::-1]
    top_count = max(1, int(np.ceil(0.10 * len(descending))))
    quantized_mm = np.rint(selected * 1000.0).astype(np.int32)
    unique_mm = len(np.unique(quantized_mm, axis=0))
    return {
        "points_in_range": int(len(selected)),
        "in_range_fraction": float(len(selected) / len(points)),
        "occupied_voxels": int(len(counts)),
        "points_per_voxel_mean": float(np.mean(counts)),
        "points_per_voxel_p90": percentile(counts, 0.90),
        "points_per_voxel_p99": percentile(counts, 0.99),
        "top10pct_voxel_point_fraction": float(np.sum(descending[:top_count]) / len(selected)),
        "unique_1mm_fraction": float(unique_mm / len(selected)),
        "far_gt40m_fraction": float(np.mean(np.linalg.norm(selected[:, :2], axis=1) > 40.0)),
    }


def geometry_audit(
    frames: list[str], point_samples: dict[tuple[str, str], np.ndarray]
) -> tuple[list[dict], list[dict]]:
    scene_rows: list[dict] = []
    object_rows: list[dict] = []
    for position, frame in enumerate(frames, start=1):
        observed = read_bin(OBSERVED / f"{frame}.bin")
        old = read_bin(OLD_POINTS / f"{frame}.bin")
        new = read_bin(NEW_POINTS / f"{frame}.bin")
        conditions = {
            "old_strict4n": old,
            "new_strict4n": new,
            "old_centerpoint_input": centerpoint_input(frame, observed, old),
            "new_centerpoint_input": centerpoint_input(frame, observed, new),
            "old_pointrcnn_sample": point_samples[("old_pointrcnn_sample", frame)],
            "new_pointrcnn_sample": point_samples[("new_pointrcnn_sample", frame)],
        }
        calibration = Calibration(str(TRAINING / "calib" / f"{frame}.txt"))
        observed_rect = calibration.lidar_to_rect(observed[:, :3])
        objects = [
            obj
            for obj in kitti_utils.get_objects_from_label(str(TRAINING / "label_2" / f"{frame}.txt"))
            if obj.cls_type in CLASSES and obj.level in (1, 2, 3)
        ]
        for condition, points in conditions.items():
            scene_rows.append({"frame_id": frame, "condition": condition, "points": len(points), **voxel_metrics(points)})
            points_rect = calibration.lidar_to_rect(points[:, :3])
            for object_index, obj in enumerate(objects):
                observed_count = int(np.count_nonzero(box_mask(observed_rect, obj)))
                if observed_count == 0:
                    continue
                inside_count = int(np.count_nonzero(box_mask(points_rect, obj)))
                object_rows.append(
                    {
                        "frame_id": frame,
                        "condition": condition,
                        "object_index": object_index,
                        "class": obj.cls_type,
                        "difficulty": obj.level_str,
                        "observed_inside_points": observed_count,
                        "inside_points": inside_count,
                        "inside_over_observed": inside_count / observed_count,
                    }
                )
        print(f"GEOMETRY {position}/{len(frames)} {frame}", flush=True)
    return scene_rows, object_rows


def geometry_summary(scene_rows: list[dict], object_rows: list[dict]) -> dict:
    scene: list[dict] = []
    for condition in sorted({str(row["condition"]) for row in scene_rows}):
        selected = [row for row in scene_rows if row["condition"] == condition]
        entry: dict[str, object] = {"condition": condition, "frames": len(selected)}
        for metric in (
            "in_range_fraction",
            "occupied_voxels",
            "points_per_voxel_mean",
            "points_per_voxel_p90",
            "points_per_voxel_p99",
            "top10pct_voxel_point_fraction",
            "unique_1mm_fraction",
            "far_gt40m_fraction",
        ):
            entry[f"{metric}_frame_median"] = percentile([row.get(metric, np.nan) for row in selected], 0.50)
        scene.append(entry)
    objects: list[dict] = []
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in object_rows:
        groups[(str(row["condition"]), str(row["class"]), str(row["difficulty"]))].append(row)
    for (condition, class_name, difficulty), selected in sorted(groups.items()):
        objects.append(
            {
                "condition": condition,
                "class": class_name,
                "difficulty": difficulty,
                "objects": len(selected),
                "inside_points_sum": sum(int(row["inside_points"]) for row in selected),
                "inside_over_observed_median": percentile([row["inside_over_observed"] for row in selected], 0.50),
                "inside_over_observed_p10": percentile([row["inside_over_observed"] for row in selected], 0.10),
            }
        )
    return {"scene": scene, "objects": objects}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    frames = [str(item) for item in json.loads(PROTOCOL.read_text())["frame_ids"]]
    geometry_frames = [line.strip() for line in GEOMETRY20.read_text().splitlines() if line.strip()]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    per_gt, prediction_rows = detector_audit(frames)
    detector_aggregate = detector_summary(per_gt, prediction_rows)
    point_samples = reproduce_pointrcnn_samples(frames, set(geometry_frames))
    scene_rows, object_rows = geometry_audit(geometry_frames, point_samples)
    geometry_aggregate = geometry_summary(scene_rows, object_rows)

    write_csv(output / "detector_gt_transitions.csv", per_gt)
    write_csv(output / "detector_prediction_stats_per_frame.csv", prediction_rows)
    write_csv(output / "detector_transition_summary.csv", detector_aggregate)
    write_csv(output / "geometry_scene20.csv", scene_rows)
    write_csv(output / "geometry_object20.csv", object_rows)
    write_csv(output / "geometry_scene20_summary.csv", geometry_aggregate["scene"])
    write_csv(output / "geometry_object20_summary.csv", geometry_aggregate["objects"])
    payload = {
        "status": "PASS",
        "detector_frames": len(frames),
        "geometry_frames": len(geometry_frames),
        "detector_transition_summary": detector_aggregate,
        "geometry_summary": geometry_aggregate,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
