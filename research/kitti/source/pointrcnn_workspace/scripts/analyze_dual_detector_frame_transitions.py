#!/usr/bin/env python3
"""Rank KITTI frames by baseline-to-upsampling detector transitions.

This analysis is read-only with respect to all existing experiment outputs.  It
uses the frozen PointRCNN E2 predictions and the frozen CenterPoint exact-4N E1
``result.pkl`` files, then audits the same KITTI validation frames for both
detectors, both experiment lines, and all four formal upsampling methods.

The resulting CSV files are explanatory frame/object audits.  Official AP
continues to come from the existing full validation evaluations.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[1]
KITTI = REPO / "data/KITTI/object/training"
VAL_SPLIT = REPO / "data/KITTI/ImageSets/val.txt"
POINT_WORKSPACE = REPO / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
CENTER_WORKSPACE = REPO / "results/centerpoint_exact4n_e1_20260729"
DEFAULT_OUTPUT = REPO / "results/dual_detector_three_frame_root_cause_20260730"

METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
METHOD_LABEL = {
    "pdans": "PDANS",
    "pu_gcn": "PU-GCN",
    "pu_edgeformer": "PU-EdgeFormer",
    "pu_net": "PU-Net",
}
LINES = {
    "A": {
        "baseline": "original_baseline",
        "prefix": "original_x4",
        "label": "Line A",
    },
    "B": {
        "baseline": "downsampled_x4_baseline",
        "prefix": "downsampled_x4",
        "label": "Line B",
    },
}
CLASS_IOU_THRESHOLD = {"Car": 0.70, "Pedestrian": 0.50, "Cyclist": 0.50}
DETECTOR_CLASSES = {
    "pointrcnn": ("Car",),
    "centerpoint": ("Car", "Pedestrian", "Cyclist"),
}


def load_detection_helpers():
    path = REPO / "tools/generate_pointrcnn_detection_box_visualization_v1.py"
    spec = importlib.util.spec_from_file_location("dual_detector_detection_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import detection helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


det = load_detection_helpers()


def read_frames(path: Path = VAL_SPLIT) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def point_pred_dir(variant: str) -> Path:
    return (
        POINT_WORKSPACE
        / "eval_outputs/e2"
        / variant
        / "inference/eval/epoch_no_number/val/final_result/data"
    )


def center_result_path(variant: str) -> Path:
    return (
        CENTER_WORKSPACE
        / variant
        / "full/openpcdet_output/eval/epoch_80/val/frozen_exact4n_e1/result.pkl"
    )


def lidar_box_corners(box: Iterable[float]) -> list[list[float]]:
    """Return eight LiDAR-frame corners for [x,y,z,dx,dy,dz,heading]."""
    x, y, z, dx, dy, dz, heading = [float(value) for value in box]
    c, s = math.cos(heading), math.sin(heading)
    local = (
        (dx / 2.0, dy / 2.0),
        (-dx / 2.0, dy / 2.0),
        (-dx / 2.0, -dy / 2.0),
        (dx / 2.0, -dy / 2.0),
    )
    xy = [(x + lx * c - ly * s, y + lx * s + ly * c) for lx, ly in local]
    bottom = [[px, py, z - dz / 2.0] for px, py in xy]
    top = [[px, py, z + dz / 2.0] for px, py in xy]
    return bottom + top


def centerpoint_annotations(path: Path) -> dict[str, dict]:
    with path.open("rb") as handle:
        annotations = pickle.load(handle)
    return {str(annotation["frame_id"]): annotation for annotation in annotations}


def centerpoint_boxes(annotation: dict, source: str) -> list[dict]:
    boxes = []
    for idx, (name, score, box) in enumerate(
        zip(annotation["name"], annotation["score"], annotation["boxes_lidar"])
    ):
        box_values = np.asarray(box, dtype=np.float64)
        boxes.append(
            {
                "id": idx,
                "class": str(name),
                "score": float(score),
                "source": source,
                "center_lidar": box_values[:3].tolist(),
                "dimensions_lidar": box_values[3:6].tolist(),
                "heading_lidar": float(box_values[6]),
                "corners_lidar": lidar_box_corners(box_values),
            }
        )
    return boxes


def box_center(box: dict) -> np.ndarray:
    if "center_lidar" in box:
        return np.asarray(box["center_lidar"], dtype=np.float64)
    return np.asarray(box["corners_lidar"], dtype=np.float64).mean(axis=0)


def polygon_xy(box: dict) -> list[tuple[float, float]]:
    points = [(float(x), float(y)) for x, y, _ in box["corners_lidar"][:4]]
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    points.sort(key=lambda point: math.atan2(point[1] - cy, point[0] - cx))
    return points


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            points[idx][0] * points[(idx + 1) % len(points)][1]
            - points[(idx + 1) % len(points)][0] * points[idx][1]
            for idx in range(len(points))
        )
    ) * 0.5


def cross(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def line_intersection(a, b, p, q):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = p
    x4, y4 = q
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12:
        return b
    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denominator
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denominator
    return px, py


def clip_polygon(subject, clip):
    output = list(subject)
    if not output:
        return []
    orientation = 1.0 if sum(
        clip[idx][0] * clip[(idx + 1) % len(clip)][1]
        - clip[(idx + 1) % len(clip)][0] * clip[idx][1]
        for idx in range(len(clip))
    ) >= 0 else -1.0
    for idx in range(len(clip)):
        edge_a = clip[idx]
        edge_b = clip[(idx + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = orientation * cross(edge_a, edge_b, previous) >= -1e-9
        for current in input_points:
            current_inside = orientation * cross(edge_a, edge_b, current) >= -1e-9
            if current_inside:
                if not previous_inside:
                    output.append(line_intersection(previous, current, edge_a, edge_b))
                output.append(current)
            elif previous_inside:
                output.append(line_intersection(previous, current, edge_a, edge_b))
            previous, previous_inside = current, current_inside
    return output


def oriented_iou(box_a: dict, box_b: dict) -> tuple[float, float]:
    polygon_a, polygon_b = polygon_xy(box_a), polygon_xy(box_b)
    area_a, area_b = polygon_area(polygon_a), polygon_area(polygon_b)
    intersection_area = polygon_area(clip_polygon(polygon_a, polygon_b))
    area_union = area_a + area_b - intersection_area
    bev_iou = intersection_area / area_union if area_union > 0 else 0.0
    z_a = [float(point[2]) for point in box_a["corners_lidar"]]
    z_b = [float(point[2]) for point in box_b["corners_lidar"]]
    intersection_height = max(0.0, min(max(z_a), max(z_b)) - max(min(z_a), min(z_b)))
    intersection_volume = intersection_area * intersection_height
    volume_a = area_a * (max(z_a) - min(z_a))
    volume_b = area_b * (max(z_b) - min(z_b))
    volume_union = volume_a + volume_b - intersection_volume
    iou3d = intersection_volume / volume_union if volume_union > 0 else 0.0
    return bev_iou, iou3d


def gt_for_frame(frame: str, calib: dict) -> list[dict]:
    return det.maybe_gt(frame, calib)


def point_predictions(frame: str, variant: str, calib: dict) -> list[dict]:
    return det.parse_kitti_boxes(point_pred_dir(variant) / f"{frame}.txt", calib, variant)


def match_gt_predictions(
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    classes: Iterable[str],
) -> dict:
    """Greedily match by oriented 3D IoU and apply KITTI class thresholds."""
    classes = tuple(classes)
    gt = [box for box in gt_boxes if box["class"] in classes]
    pred = [box for box in pred_boxes if box["class"] in classes]
    candidates = []
    for gt_idx, gt_box in enumerate(gt):
        for pred_idx, pred_box in enumerate(pred):
            if gt_box["class"] != pred_box["class"]:
                continue
            center_distance = float(np.linalg.norm(box_center(gt_box) - box_center(pred_box)))
            if center_distance > 6.0:
                continue
            bev_iou, iou3d = oriented_iou(gt_box, pred_box)
            candidates.append((-iou3d, center_distance, gt_idx, pred_idx, bev_iou, iou3d))
    candidates.sort()
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches = []
    for _, distance, gt_idx, pred_idx, bev_iou, iou3d in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        threshold = CLASS_IOU_THRESHOLD[gt[gt_idx]["class"]]
        if iou3d < threshold:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append(
            {
                "gt_idx": gt_idx,
                "pred_idx": pred_idx,
                "class": gt[gt_idx]["class"],
                "center_distance_m": distance,
                "bev_iou": bev_iou,
                "iou3d": iou3d,
                "threshold": threshold,
                "score": pred[pred_idx].get("score"),
            }
        )
    nearest = {}
    for gt_idx, gt_box in enumerate(gt):
        same_class = [
            (oriented_iou(gt_box, pred_box)[1], pred_idx)
            for pred_idx, pred_box in enumerate(pred)
            if pred_box["class"] == gt_box["class"]
        ]
        if same_class:
            best_iou, best_idx = max(same_class)
            nearest[gt_idx] = {
                "pred_idx": best_idx,
                "iou3d": best_iou,
                "score": pred[best_idx].get("score"),
            }
    return {
        "gt": gt,
        "pred": pred,
        "matches": matches,
        "match_by_gt": {row["gt_idx"]: row for row in matches},
        "matched_gt": used_gt,
        "matched_pred": used_pred,
        "missed_gt": [idx for idx in range(len(gt)) if idx not in used_gt],
        "false_positive_pred": [idx for idx in range(len(pred)) if idx not in used_pred],
        "nearest_by_gt": nearest,
    }


def transition_rows(
    detector: str,
    line: str,
    method: str,
    frame: str,
    baseline_audit: dict,
    upsampled_audit: dict,
) -> list[dict]:
    rows = []
    base_by_gt = baseline_audit["match_by_gt"]
    up_by_gt = upsampled_audit["match_by_gt"]
    for gt_idx, gt_box in enumerate(baseline_audit["gt"]):
        baseline_match = base_by_gt.get(gt_idx)
        upsampled_match = up_by_gt.get(gt_idx)
        if baseline_match and upsampled_match:
            iou_delta = upsampled_match["iou3d"] - baseline_match["iou3d"]
            score_delta = float(upsampled_match["score"] or 0.0) - float(
                baseline_match["score"] or 0.0
            )
            if iou_delta <= -0.10:
                transition = "localization_degraded"
            elif score_delta <= -0.10:
                transition = "confidence_degraded"
            else:
                transition = "maintained"
        elif baseline_match:
            transition = "lost_after_upsampling"
        elif upsampled_match:
            transition = "recovered_after_upsampling"
        else:
            transition = "missed_both"
        center = box_center(gt_box)
        rows.append(
            {
                "detector": detector,
                "line": line,
                "method": METHOD_LABEL[method],
                "variant": f"{LINES[line]['prefix']}_{method}",
                "frame_id": frame,
                "gt_index": gt_idx,
                "class": gt_box["class"],
                "gt_center_x_m": round(float(center[0]), 4),
                "gt_center_y_m": round(float(center[1]), 4),
                "gt_center_z_m": round(float(center[2]), 4),
                "transition": transition,
                "baseline_pred_index": "" if not baseline_match else baseline_match["pred_idx"],
                "baseline_score": "" if not baseline_match else baseline_match["score"],
                "baseline_bev_iou": "" if not baseline_match else baseline_match["bev_iou"],
                "baseline_3d_iou": "" if not baseline_match else baseline_match["iou3d"],
                "upsampled_pred_index": "" if not upsampled_match else upsampled_match["pred_idx"],
                "upsampled_score": "" if not upsampled_match else upsampled_match["score"],
                "upsampled_bev_iou": "" if not upsampled_match else upsampled_match["bev_iou"],
                "upsampled_3d_iou": "" if not upsampled_match else upsampled_match["iou3d"],
                "delta_3d_iou": (
                    ""
                    if not baseline_match or not upsampled_match
                    else upsampled_match["iou3d"] - baseline_match["iou3d"]
                ),
            }
        )
    return rows


def load_centerpoint_matrix() -> dict[str, dict[str, dict]]:
    variants = [cfg["baseline"] for cfg in LINES.values()]
    variants.extend(
        f"{cfg['prefix']}_{method}" for cfg in LINES.values() for method in METHODS
    )
    return {variant: centerpoint_annotations(center_result_path(variant)) for variant in variants}


def analyze(frames: list[str]) -> tuple[list[dict], list[dict]]:
    centerpoint = load_centerpoint_matrix()
    ranking = defaultdict(
        lambda: {
            "score": 0.0,
            "lost": 0,
            "recovered": 0,
            "localization_degraded": 0,
            "confidence_degraded": 0,
            "maintained": 0,
            "missed_both": 0,
            "new_fp": 0,
            "removed_fp": 0,
            "pointrcnn_events": 0,
            "centerpoint_events": 0,
        }
    )
    transitions: list[dict] = []
    for position, frame in enumerate(frames, start=1):
        calib = det.parse_calib(KITTI / "calib" / f"{frame}.txt")
        gt_boxes = gt_for_frame(frame, calib)
        ranking[frame]["gt_car"] = sum(box["class"] == "Car" for box in gt_boxes)
        ranking[frame]["gt_pedestrian"] = sum(
            box["class"] == "Pedestrian" for box in gt_boxes
        )
        ranking[frame]["gt_cyclist"] = sum(box["class"] == "Cyclist" for box in gt_boxes)
        for detector in ("pointrcnn", "centerpoint"):
            classes = DETECTOR_CLASSES[detector]
            for line, line_cfg in LINES.items():
                baseline_variant = str(line_cfg["baseline"])
                if detector == "pointrcnn":
                    baseline_predictions = point_predictions(
                        frame, baseline_variant, calib
                    )
                else:
                    baseline_predictions = centerpoint_boxes(
                        centerpoint[baseline_variant][frame], baseline_variant
                    )
                baseline_audit = match_gt_predictions(gt_boxes, baseline_predictions, classes)
                for method in METHODS:
                    variant = f"{line_cfg['prefix']}_{method}"
                    if detector == "pointrcnn":
                        upsampled_predictions = point_predictions(frame, variant, calib)
                    else:
                        upsampled_predictions = centerpoint_boxes(
                            centerpoint[variant][frame], variant
                        )
                    upsampled_audit = match_gt_predictions(
                        gt_boxes, upsampled_predictions, classes
                    )
                    case_rows = transition_rows(
                        detector,
                        line,
                        method,
                        frame,
                        baseline_audit,
                        upsampled_audit,
                    )
                    transitions.extend(case_rows)
                    counts = Counter(row["transition"] for row in case_rows)
                    new_fp = max(
                        0,
                        len(upsampled_audit["false_positive_pred"])
                        - len(baseline_audit["false_positive_pred"]),
                    )
                    removed_fp = max(
                        0,
                        len(baseline_audit["false_positive_pred"])
                        - len(upsampled_audit["false_positive_pred"]),
                    )
                    info = ranking[frame]
                    info["lost"] += counts["lost_after_upsampling"]
                    info["recovered"] += counts["recovered_after_upsampling"]
                    info["localization_degraded"] += counts["localization_degraded"]
                    info["confidence_degraded"] += counts["confidence_degraded"]
                    info["maintained"] += counts["maintained"]
                    info["missed_both"] += counts["missed_both"]
                    info["new_fp"] += new_fp
                    info["removed_fp"] += removed_fp
                    event_count = (
                        counts["lost_after_upsampling"]
                        + counts["recovered_after_upsampling"]
                        + counts["localization_degraded"]
                        + counts["confidence_degraded"]
                        + new_fp
                        + removed_fp
                    )
                    info[f"{detector}_events"] += event_count
                    info["score"] += (
                        7.0 * counts["lost_after_upsampling"]
                        + 5.0 * counts["recovered_after_upsampling"]
                        + 3.0 * counts["localization_degraded"]
                        + 1.5 * counts["confidence_degraded"]
                        + 1.5 * new_fp
                        + 0.75 * removed_fp
                    )
        if position == 1 or position % 250 == 0:
            print(f"audited {position}/{len(frames)} frames", flush=True)
    rank_rows = []
    for frame, info in ranking.items():
        class_diversity = sum(
            int(info.get(key, 0) > 0)
            for key in ("gt_car", "gt_pedestrian", "gt_cyclist")
        )
        cross_detector = int(info["pointrcnn_events"] > 0 and info["centerpoint_events"] > 0)
        info["score"] += 4.0 * class_diversity + 8.0 * cross_detector
        rank_rows.append({"frame_id": frame, **info, "class_diversity": class_diversity})
    rank_rows.sort(key=lambda row: (-float(row["score"]), row["frame_id"]))
    for rank, row in enumerate(rank_rows, start=1):
        row["rank"] = rank
    return rank_rows, transitions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Audit only the first N validation frames (0 means all 3,769).",
    )
    args = parser.parse_args()
    frames = read_frames()
    if args.limit:
        frames = frames[: args.limit]
    ranking, transitions = analyze(frames)
    write_csv(args.output / "analysis/frame_candidate_ranking.csv", ranking)
    write_csv(args.output / "analysis/all_frame_gt_transitions.csv", transitions)
    print(f"ranking={args.output / 'analysis/frame_candidate_ranking.csv'}")
    print(f"transitions={args.output / 'analysis/all_frame_gt_transitions.csv'}")
    print("top_frames=" + ",".join(row["frame_id"] for row in ranking[:20]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
