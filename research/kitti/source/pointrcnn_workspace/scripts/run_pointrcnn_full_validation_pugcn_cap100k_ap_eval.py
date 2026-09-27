#!/usr/bin/env python3
"""Run full-validation AP evaluation for existing PU-GCN cap_100k inference outputs."""

from __future__ import annotations

import csv
import argparse
import math
import sys
import traceback
import types
from pathlib import Path
from typing import Dict, List

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_ROOT = PROJECT_ROOT / "results" / "pointrcnn_full_validation_pugcn_cap100k"
FULL_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
LABEL_DIR = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "label_2"
DETECTION_DIR = RESULT_ROOT / "inference" / "eval" / "epoch_no_number" / "val" / "test_mode" / "final_result" / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--full-split", type=Path, default=FULL_SPLIT)
    parser.add_argument("--label-dir", type=Path, default=LABEL_DIR)
    parser.add_argument("--detection-dir", type=Path, default=None)
    parser.add_argument("--method", type=str, default="PU-GCN cap_100k")
    parser.add_argument("--detector-variant", type=str, default="PU-GCN detector-normalized")
    parser.add_argument("--summary-prefix", type=str, default="full_validation_ap_eval")
    return parser.parse_args()


def _rbbox_to_corners(rbbox: np.ndarray) -> np.ndarray:
    center_x, center_z, dim_x, dim_z, angle = [float(v) for v in rbbox]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    local_x = np.array([-dim_x / 2, -dim_x / 2, dim_x / 2, dim_x / 2], dtype=np.float32)
    local_z = np.array([-dim_z / 2, dim_z / 2, dim_z / 2, -dim_z / 2], dtype=np.float32)
    corners = np.zeros((4, 2), dtype=np.float32)
    for i in range(4):
        corners[i, 0] = a_cos * local_x[i] + a_sin * local_z[i] + center_x
        corners[i, 1] = -a_sin * local_x[i] + a_cos * local_z[i] + center_z
    return corners


def _polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _polygon_orientation(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _segment_intersection(p1: np.ndarray, p2: np.ndarray, cp1: np.ndarray, cp2: np.ndarray) -> np.ndarray:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = cp1
    x4, y4 = cp2
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-8:
        return np.array([x2, y2], dtype=np.float32)
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    return np.array([x1 + ua * (x2 - x1), y1 + ua * (y2 - y1)], dtype=np.float32)


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray, clip_orientation: float) -> bool:
    cross = (
        (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])
    )
    return bool(clip_orientation * cross >= -1e-8)


def _convex_polygon_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    if subject.shape[0] == 0:
        return subject
    output = subject.copy()
    clip_orientation = np.sign(_polygon_orientation(clip))
    if clip_orientation == 0:
        clip_orientation = 1.0
    for i in range(len(clip)):
        edge_start = clip[i]
        edge_end = clip[(i + 1) % len(clip)]
        input_list = output
        output_points = []
        if len(input_list) == 0:
            break
        previous = input_list[-1]
        for current in input_list:
            current_inside = _inside(current, edge_start, edge_end, clip_orientation)
            previous_inside = _inside(previous, edge_start, edge_end, clip_orientation)
            if current_inside:
                if not previous_inside:
                    output_points.append(_segment_intersection(previous, current, edge_start, edge_end))
                output_points.append(np.asarray(current, dtype=np.float32))
            elif previous_inside:
                output_points.append(_segment_intersection(previous, current, edge_start, edge_end))
            previous = current
        output = [np.asarray(point, dtype=np.float32) for point in output_points]
    if len(output) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack(output, axis=0)


def intersection_area(box1: np.ndarray, box2: np.ndarray) -> float:
    clipped = _convex_polygon_clip(_rbbox_to_corners(box1), _rbbox_to_corners(box2))
    if clipped.size == 0:
        return 0.0
    return _polygon_area(clipped)


def rotate_iou_cpu_eval(boxes: np.ndarray, query_boxes: np.ndarray, criterion: int = -1, device_id: int = 0) -> np.ndarray:
    del device_id
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    out = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=np.float32)
    for i in range(boxes.shape[0]):
        area1 = float(boxes[i, 2] * boxes[i, 3])
        for j in range(query_boxes.shape[0]):
            area2 = float(query_boxes[j, 2] * query_boxes[j, 3])
            inter = intersection_area(boxes[i], query_boxes[j])
            if inter <= 0.0:
                continue
            if criterion == -1:
                denom = area1 + area2 - inter
                out[i, j] = inter / denom if denom > 0.0 else 0.0
            elif criterion == 0:
                out[i, j] = inter / area1 if area1 > 0.0 else 0.0
            elif criterion == 1:
                out[i, j] = inter / area2 if area2 > 0.0 else 0.0
            else:
                out[i, j] = inter
    return out


def patch_evaluator() -> None:
    fake_rotate_iou = types.ModuleType("tools.kitti_object_eval_python.rotate_iou")
    fake_rotate_iou.rotate_iou_gpu_eval = rotate_iou_cpu_eval
    sys.modules["tools.kitti_object_eval_python.rotate_iou"] = fake_rotate_iou
    import tools.kitti_object_eval_python.eval as eval_mod

    eval_mod.rotate_iou_gpu_eval = rotate_iou_cpu_eval


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    detection_dir = args.detection_dir if args.detection_dir is not None else args.result_root / "inference" / "eval" / "epoch_no_number" / "val" / "test_mode" / "final_result" / "data"
    frame_ids = [line.strip() for line in args.full_split.read_text(encoding="utf-8").splitlines() if line.strip()]
    patch_evaluator()
    from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate

    missing_labels = [frame_id for frame_id in frame_ids if not (args.label_dir / ("%s.txt" % frame_id)).exists()]
    missing_dets = [frame_id for frame_id in frame_ids if not (detection_dir / ("%s.txt" % frame_id)).exists()]
    row: Dict[str, object] = {
        "method": args.method,
        "detector_variant": args.detector_variant,
        "evaluated_frames": len(frame_ids),
        "detection_dir": str(detection_dir),
        "missing_labels": ",".join(missing_labels) if missing_labels else "none",
        "missing_detection_files": ",".join(missing_dets) if missing_dets else "none",
        "status": "pending",
        "warnings": "",
    }
    if missing_labels or missing_dets:
        row["status"] = "missing_inputs"
        write_csv(args.result_root / ("%s_summary.csv" % args.summary_prefix), [row])
        return
    try:
        ap_result_str, ap_dict = kitti_evaluate(str(args.label_dir), str(detection_dir), label_split_file=str(args.full_split), current_class=0)
        row.update(
            {
                "bbox_AP_easy": float(ap_dict["Car_image_easy"]),
                "bbox_AP_moderate": float(ap_dict["Car_image_moderate"]),
                "bbox_AP_hard": float(ap_dict["Car_image_hard"]),
                "bev_AP_easy": float(ap_dict["Car_bev_easy"]),
                "bev_AP_moderate": float(ap_dict["Car_bev_moderate"]),
                "bev_AP_hard": float(ap_dict["Car_bev_hard"]),
                "3d_AP_easy": float(ap_dict["Car_3d_easy"]),
                "3d_AP_moderate": float(ap_dict["Car_3d_moderate"]),
                "3d_AP_hard": float(ap_dict["Car_3d_hard"]),
                "status": "success",
            }
        )
        (args.result_root / "raw_eval_output.txt").write_text(ap_result_str, encoding="utf-8")
    except Exception as exc:
        row["status"] = "failed"
        row["warnings"] = repr(exc)
        (args.result_root / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    write_csv(args.result_root / ("%s_summary.csv" % args.summary_prefix), [row])
    print(row["status"])


if __name__ == "__main__":
    main()
