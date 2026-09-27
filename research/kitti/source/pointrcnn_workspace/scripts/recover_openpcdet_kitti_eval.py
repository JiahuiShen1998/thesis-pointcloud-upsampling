#!/usr/bin/env python3
"""Re-run OpenPCDet KITTI AP evaluation from an already saved result.pkl."""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np


REPO = Path(__file__).resolve().parents[1]
OPENPCDET = REPO / "external/OpenPCDet"
sys.path.insert(0, str(OPENPCDET))

from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-pkl", type=Path, required=True)
    parser.add_argument("--infos-pkl", type=Path, required=True)
    parser.add_argument("--output-log", type=Path, required=True)
    parser.add_argument(
        "--class-name",
        action="append",
        choices=("Car", "Pedestrian", "Cyclist"),
        help="Evaluate only the selected class(es); defaults to all three.",
    )
    parser.add_argument(
        "--cpu-rotate-iou",
        action="store_true",
        help="Use an OpenCV CPU implementation for rotated IoU instead of Numba CUDA.",
    )
    return parser.parse_args()


def rotated_corners(box: np.ndarray) -> np.ndarray:
    center_x, center_y, width, height, angle = [float(value) for value in box]
    local = np.asarray(
        [
            [-width * 0.5, -height * 0.5],
            [-width * 0.5, height * 0.5],
            [width * 0.5, height * 0.5],
            [width * 0.5, -height * 0.5],
        ],
        dtype=np.float32,
    )
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float32)
    return local @ rotation.T + np.asarray([center_x, center_y], dtype=np.float32)


def cpu_rotate_iou_eval(
    boxes: np.ndarray,
    query_boxes: np.ndarray,
    criterion: int = -1,
    device_id: int = 0,
) -> np.ndarray:
    """Drop-in CPU equivalent of OpenPCDet's rotate_iou_gpu_eval."""
    del device_id
    output = np.zeros((boxes.shape[0], query_boxes.shape[0]), dtype=boxes.dtype)
    if boxes.shape[0] == 0 or query_boxes.shape[0] == 0:
        return output
    box_polygons = [rotated_corners(box) for box in boxes]
    query_polygons = [rotated_corners(box) for box in query_boxes]
    box_areas = boxes[:, 2] * boxes[:, 3]
    query_areas = query_boxes[:, 2] * query_boxes[:, 3]
    box_bounds = [
        (poly[:, 0].min(), poly[:, 0].max(), poly[:, 1].min(), poly[:, 1].max())
        for poly in box_polygons
    ]
    query_bounds = [
        (poly[:, 0].min(), poly[:, 0].max(), poly[:, 1].min(), poly[:, 1].max())
        for poly in query_polygons
    ]
    for row, polygon in enumerate(box_polygons):
        left_min_x, left_max_x, left_min_y, left_max_y = box_bounds[row]
        for column, query_polygon in enumerate(query_polygons):
            right_min_x, right_max_x, right_min_y, right_max_y = query_bounds[column]
            if (
                left_max_x <= right_min_x
                or right_max_x <= left_min_x
                or left_max_y <= right_min_y
                or right_max_y <= left_min_y
            ):
                continue
            intersection, _ = cv2.intersectConvexConvex(polygon, query_polygon)
            intersection = float(max(intersection, 0.0))
            if intersection <= 0:
                continue
            if criterion == -1:
                denominator = float(box_areas[row] + query_areas[column] - intersection)
            elif criterion == 0:
                denominator = float(box_areas[row])
            elif criterion == 1:
                denominator = float(query_areas[column])
            else:
                denominator = 1.0
            output[row, column] = intersection / denominator if denominator > 0 else 0.0
    return output


def main() -> int:
    args = parse_args()
    if args.cpu_rotate_iou:
        kitti_eval.rotate_iou_gpu_eval = cpu_rotate_iou_eval
    with args.result_pkl.open("rb") as handle:
        det_annos = pickle.load(handle)
    with args.infos_pkl.open("rb") as handle:
        infos = pickle.load(handle)
    gt_annos = [copy.deepcopy(info["annos"]) for info in infos]
    if len(gt_annos) != len(det_annos):
        raise ValueError(
            f"annotation count mismatch: gt={len(gt_annos)} det={len(det_annos)}"
        )
    class_names = args.class_name or ["Car", "Pedestrian", "Cyclist"]
    result_str, _ = kitti_eval.get_official_eval_result(
        gt_annos, copy.deepcopy(det_annos), class_names
    )
    args.output_log.parent.mkdir(parents=True, exist_ok=True)
    args.output_log.write_text(
        result_str + "\nEvaluation done.\n", encoding="utf-8"
    )
    print(result_str, flush=True)
    print("Evaluation done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
