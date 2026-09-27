#!/usr/bin/env python3
"""20-frame KITTI AP dry evaluation for existing PointRCNN outputs."""

from __future__ import annotations

import csv
import math
import shutil
import sys
import traceback
import types
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_ROOT = PROJECT_ROOT / "results" / "pointrcnn_20frame_default_fair_comparison"
SOURCE_FRAME_LIST = PROJECT_ROOT / "results" / "pointrcnn_larger_default_validation_cap100k" / "shared_20_frame_list.txt"
DRY_SPLIT_FILE = RESULT_ROOT / "shared_20_val.txt"
LABEL_DIR = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "label_2"
AP_ROOT = RESULT_ROOT / "ap_eval"
SPLIT_NAME = "val_pugcn_cap100k_20"

METHODS = [
    ("Original KITTI", "original", "original_default"),
    ("EAR", "ear", "ear_default"),
    ("PU-Net", "punet", "punet_default"),
    ("PU-GCN cap_100k", "pugcn_cap100k", "pugcn_cap100k_default"),
]


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
    """CPU replacement for evaluator rotated IoU, preserving 3D criterion=2 behavior."""
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

    def safe_get_split_parts(num: int, num_part: int) -> List[int]:
        num_part = max(1, min(int(num_part), int(num)))
        same_part = num // num_part
        remain_num = num % num_part
        parts = [same_part] * num_part
        if remain_num:
            parts.append(remain_num)
        return [part for part in parts if part > 0]

    eval_mod.get_split_parts = safe_get_split_parts


def read_frame_ids() -> List[str]:
    return [line.strip() for line in SOURCE_FRAME_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]


def detection_dir(run_name: str) -> Path:
    return RESULT_ROOT / run_name / "eval" / "epoch_no_number" / SPLIT_NAME / "test_mode" / "final_result" / "data"


def count_detection_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def check_inputs(frame_ids: Sequence[str], det_dir: Path) -> Dict[str, object]:
    missing_labels = []
    missing_detections = []
    total_detections = 0
    for frame_id in frame_ids:
        if not (LABEL_DIR / f"{frame_id}.txt").exists():
            missing_labels.append(frame_id)
        det_file = det_dir / f"{frame_id}.txt"
        if not det_file.exists():
            missing_detections.append(frame_id)
        else:
            total_detections += count_detection_lines(det_file)
    return {
        "missing_labels": missing_labels,
        "missing_detections": missing_detections,
        "total_detections": total_detections,
    }


def fmt_list(values: Sequence[str]) -> str:
    return ",".join(values) if values else "none"


def run_eval(method_name: str, output_name: str, run_name: str, frame_ids: Sequence[str]) -> Dict[str, object]:
    from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate

    out_dir = AP_ROOT / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    det_dir = detection_dir(run_name)
    checks = check_inputs(frame_ids, det_dir)
    row: Dict[str, object] = {
        "method": method_name,
        "output_name": output_name,
        "evaluated_frames": len(frame_ids),
        "detection_dir": str(det_dir),
        "total_detections": checks["total_detections"],
        "missing_labels": fmt_list(checks["missing_labels"]),
        "missing_detection_files": fmt_list(checks["missing_detections"]),
        "status": "pending",
        "warnings": "",
    }
    if checks["missing_labels"] or checks["missing_detections"]:
        row["status"] = "missing_inputs"
        row["warnings"] = "Missing labels or detection files."
        return row

    try:
        ap_result_str, ap_dict = kitti_evaluate(str(LABEL_DIR), str(det_dir), label_split_file=str(DRY_SPLIT_FILE), current_class=0)
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
        (out_dir / "raw_eval_output.txt").write_text(ap_result_str, encoding="utf-8")
    except Exception as exc:
        row["status"] = "failed"
        row["warnings"] = repr(exc)
        (out_dir / "error_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return row


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
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


def ap_fmt(row: Dict[str, object], key: str) -> str:
    value = row.get(key, "")
    if value == "":
        return ""
    return f"{float(value):.4f}"


def frame_detection_counts(frame_ids: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for frame_id in frame_ids:
        row: Dict[str, object] = {"frame_id": frame_id}
        for method_name, _, run_name in METHODS:
            row[method_name] = count_detection_lines(detection_dir(run_name) / f"{frame_id}.txt")
        rows.append(row)
    return rows


def write_markdown(rows: Sequence[Dict[str, object]], frame_rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# 20-Frame AP Dry Evaluation",
        "",
        "This is a 20-frame subset AP dry evaluation for evaluator feasibility and preliminary AP-like comparison only.",
        "It is not final KITTI validation AP and should not be interpreted as full validation performance.",
        "",
        f"Frame list: `{SOURCE_FRAME_LIST.relative_to(PROJECT_ROOT)}`",
        f"Temporary subset split: `{DRY_SPLIT_FILE.relative_to(PROJECT_ROOT)}`",
        f"Label directory: `{LABEL_DIR.relative_to(PROJECT_ROOT)}`",
        f"Inference output root: `{RESULT_ROOT.relative_to(PROJECT_ROOT)}`",
        "",
        "Evaluator: `tools/kitti_object_eval_python/evaluate.py` with an in-memory CPU rotated-IoU patch, matching the existing AP recovery approach used elsewhere in this workspace.",
        "",
        "## AP Summary",
        "",
        "| Method | Status | Frames | Detections | bbox Easy | bbox Mod | bbox Hard | BEV Easy | BEV Mod | BEV Hard | 3D Easy | 3D Mod | 3D Hard | Missing labels | Missing detections | Warnings |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {status} | {evaluated_frames} | {total_detections} | {bbox_easy} | {bbox_mod} | {bbox_hard} | "
            "{bev_easy} | {bev_mod} | {bev_hard} | {d3_easy} | {d3_mod} | {d3_hard} | {missing_labels} | "
            "{missing_detection_files} | {warnings} |".format(
                method=row["method"],
                status=row["status"],
                evaluated_frames=row["evaluated_frames"],
                total_detections=row["total_detections"],
                bbox_easy=ap_fmt(row, "bbox_AP_easy"),
                bbox_mod=ap_fmt(row, "bbox_AP_moderate"),
                bbox_hard=ap_fmt(row, "bbox_AP_hard"),
                bev_easy=ap_fmt(row, "bev_AP_easy"),
                bev_mod=ap_fmt(row, "bev_AP_moderate"),
                bev_hard=ap_fmt(row, "bev_AP_hard"),
                d3_easy=ap_fmt(row, "3d_AP_easy"),
                d3_mod=ap_fmt(row, "3d_AP_moderate"),
                d3_hard=ap_fmt(row, "3d_AP_hard"),
                missing_labels=row["missing_labels"],
                missing_detection_files=row["missing_detection_files"],
                warnings=row["warnings"],
            )
        )

    lines.extend(
        [
            "",
            "## Watch Frames",
            "",
            "| Frame | Original detections | EAR detections | PU-Net detections | PU-GCN cap_100k detections | Note |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    notes = {
        "000004": "PU-GCN had a large detection drop in inference-only comparison.",
        "000006": "PU-GCN had a large detection drop in inference-only comparison.",
        "000035": "PU-GCN had the largest detection drop in inference-only comparison.",
    }
    for row in frame_rows:
        if row["frame_id"] in notes:
            lines.append(
                "| `{frame_id}` | {orig} | {ear} | {punet} | {pugcn} | {note} |".format(
                    frame_id=row["frame_id"],
                    orig=row["Original KITTI"],
                    ear=row["EAR"],
                    punet=row["PU-Net"],
                    pugcn=row["PU-GCN cap_100k"],
                    note=notes[row["frame_id"]],
                )
            )

    status_ok = all(row["status"] == "success" for row in rows)
    totals = {row["method"]: row["total_detections"] for row in rows}
    pugcn = next(row for row in rows if row["method"] == "PU-GCN cap_100k")
    original = next(row for row in rows if row["method"] == "Original KITTI")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"Evaluation pipeline works on the 20-frame subset: **{'yes' if status_ok else 'no'}**.",
            "PU-GCN cap_100k is evaluator-compatible: **%s**." % ("yes" if pugcn["status"] == "success" else "no"),
            (
                "Detection totals entering the evaluator were Original `{Original KITTI}`, EAR `{EAR}`, "
                "PU-Net `{PU-Net}`, PU-GCN cap_100k `{PU-GCN cap_100k}`."
            ).format(**totals),
            (
                "Preliminary 3D AP Moderate on this tiny subset was Original `{orig:.4f}` vs PU-GCN cap_100k `{pugcn:.4f}`."
            ).format(orig=float(original.get("3d_AP_moderate", 0.0)), pugcn=float(pugcn.get("3d_AP_moderate", 0.0))),
            "Because this is only 20 frames, the AP values are noisy and should be used to validate the pipeline, not to make final claims.",
            "Recommended next step: run a 50-frame validation subset before full validation AP, unless time requires jumping directly to full validation.",
            "",
        ]
    )
    (RESULT_ROOT / "20frame_ap_dry_eval_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame_ids = read_frame_ids()
    AP_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_FRAME_LIST, DRY_SPLIT_FILE)
    patch_evaluator()
    rows = [run_eval(method_name, output_name, run_name, frame_ids) for method_name, output_name, run_name in METHODS]
    frame_rows = frame_detection_counts(frame_ids)
    write_csv(RESULT_ROOT / "20frame_ap_dry_eval_summary.csv", rows)
    write_csv(RESULT_ROOT / "20frame_ap_dry_eval_frame_detection_counts.csv", frame_rows)
    write_markdown(rows, frame_rows)
    for row in rows:
        print(row["method"], row["status"], row.get("3d_AP_moderate", ""))


if __name__ == "__main__":
    main()
