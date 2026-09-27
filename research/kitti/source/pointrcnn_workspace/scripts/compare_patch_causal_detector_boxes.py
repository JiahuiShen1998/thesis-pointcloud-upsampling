#!/usr/bin/env python3
"""Classify detector boxes as kept, lost, or added between old/new patch runs."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("kitti_txt", "openpcdet_pkl"), required=True)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bev-iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def rectangle(box: np.ndarray) -> np.ndarray:
    x, y, dx, dy, heading = [float(item) for item in box]
    local = np.asarray(
        [[dx / 2, dy / 2], [-dx / 2, dy / 2], [-dx / 2, -dy / 2], [dx / 2, -dy / 2]],
        dtype=np.float64,
    )
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return local @ rotation.T + np.asarray([x, y])


def cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def segment_line_intersection(
    start: np.ndarray,
    end: np.ndarray,
    clip_a: np.ndarray,
    clip_b: np.ndarray,
) -> np.ndarray:
    direction = end - start
    edge = clip_b - clip_a
    denominator = cross(direction, edge)
    if abs(denominator) < 1e-12:
        return end.copy()
    t = cross(clip_a - start, edge) / denominator
    return start + t * direction


def polygon_clip(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    output = [point.copy() for point in subject]
    for index, clip_a in enumerate(clipper):
        clip_b = clipper[(index + 1) % len(clipper)]
        inputs = output
        output = []
        if not inputs:
            break
        start = inputs[-1]
        for end in inputs:
            end_inside = cross(clip_b - clip_a, end - clip_a) >= -1e-10
            start_inside = cross(clip_b - clip_a, start - clip_a) >= -1e-10
            if end_inside:
                if not start_inside:
                    output.append(segment_line_intersection(start, end, clip_a, clip_b))
                output.append(end.copy())
            elif start_inside:
                output.append(segment_line_intersection(start, end, clip_a, clip_b))
            start = end
    return np.asarray(output, dtype=np.float64)


def polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        float(
            np.dot(points[:, 0], np.roll(points[:, 1], -1))
            - np.dot(points[:, 1], np.roll(points[:, 0], -1))
        )
    ) / 2.0


def bev_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_poly = rectangle(first)
    second_poly = rectangle(second)
    intersection = polygon_area(polygon_clip(first_poly, second_poly))
    union = float(first[2] * first[3] + second[2] * second[3] - intersection)
    return intersection / union if union > 0 else 0.0


def load_txt(path: Path, frames: list[str]) -> dict[str, dict[str, list[tuple[np.ndarray, float]]]]:
    result: dict[str, dict[str, list[tuple[np.ndarray, float]]]] = {}
    for frame in frames:
        classes: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)
        file_path = path / f"{frame}.txt"
        for line in file_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 16:
                continue
            class_name = fields[0]
            h, w, length = [float(item) for item in fields[8:11]]
            x, _, z = [float(item) for item in fields[11:14]]
            heading = float(fields[14])
            score = float(fields[15])
            classes[class_name].append(
                (np.asarray([x, z, length, w, heading], dtype=np.float64), score)
            )
        result[frame] = classes
    return result


def load_pkl(path: Path, frames: list[str]) -> dict[str, dict[str, list[tuple[np.ndarray, float]]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    by_frame = {str(item["frame_id"]): item for item in payload}
    result: dict[str, dict[str, list[tuple[np.ndarray, float]]]] = {}
    for frame in frames:
        if frame not in by_frame:
            raise KeyError(f"{path} has no frame {frame}")
        item = by_frame[frame]
        classes: dict[str, list[tuple[np.ndarray, float]]] = defaultdict(list)
        for class_name, box, score in zip(item["name"], item["boxes_lidar"], item["score"]):
            classes[str(class_name)].append(
                (
                    np.asarray([box[0], box[1], box[3], box[4], box[6]], dtype=np.float64),
                    float(score),
                )
            )
        result[frame] = classes
    return result


def match_boxes(
    old: list[tuple[np.ndarray, float]],
    new: list[tuple[np.ndarray, float]],
    threshold: float,
) -> tuple[int, int, int, list[float], list[float]]:
    if not old:
        return 0, 0, len(new), [], []
    if not new:
        return 0, len(old), 0, [], []
    matrix = np.asarray(
        [[bev_iou(old_box, new_box) for new_box, _ in new] for old_box, _ in old],
        dtype=np.float64,
    )
    old_index, new_index = linear_sum_assignment(1.0 - matrix)
    accepted = [
        (int(i), int(j))
        for i, j in zip(old_index, new_index)
        if matrix[i, j] >= threshold
    ]
    kept = len(accepted)
    ious = [float(matrix[i, j]) for i, j in accepted]
    score_deltas = [float(new[j][1] - old[i][1]) for i, j in accepted]
    return kept, len(old) - kept, len(new) - kept, ious, score_deltas


def main() -> int:
    args = parse_args()
    protocol = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    frames = [str(item) for item in protocol["frame_ids"]]
    loader = load_txt if args.kind == "kitti_txt" else load_pkl
    old = loader(args.old.resolve(), frames)
    new = loader(args.new.resolve(), frames)
    rows: list[dict[str, object]] = []
    all_classes = sorted(
        {
            class_name
            for frame in frames
            for class_name in set(old[frame]) | set(new[frame])
        }
    )
    for frame in frames:
        for class_name in all_classes:
            old_boxes = old[frame].get(class_name, [])
            new_boxes = new[frame].get(class_name, [])
            kept, lost, added, ious, score_deltas = match_boxes(
                old_boxes, new_boxes, args.bev_iou_threshold
            )
            rows.append(
                {
                    "frame_id": frame,
                    "class": class_name,
                    "old_boxes": len(old_boxes),
                    "new_boxes": len(new_boxes),
                    "kept": kept,
                    "lost": lost,
                    "added": added,
                    "kept_bev_iou_mean": float(np.mean(ious)) if ious else float("nan"),
                    "kept_score_delta_mean": (
                        float(np.mean(score_deltas)) if score_deltas else float("nan")
                    ),
                }
            )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_frame_box_changes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for class_name in all_classes:
        selected = [row for row in rows if row["class"] == class_name]
        totals = {
            key: sum(int(row[key]) for row in selected)
            for key in ("old_boxes", "new_boxes", "kept", "lost", "added")
        }
        aggregates.append(
            {
                "class": class_name,
                **totals,
                "kept_fraction_of_old": totals["kept"] / max(totals["old_boxes"], 1),
                "lost_fraction_of_old": totals["lost"] / max(totals["old_boxes"], 1),
                "added_fraction_of_new": totals["added"] / max(totals["new_boxes"], 1),
            }
        )
    with (output / "aggregate_box_changes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
        writer.writeheader()
        writer.writerows(aggregates)
    metadata = {
        "status": "PASS",
        "kind": args.kind,
        "frames": len(frames),
        "old": str(args.old.resolve()),
        "new": str(args.new.resolve()),
        "matching": "same-class Hungarian assignment on rotated BEV IoU",
        "bev_iou_threshold": args.bev_iou_threshold,
        "aggregates": aggregates,
    }
    (output / "box_change_summary.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
