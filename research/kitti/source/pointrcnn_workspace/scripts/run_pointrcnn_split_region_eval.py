#!/usr/bin/env python3
"""Run and merge lossless split-region PointRCNN inference.

Each non-empty region is inferred independently with global KITTI coordinates.
Predictions are owned by the disjoint region containing their x/z center, then
merged with class-wise rotated-BEV NMS.  The same ownership and NMS rules are
used for the baseline and every upsampling method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = REPO / "results/pointrcnn_split_region_surface256_v1_20260804"
IMAGESETS = REPO / "data/KITTI/ImageSets"
RECOVERY_RUNNER = REPO / "scripts/run_variant_eval_detector_recovery.py"
SPEC = importlib.util.spec_from_file_location("split_region_recovery_runner", RECOVERY_RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {RECOVERY_RUNNER}")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)
base = recovery.base


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_frames(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def prediction_dir(out_dir: Path, split_name: str) -> Path:
    return out_dir / "inference/eval/epoch_no_number" / split_name / "final_result/data"


def complete_prediction_dir(path: Path, frames: list[str]) -> bool:
    return path.is_dir() and all((path / f"{frame}.txt").is_file() for frame in frames)


def temporary_split_link(split_file: Path, token: str) -> tuple[Path, bool]:
    digest = hashlib.sha256(str(split_file).encode()).hexdigest()[:10]
    link = IMAGESETS / f"splitreg_{token}_{digest}.txt"
    if link.exists() or link.is_symlink():
        if not link.is_symlink() or link.resolve() != split_file.resolve():
            raise RuntimeError(f"refusing to replace existing split path: {link}")
        return link, False
    link.symlink_to(split_file.resolve())
    return link, True


def run_slots(workspace: Path, source: str) -> None:
    split_paths = sorted((workspace / "splits" / source).glob("slot_*.txt"))
    if not split_paths:
        raise FileNotFoundError(f"no split-region slots for {source}")
    token = source.replace("_", "")[:18]
    for position, split_file in enumerate(split_paths, start=1):
        slot = split_file.stem
        frames = read_frames(split_file)
        input_dir = workspace / "inputs" / source / slot
        out_dir = workspace / "detector" / source / "slots" / slot
        link, created = temporary_split_link(split_file, token)
        try:
            pred_dir = prediction_dir(out_dir, link.stem)
            if complete_prediction_dir(pred_dir, frames):
                print(
                    f"SPLIT_REGION_SLOT_REUSE {source} {slot} frames={len(frames)} "
                    f"progress={position}/{len(split_paths)}",
                    flush=True,
                )
                continue
            # An interrupted/crashed slot may contain a valid-looking prefix.
            # Remove only this regenerable inference directory before retrying
            # so stale files cannot satisfy the later coverage check.
            inference_dir = out_dir / "inference"
            if inference_dir.exists():
                shutil.rmtree(inference_dir)
            elapsed, returncode = base.run_inference(
                f"split_region_{source}_{slot}", input_dir, out_dir, link
            )
            pred_dir, produced, empty = base.coverage(out_dir, link)
            if produced != len(frames):
                raise RuntimeError(f"{source}/{slot}: produced {produced}, expected {len(frames)}")
            print(
                f"SPLIT_REGION_SLOT_DONE {source} {slot} frames={len(frames)} "
                f"empty={empty} runtime={elapsed:.1f}s returncode={returncode} "
                f"progress={position}/{len(split_paths)}",
                flush=True,
            )
        finally:
            if created:
                link.unlink(missing_ok=True)


def signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_area(poly: np.ndarray) -> float:
    return abs(signed_area(poly)) if poly.shape[0] >= 3 else 0.0


def line_intersection(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r = p2 - p1
    s = b - a
    denominator = r[0] * s[1] - r[1] * s[0]
    if abs(float(denominator)) < 1e-12:
        return p2
    t = ((a[0] - p1[0]) * s[1] - (a[1] - p1[1]) * s[0]) / denominator
    return p1 + t * r


def polygon_clip(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = subject.copy()
    orientation = 1.0 if signed_area(clip) >= 0 else -1.0
    for index in range(clip.shape[0]):
        a = clip[index]
        b = clip[(index + 1) % clip.shape[0]]
        input_poly = output
        if input_poly.shape[0] == 0:
            break
        points: list[np.ndarray] = []

        def inside(point: np.ndarray) -> bool:
            cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
            return orientation * cross >= -1e-8

        previous = input_poly[-1]
        previous_inside = inside(previous)
        for current in input_poly:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    points.append(line_intersection(previous, current, a, b))
                points.append(current)
            elif previous_inside:
                points.append(line_intersection(previous, current, a, b))
            previous = current
            previous_inside = current_inside
        output = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return output


def bev_corners(fields: list[str]) -> np.ndarray:
    width = float(fields[9])
    length = float(fields[10])
    center_x = float(fields[11])
    center_z = float(fields[13])
    yaw = float(fields[14])
    local = np.asarray(
        [
            [length * 0.5, width * 0.5],
            [-length * 0.5, width * 0.5],
            [-length * 0.5, -width * 0.5],
            [length * 0.5, -width * 0.5],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=np.float64)
    return local @ rotation.T + np.asarray([center_x, center_z])


def bev_iou(left: list[str], right: list[str]) -> float:
    poly_left = bev_corners(left)
    poly_right = bev_corners(right)
    intersection = polygon_area(polygon_clip(poly_left, poly_right))
    if intersection <= 0:
        return 0.0
    union = polygon_area(poly_left) + polygon_area(poly_right) - intersection
    return intersection / union if union > 0 else 0.0


def rotated_nms(rows: list[list[str]], threshold: float) -> list[list[str]]:
    kept: list[list[str]] = []
    for row in sorted(rows, key=lambda item: float(item[15]), reverse=True):
        if all(row[0] != prior[0] or bev_iou(row, prior) <= threshold for prior in kept):
            kept.append(row)
    return kept


def load_regions(path: Path) -> dict[tuple[str, int], tuple[float, float, float, float]]:
    output = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output[(row["frame_id"], int(row["slot"]))] = (
                float(row["x_min_rect_m"]),
                float(row["x_max_rect_m"]),
                float(row["z_min_rect_m"]),
                float(row["z_max_rect_m"]),
            )
    return output


def merge_predictions(workspace: Path, source: str, nms_threshold: float) -> tuple[Path, dict]:
    protocol = read_json(workspace / "protocol.json")
    master_frames = read_frames(Path(protocol["split_file"]))
    regions = load_regions(workspace / "manifests/regions.csv")
    merged_dir = workspace / "detector" / source / "merged_predictions"
    merged_dir.mkdir(parents=True, exist_ok=True)
    candidate_count = owned_count = kept_count = 0

    for frame in master_frames:
        owned: list[list[str]] = []
        frame_regions = sorted((key, value) for key, value in regions.items() if key[0] == frame)
        for (unused_frame, slot), bounds in frame_regions:
            split_file = workspace / "splits" / source / f"slot_{slot:03d}.txt"
            if not split_file.is_file() or frame not in set(read_frames(split_file)):
                continue
            digest = hashlib.sha256(str(split_file).encode()).hexdigest()[:10]
            token = source.replace("_", "")[:18]
            split_name = f"splitreg_{token}_{digest}"
            path = prediction_dir(
                workspace / "detector" / source / "slots" / f"slot_{slot:03d}", split_name
            ) / f"{frame}.txt"
            if not path.is_file():
                raise FileNotFoundError(path)
            x_min, x_max, z_min, z_max = bounds
            for line in path.read_text().splitlines():
                fields = line.split()
                if len(fields) < 16:
                    continue
                candidate_count += 1
                center_x, center_z = float(fields[11]), float(fields[13])
                if x_min <= center_x < x_max and z_min <= center_z < z_max:
                    owned.append(fields)
                    owned_count += 1
        kept = rotated_nms(owned, nms_threshold)
        kept_count += len(kept)
        text = "\n".join(" ".join(row) for row in kept)
        (merged_dir / f"{frame}.txt").write_text(text + ("\n" if text else ""))

    stats = {
        "source": source,
        "frames": len(master_frames),
        "raw_region_predictions": candidate_count,
        "center_owned_predictions": owned_count,
        "post_global_rotated_bev_nms_predictions": kept_count,
        "global_nms_threshold": nms_threshold,
        "ownership": "predicted box center inside disjoint core region",
    }
    (workspace / "detector" / source / "merge_summary.json").write_text(
        json.dumps(stats, indent=2) + "\n"
    )
    return merged_dir, stats


def score_merged(workspace: Path, source: str, merged_dir: Path) -> dict:
    protocol = read_json(workspace / "protocol.json")
    split_file = Path(protocol["split_file"])
    out_dir = workspace / "detector" / source / "evaluation"
    artifacts = recovery.run_cpp_eval_scoped(out_dir, merged_dir, split_file)
    frames = read_frames(split_file)
    empty = sum((merged_dir / f"{frame}.txt").stat().st_size == 0 for frame in frames)
    summary = base.write_summary(
        f"split_region_{source}",
        Path(protocol["sources"][source]),
        out_dir,
        len(frames),
        empty,
        0.0,
        0,
        artifacts,
    )
    summary["split_region_protocol"] = str(workspace / "protocol.json")
    summary["merged_prediction_dir"] = str(merged_dir)
    (out_dir / "result_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source", action="append")
    parser.add_argument("--nms-threshold", type=float, default=0.1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    protocol = read_json(workspace / "protocol.json")
    selected = args.source or list(protocol["sources"])
    unknown = set(selected) - set(protocol["sources"])
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")

    for source in selected:
        if not args.merge_only:
            run_slots(workspace, source)
        merged_dir, merge_stats = merge_predictions(workspace, source, args.nms_threshold)
        summary = score_merged(workspace, source, merged_dir)
        moderate = summary["ap_r40_percent"]["3d_ap"]["moderate"]
        print(
            f"SPLIT_REGION_SOURCE_PASS {source} mod3d={moderate:.6f} "
            f"kept={merge_stats['post_global_rotated_bev_nms_predictions']}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
