#!/usr/bin/env python3
"""Run paired, lossless region-split evaluation with three-class PointRCNN.

The input workspace is produced by ``prepare_pointrcnn_split_region_inputs.py``.
Every source uses the same disjoint core regions and halo policy.  Predictions
are retained only when their camera-frame x/z center belongs to the core, then
merged with class-wise rotated-BEV NMS before one KITTI three-class evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np

import run_patch_causal_centerpoint_eval as eval_base
import run_patch_causal_openpcdet_pointrcnn_eval as point_runner


REPO = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = (
    REPO
    / "results/openpcdet_pointrcnn_three_class_region_split_pilot256_20260808/prepared"
)
DEFAULT_RESULT_ROOT = (
    REPO / "results/openpcdet_pointrcnn_three_class_region_split_pilot256_20260808"
)
RECOVERY = REPO / "scripts/recover_openpcdet_kitti_eval.py"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_frames(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_regions(path: Path) -> dict[tuple[str, int], tuple[float, float, float, float]]:
    output: dict[tuple[str, int], tuple[float, float, float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output[(row["frame_id"], int(row["slot"]))] = (
                float(row["x_min_rect_m"]),
                float(row["x_max_rect_m"]),
                float(row["z_min_rect_m"]),
                float(row["z_max_rect_m"]),
            )
    return output


def find_result_pickle(run_root: Path) -> Path:
    paths = sorted((run_root / "openpcdet_output").rglob("result.pkl"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one result.pkl below {run_root}, found {paths}")
    return paths[0]


def frame_id(annotation: dict) -> str:
    return str(annotation["frame_id"])


def bev_corners(box: np.ndarray) -> np.ndarray:
    """Return four lidar-frame BEV corners for [x,y,z,dx,dy,dz,heading]."""
    center_x, center_y = float(box[0]), float(box[1])
    length, width = float(box[3]), float(box[4])
    heading = float(box[6])
    local = np.asarray(
        [
            [length * 0.5, width * 0.5],
            [-length * 0.5, width * 0.5],
            [-length * 0.5, -width * 0.5],
            [length * 0.5, -width * 0.5],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(heading), math.sin(heading)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return local @ rotation.T + np.asarray([center_x, center_y])


def signed_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_area(poly: np.ndarray) -> float:
    return abs(signed_area(poly)) if poly.shape[0] >= 3 else 0.0


def line_intersection(
    p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
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
            cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (
                point[0] - a[0]
            )
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


def bev_iou(left: tuple[dict, int], right: tuple[dict, int]) -> float:
    left_anno, left_index = left
    right_anno, right_index = right
    poly_left = bev_corners(left_anno["boxes_lidar"][left_index])
    poly_right = bev_corners(right_anno["boxes_lidar"][right_index])
    intersection = polygon_area(polygon_clip(poly_left, poly_right))
    if intersection <= 0:
        return 0.0
    union = polygon_area(poly_left) + polygon_area(poly_right) - intersection
    return intersection / union if union > 0 else 0.0


def classwise_rotated_nms(
    candidates: list[tuple[dict, int]], threshold: float
) -> list[tuple[dict, int]]:
    ordered = sorted(
        candidates,
        key=lambda item: float(item[0]["score"][item[1]]),
        reverse=True,
    )
    kept: list[tuple[dict, int]] = []
    for candidate in ordered:
        annotation, index = candidate
        name = str(annotation["name"][index])
        if all(
            name != str(prior[0]["name"][prior[1]])
            or bev_iou(candidate, prior) <= threshold
            for prior in kept
        ):
            kept.append(candidate)
    return kept


def merge_annotation(
    frame: str, template: dict, candidates: list[tuple[dict, int]]
) -> dict:
    output: dict = {"frame_id": np.str_(frame)}
    for key, value in template.items():
        if key == "frame_id":
            continue
        array = np.asarray(value)
        if candidates:
            output[key] = np.concatenate(
                [np.asarray(annotation[key])[index : index + 1] for annotation, index in candidates],
                axis=0,
            )
        else:
            output[key] = array[:0].copy()
    return output


def run_slot(
    result_root: Path,
    workspace: Path,
    source: str,
    split_file: Path,
    batch_size: int,
    workers: int,
    resume: bool,
) -> Path:
    slot = split_file.stem
    frames = read_frames(split_file)
    input_dir = workspace / "inputs" / source / slot
    run_root = result_root / "slot_runs" / source / slot
    run_root.mkdir(parents=True, exist_ok=True)
    overlay = eval_base.prepare_overlay(
        run_root, input_dir, eval_base.selected_infos(frames)
    )
    experiment_token = hashlib.sha256(str(result_root).encode("utf-8")).hexdigest()[:8]
    point_runner.run_eval(
        f"rs3_{experiment_token}_{source}_{slot}",
        run_root,
        overlay,
        batch_size,
        workers,
        resume,
    )
    return find_result_pickle(run_root)


def merge_source(
    result_root: Path,
    workspace: Path,
    source: str,
    master_frames: list[str],
    regions: dict[tuple[str, int], tuple[float, float, float, float]],
    batch_size: int,
    workers: int,
    resume: bool,
    nms_threshold: float,
) -> dict:
    split_paths = sorted((workspace / "splits" / source).glob("slot_*.txt"))
    if not split_paths:
        raise FileNotFoundError(f"no split files for {source}")

    by_frame_slot: dict[tuple[str, int], dict] = {}
    for position, split_file in enumerate(split_paths, start=1):
        slot = int(split_file.stem.split("_")[-1])
        result_path = run_slot(
            result_root,
            workspace,
            source,
            split_file,
            batch_size,
            workers,
            resume,
        )
        with result_path.open("rb") as handle:
            annotations = pickle.load(handle)
        expected = read_frames(split_file)
        if [frame_id(item) for item in annotations] != expected:
            raise RuntimeError(f"prediction order/coverage mismatch for {source}/{split_file.stem}")
        for annotation in annotations:
            by_frame_slot[(frame_id(annotation), slot)] = annotation
        print(
            f"REGION_SLOT_READY source={source} slot={slot:03d} "
            f"frames={len(annotations)} progress={position}/{len(split_paths)}",
            flush=True,
        )

    merged: list[dict] = []
    raw_predictions = owned_predictions = kept_predictions = 0
    for frame in master_frames:
        frame_items = sorted(
            (key, bounds) for key, bounds in regions.items() if key[0] == frame
        )
        candidates: list[tuple[dict, int]] = []
        template: dict | None = None
        for (_, slot), bounds in frame_items:
            annotation = by_frame_slot.get((frame, slot))
            if annotation is None:
                continue
            template = annotation
            x_min, x_max, z_min, z_max = bounds
            raw_predictions += len(annotation["name"])
            for index, location in enumerate(annotation["location"]):
                center_x, center_z = float(location[0]), float(location[2])
                if x_min <= center_x < x_max and z_min <= center_z < z_max:
                    candidates.append((annotation, index))
                    owned_predictions += 1
        if template is None:
            raise RuntimeError(f"no region annotation template for {source}/{frame}")
        kept = classwise_rotated_nms(candidates, nms_threshold)
        kept_predictions += len(kept)
        merged.append(merge_annotation(frame, template, kept))

    source_root = result_root / "merged" / source
    source_root.mkdir(parents=True, exist_ok=True)
    merged_pickle = source_root / "result.pkl"
    with merged_pickle.open("wb") as handle:
        pickle.dump(merged, handle)
    infos_pickle = source_root / "kitti_infos_val.pkl"
    with infos_pickle.open("wb") as handle:
        pickle.dump(eval_base.selected_infos(master_frames), handle)
    eval_log = source_root / "official_eval.txt"
    command = [
        str(point_runner.PYTHON),
        str(RECOVERY),
        "--result-pkl",
        str(merged_pickle),
        "--infos-pkl",
        str(infos_pickle),
        "--output-log",
        str(eval_log),
        "--cpu-rotate-iou",
    ]
    recovery_result: subprocess.CompletedProcess | None = None
    for attempt in range(1, 4):
        recovery_result = subprocess.run(command, cwd=REPO, check=False)
        if recovery_result.returncode == 0:
            break
        if recovery_result.returncode != -11:
            break
        print(
            f"RECOVERY_RETRY source={source} attempt={attempt}/3 returncode=-11",
            flush=True,
        )
        time.sleep(2)
    if recovery_result is None or recovery_result.returncode != 0:
        code = None if recovery_result is None else recovery_result.returncode
        raise RuntimeError(f"official KITTI evaluation failed for {source}: {code}")
    metrics = eval_base.parse_metrics(eval_log.read_text(encoding="utf-8"))
    summary = {
        "status": "PASS",
        "source": source,
        "frames": len(master_frames),
        "virtual_regions": len(by_frame_slot),
        "raw_region_predictions": raw_predictions,
        "center_owned_predictions": owned_predictions,
        "post_classwise_rotated_bev_nms_predictions": kept_predictions,
        "nms_threshold": nms_threshold,
        "point_sampling": "all region points retained; OpenPCDet repeats to 16384 when needed",
        "metrics_percent": metrics,
        "result_pickle": str(merged_pickle.resolve()),
        "official_eval_log": str(eval_log.resolve()),
    }
    (source_root / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def write_comparison(result_root: Path, summaries: dict[str, dict]) -> None:
    baseline = summaries["original_baseline"]["metrics_percent"]
    rows: list[dict[str, object]] = []
    method_names = [name for name in summaries if name != "original_baseline"]
    for method_name in method_names:
        method = summaries[method_name]["metrics_percent"]
        for class_name in ("Car", "Pedestrian", "Cyclist"):
            for metric in ("3d_ap_r40", "bev_ap_r40"):
                for difficulty in ("easy", "moderate", "hard"):
                    before = float(baseline[class_name][metric][difficulty])
                    after = float(method[class_name][metric][difficulty])
                    rows.append(
                        {
                            "method": method_name,
                            "class": class_name,
                            "metric": metric,
                            "difficulty": difficulty,
                            "baseline": before,
                            "method_ap": after,
                            "delta": after - before,
                        }
                    )
    csv_path = result_root / "paired_region_split_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    moderate = [row for row in rows if row["difficulty"] == "moderate"]
    lines = [
        "# OpenPCDet PointRCNN 三类别成对 Region-Split 对比",
        "",
        "Baseline 与所有方法使用完全相同的核心区域、3 m halo、检测器和合并规则。",
        "核心区域内点不被 16,384 点上限截断；低于上限时由原生 PointRCNN 重复采样补满。",
        "",
        "## AP_R40 Moderate（%）",
        "",
        "| Method | Class | Metric | Baseline | Method | Δ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in moderate:
        lines.append(
            f"| {row['method']} | {row['class']} | "
            f"{str(row['metric']).replace('_ap_r40', '').upper()} | "
            f"{float(row['baseline']):.4f} | {float(row['method_ap']):.4f} | "
            f"{float(row['delta']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "该表的 Δ 只比较同一 region-split 推理协议下各方法与 baseline；不能与整帧原生采样结果直接作差。",
            "",
        ]
    )
    (result_root / "PAIRED_REGION_SPLIT_THREE_CLASS_ZH.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--source", action="append")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--nms-threshold", type=float, default=0.1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    result_root = args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    protocol = read_json(workspace / "protocol.json")
    master_frames = read_frames(Path(protocol["split_file"]))
    regions = load_regions(workspace / "manifests/regions.csv")
    available_sources = list(protocol["sources"])
    sources = args.source or available_sources
    unknown = sorted(set(sources) - set(available_sources))
    if unknown:
        raise ValueError(f"sources are not present in the region protocol: {unknown}")
    summaries: dict[str, dict] = {}
    for source in sources:
        summaries[source] = merge_source(
            result_root,
            workspace,
            source,
            master_frames,
            regions,
            args.batch_size,
            args.workers,
            args.resume,
            args.nms_threshold,
        )
    if "original_baseline" in summaries and len(summaries) > 1:
        write_comparison(result_root, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
