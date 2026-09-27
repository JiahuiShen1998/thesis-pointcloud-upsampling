#!/usr/bin/env python3
"""Generate three-frame PointRCNN/CenterPoint visual evidence.

The exporter consumes only completed experiment artifacts.  It does not run
upsampling, detector inference, training, or official evaluation, and it never
modifies the existing result directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analyze_dual_detector_frame_transitions as audit  # noqa: E402
import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402

from lib.utils.calibration import Calibration  # noqa: E402


OUT = audit.DEFAULT_OUTPUT
KITTI = audit.KITTI
WORKSPACE = audit.POINT_WORKSPACE
CENTER_WORKSPACE = audit.CENTER_WORKSPACE
SOURCE = REPO / "results/kitti_unified_x4_current_methods_no_detector"
POINT_AP = WORKSPACE / "reports/e1_e2_live_ap_summary.csv"
CENTER_AP = CENTER_WORKSPACE / "full_ap_summary.csv"
CENTER_GEOMETRY = (
    REPO
    / "results/centerpoint_input_mechanism_analysis_20260729/centerpoint_voxel_summary.csv"
)
RATIO_RESULTS = (
    REPO
    / "results/kitti_x4_detector_recovery_combined_analysis_20260725/combined_ratio_results.csv"
)

COLORS = {
    "baseline": "#8B9098",
    "upsampled": "#1976D2",
    "generated": "#1976D2",
    "observed": "#8B9098",
    "effective": "#2563EB",
    "gt": "#E6AB02",
    "maintained": "#1B9E77",
    "localization_degraded": "#E67E22",
    "confidence_degraded": "#9467BD",
    "lost_after_upsampling": "#D62728",
    "recovered_after_upsampling": "#00A6A6",
    "missed_both": "#7F7F7F",
    "false_positive": "#D62728",
    "ignored_gt": "#A0A0A0",
}
BOX_EDGES = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
DISPLAY_FULL = 45_000
DISPLAY_EFFECTIVE = 26_000
VOXEL_SIZE = np.asarray([0.05, 0.05, 0.1], dtype=np.float32)
POINT_RANGE = np.asarray([0.0, -40.0, -3.0, 70.4, 40.0, 1.0], dtype=np.float32)
MAX_POINTS_PER_VOXEL = 5
MAX_VOXELS = 40_000
MAX_STATIC_CROPS_PER_CASE = 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def display_sample(points: np.ndarray, limit: int, *parts: object) -> np.ndarray:
    if len(points) <= limit:
        return points
    rng = np.random.default_rng(stable_seed(*parts))
    return points[rng.choice(len(points), size=limit, replace=False)]


def exact_row_mask(reference: np.ndarray, points: np.ndarray) -> np.ndarray:
    reference = np.ascontiguousarray(reference, dtype=np.float32)
    points = np.ascontiguousarray(points, dtype=np.float32)
    dtype = np.dtype((np.void, reference.dtype.itemsize * reference.shape[1]))
    reference_keys = np.unique(reference.view(dtype).ravel())
    return np.isin(points.view(dtype).ravel(), reference_keys, assume_unique=False)


def point_paths(line: str, method: str, frame: str) -> dict[str, Path]:
    line_cfg = audit.LINES[line]
    variant = f"{line_cfg['prefix']}_{method}"
    baseline_variant = str(line_cfg["baseline"])
    observed = prep.ORIGINAL if line == "A" else prep.DOWNSAMPLED
    return {
        "observed": observed / f"{frame}.bin",
        "original_reference": prep.ORIGINAL / f"{frame}.bin",
        "upsampled_e1": WORKSPACE / "inputs/e1_default_16384" / variant / f"{frame}.bin",
        "baseline_e2": WORKSPACE
        / "inputs/e2_canonical_16384"
        / baseline_variant
        / f"{frame}.bin",
        "upsampled_e2": WORKSPACE
        / "inputs/e2_canonical_16384"
        / variant
        / f"{frame}.bin",
    }


def centerpoint_fov_range(points: np.ndarray, frame: str) -> np.ndarray:
    calibration = Calibration(str(KITTI / "calib" / f"{frame}.txt"))
    with Image.open(KITTI / "image_2" / f"{frame}.png") as image:
        width, height = image.size
    rect = calibration.lidar_to_rect(points[:, :3])
    image_xy, depth = calibration.rect_to_img(rect)
    valid = (
        (image_xy[:, 0] >= 0.0)
        & (image_xy[:, 0] < width)
        & (image_xy[:, 1] >= 0.0)
        & (image_xy[:, 1] < height)
        & (depth >= 0.0)
        & (points[:, 0] >= POINT_RANGE[0])
        & (points[:, 0] < POINT_RANGE[3])
        & (points[:, 1] >= POINT_RANGE[1])
        & (points[:, 1] < POINT_RANGE[4])
        & (points[:, 2] >= POINT_RANGE[2])
        & (points[:, 2] < POINT_RANGE[5])
    )
    return points[valid]


def centerpoint_effective_points(points: np.ndarray, frame: str) -> tuple[np.ndarray, dict]:
    """Reproduce the order-sensitive first-5/first-40k detector point budget."""
    selected = centerpoint_fov_range(points, frame)
    coords = np.floor((selected[:, :3] - POINT_RANGE[:3]) / VOXEL_SIZE).astype(np.int32)
    voxel_counts: dict[tuple[int, int, int], int] = {}
    retained: list[int] = []
    cap_dropped_new_voxels = 0
    max5_dropped_points = 0
    for idx, coord in enumerate(coords):
        key = (int(coord[0]), int(coord[1]), int(coord[2]))
        count = voxel_counts.get(key)
        if count is None:
            if len(voxel_counts) >= MAX_VOXELS:
                cap_dropped_new_voxels += 1
                continue
            voxel_counts[key] = 1
            retained.append(idx)
        elif count < MAX_POINTS_PER_VOXEL:
            voxel_counts[key] = count + 1
            retained.append(idx)
        else:
            max5_dropped_points += 1
    retained_points = selected[np.asarray(retained, dtype=np.int64)]
    return retained_points, {
        "fov_range_points": len(selected),
        "retained_points": len(retained_points),
        "retained_voxels": len(voxel_counts),
        "voxel_cap_hit": len(voxel_counts) >= MAX_VOXELS and cap_dropped_new_voxels > 0,
        "cap_dropped_new_voxel_points": cap_dropped_new_voxels,
        "max5_dropped_points": max5_dropped_points,
    }


def transition_map(base_audit: dict, up_audit: dict) -> dict[int, str]:
    base = base_audit["match_by_gt"]
    up = up_audit["match_by_gt"]
    output = {}
    for gt_idx in range(len(base_audit["gt"])):
        base_match, up_match = base.get(gt_idx), up.get(gt_idx)
        if base_match and up_match:
            if up_match["iou3d"] - base_match["iou3d"] <= -0.10:
                status = "localization_degraded"
            elif float(up_match["score"] or 0.0) - float(base_match["score"] or 0.0) <= -0.10:
                status = "confidence_degraded"
            else:
                status = "maintained"
        elif base_match:
            status = "lost_after_upsampling"
        elif up_match:
            status = "recovered_after_upsampling"
        else:
            status = "missed_both"
        output[gt_idx] = status
    return output


def prediction_statuses(audit_result: dict, gt_status: dict[int, str]) -> list[str]:
    by_pred = {match["pred_idx"]: match for match in audit_result["matches"]}
    output = []
    for pred_idx in range(len(audit_result["pred"])):
        match = by_pred.get(pred_idx)
        output.append("false_positive" if match is None else gt_status[match["gt_idx"]])
    return output


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def box_center(box: dict) -> np.ndarray:
    return audit.box_center(box)


def crop_points(points: np.ndarray, box: dict, margin: float = 2.0) -> np.ndarray:
    corners = np.asarray(box["corners_lidar"], dtype=np.float64)
    lower = corners.min(axis=0) - margin
    upper = corners.max(axis=0) + margin
    mask = np.all((points[:, :3] >= lower) & (points[:, :3] <= upper), axis=1)
    return points[mask]


def box_dimensions(box: dict) -> tuple[float, float, float]:
    corners = np.asarray(box["corners_lidar"], dtype=np.float64)
    edge_lengths = [
        float(np.linalg.norm(corners[edge[0]] - corners[edge[1]]))
        for edge in ((0, 1), (1, 2), (0, 4))
    ]
    return tuple(edge_lengths)


def draw_box_3d(ax, box: dict, color: str, linewidth: float = 1.2, linestyle: str = "-"):
    corners = np.asarray(box["corners_lidar"], dtype=np.float64)
    for start, end in BOX_EDGES:
        ax.plot(
            corners[[start, end], 0],
            corners[[start, end], 1],
            corners[[start, end], 2],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
        )


def draw_box_bev(ax, box: dict, color: str, linewidth: float = 1.2, linestyle: str = "-"):
    corners = np.asarray(box["corners_lidar"], dtype=np.float64)
    ring = np.vstack((corners[:4, :2], corners[0, :2]))
    ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=linewidth, linestyle=linestyle)


def draw_boxes(
    ax,
    gt_all: list[dict],
    detector_classes: tuple[str, ...],
    audit_result: dict,
    status_by_gt: dict[int, str],
    prediction_status: list[str],
    projection: str,
):
    draw = draw_box_3d if projection == "3d" else draw_box_bev
    eval_gt_by_id = {box["id"]: idx for idx, box in enumerate(audit_result["gt"])}
    for gt_box in gt_all:
        eval_idx = eval_gt_by_id.get(gt_box["id"])
        color = (
            COLORS[status_by_gt[eval_idx]]
            if eval_idx is not None
            else COLORS["ignored_gt"]
        )
        draw(ax, gt_box, color, linewidth=1.0, linestyle="--")
    for pred_idx, pred_box in enumerate(audit_result["pred"]):
        draw(
            ax,
            pred_box,
            COLORS[prediction_status[pred_idx]],
            linewidth=1.45,
            linestyle="-" if prediction_status[pred_idx] != "false_positive" else ":",
        )


def configure_3d(ax, title: str, points: np.ndarray):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x forward (m)", fontsize=7)
    ax.set_ylabel("y left (m)", fontsize=7)
    ax.set_zlabel("z (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    if len(points):
        low, high = np.percentile(points[:, :3], [0.5, 99.5], axis=0)
        ax.set_xlim(max(0.0, low[0]), min(72.0, high[0]))
        ax.set_ylim(max(-40.0, low[1]), min(40.0, high[1]))
        ax.set_zlim(max(-3.5, low[2]), min(2.0, high[2]))
    ax.view_init(elev=22, azim=-105)


def configure_bev(ax, title: str):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x forward (m)", fontsize=7)
    ax.set_ylabel("y left (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_xlim(0, 72)
    ax.set_ylim(-40, 40)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.25, alpha=0.25)


def scatter_3d(ax, points: np.ndarray, color: str, size: float = 0.18, alpha: float = 0.58):
    if len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=color,
            s=size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )


def scatter_bev(ax, points: np.ndarray, color: str, size: float = 0.16, alpha: float = 0.50):
    if len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=color,
            s=size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )


def render_full_cloud(
    output: Path,
    observed: np.ndarray,
    e1: np.ndarray,
    observed_mask: np.ndarray,
    title: str,
):
    observed_display = display_sample(observed, DISPLAY_FULL, title, "observed")
    e1_observed = display_sample(e1[observed_mask], DISPLAY_FULL // 2, title, "e1-observed")
    e1_generated = display_sample(
        e1[~observed_mask], DISPLAY_FULL, title, "e1-generated"
    )
    fig = plt.figure(figsize=(15, 6.5), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    scatter_3d(ax1, observed_display, COLORS["baseline"], 0.16, 0.58)
    configure_3d(ax1, "Observed baseline — full frame", observed_display)
    scatter_3d(ax2, e1_observed, COLORS["observed"], 0.13, 0.45)
    scatter_3d(ax2, e1_generated, COLORS["generated"], 0.14, 0.50)
    configure_3d(ax2, "Exact-4N E1 — observed gray, generated blue", e1)
    fig.suptitle(title, fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def render_detector(
    output: Path,
    detector: str,
    baseline_points: np.ndarray,
    upsampled_points: np.ndarray,
    gt_all: list[dict],
    baseline_audit: dict,
    upsampled_audit: dict,
    title: str,
):
    classes = audit.DETECTOR_CLASSES[detector]
    status = transition_map(baseline_audit, upsampled_audit)
    base_pred_status = prediction_statuses(baseline_audit, status)
    up_pred_status = prediction_statuses(upsampled_audit, status)
    base_display = display_sample(
        baseline_points, DISPLAY_EFFECTIVE, title, detector, "baseline"
    )
    up_display = display_sample(
        upsampled_points, DISPLAY_EFFECTIVE, title, detector, "upsampled"
    )
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4),
    ]
    scatter_3d(axes[0], base_display, COLORS["baseline"], 0.20, 0.56)
    scatter_3d(axes[1], up_display, COLORS["upsampled"], 0.20, 0.55)
    draw_boxes(
        axes[0],
        gt_all,
        classes,
        baseline_audit,
        status,
        base_pred_status,
        "3d",
    )
    draw_boxes(
        axes[1],
        gt_all,
        classes,
        upsampled_audit,
        status,
        up_pred_status,
        "3d",
    )
    configure_3d(
        axes[0],
        f"Baseline detector input ({len(baseline_points):,} retained points)",
        baseline_points,
    )
    configure_3d(
        axes[1],
        f"Upsampled detector input ({len(upsampled_points):,} retained points)",
        upsampled_points,
    )
    scatter_bev(axes[2], base_display, COLORS["baseline"], 0.18, 0.52)
    scatter_bev(axes[3], up_display, COLORS["upsampled"], 0.18, 0.50)
    draw_boxes(
        axes[2],
        gt_all,
        classes,
        baseline_audit,
        status,
        base_pred_status,
        "bev",
    )
    draw_boxes(
        axes[3],
        gt_all,
        classes,
        upsampled_audit,
        status,
        up_pred_status,
        "bev",
    )
    configure_bev(axes[2], "Baseline BEV — every GT and final prediction box")
    configure_bev(axes[3], "Upsampled BEV — every GT and final prediction box")
    counts = Counter(status.values())
    fig.suptitle(
        f"{title}\n"
        f"{detector}: lost={counts['lost_after_upsampling']}, "
        f"recovered={counts['recovered_after_upsampling']}, "
        f"localized↓={counts['localization_degraded']}, "
        f"confidence↓={counts['confidence_degraded']}; "
        "dashed=GT, solid/dotted=final prediction",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def render_crop(
    output: Path,
    baseline_points: np.ndarray,
    upsampled_points: np.ndarray,
    focus_box: dict,
    baseline_audit: dict,
    upsampled_audit: dict,
    gt_all: list[dict],
    detector: str,
    title: str,
):
    baseline_crop = crop_points(baseline_points, focus_box)
    upsampled_crop = crop_points(upsampled_points, focus_box)
    status = transition_map(baseline_audit, upsampled_audit)
    base_pred_status = prediction_statuses(baseline_audit, status)
    up_pred_status = prediction_statuses(upsampled_audit, status)
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    axes = [
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3),
        fig.add_subplot(2, 2, 4),
    ]
    scatter_3d(axes[0], baseline_crop, COLORS["baseline"], 1.0, 0.65)
    scatter_3d(axes[1], upsampled_crop, COLORS["upsampled"], 1.0, 0.62)
    scatter_bev(axes[2], baseline_crop, COLORS["baseline"], 1.0, 0.65)
    scatter_bev(axes[3], upsampled_crop, COLORS["upsampled"], 1.0, 0.62)
    for ax, projection, case_audit, pred_status in (
        (axes[0], "3d", baseline_audit, base_pred_status),
        (axes[1], "3d", upsampled_audit, up_pred_status),
        (axes[2], "bev", baseline_audit, base_pred_status),
        (axes[3], "bev", upsampled_audit, up_pred_status),
    ):
        draw = draw_box_3d if projection == "3d" else draw_box_bev
        draw(ax, focus_box, COLORS["gt"], linewidth=2.0, linestyle="--")
        for pred_idx, pred_box in enumerate(case_audit["pred"]):
            if len(crop_points(np.asarray([box_center(pred_box)]), focus_box, 3.0)):
                draw(
                    ax,
                    pred_box,
                    COLORS[pred_status[pred_idx]],
                    linewidth=1.7,
                    linestyle="-" if pred_status[pred_idx] != "false_positive" else ":",
                )
    corners = np.asarray(focus_box["corners_lidar"])
    lower, upper = corners.min(axis=0) - 2.5, corners.max(axis=0) + 2.5
    for ax, label in ((axes[0], "Baseline 3D crop"), (axes[1], "Upsampled 3D crop")):
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_zlim(lower[2], upper[2])
        ax.view_init(elev=23, azim=-105)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=6)
    for ax, label in ((axes[2], "Baseline BEV crop"), (axes[3], "Upsampled BEV crop")):
        ax.set_xlim(lower[0], upper[0])
        ax.set_ylim(lower[1], upper[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.grid(True, linewidth=0.25, alpha=0.25)
    fig.suptitle(
        f"{title}\npoints in crop: {len(baseline_crop):,} → {len(upsampled_crop):,}",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def inventory_rows(
    detector: str,
    line: str,
    method: str,
    frame: str,
    gt_all: list[dict],
    baseline_audit: dict,
    upsampled_audit: dict,
) -> list[dict]:
    rows = []
    status = transition_map(baseline_audit, upsampled_audit)
    eval_gt_by_id = {box["id"]: idx for idx, box in enumerate(baseline_audit["gt"])}
    for box in gt_all:
        center = box_center(box)
        dims = box_dimensions(box)
        eval_idx = eval_gt_by_id.get(box["id"])
        rows.append(
            {
                "detector": detector,
                "line": line,
                "method": audit.METHOD_LABEL[method],
                "frame_id": frame,
                "state": "ground_truth",
                "box_index": box["id"],
                "class": box["class"],
                "score": "",
                "transition_or_status": (
                    status[eval_idx] if eval_idx is not None else "non_evaluated_gt_class"
                ),
                "matched_gt_index": "" if eval_idx is None else eval_idx,
                "center_x_m": center[0],
                "center_y_m": center[1],
                "center_z_m": center[2],
                "dimension_1_m": dims[0],
                "dimension_2_m": dims[1],
                "dimension_3_m": dims[2],
                "corners_lidar_json": json.dumps(box["corners_lidar"]),
            }
        )
    for state, case_audit in (
        ("baseline_prediction", baseline_audit),
        ("upsampled_prediction", upsampled_audit),
    ):
        pred_status = prediction_statuses(case_audit, status)
        match_by_pred = {match["pred_idx"]: match for match in case_audit["matches"]}
        for pred_idx, box in enumerate(case_audit["pred"]):
            center = box_center(box)
            dims = box_dimensions(box)
            match = match_by_pred.get(pred_idx)
            rows.append(
                {
                    "detector": detector,
                    "line": line,
                    "method": audit.METHOD_LABEL[method],
                    "frame_id": frame,
                    "state": state,
                    "box_index": pred_idx,
                    "class": box["class"],
                    "score": box.get("score", ""),
                    "transition_or_status": pred_status[pred_idx],
                    "matched_gt_index": "" if match is None else match["gt_idx"],
                    "bev_iou": "" if match is None else match["bev_iou"],
                    "iou3d": "" if match is None else match["iou3d"],
                    "center_x_m": center[0],
                    "center_y_m": center[1],
                    "center_z_m": center[2],
                    "dimension_1_m": dims[0],
                    "dimension_2_m": dims[1],
                    "dimension_3_m": dims[2],
                    "corners_lidar_json": json.dumps(box["corners_lidar"]),
                }
            )
    return rows


def parse_centerpoint_all_metrics() -> list[dict]:
    rows = []
    for line, line_cfg in audit.LINES.items():
        variants = [str(line_cfg["baseline"])] + [
            f"{line_cfg['prefix']}_{method}" for method in audit.METHODS
        ]
        for variant in variants:
            logs = sorted(
                (
                    CENTER_WORKSPACE
                    / variant
                    / "full/openpcdet_output/eval/epoch_80/val/frozen_exact4n_e1"
                ).glob("log_eval_*.txt")
            )
            if not logs:
                raise FileNotFoundError(f"missing CenterPoint evaluator log for {variant}")
            text = logs[-1].read_text(encoding="utf-8", errors="replace")
            for class_name, threshold in (("Car", "0.70"), ("Pedestrian", "0.50"), ("Cyclist", "0.50")):
                pattern = re.compile(
                    rf"{class_name} AP_R40@{threshold}, {threshold}, {threshold}:\s*\n"
                    r"bbox AP:([0-9.]+), ([0-9.]+), ([0-9.]+)\s*\n"
                    r"bev\s+AP:([0-9.]+), ([0-9.]+), ([0-9.]+)\s*\n"
                    r"3d\s+AP:([0-9.]+), ([0-9.]+), ([0-9.]+)"
                )
                match = pattern.search(text)
                if not match:
                    raise RuntimeError(f"cannot parse {class_name} AP_R40 from {logs[-1]}")
                values = [float(value) for value in match.groups()]
                for metric, offset in (("bbox", 0), ("bev", 3), ("3d", 6)):
                    rows.append(
                        {
                            "detector": "centerpoint",
                            "variant": variant,
                            "line": (
                                "baseline_" + line
                                if variant == line_cfg["baseline"]
                                else line
                            ),
                            "method": (
                                "baseline"
                                if variant == line_cfg["baseline"]
                                else audit.METHOD_LABEL[
                                    variant.removeprefix(f"{line_cfg['prefix']}_")
                                ]
                            ),
                            "class": class_name,
                            "metric": metric,
                            "easy": values[offset],
                            "moderate": values[offset + 1],
                            "hard": values[offset + 2],
                        }
                    )
    unique = {}
    for row in rows:
        key = (row["detector"], row["variant"], row["class"], row["metric"])
        unique[key] = row
    return list(unique.values())


def parse_pointrcnn_all_metrics() -> list[dict]:
    output = []
    for row in read_csv(POINT_AP):
        if row["experiment"] != "E2":
            continue
        variant = row["variant"]
        line = "A" if variant.startswith("original") else "B"
        line_cfg = audit.LINES[line]
        method = (
            "baseline"
            if variant == line_cfg["baseline"]
            else audit.METHOD_LABEL[variant.removeprefix(f"{line_cfg['prefix']}_")]
        )
        for metric, prefix in (("bbox", "bbox_ap"), ("bev", "bev_ap"), ("3d", "3d_ap")):
            output.append(
                {
                    "detector": "pointrcnn",
                    "variant": variant,
                    "line": "baseline_" + line if method == "baseline" else line,
                    "method": method,
                    "class": "Car",
                    "metric": metric,
                    "easy": float(row[f"{prefix}_easy"]),
                    "moderate": float(row[f"{prefix}_moderate"]),
                    "hard": float(row[f"{prefix}_hard"]),
                }
            )
    return output


def add_metric_deltas(rows: list[dict]) -> list[dict]:
    baseline = {}
    for row in rows:
        if row["method"] == "baseline":
            baseline[(row["detector"], row["line"][-1], row["class"], row["metric"])] = row
    output = []
    for row in rows:
        item = dict(row)
        line = row["line"][-1]
        base = baseline[(row["detector"], line, row["class"], row["metric"])]
        for difficulty in ("easy", "moderate", "hard"):
            item[f"delta_{difficulty}"] = row[difficulty] - base[difficulty]
        output.append(item)
    return output


def plot_ap_deltas(rows: list[dict], output: Path):
    methods = [audit.METHOD_LABEL[method] for method in audit.METHODS]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for row_idx, detector in enumerate(("pointrcnn", "centerpoint")):
        for col_idx, line in enumerate(("A", "B")):
            ax = axes[row_idx, col_idx]
            classes = ["Car"] if detector == "pointrcnn" else ["Car", "Pedestrian", "Cyclist"]
            width = 0.22 if len(classes) == 3 else 0.56
            positions = np.arange(len(methods))
            for class_idx, class_name in enumerate(classes):
                values = [
                    next(
                        row["delta_moderate"]
                        for row in rows
                        if row["detector"] == detector
                        and row["line"] == line
                        and row["method"] == method
                        and row["class"] == class_name
                        and row["metric"] == "3d"
                    )
                    for method in methods
                ]
                offsets = (
                    positions
                    + (class_idx - (len(classes) - 1) / 2.0) * width
                )
                bars = ax.bar(offsets, values, width=width, label=class_name)
                for bar, value in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        value - 1.0 if value < 0 else value + 0.5,
                        f"{value:.1f}",
                        ha="center",
                        va="top" if value < 0 else "bottom",
                        fontsize=7,
                    )
            ax.axhline(0, color="#444444", linewidth=0.7)
            ax.set_xticks(positions, methods, rotation=15)
            ax.set_ylabel("Δ Moderate 3D AP_R40 (points)")
            ax.set_title(f"{detector} — Line {line}")
            ax.grid(axis="y", linewidth=0.35, alpha=0.35)
            if len(classes) > 1:
                ax.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_metric_matrix(rows: list[dict], output: Path):
    method_order = [audit.METHOD_LABEL[method] for method in audit.METHODS]
    labels = []
    values = []
    for detector in ("pointrcnn", "centerpoint"):
        classes = ["Car"] if detector == "pointrcnn" else ["Car", "Pedestrian", "Cyclist"]
        for line in ("A", "B"):
            for class_name in classes:
                for metric in ("bbox", "bev", "3d"):
                    labels.append(f"{detector[:2].upper()} L{line} {class_name[:3]} {metric}")
                    values.append(
                        [
                            next(
                                row["delta_moderate"]
                                for row in rows
                                if row["detector"] == detector
                                and row["line"] == line
                                and row["method"] == method
                                and row["class"] == class_name
                                and row["metric"] == metric
                            )
                            for method in method_order
                        ]
                    )
    matrix = np.asarray(values)
    fig, ax = plt.subplots(figsize=(9, max(8, len(labels) * 0.32)), constrained_layout=True)
    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(method_order)), method_order)
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            ax.text(x, y, f"{matrix[y, x]:.1f}", ha="center", va="center", fontsize=6)
    ax.set_title("Moderate AP_R40 delta vs each line's baseline")
    fig.colorbar(image, ax=ax, label="AP points")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_geometry_relation(metric_rows: list[dict], output: Path):
    geometry = {
        (row["line"], audit.METHOD_LABEL[row["method"]]): row
        for row in read_csv(CENTER_GEOMETRY)
        if row["method"] in audit.METHODS
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    markers = {"PDANS": "o", "PU-GCN": "s", "PU-EdgeFormer": "^", "PU-Net": "X"}
    for ax, line in zip(axes, ("A", "B")):
        for detector in ("pointrcnn", "centerpoint"):
            for method in markers:
                geom_line = "line_a" if line == "A" else "line_b"
                x = float(geometry[(geom_line, method)]["generated_0p2_voxel_precision_median"]) * 100
                y = next(
                    row["delta_moderate"]
                    for row in metric_rows
                    if row["detector"] == detector
                    and row["line"] == line
                    and row["method"] == method
                    and row["class"] == "Car"
                    and row["metric"] == "3d"
                )
                ax.scatter(
                    x,
                    y,
                    marker=markers[method],
                    s=70,
                    label=f"{detector} / {method}",
                    alpha=0.88,
                )
                ax.annotate(
                    f"{detector[:2].upper()} {method}",
                    (x, y),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=7,
                )
        ax.axhline(0, color="#444444", linewidth=0.7)
        ax.set_xlabel("Generated 0.2 m voxel precision to original scan (%)")
        ax.set_ylabel("Car Moderate 3D AP_R40 delta")
        ax.set_title(f"Line {line}: geometric support vs detector loss")
        ax.grid(True, linewidth=0.35, alpha=0.35)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def choose_frames(ranking: list[dict[str, str]]) -> list[str]:
    """Choose one overall, one Pedestrian-rich, and one Cyclist-rich frame."""
    chosen: list[str] = []

    def pick(predicate):
        for row in ranking:
            if row["frame_id"] in chosen:
                continue
            if int(row["pointrcnn_events"]) == 0 or int(row["centerpoint_events"]) == 0:
                continue
            if predicate(row):
                chosen.append(row["frame_id"])
                return

    pick(lambda row: int(row["gt_car"]) >= 2)
    pick(lambda row: int(row["gt_pedestrian"]) >= 1)
    pick(lambda row: int(row["gt_cyclist"]) >= 1)
    pick(lambda row: True)
    pick(lambda row: True)
    return chosen[:3]


def build_case(
    frame: str,
    line: str,
    method: str,
    detector: str,
    centerpoint_matrix: dict,
    full_transition_lookup: dict,
    render: bool = True,
) -> tuple[dict, list[dict], list[dict]]:
    paths = point_paths(line, method, frame)
    observed = prep.read_bin(paths["observed"])
    e1 = prep.read_bin(paths["upsampled_e1"])
    observed_mask = exact_row_mask(observed, e1)
    calib = audit.det.parse_calib(KITTI / "calib" / f"{frame}.txt")
    gt_all = audit.gt_for_frame(frame, calib)
    line_cfg = audit.LINES[line]
    baseline_variant = str(line_cfg["baseline"])
    variant = f"{line_cfg['prefix']}_{method}"
    if detector == "pointrcnn":
        baseline_predictions = audit.point_predictions(frame, baseline_variant, calib)
        upsampled_predictions = audit.point_predictions(frame, variant, calib)
        baseline_effective = prep.read_bin(paths["baseline_e2"])
        upsampled_effective = prep.read_bin(paths["upsampled_e2"])
        baseline_effective_meta = {
            "adapter": "PointRCNN E2 exact 16,384",
            "retained_points": len(baseline_effective),
        }
        upsampled_effective_meta = {
            "adapter": "PointRCNN E2 exact 16,384",
            "retained_points": len(upsampled_effective),
        }
    else:
        baseline_predictions = audit.centerpoint_boxes(
            centerpoint_matrix[baseline_variant][frame], baseline_variant
        )
        upsampled_predictions = audit.centerpoint_boxes(
            centerpoint_matrix[variant][frame], variant
        )
        baseline_effective, baseline_effective_meta = centerpoint_effective_points(
            observed, frame
        )
        upsampled_effective, upsampled_effective_meta = centerpoint_effective_points(
            e1, frame
        )
    classes = audit.DETECTOR_CLASSES[detector]
    baseline_audit = audit.match_gt_predictions(gt_all, baseline_predictions, classes)
    upsampled_audit = audit.match_gt_predictions(gt_all, upsampled_predictions, classes)
    status = transition_map(baseline_audit, upsampled_audit)
    case_dir = (
        OUT
        / "frames"
        / frame
        / f"line_{line.lower()}"
        / method
        / detector
    )
    title = (
        f"Frame {frame} · Line {line} · {audit.METHOD_LABEL[method]} · {detector}"
    )
    if render:
        render_detector(
            case_dir / "full_frame_3d_bev.png",
            detector,
            baseline_effective,
            upsampled_effective,
            gt_all,
            baseline_audit,
            upsampled_audit,
            title,
        )
    crop_rows = []
    changed_all = [
        (gt_idx, state)
        for gt_idx, state in status.items()
        if state not in ("maintained", "missed_both")
    ]
    priority = {
        "lost_after_upsampling": 5,
        "recovered_after_upsampling": 4,
        "localization_degraded": 3,
        "confidence_degraded": 2,
    }
    changed_all.sort(
        key=lambda item: (
            -priority[item[1]],
            float(box_center(baseline_audit["gt"][item[0]])[0]),
            item[0],
        )
    )
    changed = []
    # Prefer class diversity first, then fill the remaining static-crop budget.
    for class_name in audit.DETECTOR_CLASSES[detector]:
        candidate = next(
            (
                item
                for item in changed_all
                if baseline_audit["gt"][item[0]]["class"] == class_name
                and item not in changed
            ),
            None,
        )
        if candidate is not None:
            changed.append(candidate)
        if len(changed) >= MAX_STATIC_CROPS_PER_CASE:
            break
    for item in changed_all:
        if len(changed) >= MAX_STATIC_CROPS_PER_CASE:
            break
        if item not in changed:
            changed.append(item)
    for gt_idx, state in changed:
        focus = baseline_audit["gt"][gt_idx]
        crop_path = case_dir / "object_crops" / f"gt_{gt_idx:03d}_{focus['class']}_{state}.png"
        if render:
            render_crop(
                crop_path,
                baseline_effective,
                upsampled_effective,
                focus,
                baseline_audit,
                upsampled_audit,
                gt_all,
                detector,
                f"{title} · GT {gt_idx} {focus['class']} · {state}",
            )
        crop_rows.append(
            {
                "detector": detector,
                "line": line,
                "method": audit.METHOD_LABEL[method],
                "frame_id": frame,
                "gt_index": gt_idx,
                "class": focus["class"],
                "transition": state,
                "crop_png": str(crop_path.relative_to(OUT)),
            }
        )
    status_rows = [
        row
        for row in full_transition_lookup.get((detector, line, audit.METHOD_LABEL[method], frame), [])
    ]
    case_json = {
        "protocol": {
            "detector": detector,
            "line": line,
            "method": audit.METHOD_LABEL[method],
            "baseline_variant": baseline_variant,
            "upsampled_variant": variant,
            "point_cloud_visualization": "complete observed frame and exact-4N E1; detector view uses detector-specific effective input",
            "box_matching": "greedy same-class oriented 3D IoU; Car 0.70, Pedestrian/Cyclist 0.50",
            "official_metric_boundary": "frame audit is explanatory; official AP is read from completed 3,769-frame evaluation",
        },
        "frame_id": frame,
        "paths": {key: str(value) for key, value in paths.items()},
        "calibration": str(KITTI / "calib" / f"{frame}.txt"),
        "label": str(KITTI / "label_2" / f"{frame}.txt"),
        "baseline_effective_meta": baseline_effective_meta,
        "upsampled_effective_meta": upsampled_effective_meta,
        "observed_points": len(observed),
        "e1_points": len(e1),
        "e1_exact_observed_rows": int(observed_mask.sum()),
        "e1_generated_rows_lower_bound": int((~observed_mask).sum()),
        "gt_all": gt_all,
        "baseline_audit": baseline_audit,
        "upsampled_audit": upsampled_audit,
        "transition_by_gt": status,
        "baseline_prediction_status": prediction_statuses(baseline_audit, status),
        "upsampled_prediction_status": prediction_statuses(upsampled_audit, status),
        "transition_rows": status_rows,
        "visualizations": {
            "full_frame": str((case_dir / "full_frame_3d_bev.png").relative_to(OUT)),
            "object_crops": [row["crop_png"] for row in crop_rows],
        },
    }
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_manifest.json").write_text(
        json.dumps(json_safe(case_json), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    inventory = inventory_rows(
        detector,
        line,
        method,
        frame,
        gt_all,
        baseline_audit,
        upsampled_audit,
    )
    return case_json, inventory, crop_rows


def render_cloud_once(frame: str, line: str, method: str):
    paths = point_paths(line, method, frame)
    observed = prep.read_bin(paths["observed"])
    e1 = prep.read_bin(paths["upsampled_e1"])
    observed_mask = exact_row_mask(observed, e1)
    output = (
        OUT
        / "frames"
        / frame
        / f"line_{line.lower()}"
        / method
        / "pointcloud_full_frame.png"
    )
    render_full_cloud(
        output,
        observed,
        e1,
        observed_mask,
        f"Frame {frame} · Line {line} · {audit.METHOD_LABEL[method]}",
    )


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--frames",
        nargs="*",
        help="Exactly three frame IDs. Omit to select from frame_candidate_ranking.csv.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write metrics/manifests without rendering PNG figures.",
    )
    args = parser.parse_args()
    OUT = args.output
    ranking_path = OUT / "analysis/frame_candidate_ranking.csv"
    transitions_path = OUT / "analysis/all_frame_gt_transitions.csv"
    if not ranking_path.exists() or not transitions_path.exists():
        raise FileNotFoundError(
            "run scripts/analyze_dual_detector_frame_transitions.py first"
        )
    ranking = read_csv(ranking_path)
    frames = args.frames or choose_frames(ranking)
    if len(frames) != 3 or len(set(frames)) != 3:
        raise ValueError(f"expected three distinct frames, got {frames}")
    transitions = read_csv(transitions_path)
    selected_transitions = [row for row in transitions if row["frame_id"] in frames]
    write_csv(OUT / "analysis/selected_frame_gt_transitions.csv", selected_transitions)
    lookup = defaultdict(list)
    for row in selected_transitions:
        lookup[(row["detector"], row["line"], row["method"], row["frame_id"])].append(row)
    metric_rows = add_metric_deltas(
        parse_pointrcnn_all_metrics() + parse_centerpoint_all_metrics()
    )
    write_csv(OUT / "analysis/all_detector_ap_r40_bbox_bev_3d.csv", metric_rows)
    selected_rank = [row for row in ranking if row["frame_id"] in frames]
    write_csv(OUT / "analysis/selected_frame_ranking.csv", selected_rank)
    if not args.skip_plots:
        plot_ap_deltas(metric_rows, OUT / "figures/ap_3d_delta_by_detector_line_class.png")
        plot_metric_matrix(metric_rows, OUT / "figures/ap_bbox_bev_3d_delta_matrix.png")
        plot_geometry_relation(metric_rows, OUT / "figures/geometry_support_vs_car_ap.png")
    centerpoint_matrix = audit.load_centerpoint_matrix()
    inventory: list[dict] = []
    crops: list[dict] = []
    case_index = []
    for frame in frames:
        for line in ("A", "B"):
            for method in audit.METHODS:
                if not args.skip_plots:
                    render_cloud_once(frame, line, method)
                for detector in ("pointrcnn", "centerpoint"):
                    case, case_inventory, case_crops = build_case(
                        frame,
                        line,
                        method,
                        detector,
                        centerpoint_matrix,
                        lookup,
                        render=not args.skip_plots,
                    )
                    inventory.extend(case_inventory)
                    crops.extend(case_crops)
                    case_index.append(
                        {
                            "frame_id": frame,
                            "line": line,
                            "method": audit.METHOD_LABEL[method],
                            "detector": detector,
                            "manifest": str(
                                Path(case["visualizations"]["full_frame"]).parent
                                / "case_manifest.json"
                            ),
                            "full_frame_png": case["visualizations"]["full_frame"],
                            "changed_object_crops": len(case_crops),
                        }
                    )
    write_csv(OUT / "analysis/all_selected_frame_boxes.csv", inventory)
    write_csv(OUT / "analysis/object_crop_index.csv", crops)
    write_csv(OUT / "case_index.csv", case_index)
    (OUT / "selected_frames.txt").write_text("\n".join(frames) + "\n", encoding="utf-8")
    print("selected_frames=" + ",".join(frames))
    print(f"cases={len(case_index)}")
    print(f"box_inventory_rows={len(inventory)}")
    print(f"changed_object_crops={len(crops)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
