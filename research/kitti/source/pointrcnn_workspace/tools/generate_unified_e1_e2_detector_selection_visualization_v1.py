#!/usr/bin/env python3
"""Generate a fixed-frame E1 -> E2 -> PointRCNN decision visualization.

The package uses the current unified x4 E1/E2 experiment, not the older
method-specific visualization inputs.  Every method and both experiment lines
use the same two frames:

* 000002: one Moderate Car at about 34 m (sparse/distant-object case)
* 000152: seven Cars spanning about 6--47 m (mixed near/far multi-car case)

For each method/line/frame the exporter shows:

1. the observed baseline cloud;
2. the exact-x4 E1 cloud (observed versus generated rows);
3. the exact 16,384-point E2 detector input;
4. baseline and upsampled PointRCNN final decisions;
5. a focused object crop with the original KITTI scan as reference.

The E2 observed/generated split is conservative.  Any selected XYZI row that
exactly equals an observed row is attributed to the observed set, so the
reported observed fraction is an upper bound and the generated fraction is a
lower bound.  Per-frame box association is an explanatory audit, not a
replacement for the official KITTI evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import math
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline.offline import get_plotlyjs
from plotly.subplots import make_subplots


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_e2_observed_retention as retention  # noqa: E402
import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402


def load_detection_helpers():
    path = ROOT / "tools/generate_pointrcnn_detection_box_visualization_v1.py"
    spec = importlib.util.spec_from_file_location("detection_vis_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import detector visualization helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


det = load_detection_helpers()

WORKSPACE = ROOT / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
SOURCE = ROOT / "results/kitti_unified_x4_current_methods_no_detector"
TRAINING = ROOT / "data/KITTI/object/training"
OUT = ROOT / "results/unified_e1_e2_detector_selection_visualization_v1"
FRAMES = ("000002", "000152")
METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
METHOD_LABEL = {
    "pdans": "PDANS",
    "pu_gcn": "PU-GCN",
    "pu_edgeformer": "PU-EdgeFormer",
    "pu_net": "PU-Net",
}
LINES = {
    "A": {
        "label": "Line A: original baseline vs original + x4 upsampling",
        "short": "Original -> x4",
        "observed": prep.ORIGINAL,
        "baseline": "original_baseline",
        "prefix": "original_x4",
    },
    "B": {
        "label": "Line B: downsampled baseline vs downsampled + x4 upsampling",
        "short": "Downsampled -> x4",
        "observed": prep.DOWNSAMPLED,
        "baseline": "downsampled_x4_baseline",
        "prefix": "downsampled_x4",
    },
}
GEOMETRY_SUMMARY = WORKSPACE / "reports/geometry_root_cause_v1/method_line_geometry_summary.csv"
OBJECT_METRICS = WORKSPACE / "reports/geometry_root_cause_v1/object_level_metrics.csv"
AP_SUMMARY = WORKSPACE / "reports/e1_e2_live_ap_summary.csv"
ASSOC_CENTER_M = 2.0
CAR_IOU3D_THRESHOLD = 0.70
DISPLAY_MAX_POINTS = 30000
DISPLAY_MAX_3D_FULL = 45000
DISPLAY_MAX_3D_GENERATED = 60000
COLORS = {
    "observed": "#235789",
    "generated": "#F28E2B",
    "selected": "#171717",
    "reference": "#B8BEC8",
    "gt": "#F2C94C",
    "tp": "#1B9E77",
    "localized": "#D97706",
    "fp": "#D62728",
    "baseline_pred": "#3B82F6",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def fmt(value, digits: int = 2) -> str:
    if value in (None, ""):
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}"


def fnum(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def row_keys(points: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(points, dtype=np.float32)
    dtype = np.dtype((np.void, values.dtype.itemsize * values.shape[1]))
    return values.view(dtype).ravel()


def observed_membership_mask(observed: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Conservative exact-row observed mask (observed fraction upper bound)."""
    observed_keys = np.unique(row_keys(observed))
    return np.isin(row_keys(points), observed_keys, assume_unique=False)


def sample_points(points: np.ndarray, max_points: int, *seed_parts: object) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(stable_seed(*seed_parts))
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def sigmoid(value) -> float | None:
    score = fnum(value)
    if not math.isfinite(score):
        return None
    return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score)))))


def pred_dir(variant: str) -> Path:
    return (
        WORKSPACE
        / "eval_outputs/e2"
        / variant
        / "inference/eval/epoch_no_number/val/final_result/data"
    )


def load_manifest_row(variant: str, frame: str) -> dict[str, str]:
    path = WORKSPACE / "manifests/e2_canonical_16384" / f"{variant}.csv"
    for row in read_csv(path):
        if row["frame_id"] == frame:
            return row
    raise KeyError(f"manifest row missing: {variant}/{frame}")


def polygon_xy(box: dict) -> list[tuple[float, float]]:
    points = [(float(x), float(y)) for x, y, _ in box["corners_lidar"][:4]]
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    points.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return points


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    return abs(
        sum(
            points[i][0] * points[(i + 1) % len(points)][1]
            - points[(i + 1) % len(points)][0] * points[i][1]
            for i in range(len(points))
        )
        / 2.0
    )


def cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def line_intersection(a, b, p, q):
    x1, y1 = a
    x2, y2 = b
    x3, y3 = p
    x4, y4 = q
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-10:
        return b
    first = x1 * y2 - y1 * x2
    second = x3 * y4 - y3 * x4
    return (
        (first * (x3 - x4) - (x1 - x2) * second) / denominator,
        (first * (y3 - y4) - (y1 - y2) * second) / denominator,
    )


def clip_polygon(subject, clip):
    output = list(subject)
    for index, clip_a in enumerate(clip):
        clip_b = clip[(index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        start = input_points[-1]
        for end in input_points:
            end_inside = cross(clip_a, clip_b, end) >= -1e-9
            start_inside = cross(clip_a, clip_b, start) >= -1e-9
            if end_inside:
                if not start_inside:
                    output.append(line_intersection(start, end, clip_a, clip_b))
                output.append(end)
            elif start_inside:
                output.append(line_intersection(start, end, clip_a, clip_b))
            start = end
    return output


def oriented_iou(box_a: dict, box_b: dict) -> tuple[float, float]:
    poly_a = polygon_xy(box_a)
    poly_b = polygon_xy(box_b)
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    intersection_area = polygon_area(clip_polygon(poly_a, poly_b))
    bev_union = area_a + area_b - intersection_area
    bev_iou = intersection_area / bev_union if bev_union > 0 else 0.0

    z_a = [corner[2] for corner in box_a["corners_lidar"]]
    z_b = [corner[2] for corner in box_b["corners_lidar"]]
    height_overlap = max(0.0, min(max(z_a), max(z_b)) - max(min(z_a), min(z_b)))
    intersection_volume = intersection_area * height_overlap
    volume_a = area_a * (max(z_a) - min(z_a))
    volume_b = area_b * (max(z_b) - min(z_b))
    volume_union = volume_a + volume_b - intersection_volume
    iou3d = intersection_volume / volume_union if volume_union > 0 else 0.0
    return bev_iou, iou3d


def box_center(box: dict) -> np.ndarray:
    return np.asarray(box["corners_lidar"], dtype=np.float64).mean(axis=0)


def associate(gt_boxes: list[dict], pred_boxes: list[dict]) -> dict:
    gt = [box for box in gt_boxes if box["class"] == "Car"]
    pred = [box for box in pred_boxes if box["class"] == "Car"]
    candidates = []
    for gi, gt_box in enumerate(gt):
        for pi, pred_box in enumerate(pred):
            distance = float(np.linalg.norm(box_center(gt_box) - box_center(pred_box)))
            if distance <= ASSOC_CENTER_M:
                bev_iou, iou3d = oriented_iou(gt_box, pred_box)
                candidates.append((distance, -iou3d, gi, pi, bev_iou, iou3d))
    candidates.sort()
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches = []
    for distance, _, gi, pi, bev_iou, iou3d in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append(
            {
                "gt_idx": gi,
                "pred_idx": pi,
                "center_distance_m": distance,
                "bev_iou": bev_iou,
                "iou3d": iou3d,
                "score": pred[pi].get("score"),
                "iou3d_pass_0p7": iou3d >= CAR_IOU3D_THRESHOLD,
            }
        )
    strict_tp = sum(match["iou3d_pass_0p7"] for match in matches)
    return {
        "gt": gt,
        "pred": pred,
        "matches": matches,
        "matched_gt": sorted(used_gt),
        "matched_pred": sorted(used_pred),
        "missed_gt": [idx for idx in range(len(gt)) if idx not in used_gt],
        "extra_pred": [idx for idx in range(len(pred)) if idx not in used_pred],
        "associated": len(matches),
        "extra": len(pred) - len(matches),
        "missed": len(gt) - len(matches),
        "strict_tp_0p7": strict_tp,
        "strict_fp_0p7": len(pred) - strict_tp,
        "strict_fn_0p7": len(gt) - strict_tp,
    }


def match_map(audit: dict) -> dict[int, dict]:
    return {int(match["gt_idx"]): match for match in audit["matches"]}


def choose_focus_object(base_audit: dict, up_audit: dict) -> tuple[int, str]:
    base = match_map(base_audit)
    up = match_map(up_audit)
    best: tuple[tuple, int, str] | None = None
    for gi, gt_box in enumerate(base_audit["gt"]):
        depth = float(box_center(gt_box)[0])
        if gi in base and gi not in up:
            key = (5, fnum(base[gi].get("score"), 0.0), depth)
            reason = "baseline-associated object becomes missed after upsampling"
        elif gi in base and gi in up and base[gi]["iou3d_pass_0p7"] and not up[gi]["iou3d_pass_0p7"]:
            key = (4, base[gi]["iou3d"] - up[gi]["iou3d"], depth)
            reason = "upsampled prediction falls below the 0.70 Car 3D-IoU reference threshold"
        elif gi in base and gi in up:
            score_drop = fnum(base[gi].get("score"), 0.0) - fnum(up[gi].get("score"), 0.0)
            iou_drop = base[gi]["iou3d"] - up[gi]["iou3d"]
            key = (3, max(score_drop, 0.0) + 3.0 * max(iou_drop, 0.0), depth)
            reason = "same object is retained but localization or ranking score weakens"
        elif gi not in base and gi not in up:
            key = (1, depth, 0.0)
            reason = "object is missed by both inputs"
        else:
            key = (2, depth, 0.0)
            reason = "upsampled input detects an object missed by baseline"
        if best is None or key > best[0]:
            best = (key, gi, reason)
    if best is None:
        return 0, "no Car GT object available"
    return best[1], best[2]


def box_xy(box: dict) -> tuple[list[float], list[float]]:
    polygon = polygon_xy(box)
    polygon.append(polygon[0])
    return [p[1] for p in polygon], [p[0] for p in polygon]


def crop_mask(points: np.ndarray, box: dict, margin: float = 2.5) -> np.ndarray:
    corners = np.asarray(box["corners_lidar"], dtype=np.float64)
    min_xyz = corners.min(axis=0) - margin
    max_xyz = corners.max(axis=0) + margin
    return np.all((points[:, :3] >= min_xyz) & (points[:, :3] <= max_xyz), axis=1)


def plot_points(ax, points: np.ndarray, color: str, label: str, size=1.2, alpha=0.65, marker="."):
    if points.size:
        ax.scatter(points[:, 1], points[:, 0], s=size, c=color, alpha=alpha, marker=marker, label=label, linewidths=0)


def draw_box(ax, box: dict, color: str, label: str | None = None, linestyle="-", linewidth=1.6):
    lateral, forward = box_xy(box)
    ax.plot(lateral, forward, color=color, linestyle=linestyle, linewidth=linewidth, label=label)


def configure_full_axis(ax, title: str):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlim(-40, 40)
    ax.set_ylim(0, 70.4)
    ax.set_xlabel("LiDAR lateral y (m)", fontsize=8)
    ax.set_ylabel("LiDAR forward x (m)", fontsize=8)
    ax.grid(True, alpha=0.16, linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)


def draw_decisions(ax, points, audit, title: str):
    display = sample_points(points, DISPLAY_MAX_POINTS, "decision", title)
    plot_points(ax, display, "#7D8795", "E2 points", 0.8, 0.35)
    match_by_pred = {int(item["pred_idx"]): item for item in audit["matches"]}
    match_by_gt = match_map(audit)
    for gi, gt_box in enumerate(audit["gt"]):
        draw_box(ax, gt_box, COLORS["gt"], "GT Car" if gi == 0 else None, linewidth=2.0)
        center = box_center(gt_box)
        if gi not in match_by_gt:
            ax.text(center[1], center[0], f"GT{gi} MISS", color=COLORS["fp"], fontsize=7, weight="bold")
    for pi, pred_box in enumerate(audit["pred"]):
        match = match_by_pred.get(pi)
        if match is None:
            color, linestyle, status = COLORS["fp"], "--", "extra"
        elif match["iou3d_pass_0p7"]:
            color, linestyle, status = COLORS["tp"], "-", "IoU pass"
        else:
            color, linestyle, status = COLORS["localized"], "-.", "IoU<.70"
        draw_box(ax, pred_box, color, f"prediction: {status}" if pi == 0 else None, linestyle, 1.8)
        center = box_center(pred_box)
        score = pred_box.get("score")
        iou = match["iou3d"] if match else None
        text = f"P{pi} s={fmt(score, 1)}" + (f" iou={iou:.2f}" if iou is not None else "")
        ax.text(center[1], center[0], text, color=color, fontsize=6.5, weight="bold")
    configure_full_axis(ax, title)


def add_plotly_points(fig, row, col, points, name, color, size, opacity, symbol="circle", showlegend=True):
    display = sample_points(points, DISPLAY_MAX_POINTS, row, col, name)
    fig.add_trace(
        go.Scattergl(
            x=display[:, 1] if display.size else [],
            y=display[:, 0] if display.size else [],
            mode="markers",
            marker={"size": size, "color": color, "opacity": opacity, "symbol": symbol},
            name=name,
            showlegend=showlegend,
            hovertemplate="y=%{x:.2f} m<br>x=%{y:.2f} m<extra>" + html.escape(name) + "</extra>",
        ),
        row=row,
        col=col,
    )


def add_plotly_box(fig, row, col, box, name, color, dash="solid", width=3, showlegend=True, text=None):
    lateral, forward = box_xy(box)
    fig.add_trace(
        go.Scattergl(
            x=lateral,
            y=forward,
            mode="lines" if not text else "lines+text",
            line={"color": color, "width": width, "dash": dash},
            name=name,
            showlegend=showlegend,
            text=text,
            hovertemplate=html.escape(name) + "<extra></extra>",
        ),
        row=row,
        col=col,
    )


def add_plotly_decisions(fig, row, col, points, audit, prefix):
    add_plotly_points(fig, row, col, points, f"{prefix} E2 points", "#7D8795", 2, 0.35, showlegend=False)
    for gi, gt_box in enumerate(audit["gt"]):
        add_plotly_box(fig, row, col, gt_box, f"GT Car {gi}", COLORS["gt"], width=4, showlegend=(gi == 0))
    match_by_pred = {int(item["pred_idx"]): item for item in audit["matches"]}
    for pi, pred_box in enumerate(audit["pred"]):
        match = match_by_pred.get(pi)
        if match is None:
            color, dash, status = COLORS["fp"], "dash", "extra prediction"
        elif match["iou3d_pass_0p7"]:
            color, dash, status = COLORS["tp"], "solid", "3D IoU >= 0.70"
        else:
            color, dash, status = COLORS["localized"], "dashdot", "associated, 3D IoU < 0.70"
        label = f"{prefix} P{pi}: {status}; score={fmt(pred_box.get('score'), 2)}"
        if match:
            label += f"; 3D IoU={match['iou3d']:.3f}"
        add_plotly_box(fig, row, col, pred_box, label, color, dash=dash, width=3, showlegend=False)


def add_plotly_3d_points(
    fig,
    row,
    col,
    points,
    name,
    color,
    size=1.5,
    opacity=0.55,
    symbol="circle",
    max_points=DISPLAY_MAX_3D_FULL,
    showlegend=True,
    legendgroup=None,
):
    """Add a deterministic display-LOD 3D XYZI layer in LiDAR coordinates."""
    display = sample_points(points, max_points, "3d", row, col, name)
    customdata = display[:, 3] if display.size and display.shape[1] > 3 else np.zeros(display.shape[0])
    fig.add_trace(
        go.Scatter3d(
            x=display[:, 0] if display.size else [],
            y=display[:, 1] if display.size else [],
            z=display[:, 2] if display.size else [],
            customdata=customdata,
            mode="markers",
            marker={"size": size, "color": color, "opacity": opacity, "symbol": symbol},
            name=name,
            legendgroup=legendgroup or name,
            showlegend=showlegend,
            hovertemplate=(
                "forward x=%{x:.3f} m<br>lateral y=%{y:.3f} m<br>height z=%{z:.3f} m"
                "<br>intensity=%{customdata:.3f}<extra>" + html.escape(name) + "</extra>"
            ),
        ),
        row=row,
        col=col,
    )


def box_edges_xyz(box: dict) -> tuple[list, list, list]:
    corners = np.asarray(box["corners_lidar"], dtype=np.float64)
    if corners.shape[0] < 8:
        raise ValueError(f"3D box requires 8 corners, got {corners.shape}")
    edges = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
    xs, ys, zs = [], [], []
    for start, end in edges:
        xs.extend((corners[start, 0], corners[end, 0], None))
        ys.extend((corners[start, 1], corners[end, 1], None))
        zs.extend((corners[start, 2], corners[end, 2], None))
    return xs, ys, zs


def add_plotly_3d_box(
    fig,
    row,
    col,
    box,
    name,
    color,
    width=5,
    dash="solid",
    showlegend=True,
    hover_text=None,
    center_text=None,
    legendgroup=None,
):
    xs, ys, zs = box_edges_xyz(box)
    detail = hover_text or name
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line={"color": color, "width": width, "dash": dash},
            name=name,
            legendgroup=legendgroup or name,
            showlegend=showlegend,
            text=[detail] * len(xs),
            hovertemplate="%{text}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    if center_text:
        center = box_center(box)
        fig.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2] + 1.0],
                mode="text",
                text=[center_text],
                textfont={"color": color, "size": 11},
                name=center_text,
                legendgroup=legendgroup or name,
                showlegend=False,
                hovertext=[detail],
                hovertemplate="%{hovertext}<extra></extra>",
            ),
            row=row,
            col=col,
        )


def prediction_audit_detail(prefix: str, pi: int, pred_box: dict, match: dict | None) -> tuple[str, str, str, str]:
    raw_score = pred_box.get("score")
    probability = sigmoid(raw_score)
    if match is None:
        color, dash, offline_status = COLORS["fp"], "dash", "unmatched to Car GT within 2 m"
        association = "none"
        metric = "3D IoU: n/a"
    elif match["iou3d_pass_0p7"]:
        color, dash, offline_status = COLORS["tp"], "solid", "associated; explanatory IoU>=0.70 pass"
        association = f"GT{match['gt_idx']} at {match['center_distance_m']:.3f} m center distance"
        metric = f"BEV IoU={match['bev_iou']:.3f}; 3D IoU={match['iou3d']:.3f}"
    else:
        color, dash, offline_status = COLORS["localized"], "dashdot", "associated; explanatory IoU<0.70"
        association = f"GT{match['gt_idx']} at {match['center_distance_m']:.3f} m center distance"
        metric = f"BEV IoU={match['bev_iou']:.3f}; 3D IoU={match['iou3d']:.3f}"
    probability_text = fmt(probability, 3)
    short = f"{prefix} P{pi} logit={fmt(raw_score, 2)}"
    hover = (
        f"<b>{html.escape(prefix)} final output P{pi}</b><br>"
        "detector stage: survived RCNN score filtering and NMS<br>"
        f"raw RCNN logit={fmt(raw_score, 4)}; sigmoid={probability_text}<br>"
        f"offline GT association: {association}<br>{metric}<br>"
        f"offline audit status: {offline_status}"
    )
    return color, dash, short, hover


def add_plotly_3d_boxes_and_decisions(fig, row, col, audit, prefix, showlegend=True):
    match_by_gt = match_map(audit)
    match_by_pred = {int(item["pred_idx"]): item for item in audit["matches"]}
    for gi, gt_box in enumerate(audit["gt"]):
        gt_status = "matched by offline 2 m association" if gi in match_by_gt else "MISS: no final prediction within offline 2 m association"
        add_plotly_3d_box(
            fig,
            row,
            col,
            gt_box,
            f"GT Car ({prefix})",
            COLORS["gt"],
            width=6,
            showlegend=showlegend and gi == 0,
            hover_text=f"<b>GT Car {gi}</b><br>{gt_status}",
            center_text=f"GT{gi} MISS" if gi not in match_by_gt else None,
            legendgroup=f"gt-{prefix}",
        )
    for pi, pred_box in enumerate(audit["pred"]):
        color, dash, short, hover = prediction_audit_detail(prefix, pi, pred_box, match_by_pred.get(pi))
        add_plotly_3d_box(
            fig,
            row,
            col,
            pred_box,
            f"{prefix} final predictions",
            color,
            width=6,
            dash=dash,
            showlegend=showlegend and pi == 0,
            hover_text=hover,
            center_text=None,
            legendgroup=f"prediction-{prefix}",
        )


def configure_scene(fig, row, col, x_range, y_range, z_range, aspectratio):
    axis_common = {"showbackground": True, "backgroundcolor": "#F7F8FA", "gridcolor": "#D8DEE8", "zerolinecolor": "#AEB7C5"}
    fig.update_scenes(
        xaxis={**axis_common, "title": "forward x (m)", "range": x_range},
        yaxis={**axis_common, "title": "lateral y (m)", "range": y_range},
        zaxis={**axis_common, "title": "height z (m)", "range": z_range},
        aspectmode="manual",
        aspectratio=aspectratio,
        camera={"eye": {"x": 1.55, "y": 1.55, "z": 0.85}, "up": {"x": 0, "y": 0, "z": 1}},
        row=row,
        col=col,
    )


def decision_table_html(audit: dict, label: str) -> str:
    match_by_pred = {int(item["pred_idx"]): item for item in audit["matches"]}
    rows = []
    for pi, pred_box in enumerate(audit["pred"]):
        match = match_by_pred.get(pi)
        raw_score = pred_box.get("score")
        if match is None:
            association, distance, bev_iou, iou3d, audit_status = "none", "n/a", "n/a", "n/a", "unmatched prediction"
        else:
            association = f"GT{match['gt_idx']}"
            distance = fmt(match["center_distance_m"], 3)
            bev_iou = fmt(match["bev_iou"], 3)
            iou3d = fmt(match["iou3d"], 3)
            audit_status = "IoU>=0.70" if match["iou3d_pass_0p7"] else "IoU<0.70"
        rows.append(
            "<tr>"
            f"<td>P{pi}</td><td>final output</td><td>{fmt(raw_score, 4)}</td><td>{fmt(sigmoid(raw_score), 3)}</td>"
            f"<td>{association}</td><td>{distance}</td><td>{bev_iou}</td><td>{iou3d}</td><td>{audit_status}</td>"
            "</tr>"
        )
    for gi in audit["missed_gt"]:
        rows.append(f"<tr class='miss'><td>GT{gi}</td><td>no associated final box</td><td colspan='6'>—</td><td>MISS</td></tr>")
    summary = audit_summary(audit)
    table_body = "".join(rows) if rows else '<tr><td colspan="9">No final Car prediction</td></tr>'
    return (
        f"<section class='decision'><h3>{html.escape(label)}</h3>"
        f"<p>GT={summary['gt_car']}, final predictions={summary['prediction_car']}, associated/extra/miss="
        f"{summary['associated']}/{summary['extra']}/{summary['missed']}, explanatory TP/FP/FN@0.70="
        f"{summary['strict_tp_0p7']}/{summary['strict_fp_0p7']}/{summary['strict_fn_0p7']}</p>"
        "<div class='table-wrap'><table><thead><tr><th>item</th><th>detector state</th><th>raw logit</th><th>sigmoid</th>"
        "<th>offline GT</th><th>center d (m)</th><th>BEV IoU</th><th>3D IoU</th><th>offline audit</th></tr></thead>"
        f"<tbody>{table_body}</tbody></table></div></section>"
    )


def plotly_document(title: str, figure_html: str, body_html: str, plotly_src: str, sync_scenes: tuple[str, ...] = ()) -> str:
    scene_json = json.dumps(list(sync_scenes))
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><script src="{plotly_src}"></script>
<style>
body{{margin:0;background:#eef1f5;color:#172033;font-family:Inter,Arial,sans-serif}}main{{max-width:1900px;margin:auto;background:white}}
.guide,.note,.decision{{margin:0;padding:14px 24px;border-top:1px solid #d8dee8}}.guide{{background:#eaf5f8}}.note{{background:#fff8e8;color:#493f25}}
.decisions{{display:grid;grid-template-columns:1fr 1fr;gap:0;border-top:1px solid #d8dee8}}.decision{{border-top:0}}h3{{margin:4px 0 8px}}p{{line-height:1.45}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px;border:1px solid #d8dee8;text-align:left;white-space:nowrap}}th{{background:#f1f4f8}}.miss{{background:#fff0f0}}
code{{background:#edf0f4;padding:1px 4px;border-radius:3px}}a{{color:#086f83}}@media(max-width:1000px){{.decisions{{grid-template-columns:1fr}}}}
</style></head><body><main>{figure_html}{body_html}</main>
<script>
(function(){{
 const plot=document.querySelector('.plotly-graph-div'); const scenes={scene_json}; let syncing=false;
 if(!plot || scenes.length<2) return;
 plot.on('plotly_relayout', function(event){{
   if(syncing) return;
   const source=scenes.find(scene=>Object.keys(event).some(key=>key.indexOf(scene+'.camera')===0));
   if(!source || !plot.layout[source] || !plot.layout[source].camera) return;
   syncing=true; const update={{}};
   scenes.forEach(scene=>{{if(scene!==source) update[scene+'.camera']=plot.layout[source].camera;}});
   Plotly.relayout(plot, update).then(()=>{{syncing=false;}}).catch(()=>{{syncing=false;}});
 }});
}})();
</script></body></html>"""


def ap_lookup() -> dict[str, dict[str, float]]:
    output = {}
    for row in read_csv(AP_SUMMARY):
        if row["experiment"] != "E2":
            continue
        output[row["variant"]] = {
            "easy": fnum(row["3d_ap_easy"]),
            "moderate": fnum(row["3d_ap_moderate"]),
            "hard": fnum(row["3d_ap_hard"]),
        }
    return output


def geometry_lookup():
    method_rows = {(row["line"], row["method"]): row for row in read_csv(GEOMETRY_SUMMARY)}
    object_rows = {
        (row["line"], row["method"], row["frame_id"], int(row["object_index"])): row
        for row in read_csv(OBJECT_METRICS)
    }
    return method_rows, object_rows


def selection_metrics(observed: np.ndarray, observed_valid: np.ndarray, observed_rect: np.ndarray, e1_valid: np.ndarray, e2: np.ndarray, frame: str, manifest: dict) -> dict:
    exact_mask = observed_membership_mask(observed_valid, e2)
    unique_retained = retention.multiset_intersection_count(observed_valid, e2)
    _, e2_rect = prep.fov_valid_mask(e2, frame)
    return {
        "observed_total": int(observed.shape[0]),
        "observed_fov": int(observed_valid.shape[0]),
        "e1_total": int(manifest["source_points"]),
        "e1_fov": int(e1_valid.shape[0]),
        "e2_total": int(e2.shape[0]),
        "e2_exact_observed_upper": int(exact_mask.sum()),
        "e2_exact_observed_fraction_upper": float(exact_mask.mean()),
        "e2_generated_fraction_lower": float(1.0 - exact_mask.mean()),
        "e2_unique_observed_retained_upper": int(unique_retained),
        "e2_observed_point_retention_upper": float(unique_retained / max(observed_valid.shape[0], 1)),
        "e2_observed_voxel_0p1_coverage": retention.voxel_coverage(observed_rect, e2_rect),
        "e2_voxel_unique_points": int(manifest["voxel_unique_points"]),
        "fill_policy": manifest["fill_policy"],
    }


def audit_summary(audit: dict) -> dict:
    matched_scores = [fnum(item.get("score")) for item in audit["matches"] if math.isfinite(fnum(item.get("score")))]
    matched_ious = [item["iou3d"] for item in audit["matches"]]
    return {
        "gt_car": len(audit["gt"]),
        "prediction_car": len(audit["pred"]),
        "associated": audit["associated"],
        "extra": audit["extra"],
        "missed": audit["missed"],
        "strict_tp_0p7": audit["strict_tp_0p7"],
        "strict_fp_0p7": audit["strict_fp_0p7"],
        "strict_fn_0p7": audit["strict_fn_0p7"],
        "mean_associated_iou3d": float(np.mean(matched_ious)) if matched_ious else None,
        "mean_associated_score": float(np.mean(matched_scores)) if matched_scores else None,
    }


def diagnosis(base_audit: dict, up_audit: dict, selection: dict, geometry: dict, object_metric: dict | None, ap_delta: float, method: str, line: str) -> str:
    base = audit_summary(base_audit)
    up = audit_summary(up_audit)
    parts = []
    if up["missed"] > base["missed"]:
        parts.append(f"associated misses rise {base['missed']} -> {up['missed']}")
    if up["strict_tp_0p7"] < base["strict_tp_0p7"]:
        parts.append(f"3D-IoU>=0.70 matches fall {base['strict_tp_0p7']} -> {up['strict_tp_0p7']}")
    if up["extra"] > base["extra"]:
        parts.append(f"extra predictions rise {base['extra']} -> {up['extra']}")
    if up["mean_associated_score"] is not None and base["mean_associated_score"] is not None and up["mean_associated_score"] < base["mean_associated_score"]:
        parts.append(f"mean associated score falls {base['mean_associated_score']:.2f} -> {up['mean_associated_score']:.2f}")
    if not parts:
        parts.append("this frame has no single count-level failure; inspect box localization and score changes")
    geometry_bits = [
        f"E2 contains at least {100 * selection['e2_generated_fraction_lower']:.1f}% non-observed rows",
        f"global generated-voxel precision is {100 * fnum(geometry.get('voxel_0p20_generated_reference_precision_median')):.1f}%",
        f"global E1 extra-voxel fraction is {100 * fnum(geometry.get('voxel_0p20_e1_extra_fraction_median')):.1f}%",
    ]
    if object_metric:
        geometry_bits.append(
            f"focus-object generated near-surface fraction is {100 * fnum(object_metric.get('generated_inside_near_reference_0p25_fraction')):.1f}%"
        )
    if line == "A":
        mechanism = "Line A already has the complete observed scan; generated rows can only add redundant/estimated geometry and also displace observed rows in E2."
    else:
        mechanism = "Line B gains some surface coverage, but unsupported/off-surface geometry dominates the detector neighborhood change; the 16,384 cap is not the main cause."
    if method == "pu_net":
        mechanism += " Current PU-Net output is additionally invalidated by the known missing normalization/inverse-transform wrapper defect."
    return "; ".join(parts) + ". " + "; ".join(geometry_bits) + f". Full-val Moderate AP delta is {ap_delta:+.2f}. " + mechanism


def create_static(case: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(24, 15), facecolor="#F6F7F9")
    observed_disp = sample_points(case["observed_valid"], DISPLAY_MAX_POINTS, case["variant"], case["frame"], "obs")
    e1_observed_disp = sample_points(case["e1_valid"][case["e1_observed_mask"]], DISPLAY_MAX_POINTS, case["variant"], case["frame"], "e1obs")
    e1_generated_disp = sample_points(case["e1_valid"][~case["e1_observed_mask"]], DISPLAY_MAX_POINTS, case["variant"], case["frame"], "e1gen")
    e2_observed = case["e2"][case["e2_observed_mask"]]
    e2_generated = case["e2"][~case["e2_observed_mask"]]

    plot_points(axes[0, 0], observed_disp, COLORS["observed"], "observed baseline", 1.0, 0.55)
    configure_full_axis(axes[0, 0], f"1. Observed baseline (FOV): {len(case['observed_valid']):,} points")

    plot_points(axes[0, 1], e1_observed_disp, COLORS["observed"], "exact observed rows", 0.9, 0.50)
    plot_points(axes[0, 1], e1_generated_disp, COLORS["generated"], "generated rows", 0.8, 0.35)
    configure_full_axis(axes[0, 1], f"2. E1 exact x4 (FOV): {len(case['e1_valid']):,} valid points")

    plot_points(axes[0, 2], e2_observed, COLORS["observed"], "observed upper bound", 1.3, 0.70)
    plot_points(axes[0, 2], e2_generated, COLORS["generated"], "generated lower bound", 1.1, 0.55)
    configure_full_axis(
        axes[0, 2],
        "3. E2 selected 16,384: "
        f"obs<= {100 * case['selection']['e2_exact_observed_fraction_upper']:.1f}% / "
        f"gen>= {100 * case['selection']['e2_generated_fraction_lower']:.1f}%",
    )

    draw_decisions(
        axes[1, 0],
        case["baseline_e2"],
        case["baseline_audit"],
        "4. Baseline PointRCNN final decisions\n"
        f"assoc/extra/miss={case['base_summary']['associated']}/{case['base_summary']['extra']}/{case['base_summary']['missed']}; "
        f"TP/FP/FN@0.70={case['base_summary']['strict_tp_0p7']}/{case['base_summary']['strict_fp_0p7']}/{case['base_summary']['strict_fn_0p7']}",
    )
    draw_decisions(
        axes[1, 1],
        case["e2"],
        case["up_audit"],
        "5. Upsampled PointRCNN final decisions\n"
        f"assoc/extra/miss={case['up_summary']['associated']}/{case['up_summary']['extra']}/{case['up_summary']['missed']}; "
        f"TP/FP/FN@0.70={case['up_summary']['strict_tp_0p7']}/{case['up_summary']['strict_fp_0p7']}/{case['up_summary']['strict_fn_0p7']}",
    )

    focus_ax = axes[1, 2]
    focus = case["focus_gt"]
    reference_crop = case["reference"][crop_mask(case["reference"], focus)]
    observed_crop = case["observed"][crop_mask(case["observed"], focus)]
    generated_crop = case["e1_valid"][~case["e1_observed_mask"]]
    generated_crop = generated_crop[crop_mask(generated_crop, focus)]
    e2_crop = case["e2"][crop_mask(case["e2"], focus)]
    plot_points(focus_ax, reference_crop, COLORS["reference"], "original KITTI reference", 5, 0.25)
    plot_points(focus_ax, observed_crop, COLORS["observed"], "observed baseline", 6, 0.55)
    plot_points(focus_ax, generated_crop, COLORS["generated"], "E1 generated", 5, 0.45)
    plot_points(focus_ax, e2_crop, COLORS["selected"], "E2 selected", 8, 0.55, marker="x")
    draw_box(focus_ax, focus, COLORS["gt"], "focus GT", linewidth=2.5)
    base_match = match_map(case["baseline_audit"]).get(case["focus_idx"])
    up_match = match_map(case["up_audit"]).get(case["focus_idx"])
    if base_match:
        draw_box(focus_ax, case["baseline_audit"]["pred"][base_match["pred_idx"]], COLORS["baseline_pred"], "baseline pred", "--", 2.0)
    if up_match:
        color = COLORS["tp"] if up_match["iou3d_pass_0p7"] else COLORS["localized"]
        draw_box(focus_ax, case["up_audit"]["pred"][up_match["pred_idx"]], color, "upsampled pred", "-", 2.0)
    corners = np.asarray(focus["corners_lidar"])
    focus_ax.set_xlim(corners[:, 1].min() - 3.0, corners[:, 1].max() + 3.0)
    focus_ax.set_ylim(corners[:, 0].min() - 3.0, corners[:, 0].max() + 3.0)
    focus_ax.set_aspect("equal", adjustable="box")
    focus_ax.grid(True, alpha=0.18)
    focus_ax.set_xlabel("LiDAR lateral y (m)")
    focus_ax.set_ylabel("LiDAR forward x (m)")
    focus_ax.set_title(f"6. Focus GT {case['focus_idx']}: {case['focus_reason']}", fontsize=10, fontweight="bold")

    handles, labels = [], []
    for ax in axes.ravel():
        hnd, lbl = ax.get_legend_handles_labels()
        for item, label in zip(hnd, lbl):
            if label not in labels:
                handles.append(item)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=min(7, len(labels)), frameon=False, fontsize=9)
    fig.suptitle(
        f"{case['method_label']} | {case['line_label']} | frame {case['frame']}\n"
        f"E2 3D AP_R40 Moderate {case['method_ap']:.2f} vs baseline {case['baseline_ap']:.2f} ({case['ap_delta']:+.2f})",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(0.02, 0.035, case["diagnosis"], fontsize=9.5, color="#303641", wrap=True)
    fig.text(
        0.02,
        0.012,
        "Association: same-class center distance <=2 m. TP/FP/FN@0.70 uses oriented 3D IoU as an explanatory reference; official KITTI AP remains the full-validation result.",
        fontsize=8.5,
        color="#5B6472",
    )
    fig.tight_layout(rect=(0.015, 0.065, 0.99, 0.95), w_pad=1.5, h_pad=2.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_interactive_bev(case: dict, path: Path) -> None:
    titles = [
        "1. Observed baseline (FOV)",
        "2. E1 exact x4: observed vs generated",
        "3. E2 selected 16,384",
        "4. Baseline final decisions",
        "5. Upsampled final decisions",
        f"6. Focus GT {case['focus_idx']}",
    ]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, horizontal_spacing=0.055, vertical_spacing=0.10)
    add_plotly_points(fig, 1, 1, case["observed_valid"], "observed baseline", COLORS["observed"], 2, 0.55)
    add_plotly_points(fig, 1, 2, case["e1_valid"][case["e1_observed_mask"]], "E1 exact observed", COLORS["observed"], 2, 0.55)
    add_plotly_points(fig, 1, 2, case["e1_valid"][~case["e1_observed_mask"]], "E1 generated", COLORS["generated"], 2, 0.38)
    add_plotly_points(fig, 1, 3, case["e2"][case["e2_observed_mask"]], "E2 observed upper", COLORS["observed"], 2.5, 0.72)
    add_plotly_points(fig, 1, 3, case["e2"][~case["e2_observed_mask"]], "E2 generated lower", COLORS["generated"], 2.3, 0.58)
    add_plotly_decisions(fig, 2, 1, case["baseline_e2"], case["baseline_audit"], "baseline")
    add_plotly_decisions(fig, 2, 2, case["e2"], case["up_audit"], "upsampled")

    focus = case["focus_gt"]
    reference_crop = case["reference"][crop_mask(case["reference"], focus)]
    observed_crop = case["observed"][crop_mask(case["observed"], focus)]
    generated = case["e1_valid"][~case["e1_observed_mask"]]
    generated_crop = generated[crop_mask(generated, focus)]
    e2_crop = case["e2"][crop_mask(case["e2"], focus)]
    add_plotly_points(fig, 2, 3, reference_crop, "original reference crop", COLORS["reference"], 3, 0.25)
    add_plotly_points(fig, 2, 3, observed_crop, "observed crop", COLORS["observed"], 4, 0.55)
    add_plotly_points(fig, 2, 3, generated_crop, "generated crop", COLORS["generated"], 4, 0.48)
    add_plotly_points(fig, 2, 3, e2_crop, "E2 selected crop", COLORS["selected"], 5, 0.65, symbol="x")
    add_plotly_box(fig, 2, 3, focus, "focus GT", COLORS["gt"], width=4)
    base_match = match_map(case["baseline_audit"]).get(case["focus_idx"])
    up_match = match_map(case["up_audit"]).get(case["focus_idx"])
    if base_match:
        add_plotly_box(fig, 2, 3, case["baseline_audit"]["pred"][base_match["pred_idx"]], "focus baseline prediction", COLORS["baseline_pred"], "dash", 3)
    if up_match:
        color = COLORS["tp"] if up_match["iou3d_pass_0p7"] else COLORS["localized"]
        add_plotly_box(fig, 2, 3, case["up_audit"]["pred"][up_match["pred_idx"]], "focus upsampled prediction", color, "solid", 3)

    for row in (1, 2):
        for col in (1, 2, 3):
            fig.update_xaxes(title_text="lateral y (m)", row=row, col=col, showgrid=True, gridcolor="#E3E6EB", zeroline=False)
            fig.update_yaxes(title_text="forward x (m)", row=row, col=col, showgrid=True, gridcolor="#E3E6EB", zeroline=False, scaleanchor=f"x{(row - 1) * 3 + col}" if not (row == 1 and col == 1) else "x")
            if not (row == 2 and col == 3):
                fig.update_xaxes(range=[-40, 40], row=row, col=col)
                fig.update_yaxes(range=[0, 70.4], row=row, col=col)
    corners = np.asarray(focus["corners_lidar"])
    fig.update_xaxes(range=[corners[:, 1].min() - 3, corners[:, 1].max() + 3], row=2, col=3)
    fig.update_yaxes(range=[corners[:, 0].min() - 3, corners[:, 0].max() + 3], row=2, col=3)
    fig.update_layout(
        title={
            "text": html.escape(
                f"{case['method_label']} | {case['line_label']} | frame {case['frame']} | Moderate AP delta {case['ap_delta']:+.2f}"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        height=1120,
        width=1750,
        legend={"orientation": "h", "y": -0.09, "x": 0.0},
        margin={"l": 70, "r": 40, "t": 105, "b": 145},
        annotations=list(fig.layout.annotations)
        + [
            {
                "text": html.escape(case["diagnosis"]),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.16,
                "showarrow": False,
                "align": "left",
                "font": {"size": 12, "color": "#303641"},
            }
        ],
    )
    figure_html = pio.to_html(fig, full_html=False, include_plotlyjs=False, config={"displaylogo": False, "scrollZoom": True})
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(case['method_label'])} {case['line']} {case['frame']}</title>
<script src="../../../assets/plotly.min.js"></script>
<style>body{{margin:0;background:#fff;font-family:Arial,sans-serif}}.note{{padding:12px 24px;background:#fff8e8;border-top:1px solid #e0d7bd;color:#493f25}}</style></head>
<body>{figure_html}<div class="note"><b>Audit scope:</b> final predictions are associated to Car GT by a 2 m center rule and annotated with oriented 3D IoU. This selected-frame audit explains behavior; official AP comes from all 3,769 validation frames.</div></body></html>"""
    write_text(path, document)


def create_interactive_3d(case: dict, path: Path) -> None:
    titles = [
        "1. Original reference + observed baseline (complete scan)",
        "2. E1 exact x4 (complete): observed vs generated",
        "3. E2 actual detector input: exactly 16,384",
        "4. Baseline E2 + PointRCNN final outputs",
        "5. Upsampled E2 + PointRCNN final outputs",
        f"6. Focus GT {case['focus_idx']}: generated-point geometry and boxes",
    ]
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"type": "scene"}] * 3, [{"type": "scene"}] * 3],
        subplot_titles=titles,
        horizontal_spacing=0.025,
        vertical_spacing=0.075,
    )

    if case["line"] == "B":
        add_plotly_3d_points(
            fig, 1, 1, case["reference"], "original KITTI reference", COLORS["reference"],
            size=1.1, opacity=0.24, max_points=DISPLAY_MAX_3D_FULL, legendgroup="reference",
        )
    add_plotly_3d_points(
        fig, 1, 1, case["observed"], "observed baseline", COLORS["observed"],
        size=1.3, opacity=0.58, max_points=DISPLAY_MAX_3D_FULL, legendgroup="observed",
    )
    for gi, gt_box in enumerate(case["baseline_audit"]["gt"]):
        add_plotly_3d_box(
            fig, 1, 1, gt_box, "GT Car", COLORS["gt"], width=6,
            showlegend=(gi == 0), hover_text=f"GT Car {gi}", legendgroup="gt-context",
        )

    e1_observed = case["e1"][case["e1_full_observed_mask"]]
    e1_generated = case["e1"][~case["e1_full_observed_mask"]]
    add_plotly_3d_points(
        fig, 1, 2, e1_observed, "E1 exact observed rows", COLORS["observed"],
        size=1.2, opacity=0.56, max_points=DISPLAY_MAX_3D_FULL, legendgroup="e1-observed",
    )
    add_plotly_3d_points(
        fig, 1, 2, e1_generated, "E1 generated rows", COLORS["generated"],
        size=1.1, opacity=0.42, max_points=DISPLAY_MAX_3D_GENERATED, legendgroup="e1-generated",
    )
    for gi, gt_box in enumerate(case["baseline_audit"]["gt"]):
        add_plotly_3d_box(
            fig, 1, 2, gt_box, "GT Car on E1", COLORS["gt"], width=6,
            showlegend=False, hover_text=f"GT Car {gi}", legendgroup="gt-context",
        )

    e2_observed = case["e2"][case["e2_observed_mask"]]
    e2_generated = case["e2"][~case["e2_observed_mask"]]
    add_plotly_3d_points(
        fig, 1, 3, e2_observed, "E2 observed upper bound", COLORS["observed"],
        size=1.7, opacity=0.75, max_points=20000, legendgroup="e2-observed",
    )
    add_plotly_3d_points(
        fig, 1, 3, e2_generated, "E2 generated lower bound", COLORS["generated"],
        size=1.6, opacity=0.66, max_points=20000, legendgroup="e2-generated",
    )
    for gi, gt_box in enumerate(case["baseline_audit"]["gt"]):
        add_plotly_3d_box(
            fig, 1, 3, gt_box, "GT Car on E2", COLORS["gt"], width=6,
            showlegend=False, hover_text=f"GT Car {gi}", legendgroup="gt-context",
        )

    add_plotly_3d_points(
        fig, 2, 1, case["baseline_e2"], "baseline actual E2 input", "#7D8795",
        size=1.5, opacity=0.42, max_points=20000, legendgroup="baseline-e2",
    )
    add_plotly_3d_boxes_and_decisions(fig, 2, 1, case["baseline_audit"], "baseline", showlegend=True)

    add_plotly_3d_points(
        fig, 2, 2, e2_observed, "upsampled E2 observed upper", COLORS["observed"],
        size=1.6, opacity=0.70, max_points=20000, showlegend=False, legendgroup="e2-observed",
    )
    add_plotly_3d_points(
        fig, 2, 2, e2_generated, "upsampled E2 generated lower", COLORS["generated"],
        size=1.5, opacity=0.60, max_points=20000, showlegend=False, legendgroup="e2-generated",
    )
    add_plotly_3d_boxes_and_decisions(fig, 2, 2, case["up_audit"], "upsampled", showlegend=True)

    focus = case["focus_gt"]
    reference_crop = case["reference"][crop_mask(case["reference"], focus)]
    observed_crop = case["observed"][crop_mask(case["observed"], focus)]
    generated_crop = e1_generated[crop_mask(e1_generated, focus)]
    e2_observed_crop = e2_observed[crop_mask(e2_observed, focus)]
    e2_generated_crop = e2_generated[crop_mask(e2_generated, focus)]
    add_plotly_3d_points(fig, 2, 3, reference_crop, "focus original reference", COLORS["reference"], 2.3, 0.25, max_points=30000, legendgroup="focus-reference")
    add_plotly_3d_points(fig, 2, 3, observed_crop, "focus observed baseline", COLORS["observed"], 3.0, 0.70, max_points=30000, legendgroup="focus-observed")
    add_plotly_3d_points(fig, 2, 3, generated_crop, "focus E1 generated", COLORS["generated"], 2.7, 0.62, max_points=40000, legendgroup="focus-generated")
    add_plotly_3d_points(fig, 2, 3, e2_observed_crop, "focus E2 selected observed", "#21A7B7", 4.0, 0.90, "diamond", 30000, True, "focus-e2-observed")
    add_plotly_3d_points(fig, 2, 3, e2_generated_crop, "focus E2 selected generated", COLORS["selected"], 3.7, 0.88, "x", 30000, True, "focus-e2-generated")
    add_plotly_3d_box(
        fig, 2, 3, focus, "focus GT Car", COLORS["gt"], width=8, showlegend=True,
        hover_text=f"<b>focus GT Car {case['focus_idx']}</b><br>{html.escape(case['focus_reason'])}", legendgroup="focus-gt",
    )
    base_match = match_map(case["baseline_audit"]).get(case["focus_idx"])
    up_match = match_map(case["up_audit"]).get(case["focus_idx"])
    if base_match:
        pred = case["baseline_audit"]["pred"][base_match["pred_idx"]]
        color, dash, short, hover = prediction_audit_detail("baseline", base_match["pred_idx"], pred, base_match)
        add_plotly_3d_box(fig, 2, 3, pred, "focus baseline prediction", COLORS["baseline_pred"], 7, dash, True, hover, short, "focus-base-pred")
    if up_match:
        pred = case["up_audit"]["pred"][up_match["pred_idx"]]
        color, dash, short, hover = prediction_audit_detail("upsampled", up_match["pred_idx"], pred, up_match)
        add_plotly_3d_box(fig, 2, 3, pred, "focus upsampled prediction", color, 7, dash, True, hover, short, "focus-up-pred")
    if not up_match:
        center = box_center(focus)
        fig.add_trace(
            go.Scatter3d(
                x=[center[0]], y=[center[1]], z=[center[2] + 1.4], mode="text", text=["UPSAMPLED MISS"],
                textfont={"color": COLORS["fp"], "size": 13}, showlegend=False,
                hovertemplate="No upsampled final Car prediction associated within 2 m<extra></extra>",
            ), row=2, col=3,
        )

    for row, col in ((1, 1), (1, 2)):
        configure_scene(fig, row, col, [-80, 80], [-80, 80], [-5, 5], {"x": 1.0, "y": 1.0, "z": 0.18})
    for row, col in ((1, 3), (2, 1), (2, 2)):
        configure_scene(fig, row, col, [0, 70.4], [-40, 40], [-3, 4], {"x": 1.15, "y": 1.0, "z": 0.16})
    corners = np.asarray(focus["corners_lidar"], dtype=np.float64)
    configure_scene(
        fig, 2, 3,
        [float(corners[:, 0].min() - 3.0), float(corners[:, 0].max() + 3.0)],
        [float(corners[:, 1].min() - 3.0), float(corners[:, 1].max() + 3.0)],
        [float(corners[:, 2].min() - 2.0), float(corners[:, 2].max() + 2.0)],
        {"x": 1.0, "y": 1.0, "z": 0.72},
    )
    fig.update_layout(
        title={
            "text": html.escape(
                f"3D interactive pipeline | {case['method_label']} | {case['line_label']} | frame {case['frame']} | Moderate AP delta {case['ap_delta']:+.2f}"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        height=1390,
        width=1900,
        legend={"orientation": "h", "y": -0.035, "x": 0.0, "font": {"size": 11}},
        margin={"l": 25, "r": 25, "t": 105, "b": 110},
    )
    figure_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )
    body = (
        "<div class='guide'><b>操作：</b>左键旋转，滚轮缩放，右键平移；双击场景可自动缩放；点击图例开关 original / observed / generated / E2 / boxes。"
        "第 3、4、5 个场景的相机方向会同步，便于在同一视角比较 detector 输入与结果。"
        " <a href='interactive_bev.html'>打开精确 2D BEV 辅助页</a> · <a href='metadata.json'>metadata</a></div>"
        f"<div class='note'><b>生成点与选点：</b>E1 为 observed + 3N generated；E2 是 detector 实际收到的 16,384 点。"
        f"本帧 E2 observed≤{100 * case['selection']['e2_exact_observed_fraction_upper']:.1f}%，generated≥{100 * case['selection']['e2_generated_fraction_lower']:.1f}%。"
        f"完整场景为浏览器性能使用确定性显示 LOD（original≤{DISPLAY_MAX_3D_FULL:,}，generated≤{DISPLAY_MAX_3D_GENERATED:,}）；点数统计、局部 crop 和 detector 数据不因此改变。"
        "完整场景轴默认限制在常规 KITTI LiDAR 范围，双击场景可查看超范围生成点。</div>"
        "<div class='note'><b>Detector 判定边界：</b>框来自 <code>final_result/data</code>，所以它们已经通过当前 PointRCNN 的 RPN/RCNN score filtering 与 NMS。"
        "页面没有保存被过滤的 proposal，因而黄/绿/橙/红不是模型内部的接受/拒绝颜色：黄色是 GT；绿/橙/红分别是离线同类中心≤2 m 关联后 IoU≥0.70、IoU&lt;0.70、未关联。"
        "KITTI txt 的 score 是 raw RCNN logit，表中同时给出 sigmoid；正式 AP 仍来自全部 3,769 帧。</div>"
        f"<div class='note'><b>本 case 诊断：</b>{html.escape(case['diagnosis'])}</div>"
        f"<div class='decisions'>{decision_table_html(case['baseline_audit'], 'Baseline final-output audit')}{decision_table_html(case['up_audit'], 'Upsampled final-output audit')}</div>"
    )
    document = plotly_document(
        f"3D | {case['method_label']} {case['line']} {case['frame']}",
        figure_html,
        body,
        "../../../assets/plotly.min.js",
        ("scene3", "scene4", "scene5"),
    )
    write_text(path, document)


def write_interactive_alias(path: Path) -> None:
    write_text(
        path,
        """<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="0; url=interactive_3d.html">
<title>Open 3D interactive view</title></head><body><p><a href="interactive_3d.html">Open the 3D interactive view</a> · <a href="interactive_bev.html">Open the 2D BEV view</a></p></body></html>""",
    )


def create_comparison_sheet(items: list[dict], path: Path) -> None:
    """Put the shared baseline and all four methods on one fixed-frame sheet."""
    items = sorted(items, key=lambda case: METHODS.index(case["method"]))
    first = items[0]
    fig, axes = plt.subplots(2, 3, figsize=(23, 14), facecolor="#F6F7F9")
    draw_decisions(
        axes[0, 0],
        first["baseline_e2"],
        first["baseline_audit"],
        f"Baseline | AP Mod {first['baseline_ap']:.2f}\n"
        f"assoc/extra/miss={first['base_summary']['associated']}/{first['base_summary']['extra']}/{first['base_summary']['missed']}",
    )
    for ax, case in zip((axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]), items):
        draw_decisions(
            ax,
            case["e2"],
            case["up_audit"],
            f"{case['method_label']} | AP Mod {case['method_ap']:.2f} ({case['ap_delta']:+.2f})\n"
            f"assoc/extra/miss={case['up_summary']['associated']}/{case['up_summary']['extra']}/{case['up_summary']['missed']}",
        )

    note = axes[1, 2]
    note.axis("off")
    lines = [
        "Same-frame comparison",
        "",
        f"{first['line_label']}",
        f"frame {first['frame']} | GT Cars: {first['base_summary']['gt_car']}",
        "",
        "Method             assoc  extra  miss   E2 gen>=   AP delta",
    ]
    for case in items:
        lines.append(
            f"{case['method_label']:<18} {case['up_summary']['associated']:>2}     {case['up_summary']['extra']:>2}    "
            f"{case['up_summary']['missed']:>2}     {100 * case['selection']['e2_generated_fraction_lower']:>5.1f}%   {case['ap_delta']:>+7.2f}"
        )
    lines.extend(
        [
            "",
            "How to read:",
            "yellow = GT Car",
            "green = associated prediction with 3D IoU >= 0.70",
            "orange = associated but 3D IoU < 0.70",
            "red dashed = extra prediction",
            "",
            "All panels use the same E2 policy, checkpoint,",
            "axes, frame, GT, and final-output audit.",
        ]
    )
    note.text(
        0.03,
        0.97,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        color="#253041",
        linespacing=1.35,
    )
    fig.suptitle(
        f"Cross-method PointRCNN decision comparison | {first['line_label']} | frame {first['frame']}",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    handles, labels = [], []
    for ax in axes.ravel()[:5]:
        hnd, lbl = ax.get_legend_handles_labels()
        for item, label in zip(hnd, lbl):
            if label not in labels:
                handles.append(item)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), frameon=False, fontsize=9)
    fig.text(
        0.02,
        0.015,
        "Selected-frame association/IoU is explanatory; official AP values are the full 3,769-frame E2 Car 3D AP_R40 results.",
        fontsize=8.5,
        color="#5B6472",
    )
    fig.tight_layout(rect=(0.015, 0.05, 0.99, 0.95), w_pad=1.5, h_pad=1.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_comparison_3d(items: list[dict], path: Path) -> None:
    """Interactive same-camera comparison of baseline and all four upsampling methods."""
    items = sorted(items, key=lambda case: METHODS.index(case["method"]))
    first = items[0]
    titles = [
        f"Baseline E2 + final outputs (AP Mod {first['baseline_ap']:.2f})",
        *[
            f"{case['method_label']} E2 + final outputs (ΔAP {case['ap_delta']:+.2f})"
            for case in items
        ],
        "Original complete scan + GT context",
    ]
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"type": "scene"}] * 3, [{"type": "scene"}] * 3],
        subplot_titles=titles,
        horizontal_spacing=0.025,
        vertical_spacing=0.075,
    )
    add_plotly_3d_points(
        fig, 1, 1, first["baseline_e2"], "baseline actual E2 input", "#7D8795",
        size=1.6, opacity=0.43, max_points=20000, legendgroup="baseline-e2",
    )
    add_plotly_3d_boxes_and_decisions(fig, 1, 1, first["baseline_audit"], "baseline", showlegend=True)
    positions = ((1, 2), (1, 3), (2, 1), (2, 2))
    for index, (case, (row, col)) in enumerate(zip(items, positions)):
        observed = case["e2"][case["e2_observed_mask"]]
        generated = case["e2"][~case["e2_observed_mask"]]
        add_plotly_3d_points(
            fig, row, col, observed, "E2 observed upper", COLORS["observed"],
            size=1.6, opacity=0.70, max_points=20000, showlegend=(index == 0), legendgroup="comparison-observed",
        )
        add_plotly_3d_points(
            fig, row, col, generated, "E2 generated lower", COLORS["generated"],
            size=1.5, opacity=0.60, max_points=20000, showlegend=(index == 0), legendgroup="comparison-generated",
        )
        add_plotly_3d_boxes_and_decisions(fig, row, col, case["up_audit"], case["method_label"], showlegend=False)

    add_plotly_3d_points(
        fig, 2, 3, first["reference"], "original KITTI complete scan", COLORS["reference"],
        size=1.2, opacity=0.48, max_points=DISPLAY_MAX_3D_FULL, legendgroup="comparison-reference",
    )
    for gi, gt_box in enumerate(first["baseline_audit"]["gt"]):
        add_plotly_3d_box(
            fig, 2, 3, gt_box, "GT Car context", COLORS["gt"], width=6,
            showlegend=(gi == 0), hover_text=f"GT Car {gi}", legendgroup="comparison-gt-context",
        )

    for row, col in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2)):
        configure_scene(fig, row, col, [0, 70.4], [-40, 40], [-3, 4], {"x": 1.15, "y": 1.0, "z": 0.16})
    configure_scene(fig, 2, 3, [-80, 80], [-80, 80], [-5, 5], {"x": 1.0, "y": 1.0, "z": 0.18})
    fig.update_layout(
        title={
            "text": html.escape(f"3D cross-method comparison | {first['line_label']} | frame {first['frame']}"),
            "x": 0.5,
        },
        template="plotly_white",
        height=1360,
        width=1900,
        legend={"orientation": "h", "y": -0.035, "x": 0.0, "font": {"size": 11}},
        margin={"l": 25, "r": 25, "t": 105, "b": 105},
    )
    figure_html = pio.to_html(
        fig, full_html=False, include_plotlyjs=False,
        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
    )
    table_rows = []
    for case in items:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(case['method_label'])}</td><td>{case['method_ap']:.2f}</td><td>{case['ap_delta']:+.2f}</td>"
            f"<td>{100 * case['selection']['e2_generated_fraction_lower']:.1f}%</td>"
            f"<td>{case['up_summary']['prediction_car']}</td><td>{case['up_summary']['associated']}</td>"
            f"<td>{case['up_summary']['extra']}</td><td>{case['up_summary']['missed']}</td>"
            f"<td>{case['up_summary']['strict_tp_0p7']}/{case['up_summary']['strict_fp_0p7']}/{case['up_summary']['strict_fn_0p7']}</td>"
            "</tr>"
        )
    body = (
        "<div class='guide'><b>同视角比较：</b>前五个场景相机同步；旋转、缩放任一 detector 场景，其余会跟随。点击图例可分别隐藏点或框。"
        f" <a href='line_{first['line']}_frame_{first['frame']}.png'>打开 2D 固定视角对比图</a> · <a href='../index.html'>返回总索引</a></div>"
        "<div class='note'><b>颜色含义：</b>蓝/橙为 E2 中 observed 上界/generated 下界；黄色为 GT。预测框颜色是离线解释：绿色 IoU≥0.70，橙色已关联但 IoU&lt;0.70，红色未关联。"
        "所有显示的预测框本身都来自 PointRCNN final output，已经通过模型的 score filtering 和 NMS。</div>"
        "<section class='decision'><h3>Same-frame audit summary</h3><div class='table-wrap'><table><thead><tr>"
        "<th>method</th><th>AP Mod</th><th>ΔAP</th><th>E2 generated≥</th><th>final pred</th><th>assoc</th><th>extra</th><th>miss</th><th>TP/FP/FN@0.70</th>"
        f"</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div></section>"
    )
    document = plotly_document(
        f"3D comparison | line {first['line']} frame {first['frame']}",
        figure_html,
        body,
        "../assets/plotly.min.js",
        ("scene", "scene2", "scene3", "scene4", "scene5"),
    )
    write_text(path, document)


def case_metadata(case: dict) -> dict:
    return {
        "method": case["method_label"],
        "method_key": case["method"],
        "line": case["line"],
        "line_label": case["line_label"],
        "frame_id": case["frame"],
        "variant": case["variant"],
        "baseline_variant": case["baseline_variant"],
        "frame_selection_reason": "fixed shared frames: 000002 distant single-Car; 000152 mixed-distance seven-Car scene",
        "paths": case["paths"],
        "ap": {
            "metric": "Car 3D AP_R40 Moderate on full 3769-frame validation",
            "baseline": case["baseline_ap"],
            "upsampled": case["method_ap"],
            "delta": case["ap_delta"],
        },
        "selection": case["selection"],
        "baseline_detector_audit": case["base_summary"],
        "upsampled_detector_audit": case["up_summary"],
        "focus_object": {
            "car_gt_index": case["focus_idx"],
            "reason": case["focus_reason"],
            "geometry_metrics": case["object_metric"],
        },
        "method_line_geometry": case["geometry"],
        "diagnosis": case["diagnosis"],
        "policies": {
            "E1": "N observed + 3N generated, exact x4; observed multiset preserved",
            "E2": "PointRCNN FOV/range -> 0.1 m voxel representative -> proportional depth sampling/fill -> exactly 16384",
            "observed_generated_attribution": "exact-row membership; observed fraction upper bound, generated fraction lower bound",
            "box_association": "same-class center distance <=2 m; oriented BEV/3D IoU annotated",
            "iou_reference": "Car 3D IoU >=0.70 explanatory threshold; not a reimplementation of the complete KITTI evaluator",
            "3d_display_lod": f"deterministic display-only sampling: full layers <= {DISPLAY_MAX_3D_FULL}, generated layers <= {DISPLAY_MAX_3D_GENERATED}; detector E2 and focus crop retain all available rows",
            "detector_output_scope": "boxes are final_result/data outputs that survived score filtering and NMS; rejected proposals were not saved and are not visualized",
        },
    }


def build_case(line: str, method: str, frame: str, ap: dict, method_geometry: dict, object_geometry: dict) -> dict:
    line_cfg = LINES[line]
    variant = f"{line_cfg['prefix']}_{method}"
    baseline_variant = str(line_cfg["baseline"])
    observed_path = Path(line_cfg["observed"]) / f"{frame}.bin"
    reference_path = prep.ORIGINAL / f"{frame}.bin"
    e1_path = WORKSPACE / "inputs/e1_default_16384" / variant / f"{frame}.bin"
    e2_path = WORKSPACE / "inputs/e2_canonical_16384" / variant / f"{frame}.bin"
    baseline_e2_path = WORKSPACE / "inputs/e2_canonical_16384" / baseline_variant / f"{frame}.bin"
    calib_path = TRAINING / "calib" / f"{frame}.txt"
    label_path = TRAINING / "label_2" / f"{frame}.txt"
    base_pred_path = pred_dir(baseline_variant) / f"{frame}.txt"
    up_pred_path = pred_dir(variant) / f"{frame}.txt"
    required = [observed_path, reference_path, e1_path, e2_path, baseline_e2_path, calib_path, label_path, base_pred_path, up_pred_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing case inputs:\n" + "\n".join(missing))

    observed = prep.read_bin(observed_path)
    reference = prep.read_bin(reference_path)
    e1 = prep.read_bin(e1_path)
    e2 = prep.read_bin(e2_path)
    baseline_e2 = prep.read_bin(baseline_e2_path)
    observed_valid_mask, observed_rect_all = prep.fov_valid_mask(observed, frame)
    observed_valid = observed[observed_valid_mask]
    observed_rect = observed_rect_all[observed_valid_mask]
    e1_valid_mask, _ = prep.fov_valid_mask(e1, frame)
    e1_valid = e1[e1_valid_mask]
    e1_full_observed_mask = observed_membership_mask(observed, e1)
    e1_observed_mask = observed_membership_mask(observed_valid, e1_valid)
    e2_observed_mask = observed_membership_mask(observed_valid, e2)
    manifest = load_manifest_row(variant, frame)
    selection = selection_metrics(observed, observed_valid, observed_rect, e1_valid, e2, frame, manifest)

    calib = det.parse_calib(calib_path)
    gt_boxes = det.maybe_gt(frame, calib)
    base_boxes = det.parse_kitti_boxes(base_pred_path, calib, "baseline")
    up_boxes = det.parse_kitti_boxes(up_pred_path, calib, "upsampled")
    baseline_audit = associate(gt_boxes, base_boxes)
    up_audit = associate(gt_boxes, up_boxes)
    focus_idx, focus_reason = choose_focus_object(baseline_audit, up_audit)
    focus_gt = baseline_audit["gt"][focus_idx]
    object_metric = object_geometry.get((line, method, frame, focus_idx))
    baseline_ap = ap[baseline_variant]["moderate"]
    method_ap = ap[variant]["moderate"]
    ap_delta = method_ap - baseline_ap
    diagnosis_text = diagnosis(
        baseline_audit,
        up_audit,
        selection,
        method_geometry[(line, method)],
        object_metric,
        ap_delta,
        method,
        line,
    )
    paths = {
        "observed_baseline": str(observed_path),
        "original_reference": str(reference_path),
        "e1_exact_x4": str(e1_path),
        "e2_selected_16384": str(e2_path),
        "baseline_e2_selected_16384": str(baseline_e2_path),
        "baseline_final_predictions": str(base_pred_path),
        "upsampled_final_predictions": str(up_pred_path),
        "ground_truth": str(label_path),
        "calibration": str(calib_path),
        "e2_manifest": str(WORKSPACE / "manifests/e2_canonical_16384" / f"{variant}.csv"),
    }
    return {
        "line": line,
        "line_label": str(line_cfg["label"]),
        "method": method,
        "method_label": METHOD_LABEL[method],
        "frame": frame,
        "variant": variant,
        "baseline_variant": baseline_variant,
        "observed": observed,
        "observed_valid": observed_valid,
        "reference": reference,
        "e1": e1,
        "e1_full_observed_mask": e1_full_observed_mask,
        "e1_valid": e1_valid,
        "e1_observed_mask": e1_observed_mask,
        "e2": e2,
        "e2_observed_mask": e2_observed_mask,
        "baseline_e2": baseline_e2,
        "baseline_audit": baseline_audit,
        "up_audit": up_audit,
        "base_summary": audit_summary(baseline_audit),
        "up_summary": audit_summary(up_audit),
        "focus_idx": focus_idx,
        "focus_reason": focus_reason,
        "focus_gt": focus_gt,
        "selection": selection,
        "geometry": method_geometry[(line, method)],
        "object_metric": object_metric,
        "baseline_ap": baseline_ap,
        "method_ap": method_ap,
        "ap_delta": ap_delta,
        "diagnosis": diagnosis_text,
        "paths": paths,
    }


def summary_row(case: dict) -> dict:
    row = {
        "method": case["method_label"],
        "line": case["line"],
        "frame_id": case["frame"],
        "baseline_variant": case["baseline_variant"],
        "upsampled_variant": case["variant"],
        "baseline_3d_ap_r40_moderate": case["baseline_ap"],
        "upsampled_3d_ap_r40_moderate": case["method_ap"],
        "delta_3d_ap_r40_moderate": case["ap_delta"],
        **{f"baseline_{key}": value for key, value in case["base_summary"].items()},
        **{f"upsampled_{key}": value for key, value in case["up_summary"].items()},
        **case["selection"],
        "focus_gt_index": case["focus_idx"],
        "focus_reason": case["focus_reason"],
        "diagnosis": case["diagnosis"],
        "overview_png": f"line_{case['line']}/{case['method']}/frame_{case['frame']}/overview.png",
        "interactive_html": f"line_{case['line']}/{case['method']}/frame_{case['frame']}/interactive.html",
        "interactive_3d_html": f"line_{case['line']}/{case['method']}/frame_{case['frame']}/interactive_3d.html",
        "interactive_bev_html": f"line_{case['line']}/{case['method']}/frame_{case['frame']}/interactive_bev.html",
    }
    return row


def render_index(rows: list[dict]) -> str:
    comparison_cards = []
    for line in LINES:
        for frame in FRAMES:
            path = f"comparison_sheets/line_{line}_frame_{frame}.png"
            path_3d = f"comparison_sheets/line_{line}_frame_{frame}_3d.html"
            comparison_cards.append(
                f"<div class='frame'><a href='{path_3d}'><img src='{path}' alt='cross-method comparison'></a>"
                f"<p><b>{html.escape(LINES[line]['label'])} · frame {frame}</b><br>"
                f"<a class='primary' href='{path_3d}'>open synchronized 3D comparison</a> · <a href='{path}'>2D full-resolution sheet</a></p></div>"
            )
    cards = []
    for line in LINES:
        for method in METHODS:
            subset = [row for row in rows if row["line"] == line and row["method"] == METHOD_LABEL[method]]
            links = []
            for row in subset:
                links.append(
                    f"<div class='frame'><a href='{html.escape(row['interactive_3d_html'])}'><img src='{html.escape(row['overview_png'])}' alt='overview'></a>"
                    f"<p><b>frame {row['frame_id']}</b> · AP delta {float(row['delta_3d_ap_r40_moderate']):+.2f} · "
                    f"assoc {row['baseline_associated']}→{row['upsampled_associated']} · miss {row['baseline_missed']}→{row['upsampled_missed']}<br>"
                    f"<a class='primary' href='{html.escape(row['interactive_3d_html'])}'>3D interactive pipeline</a> · "
                    f"<a href='{html.escape(row['interactive_bev_html'])}'>2D BEV</a> · "
                    f"<a href='{html.escape(row['overview_png'])}'>PNG</a> · "
                    f"<a href='{html.escape(os.path.dirname(row['interactive_html']) + '/metadata.json')}'>metadata</a></p></div>"
                )
            cards.append(
                f"<section><h2>{html.escape(METHOD_LABEL[method])} · {html.escape(LINES[line]['label'])}</h2>"
                f"<div class='grid'>{''.join(links)}</div></section>"
            )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Unified E1/E2 Detector Selection Visualization</title>
<style>body{{font-family:Inter,Arial,sans-serif;margin:0;background:#eef1f5;color:#172033}}header,main{{max-width:1600px;margin:auto;padding:24px}}header{{background:white;max-width:none;border-bottom:1px solid #d8dde5}}h1{{margin:0 0 8px}}section{{background:white;border:1px solid #d8dde5;border-radius:10px;padding:18px;margin:18px 0}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}.frame img{{width:100%;border:1px solid #ccd3dd;border-radius:7px}}a{{color:#086f83;text-decoration:none}}a.primary{{font-weight:700;color:#075f70}}.note{{background:#fff8e6;border-left:4px solid #dc9b22;padding:14px;margin-top:14px;line-height:1.5}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><header><h1>3D Unified E1 → E2 → PointRCNN visualization</h1><p>Four current methods × two experiment lines × the same two frames (000002 and 000152). Every case has rotatable 3D and precise 2D BEV views.</p></header>
<main><div class="note"><b>主入口现在是 3D：</b>每个 case 左到右依次为完整 original/reference + observed baseline、完整 exact-x4 E1（observed/generated 分色）、detector 实际 16,384 点 E2、baseline 最终检测、upsampled 最终检测和同一目标 3D crop。可旋转、缩放、平移和开关图层；2D BEV 只作为精确俯视辅助。</div>
<div class="note"><b>Detector 颜色不要误读：</b>显示的预测框均已由 PointRCNN 输出，已经通过 score filtering 与 NMS。绿/橙/红是之后为了分析做的 GT 关联与 0.70 IoU 审计，并非 detector 内部“选中/拒绝”状态；逐框 raw logit、sigmoid、匹配距离和 3D IoU 可在 3D hover 与页底表格查看。</div>
<p><a href="analysis_summary_zh.md">Chinese root-cause analysis</a> · <a href="case_summary.csv">case metrics CSV</a> · <a href="method_line_summary.csv">method/line summary CSV</a> · <a href="validation_report.json">validation report</a></p>
<section><h2>Cross-method synchronized 3D comparisons</h2><div class="grid">{''.join(comparison_cards)}</div></section>
{''.join(cards)}</main></body></html>"""


def method_line_rows(cases: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case["line"], case["method"])].append(case)
    rows = []
    for (line, method), items in sorted(grouped.items()):
        geometry = items[0]["geometry"]
        rows.append(
            {
                "method": METHOD_LABEL[method],
                "line": line,
                "baseline_3d_ap_r40_moderate": items[0]["baseline_ap"],
                "upsampled_3d_ap_r40_moderate": items[0]["method_ap"],
                "delta_3d_ap_r40_moderate": items[0]["ap_delta"],
                "selected_frames": "|".join(case["frame"] for case in items),
                "baseline_associated_total": sum(case["base_summary"]["associated"] for case in items),
                "upsampled_associated_total": sum(case["up_summary"]["associated"] for case in items),
                "baseline_missed_total": sum(case["base_summary"]["missed"] for case in items),
                "upsampled_missed_total": sum(case["up_summary"]["missed"] for case in items),
                "e2_observed_fraction_upper_mean": float(np.mean([case["selection"]["e2_exact_observed_fraction_upper"] for case in items])),
                "e2_generated_fraction_lower_mean": float(np.mean([case["selection"]["e2_generated_fraction_lower"] for case in items])),
                "voxel_0p20_generated_reference_precision_median": fnum(geometry["voxel_0p20_generated_reference_precision_median"]),
                "voxel_0p20_e1_extra_fraction_median": fnum(geometry["voxel_0p20_e1_extra_fraction_median"]),
                "generated_inside_near_reference_0p25_median": fnum(geometry["generated_inside_near_reference_0p25_median"]),
                "generated_shell_to_inside_median": fnum(geometry["generated_shell_to_inside_median"]),
                "pu_net_wrapper_validity": "INVALID_CURRENT_WRAPPER" if method == "pu_net" else "no_known_method_specific_wrapper_defect",
            }
        )
    return rows


def render_analysis(cases: list[dict], summary: list[dict]) -> str:
    def row_for(line, method):
        return next(row for row in summary if row["line"] == line and row["method"] == METHOD_LABEL[method])

    table_lines = [
        "| Line | Method | 3D AP Mod baseline -> up | Delta | Selected-frame assoc | Selected-frame misses | E2 generated lower bound | Gen voxel precision | E1 extra voxels |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for line in LINES:
        for method in METHODS:
            row = row_for(line, method)
            table_lines.append(
                f"| {line} | {row['method']} | {float(row['baseline_3d_ap_r40_moderate']):.2f} -> {float(row['upsampled_3d_ap_r40_moderate']):.2f} | {float(row['delta_3d_ap_r40_moderate']):+.2f} | "
                f"{row['baseline_associated_total']} -> {row['upsampled_associated_total']} | {row['baseline_missed_total']} -> {row['upsampled_missed_total']} | "
                f">={100 * float(row['e2_generated_fraction_lower_mean']):.1f}% | {100 * float(row['voxel_0p20_generated_reference_precision_median']):.1f}% | {100 * float(row['voxel_0p20_e1_extra_fraction_median']):.1f}% |"
            )
    return "\n".join(
        [
            "# 统一两帧：E1 / E2 / PointRCNN 检测下降分析",
            "",
            "## 结论",
            "",
            "这批可视化把问题定位在两个连续环节：第一，上采样生成的 3N 点中存在大量重复、偏离真实表面或落在车辆外壳/背景的几何；第二，E2 固定 16,384 点选择会让这些生成点进入 detector，并在 Line A 中挤掉大量原始测量。PointRCNN 之后仍按同一 checkpoint 做前景打分、proposal、RCNN 框回归/打分和 NMS，所以最终表现为漏检增加、框定位不能通过 0.70 Car 3D-IoU 参考线、置信分数下降或额外框增加。",
            "",
            "当前证据不支持“只是点数不够”这一解释：E3 将容量提高到 32,768 并优先保留真实点后，Line B 的可靠方法没有恢复；Line A 虽恢复 2.84--6.68 AP，仍明显低于同容量 baseline，说明生成几何污染仍是主因。",
            "",
            "## 为什么固定这两帧",
            "",
            "- `000002`：只有一个约 34.38 m 的 Moderate Car。Line A baseline 能关联该车，但 PU-GCN、PU-EdgeFormer、PU-Net 漏掉；Line B baseline 能关联，而四个上采样方法全部漏掉。它把远距稀疏目标上的退化隔离得很清楚。",
            "- `000152`：7 个 Car，距离约 6.3--46.8 m。Line A baseline 关联 7/7，上采样后各方法只关联 4--6；Line B baseline 关联 6/7，上采样后只关联 3--5。它能同时看到近车、远车、定位和分数变化。",
            "- 两帧在四个方法、两条线中数据和完整 E2 detector 输出均齐全，并且已有生成几何/车辆级指标，所以不是只挑一张好看的图。",
            "- 这里的“全部方法”指当前统一 exact-x4 E1/E2 协议下完成全量 detector 评测的 PDANS、PU-GCN、PU-EdgeFormer、PU-Net。EAR/TULIP 属于更早的 x2/方法专用输入与 detector 配置，SPU-PMD 只有不完整 smoke 输出；把它们混入本图会破坏选点和 AP 口径，因此没有伪装成同一组可比结果。",
            "",
            "## 统一结果表",
            "",
            *table_lines,
            "",
            "## Detector 实际如何选点和判断",
            "",
            "1. E1 先形成严格 `N observed + 3N generated = 4N`，真实输入点逐点保留。",
            "2. E2 使用与 PointRCNN 范围一致的相机 FOV/range 过滤，做 0.1 m voxel 代表点选择，再按 0--20/20--40/40--60/60--70.4 m 深度层比例采样；不足时用未选有效点或确定性重复补到 16,384。整个选择不使用 GT、类别或 AP。",
            "3. 保存后的 E2 已经恰好有 16,384 个有效点，所以 PointRCNN 不再做近/远二次下采样；图中第三列就是 detector 真正接收的点。",
            "4. PointRCNN RPN 对这些点提取局部邻域特征并给前景分数：当前 `default.yaml` 的 RPN score threshold 为 0.3，测试时最多从 9,000 个 pre-NMS proposal 经 RPN NMS=0.8 留 100 个。RCNN 再池化每个 proposal、回归 3D 框并打分；sigmoid score >0.3 后再用 RCNN NMS=0.1 输出最终框。KITTI txt 保存的是 raw RCNN score/logit，所以图中可出现负分，它不是校准概率。",
            "5. 图中的逐帧关联使用同类中心距离 <=2 m，再报告 oriented 3D IoU；`TP/FP/FN@0.70` 只用于解释 Car 的定位门槛。正式结论仍以 3,769 帧官方 AP_R40 为准。",
            "6. 当前评测目录只保存 `final_result/data`，没有保存每个被 score threshold 或 NMS 淘汰的 proposal。因此 3D 页中出现的所有预测框都是 detector 已经输出的框；绿/橙/红是输出之后的 GT 审计颜色，不能解释成 detector 内部的接受/拒绝。每个框的 hover 和页底表格同时给出 raw logit、sigmoid、关联 GT、中心距离、BEV IoU 和 3D IoU。",
            "",
            "## 为什么 Line A 下降",
            "",
            "Line A 的原始扫描本来已信息完整。上采样不会恢复缺失测量，只会增加估计点。E2 中四方法的真实行上界通常只有约 23--27%，独立真实测量保留率约 21--24%；其余位置主要被生成点占用。即使 E1 保留全部原始点，生成点仍改变 PointRCNN 的 kNN/局部特征、前景分数和 proposal 证据。E3 真实点优先只能部分恢复，证明“真实点被挤掉”是一个原因，但不是全部原因。",
            "",
            "## 为什么 Line B 下降",
            "",
            "Line B 的稀疏 baseline 确实缺点，但当前公共 patch 提取器不是严格局部邻域：2048 点连续块的典型 XY 跨度约 30.46 m、p90 约 124.10 m。网络会把多个物体、地面和背景当成同一 patch 插值。新增点只把 0.2 m 参考体素召回提高约 9--12 个百分点，同时制造约 42--54% 不受原始完整扫描支持的占据体素。E3 保留所有 observed 并加倍容量仍不恢复，说明 Line B 主因是生成点位置错误，不是 16,384 cap。",
            "",
            "## 方法差异",
            "",
            "- PDANS 的表面一致性最好，因此 AP 最高，但 Line B 仍有约 41.7% E1 额外体素，仍比 baseline 低 21.67 AP。",
            "- PU-GCN 次之；车辆近表面比例和生成体素精度下降，对应更大漏检。",
            "- PU-EdgeFormer 的额外体素和车辆外壳点更多，PointRCNN 前景/框证据进一步被稀释。",
            "- PU-Net 当前 wrapper 缺少预训练要求的中心化、尺度归一化和输出反变换；它的 AP 是 pipeline-defect 结果，不能当作 PU-Net 方法能力。",
            "",
            "## 如何看图",
            "",
            "每个 case 的主入口是 `interactive_3d.html`：左键旋转、滚轮缩放、右键平移，点击图例可独立开关 original/reference、observed、generated、E2 selected、GT 和预测框；第 3--5 个场景同步相机。六个 3D 场景依次是完整 original/reference + observed baseline、完整 E1 observed/generated、detector 实际 E2、baseline 最终检测、upsampled 最终检测、同一 GT 的局部 3D 放大。完整场景使用固定随机种子的显示 LOD，但 E2 的 16,384 点和局部 crop 使用全部可用点，统计完全基于全量数据。",
            "",
            "`interactive_bev.html` 保留精确二维俯视，`overview.png` 便于快速扫图；四个 `comparison_sheets/*_3d.html` 把同一实验线/同一帧的 baseline 与四种方法放在同步相机下。蓝色是 observed，橙色是生成/非 observed，黄色是 GT；绿色框在离线关联后通过 0.70 3D-IoU，橙框能关联但低于 0.70，红框未关联。focus crop 中青色菱形/黑色叉分别是 E2 最终选择的 observed/generated 行。",
            "",
            "## 限制",
            "",
            "逐帧关联用于解释，不复刻 KITTI 的 difficulty、DontCare 和全集阈值积分；官方 AP 只从现有全量 E2 评测读取。exact-row 方法会把与 observed 完全相同的生成行也算作 observed，因此 observed 比例是上界、generated 比例是下界。",
            "",
        ]
    )


def validate(cases: list[dict], rows: list[dict]) -> dict:
    errors = []
    expected = len(FRAMES) * len(METHODS) * len(LINES)
    if len(cases) != expected:
        errors.append(f"case count {len(cases)} != {expected}")
    for case in cases:
        case_dir = OUT / f"line_{case['line']}" / case["method"] / f"frame_{case['frame']}"
        for name in ("overview.png", "interactive.html", "interactive_3d.html", "interactive_bev.html", "metadata.json"):
            if not (case_dir / name).is_file():
                errors.append(f"missing {case_dir / name}")
        page_3d = case_dir / "interactive_3d.html"
        if page_3d.is_file() and '"type":"scatter3d"' not in page_3d.read_text(encoding="utf-8"):
            errors.append(f"3D trace missing from {page_3d}")
        if case["e2"].shape != (16384, 4):
            errors.append(f"bad E2 shape {case['variant']}/{case['frame']}: {case['e2'].shape}")
        if case["e1"].shape[0] != 4 * case["observed"].shape[0]:
            errors.append(f"bad E1 x4 count {case['variant']}/{case['frame']}")
        if case["frame"] not in FRAMES:
            errors.append(f"unexpected frame {case['frame']}")
    required = ("index.html", "README.md", "analysis_summary_zh.md", "case_summary.csv", "method_line_summary.csv")
    for name in required:
        if not (OUT / name).is_file():
            errors.append(f"missing {OUT / name}")
    for line in LINES:
        for frame in FRAMES:
            comparison = OUT / "comparison_sheets" / f"line_{line}_frame_{frame}.png"
            if not comparison.is_file():
                errors.append(f"missing {comparison}")
            comparison_3d = OUT / "comparison_sheets" / f"line_{line}_frame_{frame}_3d.html"
            if not comparison_3d.is_file():
                errors.append(f"missing {comparison_3d}")
    broken_links = []
    for page in OUT.rglob("*.html"):
        source = page.read_text(encoding="utf-8")
        for target in re.findall(r"href=['\"]([^'\"#]+)", source):
            if "://" in target or target.startswith(("mailto:", "javascript:")):
                continue
            resolved = (page.parent / target).resolve()
            if not resolved.exists():
                broken_links.append(f"{page.relative_to(OUT)} -> {target}")
    errors.extend(f"broken link {item}" for item in broken_links)
    return {
        "status": "PASS" if not errors else "FAIL",
        "expected_cases": expected,
        "generated_cases": len(cases),
        "shared_frames": list(FRAMES),
        "methods": [METHOD_LABEL[method] for method in METHODS],
        "lines": list(LINES),
        "static_overviews": len(list(OUT.rglob("overview.png"))),
        "interactive_alias_pages": len(list(OUT.rglob("interactive.html"))),
        "interactive_3d_pages": len(list(OUT.rglob("interactive_3d.html"))),
        "interactive_bev_pages": len(list(OUT.rglob("interactive_bev.html"))),
        "metadata_files": len(list(OUT.rglob("metadata.json"))),
        "cross_method_comparison_sheets": len(list((OUT / "comparison_sheets").glob("*.png"))),
        "cross_method_3d_pages": len(list((OUT / "comparison_sheets").glob("*_3d.html"))),
        "broken_local_links": broken_links,
        "summary_rows": len(rows),
        "errors": errors,
    }


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()
    OUT = args.output.resolve()
    if OUT.exists() and not args.keep_existing:
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    write_text(OUT / "assets/plotly.min.js", get_plotlyjs())

    ap = ap_lookup()
    method_geometry, object_geometry = geometry_lookup()
    cases = []
    rows = []
    for line in LINES:
        for method in METHODS:
            for frame in FRAMES:
                print(f"building {line} {method} frame {frame}", flush=True)
                case = build_case(line, method, frame, ap, method_geometry, object_geometry)
                case_dir = OUT / f"line_{line}" / method / f"frame_{frame}"
                create_static(case, case_dir / "overview.png")
                create_interactive_bev(case, case_dir / "interactive_bev.html")
                create_interactive_3d(case, case_dir / "interactive_3d.html")
                write_interactive_alias(case_dir / "interactive.html")
                write_text(case_dir / "metadata.json", json.dumps(case_metadata(case), indent=2, ensure_ascii=False) + "\n")
                cases.append(case)
                rows.append(summary_row(case))

    method_summary = method_line_rows(cases)
    for line in LINES:
        for frame in FRAMES:
            subset = [case for case in cases if case["line"] == line and case["frame"] == frame]
            create_comparison_sheet(
                subset,
                OUT / "comparison_sheets" / f"line_{line}_frame_{frame}.png",
            )
            create_comparison_3d(
                subset,
                OUT / "comparison_sheets" / f"line_{line}_frame_{frame}_3d.html",
            )
    write_csv(OUT / "case_summary.csv", rows)
    write_csv(OUT / "method_line_summary.csv", method_summary)
    write_text(OUT / "analysis_summary_zh.md", render_analysis(cases, method_summary))
    write_text(OUT / "index.html", render_index(rows))
    write_text(
        OUT / "README.md",
        """# Unified E1/E2 Detector Selection Visualization v1

Open `index.html`.  The package compares every method with completed full-validation results under the current unified exact-x4 E1/E2 protocol: PDANS, PU-GCN, PU-EdgeFormer, and PU-Net.  Both Line A and Line B use the same frames `000002` and `000152`.

EAR/TULIP are older x2 or method-specific detector pipelines, and SPU-PMD has only partial smoke output.  They are intentionally not mixed into this exact-x4 detector-input analysis.

Each case contains:

- `interactive_3d.html`: the primary rotatable 3D pipeline view (original/reference, observed baseline, E1 observed/generated, actual E2 detector input, baseline/upsampled final boxes, and a 3D focus crop);
- `interactive_bev.html`: the precise 2D BEV companion;
- `interactive.html`: a compatibility redirect to the 3D view;
- `overview.png`: a high-resolution static sheet;
- `metadata.json`: the complete data provenance and audit policy.

The four `comparison_sheets/*_3d.html` pages synchronize camera motion across the baseline and all four methods for the same line/frame.  `analysis_summary_zh.md` contains the Chinese root-cause analysis.  Official AP values are read from the completed 3,769-frame E2 evaluations; the per-frame association/IoU audit is explanatory only.  The displayed prediction boxes are already final PointRCNN outputs, not rejected proposals.
""",
    )
    report = validate(cases, rows)
    write_text(OUT / "validation_report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Output: {OUT}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
