#!/usr/bin/env python3
"""Build a classified gallery of Car detections lost after upsampling.

The source package contains three selected KITTI frames, two comparison lines,
four upsampling methods, and two detectors.  This script selects at most two
real ``lost_after_upsampling`` Car objects per combination.  It deliberately
creates explicit evidence-gap folders when fewer than two qualifying Cars
exist instead of substituting another class or inventing a loss.

Each selected object is rendered as a 2x3 figure:

* original full-density reference, with the GT box;
* detector baseline input, with the matched baseline prediction;
* detector upsampled input, with the GT box and no matched prediction;
* the same three views in BEV on the second row.

All points and boxes are transformed into the selected GT car's local frame so
the complete vehicle and box remain visible and comparable in every panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_BUNDLE = (
    SCRIPT_ROOT
    / "current_task_visualization_analysis_portable_20260731"
    / "current_task_visualization_analysis_portable_20260731"
)
DEFAULT_REPORT = (
    DEFAULT_BUNDLE
    / "reports"
    / "dual_detector_three_frame_root_cause_20260730"
)
DEFAULT_OUTPUT = (
    SCRIPT_ROOT.parent
    / "report"
    / "lost_after_upsampling_car_gallery_20260818"
)

BOX_EDGES = np.asarray(
    [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ],
    dtype=np.int32,
)

VOXEL_SIZE = np.asarray([0.05, 0.05, 0.1], dtype=np.float32)
POINT_RANGE = np.asarray([0.0, -40.0, -3.0, 70.4, 40.0, 1.0], dtype=np.float32)
MAX_POINTS_PER_VOXEL = 5
MAX_VOXELS = 40_000

METHOD_SLUG = {
    "PDANS": "pdans",
    "PU-GCN": "pu_gcn",
    "PU-EdgeFormer": "pu_edgeformer",
    "PU-Net": "pu_net",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_packaged_path(bundle_root: Path, raw_path: str | Path) -> Path:
    original = Path(raw_path)
    normalized = str(original).replace("\\", "/")
    marker = "/PointRCNN/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        packaged = bundle_root / "project_data" / Path(relative)
        if packaged.exists():
            return packaged
    if original.exists():
        return original
    raise FileNotFoundError(f"Neither experiment nor packaged path exists: {original}")


@lru_cache(maxsize=10)
def read_bin_cached(path_text: str) -> np.ndarray:
    path = Path(path_text)
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"{path} is not an Nx4 float32 KITTI cloud")
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN or Inf")
    return points


def read_bin(path: Path) -> np.ndarray:
    return read_bin_cached(str(path.resolve()))


def read_png_size(path: Path) -> tuple[int, int]:
    import struct

    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Expected PNG: {path}")
    return struct.unpack(">II", header[16:24])


def read_calibration(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        values[key] = np.asarray([float(item) for item in raw.split()])
    return (
        values["P2"].reshape(3, 4),
        values["R0_rect"].reshape(3, 3),
        values["Tr_velo_to_cam"].reshape(3, 4),
    )


def centerpoint_fov_range(points: np.ndarray, frame: str, bundle_root: Path) -> np.ndarray:
    """Replay the original lab audit's float32 KITTI projection exactly."""
    import importlib.util
    source = bundle_root / "source_scripts/calibration.py"
    spec = importlib.util.spec_from_file_location("archived_lab_calibration", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = bundle_root / "project_data/data/KITTI/object/training"
    calibration = module.Calibration(str(base / "calib" / f"{frame}.txt"))
    width, height = read_png_size(base / "image_2" / f"{frame}.png")
    rect = calibration.lidar_to_rect(points[:, :3])
    image_xy, depth = calibration.rect_to_img(rect)
    valid = (
        (image_xy[:, 0] >= 0.0) & (image_xy[:, 0] < width)
        & (image_xy[:, 1] >= 0.0) & (image_xy[:, 1] < height)
        & (depth >= 0.0)
        & (points[:, 0] >= POINT_RANGE[0]) & (points[:, 0] < POINT_RANGE[3])
        & (points[:, 1] >= POINT_RANGE[1]) & (points[:, 1] < POINT_RANGE[4])
        & (points[:, 2] >= POINT_RANGE[2]) & (points[:, 2] < POINT_RANGE[5])
    )
    return points[valid]


def centerpoint_effective_points(
    points: np.ndarray, frame: str, bundle_root: Path
) -> np.ndarray:
    selected = centerpoint_fov_range(points, frame, bundle_root)
    coords = np.floor((selected[:, :3] - POINT_RANGE[:3]) / VOXEL_SIZE).astype(
        np.int32
    )
    voxel_counts: dict[tuple[int, int, int], int] = {}
    retained: list[int] = []
    for index, coord in enumerate(coords):
        key = int(coord[0]), int(coord[1]), int(coord[2])
        count = voxel_counts.get(key)
        if count is None:
            if len(voxel_counts) >= MAX_VOXELS:
                continue
            voxel_counts[key] = 1
            retained.append(index)
        elif count < MAX_POINTS_PER_VOXEL:
            voxel_counts[key] = count + 1
            retained.append(index)
    return selected[np.asarray(retained, dtype=np.int64)]


def car_basis(box: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners = np.asarray(box["corners_lidar"], dtype=np.float64)
    center = corners.mean(axis=0)
    edge_a = corners[1] - corners[0]
    edge_b = corners[3] - corners[0]
    edge_h = corners[4] - corners[0]
    if np.linalg.norm(edge_a) >= np.linalg.norm(edge_b):
        long_edge, width_edge = edge_a, edge_b
    else:
        long_edge, width_edge = edge_b, edge_a
    long_axis = long_edge / np.linalg.norm(long_edge)
    width_axis = width_edge / np.linalg.norm(width_edge)
    height_axis = edge_h / np.linalg.norm(edge_h)
    # Keep the displayed height axis pointing upward.
    if height_axis[2] < 0:
        height_axis = -height_axis
    basis = np.vstack((long_axis, width_axis, height_axis))
    local_corners = (corners - center) @ basis.T
    return center, basis, local_corners


def to_local(points_xyz: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (points_xyz - center) @ basis.T


def crop_local(
    points: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    extents: np.ndarray,
) -> np.ndarray:
    local = to_local(points[:, :3].astype(np.float64), center, basis)
    limits = extents + np.asarray([1.8, 1.5, 0.8])
    mask = np.all(np.abs(local) <= limits, axis=1)
    return local[mask]


def inside_box(local: np.ndarray, extents: np.ndarray, pad: float = 0.08) -> np.ndarray:
    return np.all(np.abs(local) <= extents + pad, axis=1)


def sample_points(points: np.ndarray, limit: int = 12_000) -> np.ndarray:
    if len(points) <= limit:
        return points
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return points[indices]


def quality_metrics(bundle_root: Path, manifest: dict, gt_index: int) -> dict:
    gt = manifest["baseline_audit"]["gt"][gt_index]
    match = manifest["baseline_audit"]["match_by_gt"][str(gt_index)]
    original = read_bin(resolve_packaged_path(bundle_root, manifest["paths"]["original_reference"]))
    center, basis, local_corners = car_basis(gt)
    extents = np.max(np.abs(local_corners), axis=0)
    local = crop_local(original, center, basis, extents)
    object_points = int(inside_box(local, extents).sum())
    truncation = max(0.0, float(gt.get("truncation", 0.0)))
    occlusion = max(0, int(gt.get("occlusion", 0)))
    distance = float(np.linalg.norm(center[:2]))
    iou3d = float(match["iou3d"])
    score = float(match["score"])
    # Point support dominates; box quality and low truncation/occlusion break ties.
    quality = (
        4.0 * math.log1p(object_points)
        + 8.0 * iou3d
        + 0.15 * score
        - 3.0 * truncation
        - 0.7 * occlusion
        - 0.015 * distance
    )
    return {
        "gt": gt,
        "match": match,
        "object_points_original": object_points,
        "truncation": truncation,
        "occlusion": occlusion,
        "distance_m": distance,
        "quality_score": quality,
    }


def effective_inputs(
    bundle_root: Path, manifest: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = {
        key: resolve_packaged_path(bundle_root, value)
        for key, value in manifest["paths"].items()
    }
    original = read_bin(paths["original_reference"])
    detector = manifest["protocol"]["detector"]
    if detector == "pointrcnn":
        baseline = read_bin(paths["baseline_e2"])
        upsampled = read_bin(paths["upsampled_e2"])
    else:
        observed = read_bin(paths["observed"])
        e1 = read_bin(paths["upsampled_e1"])
        frame = manifest["frame_id"]
        baseline = centerpoint_effective_points(observed, frame, bundle_root)
        upsampled = centerpoint_effective_points(e1, frame, bundle_root)
    return original, baseline, upsampled


def draw_box_3d(ax, corners: np.ndarray, color: str, linewidth: float, linestyle: str) -> None:
    for start, end in BOX_EDGES:
        ax.plot(
            [corners[start, 0], corners[end, 0]],
            [corners[start, 1], corners[end, 1]],
            [corners[start, 2], corners[end, 2]],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            solid_capstyle="round",
            zorder=8,
        )


def draw_box_bev(ax, corners: np.ndarray, color: str, linewidth: float, linestyle: str) -> None:
    order = [0, 1, 2, 3, 0]
    ax.plot(
        corners[order, 0],
        corners[order, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
        zorder=8,
    )


def scatter_points_3d(ax, points: np.ndarray, extents: np.ndarray, color: str) -> None:
    points = sample_points(points)
    mask = inside_box(points, extents)
    context = points[~mask]
    inside = points[mask]
    if len(context):
        ax.scatter(
            context[:, 0], context[:, 1], context[:, 2],
            s=2.0, c="#B8BEC7", alpha=0.28, depthshade=False, rasterized=True,
        )
    if len(inside):
        ax.scatter(
            inside[:, 0], inside[:, 1], inside[:, 2],
            s=7.0, c=color, alpha=0.92, depthshade=False, rasterized=True,
        )


def scatter_points_bev(ax, points: np.ndarray, extents: np.ndarray, color: str) -> None:
    points = sample_points(points)
    mask = inside_box(points, extents)
    context = points[~mask]
    inside = points[mask]
    if len(context):
        ax.scatter(
            context[:, 0], context[:, 1],
            s=2.0, c="#B8BEC7", alpha=0.25, rasterized=True,
        )
    if len(inside):
        ax.scatter(
            inside[:, 0], inside[:, 1],
            s=7.0, c=color, alpha=0.92, rasterized=True,
        )


def style_3d(ax, limits: np.ndarray) -> None:
    ax.set_xlim(-limits[0], limits[0])
    ax.set_ylim(-limits[1], limits[1])
    ax.set_zlim(-limits[2], limits[2])
    ax.set_box_aspect((2 * limits[0], 2 * limits[1], 2 * limits[2]))
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("car length (m)", fontsize=8)
    ax.set_ylabel("car width (m)", fontsize=8)
    ax.set_zlabel("height (m)", fontsize=8)
    ax.tick_params(labelsize=7, pad=0)
    ax.grid(True, alpha=0.18)


def style_bev(ax, limits: np.ndarray) -> None:
    ax.set_xlim(-limits[0], limits[0])
    ax.set_ylim(-limits[1], limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("car length (m)", fontsize=8)
    ax.set_ylabel("car width (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.18)


def render_object(
    bundle_root: Path,
    manifest: dict,
    selection: dict,
    output_path: Path,
) -> dict:
    gt_index = int(selection["gt_index"])
    gt = manifest["baseline_audit"]["gt"][gt_index]
    match = manifest["baseline_audit"]["match_by_gt"][str(gt_index)]
    pred = manifest["baseline_audit"]["pred"][int(match["pred_idx"])]
    original, baseline, upsampled = effective_inputs(bundle_root, manifest)

    center, basis, gt_corners = car_basis(gt)
    pred_corners = to_local(
        np.asarray(pred["corners_lidar"], dtype=np.float64), center, basis
    )
    extents = np.max(np.abs(gt_corners), axis=0)
    limits = extents + np.asarray([1.8, 1.5, 0.8])
    clouds = [
        crop_local(original, center, basis, extents),
        crop_local(baseline, center, basis, extents),
        crop_local(upsampled, center, basis, extents),
    ]

    figure = plt.figure(figsize=(16, 10), dpi=150, facecolor="white")
    axes_3d = [figure.add_subplot(2, 3, index + 1, projection="3d") for index in range(3)]
    axes_bev = [figure.add_subplot(2, 3, index + 4) for index in range(3)]

    titles = [
        "Original full-density reference\nvehicle-shape check",
        f"Before upsampling: {manifest['protocol']['baseline_variant']}\nmatched detection present",
        f"After upsampling: {manifest['protocol']['upsampled_variant']}\nmatched detection absent",
    ]
    point_colors = ["#243B53", "#243B53", "#1F77B4"]
    for index, (ax3d, axbev, cloud, title, color) in enumerate(
        zip(axes_3d, axes_bev, clouds, titles, point_colors)
    ):
        scatter_points_3d(ax3d, cloud, extents, color)
        scatter_points_bev(axbev, cloud, extents, color)
        if index < 2:
            draw_box_3d(ax3d, gt_corners, "#D89B00", 2.2, "--")
            draw_box_bev(axbev, gt_corners, "#D89B00", 2.2, "--")
        else:
            draw_box_3d(ax3d, gt_corners, "#D62728", 2.6, "--")
            draw_box_bev(axbev, gt_corners, "#D62728", 2.6, "--")
        if index == 1:
            draw_box_3d(ax3d, pred_corners, "#168A45", 2.8, "-")
            draw_box_bev(axbev, pred_corners, "#168A45", 2.8, "-")
        style_3d(ax3d, limits)
        style_bev(axbev, limits)
        ax3d.set_title(title, fontsize=11, fontweight="bold", pad=10)

    axes_3d[1].text2D(
        0.03, 0.03,
        f"baseline score={float(match['score']):.3f}\n3D IoU={float(match['iou3d']):.3f}",
        transform=axes_3d[1].transAxes,
        fontsize=9,
        color="#0B5D2A",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#168A45", alpha=0.92),
    )
    axes_3d[2].text2D(
        0.03, 0.03,
        "LOST AFTER UPSAMPLING\nmatched prediction: NONE",
        transform=axes_3d[2].transAxes,
        fontsize=9,
        fontweight="bold",
        color="#A11212",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF2F2", edgecolor="#D62728", alpha=0.96),
    )

    protocol = manifest["protocol"]
    gt_id = gt.get("id", gt_index)
    figure.suptitle(
        f"Frame {manifest['frame_id']} | Line {protocol['line']} | {protocol['method']} | "
        f"{protocol['detector']} | Car GT id {gt_id} (eval index {gt_index})",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    legend = [
        Line2D([0], [0], color="#D89B00", lw=2.2, ls="--", label="GT box before upsampling"),
        Line2D([0], [0], color="#168A45", lw=2.8, label="matched baseline detection box"),
        Line2D([0], [0], color="#D62728", lw=2.6, ls="--", label="GT box after upsampling; detection lost"),
    ]
    figure.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False, fontsize=10)
    figure.text(
        0.5,
        0.015,
        (
            f"distance={selection['distance_m']:.1f} m | original points inside GT="
            f"{selection['object_points_original']} | truncation={selection['truncation']:.2f} | "
            f"occlusion={selection['occlusion']} | transition=lost_after_upsampling"
        ),
        ha="center",
        va="bottom",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.07, wspace=0.18, hspace=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)

    return {
        "original_crop_points": len(clouds[0]),
        "baseline_crop_points": len(clouds[1]),
        "upsampled_crop_points": len(clouds[2]),
    }


def manifest_path(report_root: Path, case: dict[str, str]) -> Path:
    return (
        report_root
        / "frames"
        / case["frame_id"]
        / f"line_{case['line'].lower()}"
        / METHOD_SLUG[case["method"]]
        / case["detector"]
        / "case_manifest.json"
    )


def case_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return row["frame_id"], row["line"], row["method"], row["detector"]


def status_folder(available: int) -> str:
    if available >= 2:
        return "2_objects_found"
    if available == 1:
        return "1_object_only"
    return "0_object_no_evidence"


def build_gallery(bundle_root: Path, report_root: Path, output_root: Path) -> None:
    case_rows = read_csv(report_root / "case_index.csv")
    transition_rows = read_csv(report_root / "analysis/selected_frame_gt_transitions.csv")
    candidates_by_case: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in transition_rows:
        if row["transition"] == "lost_after_upsampling" and row["class"] == "Car":
            candidates_by_case[case_key(row)].append(row)

    output_root.mkdir(parents=True, exist_ok=False)
    coverage_rows: list[dict] = []
    selection_rows: list[dict] = []
    shortage_rows: list[dict] = []

    for case_number, case in enumerate(case_rows, start=1):
        key = case_key(case)
        path = manifest_path(report_root, case)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ranked: list[dict] = []
        for candidate in candidates_by_case.get(key, []):
            gt_index = int(candidate["gt_index"])
            metrics = quality_metrics(bundle_root, manifest, gt_index)
            ranked.append(
                {
                    **candidate,
                    **{name: value for name, value in metrics.items() if name not in {"gt", "match"}},
                    "gt_id": metrics["gt"].get("id", gt_index),
                    "baseline_score": float(metrics["match"]["score"]),
                    "baseline_bev_iou": float(metrics["match"]["bev_iou"]),
                    "baseline_3d_iou": float(metrics["match"]["iou3d"]),
                }
            )
        ranked.sort(
            key=lambda row: (
                row["quality_score"],
                row["object_points_original"],
                row["baseline_3d_iou"],
            ),
            reverse=True,
        )
        selected = ranked[:2]
        available = len(ranked)
        status = status_folder(available)
        relative_case_dir = Path(status) / f"frame_{case['frame_id']}" / f"line_{case['line']}" / METHOD_SLUG[case["method"]] / case["detector"]
        case_dir = output_root / relative_case_dir
        case_dir.mkdir(parents=True, exist_ok=True)

        if available < 2:
            missing = 2 - min(available, 2)
            note = textwrap.dedent(
                f"""\
                # Evidence gap: {case['frame_id']} / Line {case['line']} / {case['method']} / {case['detector']}

                Requested: two Car objects that were detected before upsampling and not detected after upsampling.

                Available in the audited selected-frame transition table: {available}.
                Missing: {missing}.

                No other class and no non-lost transition was substituted. This folder preserves the observed evidence boundary.
                """
            )
            (case_dir / "README.md").write_text(note, encoding="utf-8")
            shortage_rows.append(
                {
                    "frame_id": case["frame_id"],
                    "line": case["line"],
                    "method": case["method"],
                    "detector": case["detector"],
                    "available_lost_cars": available,
                    "missing_from_requested_two": missing,
                    "relative_folder": str(relative_case_dir).replace("\\", "/"),
                }
            )

        for rank, selection in enumerate(selected, start=1):
            filename = (
                f"rank_{rank:02d}_gt_{int(selection['gt_id']):03d}_eval_{int(selection['gt_index']):03d}_"
                "Car_lost_after_upsampling.png"
            )
            image_path = case_dir / filename
            crop_counts = render_object(bundle_root, manifest, selection, image_path)
            selection_rows.append(
                {
                    "frame_id": case["frame_id"],
                    "line": case["line"],
                    "method": case["method"],
                    "detector": case["detector"],
                    "rank": rank,
                    "gt_id": selection["gt_id"],
                    "gt_index": selection["gt_index"],
                    "transition": "lost_after_upsampling",
                    "baseline_score": f"{selection['baseline_score']:.6f}",
                    "baseline_bev_iou": f"{selection['baseline_bev_iou']:.6f}",
                    "baseline_3d_iou": f"{selection['baseline_3d_iou']:.6f}",
                    "distance_m": f"{selection['distance_m']:.3f}",
                    "original_points_inside_gt": selection["object_points_original"],
                    "truncation": f"{selection['truncation']:.3f}",
                    "occlusion": selection["occlusion"],
                    "quality_score": f"{selection['quality_score']:.6f}",
                    **crop_counts,
                    "image": str(image_path.relative_to(output_root)).replace("\\", "/"),
                }
            )

        coverage_rows.append(
            {
                "frame_id": case["frame_id"],
                "line": case["line"],
                "method": case["method"],
                "detector": case["detector"],
                "available_lost_cars": available,
                "selected": len(selected),
                "requested": 2,
                "status": status,
                "relative_folder": str(relative_case_dir).replace("\\", "/"),
            }
        )
        print(
            f"[{case_number:02d}/{len(case_rows)}] {case['frame_id']} Line {case['line']} "
            f"{case['method']} {case['detector']}: {len(selected)}/2 selected"
        )

    write_csv(
        output_root / "coverage_summary.csv",
        coverage_rows,
        [
            "frame_id", "line", "method", "detector", "available_lost_cars",
            "selected", "requested", "status", "relative_folder",
        ],
    )
    write_csv(
        output_root / "selection_index.csv",
        selection_rows,
        [
            "frame_id", "line", "method", "detector", "rank", "gt_id", "gt_index",
            "transition", "baseline_score", "baseline_bev_iou", "baseline_3d_iou",
            "distance_m", "original_points_inside_gt", "truncation", "occlusion",
            "quality_score", "original_crop_points", "baseline_crop_points",
            "upsampled_crop_points", "image",
        ],
    )
    write_csv(
        output_root / "insufficient_evidence.csv",
        shortage_rows,
        [
            "frame_id", "line", "method", "detector", "available_lost_cars",
            "missing_from_requested_two", "relative_folder",
        ],
    )

    counts = Counter(row["status"] for row in coverage_rows)
    readme = textwrap.dedent(
        f"""\
        # Lost-after-upsampling Car object gallery

        This folder contains evidence-backed screenshots for every combination of:

        - frames: `000590`, `005625`, `006682`
        - experiments: Line A and Line B
        - methods: PDANS, PU-GCN, PU-EdgeFormer, PU-Net
        - detectors: PointRCNN and CenterPoint

        ## Coverage

        - requested combinations: {len(coverage_rows)}
        - combinations with two qualifying Cars: {counts['2_objects_found']}
        - combinations with one qualifying Car: {counts['1_object_only']}
        - combinations with no qualifying Car: {counts['0_object_no_evidence']}
        - rendered Car images: {len(selection_rows)}

        All shortfalls occur in frame `006682`. No Pedestrian/Cyclist case and no
        localization/confidence-only degradation was relabeled as a lost Car.

        ## Figure layout

        Every PNG uses the same 2x3 layout. The top row is a 3D oblique view and
        the bottom row is BEV. Columns are:

        1. original full-density reference, for vehicle-shape verification;
        2. detector baseline input with the matched green prediction box;
        3. detector upsampled input with no matched prediction box.

        Yellow dashed = GT before upsampling; green solid = matched baseline
        prediction; red dashed = GT after upsampling where the detection was lost.

        ## Index files

        - `coverage_summary.csv`: all 48 combinations and their coverage status.
        - `selection_index.csv`: the exact 0-2 selected objects and quality metrics.
        - `insufficient_evidence.csv`: combinations that cannot supply two real lost Cars.

        Selection prioritizes original-reference point support, baseline 3D IoU and
        score, low truncation/occlusion, and closer distance. These images explain
        selected cases; formal AP remains the full-validation-set metric.
        """
    )
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    print(f"output={output_root}")
    print(f"rendered_images={len(selection_rows)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(
            f"Output already exists: {args.output}. Choose a new --output path; "
            "the script never deletes or overwrites an existing gallery."
        )
    build_gallery(args.bundle_root.resolve(), args.report_root.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
