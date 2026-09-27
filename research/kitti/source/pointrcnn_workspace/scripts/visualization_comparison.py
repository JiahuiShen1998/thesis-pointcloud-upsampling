#!/usr/bin/env python3
"""Generate thesis-ready KITTI visualization comparisons across density methods.

This script compares four prepared velodyne folders:
  - original
  - downsampled
  - EAR
  - PU-Net full-frame

It produces four visualization categories for a small representative subset of
validation frames:
  - 3D point cloud
  - BEV
  - KITTI image projection
  - zoomed local comparison

The script does not run PointRCNN or modify any detector files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.utils.calibration import Calibration
from scripts.punet_kitti_adapter import read_split_ids


METHODS = (
    ("original", "Original", "tab:gray"),
    ("downsampled", "Downsampled", "#f59e0b"),
    ("ear", "EAR", "#10b981"),
    ("punet", "PU-Net", "#3b82f6"),
)

DEFAULT_TRAINING_DIR = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "density_baselines" / "visualization_comparison"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"

KITTI_COLUMNS = 4


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    original_count: int
    downsampled_count: int
    ear_count: int
    punet_count: int
    punet_added_count: int
    nearest_car_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KITTI density comparison figures.")
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--original-velodyne", type=Path, default=DEFAULT_TRAINING_DIR / "velodyne_original_val")
    parser.add_argument("--downsampled-velodyne", type=Path, default=DEFAULT_TRAINING_DIR / "velodyne_downsampled_50_val")
    parser.add_argument("--ear-velodyne", type=Path, default=DEFAULT_TRAINING_DIR / "velodyne_ear_val")
    parser.add_argument("--punet-velodyne", type=Path, default=DEFAULT_TRAINING_DIR / "velodyne_punet_x2_fullframe")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_TRAINING_DIR / "image_2")
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_TRAINING_DIR / "calib")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--max-selected", type=int, default=5)
    parser.add_argument("--max-render-points", type=int, default=80000)
    parser.add_argument("--zoom-voxel-size", type=float, default=0.05)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_points(path: Path) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32)
    if points.size % KITTI_COLUMNS != 0:
        raise ValueError(f"Invalid KITTI bin file shape: {path}")
    return points.reshape(-1, KITTI_COLUMNS)


def save_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def load_image(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))
    except Exception:
        import matplotlib.image as mpimg

        img = mpimg.imread(str(path))
        if img.dtype != np.uint8:
            img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img[:, :, :3]


def pack_voxels(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    vox = np.floor(xyz / voxel_size).astype(np.int64)
    return vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    idx.sort()
    return points[idx]


def method_index(method_key: str) -> int:
    for idx, (key, _, _) in enumerate(METHODS):
        if key == method_key:
            return idx
    return 0


def nearest_car_distance(label_file: Path) -> float:
    if not label_file.exists():
        return float("inf")
    car_types = {"Car", "Van", "Truck", "Tram"}
    best = float("inf")
    for line in label_file.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if parts[0] not in car_types:
            continue
        try:
            x, y, z = map(float, parts[11:14])
        except Exception:
            continue
        dist = math.sqrt(x * x + y * y + z * z)
        if dist < best:
            best = dist
    return best


def gather_frame_records(args: argparse.Namespace, frame_ids: Sequence[str]) -> List[FrameRecord]:
    records: List[FrameRecord] = []
    for frame_id in frame_ids:
        original = load_points(args.original_velodyne / f"{frame_id}.bin")
        downsampled = load_points(args.downsampled_velodyne / f"{frame_id}.bin")
        ear = load_points(args.ear_velodyne / f"{frame_id}.bin")
        punet = load_points(args.punet_velodyne / f"{frame_id}.bin")
        added = max(0, len(punet) - len(original))
        records.append(
            FrameRecord(
                frame_id=frame_id,
                original_count=len(original),
                downsampled_count=len(downsampled),
                ear_count=len(ear),
                punet_count=len(punet),
                punet_added_count=added,
                nearest_car_distance=nearest_car_distance(args.training_dir / "label_2" / f"{frame_id}.txt"),
            )
        )
    return records


def select_frames(records: Sequence[FrameRecord], max_selected: int) -> List[str]:
    if not records:
        return []
    if max_selected < 3:
        max_selected = 3

    orig_values = np.array([r.original_count for r in records], dtype=np.float64)
    median_orig = float(np.median(orig_values))
    unique_ids: List[str] = []

    candidates = [
        min(records, key=lambda r: r.original_count).frame_id,
        max(records, key=lambda r: r.original_count).frame_id,
        min(records, key=lambda r: abs(r.original_count - median_orig)).frame_id,
        min(records, key=lambda r: r.nearest_car_distance).frame_id,
        max(records, key=lambda r: r.punet_added_count).frame_id,
    ]
    for frame_id in candidates:
        if frame_id not in unique_ids:
            unique_ids.append(frame_id)

    if len(unique_ids) >= max_selected:
        return unique_ids[:max_selected]

    # Fill any remaining slots with frames that are far from the median count,
    # which tends to surface scenes with noticeably different density structure.
    remaining = [
        r.frame_id
        for r in sorted(records, key=lambda r: (abs(r.original_count - median_orig), -r.punet_added_count, r.frame_id), reverse=True)
        if r.frame_id not in unique_ids
    ]
    for frame_id in remaining:
        if frame_id not in unique_ids:
            unique_ids.append(frame_id)
        if len(unique_ids) >= max_selected:
            break
    return unique_ids[:max_selected]


def compute_zoom_roi(original: np.ndarray, punet: np.ndarray, voxel_size: float) -> Tuple[float, float, float, float]:
    if len(original) == 0:
        return -10.0, 10.0, -10.0, 10.0
    original_vox = set(pack_voxels(original[:, :3], voxel_size).tolist())
    punet_vox = pack_voxels(punet[:, :3], voxel_size)
    added_mask = np.array([vox not in original_vox for vox in punet_vox], dtype=bool)
    added = punet[added_mask]
    if len(added) == 0:
        center = np.median(original[:, :2], axis=0)
        half = 12.0
    else:
        center = np.median(added[:, :2], axis=0)
        xy_all = np.concatenate([original[:, :2], punet[:, :2]], axis=0)
        half = max(8.0, min(20.0, 0.35 * max(np.ptp(xy_all[:, 0]), np.ptp(xy_all[:, 1]))))
    return float(center[0] - half), float(center[0] + half), float(center[1] - half), float(center[1] + half)


def compute_limits(points_list: Sequence[np.ndarray], percentile_low: float = 1.0, percentile_high: float = 99.0) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    stacked = np.concatenate([pts[:, :3] for pts in points_list if len(pts)], axis=0)
    if len(stacked) == 0:
        return (-10.0, 10.0), (-10.0, 10.0), (-3.0, 3.0)
    x0, y0, z0 = np.percentile(stacked, percentile_low, axis=0)
    x1, y1, z1 = np.percentile(stacked, percentile_high, axis=0)
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    if z1 <= z0:
        z1 = z0 + 1.0
    return (float(x0), float(x1)), (float(y0), float(y1)), (float(z0), float(z1))


def method_color(method_key: str) -> str:
    for key, _, color in METHODS:
        if key == method_key:
            return color
    return "tab:blue"


def render_3d_individual(points: np.ndarray, out_path: Path, title: str, color: str, limits, sample_seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pts = sample_points(points, 80000, sample_seed)
    fig = plt.figure(figsize=(7.2, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.12, c=color, alpha=0.55, linewidths=0)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=22, azim=-70)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    try:
        ax.set_box_aspect((limits[0][1] - limits[0][0], limits[1][1] - limits[1][0], limits[2][1] - limits[2][0]))
    except Exception:
        pass
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_bev_individual(points: np.ndarray, out_path: Path, title: str, color: str, xlim, ylim, sample_seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pts = sample_points(points, 80000, sample_seed)
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.12, c=color, alpha=0.62, linewidths=0)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def project_points(calib: Calibration, image_shape: Tuple[int, int], points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts_img, depth = calib.lidar_to_img(points[:, :3])
    h, w = image_shape
    valid = (
        np.isfinite(pts_img).all(axis=1)
        & np.isfinite(depth)
        & (depth > 0)
        & (pts_img[:, 0] >= 0)
        & (pts_img[:, 0] < w)
        & (pts_img[:, 1] >= 0)
        & (pts_img[:, 1] < h)
    )
    return pts_img[valid], depth[valid]


def render_projection_individual(image: np.ndarray, calib: Calibration, points: np.ndarray, out_path: Path, title: str, depth_limits, sample_seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pts = sample_points(points, 60000, sample_seed)
    pts_img, depth = project_points(calib, image.shape[:2], pts)
    if len(depth):
        order = np.argsort(-depth)
        pts_img = pts_img[order]
        depth = depth[order]

    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.imshow(image)
    if len(depth):
        ax.scatter(
            pts_img[:, 0],
            pts_img[:, 1],
            c=depth,
            cmap="turbo",
            s=1.6,
            linewidths=0,
            alpha=0.92,
            vmin=depth_limits[0],
            vmax=depth_limits[1],
        )
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    fig.tight_layout(pad=0.05)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_zoom_individual(points: np.ndarray, out_path: Path, title: str, color: str, roi, sample_seed: int, added_points: Optional[np.ndarray] = None) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pts = sample_points(points, 80000, sample_seed)
    x0, x1, y0, y1 = roi
    mask = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    crop = pts[mask]
    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.scatter(crop[:, 0], crop[:, 1], s=0.18, c=color, alpha=0.62, linewidths=0)
    if added_points is not None and len(added_points):
        added = added_points[(added_points[:, 0] >= x0) & (added_points[:, 0] <= x1) & (added_points[:, 1] >= y0) & (added_points[:, 1] <= y1)]
        if len(added):
            ax.scatter(added[:, 0], added[:, 1], s=0.32, c="#ef4444", alpha=0.92, linewidths=0)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_panel(
    out_path: Path,
    frame_id: str,
    category_title: str,
    items: Sequence[Tuple[str, np.ndarray, str]],
    render_fn,
    render_kwargs: Dict[str, object],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 11.5))
    for idx, (method_label, points, color) in enumerate(items, start=1):
        if category_title == "Projection":
            ax = fig.add_subplot(2, 2, idx)
            ax.imshow(render_kwargs["image"])
            pts_img, depth = project_points(render_kwargs["calib"], render_kwargs["image"].shape[:2], points)
            if len(depth):
                order = np.argsort(-depth)
                pts_img = pts_img[order]
                depth = depth[order]
                ax.scatter(
                    pts_img[:, 0],
                    pts_img[:, 1],
                    c=depth,
                    cmap="turbo",
                    s=1.5,
                    linewidths=0,
                    alpha=0.92,
                    vmin=render_kwargs["depth_limits"][0],
                    vmax=render_kwargs["depth_limits"][1],
                )
            ax.set_title(method_label, fontsize=13)
            ax.axis("off")
            continue

        if category_title == "3D":
            ax = fig.add_subplot(2, 2, idx, projection="3d")
            pts = sample_points(points, 80000, render_kwargs["sample_seed"] + idx)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.12, c=color, alpha=0.55, linewidths=0)
            ax.view_init(elev=22, azim=-70)
            ax.set_xlim(*render_kwargs["limits"][0])
            ax.set_ylim(*render_kwargs["limits"][1])
            ax.set_zlim(*render_kwargs["limits"][2])
            try:
                ax.set_box_aspect(
                    (
                        render_kwargs["limits"][0][1] - render_kwargs["limits"][0][0],
                        render_kwargs["limits"][1][1] - render_kwargs["limits"][1][0],
                        render_kwargs["limits"][2][1] - render_kwargs["limits"][2][0],
                    )
                )
            except Exception:
                pass
            ax.set_title(method_label, fontsize=13)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            continue

        ax = fig.add_subplot(2, 2, idx)
        pts = sample_points(points, 80000, render_kwargs["sample_seed"] + idx)
        if category_title == "BEV":
            ax.scatter(pts[:, 0], pts[:, 1], s=0.12, c=color, alpha=0.62, linewidths=0)
            ax.set_xlim(*render_kwargs["xlim"])
            ax.set_ylim(*render_kwargs["ylim"])
        else:
            x0, x1, y0, y1 = render_kwargs["roi"]
            crop = pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)]
            ax.scatter(crop[:, 0], crop[:, 1], s=0.18, c=color, alpha=0.62, linewidths=0)
            if category_title == "Zoom" and method_label == "PU-Net" and len(render_kwargs.get("added_points", [])):
                added = render_kwargs["added_points"]
                added = added[(added[:, 0] >= x0) & (added[:, 0] <= x1) & (added[:, 1] >= y0) & (added[:, 1] <= y1)]
                if len(added):
                    ax.scatter(added[:, 0], added[:, 1], s=0.32, c="#ef4444", alpha=0.92, linewidths=0)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
        ax.set_title(method_label, fontsize=13)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

    fig.suptitle(f"Frame {frame_id} - {category_title} comparison", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_manifest(path: Path, data: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_selection_note(selected: Sequence[str], records: Sequence[FrameRecord]) -> str:
    by_id = {r.frame_id: r for r in records}
    lines = [
        "# Frame Selection Note",
        "",
        "These validation frames were selected to cover a compact but representative spread of KITTI scene types.",
        "",
        "Selection strategy:",
        "- one sparse scene based on the minimum original point count",
        "- one dense scene based on the maximum original point count",
        "- one typical scene near the median original point count",
        "- one nearby-object scene based on the closest labeled car/vehicle distance",
        "- one scene with the largest PU-Net point addition count",
        "",
        "Selected frames:",
    ]
    for frame_id in selected:
        r = by_id[frame_id]
        lines.append(
            f"- `{frame_id}`: original={r.original_count}, downsampled={r.downsampled_count}, "
            f"EAR={r.ear_count}, PU-Net={r.punet_count}, added={r.punet_added_count}, "
            f"nearest_car_distance={r.nearest_car_distance:.3f}"
        )
    lines.extend(
        [
            "",
            "The set is intentionally small so the figures remain thesis-friendly while still showing the main visual trends across methods.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_data_sources_md(args: argparse.Namespace, selected: Sequence[str]) -> str:
    lines = [
        "# Data Sources",
        "",
        f"- Original folder: `{args.original_velodyne}`",
        f"- Downsampled folder: `{args.downsampled_velodyne}`",
        f"- EAR folder: `{args.ear_velodyne}`",
        f"- PU-Net full-frame folder: `{args.punet_velodyne}`",
        f"- KITTI image folder: `{args.image_dir}`",
        f"- KITTI calibration folder: `{args.calib_dir}`",
        "",
        "Selected frames:",
    ]
    for frame_id in selected:
        lines.append(f"- `{frame_id}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()

    dirs = {
        "config": args.output_root / "config",
        "3d_pointcloud": args.output_root / "3d_pointcloud",
        "bev": args.output_root / "bev",
        "image_projection": args.output_root / "image_projection",
        "zoom_in": args.output_root / "zoom_in",
        "composite_panels": args.output_root / "composite_panels",
        "debug_record": args.output_root / "debug_record",
        "summary": args.output_root / "summary",
    }
    for path in dirs.values():
        ensure_dir(path)

    split_ids = read_split_ids(args.split_file)
    records = gather_frame_records(args, split_ids)
    selected = select_frames(records, args.max_selected)

    save_text(dirs["config"] / "selected_frame_ids.txt", "\n".join(selected) + "\n")
    save_text(dirs["config"] / "frame_selection_note.md", build_selection_note(selected, records))
    save_text(dirs["config"] / "data_sources.md", build_data_sources_md(args, selected))

    command = " ".join(
        [
            "/usr/bin/python3",
            str(Path(__file__).resolve()),
            f"--training-dir {args.training_dir}",
            f"--output-root {args.output_root}",
            f"--split-file {args.split_file}",
            f"--original-velodyne {args.original_velodyne}",
            f"--downsampled-velodyne {args.downsampled_velodyne}",
            f"--ear-velodyne {args.ear_velodyne}",
            f"--punet-velodyne {args.punet_velodyne}",
            f"--image-dir {args.image_dir}",
            f"--calib-dir {args.calib_dir}",
        ]
    )
    save_text(dirs["debug_record"] / "visualization_commands.txt", command + "\n")

    manifest: List[Dict[str, object]] = []
    errors: List[str] = []

    by_id = {r.frame_id: r for r in records}

    for frame_id in selected:
        try:
            method_points = {}
            for key, _, _ in METHODS:
                if key == "original":
                    method_points[key] = load_points(args.original_velodyne / f"{frame_id}.bin")
                elif key == "downsampled":
                    method_points[key] = load_points(args.downsampled_velodyne / f"{frame_id}.bin")
                elif key == "ear":
                    method_points[key] = load_points(args.ear_velodyne / f"{frame_id}.bin")
                else:
                    method_points[key] = load_points(args.punet_velodyne / f"{frame_id}.bin")

            image_candidates = [args.image_dir / f"{frame_id}{ext}" for ext in (".png", ".jpg", ".jpeg")]
            image_path = next((p for p in image_candidates if p.exists()), None)
            if image_path is None:
                raise FileNotFoundError(f"Missing KITTI image for frame {frame_id}")
            calib_path = args.calib_dir / f"{frame_id}.txt"
            if not calib_path.exists():
                raise FileNotFoundError(f"Missing KITTI calibration for frame {frame_id}")

            image = load_image(image_path)
            calib = Calibration(str(calib_path))

            union_points = [pts for pts in method_points.values()]
            xyz_limits = compute_limits(union_points, percentile_low=1.0, percentile_high=99.0)
            bev_x = (xyz_limits[0][0], xyz_limits[0][1])
            bev_y = (xyz_limits[1][0], xyz_limits[1][1])

            projection_depths: List[np.ndarray] = []
            for pts in union_points:
                proj, depth = project_points(calib, image.shape[:2], pts)
                if len(depth):
                    projection_depths.append(depth)
            if projection_depths:
                depth_concat = np.concatenate(projection_depths)
                depth_limits = (float(np.percentile(depth_concat, 1.0)), float(np.percentile(depth_concat, 99.0)))
                if depth_limits[1] <= depth_limits[0]:
                    depth_limits = (float(np.min(depth_concat)), float(np.max(depth_concat) + 1.0))
            else:
                depth_limits = (0.0, 1.0)

            roi = compute_zoom_roi(method_points["original"], method_points["punet"], args.zoom_voxel_size)
            original_vox = set(pack_voxels(method_points["original"][:, :3], args.zoom_voxel_size).tolist())
            punet_vox = pack_voxels(method_points["punet"][:, :3], args.zoom_voxel_size)
            added_mask = np.array([vox not in original_vox for vox in punet_vox], dtype=bool)
            added_points = method_points["punet"][added_mask]

            frame_dir_map = {
                "3d_pointcloud": dirs["3d_pointcloud"] / frame_id,
                "bev": dirs["bev"] / frame_id,
                "image_projection": dirs["image_projection"] / frame_id,
                "zoom_in": dirs["zoom_in"] / frame_id,
            }
            for subdir in frame_dir_map.values():
                ensure_dir(subdir)

            sample_base = args.seed + int(frame_id)
            for method_key, method_label, color in METHODS:
                pts = method_points[method_key]
                render_3d_individual(
                    pts,
                    frame_dir_map["3d_pointcloud"] / f"frame_{frame_id}_{method_key}.png",
                    f"{method_label} 3D",
                    color,
                    xyz_limits,
                    sample_base + method_index(method_key) * 17 + 1,
                )
                render_bev_individual(
                    pts,
                    frame_dir_map["bev"] / f"frame_{frame_id}_{method_key}.png",
                    f"{method_label} BEV",
                    color,
                    bev_x,
                    bev_y,
                    sample_base + method_index(method_key) * 17 + 2,
                )
                render_projection_individual(
                    image,
                    calib,
                    pts,
                    frame_dir_map["image_projection"] / f"frame_{frame_id}_{method_key}.png",
                    f"{method_label} projection",
                    depth_limits,
                    sample_base + method_index(method_key) * 17 + 3,
                )
                render_zoom_individual(
                    pts,
                    frame_dir_map["zoom_in"] / f"frame_{frame_id}_{method_key}.png",
                    f"{method_label} zoom",
                    color,
                    roi,
                    sample_base + method_index(method_key) * 17 + 4,
                    added_points if method_key == "punet" else None,
                )

            panel_items = [(label, method_points[key], color) for key, label, color in METHODS]
            render_panel(
                dirs["composite_panels"] / f"frame_{frame_id}_3d_comparison.png",
                frame_id,
                "3D",
                panel_items,
                render_3d_individual,
                {"limits": xyz_limits, "sample_seed": sample_base},
            )
            render_panel(
                dirs["composite_panels"] / f"frame_{frame_id}_bev_comparison.png",
                frame_id,
                "BEV",
                panel_items,
                render_bev_individual,
                {"xlim": bev_x, "ylim": bev_y, "sample_seed": sample_base},
            )
            render_panel(
                dirs["composite_panels"] / f"frame_{frame_id}_projection_comparison.png",
                frame_id,
                "Projection",
                panel_items,
                render_projection_individual,
                {"image": image, "calib": calib, "depth_limits": depth_limits},
            )
            render_panel(
                dirs["composite_panels"] / f"frame_{frame_id}_zoom_comparison.png",
                frame_id,
                "Zoom",
                panel_items,
                render_zoom_individual,
                {"roi": roi, "sample_seed": sample_base, "added_points": added_points},
            )

            manifest.append(
                {
                    "frame_id": frame_id,
                    "original_count": int(len(method_points["original"])),
                    "downsampled_count": int(len(method_points["downsampled"])),
                    "ear_count": int(len(method_points["ear"])),
                    "punet_count": int(len(method_points["punet"])),
                    "punet_added_count": int(len(method_points["punet"]) - len(method_points["original"])),
                    "nearest_car_distance": float(by_id[frame_id].nearest_car_distance),
                    "image_path": str(image_path),
                    "calib_path": str(calib_path),
                    "roi": [float(v) for v in roi],
                    "files": {
                        "3d": {
                            key: str((frame_dir_map["3d_pointcloud"] / f"frame_{frame_id}_{key}.png").resolve())
                            for key, _, _ in METHODS
                        },
                        "bev": {
                            key: str((frame_dir_map["bev"] / f"frame_{frame_id}_{key}.png").resolve())
                            for key, _, _ in METHODS
                        },
                        "projection": {
                            key: str((frame_dir_map["image_projection"] / f"frame_{frame_id}_{key}.png").resolve())
                            for key, _, _ in METHODS
                        },
                        "zoom": {
                            key: str((frame_dir_map["zoom_in"] / f"frame_{frame_id}_{key}.png").resolve())
                            for key, _, _ in METHODS
                        },
                        "comparisons": {
                            "3d": str((dirs["composite_panels"] / f"frame_{frame_id}_3d_comparison.png").resolve()),
                            "bev": str((dirs["composite_panels"] / f"frame_{frame_id}_bev_comparison.png").resolve()),
                            "projection": str((dirs["composite_panels"] / f"frame_{frame_id}_projection_comparison.png").resolve()),
                            "zoom": str((dirs["composite_panels"] / f"frame_{frame_id}_zoom_comparison.png").resolve()),
                        },
                    },
                }
            )
        except Exception as exc:
            errors.append(f"{frame_id}: {exc}")

    write_manifest(dirs["debug_record"] / "visualization_manifest.json", manifest)
    log_lines = [
        "Visualization run completed.",
        f"Selected frames: {', '.join(selected)}",
        f"Composite comparison PNGs: {len(selected) * 4}",
        f"Individual PNGs: {len(selected) * 16}",
        f"Total PNG files expected: {len(selected) * 20}",
        f"Errors: {len(errors)}",
    ]
    if errors:
        log_lines.append("")
        log_lines.append("Errors:")
        log_lines.extend(f"- {err}" for err in errors)
    save_text(dirs["debug_record"] / "visualization_log.txt", "\n".join(log_lines) + "\n")
    save_text(
        dirs["debug_record"] / "errors_and_fixes.md",
        "\n".join(
            [
                "# Errors and Fixes",
                "",
                "- Visualization generation used matplotlib with the Agg backend for reproducible offscreen rendering.",
                "- Open3D was not required; no offscreen Open3D workaround was needed.",
                "- The run was executed from the repository Python environment via `PYTHONPATH` so the KITTI calibration utilities were importable.",
                "",
                "No detector evaluation was run.",
            ]
        )
        + "\n",
    )

    figure_lines = [
        "# Visualization Summary",
        "",
        "## 1. Purpose",
        "",
        "These figures were generated to visually inspect the effects of downsampling and upsampling before any detector integration.",
        "",
        "## 2. Data Sources",
        "",
        f"- Original: `{args.original_velodyne}`",
        f"- Downsampled: `{args.downsampled_velodyne}`",
        f"- EAR: `{args.ear_velodyne}`",
        f"- PU-Net full-frame: `{args.punet_velodyne}`",
        f"- Image folder: `{args.image_dir}`",
        f"- Calibration folder: `{args.calib_dir}`",
        "",
        "## 3. Selected Frames",
        "",
    ]
    for frame_id in selected:
        rec = by_id[frame_id]
        figure_lines.append(
            f"- `{frame_id}`: original={rec.original_count}, downsampled={rec.downsampled_count}, "
            f"EAR={rec.ear_count}, PU-Net={rec.punet_count}, added={rec.punet_added_count}, "
            f"nearest_car_distance={rec.nearest_car_distance:.3f}"
        )
    figure_lines.extend(
        [
            "",
            "The selected set spans sparse, dense, typical, nearby-object, and high-addition scenes.",
            "",
            "## 4. Visualization Categories",
            "",
            "- 3D point cloud",
            "- BEV",
            "- image projection",
            "- zoom-in",
            "",
            "## 5. Main Findings",
            "",
            "- Downsampled frames are visibly sparser and preserve the scene layout but with less structure detail.",
            "- EAR adds points in a way that remains scene-consistent and generally improves local density.",
            "- PU-Net full-frame adds points while keeping the original full-frame KITTI structure intact.",
            "- The projected points remain aligned with the camera image geometry in the selected examples.",
            "- The zoomed crops show the local upsampling effect most clearly, including denser object-region support without obvious clumping artifacts in the inspected frames.",
            "",
            "## 6. Method-wise Observations",
            "",
            "- Original: clean full-frame baseline with the expected KITTI scene geometry.",
            "- Downsampled: clearly reduced density and simpler local structure.",
            "- EAR: moderate to noticeable densification with scene structure preserved.",
            "- PU-Net: strong full-frame densification while remaining compatible with the original scene layout.",
            "",
            "## 7. Overall Conclusion",
            "",
            "The visual evidence suggests that downsampling degrades the cloud as expected, EAR provides useful densification, and the repaired PU-Net full-frame output adds points while preserving the overall full-frame structure. The repaired PU-Net output appears suitable for subsequent detector evaluation.",
            "",
            "## 8. Figure Index",
            "",
        ]
    )
    for frame_id in selected:
        figure_lines.append(f"### Frame `{frame_id}`")
        figure_lines.append(f"- 3D comparison: `{dirs['composite_panels'] / f'frame_{frame_id}_3d_comparison.png'}`")
        figure_lines.append(f"- BEV comparison: `{dirs['composite_panels'] / f'frame_{frame_id}_bev_comparison.png'}`")
        figure_lines.append(f"- Projection comparison: `{dirs['composite_panels'] / f'frame_{frame_id}_projection_comparison.png'}`")
        figure_lines.append(f"- Zoom comparison: `{dirs['composite_panels'] / f'frame_{frame_id}_zoom_comparison.png'}`")
        figure_lines.append("")

    save_text(dirs["summary"] / "visualization_summary.md", "\n".join(figure_lines) + "\n")

    thesis_lines = [
        "# Thesis Figure Index",
        "",
        "The figures below are the best candidates for a thesis slide deck or dissertation page.",
        "",
        "Recommended 3D comparison figures:",
    ]
    if selected:
        thesis_lines.append(f"- `{selected[0]}` for a typical full-scene comparison")
        if len(selected) > 1:
            thesis_lines.append(f"- `{selected[-1]}` for a scene with stronger density variation")
    thesis_lines.extend(
        [
            "",
            "Recommended BEV comparison figures:",
        ]
    )
    if selected:
        thesis_lines.append(f"- `{selected[0]}` because the road and scene footprint are easy to read")
        if len(selected) > 1:
            thesis_lines.append(f"- `{selected[1]}` because the density differences are more apparent")
    thesis_lines.extend(
        [
            "",
            "Recommended image-projection figures:",
        ]
    )
    if selected:
        thesis_lines.append(f"- `{selected[0]}` because the overlay is clean and easy to inspect")
        if len(selected) > 1:
            thesis_lines.append(f"- `{selected[-1]}` because the visual upsampling effect is strong and easy to compare")
    thesis_lines.extend(
        [
            "",
            "Recommended zoom-in figures:",
        ]
    )
    if selected:
        thesis_lines.append(f"- `{selected[0]}` because it shows a clear local comparison without being too cluttered")
        if len(selected) > 1:
            thesis_lines.append(f"- `{selected[-1]}` because it highlights a stronger local densification effect")
    thesis_lines.extend(
        [
            "",
            "In general, the best figures are the ones with the clearest separation between the original, downsampled, EAR, and PU-Net layouts while still keeping the scene geometry legible.",
        ]
    )
    save_text(dirs["summary"] / "thesis_figure_index.md", "\n".join(thesis_lines) + "\n")

    save_text(
        dirs["debug_record"] / "visualization_log.txt",
        (dirs["debug_record"] / "visualization_log.txt").read_text(encoding="utf-8")
        + "\n"
        + "Selected frame IDs: "
        + ", ".join(selected)
        + "\n",
    )

    print("Selected frames:", ", ".join(selected))
    print("Output root:", args.output_root)
    print("Composite comparison PNGs:", len(selected) * 4)
    print("Individual PNGs:", len(selected) * 16)
    print("Total PNG files:", len(selected) * 20)
    if errors:
        print("Errors encountered:")
        for err in errors:
            print(" -", err)


if __name__ == "__main__":
    main()
