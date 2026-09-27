#!/usr/bin/env python3
"""Visualize original and full-frame PU-GCN KITTI point clouds."""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_original_val" / "000001.bin"
DEFAULT_PUGCN = PROJECT_ROOT / "results" / "pugcn_full_frame" / "000001" / "patches_020" / "000001_pugcn_full_frame.bin"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "pugcn_visualization_full_frame"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize full-frame PU-GCN KITTI reconstruction.")
    parser.add_argument("--original_bin", "--original-bin", dest="original_bin", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--pugcn_bin", "--pugcn-bin", dest="pugcn_bin", type=Path, default=DEFAULT_PUGCN)
    parser.add_argument("--output_root", "--output-root", dest="output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frame_id", "--frame-id", dest="frame_id", type=str, default=None)
    parser.add_argument("--max_points", "--max-points", dest="max_points", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--zoom_center", "--zoom-center", dest="zoom_center", nargs=2, type=float, default=None)
    parser.add_argument("--zoom_half_size", "--zoom-half-size", dest="zoom_half_size", type=float, default=12.0)
    return parser.parse_args()


def load_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError("Invalid KITTI .bin shape: %s" % path)
    return raw.reshape(-1, 4)


def sample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    idx.sort()
    return points[idx]


def limits(points_a: np.ndarray, points_b: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xy = np.concatenate([points_a[:, :2], points_b[:, :2]], axis=0)
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def render_bev(path: Path, points: np.ndarray, title: str, color: str, xlim=None, ylim=None, max_points: int = 120000, seed: int = 1024) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pts = sample(points, max_points, seed)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.08, c=color, alpha=0.55)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def render_overlay(path: Path, original: np.ndarray, pugcn: np.ndarray, title: str, xlim=None, ylim=None, max_points: int = 120000, seed: int = 1024) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    orig = sample(original, max_points, seed)
    out = sample(pugcn, max_points, seed + 1)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(orig[:, 0], orig[:, 1], s=0.08, c="tab:gray", alpha=0.35, label="Original")
    ax.scatter(out[:, 0], out[:, 1], s=0.06, c="tab:blue", alpha=0.35, label="PU-GCN")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(markerscale=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def zoom_limits(args: argparse.Namespace, original: np.ndarray) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if args.zoom_center is None:
        xy = np.median(original[:, :2], axis=0)
    else:
        xy = np.asarray(args.zoom_center, dtype=np.float32)
    half = float(args.zoom_half_size)
    return (float(xy[0] - half), float(xy[0] + half)), (float(xy[1] - half), float(xy[1] + half))


def main() -> None:
    args = parse_args()
    frame_id = args.frame_id or args.original_bin.stem
    out_dir = args.output_root / frame_id
    original = load_bin(args.original_bin)
    pugcn = load_bin(args.pugcn_bin)
    xlim, ylim = limits(original, pugcn)
    render_bev(out_dir / "original_full_frame_bev.png", original, "Original KITTI full frame BEV", "tab:gray", xlim, ylim, args.max_points, args.seed)
    render_bev(out_dir / "pugcn_full_frame_bev.png", pugcn, "PU-GCN full-frame output BEV", "tab:blue", xlim, ylim, args.max_points, args.seed)
    render_overlay(out_dir / "overlay_full_frame_bev.png", original, pugcn, "Original vs PU-GCN full-frame BEV", xlim, ylim, args.max_points, args.seed)
    zx, zy = zoom_limits(args, original)
    render_overlay(out_dir / "overlay_zoom_bev.png", original, pugcn, "Original vs PU-GCN zoom BEV", zx, zy, args.max_points, args.seed)
    print("Wrote full-frame PU-GCN visualization images:", out_dir)


if __name__ == "__main__":
    main()
