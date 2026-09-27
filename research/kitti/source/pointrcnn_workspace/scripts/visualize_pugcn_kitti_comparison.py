#!/usr/bin/env python3
"""Create visual comparisons for a PU-GCN KITTI single-frame workspace."""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "results" / "pugcn_single_frame"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "pugcn_visualization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize original, PU-GCN input, and PU-GCN output.")
    parser.add_argument("--frame-workspace", type=Path, default=None)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--max-points", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=1024)
    return parser.parse_args()


def load_xyzi(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        raw = np.fromfile(path, dtype=np.float32)
        arr = raw.reshape(-1, 4)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError("Expected point cloud array with at least xyz columns: %s" % path)
    return arr.astype(np.float32)


def sample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    idx.sort()
    return points[idx]


def find_workspace(args: argparse.Namespace) -> Path:
    if args.frame_workspace is not None:
        return args.frame_workspace
    if args.sample_id is not None:
        return args.input_root / args.sample_id
    manifests = sorted(args.input_root.glob("*/manifest.json"))
    if not manifests:
        raise FileNotFoundError("No PU-GCN single-frame workspace found under %s" % args.input_root)
    return manifests[0].parent


def maybe_load_output(workspace: Path) -> Optional[np.ndarray]:
    for path in [workspace / "pugcn_output_xyzi.npy", *sorted(workspace.glob("*_pugcn.bin"))]:
        if path.exists():
            return load_xyzi(path)
    return None


def render_3d(path: Path, points: np.ndarray, title: str, color: str, max_points: int, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pts = sample(points, max_points, seed)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.25, c=color, alpha=0.55)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=18, azim=-68)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_bev(path: Path, points: np.ndarray, title: str, color: str, max_points: int, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    pts = sample(points, max_points, seed)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.18, c=color, alpha=0.55)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_overlay(path: Path, input_points: np.ndarray, output_points: Optional[np.ndarray], max_points: int, seed: int) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    inp = sample(input_points, max_points, seed)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(inp[:, 0], inp[:, 1], s=0.22, c="tab:gray", alpha=0.45, label="PU-GCN input")
    if output_points is not None:
        out = sample(output_points, max_points, seed + 1)
        ax.scatter(out[:, 0], out[:, 1], s=0.12, c="tab:blue", alpha=0.45, label="PU-GCN output")
    ax.set_title("BEV overlay")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(markerscale=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    workspace = find_workspace(args)
    manifest_path = workspace / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    sample_id = str(manifest.get("sample_id", workspace.name))
    out_dir = args.output_root / sample_id

    input_points = load_xyzi(workspace / "input_sampled_xyzi.npy")
    output_points = maybe_load_output(workspace)
    render_3d(out_dir / "original_input.png", input_points, "PU-GCN selected KITTI input", "tab:gray", args.max_points, args.seed)
    render_bev(out_dir / "original_input_bev.png", input_points, "PU-GCN selected KITTI input BEV", "tab:gray", args.max_points, args.seed)
    if output_points is not None:
        render_3d(out_dir / "pugcn_output.png", output_points, "PU-GCN upsampled output", "tab:blue", args.max_points, args.seed)
        render_bev(out_dir / "pugcn_output_bev.png", output_points, "PU-GCN upsampled output BEV", "tab:blue", args.max_points, args.seed)
    render_overlay(out_dir / "overlay.png", input_points, output_points, args.max_points, args.seed)
    render_overlay(out_dir / "overlay_bev.png", input_points, output_points, args.max_points, args.seed)
    print("Wrote PU-GCN visualization images:", out_dir)


if __name__ == "__main__":
    main()
