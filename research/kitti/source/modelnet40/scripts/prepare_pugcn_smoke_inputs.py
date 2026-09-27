#!/usr/bin/env python3
"""Prepare minimal ModelNet40 protocol smoke inputs (1024 + downsampled x4 256)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pugcn_paths import resolve_project_root  # noqa: E402


from off_sampler import sample_off  # noqa: E402


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.sqrt((points ** 2).sum(axis=1)).max()
    if scale > 0:
        points = points / scale
    return points


def deterministic_downsample_x4(points: np.ndarray, seed: int) -> np.ndarray:
    target = max(1, points.shape[0] // 4)
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=target, replace=False)
    return points[idx].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument(
        "--modelnet-root",
        default="/home/ra87racy/data/ModelNet40",
        help="source OFF tree for local smoke only",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args()

    project_root = resolve_project_root(Path(args.project_root) if args.project_root else None)
    mn_root = Path(args.modelnet_root)
    if not mn_root.is_dir():
        print(f"ERROR: modelnet root missing: {mn_root}")
        return 2

    line_a_root = project_root / "datasets" / "modelnet40_original_1024"
    line_b_root = project_root / "datasets" / "modelnet40_downsampled_x4"

    picks: list[tuple[str, str, Path]] = []
    for class_dir in sorted(p for p in mn_root.iterdir() if p.is_dir()):
        for split in ("train", "test"):
            split_dir = class_dir / split
            if not split_dir.is_dir():
                continue
            offs = sorted(split_dir.glob("*.off"))
            if offs:
                picks.append((split, class_dir.name, offs[0]))
            if len(picks) >= args.num_samples:
                break
        if len(picks) >= args.num_samples:
            break

    if not picks:
        print("ERROR: no OFF samples found")
        return 2

    for i, (split, cls, off_path) in enumerate(picks):
        pts1024 = sample_off(str(off_path), 1024, args.seed + i)
        pts256 = deterministic_downsample_x4(pts1024, args.seed + i + 1000)
        if pts256.shape[0] != 256:
            # pad/truncate to exact 256 for protocol
            if pts256.shape[0] > 256:
                pts256 = pts256[:256]
            else:
                pad = np.tile(pts256[:1], (256 - pts256.shape[0], 1))
                pts256 = np.vstack([pts256, pad])
        name = off_path.stem + ".npy"
        out_a = line_a_root / split / cls / name
        out_b = line_b_root / split / cls / name
        out_a.parent.mkdir(parents=True, exist_ok=True)
        out_b.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_a, pts1024)
        np.save(out_b, pts256)
        print(f"prepared {split}/{cls}/{name}")

    print(f"lineA root: {line_a_root} count={sum(1 for _ in line_a_root.rglob('*.npy'))}")
    print(f"lineB root: {line_b_root} count={sum(1 for _ in line_b_root.rglob('*.npy'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
