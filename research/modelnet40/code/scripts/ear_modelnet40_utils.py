#!/usr/bin/env python3
"""Shared EAR ModelNet40 upsampling utilities (Step 7a/7b)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

DEFAULT_EAR_SCRIPT = Path(
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/EAR/ear_upsampling.py"
)
# Legacy ×2 wrapper (ablation / preliminary)
TARGET_POINTS = 1024
UP_FACTOR_X2 = 2.0

# Main protocol R=×4
MAIN_PROTOCOL_UPSAMPLING_FACTOR = 4
UP_FACTOR_X4 = 4.0
INPUT_POINTS_ORIGINAL = 1024
INPUT_POINTS_DOWN = 512
TARGET_POINTS_X4_ORIGINAL = 4096
TARGET_POINTS_X4_DOWN = 2048

GLOBAL_SEED = 42


def target_points_for_line(line: str, protocol: str = "x4") -> int:
    if protocol == "x2":
        return TARGET_POINTS
    if line in ("A", "original"):
        return TARGET_POINTS_X4_ORIGINAL
    if line in ("B", "downsampled50"):
        return TARGET_POINTS_X4_DOWN
    raise ValueError(f"Unknown line: {line}")


def up_factor_for_protocol(protocol: str = "x4") -> float:
    return UP_FACTOR_X2 if protocol == "x2" else UP_FACTOR_X4


def stable_seed(global_seed: int, *parts: str) -> int:
    token = ":".join([str(global_seed), *parts]).encode("utf-8")
    digest = hashlib.md5(token).hexdigest()
    return global_seed + (int(digest[:8], 16) % 1_000_000)


def resample_to_target(points: np.ndarray, target_n: int, seed: int) -> np.ndarray:
    count = points.shape[0]
    if count == target_n:
        return points.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    if count > target_n:
        indices = rng.choice(count, size=target_n, replace=False)
    else:
        indices = rng.choice(count, size=target_n, replace=True)
    return points[indices].astype(np.float32)


def load_ear_module(ear_script: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("ear_upsampling", ear_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import EAR module from {ear_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_ear_on_xyz(
    ear_module,
    points_xyz: np.ndarray,
    target_n: int,
    seed: int,
    k_neighbors: int = 20,
    edge_sensitivity: float = 4.0,
    up_threshold: float = 0.93,
    sigma_p: float = 0.0,
    up_factor: float = UP_FACTOR_X2,
    max_iter: int = 5,
) -> np.ndarray:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) xyz, got {points_xyz.shape}")
    intensity = np.zeros((points_xyz.shape[0], 1), dtype=np.float32)
    points_xyzi = np.hstack([points_xyz.astype(np.float32), intensity])
    out_xyzi = ear_module.ear_upsample_kitti(
        points_xyzi,
        target_n=target_n,
        up_factor=up_factor,
        k_neighbors=k_neighbors,
        edge_sensitivity=edge_sensitivity,
        up_threshold=up_threshold,
        sigma_p=sigma_p,
        max_iter=max_iter,
        seed=seed,
    )
    return out_xyzi[:, :3].astype(np.float32)


def is_valid_output_npy(path: Path, target_n: int = TARGET_POINTS) -> bool:
    if not path.is_file():
        return False
    try:
        arr = np.load(path)
        return arr.shape == (target_n, 3) and np.isfinite(arr).all()
    except Exception:
        return False


def ensure_metadata_link(output_root: Path, class_to_idx_src: Path) -> None:
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    dst = metadata_dir / "class_to_idx.json"
    if dst.is_symlink() or dst.exists():
        return
    try:
        dst.symlink_to(class_to_idx_src.resolve())
    except FileExistsError:
        pass
