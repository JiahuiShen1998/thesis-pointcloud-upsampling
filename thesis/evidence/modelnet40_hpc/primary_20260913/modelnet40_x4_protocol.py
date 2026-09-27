#!/usr/bin/env python3
"""Shared constants and helpers for ModelNet40 ×4 two-line protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_original"
DOWNSAMPLED_X4_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled_x4"
LEGACY_DOWNSAMPLED50_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50"
MESH_ROOT = Path(
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/data/raw/ModelNet40"
)

GLOBAL_SEED = 42
UP_FACTOR = 4
EXPECTED_TRAIN = 9843
EXPECTED_TEST = 2468
EXPECTED_CLASSES = 40

MAIN_METHODS = ("ear", "pu_net", "pu_gcn", "pdans")
# Extended methods registered into pipeline after authenticity verification.
EXTENDED_METHODS = ("pu_edgeformer",)
ALL_PIPELINE_METHODS = MAIN_METHODS + EXTENDED_METHODS
METHOD_ALIASES = {
    "ear": "ear",
    "pu_net": "punet",
    "pu_gcn": "pugcn",
    "pdans": "pdans",
    "punet": "punet",
    "pugcn": "pugcn",
    "pu_edgeformer": "pu_edgeformer",
}

# Legacy x4 upsampling output subdirs (1024→4096 / 512→2048 old protocol)
LEGACY_LINEA_UP_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_original_up"
LEGACY_LINEB_UP_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up"


def detect_original_point_count(original_root: Path | None = None) -> int:
    root = original_root or ORIGINAL_ROOT
    for npy in sorted(root.rglob("*.npy")):
        arr = np.load(npy)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return int(arr.shape[0])
    raise RuntimeError(f"No valid .npy found under {root}")


def protocol_counts(n: int | None = None) -> dict[str, int]:
    n = n or detect_original_point_count()
    return {
        "N": n,
        "N_div_4": n // UP_FACTOR,
        "four_N": n * UP_FACTOR,
    }


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


def farthest_point_sample(points: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
    """Deterministic FPS: first index from seeded RNG, then iterative farthest."""
    n = points.shape[0]
    if n_samples >= n:
        return points.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    indices = np.zeros(n_samples, dtype=np.int64)
    indices[0] = int(rng.integers(0, n))
    dists = np.full(n, np.inf, dtype=np.float64)
    last = indices[0]
    for i in range(1, n_samples):
        diff = points - points[last]
        dists = np.minimum(dists, np.sum(diff * diff, axis=1))
        indices[i] = int(np.argmax(dists))
        last = indices[i]
    return points[indices].astype(np.float32)


def load_class_to_idx() -> dict[str, int]:
    path = ORIGINAL_ROOT / "metadata" / "class_to_idx.json"
    return json.loads(path.read_text(encoding="utf-8"))


def method_dir_name(method: str) -> str:
    m = method.strip().lower().replace("-", "_")
    if m in ALL_PIPELINE_METHODS:
        return m
    raise ValueError(f"Unknown method: {method}")


def legacy_method_subdir(method: str) -> str:
    return f"{METHOD_ALIASES[method_dir_name(method)]}_x4"


def lineA_paths(method: str | None = None) -> dict[str, Path]:
    base_raw = PROJECT_ROOT / "datasets" / "lineA_original_up" / "raw"
    base_strict = PROJECT_ROOT / "datasets" / "lineA_original_up" / "strict_4N"
    if method is None:
        return {"raw_root": base_raw, "strict_root": base_strict}
    m = method_dir_name(method)
    return {
        "raw": base_raw / m,
        "strict_4N": base_strict / m,
    }


def lineB_paths(method: str | None = None) -> dict[str, Path]:
    base_raw = PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up" / "raw"
    base_strict = PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up" / "strict_N"
    if method is None:
        return {"raw_root": base_raw, "strict_root": base_strict}
    m = method_dir_name(method)
    return {
        "raw": base_raw / m,
        "strict_N": base_strict / m,
    }


def pointnet2_input_paths() -> dict[str, Path]:
    root = PROJECT_ROOT / "pointnet2_inputs"
    return {
        "lineA_baseline": root / "lineA_original_baseline",
        "lineA_up_root": root / "lineA_original_up",
        "lineB_baseline": root / "lineB_downsampled_x4_baseline",
        "lineB_up_root": root / "lineB_downsampled_x4_up",
    }


def reports_dir() -> Path:
    return PROJECT_ROOT / "reports"
