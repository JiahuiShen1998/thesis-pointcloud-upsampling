#!/usr/bin/env python3
"""Deterministic strict point-count normalization for ModelNet40 PU-GCN outputs."""

import numpy as np


def _deterministic_fps(points: np.ndarray, target: int, seed: int) -> np.ndarray:
    """Farthest point sampling with deterministic tie-breaking via seeded RNG."""
    n = points.shape[0]
    if target >= n:
        return points.copy()
    rng = np.random.default_rng(seed)
    selected = np.empty((target,), dtype=np.int64)
    # Start from a deterministic index derived from seed.
    first = int(rng.integers(0, n))
    selected[0] = first
    dists = np.sum((points - points[first]) ** 2, axis=1)
    for i in range(1, target):
        idx = int(np.argmax(dists))
        selected[i] = idx
        new_d = np.sum((points - points[idx]) ** 2, axis=1)
        dists = np.minimum(dists, new_d)
    return points[selected].astype(np.float32)


def normalize_strict(raw_xyz: np.ndarray, target_points: int, seed: int) -> np.ndarray:
    raw_xyz = np.asarray(raw_xyz, dtype=np.float32)
    if raw_xyz.ndim != 2 or raw_xyz.shape[1] != 3:
        raise ValueError(f"raw output must be (N,3), got {raw_xyz.shape}")
    if not np.isfinite(raw_xyz).all():
        raise ValueError("raw output contains NaN or Inf")
    n = raw_xyz.shape[0]
    if n == 0:
        raise ValueError("raw output is empty")
    if n == target_points:
        return raw_xyz.copy()
    if n > target_points:
        return _deterministic_fps(raw_xyz, target_points, seed)
    # raw < target: deterministic nearest-neighbor duplication padding
    rng = np.random.default_rng(seed)
    out = np.empty((target_points, 3), dtype=np.float32)
    out[:n] = raw_xyz
    if n == 1:
        out[n:] = raw_xyz[0]
        return out
    # Pick seed points deterministically, then NN duplicate from raw set.
    base_idx = np.arange(n, dtype=np.int64)
    reps = target_points - n
    pick = rng.choice(base_idx, size=reps, replace=True)
    out[n:] = raw_xyz[pick]
    return out


def audit_pass(path, expected_points: int) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path)
        if arr.shape != (expected_points, 3):
            return False
        return bool(np.isfinite(arr).all())
    except Exception:
        return False
