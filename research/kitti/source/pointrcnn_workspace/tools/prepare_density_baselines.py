#!/usr/bin/env python3
"""Prepare density-variant KITTI velodyne folders for PointRCNN experiments.

This script only rewrites point clouds under the KITTI training tree. It does
not touch the detector, labels, calibration, images, or evaluation code.

Generated variants:
  - velodyne_upsampled
  - velodyne_downsampled
  - velodyne_down_up

The script also writes reproducibility records when ``--record-root`` is set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional dependency fallback
    cKDTree = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare KITTI density baselines")
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("data/KITTI/object/training"),
        help="KITTI training directory that contains velodyne and related folders.",
    )
    parser.add_argument(
        "--source-subdir",
        type=str,
        default="velodyne_original",
        help="Source point cloud folder. Falls back to velodyne if velodyne_original does not exist.",
    )
    parser.add_argument(
        "--record-root",
        type=Path,
        default=Path("experiments/density_baselines/_preprocessing_records"),
        help="Where to store preprocessing records for each generated variant.",
    )
    parser.add_argument(
        "--downsample-ratio",
        type=float,
        default=0.5,
        help="Random downsampling ratio for the sparse baseline.",
    )
    parser.add_argument(
        "--upsample-factor",
        type=float,
        default=1.5,
        help="Target multiplier for direct upsampling.",
    )
    parser.add_argument(
        "--down-up-factor",
        type=float,
        default=1.0,
        help="Target multiplier for down_up after downsampling. Default preserves the original count.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=8,
        help="K in KNN interpolation for synthetic points.",
    )
    parser.add_argument(
        "--interp-min",
        type=float,
        default=0.35,
        help="Minimum interpolation weight for synthetic point generation.",
    )
    parser.add_argument(
        "--interp-max",
        type=float,
        default=0.65,
        help="Maximum interpolation weight for synthetic point generation.",
    )
    parser.add_argument(
        "--jitter-std",
        type=float,
        default=0.01,
        help="Gaussian jitter standard deviation added to synthetic xyz points.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Global random seed used for reproducible sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for future parallelization. The current implementation runs serially for reproducibility.",
    )
    return parser.parse_args()


def load_points(path: Path) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"Invalid KITTI bin file shape: {path}")
    return points.reshape(-1, 4)


def save_points(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points.astype(np.float32, copy=False).tofile(path)


def ensure_source_dir(training_dir: Path, source_subdir: str) -> Path:
    preferred = training_dir / source_subdir
    fallback = training_dir / "velodyne"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Could not find {preferred} or {fallback}. The KITTI velodyne tree is missing."
    )


def list_bin_files(source_dir: Path) -> List[Path]:
    return sorted(source_dir.glob("*.bin"))


def random_downsample(points: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    if ratio >= 1.0 or len(points) == 0:
        return points.copy()
    target_n = max(1, int(round(len(points) * ratio)))
    if target_n >= len(points):
        return points.copy()
    indices = rng.choice(len(points), size=target_n, replace=False)
    indices.sort()
    return points[indices]


def _nearest_neighbor_upsample(
    points: np.ndarray,
    target_n: int,
    rng: np.random.Generator,
    knn_k: int,
    interp_min: float,
    interp_max: float,
    jitter_std: float,
) -> np.ndarray:
    if target_n <= len(points):
        indices = rng.choice(len(points), size=target_n, replace=False)
        indices.sort()
        return points[indices]

    extra_n = target_n - len(points)
    xyz = points[:, :3]
    if len(points) == 1:
        base = np.repeat(points, extra_n, axis=0)
        base[:, :3] += rng.normal(0.0, jitter_std, size=(extra_n, 3))
        return np.concatenate([points, base], axis=0)

    if cKDTree is None:
        return _bruteforce_upsample(points, target_n, rng, knn_k, interp_min, interp_max, jitter_std)

    k_eff = min(max(knn_k, 2), len(points))
    tree = cKDTree(xyz)
    anchor_idx = rng.integers(0, len(points), size=extra_n)
    query_xyz = xyz[anchor_idx]
    _, nn_idx = tree.query(query_xyz, k=k_eff)
    if k_eff == 1:
        nn_idx = nn_idx[:, None]
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]

    synth = np.empty((extra_n, 4), dtype=np.float32)
    for i in range(extra_n):
        candidates = np.asarray(nn_idx[i]).reshape(-1)
        if candidates.size == 0:
            candidates = np.array([anchor_idx[i]])
        if candidates.size == 1:
            a = b = int(candidates[0])
        else:
            a = int(candidates[0])
            b = int(candidates[rng.integers(1, candidates.size)])
        t = float(rng.uniform(interp_min, interp_max))
        synth_xyz = (1.0 - t) * points[a, :3] + t * points[b, :3]
        synth_xyz += rng.normal(0.0, jitter_std, size=3)
        synth_intensity = 0.5 * (points[a, 3] + points[b, 3])
        synth[i, :3] = synth_xyz.astype(np.float32)
        synth[i, 3] = np.float32(synth_intensity)

    return np.concatenate([points, synth], axis=0)


def _bruteforce_upsample(
    points: np.ndarray,
    target_n: int,
    rng: np.random.Generator,
    knn_k: int,
    interp_min: float,
    interp_max: float,
    jitter_std: float,
) -> np.ndarray:
    extra_n = target_n - len(points)
    xyz = points[:, :3]
    synth = np.empty((extra_n, 4), dtype=np.float32)
    for i in range(extra_n):
        anchor = int(rng.integers(0, len(points)))
        diff = xyz - xyz[anchor]
        dist2 = np.sum(diff * diff, axis=1)
        nn_idx = np.argsort(dist2)[: min(max(knn_k, 2), len(points))]
        if nn_idx.size == 1:
            a = b = int(nn_idx[0])
        else:
            a = int(nn_idx[0])
            b = int(nn_idx[rng.integers(1, nn_idx.size)])
        t = float(rng.uniform(interp_min, interp_max))
        synth_xyz = (1.0 - t) * points[a, :3] + t * points[b, :3]
        synth_xyz += rng.normal(0.0, jitter_std, size=3)
        synth_intensity = 0.5 * (points[a, 3] + points[b, 3])
        synth[i, :3] = synth_xyz.astype(np.float32)
        synth[i, 3] = np.float32(synth_intensity)
    return np.concatenate([points, synth], axis=0)


def point_stats(points: np.ndarray) -> Dict[str, float]:
    xyz = points[:, :3]
    intensity = points[:, 3]
    return {
        "num_points": int(points.shape[0]),
        "x_mean": float(np.mean(xyz[:, 0])) if len(points) else 0.0,
        "y_mean": float(np.mean(xyz[:, 1])) if len(points) else 0.0,
        "z_mean": float(np.mean(xyz[:, 2])) if len(points) else 0.0,
        "x_std": float(np.std(xyz[:, 0])) if len(points) else 0.0,
        "y_std": float(np.std(xyz[:, 1])) if len(points) else 0.0,
        "z_std": float(np.std(xyz[:, 2])) if len(points) else 0.0,
        "intensity_mean": float(np.mean(intensity)) if len(points) else 0.0,
        "intensity_std": float(np.std(intensity)) if len(points) else 0.0,
        "x_min": float(np.min(xyz[:, 0])) if len(points) else 0.0,
        "y_min": float(np.min(xyz[:, 1])) if len(points) else 0.0,
        "z_min": float(np.min(xyz[:, 2])) if len(points) else 0.0,
        "x_max": float(np.max(xyz[:, 0])) if len(points) else 0.0,
        "y_max": float(np.max(xyz[:, 1])) if len(points) else 0.0,
        "z_max": float(np.max(xyz[:, 2])) if len(points) else 0.0,
    }


def build_variant_records(
    source_dir: Path,
    output_dir: Path,
    record_dir: Path,
    variant_name: str,
    transform_note: str,
    rows: List[Dict[str, object]],
    params: Dict[str, object],
) -> None:
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / "preprocessing_params.json").write_text(
        json.dumps(
            {
                "variant_name": variant_name,
                "source_dir": str(source_dir.resolve()),
                "output_dir": str(output_dir.resolve()),
                "transform_note": transform_note,
                **params,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (record_dir / "preprocessing_log.txt").write_text(
        "\n".join(
            [
                f"variant={variant_name}",
                f"source_dir={source_dir.resolve()}",
                f"output_dir={output_dir.resolve()}",
                f"transform_note={transform_note}",
                f"num_files={len(rows)}",
                f"source_files={source_dir.resolve()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (record_dir / "file_list.txt").write_text(
        "\n".join(str(row["sample_id"]) for row in rows) + "\n",
        encoding="utf-8",
    )

    csv_path = record_dir / "point_statistics.csv"
    fieldnames = list(rows[0].keys()) if rows else ["sample_id", "status"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_variant(
    source_dir: Path,
    output_dir: Path,
    record_dir: Path,
    variant_name: str,
    downsample_ratio: float,
    upsample_factor: float,
    knn_k: int,
    interp_min: float,
    interp_max: float,
    jitter_std: float,
    seed: int,
    mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_paths = list_bin_files(source_dir)
    rows: List[Dict[str, object]] = []

    for file_path in file_paths:
        sample_id = file_path.stem
        rng = np.random.default_rng(seed + int(sample_id))
        original = load_points(file_path)
        source_stats = point_stats(original)

        if mode == "upsampled":
            target_n = max(1, int(math.ceil(len(original) * upsample_factor)))
            transformed = _nearest_neighbor_upsample(
                original,
                target_n=target_n,
                rng=rng,
                knn_k=knn_k,
                interp_min=interp_min,
                interp_max=interp_max,
                jitter_std=jitter_std,
            )
            downsampled_n = ""
            upsampled_n = transformed.shape[0]
            down_up_n = ""
        elif mode == "downsampled":
            transformed = random_downsample(original, downsample_ratio, rng)
            downsampled_n = transformed.shape[0]
            upsampled_n = ""
            down_up_n = ""
        elif mode == "down_up":
            downsampled = random_downsample(original, downsample_ratio, rng)
            target_n = max(1, int(math.ceil(len(original) * upsample_factor)))
            target_n = max(target_n, len(original))
            transformed = _nearest_neighbor_upsample(
                downsampled,
                target_n=target_n,
                rng=rng,
                knn_k=knn_k,
                interp_min=interp_min,
                interp_max=interp_max,
                jitter_std=jitter_std,
            )
            downsampled_n = downsampled.shape[0]
            upsampled_n = transformed.shape[0]
            down_up_n = transformed.shape[0]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        save_points(output_dir / f"{sample_id}.bin", transformed)
        target_stats = point_stats(transformed)
        rows.append(
            {
                "sample_id": sample_id,
                "mode": mode,
                "original_points": int(source_stats["num_points"]),
                "output_points": int(target_stats["num_points"]),
                "downsampled_points": downsampled_n,
                "upsampled_points": upsampled_n,
                "down_up_points": down_up_n,
                "original_intensity_mean": round(source_stats["intensity_mean"], 6),
                "output_intensity_mean": round(target_stats["intensity_mean"], 6),
                "original_x_mean": round(source_stats["x_mean"], 6),
                "original_y_mean": round(source_stats["y_mean"], 6),
                "original_z_mean": round(source_stats["z_mean"], 6),
                "output_x_mean": round(target_stats["x_mean"], 6),
                "output_y_mean": round(target_stats["y_mean"], 6),
                "output_z_mean": round(target_stats["z_mean"], 6),
            }
        )

    params = {
        "seed": seed,
        "downsample_ratio": downsample_ratio,
        "upsample_factor": upsample_factor,
        "knn_k": knn_k,
        "interp_min": interp_min,
        "interp_max": interp_max,
        "jitter_std": jitter_std,
        "mode": mode,
    }
    build_variant_records(
        source_dir=source_dir,
        output_dir=output_dir,
        record_dir=record_dir,
        variant_name=variant_name,
        transform_note=mode,
        rows=rows,
        params=params,
    )


def main() -> int:
    args = parse_args()
    training_dir = args.training_dir.resolve()
    source_dir = ensure_source_dir(training_dir, args.source_subdir).resolve()
    record_root = args.record_root.resolve()
    record_root.mkdir(parents=True, exist_ok=True)

    variants = [
        ("velodyne_upsampled", "upsampled", args.upsample_factor),
        ("velodyne_downsampled", "downsampled", args.downsample_ratio),
        ("velodyne_down_up", "down_up", args.down_up_factor),
    ]

    summary_lines = [
        "PointRCNN density baseline preprocessing",
        f"training_dir={training_dir}",
        f"source_dir={source_dir}",
        f"seed={args.seed}",
        f"downsample_ratio={args.downsample_ratio}",
        f"upsample_factor={args.upsample_factor}",
        f"down_up_factor={args.down_up_factor}",
        f"knn_k={args.knn_k}",
        f"interp_min={args.interp_min}",
        f"interp_max={args.interp_max}",
        f"jitter_std={args.jitter_std}",
    ]
    print("\n".join(summary_lines))

    for output_name, mode, factor in variants:
        output_dir = training_dir / output_name
        record_dir = record_root / output_name
        print(f"[prepare] generating {output_name} -> {output_dir}")
        if mode == "upsampled":
            process_variant(
                source_dir=source_dir,
                output_dir=output_dir,
                record_dir=record_dir,
                variant_name=output_name,
                downsample_ratio=args.downsample_ratio,
                upsample_factor=factor,
                knn_k=args.knn_k,
                interp_min=args.interp_min,
                interp_max=args.interp_max,
                jitter_std=args.jitter_std,
                seed=args.seed,
                mode=mode,
            )
        elif mode == "downsampled":
            process_variant(
                source_dir=source_dir,
                output_dir=output_dir,
                record_dir=record_dir,
                variant_name=output_name,
                downsample_ratio=factor,
                upsample_factor=args.upsample_factor,
                knn_k=args.knn_k,
                interp_min=args.interp_min,
                interp_max=args.interp_max,
                jitter_std=args.jitter_std,
                seed=args.seed,
                mode=mode,
            )
        elif mode == "down_up":
            process_variant(
                source_dir=source_dir,
                output_dir=output_dir,
                record_dir=record_dir,
                variant_name=output_name,
                downsample_ratio=args.downsample_ratio,
                upsample_factor=args.down_up_factor,
                knn_k=args.knn_k,
                interp_min=args.interp_min,
                interp_max=args.interp_max,
                jitter_std=args.jitter_std,
                seed=args.seed,
                mode=mode,
            )

    print(f"[prepare] preprocessing records written to {record_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
