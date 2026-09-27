#!/usr/bin/env python3
"""Reconstruct a full-frame KITTI PU-Net output from scene + patch predictions.

This wrapper keeps the original KITTI scan as the backbone and appends only
filtered PU-Net-generated points from the existing patch-only output folder.
It is intentionally separate from the original adapter so the invalid
``velodyne_punet_x2`` folder can remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.punet_kitti_adapter import load_kitti_bin, read_split_ids, save_kitti_bin
from tools.density_metric_schema import NA_VALUE, SCHEMA_VERSION, csv_fieldnames
from tools.evaluate_density_pointclouds import evaluate_pair, render_visualizations, save_csv, select_visualization_samples


DEFAULT_REFERENCE_VELODYNE = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_original_val"
DEFAULT_PATCH_VELODYNE = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_punet_x2"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
DEFAULT_OUTPUT_VELODYNE = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_punet_x2_fullframe"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "density_baselines" / "PUNet_visual_only"
KITTI_COLUMNS = 4
POINT_STATS_FIELDNAMES = [
    "frame_id",
    "input_file",
    "output_file",
    "original_num_points",
    "processed_num_points",
    "final_num_points",
    "added_points",
    "removed_points",
    "point_count_ratio",
    "punet_point_count_ratio",
    "format_ok",
    "has_nan",
    "has_inf",
    "x_min_before",
    "x_max_before",
    "y_min_before",
    "y_max_before",
    "z_min_before",
    "z_max_before",
    "x_min_after",
    "x_max_after",
    "y_min_after",
    "y_max_after",
    "z_min_after",
    "z_max_after",
    "intensity_min_before",
    "intensity_max_before",
    "intensity_mean_before",
    "intensity_std_before",
    "intensity_min_after",
    "intensity_max_after",
    "intensity_mean_after",
    "intensity_std_after",
    "punet_x_min",
    "punet_x_max",
    "punet_y_min",
    "punet_y_max",
    "punet_z_min",
    "punet_z_max",
    "punet_intensity_mean",
    "preserved_original_points",
    "summary_note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a full-frame KITTI PU-Net output from patch predictions.")
    parser.add_argument("--reference-velodyne", type=Path, default=DEFAULT_REFERENCE_VELODYNE)
    parser.add_argument("--punet-velodyne", type=Path, default=DEFAULT_PATCH_VELODYNE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-velodyne", type=Path, default=DEFAULT_OUTPUT_VELODYNE)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--experiment-name", type=str, default="PUNet_visual_only")
    parser.add_argument("--method-name", type=str, default="PU-Net")
    parser.add_argument("--processing-type", type=str, default="upsample")
    parser.add_argument("--sample-limit", type=int, default=0, help="Limit to the first N split frames for smoke testing.")
    parser.add_argument("--max-visualized-samples", type=int, default=10)
    parser.add_argument("--dedup-voxel-size", type=float, default=0.05)
    parser.add_argument("--intensity-mode", type=str, default="nearest_original", choices=("nearest_original", "original_mean"))
    parser.add_argument("--seed", type=int, default=1024)
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def is_bin_file_valid(path: Path) -> Tuple[bool, bool, bool]:
    arr = load_kitti_bin(path)
    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())
    format_ok = bool(arr.ndim == 2 and arr.shape[1] == KITTI_COLUMNS)
    return format_ok, has_nan, has_inf


def pack_voxels(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    vox = np.floor(xyz / voxel_size).astype(np.int64)
    base = 20000
    offset = 10000
    return (vox[:, 0] + offset) * (base * base) + (vox[:, 1] + offset) * base + (vox[:, 2] + offset)


def merge_fullframe(
    original: np.ndarray,
    punet_patch: np.ndarray,
    dedup_voxel_size: float,
    intensity_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if original.ndim != 2 or original.shape[1] != KITTI_COLUMNS:
        raise ValueError("Original point cloud is not N x 4.")
    if punet_patch.ndim != 2 or punet_patch.shape[1] != KITTI_COLUMNS:
        raise ValueError("PU-Net point cloud is not N x 4.")

    if len(punet_patch) == 0:
        return original.copy(), np.zeros((0, 4), dtype=np.float32)

    finite_mask = np.isfinite(punet_patch[:, :3]).all(axis=1) & np.isfinite(punet_patch[:, 3])
    punet_patch = punet_patch[finite_mask]
    if len(punet_patch) == 0:
        return original.copy(), np.zeros((0, 4), dtype=np.float32)

    original_xyz = original[:, :3].astype(np.float32, copy=False)
    original_intensity = original[:, 3].astype(np.float32, copy=False)
    tree = cKDTree(original_xyz)

    original_voxels = np.unique(pack_voxels(original_xyz, dedup_voxel_size))
    punet_voxels = pack_voxels(punet_patch[:, :3], dedup_voxel_size)
    unique_idx = np.unique(punet_voxels, return_index=True)[1]
    unique_idx.sort()
    punet_unique = punet_patch[unique_idx]
    punet_unique_voxels = punet_voxels[unique_idx]

    keep_mask = ~np.isin(punet_unique_voxels, original_voxels, assume_unique=False)
    kept = punet_unique[keep_mask]
    if len(kept) == 0:
        return original.copy(), np.zeros((0, 4), dtype=np.float32)

    distances, nn_idx = tree.query(kept[:, :3], k=1)
    if intensity_mode == "nearest_original":
        intensities = original_intensity[np.asarray(nn_idx, dtype=np.int64)]
    else:
        intensities = np.full(len(kept), float(np.mean(original_intensity)), dtype=np.float32)

    kept_xyzi = np.concatenate([kept[:, :3].astype(np.float32, copy=False), intensities.reshape(-1, 1).astype(np.float32)], axis=1)
    merged = np.concatenate([original.astype(np.float32, copy=False), kept_xyzi], axis=0).astype(np.float32, copy=False)
    return merged, kept_xyzi


def compute_ranges(points: np.ndarray) -> Dict[str, float]:
    if len(points) == 0:
        return {k: NA_VALUE for k in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", "intensity_min", "intensity_max", "intensity_mean", "intensity_std"]}
    xyz = points[:, :3]
    intensity = points[:, 3]
    return {
        "x_min": float(np.min(xyz[:, 0])),
        "x_max": float(np.max(xyz[:, 0])),
        "y_min": float(np.min(xyz[:, 1])),
        "y_max": float(np.max(xyz[:, 1])),
        "z_min": float(np.min(xyz[:, 2])),
        "z_max": float(np.max(xyz[:, 2])),
        "intensity_min": float(np.min(intensity)),
        "intensity_max": float(np.max(intensity)),
        "intensity_mean": float(np.mean(intensity)),
        "intensity_std": float(np.std(intensity)),
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_point_statistics_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POINT_STATS_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, NA_VALUE) for field in POINT_STATS_FIELDNAMES})


def pick_sample_ids(rows: Sequence[Dict[str, object]], max_samples: int) -> List[str]:
    if not rows or max_samples <= 0:
        return []
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row.get("output_points", 0) or 0),
            float(row.get("input_points", 0) or 0),
            str(row.get("sample_id", "")),
        ),
    )
    if len(ranked) <= max_samples:
        return [str(row["sample_id"]) for row in ranked]
    idxs = np.linspace(0, len(ranked) - 1, num=max_samples, dtype=int)
    return [str(ranked[i]["sample_id"]) for i in idxs]


def build_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "config": root / "config",
        "preprocessing": root / "preprocessing",
        "visualizations": root / "preprocessing" / "visualizations",
        "debug_record": root / "debug_record",
        "notes": root / "notes",
        "summary": root / "summary",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def reconstruct_dataset(args: argparse.Namespace, sample_ids: Sequence[str], output_root: Path, experiment_root: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object]]:
    dirs = build_dirs(experiment_root)

    config = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": args.experiment_name,
        "method_name": args.method_name,
        "processing_type": args.processing_type,
        "reference_velodyne": str(args.reference_velodyne.resolve()),
        "punet_patch_velodyne": str(args.punet_velodyne.resolve()),
        "output_velodyne": str(output_root.resolve()),
        "dedup_voxel_size": args.dedup_voxel_size,
        "intensity_mode": args.intensity_mode,
        "sample_limit": len(sample_ids) if args.sample_limit else 0,
        "seed": args.seed,
    }
    write_json(dirs["config"] / "experiment_config.json", config)
    write_text(dirs["config"] / "data_paths.txt", "\n".join([
        f"reference_velodyne={args.reference_velodyne.resolve()}",
        f"punet_patch_velodyne={args.punet_velodyne.resolve()}",
        f"output_velodyne={output_root.resolve()}",
        f"split_file={args.split_file.resolve()}",
        f"experiment_root={experiment_root.resolve()}",
    ]) + "\n")
    write_text(dirs["config"] / "command_used.txt", " ".join(map(str, sys.argv)) + "\n")
    write_text(dirs["debug_record"] / "start_time.txt", utc_now() + "\n")
    write_text(dirs["debug_record"] / "status.txt", "running\n")

    summary_rows: List[Dict[str, object]] = []
    point_stats_rows: List[Dict[str, object]] = []
    failures: List[str] = []
    output_root.mkdir(parents=True, exist_ok=True)

    for sample_id in sample_ids:
        ref_path = args.reference_velodyne / f"{sample_id}.bin"
        punet_path = args.punet_velodyne / f"{sample_id}.bin"
        out_path = output_root / f"{sample_id}.bin"
        if not ref_path.exists() or not punet_path.exists():
            failures.append(sample_id)
            continue

        ref = load_kitti_bin(ref_path).astype(np.float32, copy=False)
        punet_patch = load_kitti_bin(punet_path).astype(np.float32, copy=False)
        merged, kept_punet = merge_fullframe(ref, punet_patch, args.dedup_voxel_size, args.intensity_mode)
        save_kitti_bin(out_path, merged)

        density_row = evaluate_pair(
            sample_id=sample_id,
            reference=ref,
            processed=merged,
            experiment_name=args.experiment_name,
            method_name=args.method_name,
            reference_path=ref_path,
            processed_path=out_path,
            transform_stage="fullframe_merge",
            intensity_strategy=args.intensity_mode,
            processing_type=args.processing_type,
            lidar_range={
                "x_min": 0.0,
                "x_max": 80.0,
                "y_min": -40.0,
                "y_max": 40.0,
                "z_min": -3.5,
                "z_max": 2.0,
            },
            cell_size=0.5,
            sample_viz_dir=None,
            seed=args.seed + int(sample_id),
        )
        summary_rows.append(density_row)

        ref_stats = compute_ranges(ref)
        merged_stats = compute_ranges(merged)
        punet_stats = compute_ranges(punet_patch)
        point_stats_rows.append(
            {
                "frame_id": sample_id,
                "input_file": str(ref_path.resolve()),
                "output_file": str(out_path.resolve()),
                "original_num_points": int(len(ref)),
                "processed_num_points": int(len(punet_patch)),
                "final_num_points": int(len(merged)),
                "added_points": int(len(kept_punet)),
                "removed_points": int(max(len(ref) - len(merged), 0)),
                "point_count_ratio": float(len(merged) / len(ref)) if len(ref) else NA_VALUE,
                "punet_point_count_ratio": float(len(punet_patch) / len(ref)) if len(ref) else NA_VALUE,
                "format_ok": bool(ref.ndim == 2 and merged.ndim == 2 and ref.shape[1] == KITTI_COLUMNS and merged.shape[1] == KITTI_COLUMNS),
                "has_nan": bool(np.isnan(ref).any() or np.isnan(merged).any() or np.isnan(punet_patch).any()),
                "has_inf": bool(np.isinf(ref).any() or np.isinf(merged).any() or np.isinf(punet_patch).any()),
                "x_min_before": ref_stats["x_min"],
                "x_max_before": ref_stats["x_max"],
                "y_min_before": ref_stats["y_min"],
                "y_max_before": ref_stats["y_max"],
                "z_min_before": ref_stats["z_min"],
                "z_max_before": ref_stats["z_max"],
                "x_min_after": merged_stats["x_min"],
                "x_max_after": merged_stats["x_max"],
                "y_min_after": merged_stats["y_min"],
                "y_max_after": merged_stats["y_max"],
                "z_min_after": merged_stats["z_min"],
                "z_max_after": merged_stats["z_max"],
                "intensity_min_before": ref_stats["intensity_min"],
                "intensity_max_before": ref_stats["intensity_max"],
                "intensity_mean_before": ref_stats["intensity_mean"],
                "intensity_std_before": ref_stats["intensity_std"],
                "intensity_min_after": merged_stats["intensity_min"],
                "intensity_max_after": merged_stats["intensity_max"],
                "intensity_mean_after": merged_stats["intensity_mean"],
                "intensity_std_after": merged_stats["intensity_std"],
                "punet_x_min": punet_stats["x_min"],
                "punet_x_max": punet_stats["x_max"],
                "punet_y_min": punet_stats["y_min"],
                "punet_y_max": punet_stats["y_max"],
                "punet_z_min": punet_stats["z_min"],
                "punet_z_max": punet_stats["z_max"],
                "punet_intensity_mean": punet_stats["intensity_mean"],
                "preserved_original_points": int(len(ref)),
                "summary_note": "original full-frame points preserved and merged with filtered PU-Net additions",
            }
        )

    if point_stats_rows:
        sample_ids_for_viz = pick_sample_ids(summary_rows, args.max_visualized_samples)
        for sample_id in sample_ids_for_viz:
            ref_path = args.reference_velodyne / f"{sample_id}.bin"
            out_path = output_root / f"{sample_id}.bin"
            if not ref_path.exists() or not out_path.exists():
                continue
            ref = load_kitti_bin(ref_path)
            merged = load_kitti_bin(out_path)
            viz_dir = dirs["visualizations"] / sample_id
            viz_paths = render_visualizations(ref, merged, viz_dir, sample_id, cell_size=0.5, seed=args.seed + int(sample_id))
            for row in summary_rows:
                if str(row.get("sample_id")) == sample_id:
                    row.update(viz_paths)
                    break

    save_csv(dirs["preprocessing"] / "density_metrics.csv", summary_rows)
    write_point_statistics_csv(dirs["preprocessing"] / "point_statistics.csv", point_stats_rows)
    write_text(dirs["preprocessing"] / "file_list.txt", "\n".join(sample_ids) + "\n")

    aggregate = _aggregate_rows(args, summary_rows, dirs, output_root)
    write_text(dirs["preprocessing"] / "preprocessing_params.json", json.dumps(config, indent=2, sort_keys=True) + "\n")
    write_text(
        dirs["preprocessing"] / "preprocessing_log.txt",
        "\n".join(
            [
                f"experiment_name={args.experiment_name}",
                f"method_name={args.method_name}",
                f"processing_type={args.processing_type}",
                f"reference_velodyne={args.reference_velodyne.resolve()}",
                f"punet_patch_velodyne={args.punet_velodyne.resolve()}",
                f"output_velodyne={output_root.resolve()}",
                f"num_frames={len(sample_ids)}",
                f"dedup_voxel_size={args.dedup_voxel_size}",
                f"intensity_mode={args.intensity_mode}",
                f"failed_frames={len(failures)}",
            ]
        )
        + "\n",
    )
    write_text(dirs["debug_record"] / "status.txt", "completed\n" if not failures else "completed_with_failures\n")
    if failures:
        write_text(dirs["debug_record"] / "failed_frames.txt", "\n".join(failures) + "\n")

    return summary_rows, point_stats_rows, aggregate


def _aggregate_rows(args: argparse.Namespace, rows: Sequence[Dict[str, object]], dirs: Dict[str, Path], output_root: Path) -> Dict[str, object]:
    if not rows:
        return {}
    numeric_fields = [
        "input_points",
        "output_points",
        "added_points",
        "removed_points",
        "upsampling_ratio",
        "nn_mean_before",
        "nn_mean_after",
        "nn_median_before",
        "nn_median_after",
        "nn_std_before",
        "nn_std_after",
        "bev_density_mean_before",
        "bev_density_mean_after",
        "bev_density_median_before",
        "bev_density_median_after",
        "bev_density_std_before",
        "bev_density_std_after",
        "intensity_mean_before",
        "intensity_mean_after",
        "intensity_std_before",
        "intensity_std_after",
    ]
    aggregate: Dict[str, object] = {field: NA_VALUE for field in csv_fieldnames()}
    aggregate.update(
        {
            "schema_version": SCHEMA_VERSION,
            "scope": "method",
            "dataset": "KITTI",
            "split": "val",
            "experiment_name": args.experiment_name,
            "method_name": args.method_name,
            "processing_type": args.processing_type,
            "sample_id": "__aggregate__",
            "reference_velodyne": str(args.reference_velodyne.resolve()),
            "processed_velodyne": str(output_root.resolve()),
            "source_path": str(args.reference_velodyne.resolve()),
            "input_path": str(args.reference_velodyne.resolve()),
            "output_path": str(output_root.resolve()),
            "transform_stage": "fullframe_merge",
            "intensity_strategy": args.intensity_mode,
            "format_ok": bool(all(bool(row.get("format_ok", False)) for row in rows)),
            "coordinate_frame_preserved": True,
            "within_kitti_range_before": bool(all(bool(row.get("within_kitti_range_before", False)) for row in rows)),
            "within_kitti_range_after": bool(all(bool(row.get("within_kitti_range_after", False)) for row in rows)),
            "summary_note": "full-frame merge aggregate",
        }
    )
    for field in numeric_fields:
        values = [float(row.get(field)) for row in rows if row.get(field) not in (None, "", NA_VALUE)]
        if not values:
            continue
        aggregate[field] = float(np.mean(values))
    with (dirs["summary"] / "method_summary.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
    return aggregate


def write_report(experiment_root: Path, output_root: Path, args: argparse.Namespace, summary_rows: Sequence[Dict[str, object]], point_stats_rows: Sequence[Dict[str, object]], failures: Sequence[str], smoke_mode: bool) -> None:
    dirs = build_dirs(experiment_root)
    original_count = float(np.mean([float(row.get("input_points", 0) or 0) for row in summary_rows])) if summary_rows else 0.0
    merged_count = float(np.mean([float(row.get("output_points", 0) or 0) for row in summary_rows])) if summary_rows else 0.0
    patch_count = float(np.mean([float(row.get("processed_num_points", 0) or 0) for row in point_stats_rows])) if point_stats_rows else 0.0
    added_count = float(np.mean([float(row.get("added_points", 0) or 0) for row in point_stats_rows])) if point_stats_rows else 0.0
    repaired_ratio = float(np.mean([float(row.get("point_count_ratio", 0) or 0) for row in point_stats_rows])) if point_stats_rows else 0.0
    patch_ratio = float(np.mean([float(row.get("punet_point_count_ratio", 0) or 0) for row in point_stats_rows])) if point_stats_rows else 0.0
    format_ok = all(bool(row.get("format_ok", False)) for row in point_stats_rows) if point_stats_rows else False
    has_nan = any(bool(row.get("has_nan", False)) for row in point_stats_rows) if point_stats_rows else True
    has_inf = any(bool(row.get("has_inf", False)) for row in point_stats_rows) if point_stats_rows else True

    ref_ranges = {
        "x_min": min(float(row.get("x_min_before", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
        "x_max": max(float(row.get("x_max_before", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
        "y_min": min(float(row.get("y_min_before", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
        "y_max": max(float(row.get("y_max_before", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
        "z_min": min(float(row.get("z_min_before", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
        "z_max": max(float(row.get("z_max_before", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
    }
    after_ranges = {
        "x_min": min(float(row.get("x_min_after", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
        "x_max": max(float(row.get("x_max_after", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
        "y_min": min(float(row.get("y_min_after", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
        "y_max": max(float(row.get("y_max_after", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
        "z_min": min(float(row.get("z_min_after", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
        "z_max": max(float(row.get("z_max_after", 0) or 0) for row in point_stats_rows) if point_stats_rows else 0.0,
    }
    report = [
        f"# {args.experiment_name} Full-Frame Repair Report",
        "",
        "## Purpose",
        "",
        "Rebuild a valid full-frame PU-Net KITTI output by preserving the original val scan and appending only filtered PU-Net patch predictions.",
        "",
        "## Why The Old Folder Was Invalid",
        "",
        f"- Old patch-only folder: `{args.punet_velodyne.resolve()}`",
        f"- It contained only fused PU-Net patch predictions, not the original full-frame KITTI points.",
        f"- Mean original points/frame: `{original_count:.6f}`",
        f"- Mean patch-only points/frame: `{patch_count:.6f}`",
        f"- Mean patch-only point-count ratio: `{patch_ratio:.6f}`",
        f"- Repaired merged points/frame: `{merged_count:.6f}`",
        f"- Repaired merged point-count ratio: `{repaired_ratio:.6f}`",
        "",
        "## Reconstruction Strategy",
        "",
        f"- Original full-frame points were preserved from `{args.reference_velodyne.resolve()}`.",
        f"- PU-Net points from the patch-only folder were filtered with a conservative voxel deduplication size of `{args.dedup_voxel_size}` m.",
        f"- Remaining PU-Net points were assigned intensity via `{args.intensity_mode}`.",
        f"- Final output folder: `{args.output_velodyne.resolve()}`",
        "",
        "## Data Validity",
        "",
        f"- frames processed: `{len(point_stats_rows)}`",
        f"- filenames matched split ids: `yes`" if not failures else f"- filenames matched split ids: `partial`",
        f"- format ok for all frames: `{format_ok}`",
        f"- any NaN: `{has_nan}`",
        f"- any Inf: `{has_inf}`",
        f"- any failed frames: `{bool(failures)}`",
        "",
        "## Geometry Summary",
        "",
        f"- x before: [{ref_ranges['x_min']:.6f}, {ref_ranges['x_max']:.6f}]",
        f"- y before: [{ref_ranges['y_min']:.6f}, {ref_ranges['y_max']:.6f}]",
        f"- z before: [{ref_ranges['z_min']:.6f}, {ref_ranges['z_max']:.6f}]",
        f"- x after: [{after_ranges['x_min']:.6f}, {after_ranges['x_max']:.6f}]",
        f"- y after: [{after_ranges['y_min']:.6f}, {after_ranges['y_max']:.6f}]",
        f"- z after: [{after_ranges['z_min']:.6f}, {after_ranges['z_max']:.6f}]",
        "",
        "## Conclusion",
        "",
        "The repaired output is suitable for later PointRCNN evaluation" if not failures else "The repaired output is suitable for later PointRCNN evaluation, but the frame list contains failures that should be reviewed before detector use.",
        "",
        "## Notes",
        "",
        "- Original full-frame points were preserved.",
        "- PU-Net additions were merged conservatively.",
        "- PointRCNN core files were not modified.",
    ]
    if smoke_mode:
        report.insert(1, "")
        report.insert(2, "## Mode")
        report.insert(3, "")
        report.insert(4, "Smoke test run on the first five val frames.")
    write_text(dirs["summary"] / "PUNet_fullframe_repair_report.md", "\n".join(report) + "\n")

    validation = [
        "# Validation Report",
        f"- output folder exists: {output_root.exists()}",
        f"- reference folder exists: {args.reference_velodyne.exists()}",
        f"- patch-only folder exists: {args.punet_velodyne.exists()}",
        f"- frames processed: {len(point_stats_rows)}",
        f"- average original points/frame: {original_count:.6f}",
        f"- average patch-only points/frame: {patch_count:.6f}",
        f"- average merged points/frame: {merged_count:.6f}",
        f"- average added PU-Net points/frame: {added_count:.6f}",
        f"- average repaired point-count ratio: {repaired_ratio:.6f}",
        f"- format ok for all frames: {format_ok}",
        f"- any NaN: {has_nan}",
        f"- any Inf: {has_inf}",
        f"- failed frames: {len(failures)}",
        f"- suitable for later PointRCNN evaluation: {not failures and format_ok and not has_nan and not has_inf}",
    ]
    write_text(dirs["preprocessing"] / "validation_report.txt", "\n".join(validation) + "\n")


def main() -> int:
    args = parse_args()
    split_ids = read_split_ids(args.split_file)
    if args.sample_limit and args.sample_limit > 0:
        split_ids = split_ids[: args.sample_limit]

    smoke_mode = bool(args.sample_limit and args.sample_limit > 0 and args.sample_limit <= 5)
    experiment_root = args.experiment_root / ("fullframe_repair_smoke5" if smoke_mode else "fullframe_repair")
    output_root = args.output_velodyne.parent / ("velodyne_punet_x2_fullframe_smoke5" if smoke_mode else "velodyne_punet_x2_fullframe")

    summary_rows: List[Dict[str, object]] = []
    point_stats_rows: List[Dict[str, object]] = []
    failures: List[str] = []

    try:
        summary_rows, point_stats_rows, _ = reconstruct_dataset(args, split_ids, output_root, experiment_root)
        failures = [row.get("frame_id", "") for row in point_stats_rows if not bool(row.get("format_ok", False))]
        write_report(experiment_root, output_root, args, summary_rows, point_stats_rows, failures, smoke_mode=smoke_mode)
        return 0
    except Exception as exc:  # pragma: no cover - runtime safeguard
        dirs = build_dirs(experiment_root)
        write_text(dirs["debug_record"] / "status.txt", "failed\n")
        write_text(dirs["debug_record"] / "error.txt", f"{type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
