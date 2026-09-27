#!/usr/bin/env python3
"""Patch-based full-frame KITTI reconstruction with official PU-GCN inference."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUGCN_ROOT = PROJECT_ROOT / "external" / "PU-GCN"
DEFAULT_RESTORE = DEFAULT_PUGCN_ROOT / "pretrained" / "pu1k-pugcn"
DEFAULT_INPUT_BIN = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_original_val" / "000001.bin"
DEFAULT_OUTPUT_BIN = PROJECT_ROOT / "results" / "pugcn_full_frame" / "000001" / "pugcn_full_frame.bin"
KITTI_COLUMNS = 4
DEFAULT_KITTI_RANGE = {
    "x_min": 0.0,
    "x_max": 80.0,
    "y_min": -40.0,
    "y_max": 40.0,
    "z_min": -3.5,
    "z_max": 2.0,
}


def str2bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("Expected true/false, got %r" % value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-frame KITTI reconstruction with PU-GCN patches.")
    parser.add_argument("--input_bin", "--input-bin", dest="input_bin", type=Path, default=DEFAULT_INPUT_BIN)
    parser.add_argument("--output_bin", "--output-bin", dest="output_bin", type=Path, default=DEFAULT_OUTPUT_BIN)
    parser.add_argument("--num_patches", "--num-patches", dest="num_patches", type=int, default=20)
    parser.add_argument("--patch_size", "--patch-size", dest="patch_size", type=int, default=1024)
    parser.add_argument("--upsample_ratio", "--upsample-ratio", dest="upsample_ratio", type=int, default=4)
    parser.add_argument("--max_output_points", "--max-output-points", dest="max_output_points", type=int, default=0)
    parser.add_argument("--filter_out_of_range", "--filter-out-of-range", dest="filter_out_of_range", type=str2bool, default=False)
    parser.add_argument("--save_debug_patches", "--save-debug-patches", dest="save_debug_patches", type=str2bool, default=False)
    parser.add_argument("--include_original", "--include-original", dest="include_original", type=str2bool, default=True)
    parser.add_argument("--work_dir", "--work-dir", dest="work_dir", type=Path, default=None)
    parser.add_argument("--pugcn_root", "--pugcn-root", dest="pugcn_root", type=Path, default=DEFAULT_PUGCN_ROOT)
    parser.add_argument("--restore", type=Path, default=DEFAULT_RESTORE)
    parser.add_argument("--pugcn_python", "--pugcn-python", dest="pugcn_python", type=str, default=sys.executable)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--internal_patch_num_point", "--internal-patch-num-point", dest="internal_patch_num_point", type=int, default=256)
    parser.add_argument("--internal_patch_num_ratio", "--internal-patch-num-ratio", dest="internal_patch_num_ratio", type=int, default=3)
    parser.add_argument("--dedup_voxel_size", "--dedup-voxel-size", dest="dedup_voxel_size", type=float, default=0.0)
    return parser.parse_args()


def load_kitti_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % KITTI_COLUMNS != 0:
        raise ValueError("Invalid KITTI .bin shape: %s" % path)
    return raw.reshape(-1, KITTI_COLUMNS)


def save_kitti_bin(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points.astype(np.float32).tofile(path)


def save_xyz(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, xyz.astype(np.float32), fmt="%.6f")


def load_xyz(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError("Invalid xyz file: %s" % path)
    return arr[:, :3].astype(np.float32)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def farthest_centers(xyz: np.ndarray, num_centers: int, seed: int) -> np.ndarray:
    if len(xyz) == 0:
        raise ValueError("Cannot select patch centers from an empty point cloud.")
    num_centers = min(max(num_centers, 1), len(xyz))
    rng = np.random.default_rng(seed)
    centers = np.empty(num_centers, dtype=np.int64)
    finite_xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(finite_xyz) == 0:
        raise ValueError("No finite xyz points available.")
    start_point = np.median(finite_xyz, axis=0)
    distances = np.sum((xyz - start_point.reshape(1, 3)) ** 2, axis=1)
    centers[0] = int(np.argmin(distances))
    min_dist = np.sum((xyz - xyz[centers[0]].reshape(1, 3)) ** 2, axis=1)
    for i in range(1, num_centers):
        centers[i] = int(np.argmax(min_dist))
        new_dist = np.sum((xyz - xyz[centers[i]].reshape(1, 3)) ** 2, axis=1)
        min_dist = np.minimum(min_dist, new_dist)
    if len(np.unique(centers)) < num_centers:
        missing = num_centers - len(np.unique(centers))
        extras = rng.choice(len(xyz), size=missing, replace=False)
        centers = np.unique(np.concatenate([centers, extras]))[:num_centers]
    return centers


def knn_patches(points_xyzi: np.ndarray, center_idx: np.ndarray, patch_size: int) -> Tuple[List[np.ndarray], List[Dict[str, object]]]:
    xyz = points_xyzi[:, :3]
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(xyz)
        distances, indices = tree.query(xyz[center_idx], k=patch_size, workers=-1)
    except TypeError:
        from scipy.spatial import cKDTree

        tree = cKDTree(xyz)
        distances, indices = tree.query(xyz[center_idx], k=patch_size)
    except Exception:
        all_indices = []
        all_distances = []
        for idx in center_idx:
            diff = xyz - xyz[idx]
            dist = np.sum(diff * diff, axis=1)
            order = np.argsort(dist)[:patch_size]
            all_indices.append(order)
            all_distances.append(np.sqrt(dist[order]))
        indices = np.asarray(all_indices)
        distances = np.asarray(all_distances)

    if indices.ndim == 1:
        indices = indices.reshape(1, -1)
        distances = distances.reshape(1, -1)

    patches = []
    metadata = []
    for patch_id, (source_center_idx, idxs, dists) in enumerate(zip(center_idx, indices, distances)):
        idxs = np.asarray(idxs, dtype=np.int64)
        patch = points_xyzi[idxs]
        patches.append(patch.astype(np.float32))
        metadata.append(
            {
                "patch_id": patch_id,
                "center_source_index": int(source_center_idx),
                "center_xyz": xyz[source_center_idx].astype(float).tolist(),
                "source_indices_min": int(np.min(idxs)),
                "source_indices_max": int(np.max(idxs)),
                "knn_radius": float(np.max(dists)),
                "input_points": int(len(patch)),
            }
        )
    return patches, metadata


def nearest_intensity(original: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
    if len(query_xyz) == 0:
        return np.zeros((0, 1), dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(original[:, :3]).query(query_xyz, k=1, workers=-1)
    except TypeError:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(original[:, :3]).query(query_xyz, k=1)
    except Exception:
        idx = []
        for point in query_xyz:
            diff = original[:, :3] - point.reshape(1, 3)
            idx.append(int(np.argmin(np.sum(diff * diff, axis=1))))
        idx = np.asarray(idx, dtype=np.int64)
    return original[np.asarray(idx, dtype=np.int64), 3].reshape(-1, 1).astype(np.float32)


def range_mask(points: np.ndarray) -> np.ndarray:
    xyz = points[:, :3]
    return (
        (xyz[:, 0] >= DEFAULT_KITTI_RANGE["x_min"])
        & (xyz[:, 0] <= DEFAULT_KITTI_RANGE["x_max"])
        & (xyz[:, 1] >= DEFAULT_KITTI_RANGE["y_min"])
        & (xyz[:, 1] <= DEFAULT_KITTI_RANGE["y_max"])
        & (xyz[:, 2] >= DEFAULT_KITTI_RANGE["z_min"])
        & (xyz[:, 2] <= DEFAULT_KITTI_RANGE["z_max"])
    )


def voxel_dedup(points: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, int]:
    if voxel_size <= 0 or len(points) == 0:
        return points, 0
    coords = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, keep_idx = np.unique(coords, axis=0, return_index=True)
    keep_idx.sort()
    return points[keep_idx], int(len(points) - len(keep_idx))


def sample_output(points: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, int]:
    if max_points <= 0 or len(points) <= max_points:
        return points, 0
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    idx.sort()
    return points[idx], int(len(points) - max_points)


def run_official_pugcn(args: argparse.Namespace, input_dir: Path) -> Path:
    pugcn_root = args.pugcn_root.resolve()
    restore = args.restore.resolve()
    input_dir = input_dir.resolve()
    required = [
        restore / "checkpoint",
        pugcn_root / "tf_ops" / "grouping" / "tf_grouping_so.so",
        pugcn_root / "tf_ops" / "sampling" / "tf_sampling_so.so",
        pugcn_root / "tf_ops" / "nn_distance" / "tf_nndistance_so.so",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing PU-GCN runtime files: %s" % ", ".join(missing))
    cmd = [
        args.pugcn_python,
        "main.py",
        "--phase",
        "test",
        "--restore",
        str(restore),
        "--data_dir",
        str(input_dir),
        "--model",
        "pugcn",
        "--k",
        str(args.k),
        "--up_ratio",
        str(args.upsample_ratio),
        "--patch_num_point",
        str(args.internal_patch_num_point),
        "--patch_num_ratio",
        str(args.internal_patch_num_ratio),
    ]
    env = dict(os.environ)
    conda_prefix = env.get("CONDA_PREFIX", "")
    ld_parts = []
    if conda_prefix:
        ld_parts.append(str(Path(conda_prefix) / "lib"))
        ld_parts.append(str(Path(conda_prefix) / "lib" / "python3.6" / "site-packages" / "tensorflow"))
    ld_parts.append(env.get("LD_LIBRARY_PATH", ""))
    env["LD_LIBRARY_PATH"] = ":".join([part for part in ld_parts if part])
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    start = time.time()
    subprocess.run(cmd, cwd=pugcn_root, env=env, check=True)
    runtime = time.time() - start
    return pugcn_root / "evaluation_code" / "result", runtime


def build_dirs(args: argparse.Namespace, frame_id: str) -> Dict[str, Path]:
    if args.work_dir is None:
        work_dir = args.output_bin.parent / ("work_%s_patches_%03d" % (frame_id, args.num_patches))
    else:
        work_dir = args.work_dir
    dirs = {
        "work": work_dir,
        "input_xyz": work_dir / "official_input_xyz",
        "official_result": work_dir / "official_result_xyz",
        "debug_patches": work_dir / "debug_patches",
    }
    if work_dir.exists():
        shutil.rmtree(work_dir)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def main() -> None:
    args = parse_args()
    frame_id = args.input_bin.stem
    dirs = build_dirs(args, frame_id)
    original = load_kitti_bin(args.input_bin)
    finite_input = original[np.isfinite(original[:, :3]).all(axis=1)]
    if len(finite_input) != len(original):
        original = finite_input

    t0 = time.time()
    center_idx = farthest_centers(original[:, :3], args.num_patches, args.seed)
    patches, patch_meta = knn_patches(original, center_idx, args.patch_size)
    prep_runtime = time.time() - t0

    for patch, meta in zip(patches, patch_meta):
        patch_name = "%s_patch%04d" % (frame_id, meta["patch_id"])
        save_xyz(dirs["input_xyz"] / ("%s.xyz" % patch_name), patch[:, :3])
        meta["input_xyz"] = str((dirs["input_xyz"] / ("%s.xyz" % patch_name)).resolve())
        if args.save_debug_patches:
            np.save(dirs["debug_patches"] / ("%s_xyzi.npy" % patch_name), patch.astype(np.float32))

    result_dir, inference_runtime = run_official_pugcn(args, dirs["input_xyz"])
    if dirs["official_result"].exists():
        shutil.rmtree(dirs["official_result"])
    shutil.copytree(result_dir, dirs["official_result"])

    pred_xyz_list = []
    missing_outputs = []
    for meta in patch_meta:
        patch_name = "%s_patch%04d" % (frame_id, meta["patch_id"])
        pred_path = dirs["official_result"] / ("%s.xyz" % patch_name)
        if pred_path.exists():
            pred = load_xyz(pred_path)
            pred_xyz_list.append(pred)
            meta["output_xyz"] = str(pred_path.resolve())
            meta["output_points"] = int(len(pred))
        else:
            missing_outputs.append(str(pred_path))
            meta["output_points"] = 0
    if missing_outputs:
        raise FileNotFoundError("Missing PU-GCN patch outputs: %s" % ", ".join(missing_outputs[:5]))
    merged_xyz = np.concatenate(pred_xyz_list, axis=0) if pred_xyz_list else np.zeros((0, 3), dtype=np.float32)
    raw_generated_points = int(len(merged_xyz))
    intensity = nearest_intensity(original, merged_xyz)
    generated_xyzi = np.concatenate([merged_xyz.astype(np.float32), intensity], axis=1)
    if args.include_original:
        merged_xyzi = np.concatenate([original.astype(np.float32), generated_xyzi], axis=0)
    else:
        merged_xyzi = generated_xyzi

    invalid_mask = ~np.isfinite(merged_xyzi[:, :3]).all(axis=1)
    invalid_points = int(np.sum(invalid_mask))
    merged_xyzi = merged_xyzi[~invalid_mask]

    out_of_range_before_filter = int(np.sum(~range_mask(merged_xyzi))) if len(merged_xyzi) else 0
    filtered_out_of_range = 0
    if args.filter_out_of_range and len(merged_xyzi):
        mask = range_mask(merged_xyzi)
        filtered_out_of_range = int(np.sum(~mask))
        merged_xyzi = merged_xyzi[mask]

    before_dedup = len(merged_xyzi)
    merged_xyzi, dedup_removed = voxel_dedup(merged_xyzi, args.dedup_voxel_size)
    before_sample = len(merged_xyzi)
    merged_xyzi, sampled_removed = sample_output(merged_xyzi, args.max_output_points, args.seed)
    save_kitti_bin(args.output_bin, merged_xyzi)

    total_runtime = time.time() - t0
    manifest = {
        "frame_id": frame_id,
        "input_bin": str(args.input_bin.resolve()),
        "output_bin": str(args.output_bin.resolve()),
        "work_dir": str(dirs["work"].resolve()),
        "num_patches": int(args.num_patches),
        "patch_size": int(args.patch_size),
        "upsample_ratio": int(args.upsample_ratio),
        "original_points": int(len(original)),
        "raw_generated_points": raw_generated_points,
        "include_original": bool(args.include_original),
        "invalid_points_removed": invalid_points,
        "out_of_range_points_before_filter": out_of_range_before_filter,
        "filtered_out_of_range": filtered_out_of_range,
        "dedup_voxel_size": float(args.dedup_voxel_size),
        "dedup_removed": dedup_removed,
        "sampled_removed": sampled_removed,
        "output_points": int(len(merged_xyzi)),
        "effective_upsampling_ratio": float(len(merged_xyzi) / len(original)) if len(original) else 0.0,
        "prep_runtime_sec": prep_runtime,
        "inference_runtime_sec": inference_runtime,
        "total_runtime_sec": total_runtime,
        "filter_out_of_range": bool(args.filter_out_of_range),
        "max_output_points": int(args.max_output_points),
        "intensity_strategy": "nearest-neighbor interpolation from the original KITTI frame",
    }
    write_json(dirs["work"] / "manifest.json", manifest)
    write_csv(
        dirs["work"] / "patch_manifest.csv",
        patch_meta,
        ["patch_id", "center_source_index", "center_xyz", "knn_radius", "input_points", "output_points", "input_xyz", "output_xyz"],
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
