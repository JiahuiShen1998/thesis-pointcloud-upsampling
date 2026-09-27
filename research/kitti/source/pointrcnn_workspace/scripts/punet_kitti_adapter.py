#!/usr/bin/env python3
"""KITTI adapter for the official PU-Net repository.

This utility prepares KITTI velodyne scenes as patch folders for PU-Net,
optionally launches the official PU-Net tester, and merges the patch outputs
back into KITTI .bin files that can be consumed by PointRCNN.

The official PU-Net repository is patch-based and TensorFlow 1.x oriented, so
this script keeps the adaptation at the data layer instead of modifying the
detector.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KITTI_ROOT = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
DEFAULT_WORK_DIR = PROJECT_ROOT / "punet_work"
DEFAULT_PUNET_ROOT = PROJECT_ROOT.parents[1] / "upsampling" / "PU-Net"


def read_split_ids(split_file: Path) -> list[str]:
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_kitti_bin(bin_path: Path) -> np.ndarray:
    points = np.fromfile(str(bin_path), dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError("Invalid KITTI bin file: %s" % bin_path)
    return points.reshape(-1, 4)


def save_kitti_bin(bin_path: Path, points_xyzi: np.ndarray) -> None:
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    points_xyzi.astype(np.float32).tofile(str(bin_path))


def save_xyz(xyz_path: Path, points_xyz: np.ndarray) -> None:
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(xyz_path), points_xyz.astype(np.float32), fmt="%.6f")


def normalize_patch(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    centroid = points_xyz.mean(axis=0, keepdims=True)
    centered = points_xyz - centroid
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    scale = max(scale, 1e-6)
    return centered / scale, centroid.reshape(3), scale


def denormalize_patch(points_xyz: np.ndarray, centroid: np.ndarray, scale: float) -> np.ndarray:
    return points_xyz * scale + centroid.reshape(1, 3)


def extract_bev_patches(
    points_xyzi: np.ndarray,
    patch_size: float,
    patch_stride: float,
    min_points: int,
    max_input_points: int,
) -> list[dict]:
    """Extract overlapping KITTI patches in the x-z plane.

    This is a simple scene-to-patch wrapper for the official patch-level PU-Net.
    """
    xyz = points_xyzi[:, :3]
    if xyz.shape[0] == 0:
        return []

    x_min, x_max = float(np.min(xyz[:, 0])), float(np.max(xyz[:, 0]))
    z_min, z_max = float(np.min(xyz[:, 2])), float(np.max(xyz[:, 2]))
    half = patch_size / 2.0

    if x_max <= x_min or z_max <= z_min:
        return []

    x_centers = np.arange(x_min + half, x_max + half, patch_stride)
    z_centers = np.arange(z_min + half, z_max + half, patch_stride)

    patches = []
    patch_idx = 0
    for cx in x_centers:
        for cz in z_centers:
            mask = (
                (xyz[:, 0] >= cx - half)
                & (xyz[:, 0] <= cx + half)
                & (xyz[:, 2] >= cz - half)
                & (xyz[:, 2] <= cz + half)
            )
            patch_xyzi = points_xyzi[mask]
            if patch_xyzi.shape[0] < min_points:
                continue

            if patch_xyzi.shape[0] > max_input_points:
                choice = np.random.choice(patch_xyzi.shape[0], max_input_points, replace=False)
                patch_xyzi = patch_xyzi[choice]
            elif patch_xyzi.shape[0] < max_input_points:
                choice = np.random.choice(patch_xyzi.shape[0], max_input_points, replace=True)
                patch_xyzi = patch_xyzi[choice]

            patches.append(
                {
                    "patch_idx": patch_idx,
                    "center_x": float(cx),
                    "center_z": float(cz),
                    "half_size": float(half),
                    "points_xyzi": patch_xyzi.astype(np.float32),
                }
            )
            patch_idx += 1

    if not patches:
        patch_xyzi = points_xyzi
        if patch_xyzi.shape[0] > max_input_points:
            choice = np.random.choice(patch_xyzi.shape[0], max_input_points, replace=False)
            patch_xyzi = patch_xyzi[choice]
        elif patch_xyzi.shape[0] < max_input_points:
            choice = np.random.choice(patch_xyzi.shape[0], max_input_points, replace=True)
            patch_xyzi = patch_xyzi[choice]
        patches.append(
            {
                "patch_idx": 0,
                "center_x": float(np.mean(xyz[:, 0])),
                "center_z": float(np.mean(xyz[:, 2])),
                "half_size": float(max(patch_size / 2.0, 1.0)),
                "points_xyzi": patch_xyzi.astype(np.float32),
            }
        )

    return patches


def prepare_scene_patch_folder(
    sample_id: str,
    points_xyzi: np.ndarray,
    sample_dir: Path,
    patch_size: float,
    patch_stride: float,
    min_points: int,
    max_input_points: int,
) -> Path:
    """Create a PU-Net input folder for one KITTI scene."""
    patch_root = sample_dir / sample_id / "patches"
    patch_root.mkdir(parents=True, exist_ok=True)

    patches = extract_bev_patches(
        points_xyzi=points_xyzi,
        patch_size=patch_size,
        patch_stride=patch_stride,
        min_points=min_points,
        max_input_points=max_input_points,
    )

    manifest = {
        "sample_id": sample_id,
        "source_points": int(points_xyzi.shape[0]),
        "patch_count": len(patches),
        "patch_size": patch_size,
        "patch_stride": patch_stride,
        "min_points": min_points,
        "max_input_points": max_input_points,
        "patches": [],
    }

    for patch in patches:
        patch_name = "%s_patch%04d" % (sample_id, patch["patch_idx"])
        normalized_xyz, centroid, scale = normalize_patch(patch["points_xyzi"][:, :3])
        patch_xyz_path = patch_root / (patch_name + ".xyz")
        patch_npy_path = patch_root / (patch_name + ".npy")
        patch_meta_path = patch_root / (patch_name + ".json")

        save_xyz(patch_xyz_path, normalized_xyz)
        np.save(str(patch_npy_path), patch["points_xyzi"].astype(np.float32))

        patch_meta = {
            "patch_name": patch_name,
            "patch_idx": patch["patch_idx"],
            "center_x": patch["center_x"],
            "center_z": patch["center_z"],
            "half_size": patch["half_size"],
            "centroid": centroid.tolist(),
            "scale": float(scale),
            "input_xyz": patch_xyz_path.name,
            "source_npy": patch_npy_path.name,
        }
        with open(patch_meta_path, "w") as f:
            json.dump(patch_meta, f, indent=2, sort_keys=True)

        manifest["patches"].append(patch_meta)

    with open(patch_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    return patch_root


def run_official_punet(
    punet_root: Path,
    punet_python: str,
    patch_root: Path,
    log_dir: Path,
    gpu: str,
    up_ratio: int,
    num_point: int,
) -> Path:
    """Invoke the official PU-Net tester on a patch directory."""
    punet_code_dir = punet_root / "code"
    cmd = [
        punet_python,
        str(punet_code_dir / "main.py"),
        "--phase",
        "test",
        "--gpu",
        gpu,
        "--log_dir",
        str(log_dir),
        "--data_folder",
        str(patch_root),
        "--num_point",
        str(num_point),
        "--up_ratio",
        str(up_ratio),
    ]
    subprocess.run(cmd, check=True, cwd=str(punet_code_dir))
    phase = patch_root.parent.name + patch_root.name
    return log_dir / "result" / phase


def fuse_voxels(points_xyzi: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    if points_xyzi.shape[0] == 0:
        return points_xyzi

    coords = np.floor(points_xyzi[:, :3] / voxel_size).astype(np.int64)
    _, keep_idx = np.unique(coords, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    return points_xyzi[keep_idx]


def load_pred_xyz(pred_xyz_path: Path, sample_id: str) -> np.ndarray:
    """Load a PU-Net prediction file with explicit text validation.

    The official PU-Net tester writes human-readable ``.xyz`` files, so a
    decode error or non-numeric payload usually means the file is stale,
    partially written, or otherwise corrupted. We surface that failure with
    frame and path context so resume jobs can quarantine the right artifact.
    """
    try:
        with open(pred_xyz_path, "r", encoding="utf-8") as f:
            pred_xyz = np.loadtxt(f)
    except UnicodeDecodeError as exc:
        size = pred_xyz_path.stat().st_size if pred_xyz_path.exists() else -1
        raise ValueError(
            "Frame %s: unreadable PU-Net prediction text file: %s (size=%d bytes)"
            % (sample_id, pred_xyz_path, size)
        ) from exc
    except Exception as exc:
        raise ValueError(
            "Frame %s: failed to parse PU-Net prediction file as numeric text: %s"
            % (sample_id, pred_xyz_path)
        ) from exc

    pred_xyz = np.asarray(pred_xyz, dtype=np.float32)
    if pred_xyz.size == 0:
        raise ValueError("Frame %s: PU-Net prediction file is empty: %s" % (sample_id, pred_xyz_path))
    if pred_xyz.ndim == 1:
        pred_xyz = pred_xyz.reshape(1, -1)
    if pred_xyz.shape[1] < 3:
        raise ValueError(
            "Frame %s: PU-Net prediction has fewer than 3 columns: %s"
            % (sample_id, pred_xyz_path)
        )
    if not np.isfinite(pred_xyz[:, :3]).all():
        raise ValueError(
            "Frame %s: PU-Net prediction contains NaN/Inf values: %s"
            % (sample_id, pred_xyz_path)
        )
    return pred_xyz


def merge_scene_result(
    sample_id: str,
    patch_root: Path,
    punet_result_root: Path,
    output_bin_path: Path,
    voxel_size: float = 0.05,
) -> None:
    manifest_path = patch_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Missing patch manifest: %s" % manifest_path)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    merged_xyz = []
    merged_intensity = []

    for patch_meta in manifest["patches"]:
        patch_name = patch_meta["patch_name"]
        pred_xyz_path = punet_result_root / (patch_name + ".xyz")
        if not pred_xyz_path.exists():
            raise FileNotFoundError("Missing PU-Net prediction: %s" % pred_xyz_path)

        pred_xyz = load_pred_xyz(pred_xyz_path, sample_id)
        if pred_xyz.shape[1] > 3:
            pred_xyz = pred_xyz[:, :3]

        denorm_xyz = denormalize_patch(pred_xyz, np.asarray(patch_meta["centroid"], dtype=np.float32), patch_meta["scale"])

        original_patch = np.load(str(patch_root / patch_meta["source_npy"]))
        original_xyz = original_patch[:, :3].astype(np.float32)
        original_intensity = original_patch[:, 3].astype(np.float32)
        tree = cKDTree(original_xyz)
        _, nn_idx = tree.query(denorm_xyz, k=1)
        nn_idx = np.asarray(nn_idx, dtype=np.int64)
        nn_intensity = original_intensity[nn_idx]

        merged_xyz.append(denorm_xyz.astype(np.float32))
        merged_intensity.append(nn_intensity.reshape(-1, 1).astype(np.float32))

    if merged_xyz:
        xyz = np.concatenate(merged_xyz, axis=0)
        intensity = np.concatenate(merged_intensity, axis=0)
        fused = np.concatenate([xyz, intensity], axis=1)
        fused = fuse_voxels(fused, voxel_size=voxel_size)
    else:
        fused = np.zeros((0, 4), dtype=np.float32)

    save_kitti_bin(output_bin_path, fused)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare KITTI scenes for PU-Net and merge the outputs back.")
    parser.add_argument("--kitti_root", type=Path, default=DEFAULT_KITTI_ROOT, help="KITTI object training directory")
    parser.add_argument("--split_file", type=Path, default=DEFAULT_SPLIT_FILE, help="KITTI split file, such as val.txt")
    parser.add_argument("--input_subdir", type=str, default="velodyne", help="Source KITTI lidar directory name")
    parser.add_argument("--output_subdir", type=str, default="velodyne_punet_x2", help="Destination KITTI lidar directory name")
    parser.add_argument("--punet_root", type=Path, default=DEFAULT_PUNET_ROOT, help="Local PU-Net repository root")
    parser.add_argument("--punet_python", type=str, default="python", help="Python executable for the official PU-Net repo")
    parser.add_argument("--punet_log_dir", type=Path, default=DEFAULT_PUNET_ROOT / "model" / "generator2_new6", help="PU-Net log directory")
    parser.add_argument("--work_dir", type=Path, default=DEFAULT_WORK_DIR, help="Temporary workspace for patches")
    parser.add_argument("--gpu", type=str, default="0", help="GPU index used by PU-Net")
    parser.add_argument("--up_ratio", type=int, default=2, help="PU-Net upsampling ratio")
    parser.add_argument("--num_point", type=int, default=1024, help="PU-Net input point count per patch")
    parser.add_argument("--patch_size", type=float, default=12.0, help="Patch size in meters")
    parser.add_argument("--patch_stride", type=float, default=8.0, help="Patch stride in meters")
    parser.add_argument("--min_points", type=int, default=256, help="Minimum points required to keep a patch")
    parser.add_argument("--max_input_points", type=int, default=1024, help="Maximum input points per PU-Net patch")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size used to fuse overlapping outputs")
    parser.add_argument("--prepare_only", action="store_true", help="Only prepare PU-Net patches, do not run PU-Net")
    parser.add_argument("--merge_only", action="store_true", help="Only merge existing PU-Net outputs")
    parser.add_argument("--skip_existing", action="store_true", help="Skip samples whose output .bin already exists")
    parser.add_argument(
        "--retry_failed_frame_clean",
        action="store_true",
        help="Delete the current frame's patch workspace and PU-Net result folder before reprocessing it",
    )
    parser.add_argument("--sample_limit", type=int, default=0, help="Optional limit on the number of KITTI frames")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    split_ids = read_split_ids(args.split_file)
    if args.sample_limit and args.sample_limit > 0:
        split_ids = split_ids[: args.sample_limit]

    source_lidar_dir = args.kitti_root / args.input_subdir
    target_lidar_dir = args.kitti_root / args.output_subdir
    target_lidar_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for sample_id in split_ids:
        source_bin = source_lidar_dir / ("%s.bin" % sample_id)
        if not source_bin.exists():
            raise FileNotFoundError("Missing KITTI file: %s" % source_bin)

        output_bin_path = target_lidar_dir / ("%s.bin" % sample_id)
        if args.retry_failed_frame_clean:
            shutil.rmtree(args.work_dir / sample_id, ignore_errors=True)
            shutil.rmtree(args.punet_log_dir / "result" / (sample_id + "patches"), ignore_errors=True)
        if args.skip_existing and output_bin_path.exists():
            summary.append(
                {
                    "sample_id": sample_id,
                    "source_bin": str(source_bin),
                    "patch_root": None,
                    "punet_result_root": None,
                    "output_bin": str(output_bin_path),
                    "status": "skipped_existing",
                }
            )
            continue

        points_xyzi = load_kitti_bin(source_bin)
        patch_root = prepare_scene_patch_folder(
            sample_id=sample_id,
            points_xyzi=points_xyzi,
            sample_dir=args.work_dir,
            patch_size=args.patch_size,
            patch_stride=args.patch_stride,
            min_points=args.min_points,
            max_input_points=args.max_input_points,
        )

        if not args.prepare_only and not args.merge_only:
            punet_result_root = run_official_punet(
                punet_root=args.punet_root,
                punet_python=args.punet_python,
                patch_root=patch_root,
                log_dir=args.punet_log_dir,
                gpu=args.gpu,
                up_ratio=args.up_ratio,
                num_point=args.num_point,
            )
        else:
            phase = patch_root.parent.name + patch_root.name
            punet_result_root = args.punet_log_dir / "result" / phase

        if not args.prepare_only:
            merge_scene_result(
                sample_id=sample_id,
                patch_root=patch_root,
                punet_result_root=punet_result_root,
                output_bin_path=output_bin_path,
                voxel_size=args.voxel_size,
            )

        summary.append(
            {
                "sample_id": sample_id,
                "source_bin": str(source_bin),
                "patch_root": str(patch_root),
                "punet_result_root": str(punet_result_root),
                "output_bin": str(output_bin_path),
                "status": "completed",
            }
        )

    summary_path = args.work_dir / "punet_kitti_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("PU-Net KITTI adaptation completed.")
    print("Summary written to: %s" % summary_path)


if __name__ == "__main__":
    main()
