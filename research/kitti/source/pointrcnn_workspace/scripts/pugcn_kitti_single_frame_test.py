#!/usr/bin/env python3
"""Prepare and optionally run official PU-GCN inference on one KITTI frame.

The official PU-GCN tester consumes xyz text files and writes xyz predictions.
This adapter keeps the conversion outside the official repository:

1. read a KITTI ``N x 4`` velodyne ``.bin`` file,
2. crop/sample a manageable xyz patch,
3. save the patch as ``.npy``, ``.ply``, and official ``.xyz`` input,
4. optionally call PU-GCN's ``main.py --phase test``,
5. convert the xyz output back to KITTI-compatible ``M x 4`` using nearest
   original intensity.

It never overwrites KITTI raw data.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KITTI_ROOT = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
DEFAULT_INPUT_BIN = DEFAULT_KITTI_ROOT / "velodyne_original_val" / "000001.bin"
DEFAULT_PUGCN_ROOT = PROJECT_ROOT / "external" / "PU-GCN"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "pugcn_single_frame"
KITTI_COLUMNS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-frame KITTI adapter for PU-GCN.")
    parser.add_argument("--input-bin", type=Path, default=DEFAULT_INPUT_BIN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pugcn-root", type=Path, default=DEFAULT_PUGCN_ROOT)
    parser.add_argument("--restore", type=Path, default=DEFAULT_PUGCN_ROOT / "pretrained" / "pu1k-pugcn")
    parser.add_argument("--pugcn-python", type=str, default=sys.executable)
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--crop-mode", choices=("front", "near", "all"), default="front")
    parser.add_argument("--x-range", nargs=2, type=float, default=(0.0, 45.0))
    parser.add_argument("--y-range", nargs=2, type=float, default=(-20.0, 20.0))
    parser.add_argument("--z-range", nargs=2, type=float, default=(-3.0, 2.0))
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--patch-num-point", type=int, default=256)
    parser.add_argument("--patch-num-ratio", type=int, default=3)
    parser.add_argument("--up-ratio", type=int, default=4)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--keep-official-result", action="store_true")
    return parser.parse_args()


def load_kitti_bin(path: Path) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32)
    if points.size % KITTI_COLUMNS != 0:
        raise ValueError("Invalid KITTI .bin shape: %s" % path)
    return points.reshape(-1, KITTI_COLUMNS)


def save_kitti_bin(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points.astype(np.float32).tofile(path)


def save_xyz(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, xyz.astype(np.float32), fmt="%.6f")


def save_ply(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % len(xyz))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for point in xyz:
            f.write("%.6f %.6f %.6f\n" % (point[0], point[1], point[2]))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def crop_points(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    xyz = points[:, :3]
    if args.crop_mode == "all":
        cropped = points
    elif args.crop_mode == "near":
        dist = np.linalg.norm(xyz[:, :2], axis=1)
        order = np.argsort(dist)
        cropped = points[order[: max(args.max_points, args.patch_num_point)]]
    else:
        x0, x1 = args.x_range
        y0, y1 = args.y_range
        z0, z1 = args.z_range
        mask = (
            (xyz[:, 0] >= x0)
            & (xyz[:, 0] <= x1)
            & (xyz[:, 1] >= y0)
            & (xyz[:, 1] <= y1)
            & (xyz[:, 2] >= z0)
            & (xyz[:, 2] <= z1)
        )
        cropped = points[mask]
    if len(cropped) == 0:
        raise ValueError("Crop produced no points; loosen the crop bounds.")
    return cropped


def sample_points(points: np.ndarray, max_points: int, min_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(points) > max_points:
        idx = rng.choice(len(points), size=max_points, replace=False)
        idx.sort()
        return points[idx]
    if len(points) < min_points:
        idx = rng.choice(len(points), size=min_points, replace=True)
        return points[idx]
    return points


def load_xyz(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError("Expected xyz columns in %s" % path)
    return arr[:, :3].astype(np.float32)


def nearest_intensity(source_xyzi: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(source_xyzi[:, :3]).query(query_xyz, k=1)
    except Exception:
        diff = query_xyz[:, None, :] - source_xyzi[None, :, :3]
        idx = np.argmin(np.sum(diff * diff, axis=2), axis=1)
    return source_xyzi[np.asarray(idx, dtype=np.int64), 3].reshape(-1, 1).astype(np.float32)


def run_official_pugcn(args: argparse.Namespace, input_dir: Path) -> Path:
    pugcn_root = args.pugcn_root.resolve()
    restore = args.restore.resolve()
    input_dir = input_dir.resolve()
    if not (restore / "checkpoint").exists():
        raise FileNotFoundError(
            "No PU-GCN checkpoint found at %s. Download pretrained weights into this folder first."
            % restore
        )
    required_ops = [
        pugcn_root / "tf_ops" / "sampling" / "tf_sampling_so.so",
        pugcn_root / "tf_ops" / "grouping" / "tf_grouping_so.so",
        pugcn_root / "tf_ops" / "nn_distance" / "tf_nndistance_so.so",
    ]
    missing_ops = [str(path) for path in required_ops if not path.exists()]
    if missing_ops:
        raise FileNotFoundError("PU-GCN tf_ops are not compiled: %s" % ", ".join(missing_ops))

    result_dir = pugcn_root / "evaluation_code" / "result"
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
        str(args.up_ratio),
        "--patch_num_point",
        str(args.patch_num_point),
        "--patch_num_ratio",
        str(args.patch_num_ratio),
    ]
    env = dict(**os_environ(), CUDA_VISIBLE_DEVICES=args.gpu)
    subprocess.run(cmd, cwd=pugcn_root, env=env, check=True)
    return result_dir


def os_environ() -> Dict[str, str]:
    import os

    return dict(os.environ)


def write_command_log(path: Path, rows: Sequence[Tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    sample_id = args.sample_id or args.input_bin.stem
    frame_dir = args.output_root / sample_id
    input_dir = frame_dir / "official_input_xyz"
    frame_dir.mkdir(parents=True, exist_ok=True)

    original = load_kitti_bin(args.input_bin)
    cropped = crop_points(original, args)
    selected = sample_points(cropped, args.max_points, args.patch_num_point, args.seed)

    input_npy = frame_dir / "input_sampled_xyzi.npy"
    input_bin = frame_dir / "input_sampled_xyzi.bin"
    input_ply = frame_dir / "input_sampled_xyz.ply"
    input_xyz = input_dir / f"{sample_id}.xyz"
    np.save(input_npy, selected.astype(np.float32))
    save_kitti_bin(input_bin, selected.astype(np.float32))
    save_ply(input_ply, selected[:, :3])
    save_xyz(input_xyz, selected[:, :3])

    manifest = {
        "sample_id": sample_id,
        "source_bin": str(args.input_bin.resolve()),
        "source_points": int(len(original)),
        "cropped_points": int(len(cropped)),
        "selected_points": int(len(selected)),
        "input_bin": str(input_bin.resolve()),
        "input_xyz": str(input_xyz.resolve()),
        "intensity_strategy": "PU-GCN receives xyz only; output intensity is nearest-neighbor interpolated from selected KITTI input points.",
        "official_repo": str(args.pugcn_root.resolve()),
        "restore": str(args.restore.resolve()),
        "ran_inference": bool(args.run_inference),
    }

    if args.run_inference:
        result_dir = run_official_pugcn(args, input_dir)
        pred_xyz = result_dir / f"{sample_id}.xyz"
        if not pred_xyz.exists():
            raise FileNotFoundError("PU-GCN did not produce expected output %s" % pred_xyz)
        output_xyz = load_xyz(pred_xyz)
        output_xyzi = np.concatenate([output_xyz, nearest_intensity(selected, output_xyz)], axis=1)
        np.save(frame_dir / "pugcn_output_xyzi.npy", output_xyzi.astype(np.float32))
        save_ply(frame_dir / "pugcn_output_xyz.ply", output_xyz)
        save_kitti_bin(frame_dir / f"{sample_id}_pugcn.bin", output_xyzi)
        if args.keep_official_result:
            shutil.copy2(pred_xyz, frame_dir / "pugcn_output.xyz")
        manifest["output_points"] = int(len(output_xyzi))
        manifest["output_bin"] = str((frame_dir / f"{sample_id}_pugcn.bin").resolve())

    write_json(frame_dir / "manifest.json", manifest)
    write_command_log(
        frame_dir / "command_record.csv",
        [
            ("script", str(Path(__file__).resolve())),
            ("input_bin", str(args.input_bin)),
            ("output_root", str(args.output_root)),
            ("run_inference", str(args.run_inference)),
            ("crop_mode", args.crop_mode),
            ("max_points", str(args.max_points)),
        ],
    )
    print("Prepared PU-GCN KITTI single-frame workspace:", frame_dir)


if __name__ == "__main__":
    main()
