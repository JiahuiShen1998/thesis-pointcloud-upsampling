#!/usr/bin/env python3
"""PU-GCN ModelNet40 ×4 upsampling wrapper (TensorFlow 1.x)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_PUGCN_ROOT = Path(
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-GCN"
)
DEFAULT_RESTORE = DEFAULT_PUGCN_ROOT / "pretrained" / "pu1k-pugcn"
DEFAULT_TF_PYTHON = Path.home() / ".conda/envs/tf15_upsampling/bin/python"
DEFAULT_UP_RATIO = 4


def _save_xyz(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, xyz.astype(np.float32), fmt="%.6f")


def _load_xyz(path: Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, :3].astype(np.float32)


class PUGCNUpsampler:
    def __init__(
        self,
        pugcn_root: Path = DEFAULT_PUGCN_ROOT,
        restore_dir: Path = DEFAULT_RESTORE,
        tf_python: Path = DEFAULT_TF_PYTHON,
        up_ratio: int = DEFAULT_UP_RATIO,
        gpu: str = "0",
        k: int = 20,
    ) -> None:
        self.pugcn_root = pugcn_root
        self.restore_dir = restore_dir
        self.tf_python = tf_python
        self.up_ratio = up_ratio
        self.gpu = gpu
        self.k = k
        if not self.tf_python.is_file():
            raise FileNotFoundError(f"PU-GCN python missing: {self.tf_python}")
        if not (self.restore_dir / "checkpoint").is_file():
            raise FileNotFoundError(f"PU-GCN checkpoint missing under {self.restore_dir}")

    def upsample_xyz(self, points_xyz: np.ndarray, sample_name: str = "input") -> np.ndarray:
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) xyz, got {points_xyz.shape}")
        if not np.isfinite(points_xyz).all():
            raise ValueError("Input NaN/Inf")

        with tempfile.TemporaryDirectory(prefix="pugcn_m40_") as tmp:
            data_dir = Path(tmp) / "in"
            data_dir.mkdir(parents=True, exist_ok=True)
            in_path = data_dir / f"{sample_name}.xyz"
            _save_xyz(in_path, points_xyz)

            result_dir = Path(tmp) / "pugcn_out"
            result_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
            env["PUGCN_OUT_FOLDER"] = str(result_dir)
            cmd = [
                str(self.tf_python),
                "main.py",
                "--phase",
                "test",
                "--restore",
                str(self.restore_dir),
                "--data_dir",
                str(data_dir),
                "--model",
                "pugcn",
                "--upsampler",
                "nodeshuffle",
                "--up_ratio",
                str(self.up_ratio),
                "--num_point",
                str(points_xyz.shape[0]),
                "--k",
                str(self.k),
                "--seed",
                "42",
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(self.pugcn_root),
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"PU-GCN inference failed (rc={proc.returncode}):\n"
                    f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
                )

            out_path = result_dir / f"{sample_name}.xyz"
            if not out_path.is_file():
                candidates = list(result_dir.glob("*.xyz"))
                if not candidates:
                    raise FileNotFoundError(f"PU-GCN output not found for {sample_name}")
                out_path = candidates[0]

            out = _load_xyz(out_path)
            if not np.isfinite(out).all():
                raise ValueError("PU-GCN output NaN/Inf")
            return out


def load_pugcn_upsampler(**kwargs) -> PUGCNUpsampler:
    return PUGCNUpsampler(**kwargs)
