#!/usr/bin/env python3
"""PU-Net ModelNet40 ×4 upsampling wrapper (TensorFlow 1.x)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_PUNET_ROOT = Path(
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PU-Net"
)
DEFAULT_MODEL_DIR = DEFAULT_PUNET_ROOT / "model" / "generator2_new6"
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


class PUNetUpsampler:
    def __init__(
        self,
        punet_root: Path = DEFAULT_PUNET_ROOT,
        model_dir: Path = DEFAULT_MODEL_DIR,
        tf_python: Path = DEFAULT_TF_PYTHON,
        up_ratio: int = DEFAULT_UP_RATIO,
        gpu: str = "0",
    ) -> None:
        self.punet_root = punet_root
        self.model_dir = model_dir
        self.tf_python = tf_python
        self.up_ratio = up_ratio
        self.gpu = gpu
        if not self.tf_python.is_file():
            raise FileNotFoundError(f"PU-Net python missing: {self.tf_python}")
        if not (self.model_dir / "checkpoint").is_file():
            raise FileNotFoundError(f"PU-Net checkpoint missing under {self.model_dir}")

    def upsample_xyz(self, points_xyz: np.ndarray, sample_name: str = "input") -> np.ndarray:
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) xyz, got {points_xyz.shape}")
        if not np.isfinite(points_xyz).all():
            raise ValueError("Input NaN/Inf")

        with tempfile.TemporaryDirectory(prefix="punet_m40_") as tmp:
            data_dir = Path(tmp) / "in"
            data_dir.mkdir(parents=True, exist_ok=True)
            in_path = data_dir / f"{sample_name}.xyz"
            _save_xyz(in_path, points_xyz)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.gpu
            # Ensure TF1.15 cuda compat + PU-Net code on path for subprocess.
            punet_code = str(self.punet_root / "code")
            env["PYTHONPATH"] = punet_code + os.pathsep + env.get("PYTHONPATH", "")
            conda_lib = Path.home() / ".conda/envs/tf15_upsampling/lib"
            if conda_lib.is_dir():
                env["LD_LIBRARY_PATH"] = (
                    f"{conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"
                )
            cmd = [
                str(self.tf_python),
                "main.py",
                "--phase",
                "test",
                "--gpu",
                self.gpu,
                "--log_dir",
                str(self.model_dir),
                "--data_folder",
                str(data_dir),
                "--up_ratio",
                str(self.up_ratio),
                "--num_point",
                str(points_xyz.shape[0]),
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(self.punet_root / "code"),
                env=env,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"PU-Net inference failed (rc={proc.returncode}):\n"
                    f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
                )

            out_path = self.model_dir / "result" / data_dir.name / f"{sample_name}.xyz"
            if not out_path.is_file():
                candidates = list((self.model_dir / "result").rglob(f"{sample_name}.xyz"))
                if not candidates:
                    raise FileNotFoundError(f"PU-Net output not found for {sample_name}")
                out_path = candidates[0]

            out = _load_xyz(out_path)
            if not np.isfinite(out).all():
                raise ValueError("PU-Net output NaN/Inf")
            return out


def load_punet_upsampler(**kwargs) -> PUNetUpsampler:
    return PUNetUpsampler(**kwargs)
