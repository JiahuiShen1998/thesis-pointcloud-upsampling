#!/usr/bin/env python3
"""PDANS ModelNet40 ×4 upsampling wrapper."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

DEFAULT_PDANS_ROOT = Path(
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/PDANS"
)
DEFAULT_CHECKPOINT = DEFAULT_PDANS_ROOT / "checkpoints" / "PU1K_PDANS.pkl"
DEFAULT_CONFIG = DEFAULT_PDANS_ROOT / "pointnet2" / "exp_configs" / "PU1K.json"
DEFAULT_R = 4
DEFAULT_STEP = 30
DEFAULT_GAMMA = 0.5


def _ensure_pdans_paths() -> None:
    root = str(DEFAULT_PDANS_ROOT)
    pointnet2 = str(DEFAULT_PDANS_ROOT / "pointnet2")
    ops_lib = str(DEFAULT_PDANS_ROOT / "pointnet2_ops_lib")
    for entry in (root, pointnet2, ops_lib):
        if entry not in sys.path:
            sys.path.insert(0, entry)


class PDANSUpsampler:
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        config_path: Path = DEFAULT_CONFIG,
        r: int = DEFAULT_R,
        step: int = DEFAULT_STEP,
        gamma: float = DEFAULT_GAMMA,
    ) -> None:
        _ensure_pdans_paths()
        from json_reader import restore_string_to_list_in_a_dict
        from models.pointnet2_with_pcld_condition import PointNet2CloudCondition
        from util import calc_diffusion_hyperparams, sampling_ddim

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"PDANS checkpoint missing: {checkpoint_path}")
        if not config_path.is_file():
            raise FileNotFoundError(f"PDANS config missing: {config_path}")

        with open(config_path, encoding="utf-8") as handle:
            config = restore_string_to_list_in_a_dict(json.load(handle))

        diffusion_config = config["diffusion_config"]
        diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_config)
        for key in diffusion_hyperparams:
            if key != "T":
                diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

        self.net = PointNet2CloudCondition(config["pointnet_config"]).cuda()
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.net.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.net.eval()

        self.diffusion_hyperparams = diffusion_hyperparams
        self.diffusion_config = diffusion_config
        self.data_scale = config["pu1k_dataset_config"]["scale"]
        self.r = r
        self.step = step
        self.gamma = gamma
        self._sampling_ddim = sampling_ddim

    def upsample_xyz(self, points_xyz: np.ndarray, sample_name: str = "input") -> np.ndarray:
        if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
            raise ValueError(f"Expected (N, 3) xyz, got {points_xyz.shape}")
        if not np.isfinite(points_xyz).all():
            raise ValueError("Input NaN/Inf")

        condition = (
            torch.from_numpy(points_xyz.astype(np.float32))
            .unsqueeze(0)
            .cuda()
        )
        batch = 1
        num_points = self.r * condition.shape[1]
        label = torch.from_numpy(
            np.full(shape=(1,), fill_value=self.r - 1, dtype=np.int64)
        ).cuda()
        self.net.reset_cond_features()

        with torch.no_grad():
            generated_data, _, _ = self._sampling_ddim(
                net=self.net,
                size=(batch, num_points, 3),
                diffusion_hyperparams=self.diffusion_hyperparams,
                label=label,
                condition=condition,
                R=self.r,
                gamma=self.gamma,
                step=self.step,
            )
        out = (generated_data[0].detach().cpu().numpy() / self.data_scale).astype(
            np.float32
        )
        if not np.isfinite(out).all():
            raise ValueError("PDANS output NaN/Inf")
        return out


def load_pdans_upsampler(**kwargs) -> PDANSUpsampler:
    if not torch.cuda.is_available():
        raise RuntimeError("PDANS requires CUDA")
    return PDANSUpsampler(**kwargs)
