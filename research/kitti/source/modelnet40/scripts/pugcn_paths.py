#!/usr/bin/env python3
"""Shared path/config helpers for ModelNet40 PU-GCN x4 protocol."""

import os
from pathlib import Path

WOODY_PROJECT_ROOT = Path(
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling"
)

# Override via env on agent/dev machine when woody path is not mounted.
LOCAL_FALLBACK_ROOT = Path("/home/ra87racy/projects/modelnet40_pointnet2_upsampling")
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("MODELNET40_PROJECT_ROOT", str(WOODY_PROJECT_ROOT))
)

DEFAULT_PUGCN_REPO = Path(
    os.environ.get(
        "PUGCN_REPO",
        "/home/ra87racy/projects/baseline_detectors/PointRCNN/external/PU-GCN",
    )
)
DEFAULT_PUGCN_CKPT = Path(
    os.environ.get(
        "PUGCN_CKPT",
        str(DEFAULT_PUGCN_REPO / "pretrained" / "pu1k-pugcn"),
    )
)
DEFAULT_PUGCN_PYTHON = Path(
    os.environ.get(
        "PUGCN_PYTHON",
        "/home/ra87racy/miniconda3/envs/pugcn/bin/python",
    )
)

DEFAULT_SEED = 20260705

LINE_DEFAULTS = {
    "lineA": {
        "input_root": "datasets/modelnet40_original_1024",
        "raw_output_root": "datasets/lineA_original_up/raw/pu_gcn",
        "strict_output_root": "datasets/lineA_original_up/strict_4N/pu_gcn",
        "expected_input_points": 1024,
        "expected_output_points": 4096,
    },
    "lineB": {
        "input_root": "datasets/modelnet40_downsampled_x4",
        "raw_output_root": "datasets/lineB_downsampled_x4_up/raw/pu_gcn",
        "strict_output_root": "datasets/lineB_downsampled_x4_up/strict_N/pu_gcn",
        "expected_input_points": 256,
        "expected_output_points": 1024,
    },
}


def resolve_project_root(project_root=None):
    root = Path(project_root) if project_root else DEFAULT_PROJECT_ROOT
    if root.exists():
        return root.resolve()
    if LOCAL_FALLBACK_ROOT.exists():
        return LOCAL_FALLBACK_ROOT.resolve()
    return root


def line_paths(line, project_root=None):
    root = resolve_project_root(project_root)
    if line not in LINE_DEFAULTS:
        raise ValueError(f"unknown line: {line}")
    cfg = LINE_DEFAULTS[line]
    return {
        "project_root": root,
        "input_root": root / cfg["input_root"],
        "raw_output_root": root / cfg["raw_output_root"],
        "strict_output_root": root / cfg["strict_output_root"],
        "expected_input_points": int(cfg["expected_input_points"]),
        "expected_output_points": int(cfg["expected_output_points"]),
    }
