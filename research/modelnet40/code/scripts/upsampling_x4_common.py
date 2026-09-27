#!/usr/bin/env python3
"""Shared ModelNet40 ×4 upsampling utilities (all main methods)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ear_modelnet40_utils import (
    GLOBAL_SEED,
    INPUT_POINTS_DOWN,
    INPUT_POINTS_ORIGINAL,
    TARGET_POINTS_X4_DOWN,
    TARGET_POINTS_X4_ORIGINAL,
    UP_FACTOR_X4,
    ensure_metadata_link,
    is_valid_output_npy,
    resample_to_target,
    stable_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_TO_IDX = PROJECT_ROOT / "datasets" / "modelnet40_original" / "metadata" / "class_to_idx.json"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "step7_smoke_sample_manifest.csv"
LEGACY_EAR_X2 = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear"

METHOD_OUTPUT_SUBDIRS = {
    "pdans": "pdans_x4",
    "punet": "punet_x4",
    "pugcn": "pugcn_x4",
}

LINE_VARIANTS = {
    "A": {
        "line": "A",
        "source_variant": "original",
        "input_key": "smoke_original_path",
        "input_root": PROJECT_ROOT / "datasets" / "modelnet40_original",
        "expected_input": INPUT_POINTS_ORIGINAL,
        "target_points": TARGET_POINTS_X4_ORIGINAL,
        "output_suffix": "original_up",
    },
    "B": {
        "line": "B",
        "source_variant": "downsampled50",
        "input_key": "smoke_downsampled50_path",
        "input_root": PROJECT_ROOT / "datasets" / "modelnet40_downsampled50",
        "expected_input": INPUT_POINTS_DOWN,
        "target_points": TARGET_POINTS_X4_DOWN,
        "output_suffix": "downsampled50_up",
    },
}


def output_root(method: str, line: str, smoke: bool = False, smoke_tag: str = "") -> Path:
    from upsampling_x4_factory import normalize_method

    method = normalize_method(method)
    if method not in METHOD_OUTPUT_SUBDIRS:
        raise ValueError(f"Unknown method: {method}")
    subdir = METHOD_OUTPUT_SUBDIRS[method]
    if smoke:
        subdir = f"{subdir}_smoke"
        if smoke_tag:
            subdir = f"{subdir}_{smoke_tag}"
    variant = LINE_VARIANTS[line]
    return PROJECT_ROOT / "datasets" / f"modelnet40_{variant['output_suffix']}" / subdir


def smoke_variant_config(method: str, line: str, smoke_tag: str = "") -> dict:
    from upsampling_x4_factory import normalize_method

    base = dict(LINE_VARIANTS[line])
    base["method"] = normalize_method(method)
    base["output_root"] = output_root(method, line, smoke=True, smoke_tag=smoke_tag)
    base["smoke_tag"] = smoke_tag
    return base


def full_line_config(method: str, line: str) -> dict:
    from upsampling_x4_factory import normalize_method

    base = dict(LINE_VARIANTS[line])
    base["method"] = normalize_method(method)
    base["output_root"] = output_root(method, line, smoke=False)
    base["chunk_audit_dir"] = (
        PROJECT_ROOT
        / "logs"
        / f"{method.lower()}_x4_generation"
        / f"line{line}_chunk_audits"
    )
    return base


def load_smoke_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> list[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_full_manifest(input_root: Path) -> list[dict]:
    rows: list[dict] = []
    metadata_dir = input_root / "metadata"
    for split in ("train", "test"):
        manifest_path = metadata_dir / f"{split}_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["split"] = row.get("split") or split
                rows.append(row)
    rows.sort(key=lambda r: (r["split"], r["class_name"], r["shape_id"]))
    return rows


def chunk_rows(rows: list[dict], chunk_id: int, num_chunks: int) -> list[dict]:
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError(f"chunk_id={chunk_id} out of range for num_chunks={num_chunks}")
    chunk_size = (len(rows) + num_chunks - 1) // num_chunks
    start = chunk_id * chunk_size
    end = min(start + chunk_size, len(rows))
    return rows[start:end]


def load_class_to_idx(path: Path | None = None) -> dict:
    src = path or CLASS_TO_IDX
    return json.loads(src.read_text(encoding="utf-8"))


def finalize_output(
    raw_xyz: np.ndarray,
    input_points: int,
    target_n: int,
    seed: int,
) -> np.ndarray:
    final_xyz = resample_to_target(raw_xyz, target_n, seed)
    if final_xyz.shape != (target_n, 3):
        raise ValueError(f"Bad shape {final_xyz.shape}, expected ({target_n}, 3)")
    if not np.isfinite(final_xyz).all():
        raise ValueError("Output NaN/Inf")
    return final_xyz


def count_npy_files(root: Path) -> dict[str, int]:
    counts = {"train": 0, "test": 0}
    for split in ("train", "test"):
        split_dir = root / split
        if split_dir.is_dir():
            counts[split] = sum(1 for _ in split_dir.rglob("*.npy"))
    return counts


__all__ = [
    "GLOBAL_SEED",
    "UP_FACTOR_X4",
    "PROJECT_ROOT",
    "CLASS_TO_IDX",
    "DEFAULT_MANIFEST",
    "LEGACY_EAR_X2",
    "METHOD_OUTPUT_SUBDIRS",
    "LINE_VARIANTS",
    "output_root",
    "smoke_variant_config",
    "full_line_config",
    "load_smoke_manifest",
    "load_full_manifest",
    "chunk_rows",
    "load_class_to_idx",
    "ensure_metadata_link",
    "is_valid_output_npy",
    "stable_seed",
    "finalize_output",
    "count_npy_files",
]
