#!/usr/bin/env python3
"""DataLoader for preprocessed ModelNet40 .npy point clouds."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    max_dist = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if max_dist <= 0:
        raise ValueError("Degenerate point cloud in pc_normalize")
    return pc / max_dist


def stable_seed(global_seed: int, *parts: str) -> int:
    token = ":".join([str(global_seed), *parts]).encode("utf-8")
    digest = hashlib.md5(token).hexdigest()
    return global_seed + (int(digest[:8], 16) % 1_000_000)


def resample_points(points: np.ndarray, num_points: int, seed: int) -> np.ndarray:
    """Resample point cloud to target count (upsample with replacement or downsample without)."""
    count = points.shape[0]
    if count == num_points:
        return points.copy()
    rng = np.random.default_rng(seed)
    if count > num_points:
        indices = rng.choice(count, size=num_points, replace=False)
    else:
        indices = rng.choice(count, size=num_points, replace=True)
    return points[indices].astype(np.float32)


class ModelNetNPYDataset(Dataset):
    """Load ModelNet40 samples from train/test .npy layout + manifest metadata."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        num_points: int = 1024,
        normalize: bool = True,
        allow_resample: bool = True,
        resample_seed: int = 42,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.allow_resample = allow_resample
        self.resample_seed = resample_seed

        metadata_dir = self.data_root / "metadata"
        class_to_idx_path = metadata_dir / "class_to_idx.json"
        manifest_path = metadata_dir / f"{split}_manifest.csv"

        if not class_to_idx_path.is_file():
            raise FileNotFoundError(f"Missing label map: {class_to_idx_path}")
        with open(class_to_idx_path, encoding="utf-8") as handle:
            self.class_to_idx = json.load(handle)

        self.samples: list[tuple[Path, int, str, str]] = []
        if manifest_path.is_file():
            with open(manifest_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    npy_key = "output_npy" if "output_npy" in row else "output_path"
                    npy_path = Path(row[npy_key])
                    if not npy_path.is_file():
                        npy_path = self.data_root.parent / npy_path
                    if not npy_path.is_file():
                        npy_path = self.data_root / split / row["class_name"] / f"{row['shape_id']}.npy"
                    label = int(row["label"])
                    shape_id = row.get("shape_id", npy_path.stem)
                    self.samples.append((npy_path, label, row["class_name"], shape_id))
        else:
            split_dir = self.data_root / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Split directory not found: {split_dir}")
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    raise KeyError(f"Class {class_name} missing from class_to_idx.json")
                label = self.class_to_idx[class_name]
                for npy_path in sorted(class_dir.glob("*.npy")):
                    self.samples.append((npy_path, label, class_name, npy_path.stem))

        if not self.samples:
            raise RuntimeError(f"No samples found for split={split} under {self.data_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        npy_path, label, class_name, shape_id = self.samples[index]
        points = np.load(npy_path).astype(np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"{npy_path} has invalid shape {points.shape}, expected (N, 3)")

        if points.shape[0] != self.num_points:
            if not self.allow_resample:
                raise ValueError(
                    f"{npy_path} has shape {points.shape}, expected ({self.num_points}, 3)"
                )
            seed = stable_seed(self.resample_seed, self.split, class_name, shape_id, "to1024")
            points = resample_points(points, self.num_points, seed)

        if points.shape != (self.num_points, 3):
            raise ValueError(f"{npy_path} resampled to {points.shape}, expected ({self.num_points}, 3)")
        if not np.isfinite(points).all():
            raise ValueError(f"{npy_path} contains non-finite values after resampling")
        if self.normalize:
            points = pc_normalize(points)
        return points, label
