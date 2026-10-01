#!/usr/bin/env python3
"""Run the archived PointNet++ baseline on one real, included ModelNet40 cloud.

This is a model-inference demonstration on a selected sample, not a dataset
accuracy estimate, new training run, or implementation of an upsampling method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "research/modelnet40/results/pointnet2_final_runs/lineA_original_baseline"
DEFAULT_SAMPLE = (
    ROOT
    / "thesis/evidence/modelnet40_hpc/data/pointcloud_examples/lineA_airplane_0627_original.npy"
)


def require_asset(path):
    if not path.is_file():
        raise ValueError(
            f"Missing asset: {path}. Run the selective Git LFS download in docs/REPRODUCING.md."
        )
    with path.open("rb") as stream:
        if stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(
                f"{path.name} is a Git LFS pointer, not the asset. Run git lfs pull as documented."
            )


def label_map():
    path = RUN / "lineA_original_baseline_result.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    start = rows.index(["class_idx", "class_name", "accuracy"]) + 1
    labels = {int(row[0]): row[1] for row in rows[start:] if row}
    if set(labels) != set(range(40)):
        raise ValueError(
            "The archived classifier label map must contain all 40 classes."
        )
    return labels, path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=Path,
        default=DEFAULT_SAMPLE,
        help="An XYZ .npy cloud with exactly 1024 points",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4, help="CPU intra-op threads")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/demo/prediction.json"
    )
    args = parser.parse_args()
    try:
        import numpy as np
        import torch
    except ImportError as error:
        parser.exit(
            2,
            f"Demo dependency unavailable: {error}. Install requirements.txt in a Python 3.12 virtual environment.\n",
        )
    try:
        if args.threads < 1:
            raise ValueError("--threads must be positive")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA is unavailable. Use --device cpu, or install a compatible GPU PyTorch build."
            )
        checkpoint_path = RUN / "checkpoints/best_model.pth"
        for path in [
            checkpoint_path,
            args.sample,
            RUN / "pointnet2_cls_ssg.py",
            RUN / "pointnet2_utils.py",
        ]:
            require_asset(path)
        labels, label_source = label_map()
        points = np.load(args.sample, allow_pickle=False).astype(np.float32)
        if points.shape != (1024, 3) or not np.isfinite(points).all():
            raise ValueError(
                f"Expected finite XYZ shape (1024, 3), got {points.shape}. No silent resampling is performed."
            )
        # Match the archived ModelNetNPYDataset centering and unit-sphere normalization.
        points = points - points.mean(axis=0)
        radius = np.max(np.sqrt(np.sum(points**2, axis=1)))
        if radius <= 0:
            raise ValueError("Degenerate point cloud: zero radius after centering")
        points = points / radius
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.threads)
        sys.path.insert(0, str(RUN))
        model_module = importlib.import_module("pointnet2_cls_ssg")
        model = model_module.get_model(40, normal_channel=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(args.device).eval()
        tensor = torch.from_numpy(points).transpose(0, 1).unsqueeze(0).to(args.device)
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            log_probabilities, _ = model(tensor)
            probabilities = log_probabilities.exp()[0]
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if not torch.isfinite(probabilities).all() or not torch.isclose(
            probabilities.sum(), torch.tensor(1.0, device=args.device), atol=1e-5
        ):
            raise ValueError("Invalid model probability distribution")
        scores, indices = probabilities.topk(5)
        result = {
            "scope": "selected-sample inference demonstration; not a benchmark accuracy estimate",
            "model": "archived PointNet++ SSG Original 1024 baseline",
            "sample": args.sample.name,
            "points": 1024,
            "normalization": "centroid subtraction and maximum-radius scaling; identical to archived loader",
            "device": args.device,
            "seed": args.seed,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_variant": checkpoint["variant"],
            "forward_seconds": elapsed,
            "top5": [
                {
                    "class_index": int(i),
                    "class_name": labels[int(i)],
                    "probability": float(s),
                }
                for s, i in zip(scores.cpu(), indices.cpu())
            ],
            "sha256": {
                "checkpoint": digest(checkpoint_path),
                "sample": digest(args.sample),
                "label_source": digest(label_source),
                "model_source": digest(RUN / "pointnet2_cls_ssg.py"),
                "model_utils": digest(RUN / "pointnet2_utils.py"),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in ["sample", "points", "device", "checkpoint_epoch", "top5"]
                },
                indent=2,
            )
        )
        print(f"Prediction saved to {args.output.resolve()}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Inference failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
