#!/usr/bin/env python3
"""Run the local EAR KITTI method entrypoint and save raw output provenance."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


EAR_CODE_DIR = Path("/home/ra87racy/projects/upsampling/EAR")
EAR_ENTRYPOINT = EAR_CODE_DIR / "ear_upsampling.py"


def load_kitti_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(f"{path} is not KITTI XYZI: value count is not divisible by 4")
    return values.reshape(-1, 4)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real EAR raw inference for one KITTI .bin frame")
    parser.add_argument("--input_bin", required=True)
    parser.add_argument("--raw_output", required=True)
    parser.add_argument("--provenance_json", required=True)
    parser.add_argument("--line", required=True)
    parser.add_argument("--frame_id", required=True)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--up_ratio", type=float, default=4.0)
    parser.add_argument("--k_neighbors", type=int, default=20)
    parser.add_argument("--edge_sensitivity", type=float, default=4.0)
    parser.add_argument("--up_threshold", type=float, default=0.93)
    parser.add_argument("--sigma_p", type=float, default=0.0)
    parser.add_argument("--max_iter", type=int, default=5)
    args = parser.parse_args()

    input_bin = Path(args.input_bin).resolve()
    raw_output = Path(args.raw_output).resolve()
    provenance_json = Path(args.provenance_json).resolve()

    command = " ".join(sys.argv)
    provenance = {
        "method": "EAR",
        "line": args.line,
        "frame_id": args.frame_id,
        "input_bin": str(input_bin),
        "input_point_count": None,
        "checkpoint_required": False,
        "checkpoint_path": None,
        "checkpoint_loaded": "not_required",
        "method_entrypoint": str(EAR_ENTRYPOINT),
        "method_inference_called": False,
        "method_algorithm_called": False,
        "raw_output_path": str(raw_output),
        "raw_output_point_count": None,
        "raw_output_format": "KITTI_XYZI_float32_bin",
        "raw_output_generated_from": None,
        "fallback_used": False,
        "generic_adapter_used_before_raw": False,
        "env_name": os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("VIRTUAL_ENV") or "unknown",
        "python_path": sys.executable,
        "python_version": platform.python_version(),
        "command": command,
        "status": "STARTED",
        "notes": [],
    }

    start = time.time()
    try:
        if not EAR_ENTRYPOINT.exists():
            raise FileNotFoundError(f"EAR entrypoint not found: {EAR_ENTRYPOINT}")
        sys.path.insert(0, str(EAR_CODE_DIR))
        from ear_upsampling import ear_upsample_kitti  # type: ignore

        points = load_kitti_bin(input_bin)
        provenance["input_point_count"] = int(points.shape[0])
        target_n = int(points.shape[0] * args.up_ratio)
        if target_n <= points.shape[0]:
            raise ValueError("--up_ratio must request more raw points than the input")

        raw_output.parent.mkdir(parents=True, exist_ok=True)
        provenance["method_inference_called"] = True
        raw_points = ear_upsample_kitti(
            points,
            target_n=target_n,
            up_factor=args.up_ratio,
            k_neighbors=args.k_neighbors,
            edge_sensitivity=args.edge_sensitivity,
            up_threshold=args.up_threshold,
            sigma_p=args.sigma_p,
            max_iter=args.max_iter,
            seed=args.seed,
        )
        provenance["method_algorithm_called"] = True
        raw_points = np.asarray(raw_points, dtype=np.float32)
        if raw_points.ndim != 2 or raw_points.shape[1] != 4:
            raise ValueError(f"EAR raw output has invalid shape {raw_points.shape}; expected Nx4")
        if raw_points.shape[0] == 0:
            raise ValueError("EAR raw output is empty")
        if not np.isfinite(raw_points).all():
            raise ValueError("EAR raw output contains NaN or Inf")
        raw_points.tofile(raw_output)

        provenance["raw_output_point_count"] = int(raw_points.shape[0])
        provenance["raw_output_generated_from"] = "EAR_algorithm"
        provenance["status"] = "PASS"
        provenance["runtime_seconds"] = round(time.time() - start, 3)
        save_json(provenance_json, provenance)
        return 0
    except Exception as exc:  # noqa: BLE001
        provenance["status"] = "FAIL"
        provenance["raw_output_generated_from"] = provenance.get("raw_output_generated_from") or "NONE"
        provenance["notes"].append(str(exc))
        provenance["runtime_seconds"] = round(time.time() - start, 3)
        save_json(provenance_json, provenance)
        print(f"EAR raw wrapper failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
