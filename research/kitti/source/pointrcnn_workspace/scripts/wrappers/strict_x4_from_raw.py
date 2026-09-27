#!/usr/bin/env python3
"""Normalize raw method output to strict x4 KITTI XYZI with nearest input intensity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def load_kitti_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(f"{path} is not KITTI XYZI: value count is not divisible by 4")
    return values.reshape(-1, 4)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def choose_strict(raw: np.ndarray, target_points: int, seed: int) -> np.ndarray:
    if raw.shape[0] == target_points:
        return raw[:, :3].copy()
    rng = np.random.default_rng(seed)
    if raw.shape[0] > target_points:
        idx = rng.choice(raw.shape[0], size=target_points, replace=False)
        return raw[idx, :3].copy()
    raise ValueError(
        "raw method output has fewer points than required: "
        f"raw={raw.shape[0]}, required={target_points}. "
        "Strict x4 output cannot be filled by copying input, duplicating raw points, "
        "or synthesizing fake points."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Final strict x4 adapter from raw method output")
    parser.add_argument("--raw_output", required=True)
    parser.add_argument("--input_bin", required=True)
    parser.add_argument("--final_output", required=True)
    parser.add_argument("--provenance_json", required=True)
    parser.add_argument("--raw_provenance_json", required=True)
    parser.add_argument("--method", default="EAR")
    parser.add_argument("--line", required=True)
    parser.add_argument("--frame_id", required=True)
    parser.add_argument("--up_ratio", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()

    raw_path = Path(args.raw_output).resolve()
    input_path = Path(args.input_bin).resolve()
    final_path = Path(args.final_output).resolve()
    provenance_path = Path(args.provenance_json).resolve()
    raw_provenance_path = Path(args.raw_provenance_json).resolve()

    provenance = {
        "method": args.method,
        "line": args.line,
        "frame_id": args.frame_id,
        "adapter_input_is_raw_method_output": False,
        "raw_output_path": str(raw_path),
        "raw_provenance_json": str(raw_provenance_path),
        "input_bin_for_intensity": str(input_path),
        "target_points": None,
        "final_output_points": None,
        "ratio": None,
        "intensity_policy": "nearest input intensity assignment",
        "frame_count_rule": "target_points = input_point_count * up_ratio; Line A input=N => 4N, Line B input=M=floor(N/4) => 4M",
        "patch_size_policy": "patch sizes such as 2048/8192 are internal inference settings only and do not define frame-level output counts",
        "status": "STARTED",
        "notes": [],
    }

    try:
        raw_prov = json.loads(raw_provenance_path.read_text(encoding="utf-8"))
        if raw_prov.get("method") != args.method:
            raise ValueError("raw provenance method mismatch")
        if raw_prov.get("raw_output_path") != str(raw_path):
            raise ValueError("raw provenance does not point to raw_output")
        if raw_prov.get("status") != "PASS":
            raise ValueError("raw provenance is not PASS")
        if raw_prov.get("method_algorithm_called") is not True:
            raise ValueError("raw provenance does not prove method algorithm call")
        if raw_prov.get("raw_output_generated_from") != "EAR_algorithm":
            raise ValueError("raw output was not generated from EAR_algorithm")
        if raw_prov.get("fallback_used") is not False:
            raise ValueError("raw provenance fallback_used is not false")
        if raw_prov.get("generic_adapter_used_before_raw") is not False:
            raise ValueError("raw provenance generic_adapter_used_before_raw is not false")
        provenance["adapter_input_is_raw_method_output"] = True

        raw = load_kitti_bin(raw_path)
        inp = load_kitti_bin(input_path)
        target_points = int(inp.shape[0] * args.up_ratio)
        provenance["input_points"] = int(inp.shape[0])
        provenance["raw_output_points"] = int(raw.shape[0])
        xyz = choose_strict(raw, target_points, args.seed)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs.fit(inp[:, :3])
        nearest = nbrs.kneighbors(xyz, return_distance=False).reshape(-1)
        final = np.hstack([xyz, inp[nearest, 3:4]]).astype(np.float32)
        if final.shape != (target_points, 4):
            raise ValueError(f"final output shape {final.shape} is not ({target_points}, 4)")
        if not np.isfinite(final).all():
            raise ValueError("final output contains NaN or Inf")

        final_path.parent.mkdir(parents=True, exist_ok=True)
        final.tofile(final_path)
        provenance["target_points"] = target_points
        provenance["raw_shortfall_points"] = 0
        provenance["final_output_points"] = int(final.shape[0])
        provenance["ratio"] = float(final.shape[0] / inp.shape[0])
        provenance["status"] = "PASS"
        save_json(provenance_path, provenance)
        return 0
    except Exception as exc:  # noqa: BLE001
        provenance["status"] = "FAIL"
        provenance["notes"].append(str(exc))
        if provenance.get("target_points") is None:
            try:
                provenance["target_points"] = int(load_kitti_bin(input_path).shape[0] * args.up_ratio)
            except Exception:
                pass
        if "raw_output_points" in provenance and provenance.get("target_points") is not None:
            provenance["raw_shortfall_points"] = int(max(0, provenance["target_points"] - provenance["raw_output_points"]))
        save_json(provenance_path, provenance)
        print(f"strict adapter failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
