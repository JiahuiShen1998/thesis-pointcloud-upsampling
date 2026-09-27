#!/usr/bin/env python3
"""Normalize merged raw method output to strict x4 KITTI XYZI."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def load_kitti_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(f"{path} is not KITTI XYZI: value count is not divisible by 4")
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN or Inf")
    return points


def load_raw_xyz(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        raw = np.load(path)
    else:
        values = np.fromfile(path, dtype=np.float32)
        if values.size % 3 == 0:
            raw = values.reshape(-1, 3)
        elif values.size % 4 == 0:
            raw = values.reshape(-1, 4)[:, :3]
        else:
            raise ValueError(f"{path} cannot be interpreted as float32 XYZ or XYZI")
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(f"{path} raw output shape must be Nx3 or NxC with C>=3, got {raw.shape}")
    raw = raw[:, :3]
    if raw.shape[0] == 0:
        raise ValueError("merged raw output is empty")
    if not np.isfinite(raw).all():
        raise ValueError("merged raw output contains NaN or Inf")
    return raw


def float32_sha256(points: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(points, dtype=np.float32).tobytes()).hexdigest()


def raw_xyz_from_payload(path: Path, payload: bytes) -> np.ndarray:
    if path.suffix == ".npy":
        raw = np.load(io.BytesIO(payload))
    else:
        values = np.frombuffer(payload, dtype=np.float32)
        if values.size % 3 == 0:
            raw = values.reshape(-1, 3)
        elif values.size % 4 == 0:
            raw = values.reshape(-1, 4)[:, :3]
        else:
            raise ValueError(f"{path} cannot be interpreted as float32 XYZ or XYZI")
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] < 3 or raw.shape[0] == 0:
        raise ValueError(f"{path} raw output has invalid shape {raw.shape}")
    raw = raw[:, :3]
    if not np.isfinite(raw).all():
        raise ValueError("merged raw output contains NaN or Inf")
    return raw


def reconstruct_from_raw_patches(raw_patch_dir: Path) -> np.ndarray:
    patch_files = sorted(raw_patch_dir.glob("patch_*.npy"))
    if not patch_files:
        raise ValueError(f"no method raw patches found in {raw_patch_dir}")
    arrays = []
    for path in patch_files:
        first = path.read_bytes()
        second = path.read_bytes()
        if first != second:
            raise ValueError(f"method raw patch is not stable across reads: {path}")
        array = raw_xyz_from_payload(path, first)
        arrays.append(array)
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def load_raw_xyz_verified(path: Path, merged_provenance: dict) -> tuple[np.ndarray, bool, str]:
    """Load a stable merged array or losslessly rebuild it from method raw patches."""
    expected = merged_provenance.get("merged_float32_sha256")
    first = path.read_bytes()
    second = path.read_bytes()
    failure_reason = ""
    if first == second:
        try:
            raw = raw_xyz_from_payload(path, first)
            digest = float32_sha256(raw)
            if expected is None or digest == expected:
                return raw, False, "stable_merged_raw"
            failure_reason = f"merged checksum mismatch: {digest} != {expected}"
        except (OSError, ValueError) as error:
            failure_reason = str(error)
    else:
        failure_reason = "merged raw file is not byte-stable across consecutive reads"

    raw_patch_dir_value = merged_provenance.get("raw_patch_output_dir")
    if not raw_patch_dir_value:
        raise ValueError(f"{failure_reason}; no raw_patch_output_dir is available")
    reconstructed = reconstruct_from_raw_patches(Path(raw_patch_dir_value).resolve())
    digest = float32_sha256(reconstructed)
    if expected is not None and digest != expected:
        raise ValueError(
            f"{failure_reason}; reconstructed raw checksum mismatch: {digest} != {expected}"
        )
    return reconstructed, True, failure_reason


def choose_strict(raw_xyz: np.ndarray, target_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if raw_xyz.shape[0] == target_points:
        return raw_xyz.copy()
    if raw_xyz.shape[0] > target_points:
        idx = rng.choice(raw_xyz.shape[0], size=target_points, replace=False)
        return raw_xyz[idx].copy()
    raise ValueError(
        "merged raw method output has fewer points than required: "
        f"raw={raw_xyz.shape[0]}, required={target_points}. "
        "Strict x4 output cannot be filled by copying input, duplicating raw points, "
        "or synthesizing fake points."
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged_raw_output", required=True)
    parser.add_argument("--merged_raw_provenance_json", required=True)
    parser.add_argument("--input_bin", required=True)
    parser.add_argument("--final_output", required=True)
    parser.add_argument("--provenance_json", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--line", required=True)
    parser.add_argument("--frame_id", required=True)
    parser.add_argument("--up_ratio", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260630)
    args = parser.parse_args()

    started = time.time()
    merged_path = Path(args.merged_raw_output).resolve()
    merged_prov_path = Path(args.merged_raw_provenance_json).resolve()
    input_path = Path(args.input_bin).resolve()
    final_path = Path(args.final_output).resolve()
    provenance_path = Path(args.provenance_json).resolve()
    provenance = {
        "method": args.method,
        "line": args.line,
        "frame_id": args.frame_id,
        "input_bin": str(input_path),
        "merged_raw_output_path": str(merged_path),
        "merged_raw_provenance_json": str(merged_prov_path),
        "final_output_path": str(final_path),
        "final_adapter_input_is_merged_raw_output": False,
        "generic_adapter_used_before_raw": False,
        "fallback_used": False,
        "intensity_policy": "nearest input intensity assignment",
        "frame_count_rule": "target_output_points = input_point_count * up_ratio; Line A input=N => 4N, Line B input=M=floor(N/4) => 4M",
        "patch_size_policy": "patch sizes such as 2048/8192 are internal inference settings only and do not define frame-level output counts",
        "ratio": None,
        "status": "STARTED",
        "notes": [],
    }

    try:
        merged_prov = json.loads(merged_prov_path.read_text(encoding="utf-8"))
        if merged_prov.get("method") != args.method:
            raise ValueError("merged raw provenance method mismatch")
        if Path(merged_prov.get("merged_raw_output_path", "")).resolve() != merged_path:
            raise ValueError("merged raw provenance does not point to merged_raw_output")
        if merged_prov.get("status") != "PASS":
            raise ValueError("merged raw provenance status is not PASS")
        if merged_prov.get("method_inference_called") is not True:
            raise ValueError("method_inference_called is not true")
        if merged_prov.get("checkpoint_loaded") is not True:
            raise ValueError("checkpoint_loaded is not true")
        if merged_prov.get("fallback_used") is not False:
            raise ValueError("fallback_used is not false")
        if merged_prov.get("generic_adapter_used_before_raw") is not False:
            raise ValueError("generic_adapter_used_before_raw is not false")

        inp = load_kitti_bin(input_path)
        raw_xyz, io_recovery_used, io_recovery_reason = load_raw_xyz_verified(
            merged_path, merged_prov
        )
        target_points = int(round(inp.shape[0] * args.up_ratio))
        provenance.update(
            {
                "input_point_count": int(inp.shape[0]),
                "merged_raw_output_point_count": int(raw_xyz.shape[0]),
                "target_output_points": int(target_points),
                "raw_shortfall_points": int(max(0, target_points - raw_xyz.shape[0])),
                "merged_io_recovery_used": io_recovery_used,
                "merged_io_recovery_reason": io_recovery_reason,
                "merged_float32_sha256_verified": float32_sha256(raw_xyz),
            }
        )
        strict_xyz = choose_strict(raw_xyz, target_points, args.seed)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nbrs.fit(inp[:, :3])
        nearest = nbrs.kneighbors(strict_xyz, return_distance=False).reshape(-1)
        final = np.hstack([strict_xyz, inp[nearest, 3:4]]).astype(np.float32)
        if final.shape != (target_points, 4):
            raise ValueError(f"final shape {final.shape} is not ({target_points}, 4)")
        if not np.isfinite(final).all():
            raise ValueError("final output contains NaN or Inf")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final.tofile(final_path)

        provenance.update(
            {
                "input_point_count": int(inp.shape[0]),
                "merged_raw_output_point_count": int(raw_xyz.shape[0]),
                "target_output_points": int(target_points),
                "raw_shortfall_points": 0,
                "final_output_point_count": int(final.shape[0]),
                "ratio": float(final.shape[0] / inp.shape[0]),
                "final_adapter_input_is_merged_raw_output": True,
                "runtime_final_adapter_sec": float(time.time() - started),
                "status": "PASS",
            }
        )
        save_json(provenance_path, provenance)
        print(json.dumps({k: provenance[k] for k in ("status", "final_output_point_count", "ratio", "runtime_final_adapter_sec")}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        provenance["status"] = "FAIL"
        provenance["notes"].append(str(exc))
        provenance["runtime_final_adapter_sec"] = float(time.time() - started)
        save_json(provenance_path, provenance)
        print(f"strict_x4_from_merged_raw failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
