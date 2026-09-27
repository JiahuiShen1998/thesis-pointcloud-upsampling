#!/usr/bin/env python3
"""Batch strict x4 KITTI finalization for merged raw method outputs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strict_x4_from_merged_raw as strict_one  # noqa: E402


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_job(job: dict, up_ratio: float) -> dict:
    started = time.time()
    method = job["method"]
    line = job["line"]
    frame_id = str(job["frame_id"])
    input_path = Path(job["input_bin"]).resolve()
    merged_path = Path(job["merged_raw_output"]).resolve()
    merged_prov_path = Path(job["merged_raw_provenance_json"]).resolve()
    final_path = Path(job["final_output"]).resolve()
    provenance_path = Path(job["provenance_json"]).resolve()
    provenance = {
        "method": method,
        "line": line,
        "frame_id": frame_id,
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
        if merged_prov.get("method") != method:
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

        inp = strict_one.load_kitti_bin(input_path)
        raw_xyz, io_recovery_used, io_recovery_reason = strict_one.load_raw_xyz_verified(
            merged_path, merged_prov
        )
        target_points = int(round(inp.shape[0] * up_ratio))
        provenance.update(
            {
                "input_point_count": int(inp.shape[0]),
                "merged_raw_output_point_count": int(raw_xyz.shape[0]),
                "target_output_points": int(target_points),
                "raw_shortfall_points": int(max(0, target_points - raw_xyz.shape[0])),
                "merged_io_recovery_used": io_recovery_used,
                "merged_io_recovery_reason": io_recovery_reason,
                "merged_float32_sha256_verified": strict_one.float32_sha256(raw_xyz),
            }
        )
        strict_xyz = strict_one.choose_strict(raw_xyz, target_points, int(job["seed"]))
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
    except Exception as exc:  # noqa: BLE001
        provenance["status"] = "FAIL"
        provenance["notes"].append(str(exc))
        provenance["runtime_final_adapter_sec"] = float(time.time() - started)
    save_json(provenance_path, provenance)
    return provenance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs_json", required=True)
    parser.add_argument("--up_ratio", type=float, default=4.0)
    args = parser.parse_args()

    started = time.time()
    jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))
    failures = 0
    for index, job in enumerate(jobs, start=1):
        provenance = run_job(job, args.up_ratio)
        if provenance["status"] != "PASS":
            failures += 1
        print(
            json.dumps(
                {
                    "frame_id": provenance["frame_id"],
                    "index": index,
                    "jobs": len(jobs),
                    "status": provenance["status"],
                    "final_output_point_count": provenance.get("final_output_point_count"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "status": "PASS" if failures == 0 else "FAIL",
                "jobs": len(jobs),
                "failures": failures,
                "runtime_total_sec": time.time() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
