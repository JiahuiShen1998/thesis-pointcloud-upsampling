#!/usr/bin/env python3
"""Run PDANS patch inference on unified .npy patches."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def query_free_gpu_mib() -> int:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(proc.stdout.strip().splitlines()[0].strip())
    except Exception:
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            return free // (1024 * 1024)
        return 0


def wait_for_gpu_memory(min_free_mib: int, timeout_sec: int) -> None:
    if not torch.cuda.is_available():
        return
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        free_mib = query_free_gpu_mib()
        if free_mib >= min_free_mib:
            return
        time.sleep(15)
    raise RuntimeError(
        f"insufficient free GPU memory after {timeout_sec}s wait: free={query_free_gpu_mib()}MiB required>={min_free_mib}MiB"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch_metadata_json", required=True)
    parser.add_argument("--raw_patch_output_dir", required=True)
    parser.add_argument("--merged_raw_output", required=True)
    parser.add_argument("--merged_raw_provenance_json", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--line", required=True)
    parser.add_argument("--frame_id", required=True)
    parser.add_argument("--patch_input_points", type=int, required=True)
    parser.add_argument("--patch_output_points", type=int, required=True)
    parser.add_argument("--up_ratio", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--step", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_free_gpu_mib", type=int, default=3000)
    parser.add_argument("--gpu_wait_timeout_sec", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=20260630)
    args = parser.parse_args()

    started = time.time()
    patch_meta_path = Path(args.patch_metadata_json).resolve()
    raw_patch_dir = Path(args.raw_patch_output_dir).resolve()
    merged_path = Path(args.merged_raw_output).resolve()
    provenance_path = Path(args.merged_raw_provenance_json).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()
    raw_patch_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "method": "PDANS",
        "line": args.line,
        "frame_id": args.frame_id,
        "method_entrypoint": str(Path(__file__).resolve()),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "checkpoint_loaded": False,
        "method_inference_called": False,
        "patch_protocol_id": None,
        "patch_selection": None,
        "patch_input_points": int(args.patch_input_points),
        "patch_output_points": int(args.patch_output_points),
        "up_ratio": int(args.up_ratio),
        "raw_patch_output_dir": str(raw_patch_dir),
        "merged_raw_output_path": str(merged_path),
        "fallback_used": False,
        "generic_adapter_used_before_raw": False,
        "status": "STARTED",
        "notes": [],
    }
    try:
        if not checkpoint_path.exists():
            raise ValueError(f"checkpoint missing: {checkpoint_path}")
        if not config_path.exists():
            raise ValueError(f"config missing: {config_path}")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if args.device == "cuda":
            wait_for_gpu_memory(args.min_free_gpu_mib, args.gpu_wait_timeout_sec)
        if args.patch_output_points != args.patch_input_points * args.up_ratio:
            raise ValueError("patch_output_points must equal patch_input_points * up_ratio")

        metadata = json.loads(patch_meta_path.read_text(encoding="utf-8"))
        provenance["patch_protocol_id"] = metadata.get(
            "patch_protocol_id", f"{metadata.get('patch_selection', 'unknown')}_k{args.patch_input_points}"
        )
        provenance["patch_selection"] = metadata.get("patch_selection")
        affected_tail_patches = [
            patch for patch in metadata["patches"] if patch.get("tail_support_rule") is not None
        ]
        if affected_tail_patches:
            provenance["tail_support_rule"] = "degenerate_tail_neighbor_support_v1"
            provenance["tail_support_affected_patches"] = affected_tail_patches
            provenance["no_final_output_padding"] = True
        patch_files = [Path(metadata["output_patch_dir"]) / p["file"] for p in metadata["patches"]]
        if not patch_files:
            raise ValueError("no patches listed in metadata")

        sys.path.insert(0, str(SCRIPTS_DIR))
        from pdans_kitti_adapter import load_pdans, normalize_patch  # noqa: PLC0415

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            torch.cuda.empty_cache()
            gc.collect()

        net, diffusion, config, sampling_ddim, checkpoint = load_pdans(config_path, checkpoint_path, device)
        provenance["checkpoint_loaded"] = True
        provenance["checkpoint_epoch"] = checkpoint.get("epoch")
        provenance["checkpoint_iter"] = checkpoint.get("iter")
        provenance["config_dataset"] = config.get("train_config", {}).get("dataset")

        raw_outputs: list[np.ndarray] = []
        infer_started = time.time()
        label_full = torch.full((args.batch_size,), fill_value=args.up_ratio - 1, dtype=torch.long, device=device)
        for start_idx in range(0, len(patch_files), args.batch_size):
            batch_files = patch_files[start_idx : start_idx + args.batch_size]
            normed: list[np.ndarray] = []
            centers: list[np.ndarray] = []
            scales: list[float] = []
            for patch_file in batch_files:
                patch = np.load(patch_file).astype(np.float32)
                if patch.shape != (args.patch_input_points, 3):
                    raise ValueError(f"{patch_file} shape {patch.shape} != ({args.patch_input_points}, 3)")
                patch_norm, center, scale = normalize_patch(patch)
                normed.append(patch_norm)
                centers.append(center)
                scales.append(scale)

            last_error: Exception | None = None
            dense_np = None
            for attempt in range(5):
                try:
                    if device.type == "cuda":
                        wait_for_gpu_memory(args.min_free_gpu_mib, args.gpu_wait_timeout_sec)
                    condition = torch.from_numpy(np.stack(normed, axis=0)).to(device=device, dtype=torch.float32)
                    net.reset_cond_features()
                    with torch.no_grad():
                        dense, _, _ = sampling_ddim(
                            net=net,
                            size=(condition.shape[0], args.patch_output_points, 3),
                            diffusion_hyperparams=diffusion,
                            label=label_full[: condition.shape[0]],
                            condition=condition,
                            R=args.up_ratio,
                            gamma=args.gamma,
                            step=args.step,
                        )
                    dense_np = dense.detach().cpu().numpy()
                    del dense, condition
                    last_error = None
                    break
                except RuntimeError as exc:
                    last_error = exc
                    if device.type == "cuda" and "out of memory" in str(exc).lower() and attempt < 4:
                        if "condition" in locals():
                            del condition
                        if "dense" in locals():
                            del dense
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(30)
                        continue
                    raise
            if dense_np is None:
                raise RuntimeError(f"PDANS patch inference failed after retries: {last_error}")
            for local_idx in range(dense_np.shape[0]):
                out = (dense_np[local_idx] * scales[local_idx] + centers[local_idx]).astype(np.float32)
                if not np.isfinite(out).all():
                    raise ValueError(f"PDANS patch output contains NaN or Inf: patch_{start_idx + local_idx:06d}")
                if out.shape != (args.patch_output_points, 3):
                    raise ValueError(f"PDANS prediction shape {out.shape} != ({args.patch_output_points}, 3)")
                out_path = raw_patch_dir / f"patch_{start_idx + local_idx:06d}.npy"
                np.save(out_path, out)
                raw_outputs.append(out)
            del dense_np
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()
        provenance["runtime_method_inference_sec"] = float(time.time() - infer_started)
        provenance["method_inference_called"] = True

        merge_started = time.time()
        merged = np.concatenate(raw_outputs, axis=0).astype(np.float32)
        merged_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(merged_path, merged)
        provenance.update(
            {
                "patch_count": int(len(raw_outputs)),
                "raw_patch_output_count_each": int(args.patch_output_points),
                "merged_raw_output_point_count": int(merged.shape[0]),
                "merged_float32_sha256": hashlib.sha256(
                    np.ascontiguousarray(merged).tobytes()
                ).hexdigest(),
                "runtime_merge_sec": float(time.time() - merge_started),
                "runtime_total_method_runner_sec": float(time.time() - started),
                "status": "PASS",
            }
        )
        save_json(provenance_path, provenance)
        print(json.dumps({"status": "PASS", "patch_count": len(raw_outputs), "merged_raw_output_point_count": int(merged.shape[0])}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        provenance["status"] = "FAIL"
        provenance["notes"].append(str(exc))
        provenance["runtime_total_method_runner_sec"] = float(time.time() - started)
        save_json(provenance_path, provenance)
        print(f"pdans_patch_infer failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
