#!/usr/bin/env python3
"""Run PU-Net TF1 patch inference for many frames with one checkpoint restore."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def pre_load_checkpoint(tf, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(str(checkpoint_dir))
    if ckpt and ckpt.model_checkpoint_path:
        return ckpt.model_checkpoint_path
    return None


def normalize_patch_unit_sphere(patch):
    center = patch.mean(axis=0, keepdims=True).astype(np.float32)
    centered = patch - center
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if not np.isfinite(scale) or scale < 1e-6:
        raise ValueError("PU-Net patch normalization scale is non-finite or degenerate")
    return (centered / scale).astype(np.float32), center, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code_dir", required=True)
    parser.add_argument("--jobs_json", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--tf_ops_repo_dir", default=None)
    parser.add_argument("--line", required=True)
    parser.add_argument("--patch_input_points", type=int, required=True)
    parser.add_argument("--patch_output_points", type=int, required=True)
    parser.add_argument("--up_ratio", type=int, default=4)
    parser.add_argument("--max_jobs", type=int, default=None)
    parser.add_argument(
        "--normalization_mode",
        choices=["legacy_none", "unit_sphere_v1"],
        default="legacy_none",
    )
    args = parser.parse_args()

    started = time.time()
    code_dir = Path(args.code_dir).resolve()
    tf_ops_repo_dir = Path(args.tf_ops_repo_dir).resolve() if args.tf_ops_repo_dir else None
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    if not jobs:
        raise ValueError("jobs_json contains no jobs")
    if not (checkpoint_dir / "checkpoint").exists():
        raise ValueError(f"checkpoint file missing in {checkpoint_dir}")

    sys.path.insert(0, str(code_dir))
    if tf_ops_repo_dir is not None:
        if not (tf_ops_repo_dir / "tf_ops").exists():
            raise ValueError(f"tf_ops_repo_dir does not contain tf_ops: {tf_ops_repo_dir}")
        sys.path.insert(0, str(tf_ops_repo_dir))

    import tensorflow as tf  # noqa: PLC0415

    if tf_ops_repo_dir is not None:
        import tf_ops  # noqa: PLC0415

        tf_ops.__path__ = [str(tf_ops_repo_dir / "tf_ops"), str(code_dir / "tf_ops")]
    import model_generator2_2new6 as model_gen  # noqa: PLC0415

    pointclouds_ipt = tf.placeholder(tf.float32, shape=(1, args.patch_input_points, 3))
    pred, _ = model_gen.get_gen_model(
        pointclouds_ipt,
        is_training=False,
        scope="generator",
        bradius=1.0,
        reuse=None,
        use_normal=False,
        use_bn=False,
        use_ibn=False,
        bn_decay=0.95,
        up_ratio=args.up_ratio,
    )
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    restore_model_path = pre_load_checkpoint(tf, checkpoint_dir)
    if restore_model_path is None:
        raise ValueError(f"no TensorFlow checkpoint state in {checkpoint_dir}")

    failures = 0
    with tf.Session(config=config) as sess:
        saver.restore(sess, restore_model_path)
        for job_index, job in enumerate(jobs, start=1):
            job_started = time.time()
            frame_id = str(job["frame_id"])
            patch_meta_path = Path(job["patch_metadata_json"]).resolve()
            raw_patch_dir = Path(job["raw_patch_output_dir"]).resolve()
            merged_path = Path(job["merged_raw_output"]).resolve()
            provenance_path = Path(job["merged_raw_provenance_json"]).resolve()
            raw_patch_dir.mkdir(parents=True, exist_ok=True)
            provenance = {
                "method": "PU-Net",
                "line": args.line,
                "frame_id": frame_id,
                "method_entrypoint": str(Path(__file__).resolve()),
                "checkpoint_path": str(checkpoint_dir),
                "checkpoint_loaded": True,
                "resolved_checkpoint_path": restore_model_path,
                "method_inference_called": False,
                "patch_protocol_id": None,
                "patch_selection": None,
                "normalization_mode": args.normalization_mode,
                "patch_input_points": int(args.patch_input_points),
                "patch_output_points": int(args.patch_output_points),
                "up_ratio": int(args.up_ratio),
                "raw_patch_output_dir": str(raw_patch_dir),
                "merged_raw_output_path": str(merged_path),
                "fallback_used": False,
                "generic_adapter_used_before_raw": False,
                "job_index": int(job_index),
                "job_count": int(len(jobs)),
                "status": "STARTED",
                "notes": [],
            }
            try:
                metadata = json.loads(patch_meta_path.read_text(encoding="utf-8"))
                provenance["patch_protocol_id"] = metadata.get(
                    "patch_protocol_id", f"{metadata.get('patch_selection', 'unknown')}_k{args.patch_input_points}"
                )
                provenance["patch_selection"] = metadata.get("patch_selection")
                patch_files = [Path(metadata["output_patch_dir"]) / p["file"] for p in metadata["patches"]]
                if not patch_files:
                    raise ValueError("no patches listed in metadata")
                raw_outputs = []
                normalization_records = []
                infer_started = time.time()
                for i, patch_file in enumerate(patch_files):
                    patch = np.load(patch_file).astype(np.float32)
                    if patch.shape != (args.patch_input_points, 3):
                        raise ValueError(f"{patch_file} shape {patch.shape} != ({args.patch_input_points}, 3)")
                    model_input = patch
                    center = None
                    scale = None
                    if args.normalization_mode == "unit_sphere_v1":
                        model_input, center, scale = normalize_patch_unit_sphere(patch)
                    out = sess.run(pred, feed_dict={pointclouds_ipt: model_input[None, :, :]})
                    out = np.asarray(out[0], dtype=np.float32).reshape(-1, 3)
                    if args.normalization_mode == "unit_sphere_v1":
                        out = (out * scale + center).astype(np.float32)
                        normalization_records.append(
                            {
                                "patch_id": int(i),
                                "center_xyz": center.reshape(3).astype(float).tolist(),
                                "scale": scale,
                            }
                        )
                    if out.shape != (args.patch_output_points, 3):
                        raise ValueError(f"{patch_file} prediction shape {out.shape} != ({args.patch_output_points}, 3)")
                    if not np.isfinite(out).all():
                        raise ValueError(f"{patch_file} prediction contains NaN or Inf")
                    np.save(raw_patch_dir / f"patch_{i:06d}.npy", out)
                    raw_outputs.append(out)
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
                        "runtime_total_method_runner_sec": float(time.time() - job_started),
                        "status": "PASS",
                        "normalization_records": normalization_records,
                    }
                )
                print(json.dumps({"status": "PASS", "frame_id": frame_id, "patch_count": len(raw_outputs)}, sort_keys=True), flush=True)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                provenance["status"] = "FAIL"
                provenance["notes"].append(str(exc))
                provenance["runtime_total_method_runner_sec"] = float(time.time() - job_started)
                print(f"tf_punet_patch_infer_many frame {frame_id} failed: {exc}", flush=True)
            save_json(provenance_path, provenance)
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
