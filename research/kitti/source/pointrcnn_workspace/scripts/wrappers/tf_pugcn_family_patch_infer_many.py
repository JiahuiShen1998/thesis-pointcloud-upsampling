#!/usr/bin/env python3
"""Run PU-GCN-family TF1 patch inference for many frames with one restore."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--jobs_json", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--line", required=True)
    parser.add_argument("--patch_input_points", type=int, required=True)
    parser.add_argument("--patch_output_points", type=int, required=True)
    parser.add_argument("--up_ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--max_jobs", type=int, default=None)
    parser.add_argument("--gpu_memory_fraction", type=float, default=0.0)
    args = parser.parse_args()

    started = time.time()
    repo_dir = Path(args.repo_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))
    if args.max_jobs is not None:
        jobs = jobs[: args.max_jobs]
    if not jobs:
        raise ValueError("jobs_json contains no jobs")
    if not (checkpoint_dir / "checkpoint").exists():
        raise ValueError(f"checkpoint file missing in {checkpoint_dir}")

    sys.path.insert(0, str(repo_dir))
    sys.argv = [
        sys.argv[0],
        "--phase=test",
        f"--model={args.model}",
        f"--log_dir={checkpoint_dir}",
        f"--patch_num_point={args.patch_input_points}",
        f"--num_point={args.patch_input_points}",
        f"--up_ratio={args.up_ratio}",
        "--batch_size=1",
        f"--seed={args.seed}",
    ]

    import tensorflow as tf  # noqa: PLC0415
    from Common import model_utils  # noqa: PLC0415
    from Common.model_utils import get_model_cls  # noqa: PLC0415
    from Upsampling.configs import FLAGS  # noqa: PLC0415
    from Upsampling.model import Model  # noqa: PLC0415

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if args.gpu_memory_fraction > 0:
        config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction

    failures = 0
    with tf.Session(config=config) as sess:
        model = Model(FLAGS, sess)
        model.inputs = tf.placeholder(tf.float32, shape=[1, args.patch_input_points, 3])
        is_training = tf.placeholder_with_default(False, shape=[], name="is_training")
        model.pc_radius = tf.ones(FLAGS.batch_size)
        model_cls = get_model_cls(FLAGS.model)
        gen = model_cls(FLAGS, is_training, name="generator")
        if FLAGS.model == "punet":
            model.pred_pc = gen(model.inputs, model.pc_radius)
        else:
            model.pred_pc = gen(model.inputs)
        saver = tf.train.Saver()
        _, checkpoint_path = model_utils.pre_load_checkpoint(str(checkpoint_dir))
        saver.restore(sess, checkpoint_path)

        for job_index, job in enumerate(jobs, start=1):
            job_started = time.time()
            frame_id = str(job["frame_id"])
            patch_meta_path = Path(job["patch_metadata_json"]).resolve()
            raw_patch_dir = Path(job["raw_patch_output_dir"]).resolve()
            merged_path = Path(job["merged_raw_output"]).resolve()
            provenance_path = Path(job["merged_raw_provenance_json"]).resolve()
            raw_patch_dir.mkdir(parents=True, exist_ok=True)
            provenance = {
                "method": args.method,
                "line": args.line,
                "frame_id": frame_id,
                "method_entrypoint": str(Path(__file__).resolve()),
                "method_model_arg": args.model,
                "checkpoint_path": str(checkpoint_dir),
                "checkpoint_loaded": True,
                "resolved_checkpoint_path": checkpoint_path,
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
                infer_started = time.time()
                for patch_index, patch_file in enumerate(patch_files):
                    patch = np.load(patch_file).astype(np.float32)
                    if patch.shape != (args.patch_input_points, 3):
                        raise ValueError(f"{patch_file} shape {patch.shape} != ({args.patch_input_points}, 3)")
                    pred = model.patch_prediction(patch)
                    pred = np.asarray(pred, dtype=np.float32).reshape(-1, 3)
                    if pred.shape != (args.patch_output_points, 3):
                        raise ValueError(f"{patch_file} prediction shape {pred.shape} != ({args.patch_output_points}, 3)")
                    if not np.isfinite(pred).all():
                        raise ValueError(f"{patch_file} prediction contains NaN or Inf")
                    np.save(raw_patch_dir / f"patch_{patch_index:06d}.npy", pred)
                    raw_outputs.append(pred)
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
                    }
                )
                print(json.dumps({"status": "PASS", "frame_id": frame_id, "patch_count": len(raw_outputs)}, sort_keys=True), flush=True)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                provenance["status"] = "FAIL"
                provenance["notes"].append(str(exc))
                provenance["runtime_total_method_runner_sec"] = float(time.time() - job_started)
                print(f"tf_pugcn_family_patch_infer_many frame {frame_id} failed: {exc}", flush=True)
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
