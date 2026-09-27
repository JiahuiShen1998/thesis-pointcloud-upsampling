#!/usr/bin/env python3
"""Run PU-GCN-family TF1 patch inference on unified .npy patches."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--patch_metadata_json", required=True)
    parser.add_argument("--raw_patch_output_dir", required=True)
    parser.add_argument("--merged_raw_output", required=True)
    parser.add_argument("--merged_raw_provenance_json", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--line", required=True)
    parser.add_argument("--frame_id", required=True)
    parser.add_argument("--patch_input_points", type=int, required=True)
    parser.add_argument("--patch_output_points", type=int, required=True)
    parser.add_argument("--up_ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260630)
    args = parser.parse_args()

    started = time.time()
    repo_dir = Path(args.repo_dir).resolve()
    patch_meta_path = Path(args.patch_metadata_json).resolve()
    raw_patch_dir = Path(args.raw_patch_output_dir).resolve()
    merged_path = Path(args.merged_raw_output).resolve()
    provenance_path = Path(args.merged_raw_provenance_json).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    raw_patch_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "method": args.method,
        "line": args.line,
        "frame_id": args.frame_id,
        "method_entrypoint": str(Path(__file__).resolve()),
        "method_model_arg": args.model,
        "checkpoint_path": str(checkpoint_dir),
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
        metadata = json.loads(patch_meta_path.read_text(encoding="utf-8"))
        provenance["patch_protocol_id"] = metadata.get(
            "patch_protocol_id", f"{metadata.get('patch_selection', 'unknown')}_k{args.patch_input_points}"
        )
        provenance["patch_selection"] = metadata.get("patch_selection")
        patch_files = [Path(metadata["output_patch_dir"]) / p["file"] for p in metadata["patches"]]
        if not patch_files:
            raise ValueError("no patches listed in metadata")
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
        from Upsampling.configs import FLAGS  # noqa: PLC0415
        from Upsampling.model import Model  # noqa: PLC0415
        from Common import model_utils  # noqa: PLC0415
        from Common.model_utils import get_model_cls  # noqa: PLC0415

        np.random.seed(args.seed)
        tf.set_random_seed(args.seed)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        raw_outputs = []
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
            provenance["checkpoint_loaded"] = True
            provenance["resolved_checkpoint_path"] = checkpoint_path

            infer_started = time.time()
            for i, patch_file in enumerate(patch_files):
                patch = np.load(patch_file).astype(np.float32)
                if patch.shape != (args.patch_input_points, 3):
                    raise ValueError(f"{patch_file} shape {patch.shape} != ({args.patch_input_points}, 3)")
                pred = model.patch_prediction(patch)
                pred = np.asarray(pred, dtype=np.float32).reshape(-1, 3)
                if pred.shape != (args.patch_output_points, 3):
                    raise ValueError(f"{patch_file} prediction shape {pred.shape} != ({args.patch_output_points}, 3)")
                out_path = raw_patch_dir / f"patch_{i:06d}.npy"
                np.save(out_path, pred)
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
        print(f"tf_pugcn_family_patch_infer failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
