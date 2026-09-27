#!/usr/bin/env python3
"""Run PU-Net TF1 patch inference on unified .npy patches."""

import argparse
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
    parser.add_argument("--patch_metadata_json", required=True)
    parser.add_argument("--raw_patch_output_dir", required=True)
    parser.add_argument("--merged_raw_output", required=True)
    parser.add_argument("--merged_raw_provenance_json", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument(
        "--tf_ops_repo_dir",
        default=None,
        help="Optional repo root whose tf_ops package should shadow PU-Net's local ops.",
    )
    parser.add_argument("--line", required=True)
    parser.add_argument("--frame_id", required=True)
    parser.add_argument("--patch_input_points", type=int, required=True)
    parser.add_argument("--patch_output_points", type=int, required=True)
    parser.add_argument("--up_ratio", type=int, default=4)
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
    patch_meta_path = Path(args.patch_metadata_json).resolve()
    raw_patch_dir = Path(args.raw_patch_output_dir).resolve()
    merged_path = Path(args.merged_raw_output).resolve()
    provenance_path = Path(args.merged_raw_provenance_json).resolve()
    raw_patch_dir.mkdir(parents=True, exist_ok=True)
    provenance = {
        "method": "PU-Net",
        "line": args.line,
        "frame_id": args.frame_id,
        "method_entrypoint": str(Path(__file__).resolve()),
        "checkpoint_path": str(checkpoint_dir),
        "tf_ops_repo_dir": str(tf_ops_repo_dir) if tf_ops_repo_dir else "",
        "checkpoint_loaded": False,
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
        if not (checkpoint_dir / "checkpoint").exists():
            raise ValueError("checkpoint file missing in {}".format(checkpoint_dir))
        sys.path.insert(0, str(code_dir))
        if tf_ops_repo_dir is not None and not (tf_ops_repo_dir / "tf_ops").exists():
            raise ValueError("tf_ops_repo_dir does not contain tf_ops: {}".format(tf_ops_repo_dir))
        if tf_ops_repo_dir is not None:
            sys.path.insert(0, str(tf_ops_repo_dir))
        import tensorflow as tf  # noqa: PLC0415
        if tf_ops_repo_dir is not None:
            import tf_ops  # noqa: PLC0415

            # Keep PU-Net-only subpackages such as emd/CD visible, while using
            # the TF1.13-compatible GPU sampling/grouping ops from PU-GCN.
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
            raise ValueError("no TensorFlow checkpoint state in {}".format(checkpoint_dir))
        raw_outputs = []
        normalization_records = []
        with tf.Session(config=config) as sess:
            saver.restore(sess, restore_model_path)
            provenance["checkpoint_loaded"] = True
            provenance["resolved_checkpoint_path"] = restore_model_path
            infer_started = time.time()
            for i, patch_file in enumerate(patch_files):
                patch = np.load(patch_file).astype(np.float32)
                if patch.shape != (args.patch_input_points, 3):
                    raise ValueError("{} shape {} != ({}, 3)".format(patch_file, patch.shape, args.patch_input_points))
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
                        {"patch_id": int(i), "center_xyz": center.reshape(3).astype(float).tolist(), "scale": scale}
                    )
                if out.shape != (args.patch_output_points, 3):
                    raise ValueError("{} prediction shape {} != ({}, 3)".format(patch_file, out.shape, args.patch_output_points))
                np.save(raw_patch_dir / "patch_{:06d}.npy".format(i), out)
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
                "runtime_merge_sec": float(time.time() - merge_started),
                "runtime_total_method_runner_sec": float(time.time() - started),
                "status": "PASS",
                "normalization_records": normalization_records,
            }
        )
        save_json(provenance_path, provenance)
        print(json.dumps({"status": "PASS", "patch_count": len(raw_outputs), "merged_raw_output_point_count": int(merged.shape[0])}, sort_keys=True))
        return 0
    except Exception as exc:
        provenance["status"] = "FAIL"
        provenance["notes"].append(str(exc))
        provenance["runtime_total_method_runner_sec"] = float(time.time() - started)
        save_json(provenance_path, provenance)
        print("tf_punet_patch_infer failed: {}".format(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
