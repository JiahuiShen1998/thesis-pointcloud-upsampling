#!/usr/bin/env python3
"""Run PU-GCN x4 upsampling for ModelNet40 two-line protocol (Line A / Line B)."""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pugcn_paths import (  # noqa: E402
    DEFAULT_PUGCN_CKPT,
    DEFAULT_PUGCN_REPO,
    DEFAULT_SEED,
    line_paths,
    resolve_project_root,
)
from pugcn_strict import audit_pass, normalize_strict  # noqa: E402


def iter_samples(input_root):
    for split in ("train", "test"):
        split_dir = input_root / split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for npy in sorted(class_dir.glob("*.npy")):
                rel = npy.relative_to(input_root)
                sample_id = "/".join(rel.with_suffix("").parts)
                yield sample_id, npy


def rel_output_path(input_root, input_path):
    return input_path.relative_to(input_root)


class PUGCNInferenceEngine:
    """TensorFlow 1.x PU-GCN whole-object inference via patch merge."""

    def __init__(
        self,
        repo_dir,
        checkpoint_dir,
        up_ratio=4,
        patch_num_point=256,
        patch_num_ratio=3,
        seed=DEFAULT_SEED,
    ):
        self.repo_dir = repo_dir.resolve()
        self.checkpoint_dir = checkpoint_dir.resolve()
        self.up_ratio = int(up_ratio)
        self.patch_num_point = int(patch_num_point)
        self.patch_num_ratio = int(patch_num_ratio)
        self.seed = int(seed)
        self._sess = None
        self._model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not (self.checkpoint_dir / "checkpoint").exists():
            raise FileNotFoundError(f"PU-GCN checkpoint missing: {self.checkpoint_dir}")

        sys.path.insert(0, str(self.repo_dir))
        os.chdir(str(self.repo_dir))
        sys.argv = [
            "run_pugcn_modelnet40_x4.py",
            "--phase=test",
            "--model=pugcn",
            f"--log_dir={self.checkpoint_dir}",
            f"--patch_num_point={self.patch_num_point}",
            f"--num_point={self.patch_num_point}",
            f"--up_ratio={self.up_ratio}",
            "--batch_size=1",
            f"--seed={self.seed}",
        ]
        import tensorflow as tf  # noqa: PLC0415
        from Common import model_utils  # noqa: PLC0415
        from Common.model_utils import get_model_cls  # noqa: PLC0415
        from Upsampling.configs import FLAGS  # noqa: PLC0415
        from Upsampling.model import Model  # noqa: PLC0415

        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        self._model = Model(FLAGS, self._sess)
        self._model.inputs = tf.placeholder(tf.float32, shape=[1, self.patch_num_point, 3])
        is_training = tf.placeholder_with_default(False, shape=[], name="is_training")
        self._model.pc_radius = tf.ones(FLAGS.batch_size)
        model_cls = get_model_cls(FLAGS.model)
        gen = model_cls(FLAGS, is_training, name="generator")
        self._model.pred_pc = gen(self._model.inputs)
        saver = tf.train.Saver()
        _, checkpoint_path = model_utils.pre_load_checkpoint(str(self.checkpoint_dir))
        saver.restore(self._sess, checkpoint_path)
        self._checkpoint_path = checkpoint_path
        self._loaded = True

    def infer(self, points_xyz):
        if not self._loaded:
            self.load()
        from Common import pc_util  # noqa: PLC0415
        from tf_ops.sampling.tf_sampling import farthest_point_sample  # noqa: PLC0415

        pc = np.asarray(points_xyz[:, :3], dtype=np.float32)
        with self._sess.as_default():
            pc_norm, centroid, furthest_distance = pc_util.normalize_point_cloud(pc)
            input_list, pred_list, _ = self._model.pc_prediction(pc_norm)
            if not pred_list:
                raise RuntimeError("PU-GCN returned no patch predictions")
            pred_pc = np.concatenate(pred_list, axis=0)
            pred_pc = (pred_pc * furthest_distance) + centroid
            pred_pc = np.reshape(pred_pc, (-1, 3)).astype(np.float32)
            out_point_num = int(pc.shape[0] * self.up_ratio)
            if pred_pc.shape[0] >= out_point_num:
                idx = farthest_point_sample(out_point_num, pred_pc[np.newaxis, ...]).eval(session=self._sess)[0]
                pred_pc = pred_pc[idx, :3]
        return pred_pc.astype(np.float32)


def write_failed_row(
    failed_csv,
    fieldnames,
    row,
    write_header,
):
    failed_csv.parent.mkdir(parents=True, exist_ok=True)
    with failed_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", required=True, choices=["lineA", "lineB"])
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--raw-output-root", default=None)
    parser.add_argument("--strict-output-root", default=None)
    parser.add_argument("--expected-input-points", type=int, default=None)
    parser.add_argument("--expected-output-points", type=int, default=None)
    parser.add_argument("--repo-dir", default=str(DEFAULT_PUGCN_REPO))
    parser.add_argument("--checkpoint", default=str(DEFAULT_PUGCN_CKPT))
    parser.add_argument("--up-ratio", type=int, default=4)
    parser.add_argument("--patch-num-point", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1, help="reserved; TF graph uses batch_size=1")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--failed-csv", default=None)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    project_root = resolve_project_root(Path(args.project_root) if args.project_root else None)
    defaults = line_paths(args.line, project_root)
    input_root = Path(args.input_root) if args.input_root else defaults["input_root"]
    raw_root = Path(args.raw_output_root) if args.raw_output_root else defaults["raw_output_root"]
    strict_root = Path(args.strict_output_root) if args.strict_output_root else defaults["strict_output_root"]
    expected_in = int(args.expected_input_points or defaults["expected_input_points"])
    expected_out = int(args.expected_output_points or defaults["expected_output_points"])
    failed_csv = Path(args.failed_csv) if args.failed_csv else (
        project_root / "reports" / f"modelnet40_pugcn_{args.line}_failed.csv"
    )

    if not input_root.is_dir():
        print(f"ERROR: input_root missing: {input_root}")
        return 2

    samples = list(iter_samples(input_root))
    if args.max_samples and args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        print(f"ERROR: no .npy samples under {input_root}")
        return 2

    failed_fields = [
        "line",
        "sample",
        "input_path",
        "error_message",
        "runtime_sec",
    ]
    failed_header_needed = not failed_csv.exists() or failed_csv.stat().st_size == 0

    engine = PUGCNInferenceEngine(
        repo_dir=Path(args.repo_dir),
        checkpoint_dir=Path(args.checkpoint),
        up_ratio=args.up_ratio,
        patch_num_point=args.patch_num_point,
        seed=args.seed,
    )
    engine.load()
    print(
        json.dumps(
            {
                "status": "engine_ready",
                "line": args.line,
                "project_root": str(project_root),
                "input_root": str(input_root),
                "raw_output_root": str(raw_root),
                "strict_output_root": str(strict_root),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "checkpoint_loaded_path": engine._checkpoint_path,
                "sample_count": len(samples),
                "expected_input_points": expected_in,
                "expected_output_points": expected_out,
                "seed": args.seed,
            },
            indent=2,
        )
    )

    processed = skipped = failed = 0
    started_all = time.time()
    for idx, (sample_id, input_path) in enumerate(samples, start=1):
        rel = rel_output_path(input_root, input_path)
        raw_path = raw_root / rel
        strict_path = strict_root / rel
        if args.resume and audit_pass(strict_path, expected_out):
            skipped += 1
            continue
        t0 = time.time()
        try:
            inp = np.load(input_path)
            inp = np.asarray(inp, dtype=np.float32)
            if inp.ndim != 2 or inp.shape[1] < 3:
                raise ValueError(f"input shape invalid: {inp.shape}")
            inp_xyz = inp[:, :3]
            if inp_xyz.shape[0] != expected_in:
                raise ValueError(f"input shape {inp_xyz.shape} != ({expected_in}, 3)")
            if not np.isfinite(inp_xyz).all():
                raise ValueError("input contains NaN or Inf")

            raw_xyz = engine.infer(inp_xyz)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(raw_path, raw_xyz)

            strict_xyz = normalize_strict(raw_xyz, expected_out, args.seed)
            strict_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(strict_path, strict_xyz)
            processed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            write_failed_row(
                failed_csv,
                failed_fields,
                {
                    "line": args.line,
                    "sample": sample_id,
                    "input_path": str(input_path),
                    "error_message": str(exc),
                    "runtime_sec": round(time.time() - t0, 4),
                },
                failed_header_needed,
            )
            failed_header_needed = False
            print(f"FAIL {sample_id}: {exc}")

        if idx % max(1, args.progress_every) == 0 or idx == len(samples):
            elapsed = time.time() - started_all
            print(
                f"[{args.line}] {idx}/{len(samples)} processed={processed} skipped={skipped} "
                f"failed={failed} elapsed={elapsed:.1f}s"
            )

    summary = {
        "line": args.line,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total": len(samples),
        "runtime_sec": round(time.time() - started_all, 3),
        "failed_csv": str(failed_csv),
    }
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
