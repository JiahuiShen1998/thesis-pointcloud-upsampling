#!/usr/bin/env python3
"""Validated PointRCNN evaluator with only RPN.NUM_POINTS=32768 changed."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
BASE_PATH = (
    REPO
    / "results/kitti_unified_x4_pointrcnn_eval_line_a_sampler_safe_v2_20260717_235037"
    / "commands/run_variant_eval.py"
)
SPEC = importlib.util.spec_from_file_location("validated_run_variant_eval", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load validated evaluator: {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def run_inference(name, input_dir, out_dir, split_file):
    out_dir.mkdir(parents=True, exist_ok=True)
    before = base.record_velodyne(base.VEL)
    (out_dir / "velodyne_before.txt").write_text(before + "\n")
    split_name = Path(split_file).stem
    command = [
        str(base.PYTHON),
        "eval_rcnn_official.py",
        "--cfg_file",
        "cfgs/default.yaml",
        "--ckpt",
        "PointRCNN.pth",
        "--batch_size",
        "1",
        "--workers",
        "0",
        "--eval_mode",
        "rcnn",
        "--output_dir",
        str(out_dir / "inference"),
        "--set",
        "RPN.LOC_XZ_FINE",
        "False",
        "RPN.NUM_POINTS",
        "32768",
        "TEST.SPLIT",
        split_name,
    ]
    (out_dir / "command.txt").write_text(" ".join(command) + "\n")
    (out_dir / "checkpoint_used.txt").write_text(str(base.CKPT) + "\n")
    (out_dir / "config_used.txt").write_text(
        f"{base.CFG}\noverride: RPN.NUM_POINTS=32768\n"
    )
    (out_dir / "split_used.txt").write_text(
        f"{split_name} ({len(base.frame_ids(split_file))} frames)\n"
    )
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"

    try:
        if base.VEL.is_symlink() or base.VEL.exists():
            if base.VEL.is_dir() and not base.VEL.is_symlink():
                raise RuntimeError("Refusing to replace non-symlink velodyne directory")
            base.VEL.unlink(missing_ok=True)
        base.VEL.symlink_to(input_dir)
        with (out_dir / "inference_stdout_stderr.log").open("wb") as log:
            start = time.time()
            proc = subprocess.run(
                command,
                cwd=base.TOOLS,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            elapsed = time.time() - start
        (out_dir / "runtime_seconds.txt").write_text(f"{elapsed:.2f}\n")
        (out_dir / "returncode.txt").write_text(str(proc.returncode) + "\n")
    finally:
        base.restore_velodyne(before)
        (out_dir / "velodyne_after.txt").write_text(base.record_velodyne(base.VEL) + "\n")
    return elapsed, proc.returncode


base.run_inference = run_inference

if __name__ == "__main__":
    base.main()
