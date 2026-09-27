#!/usr/bin/env python3
"""Score an existing prediction directory on a split without rerunning inference."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "scripts/run_variant_eval_detector_recovery.py"
SPEC = importlib.util.spec_from_file_location("detector_recovery_runner", RUNNER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load recovery runner: {RUNNER_PATH}")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
base = runner.base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    args = parser.parse_args()
    pred_dir = args.pred_dir.resolve()
    out_dir = args.out_dir.resolve()
    split_file = args.split_file.resolve()
    frames = base.frame_ids(split_file)
    missing = [frame for frame in frames if not (pred_dir / f"{frame}.txt").is_file()]
    if missing:
        raise FileNotFoundError(f"existing prediction coverage missing {len(missing)} frames: {missing[:20]}")
    empty = sum((pred_dir / f"{frame}.txt").stat().st_size == 0 for frame in frames)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = runner.run_cpp_eval_scoped(out_dir, pred_dir, split_file)
    summary = base.write_summary(
        args.name,
        pred_dir,
        out_dir,
        len(frames),
        empty,
        0.0,
        0,
        artifacts,
    )
    summary["source_kind"] = "existing_predictions_no_inference"
    summary["evaluation_scope"] = {
        "split_file": str(split_file),
        "frames": len(frames),
        "labels_outside_split_are_empty": True,
    }
    (out_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"EXISTING_PREDICTIONS_SCORE_PASS {args.name} frames={len(frames)} "
        f"mod3d={summary['ap_r40_percent']['3d_ap']['moderate']:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
