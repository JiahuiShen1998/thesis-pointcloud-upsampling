#!/usr/bin/env python3
"""Run one isolated PointRCNN recovery variant with split-scoped C++ AP.

Inference uses the validated original PointRCNN checkpoint/config and 16,384
input points.  The active velodyne symlink is restored in a ``finally`` block.
For screening splits, only the requested frames receive labels in the C++
evaluation workspace, so AP is scoped to the screening frames rather than
silently treating all other validation frames as false negatives.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
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


def run_cpp_eval_scoped(out_dir: Path, pred_dir: Path, split_file: Path) -> Path:
    selected = set(base.frame_ids(split_file))
    work = out_dir / "kitti_cpp_eval_work"
    if work.exists():
        shutil.rmtree(work)
    data_label = work / "data/object/label_2"
    result_sha = "eval_result"
    result_data = work / "results" / result_sha / "data"
    data_label.mkdir(parents=True)
    result_data.mkdir(parents=True)

    for index in range(7481):
        frame = f"{index:06d}"
        label = data_label / f"{frame}.txt"
        if frame in selected:
            source = base.LABEL_FULL / f"{frame}.txt"
            if not source.exists():
                source = base.LABEL_REAL / f"{frame}.txt"
            if source.exists():
                label.symlink_to(source)
            else:
                label.write_text("")
        else:
            label.write_text("")

        prediction_source = pred_dir / f"{frame}.txt"
        prediction_target = result_data / f"{frame}.txt"
        if frame in selected and prediction_source.exists():
            prediction_target.write_bytes(prediction_source.read_bytes())
        else:
            prediction_target.write_text("")

    with (out_dir / "kitti_cpp_eval.log").open("wb") as log:
        proc = subprocess.run(
            [str(base.CPP_EVAL), result_sha],
            cwd=work,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        raise RuntimeError(f"KITTI C++ eval failed with return code {proc.returncode}")

    result = work / "results" / result_sha
    artifacts = out_dir / "kitti_cpp_eval_artifacts"
    if artifacts.exists():
        shutil.rmtree(artifacts)
    artifacts.mkdir()
    for filename in (
        "stats_car_detection.txt",
        "stats_car_detection_ground.txt",
        "stats_car_detection_3d.txt",
        "stats_car_orientation.txt",
    ):
        source = result / filename
        if source.exists():
            shutil.copy2(source, artifacts / filename)
    for filename in ("car_detection.txt", "car_detection_ground.txt", "car_detection_3d.txt"):
        source = result / "plot" / filename
        if source.exists():
            shutil.copy2(source, artifacts / filename)
    shutil.rmtree(work)
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, default=base.VAL)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    out_dir = args.out_dir.resolve()
    split_file = args.split_file.resolve()
    print(f"RECOVERY_RUN_START {args.name} split={split_file}", flush=True)

    try:
        # Never combine stale predictions from an interrupted inference with
        # a resumed run.  The directory contains only regenerable outputs for
        # this isolated variant.
        inference_dir = out_dir / "inference"
        if inference_dir.exists():
            shutil.rmtree(inference_dir, ignore_errors=True)
        elapsed, returncode = base.run_inference(args.name, input_dir, out_dir, split_file)
        pred_dir, pred_count, empty_count = base.coverage(out_dir, split_file)
        if returncode != 0:
            print(
                f"WARN {args.name} inference exited {returncode}, but prediction coverage is complete",
                flush=True,
            )
        artifacts = run_cpp_eval_scoped(out_dir, pred_dir, split_file)
        summary = base.write_summary(
            args.name,
            input_dir,
            out_dir,
            pred_count,
            empty_count,
            elapsed,
            returncode,
            artifacts,
        )
        summary["evaluation_scope"] = {
            "split_file": str(split_file),
            "frames": len(base.frame_ids(split_file)),
            "labels_outside_split_are_empty": True,
        }
        (out_dir / "result_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"RECOVERY_RUN_DONE {args.name} predictions={pred_count} empty={empty_count} "
            f"runtime={elapsed:.1f}s mod3d={summary['ap_r40_percent']['3d_ap']['moderate']:.6f}",
            flush=True,
        )
    except Exception as error:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result_summary.json").write_text(
            json.dumps({"name": args.name, "status": "FAIL", "error": str(error)}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"RECOVERY_RUN_FAIL {args.name} error={error}", flush=True)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
