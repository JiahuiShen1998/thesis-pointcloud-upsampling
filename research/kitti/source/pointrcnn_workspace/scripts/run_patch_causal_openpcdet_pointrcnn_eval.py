#!/usr/bin/env python3
"""Evaluate frozen KITTI inputs with the OpenPCDet three-class PointRCNN."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import run_patch_causal_centerpoint_eval as base


REPO = Path(__file__).resolve().parents[1]
OPENPCDET = REPO / "external/OpenPCDet"
TOOLS = OPENPCDET / "tools"
PYTHON = Path("/home/ra87racy/miniconda3/envs/openpcdet_centerpoint/bin/python")
CHECKPOINT = OPENPCDET / "checkpoints/pointrcnn_8369.pth"
CONFIG = "cfgs/kitti_models/pointrcnn.yaml"
DEFAULT_RESULT_ROOT = (
    REPO / "results/openpcdet_pointrcnn_three_class_holdout64_20260808"
)


def run_eval(
    name: str,
    run_root: Path,
    overlay: Path,
    batch_size: int,
    workers: int,
    resume: bool,
) -> None:
    output = run_root / "openpcdet_output"
    output.mkdir(parents=True, exist_ok=True)
    if resume and base.evaluation_done(run_root):
        print(f"SKIP completed {name}", flush=True)
        return

    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    tag = f"frozen_three_class_{safe_name}"
    output_link = OPENPCDET / "output/kitti_models/pointrcnn" / tag
    output_link.parent.mkdir(parents=True, exist_ok=True)
    base.ensure_symlink(output_link, output)

    command = [
        str(PYTHON),
        "test.py",
        "--cfg_file",
        CONFIG,
        "--batch_size",
        str(batch_size),
        "--workers",
        str(workers),
        "--ckpt",
        str(CHECKPOINT),
        "--extra_tag",
        tag,
        "--eval_tag",
        "frozen_three_class_comparison",
        "--set",
        "DATA_CONFIG.DATA_PATH",
        str(overlay.resolve()),
    ]
    (run_root / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    print(f"START OpenPCDet-PointRCNN {name}", flush=True)
    log_path = run_root / "runner_stdout.log"
    log_path.write_text("", encoding="utf-8")
    result = None
    attempts = 0
    for attempt in range(1, 4):
        attempts = attempt
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"ATTEMPT {attempt}/3\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=TOOLS,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != -11 or base.evaluation_done(run_root):
            break
        time.sleep(2)
    assert result is not None
    elapsed = time.monotonic() - started
    completed = base.evaluation_done(run_root)
    (run_root / "raw_returncode.txt").write_text(f"{result.returncode}\n")
    (run_root / "attempts.txt").write_text(f"{attempts}\n")
    (run_root / "runtime_seconds.txt").write_text(f"{elapsed:.3f}\n")
    if not completed:
        tail = "\n".join(
            (run_root / "runner_stdout.log")
            .read_text(errors="replace")
            .splitlines()[-100:]
        )
        print(tail, file=sys.stderr)
        raise RuntimeError(f"OpenPCDet PointRCNN did not complete for {name}")
    print(f"END OpenPCDet-PointRCNN {name} runtime={elapsed:.1f}s", flush=True)


def write_summary(
    args,
    run_root: Path,
    frames: list[str],
    source: Path,
) -> None:
    logs = sorted((run_root / "openpcdet_output").rglob("log_eval_*.txt"))
    if not logs:
        raise FileNotFoundError("OpenPCDet PointRCNN evaluation log not found")
    metrics = base.parse_metrics(logs[-1].read_text(errors="replace"))
    runtime_seconds = float((run_root / "runtime_seconds.txt").read_text())
    payload = {
        "status": "PASS",
        "name": args.name,
        "line": args.line,
        "source_mode": args.source_mode,
        "source": str(source.resolve()),
        "protocol_json": str(args.protocol_json.resolve()),
        "frames": len(frames),
        "checkpoint": str(CHECKPOINT.resolve()),
        "detector_config": CONFIG,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "point_sampling": "native OpenPCDet PointRCNN test sample_points=16384",
        "runtime_seconds": runtime_seconds,
        "seconds_per_frame_wall": runtime_seconds / len(frames),
        "metrics_percent": metrics,
    }
    (run_root / "result_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, object]] = []
    for class_name, class_metrics in metrics.items():
        for metric, values in class_metrics.items():
            rows.append({"class": class_name, "metric": metric, **values})
    with (run_root / "parsed_ap_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"PASS {args.name} frames={len(frames)} runtime={runtime_seconds:.1f}s "
        f"Car/Pedestrian/Cyclist moderate 3D="
        f"{metrics['Car']['3d_ap_r40']['moderate']:.4f}/"
        f"{metrics['Pedestrian']['3d_ap_r40']['moderate']:.4f}/"
        f"{metrics['Cyclist']['3d_ap_r40']['moderate']:.4f}",
        flush=True,
    )


def main() -> int:
    args = base.parse_args()
    args.protocol_json = args.protocol_json.resolve()
    if args.result_root == base.DEFAULT_RESULT_ROOT:
        args.result_root = DEFAULT_RESULT_ROOT
    args.result_root = args.result_root.resolve()
    frames = base.protocol_frames(args.protocol_json)
    run_root = args.result_root / args.name
    run_root.mkdir(parents=True, exist_ok=True)

    if args.source_mode == "baseline":
        source = base.observed_source(args.line).resolve()
    elif args.source_mode == "direct":
        source = args.predicted_dir.resolve()
        missing = [frame for frame in frames if not (source / f"{frame}.bin").is_file()]
        if missing:
            raise FileNotFoundError(
                f"direct source missing {len(missing)} protocol frames: {missing[:10]}"
            )
    else:
        staged = run_root / "observed_first_input"
        if args.eval_only:
            source = staged
        else:
            source = base.prepare_observed_first(
                frames,
                args.line,
                args.predicted_dir,
                args.reference_token,
                staged,
            )
    if args.prepare_only:
        return 0
    overlay = base.prepare_overlay(run_root, source, base.selected_infos(frames))
    run_eval(args.name, run_root, overlay, args.batch_size, args.workers, args.resume)
    write_summary(args, run_root, frames, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
