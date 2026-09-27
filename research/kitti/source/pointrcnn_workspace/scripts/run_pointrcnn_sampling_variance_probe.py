#!/usr/bin/env python3
"""Quantify PointRCNN's point-sampling noise floor for the split-region protocol.

The frozen evaluator seeds numpy once per epoch, so every past run is a single
fixed draw from the sampling distribution rather than its expectation.  This
probe replays the same frozen workspace under different POINTRCNN_EVAL_SEED
offsets and records the resulting AP for each source, so that reported deltas
can be compared against the noise floor.

Nothing about the inputs, the checkpoint, the split, or the merge rule changes:
only the evaluator's sampling realization differs between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_pointrcnn_split_region_eval.py"
DEFAULT_SOURCE_WS = (
    REPO / "results/pointrcnn_object_preserving_detector_aware_pdans256_v1_20260805"
)
DEFAULT_OUT = REPO / "results/pointrcnn_sampling_variance_probe_20260810"
PYTHON = sys.executable


def build_seed_workspace(source_ws: Path, seed_ws: Path) -> None:
    """Symlink every read-only artifact; keep detector/ private to this seed."""
    seed_ws.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source_ws.iterdir()):
        if entry.name == "detector":
            continue
        link = seed_ws / entry.name
        if link.is_symlink() or link.exists():
            continue
        link.symlink_to(entry)
    (seed_ws / "detector").mkdir(exist_ok=True)


def read_ap(seed_ws: Path, source: str) -> dict | None:
    path = seed_ws / "detector" / source / "evaluation" / "result_summary.json"
    if not path.exists():
        return None
    summary = json.loads(path.read_text())
    ap = summary["ap_r40_percent"]
    return {
        "3d_easy": ap["3d_ap"]["easy"],
        "3d_moderate": ap["3d_ap"]["moderate"],
        "3d_hard": ap["3d_ap"]["hard"],
        "bev_easy": ap["bev_ap"]["easy"],
        "bev_moderate": ap["bev_ap"]["moderate"],
        "bev_hard": ap["bev_ap"]["hard"],
        "empty_prediction_files": summary["empty_prediction_files"],
    }


def run_one(seed: int, source: str, seed_ws: Path, log_dir: Path) -> tuple[bool, float]:
    env = os.environ.copy()
    env["POINTRCNN_EVAL_SEED"] = str(seed)
    log_path = log_dir / f"seed{seed:02d}_{source}.log"
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(
            [PYTHON, str(RUNNER), "--workspace", str(seed_ws), "--source", source],
            cwd=str(REPO),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return proc.returncode == 0, time.time() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--sources", nargs="+", default=["original_baseline", "full"],
        help="paired arms; seed 0 of 'full' reproduces the frozen 82.3995 result",
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    args = parser.parse_args()

    source_ws = args.source_workspace.resolve()
    out = args.out.resolve()
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results_path = out / "ap_by_seed.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for seed in args.seeds:
        seed_ws = out / f"seed_{seed:02d}"
        build_seed_workspace(source_ws, seed_ws)
        for source in args.sources:
            key = f"seed{seed:02d}/{source}"
            if key in results:
                print(f"PROBE_REUSE {key}", flush=True)
                continue
            print(f"PROBE_START {key}", flush=True)
            # Launching ~14 fresh CUDA subprocesses per source occasionally
            # segfaults before the child writes anything. Completed slots are
            # cached by the runner, so a retry only re-runs the failed slot.
            elapsed = 0.0
            for attempt in range(1, args.max_attempts + 1):
                ok, spent = run_one(seed, source, seed_ws, log_dir)
                elapsed += spent
                ap = read_ap(seed_ws, source)
                if ok and ap is not None:
                    break
                print(f"PROBE_RETRY {key} attempt={attempt}/{args.max_attempts}", flush=True)
            else:
                print(f"PROBE_FAIL {key} see {log_dir}/seed{seed:02d}_{source}.log", flush=True)
                return 1
            ap["attempts"] = attempt
            ap["runtime_seconds"] = round(elapsed, 1)
            ap["seed"] = seed
            ap["source"] = source
            results[key] = ap
            results_path.write_text(json.dumps(results, indent=2, sort_keys=True))
            print(
                f"PROBE_DONE {key} 3d_mod={ap['3d_moderate']:.4f} "
                f"bev_mod={ap['bev_moderate']:.4f} runtime={elapsed:.0f}s",
                flush=True,
            )

    print(f"PROBE_ALL_DONE {results_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
