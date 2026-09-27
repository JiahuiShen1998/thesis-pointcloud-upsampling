#!/usr/bin/env python3
"""Evaluate fully adapted observed-first PointRCNN checkpoints on pilot256."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "results/pugcn_full_retrain_20260824"
EVAL_ONE = ROOT / "results/pointrcnn_finetune64_six_arms_20260824/run_eval_one.py"
EVAL_INPUTS = {
    "line_a_pugcn_observed_first": EXPERIMENT
    / "evaluations/centerpoint/line_a_pugcn/observed_first_input",
    "line_b_pugcn_observed_first": EXPERIMENT
    / "evaluations/centerpoint/line_b_pugcn/observed_first_input",
}


def main() -> None:
    state_path = EXPERIMENT / "evaluations/pointrcnn_observed_first/state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    for arm, lidar_dir in EVAL_INPUTS.items():
        checkpoint = EXPERIMENT / f"pointrcnn/{arm}/combined_rpn_rcnn_adapted.pth"
        output = EXPERIMENT / f"evaluations/pointrcnn_observed_first/{arm}"
        command = [
            sys.executable,
            str(EVAL_ONE),
            "--lidar-dir",
            str(lidar_dir),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
        ]
        print(f"POINT_RCNN_OBSERVED_FIRST_EVAL_START {arm}", flush=True)
        subprocess.run(command, cwd=ROOT / "tools", check=True)
        completed.append(arm)
        state_path.write_text(
            json.dumps({"status": "RUNNING", "completed": completed}, indent=2)
            + "\n"
        )
    state_path.write_text(
        json.dumps({"status": "PASS", "completed": completed}, indent=2) + "\n"
    )
    print("POINT_RCNN_OBSERVED_FIRST_EVAL_PASS", flush=True)


if __name__ == "__main__":
    main()
