#!/usr/bin/env python3
"""Evaluate the three full-split PointRCNN adaptation arms on pilot256."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "results/pugcn_full_retrain_20260824"
EVAL_ONE = (
    ROOT
    / "results/pointrcnn_finetune64_six_arms_20260824/run_eval_one.py"
)
EVAL_INPUTS = {
    "line_a_pugcn": ROOT
    / "results/kitti_patch_punet_causal_ablation_v1_20260731/surface_candidate256_v1/pu_gcn_surface_cover_exact_pr1_c32/line_a_original_x4_up/final_bin",
    "line_b_pugcn": ROOT
    / "results/kitti_patch_punet_causal_ablation_v1_20260731/c2048_r4_pilot256_v1/pu_gcn_c2048_r4/line_b_downsampled_x4_up/final_bin",
    "line_b_baseline": ROOT
    / "results/kitti_unified_x4_current_methods_no_detector/downsampled_x4/velodyne_downsampled_x4_val",
}


def main() -> None:
    state_path = EXPERIMENT / "evaluations/pointrcnn/state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    for arm, lidar_dir in EVAL_INPUTS.items():
        checkpoint = (
            EXPERIMENT
            / f"pointrcnn/{arm}/combined_rpn_rcnn_adapted.pth"
        )
        output = EXPERIMENT / f"evaluations/pointrcnn/{arm}"
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
        print(f"POINT_RCNN_EVAL_START {arm}", flush=True)
        subprocess.run(command, cwd=ROOT / "tools", check=True)
        completed.append(arm)
        state_path.write_text(
            json.dumps({"status": "RUNNING", "completed": completed}, indent=2)
            + "\n"
        )
    state_path.write_text(
        json.dumps({"status": "PASS", "completed": completed}, indent=2) + "\n"
    )
    print("POINT_RCNN_FULL_EVAL_PASS", flush=True)


if __name__ == "__main__":
    main()
