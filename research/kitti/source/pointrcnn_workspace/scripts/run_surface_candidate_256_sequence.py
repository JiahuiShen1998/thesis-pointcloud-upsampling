#!/usr/bin/env python3
"""Gate and run the frozen surface-c32 256-frame upsampling sequence."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / "venv_pointrcnn/bin/python"
ROOT = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
FRAMES = ROOT / "splits/pilot256.txt"
CACHE = ROOT / "pilot256_surface_search_v1/surface_cover_exact_pr1_c32/line_a_original_x4_up"
RUN_KIND = "surface_candidate256_v1"
LINE = "line_a_original_x4_up"
OBSERVED = REPO / "data/KITTI/object/training/velodyne_original"
STATE = ROOT / RUN_KIND / "sequence_state.json"

COMMON = [
    "--run-kind", RUN_KIND,
    "--frames-file", str(FRAMES),
    "--patch-selection", "fps_ball_cover_knn_v3",
    "--patch-num-ratio", "1",
    "--ball-radius-m", "2",
    "--min-ball-points", "2048",
    "--cover-min-points", "32",
    "--cover-radius-m", "6",
    "--lines", LINE,
    "--reuse-patch-base", str(CACHE),
]

METHODS = [
    ("PU-GCN", "pu_gcn", "pu_gcn_surface_cover_exact_pr1_c32", []),
    ("PU-EdgeFormer", "pu_edgeformer", "pu_edgeformer_surface_cover_exact_pr1_c32", []),
    (
        "PU-Net-fixed",
        "pu_net",
        "pu_net_fixed_surface_cover_exact_pr1_c32",
        ["--normalization-mode", "unit_sphere_v1"],
    ),
    ("PDANS", "pdans", "pdans_surface_cover_exact_pr1_c32", []),
]


def save_state(payload: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed with rc={result.returncode}: {command}")


def output_dir(variant: str) -> Path:
    return ROOT / RUN_KIND / variant / LINE / "final_bin"


def final_count(variant: str) -> int:
    directory = output_dir(variant)
    return len(list(directory.glob("*.bin"))) if directory.is_dir() else 0


def verify(label: str, variant: str) -> None:
    if final_count(variant) != 256:
        raise RuntimeError(f"{label}: expected 256 final bins, got {final_count(variant)}")
    report_dir = ROOT / RUN_KIND / variant / LINE / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            str(PYTHON),
            str(REPO / "scripts/verify_patch_causal_strict_x4.py"),
            "--frames-file", str(FRAMES),
            "--observed-dir", str(OBSERVED),
            "--predicted-dir", str(output_dir(variant)),
            "--output-json", str(report_dir / "strict_x4_independent_summary.json"),
            "--output-csv", str(report_dir / "strict_x4_independent_frames.csv"),
        ]
    )


def wait_for_existing_pugcn() -> None:
    while subprocess.run(
        ["tmux", "has-session", "-t", "pugcn_surface_c32_256"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0:
        print(f"WAIT PU-GCN final={final_count(METHODS[0][2])}/256", flush=True)
        time.sleep(30)


def main() -> int:
    state: dict[str, object] = {
        "status": "RUNNING",
        "started_unix": time.time(),
        "completed": [],
        "current": "PU-GCN",
    }
    save_state(state)
    wait_for_existing_pugcn()
    verify(METHODS[0][0], METHODS[0][2])
    state["completed"].append(METHODS[0][0])
    save_state(state)

    for label, method, variant, extra in METHODS[1:]:
        free_gib = shutil.disk_usage(ROOT).free / (1024**3)
        if free_gib < 100.0:
            raise RuntimeError(f"free disk below 100 GiB before {label}: {free_gib:.3f}")
        state["current"] = label
        state["free_gib_before_current"] = free_gib
        save_state(state)
        if final_count(variant) != 256:
            run(
                [
                    str(PYTHON),
                    str(REPO / "scripts/run_patch_causal_upsampling.py"),
                    "--method", method,
                    "--variant-name", variant,
                    *COMMON,
                    *extra,
                ]
            )
        verify(label, variant)
        state["completed"].append(label)
        save_state(state)

    state["status"] = "PASS"
    state["current"] = None
    state["finished_unix"] = time.time()
    state["free_gib_after"] = shutil.disk_usage(ROOT).free / (1024**3)
    save_state(state)
    print(json.dumps(state, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        payload = {
            "status": "FAIL",
            "failed_unix": time.time(),
            "error": repr(error),
        }
        save_state(payload)
        raise
