#!/usr/bin/env python3
"""Run gated PointRCNN and CenterPoint evaluations for surface-c32 outputs."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / "venv_pointrcnn/bin/python"
ROOT = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
UPSAMPLE_ROOT = ROOT / "surface_candidate256_v1"
DETECTOR_ROOT = ROOT / "detectors"
SPLIT = REPO / "data/KITTI/ImageSets/patch_causal_pilot256.txt"
PROTOCOL = (
    REPO
    / "results/centerpoint_exact4n_reconstructed_order_safe_20260729"
    / "pilot256_protocol.json"
)
STATE = UPSAMPLE_ROOT / "detector_sequence_state.json"
LINE = "line_a_original_x4_up"

METHODS = [
    ("PU-GCN", "surface_c32_pu_gcn", "pu_gcn_surface_cover_exact_pr1_c32", "original_x4_pu_gcn"),
    (
        "PU-EdgeFormer",
        "surface_c32_pu_edgeformer",
        "pu_edgeformer_surface_cover_exact_pr1_c32",
        "original_x4_pu_edgeformer",
    ),
    (
        "PU-Net-fixed",
        "surface_c32_pu_net_fixed",
        "pu_net_fixed_surface_cover_exact_pr1_c32",
        "original_x4_pu_net",
    ),
    ("PDANS", "surface_c32_pdans", "pdans_surface_cover_exact_pr1_c32", "original_x4_pdans"),
]


def save_state(payload: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def load_pass(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "PASS"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def run(command: list[str]) -> None:
    print("RUN " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=REPO, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}: {command}")


def predicted_dir(variant: str) -> Path:
    return UPSAMPLE_ROOT / variant / LINE / "final_bin"


def verify_inputs() -> None:
    for label, _, variant, _ in METHODS:
        summary = UPSAMPLE_ROOT / variant / LINE / "reports/strict_x4_independent_summary.json"
        if not load_pass(summary):
            raise RuntimeError(f"strict-x4 summary is not PASS for {label}: {summary}")
        if len(list(predicted_dir(variant).glob("*.bin"))) != 256:
            raise RuntimeError(f"expected 256 predicted bins for {label}")


def main() -> int:
    verify_inputs()
    state: dict[str, object] = {
        "status": "RUNNING",
        "started_unix": time.time(),
        "completed": [],
        "current": None,
    }
    save_state(state)

    for label, name, variant, _ in METHODS:
        if shutil.disk_usage(ROOT).free / (1024**3) < 100.0:
            raise RuntimeError(f"free disk below 100 GiB before PointRCNN/{label}")
        state["current"] = f"PointRCNN/{label}"
        save_state(state)
        output = DETECTOR_ROOT / "pointrcnn" / name
        summary = output / "result_summary.json"
        if not load_pass(summary):
            run(
                [
                    str(PYTHON),
                    str(REPO / "scripts/run_variant_eval_detector_recovery.py"),
                    "--name", name,
                    "--input-dir", str(predicted_dir(variant)),
                    "--out-dir", str(output),
                    "--split-file", str(SPLIT),
                ]
            )
        if not load_pass(summary):
            raise RuntimeError(f"PointRCNN summary is not PASS for {label}")
        state["completed"].append(f"PointRCNN/{label}")
        save_state(state)

    for label, name, variant, reference_token in METHODS:
        if shutil.disk_usage(ROOT).free / (1024**3) < 100.0:
            raise RuntimeError(f"free disk below 100 GiB before CenterPoint/{label}")
        state["current"] = f"CenterPoint/{label}"
        save_state(state)
        output = DETECTOR_ROOT / "centerpoint" / name
        summary = output / "result_summary.json"
        if not load_pass(summary):
            run(
                [
                    str(PYTHON),
                    str(REPO / "scripts/run_patch_causal_centerpoint_eval.py"),
                    "--name", name,
                    "--line", "line_a",
                    "--source-mode", "upsampled_observed_first",
                    "--predicted-dir", str(predicted_dir(variant)),
                    "--reference-token", reference_token,
                    "--protocol-json", str(PROTOCOL),
                    "--result-root", str(DETECTOR_ROOT / "centerpoint"),
                    "--batch-size", "1",
                    "--workers", "0",
                    "--resume",
                ]
            )
        if not load_pass(summary):
            raise RuntimeError(f"CenterPoint summary is not PASS for {label}")
        state["completed"].append(f"CenterPoint/{label}")
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
        save_state({"status": "FAIL", "failed_unix": time.time(), "error": repr(error)})
        raise
