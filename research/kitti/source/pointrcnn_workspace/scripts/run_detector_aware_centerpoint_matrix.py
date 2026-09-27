#!/usr/bin/env python3
"""Run the frozen CenterPoint detector-aware PDANS 256-frame matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts/run_patch_causal_centerpoint_eval.py"
DEFAULT_INPUT_ROOT = REPO / "results/detector_aware_pdans_surface256_v1_20260804/inputs"
DEFAULT_RESULT_ROOT = REPO / "results/centerpoint_detector_aware_pdans256_v1_20260805"
DEFAULT_PROTOCOL = (
    REPO / "results/centerpoint_exact4n_reconstructed_order_safe_20260729/pilot256_protocol.json"
)
VARIANTS = (
    "fixed_2p5_pdans",
    "matched_real_dynamic",
    "full",
    "no_confidence",
    "no_sparse",
    "no_adaptive",
    "nn_only",
    "density_only",
)


def passed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "PASS"
    except (OSError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--protocol-json", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--variant", action="append", choices=VARIANTS)
    args = parser.parse_args()
    selected = tuple(args.variant or VARIANTS)
    args.result_root.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(selected, start=1):
        summary = args.result_root / name / "result_summary.json"
        if passed(summary):
            print(f"CENTERPOINT_MATRIX_REUSE {name} progress={index}/{len(selected)}", flush=True)
            continue
        command = [
            sys.executable,
            str(RUNNER),
            "--name", name,
            "--line", "line_a",
            "--source-mode", "direct",
            "--predicted-dir", str((args.input_root / name).resolve()),
            "--protocol-json", str(args.protocol_json.resolve()),
            "--result-root", str(args.result_root.resolve()),
            "--batch-size", "1",
            "--workers", "0",
            "--resume",
        ]
        completed = subprocess.run(command, cwd=REPO)
        if completed.returncode != 0 or not passed(summary):
            raise RuntimeError(f"CenterPoint failed for {name}: returncode={completed.returncode}")
        print(f"CENTERPOINT_MATRIX_DONE {name} progress={index}/{len(selected)}", flush=True)
    print("CENTERPOINT_DETECTOR_AWARE_MATRIX_PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
