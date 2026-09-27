#!/usr/bin/env python3
"""Run a resumable EAR raw + strict x4 stage for selected KITTI frames."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/ra87racy/projects/baseline_detectors/PointRCNN")
PYTHON = Path("/home/ra87racy/projects/thesis_demo/.venv/bin/python")
RAW_WRAPPER = ROOT / "scripts/wrappers/run_ear_real_raw.py"
FINAL_ADAPTER = ROOT / "scripts/wrappers/strict_x4_from_raw.py"
LINE_INPUTS = {
    "lineA": ROOT / "data/KITTI/object/training/velodyne_original_val",
    "lineB": ROOT / "data/KITTI/object/training/velodyne_downsampled_50_val",
}


def load_points(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        raise ValueError(f"{path} is not KITTI XYZI")
    return values.reshape(-1, 4)


def already_pass(raw_prov: Path, final_prov: Path, final_bin: Path, input_bin: Path) -> bool:
    if not (raw_prov.exists() and final_prov.exists() and final_bin.exists()):
        return False
    try:
        raw = json.loads(raw_prov.read_text(encoding="utf-8"))
        final = json.loads(final_prov.read_text(encoding="utf-8"))
        inp = load_points(input_bin)
        out = load_points(final_bin)
        return (
            raw.get("status") == "PASS"
            and raw.get("method_algorithm_called") is True
            and raw.get("fallback_used") is False
            and raw.get("generic_adapter_used_before_raw") is False
            and raw.get("raw_output_generated_from") == "EAR_algorithm"
            and final.get("status") == "PASS"
            and final.get("adapter_input_is_raw_method_output") is True
            and out.shape == (inp.shape[0] * 4, 4)
            and np.isfinite(out).all()
        )
    except Exception:
        return False


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"\n[exit_code={proc.returncode}]\n")
        return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run selected EAR sequential stage")
    parser.add_argument("--stage", required=True, choices=["five_frame_smoke", "full_val"])
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--max_iter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--up_ratio", type=float, default=4.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.frames:
        frames = args.frames
    elif args.stage == "five_frame_smoke":
        frames = ["000001", "000093", "000242", "003219", "006833"]
    else:
        frames = [p.stem for p in sorted(LINE_INPUTS["lineA"].glob("*.bin"))]

    out_root = ROOT / "results/kitti_real_method_sequential/EAR" / args.stage
    audit_rows = []
    failures = 0

    for frame_id in frames:
        for line, input_dir in LINE_INPUTS.items():
            input_bin = input_dir / f"{frame_id}.bin"
            raw_bin = out_root / "raw_outputs" / line / f"{frame_id}.bin"
            final_bin = out_root / "final_strict_x4" / line / f"{frame_id}.bin"
            raw_prov = out_root / "provenance" / f"{line}_{frame_id}_raw.json"
            final_prov = out_root / "provenance" / f"{line}_{frame_id}_final.json"
            log_path = out_root / "logs" / f"{line}_{frame_id}.log"

            if not input_bin.exists():
                failures += 1
                audit_rows.append({"line": line, "frame_id": frame_id, "status": "FAIL_MISSING_INPUT"})
                continue
            if args.resume and already_pass(raw_prov, final_prov, final_bin, input_bin):
                audit_rows.append({"line": line, "frame_id": frame_id, "status": "PASS_SKIPPED_EXISTING"})
                continue

            raw_cmd = [
                str(PYTHON), str(RAW_WRAPPER),
                "--input_bin", str(input_bin),
                "--raw_output", str(raw_bin),
                "--provenance_json", str(raw_prov),
                "--line", line,
                "--frame_id", frame_id,
                "--seed", str(args.seed),
                "--up_ratio", str(args.up_ratio),
                "--max_iter", str(args.max_iter),
            ]
            rc = run(raw_cmd, log_path)
            if rc != 0:
                failures += 1
                audit_rows.append({"line": line, "frame_id": frame_id, "status": "FAIL_RAW"})
                continue

            final_cmd = [
                str(PYTHON), str(FINAL_ADAPTER),
                "--raw_output", str(raw_bin),
                "--input_bin", str(input_bin),
                "--final_output", str(final_bin),
                "--provenance_json", str(final_prov),
                "--raw_provenance_json", str(raw_prov),
                "--line", line,
                "--frame_id", frame_id,
                "--up_ratio", str(args.up_ratio),
                "--seed", str(args.seed),
            ]
            rc = run(final_cmd, log_path)
            if rc != 0:
                failures += 1
                audit_rows.append({"line": line, "frame_id": frame_id, "status": "FAIL_FINAL"})
                continue

            audit_rows.append({"line": line, "frame_id": frame_id, "status": "PASS"})

    audit_path = out_root / "audit" / f"{args.stage}_run_status.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["line", "frame_id", "status"])
        w.writeheader()
        w.writerows(audit_rows)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
