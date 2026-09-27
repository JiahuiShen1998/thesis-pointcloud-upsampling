#!/usr/bin/env python3
"""After Line A completes: audit, run Line B resume, global audit, symlinks, report."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import run_pdans_two_lines_full_val as full_val
import run_pdans_resume_chunks as resume


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EAR_PY = Path("/home/ra87racy/miniconda3/envs/ear/bin/python")
LOG = full_val.AUDIT_DIR / "pdans_supervisor_after_line_a.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def count_final_bin(line: str) -> int:
    d = full_val.line_dir(line) / "final_bin"
    return len(list(d.glob("*.bin"))) if d.exists() else 0


def run(cmd: list[str]) -> int:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    log(f"exit_code={proc.returncode}")
    return proc.returncode


def wait_line_a(target: int = 3769, poll_sec: int = 300) -> bool:
    while True:
        count = count_final_bin("line_a_original_x4_up")
        log(f"line_a_final_bin={count}/{target}")
        if count >= target:
            return True
        time.sleep(poll_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=86400)
    args = parser.parse_args()

    frames = full_val.read_split()
    if not args.skip_wait:
        wait_line_a(len(frames))

    code = run(
        [
            str(EAR_PY),
            str(PROJECT_ROOT / "scripts" / "run_pdans_resume_chunks.py"),
            "--line",
            "line_a_original_x4_up",
            "--scan-only",
        ]
    )
    if code != 0:
        return code

    # Gate on the live per-frame validation (identical to --scan-only), not the
    # historical full_val manifest CSV: that manifest is only rewritten by
    # merge_manifest()/run_line() when a frame is actually re-processed, so a
    # frame that failed once and was later repaired via resume can leave a
    # stale FAIL row behind even though its final_bin is now valid.
    valid_rows, missing, invalid = resume.scan_line("line_a_original_x4_up", frames)
    final_bin_count = count_final_bin("line_a_original_x4_up")
    if missing or invalid or len(valid_rows) != len(frames) or final_bin_count != len(frames):
        log(
            f"LINE_A_AUDIT_FAIL valid={len(valid_rows)}/{len(frames)} "
            f"missing={len(missing)} invalid={len(invalid)} final_bin={final_bin_count}"
        )
        return 1
    log("LINE_A_AUDIT_PASS")

    code = run(
        [
            str(EAR_PY),
            str(PROJECT_ROOT / "scripts" / "run_pdans_resume_chunks.py"),
            "--line",
            "line_b_downsampled_x4_up",
            "--chunk-size",
            str(args.chunk_size),
            "--timeout",
            str(args.timeout),
        ]
    )
    if code != 0:
        return code

    line_b_manifest = resume.smoke.read_existing_manifest(full_val.LINE_MANIFESTS["line_b_downsampled_x4_up"])
    b_passed = sum(1 for row in line_b_manifest if row.get("status") == "PASS")
    if b_passed != len(frames):
        log(f"LINE_B_AUDIT_FAIL pass={b_passed}/{len(frames)}")
        return 1
    log("LINE_B_AUDIT_PASS")

    rows_by_line = {
        "line_a_original_x4_up": manifest,
        "line_b_downsampled_x4_up": line_b_manifest,
    }
    packaging = full_val.package_detector_ready(rows_by_line, frames)
    full_val.update_global_audit(frames)
    full_val.write_two_line_report(rows_by_line, frames, packaging)
    log("PDANS_TWO_LINE_PIPELINE_COMPLETE")
    print("DETECTOR_EVAL_STARTED=NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
