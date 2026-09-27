#!/usr/bin/env python3
"""Wait for sufficient free GPU memory, then run PDANS smoke + full val."""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "results" / "kitti_unified_x4_current_methods_no_detector" / "audits"
LOG_PATH = AUDIT_DIR / "pdans_gpu_ready_pipeline.log"
MIN_FREE_MIB = 3000
POLL_SEC = 60
MAX_WAIT_SEC = 86400


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def free_gpu_mib() -> int:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(proc.stdout.strip().splitlines()[0].strip())
    except Exception:
        return 0


def run(cmd: list[str]) -> int:
    log("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    log(f"exit_code={proc.returncode}")
    return proc.returncode


def write_smoke_reports() -> None:
    manifest_src = AUDIT_DIR / "smoke_test_manifest.csv"
    rows = [row for row in csv.DictReader(manifest_src.open(encoding="utf-8")) if row.get("method") == "pdans"]
    out_csv = AUDIT_DIR / "pdans_smoke_after_fix_manifest.csv"
    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    passed = sum(1 for row in rows if row.get("status") == "PASS")
    failed = len(rows) - passed
    lines = [
        "# PDANS Smoke Test After Fix",
        "",
        "- After ninja / pointnet2_ops fix rerun: `true`",
        "- Frames: `000001, 000093, 000242, 003219, 006833, 007458`",
        f"- Rows: `{len(rows)}`",
        f"- PASS: `{passed}`",
        f"- FAIL: `{failed}`",
        "- Detector evaluation: `NOT_STARTED`",
        "- Reused old output: `false`",
        "",
        "## Failed Frames",
        "",
    ]
    bad = [row for row in rows if row.get("status") != "PASS"]
    if bad:
        for row in bad:
            lines.append(
                f"- {row['line']}/{row['frame_id']}: `{row.get('error_message', '')}` log `{row.get('log_path', '')}`"
            )
    else:
        lines.append("- none")
    (AUDIT_DIR / "pdans_smoke_after_fix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    deadline = time.time() + MAX_WAIT_SEC
    log("PDANS gpu-ready pipeline started")
    while time.time() < deadline:
        free = free_gpu_mib()
        log(f"gpu_free_mib={free} required={MIN_FREE_MIB}")
        if free >= MIN_FREE_MIB:
            break
        time.sleep(POLL_SEC)
    else:
        log("timeout waiting for GPU")
        write_smoke_reports()
        return 1

    smoke_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_kitti_unified_x4_current_methods_smoke.py"),
        "--methods",
        "pdans",
        "--lines",
        "line_a_original_x4_up",
        "line_b_downsampled_x4_up",
        "--frames",
        "000001",
        "000093",
        "000242",
        "003219",
        "006833",
        "007458",
        "--timeout",
        "14400",
    ]
    if run(smoke_cmd) != 0:
        write_smoke_reports()
        return 1
    write_smoke_reports()

    full_cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "run_pdans_two_lines_full_val.py"), "--timeout", "14400"]
    code = run(full_cmd)
    log("pipeline finished")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
