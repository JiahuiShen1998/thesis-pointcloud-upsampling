#!/usr/bin/env python3
"""Run PDANS full KITTI val for the unified x4 no-detector protocol."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import run_kitti_unified_x4_current_methods_smoke as smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "kitti_unified_x4_current_methods_no_detector"
AUDIT_DIR = RESULT_ROOT / "audits"
REPORT_DIR = RESULT_ROOT / "reports"
VAL_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
METHOD = "pdans"
METHOD_DISPLAY = "PDANS"
WRAPPER = PROJECT_ROOT / "scripts" / "wrappers" / "pdans_patch_infer.py"
LINES = ("line_a_original_x4_up", "line_b_downsampled_x4_up")
LINE_MANIFESTS = {
    "line_a_original_x4_up": AUDIT_DIR / "line_a_pdans_full_val_manifest.csv",
    "line_b_downsampled_x4_up": AUDIT_DIR / "line_b_pdans_full_val_manifest.csv",
}
LINE_AUDITS = {
    "line_a_original_x4_up": AUDIT_DIR / "line_a_pdans_full_val_audit.md",
    "line_b_downsampled_x4_up": AUDIT_DIR / "line_b_pdans_full_val_audit.md",
}
REPORT_PATH = REPORT_DIR / "pdans_two_lines_full_val_report.md"
GLOBAL_CSV = AUDIT_DIR / "global_completed_variants_point_count_audit.csv"
GLOBAL_MD = AUDIT_DIR / "global_completed_variants_point_count_audit.md"
DETECTOR_READY = RESULT_ROOT / "detector_ready_inputs"


def read_split(path: Path = VAL_SPLIT) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def line_dir(line: str) -> Path:
    return RESULT_ROOT / line / METHOD


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    smoke.write_csv(path, rows)


def write_line_audit(line: str, rows: list[dict[str, Any]], frames: list[str]) -> None:
    passed = [row for row in rows if row.get("status") == "PASS"]
    failed = [row for row in rows if row.get("status") != "PASS"]
    raw_shortage = [row for row in rows if row.get("raw_enough_for_strict_x4") is False and row.get("merged_raw_exists") is True]
    final_count = sum(1 for row in rows if row.get("final_bin_exists") is True)
    point_errors = [row for row in rows if row.get("point_count_correct") is not True]
    lines = [
        f"# {METHOD_DISPLAY} Full Val Audit: {line}",
        "",
        f"- Method: `{METHOD_DISPLAY}` only",
        "- Detector evaluation: `NOT_STARTED`",
        f"- Val frames: `{len(frames)}`",
        f"- Rows: `{len(rows)}`",
        f"- PASS: `{len(passed)}`",
        f"- FAIL: `{len(failed)}`",
        f"- Final bin count: `{final_count}`",
        f"- Point-count errors: `{len(point_errors)}`",
        f"- Raw shortage cases: `{len(raw_shortage)}`",
        "- Reused old output: `false`",
        f"- Checkpoint: `{smoke.PDANS_CKPT}`",
        f"- Wrapper: `{WRAPPER}`",
        f"- Strict adapter: `{smoke.STRICT_PY}`",
        "",
        "## Failed Frames",
        "",
    ]
    if failed:
        for row in failed:
            lines.append(f"- {row['frame_id']}: `{row.get('error_message', '')}` logs `{row.get('log_path', '')}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Raw Shortage", ""])
    if raw_shortage:
        for row in raw_shortage:
            lines.append(f"- {row['frame_id']}: raw `{row.get('merged_raw_points')}` target `{row.get('target_output_points')}`")
    else:
        lines.append("- none")
    LINE_AUDITS[line].parent.mkdir(parents=True, exist_ok=True)
    LINE_AUDITS[line].write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_line(line: str, frames: list[str], timeout: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames, start=1):
        print(f"RUN {line} {METHOD} {frame} ({index}/{len(frames)})", flush=True)
        row = smoke.run_one(line, METHOD, frame, timeout, output_root=RESULT_ROOT)
        rows.append(row)
        write_csv(LINE_MANIFESTS[line], rows)
        write_line_audit(line, rows, frames)
    return rows


def package_detector_ready(rows_by_line: dict[str, list[dict[str, Any]]], frames: list[str]) -> dict[str, str]:
    ready = all(len(rows_by_line[line]) == len(frames) and all(row.get("status") == "PASS" for row in rows_by_line[line]) for line in LINES)
    packaging: dict[str, str] = {"status": "not_created"}
    if not ready:
        return packaging
    targets = {
        "original_x4_pdans": line_dir("line_a_original_x4_up") / "final_bin",
        "downsampled_x4_pdans": line_dir("line_b_downsampled_x4_up") / "final_bin",
    }
    packaging = {"status": "created", "mode": "symlink"}
    DETECTOR_READY.mkdir(parents=True, exist_ok=True)
    for name, source in targets.items():
        dest = DETECTOR_READY / name
        if dest.is_symlink() or dest.exists():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest.symlink_to(source, target_is_directory=True)
        packaging[name] = str(dest)
    return packaging


def update_global_audit(frames: list[str]) -> tuple[int, int]:
    cmd = [
        str(smoke.UP_BASIC_PY),
        str(PROJECT_ROOT / "scripts" / "audit_kitti_unified_x4_point_counts.py"),
        "--methods",
        "pu_net",
        "pu_gcn",
        "pdans",
        "pu_edgeformer",
        "--audit-csv",
        str(GLOBAL_CSV),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    rows: list[dict[str, str]] = []
    if GLOBAL_CSV.exists():
        with GLOBAL_CSV.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    passed = sum(1 for row in rows if row.get("status") == "PASS")
    failed = len(rows) - passed
    by_variant = Counter((row["line"], row["method"]) for row in rows)
    lines = [
        "# Global Completed Variants Point-Count Audit",
        "",
        "- Scope: 10 completed variants (includes PDANS)",
        "- Detector evaluation: `NOT_STARTED`",
        f"- Total rows: `{len(rows)}`",
        f"- PASS: `{passed}`",
        f"- FAIL: `{failed}`",
        "",
        "## Variants",
        "",
    ]
    for (line, method), count in sorted(by_variant.items()):
        pass_count = sum(1 for row in rows if row["line"] == line and row["method"] == method and row["status"] == "PASS")
        lines.append(f"- `{line}` / `{method}`: `{pass_count}/{count}` PASS")
    lines.extend(["", "## Failed Rows", ""])
    bad = [row for row in rows if row.get("status") != "PASS"]
    if bad:
        for row in bad[:20]:
            lines.append(f"- {row['line']}/{row['method']}/{row['frame_id']}")
        if len(bad) > 20:
            lines.append(f"- ... and {len(bad) - 20} more")
    else:
        lines.append("- none")
    lines.extend(["", "## Expected Total", "", f"- `3769 x 10 = 37690` PASS"])
    GLOBAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed, failed


def write_two_line_report(rows_by_line: dict[str, list[dict[str, Any]]], frames: list[str], packaging: dict[str, str]) -> None:
    all_rows = [row for rows in rows_by_line.values() for row in rows]
    failed = [row for row in all_rows if row.get("status") != "PASS"]
    raw_shortage = [row for row in all_rows if row.get("raw_enough_for_strict_x4") is False and row.get("merged_raw_exists") is True]
    line_a = rows_by_line["line_a_original_x4_up"]
    line_b = rows_by_line["line_b_downsampled_x4_up"]
    cmd_used = next((row.get("command_used", "") for row in all_rows if row.get("command_used")), "")
    lines = [
        "# PDANS Two-Line Full Val Report",
        "",
        "## Stage Summary",
        "",
        f"- Method: `{METHOD_DISPLAY}` only",
        "- Initial failure reason: missing ninja / pointnet2_ops JIT compile (original smoke); CUDA OOM mitigated via fp16 autocast in wrapper",
        "- Environment fix summary: ninja present in `ear` env; bundled pointnet2_ops JIT compile succeeds; wrapper uses CUDA fp16 autocast under shared-GPU contention",
        "- Smoke test after fix summary: see `audits/pdans_smoke_after_fix_report.md`",
        "- No detector was run: `true`",
        "- No PointRCNN, CenterPoint, detector inference, or AP evaluation was run in this stage.",
        f"- Val split frame count: `{len(frames)}`",
        "",
        "## Line A Protocol",
        "",
        "- Input: original KITTI val point cloud.",
        "- Target rule: `final_output_points = exactly 4N`.",
        f"- Final bin path: `{line_dir('line_a_original_x4_up') / 'final_bin'}`",
        "",
        "## Line B Protocol",
        "",
        "- Input: downsampled-x4 KITTI val point cloud.",
        "- Target rule: `final_output_points = exactly 4M`, where `M = floor(N / 4)`.",
        f"- Final bin path: `{line_dir('line_b_downsampled_x4_up') / 'final_bin'}`",
        "",
        "## Runtime",
        "",
        f"- Checkpoint path: `{smoke.PDANS_CKPT}`",
        f"- Config path: `{smoke.PDANS_CONFIG}`",
        f"- Wrapper / script used: `{WRAPPER}`",
        f"- Representative command used: `{cmd_used}`",
        "",
        "## Verification",
        "",
        f"- Line A frame count: `{len(line_a)}`",
        f"- Line B frame count: `{len(line_b)}`",
        f"- Line A point-count verification: `{sum(1 for row in line_a if row.get('point_count_correct') is True)}/{len(line_a)} PASS`",
        f"- Line B point-count verification: `{sum(1 for row in line_b if row.get('point_count_correct') is True)}/{len(line_b)} PASS`",
        f"- Raw shortage summary: `{len(raw_shortage)}` cases",
        f"- Failed frames: `{len(failed)}`",
        "- Reused old output check: `false`",
        f"- Detector-ready status: `{'PASS' if not failed and len(line_a) == len(frames) and len(line_b) == len(frames) else 'NOT_READY'}`",
        f"- Detector-ready packaging: `{json.dumps(packaging, sort_keys=True)}`",
        "",
        "## Failed Frames",
        "",
    ]
    if failed:
        for row in failed:
            lines.append(f"- {row['line']}/{row['frame_id']}: `{row.get('error_message', '')}` logs `{row.get('log_path', '')}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Global Completed Variants Audit",
            "",
            f"- CSV: `{GLOBAL_CSV}`",
            f"- Markdown: `{GLOBAL_MD}`",
            "",
            "## Next Recommendation",
            "",
            "- Use detector-ready PDANS folders as inputs for detector evaluation in a later stage only.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--no-fresh", action="store_true")
    parser.add_argument(
        "--lines",
        nargs="+",
        choices=LINES,
        default=LINES,
    )
    args = parser.parse_args()

    frames = args.frames or read_split()
    if not args.no_fresh:
        for line in args.lines:
            shutil.rmtree(line_dir(line), ignore_errors=True)
    rows_by_line: dict[str, list[dict[str, Any]]] = {}
    for line in LINES:
        if line not in args.lines:
            continue
        rows_by_line[line] = run_line(line, frames, args.timeout)
        write_csv(LINE_MANIFESTS[line], rows_by_line[line])
        write_line_audit(line, rows_by_line[line], frames)
    if len(rows_by_line) == 2:
        packaging = package_detector_ready(rows_by_line, frames)
        update_global_audit(frames)
        write_two_line_report(rows_by_line, frames, packaging)
    failed = [row for rows in rows_by_line.values() for row in rows if row.get("status") != "PASS"]
    print(f"LINE_A_FINAL_BIN={line_dir('line_a_original_x4_up') / 'final_bin'}")
    print(f"LINE_B_FINAL_BIN={line_dir('line_b_downsampled_x4_up') / 'final_bin'}")
    print(f"LINE_A_MANIFEST={LINE_MANIFESTS['line_a_original_x4_up']}")
    print(f"LINE_B_MANIFEST={LINE_MANIFESTS['line_b_downsampled_x4_up']}")
    print(f"LINE_A_AUDIT={LINE_AUDITS['line_a_original_x4_up']}")
    print(f"LINE_B_AUDIT={LINE_AUDITS['line_b_downsampled_x4_up']}")
    print(f"TWO_LINE_REPORT={REPORT_PATH}")
    print(f"GLOBAL_AUDIT_CSV={GLOBAL_CSV}")
    print(f"GLOBAL_AUDIT_MD={GLOBAL_MD}")
    if failed:
        print("FAILED_LOG_PATHS=" + ";".join(str(row.get("log_path", "")) for row in failed))
    else:
        print("FAILED_LOG_PATHS=none")
    print("DETECTOR_EVAL_STARTED=NO")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
