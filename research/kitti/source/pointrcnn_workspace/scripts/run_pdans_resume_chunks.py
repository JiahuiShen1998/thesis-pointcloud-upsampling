#!/usr/bin/env python3
"""Resume PDANS unified x4 full val on missing/invalid frames only."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_kitti_unified_x4_current_methods_smoke as smoke
import run_pdans_two_lines_full_val as full_val


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "kitti_unified_x4_current_methods_no_detector"
AUDIT_DIR = RESULT_ROOT / "audits"
ORIGINAL_INPUT = smoke.ORIGINAL_INPUT
DOWNSAMPLED_INPUT = smoke.DOWNSAMPLED_X4_INPUT
LINE_A = "line_a_original_x4_up"
LINE_B = "line_b_downsampled_x4_up"


def read_split() -> list[str]:
    return full_val.read_split()


def input_path(line: str, frame_id: str) -> Path:
    if line == LINE_A:
        return ORIGINAL_INPUT / f"{frame_id}.bin"
    return DOWNSAMPLED_INPUT / f"{frame_id}.bin"


def target_points(line: str, frame_id: str) -> int:
    path = input_path(line, frame_id)
    raw = np.fromfile(path, dtype=np.float32)
    n = raw.size // 4
    if line == LINE_A:
        return 4 * n
    return 4 * (n // 4) * (4 // 4)  # 4M where M = floor(N/4) for original N


def original_n_for_line_b(frame_id: str) -> int:
    path = ORIGINAL_INPUT / f"{frame_id}.bin"
    raw = np.fromfile(path, dtype=np.float32)
    return raw.size // 4


def validate_final_bin(line: str, frame_id: str, val_set: set[str]) -> tuple[bool, str, dict[str, Any]]:
    final_bin = RESULT_ROOT / line / "pdans" / "final_bin" / f"{frame_id}.bin"
    input_bin = input_path(line, frame_id)
    row: dict[str, Any] = {
        "frame_id": frame_id,
        "line": line,
        "final_bin_path": str(final_bin),
        "input_path": str(input_bin),
    }
    if frame_id not in val_set:
        return False, "not_in_val_split", row
    if not input_bin.exists():
        return False, "missing_input_bin", row
    if not final_bin.exists():
        return False, "missing_final_bin", row
    try:
        input_pts = smoke.read_kitti_bin(input_bin)
        if line == LINE_A:
            target = int(input_pts.shape[0] * 4)
        else:
            orig_n = original_n_for_line_b(frame_id)
            target = 4 * (orig_n // 4)
        row["input_points"] = int(input_pts.shape[0])
        row["target_output_points"] = int(target)
        final = smoke.read_kitti_bin(final_bin)
        row["actual_output_points"] = int(final.shape[0])
        if final.dtype != np.float32:
            return False, "dtype_not_float32", row
        if final.ndim != 2 or final.shape[1] != 4:
            return False, "shape_not_4n_x4", row
        if final.shape[0] != target:
            return False, "point_count_mismatch", row
        if not np.isfinite(final).all():
            return False, "nan_or_inf", row
        tiled = np.tile(input_pts, (int(math.ceil(target / max(1, len(input_pts)))), 1))[:target]
        if final.shape == tiled.shape and np.array_equal(final, tiled):
            return False, "repeated_input", row
        row["status"] = "VALID"
        return True, "valid", row
    except Exception as exc:  # noqa: BLE001
        row["error"] = str(exc)
        return False, f"validation_error:{exc}", row


def scan_line(line: str, frames: list[str]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    val_set = set(frames)
    valid_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    invalid: list[str] = []
    for frame_id in frames:
        ok, reason, row = validate_final_bin(line, frame_id, val_set)
        row["reason"] = reason
        if ok:
            valid_rows.append(row)
        elif reason == "missing_final_bin":
            missing.append(frame_id)
        else:
            invalid.append(frame_id)
    return valid_rows, missing, invalid


def write_resume_lists(line: str, valid_rows: list[dict[str, Any]], missing: list[str], invalid: list[str]) -> None:
    prefix = "pdans_line_a_resume" if line == LINE_A else "pdans_line_b_resume"
    valid_csv = AUDIT_DIR / f"{prefix}_existing_valid_frames.csv"
    missing_txt = AUDIT_DIR / f"{prefix}_missing_frames.txt"
    invalid_txt = AUDIT_DIR / f"{prefix}_invalid_frames.txt"
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["frame_id", "line", "input_path", "input_points", "target_output_points", "actual_output_points", "final_bin_path", "status"]
    with valid_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in valid_rows:
            writer.writerow(row)
    missing_txt.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    invalid_txt.write_text("\n".join(invalid) + ("\n" if invalid else ""), encoding="utf-8")
    print(f"VALID_FRAMES={len(valid_rows)}")
    print(f"MISSING_FRAMES={len(missing)}")
    print(f"INVALID_FRAMES={len(invalid)}")
    print(f"VALID_CSV={valid_csv}")
    print(f"MISSING_TXT={missing_txt}")
    print(f"INVALID_TXT={invalid_txt}")


def merge_manifest(line: str, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_path = full_val.LINE_MANIFESTS[line]
    existing = smoke.read_existing_manifest(manifest_path)
    by_frame = {row["frame_id"]: row for row in existing if row.get("frame_id")}
    for row in new_rows:
        by_frame[row["frame_id"]] = row
    frames = read_split()
    merged = [by_frame[fid] for fid in frames if fid in by_frame]
    smoke.write_csv(manifest_path, merged)
    full_val.write_line_audit(line, merged, frames)
    return merged


def run_chunks(
    line: str,
    todo: list[str],
    timeout: int,
    chunk_size: int,
    log_path: Path,
) -> list[dict[str, Any]]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_rows: list[dict[str, Any]] = []
    total = len(todo)
    for start in range(0, total, chunk_size):
        chunk = todo[start : start + chunk_size]
        chunk_id = start // chunk_size + 1
        msg = f"CHUNK {chunk_id} line={line} frames={len(chunk)} range={chunk[0]}..{chunk[-1]}"
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        chunk_rows: list[dict[str, Any]] = []
        for index, frame in enumerate(chunk, start=1):
            global_idx = start + index
            print(f"RUN {line} pdans {frame} ({global_idx}/{total})", flush=True)
            row = smoke.run_one(line, "pdans", frame, timeout, output_root=RESULT_ROOT)
            chunk_rows.append(row)
            new_rows.append(row)
            merge_manifest(line, chunk_rows)
        chunk_manifest = AUDIT_DIR / f"pdans_resume_{line}_chunk_{chunk_id:04d}_manifest.json"
        chunk_manifest.write_text(json.dumps(chunk_rows, indent=2), encoding="utf-8")
        with log_path.open("a", encoding="utf-8") as handle:
            passed = sum(1 for row in chunk_rows if row.get("status") == "PASS")
            handle.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CHUNK {chunk_id} done pass={passed}/{len(chunk_rows)} manifest={chunk_manifest}\n"
            )
    return new_rows


def audit_line_complete(line: str, frames: list[str]) -> int:
    merged = smoke.read_existing_manifest(full_val.LINE_MANIFESTS[line])
    passed = sum(1 for row in merged if row.get("status") == "PASS")
    failed = [row for row in merged if row.get("status") != "PASS"]
    final_count = sum(1 for fid in frames if (RESULT_ROOT / line / "pdans" / "final_bin" / f"{fid}.bin").exists())
    print(f"LINE={line}")
    print(f"MANIFEST_ROWS={len(merged)}")
    print(f"PASS={passed}")
    print(f"FAIL={len(failed)}")
    print(f"FINAL_BIN_COUNT={final_count}")
    print(f"AUDIT_MD={full_val.LINE_AUDITS[line]}")
    print("DETECTOR_EVAL_STARTED=NO")
    return 0 if passed == len(frames) and final_count == len(frames) and not failed else 1


def ensure_pdans_path() -> None:
    ear_bin = str(smoke.PDANS_PY.parent)
    os.environ["PATH"] = ear_bin + (":" + os.environ["PATH"] if os.environ.get("PATH") else "")


def main() -> int:
    ensure_pdans_path()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line", choices=[LINE_A, LINE_B], required=True)
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames processed this invocation")
    args = parser.parse_args()

    frames = read_split()
    valid_rows, missing, invalid = scan_line(args.line, frames)
    write_resume_lists(args.line, valid_rows, missing, invalid)
    if args.scan_only:
        return 0

    todo = missing + invalid
    if args.max_frames is not None:
        todo = todo[: args.max_frames]
    if not todo:
        print("NOTHING_TO_RESUME")
        return audit_line_complete(args.line, frames)

    log_path = AUDIT_DIR / f"pdans_resume_{args.line}.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] resume start line={args.line} todo={len(todo)} chunk_size={args.chunk_size} timeout={args.timeout}\n"
        )
    run_chunks(args.line, todo, args.timeout, args.chunk_size, log_path)
    return audit_line_complete(args.line, frames)


if __name__ == "__main__":
    raise SystemExit(main())
