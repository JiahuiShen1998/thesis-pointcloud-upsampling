#!/usr/bin/env python3
"""Run PU-Net full KITTI val for the unified x4 no-detector protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_kitti_unified_x4_current_methods_smoke as smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "kitti_unified_x4_current_methods_no_detector"
AUDIT_DIR = RESULT_ROOT / "audits"
REPORT_DIR = RESULT_ROOT / "reports"
VAL_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
METHOD = "pu_net"
METHOD_DISPLAY = "PU-Net"
LINES = {
    "line_a_original_x4_up": smoke.ORIGINAL_INPUT,
    "line_b_downsampled_x4_up": smoke.DOWNSAMPLED_X4_INPUT,
}
LINE_MANIFESTS = {
    "line_a_original_x4_up": AUDIT_DIR / "line_a_pu_net_full_val_manifest.csv",
    "line_b_downsampled_x4_up": AUDIT_DIR / "line_b_pu_net_full_val_manifest.csv",
}
LINE_AUDITS = {
    "line_a_original_x4_up": AUDIT_DIR / "line_a_pu_net_full_val_audit.md",
    "line_b_downsampled_x4_up": AUDIT_DIR / "line_b_pu_net_full_val_audit.md",
}
REPORT_PATH = REPORT_DIR / "pu_net_two_lines_full_val_report.md"
FRAME_AUDIT = AUDIT_DIR / "frame_level_point_count_audit.csv"
DETECTOR_READY = RESULT_ROOT / "detector_ready_inputs"
BATCH_WRAPPER = PROJECT_ROOT / "scripts" / "wrappers" / "tf_punet_patch_infer_many.py"
STRICT_BATCH_WRAPPER = PROJECT_ROOT / "scripts" / "wrappers" / "strict_x4_from_merged_raw_many.py"
SEED = smoke.SEED


def shell_join(cmd: list[str]) -> str:
    return " ".join(cmd)


def read_split(path: Path = VAL_SPLIT) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def line_dir(line: str) -> Path:
    return RESULT_ROOT / line / METHOD


def final_checks(final_bin: Path, input_points: np.ndarray, target: int) -> dict[str, Any]:
    if not final_bin.exists():
        return {
            "final_bin_exists": False,
            "actual_output_points": "",
            "point_count_correct": False,
            "no_nan_inf": False,
            "intensity_exists": False,
            "not_repeated_input": False,
        }
    final = smoke.read_kitti_bin(final_bin)
    tiled = np.tile(input_points, (int(math.ceil(target / max(1, len(input_points)))), 1))[:target]
    same_as_repeat = final.shape == tiled.shape and np.array_equal(final, tiled)
    return {
        "final_bin_exists": True,
        "actual_output_points": int(final.shape[0]),
        "point_count_correct": bool(final.shape == (target, 4)),
        "no_nan_inf": bool(np.isfinite(final).all()),
        "intensity_exists": bool(final.ndim == 2 and final.shape[1] == 4),
        "not_repeated_input": bool(not same_as_repeat),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "line",
        "method",
        "frame_id",
        "input_path",
        "input_points",
        "target_output_points",
        "raw_patch_output_path",
        "raw_patch_output_exists",
        "merged_raw_path",
        "merged_raw_exists",
        "merged_raw_points",
        "raw_enough_for_strict_x4",
        "final_bin_path",
        "final_bin_exists",
        "actual_output_points",
        "point_count_correct",
        "checkpoint_path",
        "checkpoint_loaded",
        "script_used",
        "command_used",
        "method_inference_confirmed",
        "reused_old_output",
        "no_nan_inf",
        "intensity_exists",
        "not_repeated_input",
        "status",
        "error_message",
        "log_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_one(line: str, frame: str, timeout: int) -> dict[str, Any]:
    base = line_dir(line)
    patch_dir = base / "raw_patch_outputs" / frame / "input_patches"
    raw_patch_dir = base / "raw_patch_outputs" / frame / "method_raw"
    merged_dir = base / "merged_raw"
    final_dir = base / "final_bin"
    log_dir = base / "logs"
    manifest_dir = base / "manifests"
    patch_meta = manifest_dir / f"{frame}_patch_metadata.json"
    merged_raw = merged_dir / f"{frame}.npy"
    merged_prov = manifest_dir / f"{frame}_merged_raw_provenance.json"
    final_bin = final_dir / f"{frame}.bin"
    final_prov = manifest_dir / f"{frame}_final_provenance.json"
    extract_log = log_dir / f"{frame}_patch_extract.log"
    infer_log = log_dir / f"{frame}_method_infer.log"
    strict_log = log_dir / f"{frame}_strict_final.log"
    input_path = LINES[line] / f"{frame}.bin"
    row: dict[str, Any] = {
        "line": line,
        "method": METHOD,
        "frame_id": frame,
        "input_path": str(input_path),
        "raw_patch_output_path": str(raw_patch_dir),
        "merged_raw_path": str(merged_raw),
        "final_bin_path": str(final_bin),
        "checkpoint_path": str(smoke.PUNET_CKPT),
        "script_used": str(PROJECT_ROOT / "scripts" / "wrappers" / "tf_punet_patch_infer.py"),
        "reused_old_output": False,
        "log_path": f"{extract_log};{infer_log};{strict_log}",
        "status": "FAIL",
    }
    try:
        for path in (patch_dir, raw_patch_dir):
            shutil.rmtree(path, ignore_errors=True)
        for path in (merged_raw, merged_prov, final_bin, final_prov, patch_meta):
            path.unlink(missing_ok=True)

        points = smoke.read_kitti_bin(input_path)
        target = points.shape[0] * 4
        row["input_points"] = int(points.shape[0])
        row["target_output_points"] = int(target)
        cmd, _cwd, checkpoint = smoke.command_for_method(METHOD, patch_meta, raw_patch_dir, merged_raw, merged_prov, line, frame)
        row["command_used"] = shell_join(cmd)
        if not smoke.method_checkpoint_exists(METHOD, checkpoint):
            raise FileNotFoundError(f"checkpoint missing: {checkpoint}")

        patch_cmd = [
            str(smoke.UP_BASIC_PY),
            str(smoke.PATCH_EXTRACTOR),
            "--input_bin",
            str(input_path),
            "--output_patch_dir",
            str(patch_dir),
            "--patch_input_points",
            "2048",
            "--patch_selection",
            "deterministic_spatial_chunk",
            "--seed",
            str(SEED + int(frame)),
            "--line",
            line,
            "--frame_id",
            frame,
            "--metadata_json",
            str(patch_meta),
        ]
        code, status = smoke.run_cmd(patch_cmd, extract_log, timeout=300)
        if code != 0:
            raise RuntimeError(f"patch extraction failed: {status}")

        code, status = smoke.run_cmd(cmd, infer_log, timeout=timeout, env={"CUDA_VISIBLE_DEVICES": "0"})
        if code != 0:
            raise RuntimeError(f"method inference failed: {status}")

        strict_cmd = [
            str(smoke.UP_BASIC_PY),
            str(smoke.STRICT_PY),
            "--merged_raw_output",
            str(merged_raw),
            "--merged_raw_provenance_json",
            str(merged_prov),
            "--input_bin",
            str(input_path),
            "--final_output",
            str(final_bin),
            "--provenance_json",
            str(final_prov),
            "--method",
            METHOD_DISPLAY,
            "--line",
            line,
            "--frame_id",
            frame,
            "--up_ratio",
            "4",
            "--seed",
            str(SEED + int(frame)),
        ]
        code, status = smoke.run_cmd(strict_cmd, strict_log, timeout=600)
        if code != 0:
            raise RuntimeError(f"strict final failed: {status}")

        merged_points = int(smoke.load_xyz(merged_raw).shape[0]) if merged_raw.exists() else ""
        merged_json = json.loads(merged_prov.read_text(encoding="utf-8")) if merged_prov.exists() else {}
        checks = final_checks(final_bin, points, target)
        row.update(checks)
        row.update(
            {
                "raw_patch_output_exists": bool(raw_patch_dir.exists() and any(raw_patch_dir.glob("*.npy"))),
                "merged_raw_exists": merged_raw.exists(),
                "merged_raw_points": merged_points,
                "raw_enough_for_strict_x4": bool(isinstance(merged_points, int) and merged_points >= target),
                "checkpoint_loaded": bool(merged_json.get("checkpoint_loaded") is True),
                "method_inference_confirmed": bool(merged_json.get("method_inference_called") is True),
                "error_message": "",
            }
        )
        row["status"] = (
            "PASS"
            if checks["point_count_correct"]
            and checks["no_nan_inf"]
            and checks["intensity_exists"]
            and checks["not_repeated_input"]
            and row["raw_patch_output_exists"]
            and row["merged_raw_exists"]
            and row["raw_enough_for_strict_x4"]
            and row["checkpoint_loaded"]
            and row["method_inference_confirmed"]
            else "FAIL"
        )
    except Exception as exc:  # noqa: BLE001
        row["error_message"] = str(exc)
        row["raw_patch_output_exists"] = bool(raw_patch_dir.exists() and any(raw_patch_dir.glob("*.npy")))
        row["merged_raw_exists"] = merged_raw.exists()
        row["merged_raw_points"] = int(smoke.load_xyz(merged_raw).shape[0]) if merged_raw.exists() else ""
        row["raw_enough_for_strict_x4"] = (
            bool(isinstance(row["merged_raw_points"], int) and row.get("target_output_points") and row["merged_raw_points"] >= row["target_output_points"])
        )
        if input_path.exists() and row.get("target_output_points"):
            row.update(final_checks(final_bin, smoke.read_kitti_bin(input_path), int(row["target_output_points"])))
        if merged_prov.exists():
            try:
                merged_json = json.loads(merged_prov.read_text(encoding="utf-8"))
                row["checkpoint_loaded"] = bool(merged_json.get("checkpoint_loaded") is True)
                row["method_inference_confirmed"] = bool(merged_json.get("method_inference_called") is True)
            except Exception:
                pass
    return row


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
        f"- Detector evaluation: `NOT_STARTED`",
        f"- Val frames: `{len(frames)}`",
        f"- Rows: `{len(rows)}`",
        f"- PASS: `{len(passed)}`",
        f"- FAIL: `{len(failed)}`",
        f"- Final bin count: `{final_count}`",
        f"- Point-count errors: `{len(point_errors)}`",
        f"- Raw shortage cases: `{len(raw_shortage)}`",
        f"- Reused old output: `false`",
        f"- Checkpoint: `{smoke.PUNET_CKPT}`",
        f"- Wrapper: `{PROJECT_ROOT / 'scripts' / 'wrappers' / 'tf_punet_patch_infer.py'}`",
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


def write_two_line_report(rows_by_line: dict[str, list[dict[str, Any]]], frames: list[str], packaging: dict[str, str]) -> None:
    all_rows = [row for rows in rows_by_line.values() for row in rows]
    failed = [row for row in all_rows if row.get("status") != "PASS"]
    raw_shortage = [row for row in all_rows if row.get("raw_enough_for_strict_x4") is False and row.get("merged_raw_exists") is True]
    line_a = rows_by_line["line_a_original_x4_up"]
    line_b = rows_by_line["line_b_downsampled_x4_up"]
    cmd_used = next((row.get("command_used", "") for row in all_rows if row.get("command_used")), "")
    lines = [
        "# PU-Net Two-Line Full Val Report",
        "",
        "## Stage Summary",
        "",
        f"- Method: `{METHOD_DISPLAY}` only",
        "- No detector was run: `true`",
        f"- Val split frame count: `{len(frames)}`",
        f"- Line A frame count: `{len(line_a)}`",
        f"- Line B frame count: `{len(line_b)}`",
        f"- Checkpoint path: `{smoke.PUNET_CKPT}`",
        f"- Wrapper / script used: `{PROJECT_ROOT / 'scripts' / 'wrappers' / 'tf_punet_patch_infer.py'}`",
        f"- Strict adapter used: `{smoke.STRICT_PY}`",
        f"- Representative command used: `{cmd_used}`",
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
        "## Verification",
        "",
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
    lines.extend(["", "## Next Recommendation", "", "- Use these two detector-ready PU-Net folders as inputs for detector evaluation in a later stage only."])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_detector_ready(rows_by_line: dict[str, list[dict[str, Any]]], frames: list[str]) -> dict[str, str]:
    ready = all(len(rows_by_line[line]) == len(frames) and all(row.get("status") == "PASS" for row in rows_by_line[line]) for line in LINES)
    packaging: dict[str, str] = {"status": "not_created"}
    if not ready:
        return packaging
    targets = {
        "original_x4_pu_net": line_dir("line_a_original_x4_up") / "final_bin",
        "downsampled_x4_pu_net": line_dir("line_b_downsampled_x4_up") / "final_bin",
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


def update_frame_audit() -> None:
    cmd = [
        str(smoke.UP_BASIC_PY),
        str(PROJECT_ROOT / "scripts" / "audit_kitti_unified_x4_point_counts.py"),
        "--methods",
        METHOD,
    ]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)


def patch_metadata_complete(patch_meta: Path) -> bool:
    patch_dir = patch_meta.parent.parent / "raw_patch_outputs" / patch_meta.name[:6] / "input_patches"
    return bool(patch_meta.exists() and patch_dir.exists() and any(patch_dir.glob("*.npy")))


def merged_method_complete(merged_raw: Path, merged_prov: Path, raw_patch_dir: Path) -> bool:
    try:
        if not merged_raw.exists() or not merged_prov.exists():
            return False
        if not (raw_patch_dir.exists() and any(raw_patch_dir.glob("*.npy"))):
            return False
        prov = json.loads(merged_prov.read_text(encoding="utf-8"))
        return bool(
            prov.get("status") == "PASS"
            and prov.get("checkpoint_loaded") is True
            and prov.get("method_inference_called") is True
            and prov.get("fallback_used") is False
            and prov.get("generic_adapter_used_before_raw") is False
        )
    except Exception:
        return False


def run_line_batch(line: str, frames: list[str], timeout: int, reuse_patches: bool, reuse_method_raw: bool) -> list[dict[str, Any]]:
    base = line_dir(line)
    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, str]] = []
    batch_log = base / "logs" / f"{line}_batch_method_infer.log"
    jobs_json = base / "manifests" / f"{line}_punet_batch_jobs.json"
    strict_batch_log = base / "logs" / f"{line}_batch_strict_final.log"
    strict_jobs_json = base / "manifests" / f"{line}_strict_batch_jobs.json"
    batch_cmd = [
        str(smoke.PUGCN_PY),
        str(BATCH_WRAPPER),
        "--code_dir",
        str(smoke.PUNET_CODE),
        "--jobs_json",
        str(jobs_json),
        "--checkpoint_dir",
        str(smoke.PUNET_CKPT),
        "--tf_ops_repo_dir",
        str(smoke.PUGCN_REPO),
        "--line",
        line,
        "--patch_input_points",
        "2048",
        "--patch_output_points",
        "8192",
        "--up_ratio",
        "4",
    ]

    for index, frame in enumerate(frames, start=1):
        print(f"EXTRACT {line} {METHOD} {frame} ({index}/{len(frames)})", flush=True)
        base = line_dir(line)
        patch_dir = base / "raw_patch_outputs" / frame / "input_patches"
        raw_patch_dir = base / "raw_patch_outputs" / frame / "method_raw"
        merged_raw = base / "merged_raw" / f"{frame}.npy"
        manifest_dir = base / "manifests"
        patch_meta = manifest_dir / f"{frame}_patch_metadata.json"
        merged_prov = manifest_dir / f"{frame}_merged_raw_provenance.json"
        final_bin = base / "final_bin" / f"{frame}.bin"
        final_prov = manifest_dir / f"{frame}_final_provenance.json"
        extract_log = base / "logs" / f"{frame}_patch_extract.log"
        input_path = LINES[line] / f"{frame}.bin"
        reuse_this_patch = reuse_patches and patch_metadata_complete(patch_meta)
        reuse_this_method = reuse_method_raw and merged_method_complete(merged_raw, merged_prov, raw_patch_dir)
        if reuse_this_patch:
            if not reuse_this_method:
                shutil.rmtree(raw_patch_dir, ignore_errors=True)
        else:
            shutil.rmtree(patch_dir, ignore_errors=True)
            shutil.rmtree(raw_patch_dir, ignore_errors=True)
        for path in (merged_raw, merged_prov, final_bin, final_prov, patch_meta):
            if path in (merged_raw, merged_prov) and reuse_this_method:
                continue
            if path != patch_meta or not reuse_this_patch:
                path.unlink(missing_ok=True)

        row: dict[str, Any] = {
            "line": line,
            "method": METHOD,
            "frame_id": frame,
            "input_path": str(input_path),
            "raw_patch_output_path": str(raw_patch_dir),
            "merged_raw_path": str(merged_raw),
            "final_bin_path": str(final_bin),
            "checkpoint_path": str(smoke.PUNET_CKPT),
            "script_used": str(BATCH_WRAPPER),
            "command_used": shell_join(batch_cmd),
            "reused_old_output": False,
            "log_path": f"{extract_log};{batch_log};{strict_batch_log}",
            "status": "FAIL",
        }
        try:
            points = smoke.read_kitti_bin(input_path)
            target = points.shape[0] * 4
            row["input_points"] = int(points.shape[0])
            row["target_output_points"] = int(target)
            if not smoke.method_checkpoint_exists(METHOD, smoke.PUNET_CKPT):
                raise FileNotFoundError(f"checkpoint missing: {smoke.PUNET_CKPT}")
            if not reuse_this_patch:
                patch_cmd = [
                    str(smoke.UP_BASIC_PY),
                    str(smoke.PATCH_EXTRACTOR),
                    "--input_bin",
                    str(input_path),
                    "--output_patch_dir",
                    str(patch_dir),
                    "--patch_input_points",
                    "2048",
                    "--patch_selection",
                    "deterministic_spatial_chunk",
                    "--seed",
                    str(SEED + int(frame)),
                    "--line",
                    line,
                    "--frame_id",
                    frame,
                    "--metadata_json",
                    str(patch_meta),
                ]
                code, status = smoke.run_cmd(patch_cmd, extract_log, timeout=300)
                if code != 0:
                    raise RuntimeError(f"patch extraction failed: {status}")
            if not reuse_this_method:
                jobs.append(
                    {
                        "frame_id": frame,
                        "patch_metadata_json": str(patch_meta),
                        "raw_patch_output_dir": str(raw_patch_dir),
                        "merged_raw_output": str(merged_raw),
                        "merged_raw_provenance_json": str(merged_prov),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            row["error_message"] = str(exc)
            row["raw_patch_output_exists"] = bool(raw_patch_dir.exists() and any(raw_patch_dir.glob("*.npy")))
            row["merged_raw_exists"] = merged_raw.exists()
            row["final_bin_exists"] = final_bin.exists()
        rows.append(row)
        write_csv(LINE_MANIFESTS[line], rows)
        write_line_audit(line, rows, frames)

    jobs_json.parent.mkdir(parents=True, exist_ok=True)
    jobs_json.write_text(json.dumps(jobs, indent=2, sort_keys=True), encoding="utf-8")
    if jobs:
        print(f"BATCH_INFER {line} {METHOD} jobs={len(jobs)}", flush=True)
        code, status = smoke.run_cmd(batch_cmd, batch_log, timeout=max(timeout, timeout * len(jobs)), env={"CUDA_VISIBLE_DEVICES": "0"})
        if code != 0:
            print(f"BATCH_INFER_STATUS {line} {status}", flush=True)

    strict_jobs = []
    for row in rows:
        if row.get("input_points") in ("", None):
            continue
        frame = row["frame_id"]
        strict_jobs.append(
            {
                "method": METHOD_DISPLAY,
                "line": line,
                "frame_id": frame,
                "input_bin": row["input_path"],
                "merged_raw_output": row["merged_raw_path"],
                "merged_raw_provenance_json": str(line_dir(line) / "manifests" / f"{frame}_merged_raw_provenance.json"),
                "final_output": row["final_bin_path"],
                "provenance_json": str(line_dir(line) / "manifests" / f"{frame}_final_provenance.json"),
                "seed": SEED + int(frame),
            }
        )
    strict_jobs_json.parent.mkdir(parents=True, exist_ok=True)
    strict_jobs_json.write_text(json.dumps(strict_jobs, indent=2, sort_keys=True), encoding="utf-8")
    if strict_jobs:
        strict_batch_cmd = [
            str(smoke.UP_BASIC_PY),
            str(STRICT_BATCH_WRAPPER),
            "--jobs_json",
            str(strict_jobs_json),
            "--up_ratio",
            "4",
        ]
        print(f"BATCH_STRICT {line} {METHOD} jobs={len(strict_jobs)}", flush=True)
        code, status = smoke.run_cmd(strict_batch_cmd, strict_batch_log, timeout=max(timeout, timeout * len(strict_jobs)))
        if code != 0:
            print(f"BATCH_STRICT_STATUS {line} {status}", flush=True)

    row_by_frame = {row["frame_id"]: row for row in rows}
    for index, frame in enumerate(frames, start=1):
        print(f"STRICT_AUDIT {line} {METHOD} {frame} ({index}/{len(frames)})", flush=True)
        row = row_by_frame[frame]
        input_path = Path(row["input_path"])
        raw_patch_dir = Path(row["raw_patch_output_path"])
        merged_raw = Path(row["merged_raw_path"])
        merged_prov = line_dir(line) / "manifests" / f"{frame}_merged_raw_provenance.json"
        final_bin = Path(row["final_bin_path"])
        final_prov = line_dir(line) / "manifests" / f"{frame}_final_provenance.json"
        try:
            if row.get("input_points") in ("", None):
                raise RuntimeError(row.get("error_message") or "input preparation failed")
            points = smoke.read_kitti_bin(input_path)
            target = int(row["target_output_points"])
            merged_json = json.loads(merged_prov.read_text(encoding="utf-8")) if merged_prov.exists() else {}
            if merged_json.get("status") != "PASS":
                raise RuntimeError(f"method inference failed: {merged_json.get('notes', ['unknown'])[0] if merged_json else 'missing provenance'}")
            final_json = json.loads(final_prov.read_text(encoding="utf-8")) if final_prov.exists() else {}
            if final_json.get("status") != "PASS":
                raise RuntimeError(f"strict final failed: {final_json.get('notes', ['missing provenance'])[0] if final_json else 'missing provenance'}")
            merged_points = int(smoke.load_xyz(merged_raw).shape[0]) if merged_raw.exists() else ""
            checks = final_checks(final_bin, points, target)
            row.update(checks)
            row.update(
                {
                    "raw_patch_output_exists": bool(raw_patch_dir.exists() and any(raw_patch_dir.glob("*.npy"))),
                    "merged_raw_exists": merged_raw.exists(),
                    "merged_raw_points": merged_points,
                    "raw_enough_for_strict_x4": bool(isinstance(merged_points, int) and merged_points >= target),
                    "checkpoint_loaded": bool(merged_json.get("checkpoint_loaded") is True),
                    "method_inference_confirmed": bool(merged_json.get("method_inference_called") is True),
                    "error_message": "",
                }
            )
            row["status"] = (
                "PASS"
                if checks["point_count_correct"]
                and checks["no_nan_inf"]
                and checks["intensity_exists"]
                and checks["not_repeated_input"]
                and row["raw_patch_output_exists"]
                and row["merged_raw_exists"]
                and row["raw_enough_for_strict_x4"]
                and row["checkpoint_loaded"]
                and row["method_inference_confirmed"]
                else "FAIL"
            )
        except Exception as exc:  # noqa: BLE001
            row["error_message"] = str(exc)
            row["raw_patch_output_exists"] = bool(raw_patch_dir.exists() and any(raw_patch_dir.glob("*.npy")))
            row["merged_raw_exists"] = merged_raw.exists()
            row["merged_raw_points"] = int(smoke.load_xyz(merged_raw).shape[0]) if merged_raw.exists() else ""
            row["raw_enough_for_strict_x4"] = (
                bool(isinstance(row["merged_raw_points"], int) and row.get("target_output_points") and row["merged_raw_points"] >= row["target_output_points"])
            )
            if input_path.exists() and row.get("target_output_points"):
                row.update(final_checks(final_bin, smoke.read_kitti_bin(input_path), int(row["target_output_points"])))
            if merged_prov.exists():
                try:
                    merged_json = json.loads(merged_prov.read_text(encoding="utf-8"))
                    row["checkpoint_loaded"] = bool(merged_json.get("checkpoint_loaded") is True)
                    row["method_inference_confirmed"] = bool(merged_json.get("method_inference_called") is True)
                except Exception:
                    pass
        write_csv(LINE_MANIFESTS[line], rows)
        write_line_audit(line, rows, frames)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--no-fresh", action="store_true", help="Do not clear PU-Net full-val output directories before running.")
    parser.add_argument("--reuse-patches", action="store_true", help="Reuse existing current-run input patches while regenerating raw, merged, and final outputs.")
    parser.add_argument("--reuse-current-method-raw", action="store_true", help="Reuse current-run PASS merged raw/method outputs and regenerate final outputs/audits.")
    args = parser.parse_args()

    frames = args.frames or read_split()
    if not args.no_fresh:
        for line in LINES:
            shutil.rmtree(line_dir(line), ignore_errors=True)
    rows_by_line: dict[str, list[dict[str, Any]]] = {}
    for line in ("line_a_original_x4_up", "line_b_downsampled_x4_up"):
        rows = run_line_batch(line, frames, args.timeout, args.reuse_patches, args.reuse_current_method_raw)
        rows_by_line[line] = rows
        write_csv(LINE_MANIFESTS[line], rows)
        write_line_audit(line, rows, frames)
    update_frame_audit()
    packaging = package_detector_ready(rows_by_line, frames)
    write_two_line_report(rows_by_line, frames, packaging)
    failed = [row for rows in rows_by_line.values() for row in rows if row.get("status") != "PASS"]
    print(f"LINE_A_FINAL_BIN={line_dir('line_a_original_x4_up') / 'final_bin'}")
    print(f"LINE_B_FINAL_BIN={line_dir('line_b_downsampled_x4_up') / 'final_bin'}")
    print(f"LINE_A_MANIFEST={LINE_MANIFESTS['line_a_original_x4_up']}")
    print(f"LINE_B_MANIFEST={LINE_MANIFESTS['line_b_downsampled_x4_up']}")
    print(f"LINE_A_AUDIT={LINE_AUDITS['line_a_original_x4_up']}")
    print(f"LINE_B_AUDIT={LINE_AUDITS['line_b_downsampled_x4_up']}")
    print(f"FRAME_LEVEL_POINT_COUNT_AUDIT={FRAME_AUDIT}")
    print(f"TWO_LINE_REPORT={REPORT_PATH}")
    print(f"DETECTOR_READY_ORIGINAL={DETECTOR_READY / 'original_x4_pu_net'}")
    print(f"DETECTOR_READY_DOWNSAMPLED={DETECTOR_READY / 'downsampled_x4_pu_net'}")
    if failed:
        print("FAILED_LOG_PATHS=" + ";".join(str(row.get("log_path", "")) for row in failed))
    else:
        print("FAILED_LOG_PATHS=none")
    print("DETECTOR_EVAL_STARTED=NO")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
