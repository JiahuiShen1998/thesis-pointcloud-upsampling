#!/usr/bin/env python3
"""Run current-method KITTI unified x4 smoke tests without detector evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "kitti_unified_x4_current_methods_no_detector"
SMOKE_ROOT = RESULT_ROOT / "smoke_tests"
AUDIT_DIR = RESULT_ROOT / "audits"
VAL_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
TRAINING = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
ORIGINAL_INPUT = TRAINING / "velodyne_original_val"
DOWNSAMPLED_X4_INPUT = RESULT_ROOT / "downsampled_x4" / "velodyne_downsampled_x4_val"
STRICT_PY = PROJECT_ROOT / "scripts" / "wrappers" / "strict_x4_from_merged_raw.py"
PATCH_EXTRACTOR = PROJECT_ROOT / "scripts" / "wrappers" / "kitti_patch_extractor.py"

UP_BASIC_PY = Path("/home/ra87racy/miniconda3/envs/upsampling_basic/bin/python")
PUGCN_PY = Path("/home/ra87racy/miniconda3/envs/pugcn/bin/python")
PDANS_PY = Path("/home/ra87racy/miniconda3/envs/ear/bin/python")
PUEDGE_PY = Path("/home/ra87racy/miniconda3/envs/puedgeformer/bin/python")

PUNET_CODE = Path("/home/ra87racy/projects/upsampling/PU-Net/code")
PUNET_CKPT = Path("/home/ra87racy/projects/upsampling/PU-Net/model/generator2_new6")
PUGCN_REPO = PROJECT_ROOT / "external" / "PU-GCN"
PUGCN_CKPT = PUGCN_REPO / "pretrained" / "pu1k-pugcn"
PDANS_CKPT = PROJECT_ROOT / "external" / "PDANS" / "checkpoints" / "PUGAN_PDANS.pkl"
PDANS_CONFIG = PROJECT_ROOT / "external" / "PDANS" / "pointnet2" / "exp_configs" / "PUGAN.json"
PUEDGE_REPO = (
    PROJECT_ROOT
    / "external"
    / "reproducibility_check"
    / "puedgeformer_minimal_feasibility"
    / "PU-EdgeFormer_ops_reuse"
)
PUEDGE_CKPT = (
    PUEDGE_REPO
    / "log"
    / "pu-edgeformer_full_pu1k_20260625_2314"
    / "pu1k_edgetransformer_nodeshuffle_inception_n2_C32_d2_k20_wogan_worepulse_wouniform_woreg_sample-FPS_lr0.001_cd0.0_SRx4_moreupx0_seed2_20260625-231457_b390d4d8-20ce-4963-a9e1-9474eea304e1"
)

PREFERRED_FRAMES = ["000001", "000093", "000242", "003219", "006833", "007458"]
METHODS = ["pu_net", "pu_gcn", "pdans", "pu_edgeformer"]
LINES = {
    "line_a_original_x4_up": {
        "display": "Line A",
        "input_dir": ORIGINAL_INPUT,
    },
    "line_b_downsampled_x4_up": {
        "display": "Line B",
        "input_dir": DOWNSAMPLED_X4_INPUT,
    },
}
METHOD_DISPLAY = {
    "pu_net": "PU-Net",
    "pu_gcn": "PU-GCN",
    "pdans": "PDANS",
    "pu_edgeformer": "PU-EdgeFormer",
}
SEED = 20260702


def read_split(path: Path = VAL_SPLIT) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def choose_smoke_frames(min_count: int = 5) -> list[str]:
    split = read_split()
    available = set(split)
    frames = [frame for frame in PREFERRED_FRAMES if frame in available]
    for frame in split:
        if len(frames) >= min_count:
            break
        if frame not in frames:
            frames.append(frame)
    return frames


def read_kitti_bin(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path} is not KITTI XYZI float32")
    points = raw.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN or Inf")
    return points


def load_xyz(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path)
    else:
        raw = np.fromfile(path, dtype=np.float32)
        arr = raw.reshape(-1, 3 if raw.size % 3 == 0 else 4)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{path} does not contain Nx3/NxC raw xyz")
    return arr[:, :3]


def shell_join(cmd: list[str]) -> str:
    return " ".join(cmd)


def run_cmd(cmd: list[str], log_path: Path, cwd: Path = PROJECT_ROOT, timeout: int = 3600, env: dict[str, str] | None = None) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    full_env = os.environ.copy()
    pdans_bin = str(PDANS_PY.parent)
    path_prefix = pdans_bin + (":" + full_env["PATH"] if full_env.get("PATH") else "")
    full_env["PATH"] = path_prefix
    if env:
        full_env.update(env)
        if "PATH" in env:
            full_env["PATH"] = pdans_bin + ":" + env["PATH"]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shell_join(cmd) + "\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                env=full_env,
            )
            code = proc.returncode
            status = "PASS" if code == 0 else f"FAIL_EXIT_{code}"
        except subprocess.TimeoutExpired:
            code = 124
            status = f"FAIL_TIMEOUT_{timeout}s"
            log.write(f"\nTIMEOUT after {timeout}s\n")
        log.write(f"\nCOMMAND_STATUS={status}\nRUNTIME_SEC={time.time() - started:.3f}\n")
        log.write("DETECTOR_EVAL_STARTED=NO\n")
    return code, status


def command_for_method(method: str, patch_meta: Path, raw_patch_dir: Path, merged_raw: Path, merged_prov: Path, line: str, frame: str) -> tuple[list[str], Path, Path]:
    display = METHOD_DISPLAY[method]
    if method == "pu_net":
        return (
            [
                str(PUGCN_PY),
                str(PROJECT_ROOT / "scripts" / "wrappers" / "tf_punet_patch_infer.py"),
                "--code_dir",
                str(PUNET_CODE),
                "--patch_metadata_json",
                str(patch_meta),
                "--raw_patch_output_dir",
                str(raw_patch_dir),
                "--merged_raw_output",
                str(merged_raw),
                "--merged_raw_provenance_json",
                str(merged_prov),
                "--checkpoint_dir",
                str(PUNET_CKPT),
                "--tf_ops_repo_dir",
                str(PUGCN_REPO),
                "--line",
                line,
                "--frame_id",
                frame,
                "--patch_input_points",
                "2048",
                "--patch_output_points",
                "8192",
                "--up_ratio",
                "4",
            ],
            PROJECT_ROOT,
            PUNET_CKPT,
        )
    if method == "pu_gcn":
        return (
            [
                str(PUGCN_PY),
                str(PROJECT_ROOT / "scripts" / "wrappers" / "tf_pugcn_family_patch_infer.py"),
                "--repo_dir",
                str(PUGCN_REPO),
                "--patch_metadata_json",
                str(patch_meta),
                "--raw_patch_output_dir",
                str(raw_patch_dir),
                "--merged_raw_output",
                str(merged_raw),
                "--merged_raw_provenance_json",
                str(merged_prov),
                "--method",
                display,
                "--model",
                "pugcn",
                "--checkpoint_dir",
                str(PUGCN_CKPT),
                "--line",
                line,
                "--frame_id",
                frame,
                "--patch_input_points",
                "2048",
                "--patch_output_points",
                "8192",
                "--up_ratio",
                "4",
                "--seed",
                str(SEED),
            ],
            PROJECT_ROOT,
            PUGCN_CKPT,
        )
    if method == "pdans":
        return (
            [
                str(PDANS_PY),
                str(PROJECT_ROOT / "scripts" / "wrappers" / "pdans_patch_infer.py"),
                "--patch_metadata_json",
                str(patch_meta),
                "--raw_patch_output_dir",
                str(raw_patch_dir),
                "--merged_raw_output",
                str(merged_raw),
                "--merged_raw_provenance_json",
                str(merged_prov),
                "--checkpoint",
                str(PDANS_CKPT),
                "--config",
                str(PDANS_CONFIG),
                "--line",
                line,
                "--frame_id",
                frame,
                "--patch_input_points",
                "2048",
                "--patch_output_points",
                "8192",
                "--up_ratio",
                "4",
                "--batch_size",
                "1",
                "--step",
                "30",
                "--gamma",
                "0.5",
                "--device",
                "cuda",
                "--seed",
                str(SEED),
            ],
            PROJECT_ROOT,
            PDANS_CKPT,
        )
    if method == "pu_edgeformer":
        return (
            [
                str(PUEDGE_PY),
                str(PROJECT_ROOT / "scripts" / "wrappers" / "tf_pugcn_family_patch_infer.py"),
                "--repo_dir",
                str(PUEDGE_REPO),
                "--patch_metadata_json",
                str(patch_meta),
                "--raw_patch_output_dir",
                str(raw_patch_dir),
                "--merged_raw_output",
                str(merged_raw),
                "--merged_raw_provenance_json",
                str(merged_prov),
                "--method",
                display,
                "--model",
                "edgetransformer",
                "--checkpoint_dir",
                str(PUEDGE_CKPT),
                "--line",
                line,
                "--frame_id",
                frame,
                "--patch_input_points",
                "2048",
                "--patch_output_points",
                "8192",
                "--up_ratio",
                "4",
                "--seed",
                str(SEED),
            ],
            PROJECT_ROOT,
            PUEDGE_CKPT,
        )
    raise ValueError(f"unknown method: {method}")


def method_checkpoint_exists(method: str, checkpoint: Path) -> bool:
    if method in {"pu_net", "pu_gcn", "pu_edgeformer"}:
        return (checkpoint / "checkpoint").exists()
    return checkpoint.exists()


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


def read_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    final = read_kitti_bin(final_bin)
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


def run_one(line: str, method: str, frame: str, timeout: int, output_root: Path | None = None) -> dict[str, Any]:
    base_root = output_root or SMOKE_ROOT
    line_dir = base_root / line / method
    patch_dir = line_dir / "raw_patch_outputs" / frame / "input_patches"
    raw_patch_dir = line_dir / "raw_patch_outputs" / frame / "method_raw"
    merged_dir = line_dir / "merged_raw"
    final_dir = line_dir / "final_bin"
    log_dir = line_dir / "logs"
    manifest_dir = line_dir / "manifests"
    patch_meta = manifest_dir / f"{frame}_patch_metadata.json"
    merged_raw = merged_dir / f"{frame}.npy"
    merged_prov = manifest_dir / f"{frame}_merged_raw_provenance.json"
    final_bin = final_dir / f"{frame}.bin"
    final_prov = manifest_dir / f"{frame}_final_provenance.json"
    extract_log = log_dir / f"{frame}_patch_extract.log"
    infer_log = log_dir / f"{frame}_method_infer.log"
    strict_log = log_dir / f"{frame}_strict_final.log"

    input_path = LINES[line]["input_dir"] / f"{frame}.bin"
    display = METHOD_DISPLAY[method]
    row: dict[str, Any] = {
        "line": line,
        "method": method,
        "frame_id": frame,
        "input_path": str(input_path),
        "raw_patch_output_path": str(raw_patch_dir),
        "merged_raw_path": str(merged_raw),
        "final_bin_path": str(final_bin),
        "script_used": "",
        "command_used": "",
        "reused_old_output": False,
        "log_path": str(infer_log),
        "status": "FAIL",
    }
    try:
        points = read_kitti_bin(input_path)
        target = points.shape[0] * 4
        row["input_points"] = int(points.shape[0])
        row["target_output_points"] = int(target)
        cmd, _cwd, checkpoint = command_for_method(method, patch_meta, raw_patch_dir, merged_raw, merged_prov, line, frame)
        row["checkpoint_path"] = str(checkpoint)
        row["script_used"] = cmd[1]
        row["command_used"] = shell_join(cmd)
        if not method_checkpoint_exists(method, checkpoint):
            raise FileNotFoundError(f"checkpoint missing: {checkpoint}")

        patch_cmd = [
            str(UP_BASIC_PY),
            str(PATCH_EXTRACTOR),
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
        code, status = run_cmd(patch_cmd, extract_log, timeout=300)
        if code != 0:
            raise RuntimeError(f"patch extraction failed: {status}")

        code, status = run_cmd(cmd, infer_log, timeout=timeout, env={"CUDA_VISIBLE_DEVICES": "0"})
        if code != 0:
            raise RuntimeError(f"method inference failed: {status}")

        strict_cmd = [
            str(UP_BASIC_PY),
            str(STRICT_PY),
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
            display,
            "--line",
            line,
            "--frame_id",
            frame,
            "--up_ratio",
            "4",
            "--seed",
            str(SEED + int(frame)),
        ]
        code, status = run_cmd(strict_cmd, strict_log, timeout=600)
        if code != 0:
            raise RuntimeError(f"strict final failed: {status}")

        merged_points = int(load_xyz(merged_raw).shape[0]) if merged_raw.exists() else ""
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
                "status": "PASS"
                if checks["point_count_correct"]
                and checks["no_nan_inf"]
                and checks["intensity_exists"]
                and checks["not_repeated_input"]
                and merged_raw.exists()
                and isinstance(merged_points, int)
                and merged_points >= target
                and merged_json.get("checkpoint_loaded") is True
                and merged_json.get("method_inference_called") is True
                else "FAIL",
                "error_message": "",
                "log_path": f"{extract_log};{infer_log};{strict_log}",
            }
        )
    except Exception as exc:  # noqa: BLE001
        row["error_message"] = str(exc)
        row["raw_patch_output_exists"] = bool(raw_patch_dir.exists() and any(raw_patch_dir.glob("*.npy")))
        row["merged_raw_exists"] = merged_raw.exists()
        row["merged_raw_points"] = int(load_xyz(merged_raw).shape[0]) if merged_raw.exists() else ""
        row["raw_enough_for_strict_x4"] = (
            bool(isinstance(row["merged_raw_points"], int) and row.get("target_output_points") and row["merged_raw_points"] >= row["target_output_points"])
        )
        row.update(final_checks(final_bin, read_kitti_bin(input_path), int(row.get("target_output_points") or 0)) if input_path.exists() and row.get("target_output_points") else {})
        if merged_prov.exists():
            try:
                merged_json = json.loads(merged_prov.read_text(encoding="utf-8"))
                row["checkpoint_loaded"] = bool(merged_json.get("checkpoint_loaded") is True)
                row["method_inference_confirmed"] = bool(merged_json.get("method_inference_called") is True)
            except Exception:
                pass
    return row


def write_report(path: Path, rows: list[dict[str, Any]], frames: list[str]) -> None:
    total = len(rows)
    passed = [row for row in rows if row.get("status") == "PASS"]
    failed = [row for row in rows if row.get("status") != "PASS"]
    raw_shortage = [row for row in rows if row.get("raw_enough_for_strict_x4") is False and row.get("merged_raw_exists") is True]
    by_method = {method: [row for row in rows if row["method"] == method] for method in METHODS}
    by_line = {line: [row for row in rows if row["line"] == line] for line in LINES}
    ready = []
    for line in LINES:
        for method in METHODS:
            subset = [row for row in rows if row["line"] == line and row["method"] == method]
            if subset and all(row.get("status") == "PASS" for row in subset):
                ready.append(f"{line}/{method}")
    lines = [
        "# KITTI Unified x4 Current Methods Smoke Test Report",
        "",
        f"- Tested frames: `{', '.join(frames)}`",
        f"- Tested variants: `{len(LINES) * len(METHODS)}`",
        f"- Total frame-method-line rows: `{total}`",
        f"- PASS rows: `{len(passed)}`",
        f"- FAIL rows: `{len(failed)}`",
        f"- Detector evaluation: `NOT_STARTED`",
        f"- Reused old output: `false`",
        "",
        "## Per Method",
        "",
    ]
    for method, subset in by_method.items():
        lines.append(f"- {method}: `{sum(1 for row in subset if row.get('status') == 'PASS')}/{len(subset)} PASS`")
    lines.extend(["", "## Per Line", ""])
    for line, subset in by_line.items():
        lines.append(f"- {line}: `{sum(1 for row in subset if row.get('status') == 'PASS')}/{len(subset)} PASS`")
    lines.extend(["", "## Raw Shortage Cases", ""])
    if raw_shortage:
        for row in raw_shortage:
            lines.append(f"- {row['line']}/{row['method']}/{row['frame_id']}: raw `{row.get('merged_raw_points')}` target `{row.get('target_output_points')}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Failed Frames", ""])
    if failed:
        for row in failed:
            lines.append(f"- {row['line']}/{row['method']}/{row['frame_id']}: `{row.get('error_message', '')}` log `{row.get('log_path', '')}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Wrapper And Checkpoint", ""])
    for method in METHODS:
        first = next((row for row in rows if row["method"] == method), None)
        if first:
            lines.append(f"- {method}: script `{first.get('script_used')}`, checkpoint `{first.get('checkpoint_path')}`")
    lines.extend(["", "## Ready For Full Val", ""])
    if ready:
        for item in ready:
            lines.append(f"- READY: `{item}`")
    else:
        lines.append("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", default=METHODS, choices=METHODS)
    parser.add_argument("--lines", nargs="*", default=list(LINES), choices=list(LINES))
    parser.add_argument("--frames", nargs="*", default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()

    frames = args.frames or choose_smoke_frames(min_count=5)
    manifest_path = AUDIT_DIR / "smoke_test_manifest.csv"
    report_path = AUDIT_DIR / "smoke_test_report.md"
    rows: list[dict[str, Any]] = read_existing_manifest(manifest_path)
    active_keys = {(line, method, frame) for line in args.lines for method in args.methods for frame in frames}
    rows = [row for row in rows if (row.get("line"), row.get("method"), row.get("frame_id")) not in active_keys]
    for line in args.lines:
        for method in args.methods:
            for frame in frames:
                print(f"RUN {line} {method} {frame}", flush=True)
                rows.append(run_one(line, method, frame, args.timeout))
                write_csv(manifest_path, rows)
                write_report(report_path, rows, sorted({row["frame_id"] for row in rows if row.get("frame_id")}))
    write_csv(manifest_path, rows)
    write_report(report_path, rows, sorted({row["frame_id"] for row in rows if row.get("frame_id")}))
    passed = sum(1 for row in rows if row.get("status") == "PASS")
    print(f"SMOKE_TEST_MANIFEST={manifest_path}")
    print(f"SMOKE_TEST_REPORT={report_path}")
    print(f"SMOKE_TEST_PASS={passed}")
    print(f"SMOKE_TEST_FAIL={len(rows) - passed}")
    print("DETECTOR_EVAL_STARTED=NO")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
