#!/usr/bin/env python3
"""Run resumable KITTI patch/PU-Net causal-ablation upsampling variants."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

import run_kitti_unified_x4_current_methods_smoke as smoke


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "results" / "kitti_patch_punet_causal_ablation_v1_20260731"
STRICT_MANY = PROJECT_ROOT / "scripts" / "wrappers" / "strict_x4_from_merged_raw_many.py"
PUGCN_MANY = PROJECT_ROOT / "scripts" / "wrappers" / "tf_pugcn_family_patch_infer_many.py"
PUNET_MANY = PROJECT_ROOT / "scripts" / "wrappers" / "tf_punet_patch_infer_many.py"
PDANS_ONE = PROJECT_ROOT / "scripts" / "wrappers" / "pdans_patch_infer.py"
LINES = {
    "line_a_original_x4_up": smoke.ORIGINAL_INPUT,
    "line_b_downsampled_x4_up": smoke.DOWNSAMPLED_X4_INPUT,
}
DISPLAY = {
    "pu_gcn": "PU-GCN",
    "pu_edgeformer": "PU-EdgeFormer",
    "pu_net": "PU-Net",
    "pdans": "PDANS",
}


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_frames(path: Path) -> list[str]:
    if path.suffix == ".json":
        payload = read_json(path)
        values = payload.get("frame_ids")
        if not isinstance(values, list):
            raise ValueError(f"JSON frame list has no frame_ids list: {path}")
        return [str(value) for value in values]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_logged(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    print("RUN " + " ".join(command), flush=True)
    with log_path.open("ab") as handle:
        proc = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=merged_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode:
        raise RuntimeError(f"command failed rc={proc.returncode}; log={log_path}")


def free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def variant_name(args: argparse.Namespace) -> str:
    if args.variant_name:
        return args.variant_name
    patch = args.patch_selection
    if patch in ("fps_ball_local_v1", "fps_ball_cover_v2", "fps_ball_cover_knn_v3"):
        patch += f"_pr{args.patch_num_ratio}_r{args.ball_radius_m:g}_min{args.min_ball_points}"
        if patch.startswith(("fps_ball_cover_v2", "fps_ball_cover_knn_v3")):
            patch += f"_cover{args.cover_min_points}_cr{args.cover_radius_m:g}"
    elif patch == "fps_knn_local_v1":
        patch += f"_pr{args.patch_num_ratio}"
    name = f"{args.method}__{patch}"
    if args.method == "pu_net":
        name += f"__{args.normalization_mode}"
    return name


def final_complete(final_prov: Path, final_bin: Path, expected_points: int) -> bool:
    try:
        prov = read_json(final_prov)
        size_ok = final_bin.stat().st_size == expected_points * 4 * np.dtype(np.float32).itemsize
        return bool(
            prov.get("status") == "PASS"
            and prov.get("final_output_point_count") == expected_points
            and size_ok
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def merged_complete(
    merged_prov: Path,
    merged_raw: Path,
    patch_meta: Path,
    args: argparse.Namespace,
    line: str,
    frame: str,
) -> bool:
    """Cheap resume check; strict conversion later re-reads and hashes the array."""
    try:
        prov = read_json(merged_prov)
        meta = read_json(patch_meta)
        patch_count = int(len(meta["patches"]))
        array = np.load(merged_raw, mmap_mode="r", allow_pickle=False)
        return bool(
            prov.get("status") == "PASS"
            and prov.get("method") == DISPLAY[args.method]
            and prov.get("line") == line
            and str(prov.get("frame_id")) == frame
            and prov.get("patch_selection") == args.patch_selection
            and int(prov.get("patch_count", -1)) == patch_count
            and int(prov.get("merged_raw_output_point_count", -1))
            == patch_count * 8192
            and prov.get("merged_float32_sha256")
            and array.dtype == np.float32
            and array.shape == (patch_count * 8192, 3)
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def extraction_complete(meta_path: Path, patch_dir: Path, args: argparse.Namespace) -> bool:
    try:
        meta = read_json(meta_path)
        expected = {
            "patch_selection": args.patch_selection,
            "patch_input_points": 2048,
            "status": "PASS",
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            return False
        if args.patch_selection != "deterministic_spatial_chunk":
            if int(meta.get("patch_num_ratio", -1)) != args.patch_num_ratio:
                return False
        if args.patch_selection in (
            "fps_ball_local_v1",
            "fps_ball_cover_v2",
            "fps_ball_cover_knn_v3",
        ):
            if float(meta.get("ball_radius_m", -1)) != args.ball_radius_m:
                return False
            if int(meta.get("min_ball_points", -1)) != args.min_ball_points:
                return False
            if args.patch_selection in ("fps_ball_cover_v2", "fps_ball_cover_knn_v3"):
                if int(meta.get("cover_min_points", -1)) != args.cover_min_points:
                    return False
                if float(meta.get("cover_radius_m", -1)) != args.cover_radius_m:
                    return False
        return all((patch_dir / patch["file"]).is_file() for patch in meta["patches"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def patch_command(
    args: argparse.Namespace,
    input_path: Path,
    patch_dir: Path,
    meta_path: Path,
    line: str,
    frame: str,
) -> list[str]:
    command = [
        str(smoke.UP_BASIC_PY),
        str(smoke.PATCH_EXTRACTOR),
        "--input_bin",
        str(input_path),
        "--output_patch_dir",
        str(patch_dir),
        "--patch_input_points",
        "2048",
        "--patch_selection",
        args.patch_selection,
        "--patch_num_ratio",
        str(args.patch_num_ratio),
        "--ball_radius_m",
        str(args.ball_radius_m),
        "--min_ball_points",
        str(args.min_ball_points),
        "--cover_min_points",
        str(args.cover_min_points),
        "--cover_radius_m",
        str(args.cover_radius_m),
        "--seed",
        str(smoke.SEED + int(frame)),
        "--line",
        line,
        "--frame_id",
        frame,
        "--metadata_json",
        str(meta_path),
    ]
    return command


def batch_command(
    args: argparse.Namespace,
    jobs_json: Path,
    line: str,
) -> tuple[list[str], dict[str, str]]:
    common = [
        "--jobs_json",
        str(jobs_json),
        "--line",
        line,
        "--patch_input_points",
        "2048",
        "--patch_output_points",
        "8192",
        "--up_ratio",
        "4",
    ]
    env = {"CUDA_VISIBLE_DEVICES": args.cuda_visible_devices}
    if args.method == "pu_net":
        return (
            [
                str(smoke.PUGCN_PY),
                str(PUNET_MANY),
                "--code_dir",
                str(smoke.PUNET_CODE),
                "--checkpoint_dir",
                str(smoke.PUNET_CKPT),
                "--tf_ops_repo_dir",
                str(smoke.PUGCN_REPO),
                *common,
                "--normalization_mode",
                args.normalization_mode,
            ],
            env,
        )
    if args.method == "pu_gcn":
        python = smoke.PUGCN_PY
        repo = smoke.PUGCN_REPO
        checkpoint = smoke.PUGCN_CKPT
        model = "pugcn"
    elif args.method == "pu_edgeformer":
        # The puedgeformer env segfaults while loading TensorFlow's native
        # extension (`_pywrap_tensorflow_internal.so`), so it can no longer run.
        # Both envs ship the same tensorflow_gpu-1.13.1 and the same CUDA 10.0
        # libraries; they differ only in numpy (1.16.6 vs 1.19.5).  PU-EdgeFormer
        # already runs on PU-GCN's ops (see PU-EdgeFormer_ops_reuse), and the
        # pugcn interpreter drives it unchanged, so use that one.
        python = smoke.PUGCN_PY
        repo = smoke.PUEDGE_REPO
        checkpoint = smoke.PUEDGE_CKPT
        model = "edgetransformer"
    else:
        raise ValueError(f"no batch command for {args.method}")
    return (
        [
            str(python),
            str(PUGCN_MANY),
            "--repo_dir",
            str(repo),
            "--method",
            DISPLAY[args.method],
            "--model",
            model,
            "--checkpoint_dir",
            str(checkpoint),
            *common,
            "--seed",
            str(smoke.SEED),
            "--gpu_memory_fraction",
            str(args.gpu_memory_fraction),
        ],
        env,
    )


def run_pdans_one(
    args: argparse.Namespace,
    line: str,
    frame: str,
    paths: dict[str, Path],
    patch_meta: Path,
) -> None:
    command = [
        str(smoke.PDANS_PY),
        str(PDANS_ONE),
        "--patch_metadata_json",
        str(patch_meta),
        "--raw_patch_output_dir",
        str(paths["raw_patch_dir"]),
        "--merged_raw_output",
        str(paths["merged_raw"]),
        "--merged_raw_provenance_json",
        str(paths["merged_prov"]),
        "--checkpoint",
        str(smoke.PDANS_CKPT),
        "--config",
        str(smoke.PDANS_CONFIG),
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
        str(smoke.SEED),
        "--device",
        "cuda",
    ]
    pdans_bin = str(Path(smoke.PDANS_PY).resolve().parent)
    run_logged(
        command,
        paths["infer_log"],
        {
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            "PATH": pdans_bin + os.pathsep + os.environ.get("PATH", ""),
        },
    )


def frame_paths(base: Path, frame: str) -> dict[str, Path]:
    return {
        "patch_dir": base / "work" / frame / "input_patches",
        "raw_patch_dir": base / "work" / frame / "method_raw",
        "merged_raw": base / "merged_raw" / f"{frame}.npy",
        "patch_meta": base / "manifests" / f"{frame}_patch_metadata.json",
        "merged_prov": base / "manifests" / f"{frame}_merged_raw_provenance.json",
        "final_bin": base / "final_bin" / f"{frame}.bin",
        "final_prov": base / "manifests" / f"{frame}_final_provenance.json",
        "extract_log": base / "logs" / f"{frame}_patch_extract.log",
        "infer_log": base / "logs" / f"{frame}_method_infer.log",
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_line(args: argparse.Namespace, frames: list[str], variant: str, line: str) -> list[dict[str, Any]]:
    base = args.experiment_root / args.run_kind / variant / line
    patch_base = args.reuse_patch_base or base
    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, str]] = []
    paths_by_frame: dict[str, dict[str, Path]] = {}

    def ensure_extraction(frame: str) -> str:
        if not args.extract_only:
            final_paths = frame_paths(base, frame)
            input_bytes = (args.line_inputs[line] / f"{frame}.bin").stat().st_size
            if input_bytes > 0 and input_bytes % 16 == 0 and final_complete(
                final_paths["final_prov"], final_paths["final_bin"], input_bytes // 4
            ):
                return frame
        if free_gib(args.experiment_root) < args.min_free_gib:
            raise RuntimeError(f"free disk fell below {args.min_free_gib} GiB")
        paths = frame_paths(patch_base, frame)
        input_path = args.line_inputs[line] / f"{frame}.bin"
        if not extraction_complete(paths["patch_meta"], paths["patch_dir"], args):
            if args.reuse_patch_base:
                raise RuntimeError(
                    f"reusable patch cache is incomplete or incompatible: {frame}"
                )
            paths["patch_dir"].mkdir(parents=True, exist_ok=True)
            run_logged(
                patch_command(
                    args,
                    input_path,
                    paths["patch_dir"],
                    paths["patch_meta"],
                    line,
                    frame,
                ),
                paths["extract_log"],
            )
        return frame

    if args.extract_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.extract_workers
        ) as executor:
            for position, frame in enumerate(
                executor.map(ensure_extraction, frames), start=1
            ):
                print(
                    f"EXTRACT {line} {variant} {frame} {position}/{len(frames)}",
                    flush=True,
                )

    for position, frame in enumerate(frames, start=1):
        paths = frame_paths(base, frame)
        patch_paths = frame_paths(patch_base, frame)
        paths_by_frame[frame] = paths
        input_path = args.line_inputs[line] / f"{frame}.bin"
        input_points = smoke.read_kitti_bin(input_path).shape[0]
        expected_points = int(input_points * 4)
        row = {
            "run_kind": args.run_kind,
            "variant": variant,
            "method": args.method,
            "line": line,
            "frame_id": frame,
            "input_points": int(input_points),
            "target_points": expected_points,
            "patch_selection": args.patch_selection,
            "patch_num_ratio": args.patch_num_ratio,
            "normalization_mode": args.normalization_mode if args.method == "pu_net" else "model_native",
            "status": "STARTED",
            "final_bin": str(paths["final_bin"]),
        }
        if not args.extract_only and final_complete(
            paths["final_prov"], paths["final_bin"], expected_points
        ):
            row["status"] = "PASS_REUSED"
            rows.append(row)
            print(f"REUSE_FINAL {line} {frame} {position}/{len(frames)}", flush=True)
            continue
        if free_gib(args.experiment_root) < args.min_free_gib:
            raise RuntimeError(f"free disk fell below {args.min_free_gib} GiB")
        if not extraction_complete(
            patch_paths["patch_meta"], patch_paths["patch_dir"], args
        ):
            if args.reuse_patch_base:
                raise RuntimeError(
                    f"reusable patch cache is incomplete or incompatible: {frame}"
                )
            patch_paths["patch_dir"].mkdir(parents=True, exist_ok=True)
            run_logged(
                patch_command(
                    args,
                    input_path,
                    patch_paths["patch_dir"],
                    patch_paths["patch_meta"],
                    line,
                    frame,
                ),
                patch_paths["extract_log"],
            )
        if args.extract_only:
            row["status"] = "EXTRACT_PASS"
        elif final_complete(paths["final_prov"], paths["final_bin"], expected_points):
            row["status"] = "PASS_REUSED"
        elif args.method == "pdans":
            if merged_complete(
                paths["merged_prov"],
                paths["merged_raw"],
                patch_paths["patch_meta"],
                args,
                line,
                frame,
            ):
                row["status"] = "MERGED_REUSED"
            else:
                run_pdans_one(
                    args,
                    line,
                    frame,
                    paths,
                    patch_paths["patch_meta"],
                )
        else:
            if merged_complete(
                paths["merged_prov"],
                paths["merged_raw"],
                patch_paths["patch_meta"],
                args,
                line,
                frame,
            ):
                row["status"] = "MERGED_REUSED"
            else:
                jobs.append(
                    {
                        "frame_id": frame,
                        "patch_metadata_json": str(patch_paths["patch_meta"]),
                        "raw_patch_output_dir": str(paths["raw_patch_dir"]),
                        "merged_raw_output": str(paths["merged_raw"]),
                        "merged_raw_provenance_json": str(paths["merged_prov"]),
                    }
                )
        rows.append(row)
        print(f"PREP {line} {variant} {frame} {position}/{len(frames)}", flush=True)

    if args.extract_only:
        for row in rows:
            row["free_gib_after"] = round(free_gib(args.experiment_root), 3)
        write_manifest(base / "run_manifest.csv", rows)
        return rows

    if jobs:
        jobs_json = base / "manifests" / "batch_jobs.json"
        save_json(jobs_json, jobs)
        command, env = batch_command(args, jobs_json, line)
        run_logged(command, base / "logs" / "batch_method_infer.log", env)

    strict_jobs = []
    for row in rows:
        frame = row["frame_id"]
        paths = paths_by_frame[frame]
        if row["status"] == "PASS_REUSED":
            continue
        merged = read_json(paths["merged_prov"])
        if merged.get("status") != "PASS":
            raise RuntimeError(f"merged inference not PASS for {line}/{frame}: {merged.get('notes')}")
        strict_jobs.append(
            {
                "method": DISPLAY[args.method],
                "line": line,
                "frame_id": frame,
                "input_bin": str(args.line_inputs[line] / f"{frame}.bin"),
                "merged_raw_output": str(paths["merged_raw"]),
                "merged_raw_provenance_json": str(paths["merged_prov"]),
                "final_output": str(paths["final_bin"]),
                "provenance_json": str(paths["final_prov"]),
                "seed": smoke.SEED + int(frame),
            }
        )
    if strict_jobs:
        strict_jobs_json = base / "manifests" / "strict_jobs.json"
        save_json(strict_jobs_json, strict_jobs)
        run_logged(
            [
                str(smoke.UP_BASIC_PY),
                str(STRICT_MANY),
                "--jobs_json",
                str(strict_jobs_json),
                "--up_ratio",
                "4",
            ],
            base / "logs" / "batch_strict.log",
        )

    for row in rows:
        frame = row["frame_id"]
        paths = paths_by_frame[frame]
        if not final_complete(paths["final_prov"], paths["final_bin"], int(row["target_points"])):
            row["status"] = "FAIL"
            raise RuntimeError(f"strict final audit failed for {line}/{frame}")
        row["status"] = "PASS"
        row["final_bytes"] = paths["final_bin"].stat().st_size
        # A resumed cleanup may encounter merged files already removed by a
        # previous, partially completed cleanup pass.
        row["merged_bytes"] = (
            paths["merged_raw"].stat().st_size if paths["merged_raw"].is_file() else 0
        )
        if args.keep_input_patches:
            if paths["raw_patch_dir"].is_dir():
                # NFS directory entries can disappear between scandir() and
                # unlink() during cleanup.  Cleanup is best-effort here: all
                # final outputs have already passed the strict audit above.
                shutil.rmtree(paths["raw_patch_dir"], ignore_errors=True)
        elif not args.keep_work_files:
            work_frame = base / "work" / frame
            if work_frame.is_dir() and args.experiment_root in work_frame.parents:
                shutil.rmtree(work_frame, ignore_errors=True)
        if args.drop_merged_after_final:
            paths["merged_raw"].unlink(missing_ok=True)
        row["free_gib_after"] = round(free_gib(args.experiment_root), 3)
    write_manifest(base / "run_manifest.csv", rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-kind", required=True)
    parser.add_argument("--frames-file", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(DISPLAY), required=True)
    parser.add_argument("--variant-name", default=None)
    parser.add_argument(
        "--patch-selection",
        choices=(
            "deterministic_spatial_chunk",
            "fps_knn_local_v1",
            "fps_ball_local_v1",
            "fps_ball_cover_v2",
            "fps_ball_cover_knn_v3",
        ),
        required=True,
    )
    parser.add_argument("--patch-num-ratio", type=int, default=3)
    parser.add_argument("--ball-radius-m", type=float, default=6.0)
    parser.add_argument("--min-ball-points", type=int, default=256)
    parser.add_argument("--cover-min-points", type=int, default=32)
    parser.add_argument("--cover-radius-m", type=float, default=None)
    parser.add_argument(
        "--normalization-mode",
        choices=("legacy_none", "unit_sphere_v1"),
        default="legacy_none",
    )
    parser.add_argument("--lines", nargs="+", choices=tuple(LINES), default=tuple(LINES))
    parser.add_argument(
        "--line-a-input",
        type=Path,
        default=LINES["line_a_original_x4_up"],
        help="Input directory for line_a_original_x4_up.",
    )
    parser.add_argument(
        "--line-b-input",
        type=Path,
        default=LINES["line_b_downsampled_x4_up"],
        help="Input directory for line_b_downsampled_x4_up.",
    )
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--frame-batch-size", type=int, default=0,
        help="Finalize and clean each batch before continuing; 0 keeps whole-split batching.",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=1,
        help="Independent frame-level patch extraction workers.",
    )
    parser.add_argument("--keep-work-files", action="store_true")
    parser.add_argument(
        "--keep-input-patches",
        action="store_true",
        help="Keep only reusable input patch files; method raw files are removed.",
    )
    parser.add_argument(
        "--reuse-patch-base",
        type=Path,
        default=None,
        help="Line-level run directory containing compatible work/manifests patches.",
    )
    parser.add_argument("--drop-merged-after-final", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.45)
    parser.add_argument("--min-free-gib", type=float, default=100.0)
    args = parser.parse_args()

    if args.cover_radius_m is None:
        args.cover_radius_m = args.ball_radius_m
    if args.extract_workers <= 0:
        raise ValueError("--extract-workers must be positive")
    if args.frame_batch_size < 0:
        raise ValueError("--frame-batch-size must be nonnegative")
    if args.keep_work_files and args.keep_input_patches:
        raise ValueError("--keep-work-files and --keep-input-patches are mutually exclusive")

    args.experiment_root = args.experiment_root.resolve()
    args.line_inputs = {
        "line_a_original_x4_up": args.line_a_input.resolve(),
        "line_b_downsampled_x4_up": args.line_b_input.resolve(),
    }
    for line in args.lines:
        if not args.line_inputs[line].is_dir():
            raise FileNotFoundError(args.line_inputs[line])
    if args.reuse_patch_base is not None:
        args.reuse_patch_base = args.reuse_patch_base.resolve()
        if not args.reuse_patch_base.is_dir():
            raise FileNotFoundError(args.reuse_patch_base)
    args.experiment_root.mkdir(parents=True, exist_ok=True)
    frames = read_frames(args.frames_file.resolve())
    if not frames:
        raise ValueError("frame list is empty")
    variant = variant_name(args)
    protocol = {
        "created_unix": time.time(),
        "run_kind": args.run_kind,
        "variant": variant,
        "method": args.method,
        "frames_file": str(args.frames_file.resolve()),
        "frame_ids": frames,
        "lines": args.lines,
        "line_inputs": {line: str(args.line_inputs[line]) for line in args.lines},
        "patch_selection": args.patch_selection,
        "patch_num_ratio": args.patch_num_ratio,
        "ball_radius_m": args.ball_radius_m,
        "min_ball_points": args.min_ball_points,
        "cover_min_points": args.cover_min_points,
        "cover_radius_m": args.cover_radius_m,
        "normalization_mode": args.normalization_mode,
        "patch_input_points": 2048,
        "patch_output_points": 8192,
        "strict_up_ratio": 4,
        "extract_workers": args.extract_workers,
        "frame_batch_size": args.frame_batch_size,
        "reuse_patch_base": (
            str(args.reuse_patch_base) if args.reuse_patch_base is not None else None
        ),
        "seed": smoke.SEED,
        "free_gib_before": free_gib(args.experiment_root),
    }
    save_json(args.experiment_root / args.run_kind / variant / "protocol.json", protocol)
    all_rows: list[dict[str, Any]] = []
    for line in args.lines:
        line_rows: list[dict[str, Any]] = []
        batch_size = args.frame_batch_size or len(frames)
        for start in range(0, len(frames), batch_size):
            batch_frames = frames[start:start + batch_size]
            line_rows.extend(run_line(args, batch_frames, variant, line))
            write_manifest(
                args.experiment_root / args.run_kind / variant / line / "run_manifest.csv",
                line_rows,
            )
            print(
                f"FRAME_BATCH_PASS {line} {start + len(batch_frames)}/{len(frames)}",
                flush=True,
            )
        all_rows.extend(line_rows)
    protocol["free_gib_after"] = free_gib(args.experiment_root)
    protocol["status"] = "PASS"
    save_json(args.experiment_root / args.run_kind / variant / "protocol.json", protocol)
    write_manifest(args.experiment_root / args.run_kind / variant / "run_manifest_all.csv", all_rows)
    print(json.dumps({"status": "PASS", "variant": variant, "rows": len(all_rows), "free_gib": protocol["free_gib_after"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
