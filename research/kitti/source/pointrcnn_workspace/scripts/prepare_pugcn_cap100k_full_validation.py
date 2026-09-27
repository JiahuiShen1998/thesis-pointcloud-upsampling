#!/usr/bin/env python3
"""Prepare and optionally run resumable full-validation PU-GCN cap_100k generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "pugcn_full_validation_cap100k"
FULL_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
ORIGINAL_FOLDER = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_original_val"
DEFAULT_INPUT_VELODYNE = ORIGINAL_FOLDER
EAR_FOLDER = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_ear_val"
PUNET_FOLDER = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_punet_x2_fullframe"
PUGCN_FOLDER = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "pugcn_cap_100k"
SMOKE_DENSE_FOLDER = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_pugcn_filter_range_dedup_voxel_003_smoke5"
OLD_DENSE_FOLDER = PROJECT_ROOT / "results" / "pointrcnn_larger_default_validation_cap100k" / "pugcn_dense_filter_range_dedup_voxel_003"
MID_DENSE_FOLDER = PROJECT_ROOT / "results" / "pointrcnn_50frame_default_validation_cap100k" / "pugcn_dense_filter_range_dedup_voxel_003"
PRECHECK_SOURCE = PROJECT_ROOT / "results" / "pointrcnn_50frame_default_validation_cap100k" / "pugcn_cap100k_50frame_input_precheck.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--full-split", type=Path, default=FULL_SPLIT)
    parser.add_argument("--input-velodyne", type=Path, default=DEFAULT_INPUT_VELODYNE)
    parser.add_argument("--output-folder", type=Path, default=PUGCN_FOLDER)
    parser.add_argument("--smoke-dense-folder", type=Path, default=SMOKE_DENSE_FOLDER)
    parser.add_argument(
        "--existing-dense-folder",
        type=Path,
        action="append",
        default=None,
        help="Dense source folders passed to generate_pugcn_cap100k_subset.py. Can be supplied more than once.",
    )
    parser.add_argument("--generate-missing", action="store_true")
    parser.add_argument("--run-cap-precheck", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0 means all missing frames from start-index onward.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--frame-list", type=Path, default=None, help="Optional custom frame list instead of the full validation split.")
    parser.add_argument("--pugcn-conda-env", type=str, default="pugcn")
    parser.add_argument("--metrics-conda-env", type=str, default="upsampling_basic")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--skip-existing-valid", action="store_true", default=True)
    parser.add_argument(
        "--postprocess-timeout-sec",
        type=int,
        default=1800,
        help="Maximum seconds to allow one frame postprocess subprocess to run during dense generation.",
    )
    args = parser.parse_args()
    if args.existing_dense_folder is None:
        args.existing_dense_folder = [OLD_DENSE_FOLDER, MID_DENSE_FOLDER]
    return args


def load_ids(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def slice_ids(frame_ids: Sequence[str], start_index: int, limit: int) -> List[str]:
    if start_index < 0:
        raise ValueError("start-index must be >= 0")
    sliced = list(frame_ids[start_index:])
    if limit > 0:
        sliced = sliced[:limit]
    return sliced


def count_bins(folder: Path) -> int:
    return sum(1 for _ in folder.glob("*.bin"))


def rows_from_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def run_command(cmd: Sequence[str], cwd: Path = PROJECT_ROOT, log_path: Path | None = None) -> subprocess.CompletedProcess:
    if log_path is None:
        return subprocess.run(list(cmd), cwd=str(cwd), capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ %s\n" % " ".join(cmd))
        proc = subprocess.run(list(cmd), cwd=str(cwd), stdout=log, stderr=log, text=True, check=False)
        log.write("[exit=%s]\n" % proc.returncode)
    return proc


def conda_python(env_name: str, script: Path, extra_args: Sequence[str]) -> List[str]:
    return ["conda", "run", "-n", env_name, "python", str(script)] + list(extra_args)


def estimate_runtime_seconds(missing_count: int) -> float:
    rows = rows_from_csv(PRECHECK_SOURCE)
    timings = []
    for row in rows:
        value = str(row.get("generation_runtime_sec", "")).strip()
        if not value:
            continue
        try:
            timings.append(float(value))
        except ValueError:
            continue
    if not timings:
        return float("nan")
    return missing_count * (sum(timings) / len(timings))


def format_hours(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "N/A"
    return "%.2f" % (seconds / 3600.0)


def load_points(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError("%s does not reshape to N x 4" % path)
    return raw.reshape(-1, 4)


def point_is_metric_valid(points: np.ndarray) -> bool:
    if points.ndim != 2 or points.shape[1] != 4:
        return False
    if not np.isfinite(points).all():
        return False
    xyz = points[:, :3]
    mask = (
        (xyz[:, 0] >= 0.0)
        & (xyz[:, 0] <= 80.0)
        & (xyz[:, 1] >= -40.0)
        & (xyz[:, 1] <= 40.0)
        & (xyz[:, 2] >= -3.5)
        & (xyz[:, 2] <= 2.0)
    )
    return bool(mask.all())


def existing_output_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        points = load_points(path)
    except Exception:
        return False
    return point_is_metric_valid(points)


def append_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def read_id_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def update_id_file(path: Path, add: str | None = None, remove: str | None = None) -> None:
    ids = read_id_set(path)
    if remove is not None:
        ids.discard(remove)
    if add is not None:
        ids.add(add)
    text = "\n".join(sorted(ids))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def selected_frame_ids(args: argparse.Namespace) -> List[str]:
    source = args.frame_list if args.frame_list is not None else args.full_split
    return load_ids(source)


def missing_frame_ids(frame_ids: Sequence[str], output_folder: Path) -> List[str]:
    return [frame_id for frame_id in frame_ids if not output_folder.joinpath("%s.bin" % frame_id).exists()]


def effective_targets(args: argparse.Namespace, frame_ids: Sequence[str]) -> List[str]:
    if args.skip_existing_valid:
        targets = [frame_id for frame_id in frame_ids if not existing_output_valid(args.output_folder / ("%s.bin" % frame_id))]
    else:
        targets = list(frame_ids)
    if args.resume:
        completed = set(load_ids(args.result_root / "completed_frame_ids.txt")) if (args.result_root / "completed_frame_ids.txt").exists() else set()
        failed = set(load_ids(args.result_root / "failed_frame_ids.txt")) if (args.result_root / "failed_frame_ids.txt").exists() else set()
        targets = [frame_id for frame_id in targets if frame_id not in completed and frame_id not in failed]
    return slice_ids(targets, args.start_index, args.limit)


def coverage_rows() -> List[Dict[str, object]]:
    return [
        {"method": "Original KITTI", "folder": str(ORIGINAL_FOLDER), "bin_files": count_bins(ORIGINAL_FOLDER)},
        {"method": "EAR", "folder": str(EAR_FOLDER), "bin_files": count_bins(EAR_FOLDER)},
        {"method": "PU-Net", "folder": str(PUNET_FOLDER), "bin_files": count_bins(PUNET_FOLDER)},
        {"method": "PU-GCN cap_100k", "folder": str(PUGCN_FOLDER), "bin_files": count_bins(PUGCN_FOLDER)},
    ]


def write_summary(
    args: argparse.Namespace,
    frame_ids: Sequence[str],
    targets: Sequence[str],
    estimate_sec: float,
    runtime_sec: float,
    progress_rows: Sequence[Dict[str, object]],
) -> None:
    def display_path(path: Path) -> str:
        resolved = path if path.is_absolute() else (PROJECT_ROOT / path)
        try:
            return str(resolved.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    write_csv(args.result_root / "full_validation_folder_coverage.csv", coverage_rows())
    completed = load_ids(args.result_root / "completed_frame_ids.txt") if (args.result_root / "completed_frame_ids.txt").exists() else []
    failed = load_ids(args.result_root / "failed_frame_ids.txt") if (args.result_root / "failed_frame_ids.txt").exists() else []
    lines = [
        "# PU-GCN cap_100k Full Validation Preparation",
        "",
        "This package prepares or runs resumable full-validation PU-GCN cap_100k generation under the same default PointRCNN detector setting used by the other methods.",
        "",
        "## Full Validation Split",
        "",
        "- split file: `%s`" % display_path(args.frame_list if args.frame_list is not None else args.full_split),
        "- selected frame count in this run window: `%d`" % len(frame_ids),
        "- frames still needing work in this run window before resume filtering: `%d`" % len(targets),
        "",
        "## Runtime Estimate",
        "",
        "- estimated hours for current target set from 50-frame average: `%s`" % format_hours(estimate_sec),
        "- actual runtime in this invocation: `%.2f` sec" % runtime_sec,
        "",
        "## Resume State",
        "",
        "- resume mode: `%s`" % ("yes" if args.resume else "no"),
        "- completed ids file: `%s` (%d ids)" % (args.result_root / "completed_frame_ids.txt", len(completed)),
        "- failed ids file: `%s` (%d ids)" % (args.result_root / "failed_frame_ids.txt", len(failed)),
        "- progress log: `%s`" % (args.result_root / "progress.log"),
        "",
        "## Commands",
        "",
        "Progress rows:",
        "",
        "| Frame | Status | Runtime sec | Output exists | Precheck status | Notes |",
        "|---|---|---:|---|---|---|",
    ]
    for row in progress_rows:
        lines.append(
            "| {frame_id} | {status} | {runtime_sec} | {output_exists} | {precheck_status} | {notes} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- raw KITTI data is unchanged",
            "- PointRCNN source code is unchanged in this step",
            "- output folder remains `%s/`" % args.output_folder,
            "- existing valid `pugcn_cap_100k` files are skipped",
            "- each generated frame is followed by detector-facing cap/precheck for that frame",
            "",
        ]
    )
    (args.result_root / "pugcn_full_validation_preparation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def per_frame_paths(args: argparse.Namespace, frame_id: str) -> Dict[str, Path]:
    split_file = args.result_root / "runtime_splits" / ("%s.txt" % frame_id)
    dense_frame_root = args.result_root / "pugcn_dense_generation" / frame_id
    return {
        "split_file": split_file,
        "dense_frame_root": dense_frame_root,
        "dense_output_folder": args.result_root / "pugcn_dense_filter_range_dedup_voxel_003",
        "visual_root": args.result_root / "pugcn_dense_visualization",
        "metrics_root": args.result_root / "pugcn_dense_metrics",
        "dense_output_bin": args.result_root / "pugcn_dense_filter_range_dedup_voxel_003" / ("%s.bin" % frame_id),
        "frame_precheck_root": args.result_root / "frame_precheck" / frame_id,
        "frame_log": args.result_root / "frame_logs" / ("%s.log" % frame_id),
        "selected_postprocess_bin": dense_frame_root / "postprocess" / "filter_range_dedup_voxel_003" / ("%s_pugcn_full_frame.bin" % frame_id),
        "batch_failure": dense_frame_root / "batch_failure.json",
    }


def read_batch_failure(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"error": "unreadable_batch_failure"}


def run_single_frame(args: argparse.Namespace, frame_id: str, progress_log: Path) -> Dict[str, object]:
    paths = per_frame_paths(args, frame_id)
    paths["split_file"].parent.mkdir(parents=True, exist_ok=True)
    paths["split_file"].write_text(frame_id + "\n", encoding="utf-8")
    frame_start = time.time()
    notes: List[str] = []
    status = "pending"
    precheck_status = "not_run"

    if existing_output_valid(args.output_folder / ("%s.bin" % frame_id)):
        status = "skipped_existing_valid"
        precheck_status = "already_valid"
        update_id_file(args.result_root / "completed_frame_ids.txt", add=frame_id)
        update_id_file(args.result_root / "failed_frame_ids.txt", remove=frame_id)
        append_line(progress_log, "[%s] skipped existing valid output" % frame_id)
        return {
            "frame_id": frame_id,
            "status": status,
            "runtime_sec": "%.2f" % (time.time() - frame_start),
            "output_exists": True,
            "precheck_status": precheck_status,
            "notes": "",
        }

    gen_cmd = conda_python(
        args.pugcn_conda_env,
        PROJECT_ROOT / "scripts" / "pugcn_batch_validate_full_frame.py",
        [
            "--split-file",
            str(paths["split_file"]),
            "--input-velodyne",
            str(args.input_velodyne),
            "--output-velodyne",
            str(paths["dense_output_folder"]),
            "--work-root",
            str(args.result_root / "pugcn_dense_generation"),
            "--visualization-root",
            str(args.result_root / "pugcn_dense_visualization"),
            "--metrics-root",
            str(args.result_root / "pugcn_dense_metrics"),
            "--limit",
            "1",
            "--num-patches",
            "100",
            "--patch-size",
            "1024",
            "--upsample-ratio",
            "4",
            "--skip-existing",
            "--postprocess-timeout-sec",
            str(args.postprocess_timeout_sec),
        ],
    )
    append_line(progress_log, "[%s] start dense generation" % frame_id)
    gen_proc = run_command(gen_cmd, log_path=paths["frame_log"])
    dense_exists = paths["dense_output_bin"].exists()
    batch_failure = read_batch_failure(paths["batch_failure"])
    batch_error = str(batch_failure.get("error", ""))
    postprocess_timed_out = gen_proc.returncode != 0 and batch_error == "postprocess_timeout"
    if not dense_exists and paths["selected_postprocess_bin"].exists() and not postprocess_timed_out:
        paths["dense_output_folder"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths["selected_postprocess_bin"], paths["dense_output_bin"])
        dense_exists = True
        notes.append("recovered_dense_output_from_frame_workspace")
    if gen_proc.returncode != 0:
        notes.append("dense_generation_exit_%s" % gen_proc.returncode)
        if postprocess_timed_out:
            notes.append("postprocess_timeout_%ss" % batch_failure.get("timeout_sec", args.postprocess_timeout_sec))
            status = "postprocess_timeout"
            precheck_status = "not_run"
            update_id_file(args.result_root / "failed_frame_ids.txt", add=frame_id)
            append_line(
                progress_log,
                "[%s] postprocess timeout after %s sec; marked failed and continuing"
                % (frame_id, batch_failure.get("timeout_sec", args.postprocess_timeout_sec)),
            )
            return {
                "frame_id": frame_id,
                "status": status,
                "runtime_sec": "%.2f" % (time.time() - frame_start),
                "output_exists": False,
                "precheck_status": precheck_status,
                "notes": ";".join(notes),
            }
        if not dense_exists:
            status = "dense_generation_failed"
            precheck_status = "not_run"
            update_id_file(args.result_root / "failed_frame_ids.txt", add=frame_id)
            append_line(progress_log, "[%s] failed before dense output; exit=%s" % (frame_id, gen_proc.returncode))
            return {
                "frame_id": frame_id,
                "status": status,
                "runtime_sec": "%.2f" % (time.time() - frame_start),
                "output_exists": False,
                "precheck_status": precheck_status,
                "notes": ";".join(notes),
            }
        notes.append("dense_output_exists_after_failure")

    if args.run_cap_precheck:
        precheck_cmd = conda_python(
            args.metrics_conda_env,
            PROJECT_ROOT / "scripts" / "generate_pugcn_cap100k_subset.py",
            [
                "--result-root",
                str(paths["frame_precheck_root"]),
                "--frame-list",
                str(paths["split_file"]),
                "--output-folder",
                str(args.output_folder),
                "--smoke-dense-folder",
                str(args.smoke_dense_folder),
            ],
        )
        for dense_folder in args.existing_dense_folder:
            precheck_cmd.extend(["--existing-dense-folder", str(dense_folder)])
        precheck_cmd.extend(["--new-dense-folder", str(paths["dense_output_folder"])])
        append_line(progress_log, "[%s] start cap/precheck" % frame_id)
        pre_proc = run_command(precheck_cmd, log_path=paths["frame_log"])
        output_ok = existing_output_valid(args.output_folder / ("%s.bin" % frame_id))
        if pre_proc.returncode == 0 and output_ok:
            status = "completed"
            precheck_status = "passed"
            update_id_file(args.result_root / "completed_frame_ids.txt", add=frame_id)
            update_id_file(args.result_root / "failed_frame_ids.txt", remove=frame_id)
            append_line(progress_log, "[%s] completed and passed precheck" % frame_id)
        else:
            status = "precheck_failed"
            precheck_status = "failed"
            if pre_proc.returncode != 0:
                notes.append("cap_precheck_exit_%s" % pre_proc.returncode)
            if not output_ok:
                notes.append("output_not_valid_after_precheck")
            update_id_file(args.result_root / "failed_frame_ids.txt", add=frame_id)
            append_line(progress_log, "[%s] precheck failed" % frame_id)
    else:
        output_ok = existing_output_valid(args.output_folder / ("%s.bin" % frame_id))
        status = "generated_no_precheck" if output_ok else "generated_but_output_missing"
        precheck_status = "skipped"
        append_line(progress_log, "[%s] generation finished without post-precheck" % frame_id)

    return {
        "frame_id": frame_id,
        "status": status,
        "runtime_sec": "%.2f" % (time.time() - frame_start),
        "output_exists": (args.output_folder / ("%s.bin" % frame_id)).exists(),
        "precheck_status": precheck_status,
        "notes": ";".join(notes),
    }


def initialize_resume_files(args: argparse.Namespace) -> None:
    for name in ("completed_frame_ids.txt", "failed_frame_ids.txt", "progress.log"):
        path = args.result_root / name
        if not args.resume and path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    args.result_root.mkdir(parents=True, exist_ok=True)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    all_frame_ids = selected_frame_ids(args)
    targets = effective_targets(args, all_frame_ids)
    missing_ids = missing_frame_ids(all_frame_ids, args.output_folder)
    estimate_sec = estimate_runtime_seconds(len(targets))

    (args.result_root / "full_validation_frame_list.txt").write_text("\n".join(all_frame_ids) + "\n", encoding="utf-8")
    (args.result_root / "missing_full_validation_pugcn_frames.txt").write_text("\n".join(missing_ids) + ("\n" if missing_ids else ""), encoding="utf-8")
    (args.result_root / "selected_run_frame_ids.txt").write_text("\n".join(targets) + ("\n" if targets else ""), encoding="utf-8")
    write_csv(
        args.result_root / "pugcn_full_validation_counts.csv",
        [
            {
                "selected_frames": len(all_frame_ids),
                "existing_pugcn_cap100k_files": len(all_frame_ids) - len(missing_ids),
                "missing_pugcn_cap100k_files": len(missing_ids),
                "target_frames_this_run_after_resume_and_skip": len(targets),
                "estimated_target_generation_hours_from_50frame_average": format_hours(estimate_sec),
            }
        ],
    )

    progress_rows: List[Dict[str, object]] = []
    runtime_start = time.time()

    if args.list_only:
        write_summary(args, all_frame_ids, targets, estimate_sec, 0.0, progress_rows)
        print("selected_frames", len(all_frame_ids))
        print("missing_frames", len(missing_ids))
        print("target_frames_this_run", len(targets))
        return

    initialize_resume_files(args)
    progress_log = args.result_root / "progress.log"
    append_line(progress_log, "=== run_start ts=%.3f ===" % time.time())

    if args.generate_missing:
        for index, frame_id in enumerate(targets, start=1):
            append_line(progress_log, "[%s] progress %d/%d" % (frame_id, index, len(targets)))
            progress_rows.append(run_single_frame(args, frame_id, progress_log))

    if not args.generate_missing and args.run_cap_precheck:
        for index, frame_id in enumerate(targets, start=1):
            append_line(progress_log, "[%s] precheck-only progress %d/%d" % (frame_id, index, len(targets)))
            row = run_single_frame(args, frame_id, progress_log)
            progress_rows.append(row)

    write_csv(args.result_root / "run_progress_rows.csv", progress_rows)
    write_summary(args, all_frame_ids, targets, estimate_sec, time.time() - runtime_start, progress_rows)
    print("selected_frames", len(all_frame_ids))
    print("missing_frames", len(missing_ids))
    print("target_frames_this_run", len(targets))
    print("completed_in_this_invocation", sum(1 for row in progress_rows if row["status"] in {"completed", "skipped_existing_valid"}))
    print("failed_in_this_invocation", sum(1 for row in progress_rows if "failed" in str(row["status"])))


if __name__ == "__main__":
    main()
