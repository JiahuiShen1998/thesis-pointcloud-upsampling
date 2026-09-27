#!/usr/bin/env python3
"""Batch validation for detector-facing full-frame PU-GCN outputs."""

import argparse
import csv
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val_punet_smoke5.txt"
DEFAULT_INPUT_VELODYNE = PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_original_val"
DEFAULT_OUTPUT_VELODYNE = PROJECT_ROOT / "data" / "processed" / "kitti_pugcn_full_frame_filter_range_dedup_voxel_003"
DEFAULT_WORK_ROOT = PROJECT_ROOT / "results" / "pugcn_batch_validation"
DEFAULT_VIS_ROOT = PROJECT_ROOT / "results" / "pugcn_visualization_batch_filter_range_dedup_voxel_003"
DEFAULT_METRICS_ROOT = PROJECT_ROOT / "results" / "pugcn_metrics_batch_filter_range_dedup_voxel_003"
SELECTED_VARIANT = "filter_range_dedup_voxel_003"
VOXEL_SIZE = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate detector-facing full-frame PU-GCN outputs on a small KITTI batch.")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--input-velodyne", type=Path, default=DEFAULT_INPUT_VELODYNE)
    parser.add_argument("--output-velodyne", type=Path, default=DEFAULT_OUTPUT_VELODYNE)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--visualization-root", type=Path, default=DEFAULT_VIS_ROOT)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--pugcn-conda-env", type=str, default="pugcn")
    parser.add_argument("--metrics-conda-env", type=str, default="upsampling_basic")
    parser.add_argument("--pugcn-root", type=Path, default=PROJECT_ROOT / "external" / "PU-GCN")
    parser.add_argument("--restore", type=Path, default=PROJECT_ROOT / "external" / "PU-GCN" / "pretrained" / "pu1k-pugcn")
    parser.add_argument("--num-patches", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--upsample-ratio", type=int, default=4)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--postprocess-timeout-sec",
        type=int,
        default=1800,
        help="Maximum seconds to allow one frame postprocess subprocess to run. Use 0 to disable.",
    )
    return parser.parse_args()


def read_ids(path: Path, limit: int) -> List[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return ids[:limit] if limit > 0 else ids


def run(cmd: Sequence[str], cwd: Path = PROJECT_ROOT, timeout_sec: int = 0) -> None:
    proc = subprocess.Popen(list(cmd), cwd=str(cwd), start_new_session=True)
    try:
        proc.wait(timeout=timeout_sec if timeout_sec > 0 else None)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise subprocess.TimeoutExpired(list(cmd), timeout_sec) from exc
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, list(cmd))


def conda_python(env_name: str, script_path: Path, extra_args: Sequence[str]) -> List[str]:
    return ["conda", "run", "-n", env_name, "python", str(script_path)] + list(extra_args)


def read_csv_row(path: Path) -> Dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return next(reader)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def fmt(value: object) -> str:
    if isinstance(value, float):
        return ("%.6f" % value).rstrip("0").rstrip(".")
    return str(value)


def average(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def load_existing_method_rows(path: Path, frame_ids: Sequence[str]) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    wanted = set(frame_ids)
    return [row for row in rows if row.get("sample_id") in wanted or row.get("frame_id") in wanted]


def build_method_comparison(
    batch_rows: Sequence[Dict[str, object]],
    frame_ids: Sequence[str],
    output_root: Path,
) -> None:
    methods = []

    original_rows = [
        {
            "method_name": "Original KITTI",
            "frame_count": len(batch_rows),
            "avg_input_points": average([float(row["input_points"]) for row in batch_rows]),
            "avg_output_points": average([float(row["input_points"]) for row in batch_rows]),
            "avg_upsampling_ratio": 1.0,
            "avg_out_of_range_points": "N/A",
            "avg_invalid_points": "N/A",
            "avg_near_duplicate_ratio": "N/A",
            "source": "batch input frames",
        }
    ]
    methods.extend(original_rows)

    ear_rows = load_existing_method_rows(
        PROJECT_ROOT / "experiments" / "density_baselines" / "01_EAR_original_upsampled" / "sanity_check" / "evaluation" / "sample_metrics.csv",
        frame_ids,
    )
    if ear_rows:
        methods.append(
            {
                "method_name": "EAR",
                "frame_count": len(ear_rows),
                "avg_input_points": average([float(row["input_points"]) for row in ear_rows]),
                "avg_output_points": average([float(row["output_points"]) for row in ear_rows]),
                "avg_upsampling_ratio": average([float(row["upsampling_ratio"]) for row in ear_rows]),
                "avg_out_of_range_points": "N/A",
                "avg_invalid_points": "N/A",
                "avg_near_duplicate_ratio": "N/A",
                "source": str((PROJECT_ROOT / "experiments" / "density_baselines" / "01_EAR_original_upsampled" / "sanity_check" / "evaluation" / "sample_metrics.csv").resolve()),
            }
        )

    punet_rows = load_existing_method_rows(
        PROJECT_ROOT / "experiments" / "density_baselines" / "PUNet_visual_only" / "fullframe_repair_smoke5" / "preprocessing" / "density_metrics.csv",
        frame_ids,
    )
    if punet_rows:
        methods.append(
            {
                "method_name": "PU-Net",
                "frame_count": len(punet_rows),
                "avg_input_points": average([float(row["input_points"]) for row in punet_rows]),
                "avg_output_points": average([float(row["output_points"]) for row in punet_rows]),
                "avg_upsampling_ratio": average([float(row["upsampling_ratio"]) for row in punet_rows]),
                "avg_out_of_range_points": "N/A",
                "avg_invalid_points": "N/A",
                "avg_near_duplicate_ratio": "N/A",
                "source": str((PROJECT_ROOT / "experiments" / "density_baselines" / "PUNet_visual_only" / "fullframe_repair_smoke5" / "preprocessing" / "density_metrics.csv").resolve()),
            }
        )

    methods.append(
        {
            "method_name": "PU-GCN filter_range_dedup_voxel_003",
            "frame_count": len(batch_rows),
            "avg_input_points": average([float(row["input_points"]) for row in batch_rows]),
            "avg_output_points": average([float(row["output_points"]) for row in batch_rows]),
            "avg_upsampling_ratio": average([float(row["effective_upsampling_ratio"]) for row in batch_rows]),
            "avg_out_of_range_points": average([float(row["output_out_of_range_points"]) for row in batch_rows]),
            "avg_invalid_points": average([float(row["output_invalid_xyz_points"]) for row in batch_rows]),
            "avg_near_duplicate_ratio": average([float(row["near_duplicate_ratio"]) for row in batch_rows]),
            "source": "results/pugcn_batch_validation frame metrics",
        }
    )

    write_csv(output_root / "filter_range_dedup_voxel_003_method_comparison.csv", methods)
    lines = [
        "# Batch Method Comparison: PU-GCN vs EAR vs PU-Net",
        "",
        "| Method | Frames | Avg input points | Avg output points | Avg upsampling ratio | Avg out-of-range points | Avg invalid points | Avg near-duplicate ratio | Source |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in methods:
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (
                row["method_name"],
                fmt(row["frame_count"]),
                fmt(row["avg_input_points"]),
                fmt(row["avg_output_points"]),
                fmt(row["avg_upsampling_ratio"]),
                fmt(row["avg_out_of_range_points"]),
                fmt(row["avg_invalid_points"]),
                fmt(row["avg_near_duplicate_ratio"]),
                row["source"],
            )
        )
    (output_root / "filter_range_dedup_voxel_003_method_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame_ids = read_ids(args.split_file, args.limit)
    args.output_velodyne.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.visualization_root.mkdir(parents=True, exist_ok=True)
    args.metrics_root.mkdir(parents=True, exist_ok=True)

    batch_rows: List[Dict[str, object]] = []
    failed_frames: List[str] = []
    suspicious_frames: List[str] = []

    for frame_id in frame_ids:
        input_bin = args.input_velodyne / ("%s.bin" % frame_id)
        frame_root = args.work_root / frame_id
        raw_dir = frame_root / "raw_100"
        post_root = frame_root / "postprocess"
        metrics_root = frame_root / "metrics"
        final_debug_bin = post_root / SELECTED_VARIANT / ("%s_pugcn_full_frame.bin" % frame_id)
        final_output_bin = args.output_velodyne / ("%s.bin" % frame_id)
        raw_output_bin = raw_dir / ("%s_pugcn_full_frame.bin" % frame_id)
        raw_work_dir = raw_dir / "work"
        postprocess_manifest = post_root / SELECTED_VARIANT / "postprocess_manifest.json"
        frame_start = time.time()

        try:
            if not (args.skip_existing and raw_output_bin.exists()):
                raw_dir.mkdir(parents=True, exist_ok=True)
                run(
                    conda_python(
                        args.pugcn_conda_env,
                        PROJECT_ROOT / "scripts" / "pugcn_kitti_full_frame_reconstruct.py",
                        [
                            "--input_bin",
                            str(input_bin),
                            "--output_bin",
                            str(raw_output_bin),
                            "--num_patches",
                            str(args.num_patches),
                            "--patch_size",
                            str(args.patch_size),
                            "--upsample_ratio",
                            str(args.upsample_ratio),
                            "--filter_out_of_range",
                            "false",
                            "--save_debug_patches",
                            "false",
                            "--dedup_voxel_size",
                            "0.0",
                            "--work_dir",
                            str(raw_work_dir),
                            "--pugcn_root",
                            str(args.pugcn_root),
                            "--restore",
                            str(args.restore),
                            "--pugcn_python",
                            "python",
                        ],
                    )
                )

            if not (args.skip_existing and final_debug_bin.exists() and postprocess_manifest.exists()):
                run(
                    conda_python(
                        args.pugcn_conda_env,
                        PROJECT_ROOT / "scripts" / "postprocess_pugcn_full_frame.py",
                        [
                            "--original_bin",
                            str(input_bin),
                            "--raw_bin",
                            str(raw_output_bin),
                            "--output_root",
                            str(post_root),
                            "--frame_id",
                            frame_id,
                            "--variants",
                            SELECTED_VARIANT,
                        ],
                    ),
                    timeout_sec=args.postprocess_timeout_sec,
                )

            final_output_bin.write_bytes(final_debug_bin.read_bytes())

            run(
                conda_python(
                    args.pugcn_conda_env,
                    PROJECT_ROOT / "scripts" / "visualize_pugcn_full_frame_comparison.py",
                    [
                        "--original_bin",
                        str(input_bin),
                        "--pugcn_bin",
                        str(final_debug_bin),
                        "--output_root",
                        str(args.visualization_root),
                        "--frame_id",
                        frame_id,
                        "--max_points",
                        "160000",
                    ],
                )
            )

            run(
                conda_python(
                    args.metrics_conda_env,
                    PROJECT_ROOT / "scripts" / "evaluate_upsampling_quality.py",
                    [
                        "--input-bin",
                        str(input_bin),
                        "--output-bin",
                        str(final_debug_bin),
                        "--sample-id",
                        frame_id,
                        "--experiment-name",
                        "pugcn_batch_filter_range_dedup_voxel_003",
                        "--output-root",
                        str(metrics_root),
                        "--duplicate-voxel-size",
                        str(VOXEL_SIZE),
                        "--near-duplicate-radius",
                        str(VOXEL_SIZE),
                    ],
                )
            )

            recon_manifest = json.loads((raw_work_dir / "manifest.json").read_text(encoding="utf-8"))
            post_manifest = json.loads(postprocess_manifest.read_text(encoding="utf-8"))
            metric_row = read_csv_row(metrics_root / "single_frame_metrics.csv")

            refresh_runtime = time.time() - frame_start
            reconstruction_runtime = float(recon_manifest["total_runtime_sec"])
            postprocess_runtime = float(post_manifest["runtime_sec"])
            row = {
                "frame_id": frame_id,
                "input_bin": str(input_bin.resolve()),
                "raw_output_bin": str(raw_output_bin.resolve()),
                "final_output_bin": str(final_output_bin.resolve()),
                "input_points": int(metric_row["input_points"]),
                "output_points": int(metric_row["output_points"]),
                "effective_upsampling_ratio": float(metric_row["upsampling_ratio"]),
                "output_invalid_xyz_points": int(metric_row["output_invalid_xyz_points"]),
                "input_out_of_range_points": int(metric_row.get("input_out_of_range_points", 0) or 0),
                "output_out_of_range_points": int(metric_row["output_out_of_range_points"]),
                "exact_duplicate_points": int(float(metric_row.get("exact_duplicate_points", 0) or 0)),
                "near_duplicate_ratio": float(metric_row["near_duplicate_ratio"]),
                "output_to_input_nn_mean": float(metric_row["output_to_input_nn_mean"]),
                "output_to_input_nn_median": float(metric_row["output_to_input_nn_median"]),
                "output_to_input_nn_min": float(metric_row["output_to_input_nn_min"]),
                "output_to_input_nn_max": float(metric_row["output_to_input_nn_max"]),
                "x_min_before": float(metric_row["x_min_before"]),
                "x_max_before": float(metric_row["x_max_before"]),
                "y_min_before": float(metric_row["y_min_before"]),
                "y_max_before": float(metric_row["y_max_before"]),
                "z_min_before": float(metric_row["z_min_before"]),
                "z_max_before": float(metric_row["z_max_before"]),
                "x_min_after": float(metric_row["x_min_after"]),
                "x_max_after": float(metric_row["x_max_after"]),
                "y_min_after": float(metric_row["y_min_after"]),
                "y_max_after": float(metric_row["y_max_after"]),
                "z_min_after": float(metric_row["z_min_after"]),
                "z_max_after": float(metric_row["z_max_after"]),
                "local_neighbor_count_mean": float(metric_row["output_local_neighbor_count_mean"]),
                "reconstruction_runtime_sec": reconstruction_runtime,
                "postprocess_runtime_sec": postprocess_runtime,
                "total_runtime_sec": reconstruction_runtime + postprocess_runtime,
                "refresh_runtime_sec": float(refresh_runtime),
                "visualization_dir": str((args.visualization_root / frame_id).resolve()),
                "metrics_dir": str(metrics_root.resolve()),
                "suspicious": "no",
                "suspicious_reason": "",
            }
            if row["output_invalid_xyz_points"] > 0 or row["output_out_of_range_points"] > 0 or row["near_duplicate_ratio"] > 0.5:
                row["suspicious"] = "yes"
                reasons = []
                if row["output_invalid_xyz_points"] > 0:
                    reasons.append("invalid_points")
                if row["output_out_of_range_points"] > 0:
                    reasons.append("out_of_range")
                if row["near_duplicate_ratio"] > 0.5:
                    reasons.append("near_duplicate_ratio_gt_0.5")
                row["suspicious_reason"] = ",".join(reasons)
                suspicious_frames.append(frame_id)

            write_json(frame_root / "batch_manifest.json", row)
            batch_rows.append(row)
        except subprocess.TimeoutExpired as exc:
            failed_frames.append(frame_id)
            write_json(
                frame_root / "batch_failure.json",
                {
                    "frame_id": frame_id,
                    "error": "postprocess_timeout",
                    "timeout_sec": args.postprocess_timeout_sec,
                    "command": " ".join(str(part) for part in exc.cmd),
                },
            )
        except Exception as exc:
            failed_frames.append(frame_id)
            write_json(frame_root / "batch_failure.json", {"frame_id": frame_id, "error": str(exc)})

    summary_csv = args.work_root / "filter_range_dedup_voxel_003_batch_summary.csv"
    write_csv(summary_csv, batch_rows)

    summary_lines = [
        "# PU-GCN Batch Validation Summary: filter_range_dedup_voxel_003",
        "",
        "- processed_frames: %d" % len(batch_rows),
        "- requested_frames: %d" % len(frame_ids),
        "- failed_frames: %s" % (", ".join(failed_frames) if failed_frames else "none"),
        "- visually_suspicious_frames: %s" % (", ".join(suspicious_frames) if suspicious_frames else "none"),
    ]
    if batch_rows:
        summary_lines.extend(
            [
                "- average_input_points: %s" % fmt(average([float(row["input_points"]) for row in batch_rows])),
                "- average_output_points: %s" % fmt(average([float(row["output_points"]) for row in batch_rows])),
                "- average_effective_upsampling_ratio: %s" % fmt(average([float(row["effective_upsampling_ratio"]) for row in batch_rows])),
                "- average_runtime_sec: %s" % fmt(average([float(row["total_runtime_sec"]) for row in batch_rows])),
                "- total_invalid_points: %s" % fmt(sum(int(row["output_invalid_xyz_points"]) for row in batch_rows)),
                "- total_out_of_range_points: %s" % fmt(sum(int(row["output_out_of_range_points"]) for row in batch_rows)),
                "- average_near_duplicate_ratio: %s" % fmt(average([float(row["near_duplicate_ratio"]) for row in batch_rows])),
                "",
                "| Frame | Input points | Output points | Eff. ratio | Invalid | Out-of-range | Exact dup | Near dup ratio | Reconstruction sec | Postprocess sec | Total sec | Suspicious |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in batch_rows:
            summary_lines.append(
                "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    row["frame_id"],
                    fmt(row["input_points"]),
                    fmt(row["output_points"]),
                    fmt(row["effective_upsampling_ratio"]),
                    fmt(row["output_invalid_xyz_points"]),
                    fmt(row["output_out_of_range_points"]),
                    fmt(row["exact_duplicate_points"]),
                    fmt(row["near_duplicate_ratio"]),
                    fmt(row["reconstruction_runtime_sec"]),
                    fmt(row["postprocess_runtime_sec"]),
                    fmt(row["total_runtime_sec"]),
                    row["suspicious_reason"] or "no",
                )
            )
    (args.work_root / "filter_range_dedup_voxel_003_batch_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    build_method_comparison(batch_rows, frame_ids, args.work_root)

    print("Processed frames:", len(batch_rows))
    print("Failed frames:", failed_frames if failed_frames else "none")
    print("Final detector-facing outputs:", args.output_velodyne)
    print("Batch summary:", args.work_root / "filter_range_dedup_voxel_003_batch_summary.md")
    if failed_frames:
        sys.exit(1)


if __name__ == "__main__":
    main()
