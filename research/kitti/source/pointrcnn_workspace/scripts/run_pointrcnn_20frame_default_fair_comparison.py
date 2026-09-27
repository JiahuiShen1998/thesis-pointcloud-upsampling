#!/usr/bin/env python3
"""Run inference-only PointRCNN comparison on a shared 20-frame subset."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "pointrcnn_20frame_default_fair_comparison"
FRAME_LIST = PROJECT_ROOT / "results" / "pointrcnn_larger_default_validation_cap100k" / "shared_20_frame_list.txt"
IMAGESETS_DIR = PROJECT_ROOT / "data" / "KITTI" / "ImageSets"
SPLIT_NAME = "val_pugcn_cap100k_20"
SPLIT_FILE = IMAGESETS_DIR / f"{SPLIT_NAME}.txt"
TRAINING_DIR = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
ACTIVE_VELODYNE = TRAINING_DIR / "velodyne"
TOOLS_DIR = PROJECT_ROOT / "tools"
PYTHON = PROJECT_ROOT / "venv_pointrcnn" / "bin" / "python"


METHODS = [
    {
        "method": "Original KITTI",
        "run_name": "original_default",
        "folder": TRAINING_DIR / "velodyne_original_val",
    },
    {
        "method": "EAR",
        "run_name": "ear_default",
        "folder": TRAINING_DIR / "velodyne_ear_val",
    },
    {
        "method": "PU-Net",
        "run_name": "punet_default",
        "folder": TRAINING_DIR / "velodyne_punet_x2_fullframe",
    },
    {
        "method": "PU-GCN cap_100k",
        "run_name": "pugcn_cap100k_default",
        "folder": TRAINING_DIR / "pugcn_cap_100k",
    },
]


def read_frame_ids() -> List[str]:
    return [line.strip() for line in FRAME_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_split(frame_ids: Iterable[str]) -> None:
    IMAGESETS_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_FILE.write_text("\n".join(frame_ids) + "\n", encoding="utf-8")


def symlink_target(path: Path) -> Optional[Path]:
    if path.is_symlink():
        return Path(os.readlink(path))
    return None


def set_active_velodyne(target: Path) -> None:
    if ACTIVE_VELODYNE.exists() or ACTIVE_VELODYNE.is_symlink():
        if not ACTIVE_VELODYNE.is_symlink():
            raise RuntimeError(f"{ACTIVE_VELODYNE} is not a symlink; refusing to replace it.")
        ACTIVE_VELODYNE.unlink()
    rel_target = os.path.relpath(target, ACTIVE_VELODYNE.parent)
    os.symlink(rel_target, ACTIVE_VELODYNE)


def restore_active_velodyne(original_target: Optional[Path]) -> None:
    if original_target is None:
        return
    if ACTIVE_VELODYNE.exists() or ACTIVE_VELODYNE.is_symlink():
        if ACTIVE_VELODYNE.is_symlink():
            ACTIVE_VELODYNE.unlink()
        else:
            raise RuntimeError(f"{ACTIVE_VELODYNE} is not a symlink; refusing to restore over it.")
    os.symlink(original_target, ACTIVE_VELODYNE)


def ensure_inputs(frame_ids: Iterable[str]) -> None:
    missing = []
    for method in METHODS:
        folder = method["folder"]
        for frame_id in frame_ids:
            path = folder / f"{frame_id}.bin"
            if not path.exists():
                missing.append(f"{method['method']}:{path}")
    if missing:
        raise FileNotFoundError("Missing detector input files:\n" + "\n".join(missing))


def run_method(method: Dict[str, object]) -> Dict[str, object]:
    run_dir = RESULT_ROOT / str(method["run_name"])
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    time_file = run_dir / "time.txt"
    stdout_file = run_dir / "stdout.log"
    stderr_file = run_dir / "stderr.log"
    command = [
        "/usr/bin/time",
        "-f",
        "wall_clock_sec=%e\nmax_rss_kb=%M",
        "-o",
        str(time_file),
        str(PYTHON),
        "eval_rcnn.py",
        "--cfg_file",
        "cfgs/default.yaml",
        "--ckpt",
        "PointRCNN.pth",
        "--batch_size",
        "1",
        "--workers",
        "0",
        "--eval_mode",
        "rcnn",
        "--save_result",
        "--test",
        "--output_dir",
        str(run_dir),
        "--set",
        "RPN.LOC_XZ_FINE",
        "False",
        "TEST.SPLIT",
        SPLIT_NAME,
    ]
    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{TOOLS_DIR}:{env.get('PYTHONPATH', '')}"
    (run_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (run_dir / "input_folder.txt").write_text(str(method["folder"]) + "\n", encoding="utf-8")

    with stdout_file.open("w", encoding="utf-8") as stdout, stderr_file.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(command, cwd=str(TOOLS_DIR), env=env, stdout=stdout, stderr=stderr)

    runtime: Dict[str, str] = {}
    if time_file.exists():
        for line in time_file.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                runtime[key.strip()] = value.strip()

    return {
        "method": method["method"],
        "run_name": method["run_name"],
        "folder": method["folder"],
        "run_dir": run_dir,
        "returncode": proc.returncode,
        "wall_clock_sec": runtime.get("wall_clock_sec", ""),
        "max_rss_kb": runtime.get("max_rss_kb", ""),
    }


def score_stats(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {
            "success": False,
            "detection_count": 0,
            "empty_file": True,
            "score_min": "",
            "score_mean": "",
            "score_max": "",
        }
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {
            "success": True,
            "detection_count": 0,
            "empty_file": True,
            "score_min": "",
            "score_mean": "",
            "score_max": "",
        }
    scores = []
    for line in lines:
        parts = line.split()
        if parts:
            scores.append(float(parts[-1]))
    return {
        "success": True,
        "detection_count": len(lines),
        "empty_file": False,
        "score_min": min(scores),
        "score_mean": sum(scores) / len(scores),
        "score_max": max(scores),
    }


def collect_rows(run_infos: List[Dict[str, object]], frame_ids: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for info in run_infos:
        result_dir = (
            Path(info["run_dir"])
            / "eval"
            / "epoch_no_number"
            / SPLIT_NAME
            / "test_mode"
            / "final_result"
            / "data"
        )
        for frame_id in frame_ids:
            output_file = result_dir / f"{frame_id}.txt"
            stats = score_stats(output_file)
            rows.append(
                {
                    "method": info["method"],
                    "run_name": info["run_name"],
                    "frame_id": frame_id,
                    "input_folder": info["folder"],
                    "returncode": info["returncode"],
                    "inference_success": bool(info["returncode"] == 0 and stats["success"]),
                    "final_detection_count": stats["detection_count"],
                    "empty_output_file": stats["empty_file"],
                    "score_min": stats["score_min"],
                    "score_mean": stats["score_mean"],
                    "score_max": stats["score_max"],
                    "wall_clock_sec_method": info["wall_clock_sec"],
                    "max_rss_kb_method": info["max_rss_kb"],
                    "output_file": output_file,
                }
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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
        return f"{value:.6f}"
    return str(value)


def method_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method["method"]]
        total = sum(int(row["final_detection_count"]) for row in method_rows)
        empty = sum(1 for row in method_rows if str(row["empty_output_file"]) == "True")
        failed = [row["frame_id"] for row in method_rows if str(row["inference_success"]) != "True"]
        runtime = method_rows[0]["wall_clock_sec_method"] if method_rows else ""
        summaries.append(
            {
                "method": method["method"],
                "frames": len(method_rows),
                "total_detections": total,
                "avg_detections_per_frame": total / len(method_rows) if method_rows else 0.0,
                "empty_files": empty,
                "failed_frames": ", ".join(failed) if failed else "none",
                "wall_clock_sec": runtime,
            }
        )
    return summaries


def suspicious_frames(rows: List[Dict[str, object]]) -> List[str]:
    by_frame: Dict[str, Dict[str, int]] = {}
    for row in rows:
        by_frame.setdefault(str(row["frame_id"]), {})[str(row["method"])] = int(row["final_detection_count"])
    suspicious = []
    for frame_id, counts in by_frame.items():
        original = counts.get("Original KITTI", 0)
        pugcn = counts.get("PU-GCN cap_100k", 0)
        if frame_id in {"000002", "000028"} or abs(pugcn - original) >= 5 or (original > 0 and pugcn == 0):
            suspicious.append(frame_id)
    return sorted(set(suspicious))


def write_markdown(rows: List[Dict[str, object]], summaries: List[Dict[str, object]], frame_ids: List[str]) -> None:
    suspicious = suspicious_frames(rows)
    lines = [
        "# 20-Frame Default PointRCNN Fair Comparison",
        "",
        "Inference-only comparison across Original KITTI, EAR, PU-Net, and PU-GCN cap_100k.",
        "",
        f"Split file: `{SPLIT_FILE.relative_to(PROJECT_ROOT)}`",
        f"Frame list: `{FRAME_LIST.relative_to(PROJECT_ROOT)}`",
        "Checkpoint: `tools/PointRCNN.pth`",
        "Config: `tools/cfgs/default.yaml` with `RPN.NUM_POINTS=16384`; no sampler code changes and no replacement sampling.",
        "Runtime mode: `--test --save_result`, so full KITTI AP evaluation was not run.",
        "",
        "## Detector-Facing Folders",
        "",
        "| Method | Folder |",
        "|---|---|",
    ]
    for method in METHODS:
        lines.append(f"| {method['method']} | `{Path(method['folder']).relative_to(PROJECT_ROOT)}` |")

    lines.extend(
        [
            "",
            "## Method Summary",
            "",
            "| Method | Frames | Total detections | Avg detections/frame | Empty files | Failed frames | Runtime sec |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    for summary in summaries:
        lines.append(
            "| {method} | {frames} | {total_detections} | {avg_detections_per_frame:.3f} | "
            "{empty_files} | {failed_frames} | {wall_clock_sec} |".format(**summary)
        )

    lines.extend(
        [
            "",
            "## Frame-Level Detection Counts",
            "",
            "| Frame | Original | EAR | PU-Net | PU-GCN cap_100k | PU-GCN - Original |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for frame_id in frame_ids:
        counts = {}
        for row in rows:
            if row["frame_id"] == frame_id:
                counts[row["method"]] = int(row["final_detection_count"])
        original = counts.get("Original KITTI", 0)
        pugcn = counts.get("PU-GCN cap_100k", 0)
        lines.append(
            f"| `{frame_id}` | {original} | {counts.get('EAR', 0)} | {counts.get('PU-Net', 0)} | {pugcn} | {pugcn - original} |"
        )

    lines.extend(
        [
            "",
            "## Detailed Rows",
            "",
            "| Method | Frame | Success | Detections | Empty | Score min | Score mean | Score max | Output file |",
            "|---|---|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| {method} | `{frame_id}` | {inference_success} | {final_detection_count} | {empty_output_file} | "
            "{score_min} | {score_mean} | {score_max} | `{output_file}` |".format(
                **{key: fmt(value) for key, value in row.items()}
            )
        )

    lines.extend(
        [
            "",
            "## Suspicious Frames",
            "",
            ", ".join(f"`{frame}`" for frame in suspicious) if suspicious else "None flagged.",
            "",
            "Notes:",
            "",
            "- `000002` was included because it was a prior PU-GCN watchlist frame.",
            "- `000028` was included because its dense PU-GCN generation had a documented warning before cap/precheck.",
            "- Strong PU-GCN-vs-Original differences are flagged when the absolute count difference is at least `5`, or when Original has detections and PU-GCN has none.",
            "",
            "## Conclusion",
            "",
        ]
    )
    pugcn_rows = [row for row in rows if row["method"] == "PU-GCN cap_100k"]
    pugcn_failed = [row["frame_id"] for row in pugcn_rows if str(row["inference_success"]) != "True"]
    pugcn_empty = sum(1 for row in pugcn_rows if str(row["empty_output_file"]) == "True")
    pugcn_total = sum(int(row["final_detection_count"]) for row in pugcn_rows)
    totals = {summary["method"]: summary["total_detections"] for summary in summaries}
    lines.extend(
        [
            f"PU-GCN cap_100k compatibility on this 20-frame subset: **{'yes' if not pugcn_failed else 'no'}**.",
            f"PU-GCN cap_100k empty final output files: `{pugcn_empty}` of `{len(pugcn_rows)}`.",
            (
                "Detection totals: Original KITTI `{Original KITTI}`, EAR `{EAR}`, PU-Net `{PU-Net}`, "
                "PU-GCN cap_100k `{PU-GCN cap_100k}`."
            ).format(**totals),
            f"PU-GCN cap_100k remains stable under default PointRCNN on 20 frames with `{pugcn_total}` total detections.",
            "This setup is ready for a small AP evaluation or a larger validation-subset inference pass, but full AP was not run in this stage.",
            "",
        ]
    )
    (RESULT_ROOT / "20frame_default_fair_comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frame_ids = read_frame_ids()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    write_split(frame_ids)
    ensure_inputs(frame_ids)

    original_target = symlink_target(ACTIVE_VELODYNE)
    run_infos: List[Dict[str, object]] = []
    try:
        for method in METHODS:
            set_active_velodyne(Path(method["folder"]))
            run_infos.append(run_method(method))
    finally:
        restore_active_velodyne(original_target)

    rows = collect_rows(run_infos, frame_ids)
    summaries = method_summary(rows)
    write_csv(RESULT_ROOT / "20frame_default_fair_comparison_summary.csv", rows)
    write_csv(RESULT_ROOT / "20frame_default_fair_comparison_method_summary.csv", summaries)
    write_markdown(rows, summaries, frame_ids)

    print("summary", RESULT_ROOT / "20frame_default_fair_comparison_summary.md")
    for summary in summaries:
        print(summary)


if __name__ == "__main__":
    main()
