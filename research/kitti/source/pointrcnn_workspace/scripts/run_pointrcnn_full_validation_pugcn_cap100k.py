#!/usr/bin/env python3
"""Run full-validation PointRCNN inference for PU-GCN cap_100k under the default setting."""

from __future__ import annotations

import csv
import os
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = PROJECT_ROOT / "results" / "pointrcnn_full_validation_pugcn_cap100k"
TRAINING_DIR = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
ACTIVE_VELODYNE = TRAINING_DIR / "velodyne"
PUGCN_FOLDER = TRAINING_DIR / "pugcn_cap_100k"
FULL_SPLIT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
TOOLS_DIR = PROJECT_ROOT / "tools"
PYTHON = PROJECT_ROOT / "venv_pointrcnn" / "bin" / "python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--input-folder", type=Path, default=PUGCN_FOLDER)
    parser.add_argument("--full-split", type=Path, default=FULL_SPLIT)
    parser.add_argument("--split-name", type=str, default="val")
    parser.add_argument("--overwrite", action="store_true", default=True)
    return parser.parse_args()


def read_frame_ids(split_file: Path) -> List[str]:
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def symlink_target(path: Path) -> Optional[Path]:
    if path.is_symlink():
        return Path(os.readlink(path))
    return None


def set_active_velodyne(target: Path) -> None:
    if ACTIVE_VELODYNE.exists() or ACTIVE_VELODYNE.is_symlink():
        if not ACTIVE_VELODYNE.is_symlink():
            raise RuntimeError(f"{ACTIVE_VELODYNE} is not a symlink; refusing to replace it.")
        ACTIVE_VELODYNE.unlink()
    os.symlink(os.path.relpath(target, ACTIVE_VELODYNE.parent), ACTIVE_VELODYNE)


def restore_active_velodyne(original_target: Optional[Path]) -> None:
    if original_target is None:
        return
    if ACTIVE_VELODYNE.exists() or ACTIVE_VELODYNE.is_symlink():
        if ACTIVE_VELODYNE.is_symlink():
            ACTIVE_VELODYNE.unlink()
        else:
            raise RuntimeError(f"{ACTIVE_VELODYNE} is not a symlink; refusing to restore over it.")
    os.symlink(original_target, ACTIVE_VELODYNE)


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


def score_stats(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"exists": False, "count": 0, "empty": True, "score_min": "", "score_mean": "", "score_max": ""}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {"exists": True, "count": 0, "empty": True, "score_min": "", "score_mean": "", "score_max": ""}
    scores = [float(line.split()[-1]) for line in lines]
    return {
        "exists": True,
        "count": len(lines),
        "empty": False,
        "score_min": min(scores),
        "score_mean": sum(scores) / len(scores),
        "score_max": max(scores),
    }


def main() -> None:
    args = parse_args()
    frame_ids = read_frame_ids(args.full_split)
    missing = [frame_id for frame_id in frame_ids if not (args.input_folder / ("%s.bin" % frame_id)).exists()]
    if missing:
        raise RuntimeError("Missing %d PU-GCN cap_100k files; full inference is blocked" % len(missing))

    if args.result_root.exists() and args.overwrite:
        shutil.rmtree(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)

    command = [
        "/usr/bin/time",
        "-f",
        "wall_clock_sec=%e\nmax_rss_kb=%M",
        "-o",
        str(args.result_root / "time.txt"),
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
        str(args.result_root / "inference"),
        "--set",
        "RPN.LOC_XZ_FINE",
        "False",
        "TEST.SPLIT",
        args.split_name,
    ]
    (args.result_root / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
    (args.result_root / "input_folder.txt").write_text(str(args.input_folder) + "\n", encoding="utf-8")
    (args.result_root / "split_file.txt").write_text(str(args.full_split) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["NUMBA_ENABLE_CUDASIM"] = "1"
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{TOOLS_DIR}:{env.get('PYTHONPATH', '')}"

    original_target = symlink_target(ACTIVE_VELODYNE)
    try:
        set_active_velodyne(args.input_folder)
        with (args.result_root / "stdout.log").open("w", encoding="utf-8") as stdout, (args.result_root / "stderr.log").open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(command, cwd=str(TOOLS_DIR), env=env, stdout=stdout, stderr=stderr)
    finally:
        restore_active_velodyne(original_target)

    result_dir = args.result_root / "inference" / "eval" / "epoch_no_number" / args.split_name / "test_mode" / "final_result" / "data"
    rows = []
    for frame_id in frame_ids:
        stats = score_stats(result_dir / ("%s.txt" % frame_id))
        rows.append(
            {
                "frame_id": frame_id,
                "returncode": proc.returncode,
                "detection_file_exists": stats["exists"],
                "detection_count": stats["count"],
                "empty_file": stats["empty"],
                "score_min": stats["score_min"],
                "score_mean": stats["score_mean"],
                "score_max": stats["score_max"],
                "output_file": str(result_dir / ("%s.txt" % frame_id)),
            }
        )
    write_csv(args.result_root / "full_validation_inference_rows.csv", rows)
    print("frames", len(frame_ids))
    print("returncode", proc.returncode)
    print("detection_rows", len(rows))


if __name__ == "__main__":
    main()
