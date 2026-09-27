#!/usr/bin/env python3
"""Run the PU-GCN single-frame adapter over a small KITTI frame batch."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_DIR = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data" / "KITTI" / "ImageSets" / "val.txt"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "kitti_pugcn"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small-batch PU-GCN KITTI processing wrapper.")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--input-velodyne", type=Path, default=DEFAULT_TRAINING_DIR / "velodyne_original_val")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-root", type=Path, default=PROJECT_ROOT / "results" / "pugcn_batch_work")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--pugcn-python", type=str, default=sys.executable)
    parser.add_argument("--restore", type=Path, default=PROJECT_ROOT / "external" / "PU-GCN" / "pretrained" / "pu1k-pugcn")
    return parser.parse_args()


def read_ids(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    ids = read_ids(args.split_file)[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    for frame_id in ids:
        input_bin = args.input_velodyne / f"{frame_id}.bin"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "pugcn_kitti_single_frame_test.py"),
            "--input-bin",
            str(input_bin),
            "--output-root",
            str(args.work_root),
            "--sample-id",
            frame_id,
            "--pugcn-python",
            args.pugcn_python,
            "--restore",
            str(args.restore),
        ]
        if args.run_inference:
            cmd.append("--run-inference")
        subprocess.run(cmd, check=True)
        produced = args.work_root / frame_id / f"{frame_id}_pugcn.bin"
        if produced.exists():
            target = args.output_root / f"{frame_id}.bin"
            target.write_bytes(produced.read_bytes())
    print("Processed PU-GCN batch workspace:", args.work_root)
    print("KITTI-compatible outputs, when inference succeeds:", args.output_root)


if __name__ == "__main__":
    main()
