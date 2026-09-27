#!/usr/bin/env python3
"""Export and merge PointRCNN's true pre-RCNN RPN proposals.

The export is GT-free: it reruns the frozen detector with ``--save_result``
on the already frozen lossless split-region baseline inputs, keeps proposals
whose centers belong to each disjoint core, and writes one KITTI-format
proposal file per physical frame.  The last column remains the raw RPN logit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / "venv_pointrcnn/bin/python"
TOOLS = REPO / "tools"
IMAGESETS = REPO / "data/KITTI/ImageSets"
TRAINING = REPO / "data/KITTI/object/training"
VELODYNE = TRAINING / "velodyne"
DEFAULT_SOURCE_WORKSPACE = REPO / "results/pointrcnn_split_region_object_preserving256_v1_20260804"
DEFAULT_OUTPUT = REPO / "results/detector_aware_pdans_v3_20260805/internal_proposals/pointrcnn_rpn"


def read_frames(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def record_velodyne() -> str:
    if VELODYNE.is_symlink():
        return os.readlink(VELODYNE)
    if VELODYNE.exists():
        return "EXISTS_NON_SYMLINK"
    return "ABSENT"


def restore_velodyne(previous: str) -> None:
    if previous == "EXISTS_NON_SYMLINK":
        raise RuntimeError("original velodyne was a non-symlink; refusing automatic restore")
    if VELODYNE.is_symlink() or VELODYNE.exists():
        if VELODYNE.is_dir() and not VELODYNE.is_symlink():
            raise RuntimeError("refusing to replace non-symlink velodyne directory")
        VELODYNE.unlink(missing_ok=True)
    if previous != "ABSENT":
        VELODYNE.symlink_to(previous)


def split_link(split_file: Path, token: str) -> tuple[Path, bool]:
    digest = hashlib.sha256(str(split_file.resolve()).encode()).hexdigest()[:10]
    link = IMAGESETS / f"rpnv3_{token}_{digest}.txt"
    if link.is_symlink():
        if link.resolve() != split_file.resolve():
            raise RuntimeError(f"unexpected existing split link: {link}")
        return link, False
    if link.exists():
        raise RuntimeError(f"refusing to replace existing split file: {link}")
    link.symlink_to(split_file.resolve())
    return link, True


def result_dir(run_root: Path, split_name: str) -> Path:
    return run_root / "inference/eval/epoch_no_number" / split_name


def complete(path: Path, frames: list[str]) -> bool:
    proposal_dir = path / "roi_result/data"
    return proposal_dir.is_dir() and all((proposal_dir / f"{frame}.txt").is_file() for frame in frames)


def run_slot(input_dir: Path, split_file: Path, run_root: Path) -> dict:
    frames = read_frames(split_file)
    link, created = split_link(split_file, split_file.stem.replace("_", "")[:18])
    target_result = result_dir(run_root, link.stem)
    try:
        if complete(target_result, frames):
            return {"status": "REUSED", "frames": len(frames), "split_name": link.stem, "runtime_seconds": 0.0}
        inference = run_root / "inference"
        if inference.exists():
            shutil.rmtree(inference)
        run_root.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON), "eval_rcnn_official.py",
            "--cfg_file", "cfgs/default.yaml",
            "--ckpt", "PointRCNN.pth",
            "--batch_size", "1", "--workers", "0",
            "--eval_mode", "rcnn", "--save_result",
            "--output_dir", str(inference),
            "--set", "RPN.LOC_XZ_FINE", "False", "TEST.SPLIT", link.stem,
        ]
        (run_root / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        env = os.environ.copy()
        env["NUMBA_ENABLE_CUDASIM"] = "1"
        env["POINT_RCNN_SKIP_PYTHON_AP"] = "1"
        previous = record_velodyne()
        started = time.monotonic()
        try:
            if VELODYNE.is_symlink() or VELODYNE.exists():
                if VELODYNE.is_dir() and not VELODYNE.is_symlink():
                    raise RuntimeError("refusing to replace non-symlink velodyne directory")
                VELODYNE.unlink(missing_ok=True)
            VELODYNE.symlink_to(input_dir.resolve())
            with (run_root / "stdout_stderr.log").open("wb") as log:
                proc = subprocess.run(command, cwd=TOOLS, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        finally:
            restore_velodyne(previous)
        elapsed = time.monotonic() - started
        (run_root / "returncode.txt").write_text(f"{proc.returncode}\n")
        (run_root / "runtime_seconds.txt").write_text(f"{elapsed:.3f}\n")
        if not complete(target_result, frames):
            tail = "\n".join((run_root / "stdout_stderr.log").read_text(errors="replace").splitlines()[-80:])
            raise RuntimeError(f"incomplete RPN export for {split_file.stem}:\n{tail}")
        return {"status": "DONE", "frames": len(frames), "split_name": link.stem,
                "runtime_seconds": elapsed, "returncode": proc.returncode}
    finally:
        if created:
            link.unlink(missing_ok=True)


def load_regions(path: Path) -> dict[tuple[str, int], tuple[float, float, float, float]]:
    regions = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            regions[(row["frame_id"], int(row["slot"]))] = (
                float(row["x_min_rect_m"]), float(row["x_max_rect_m"]),
                float(row["z_min_rect_m"]), float(row["z_max_rect_m"]),
            )
    return regions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-workspace", type=Path, default=DEFAULT_SOURCE_WORKSPACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20, help="0 means all protocol frames")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source = args.source_workspace.resolve()
    output = args.output.resolve()
    protocol = json.loads((source / "protocol.json").read_text())
    all_frames = read_frames(Path(protocol["split_file"]))
    frames = all_frames if args.limit == 0 else all_frames[: args.limit]
    frame_set = set(frames)
    filtered_root = output / "splits"
    run_root = output / "slot_runs"
    filtered_root.mkdir(parents=True, exist_ok=True)
    records = []

    for original_split in sorted((source / "splits/original_baseline").glob("slot_*.txt")):
        selected = [frame for frame in read_frames(original_split) if frame in frame_set]
        if not selected:
            continue
        filtered = filtered_root / original_split.name
        filtered.write_text("\n".join(selected) + "\n")
        slot = original_split.stem
        record = run_slot(source / "inputs/original_baseline" / slot, filtered, run_root / slot)
        record["slot"] = int(slot.split("_")[-1])
        records.append(record)
        print(f"RPN_EXPORT {slot} frames={len(selected)} status={record['status']}", flush=True)

    regions = load_regions(source / "manifests/regions.csv")
    merged = output / "merged_kitti"
    merged.mkdir(parents=True, exist_ok=True)
    raw_count = owned_count = 0
    for frame in frames:
        owned_rows: list[str] = []
        for record in records:
            slot = int(record["slot"])
            bounds = regions.get((frame, slot))
            if bounds is None:
                continue
            proposal_file = result_dir(run_root / f"slot_{slot:03d}", str(record["split_name"])) / "roi_result/data" / f"{frame}.txt"
            if not proposal_file.is_file():
                continue
            x_min, x_max, z_min, z_max = bounds
            for line in proposal_file.read_text().splitlines():
                fields = line.split()
                if len(fields) < 16:
                    continue
                raw_count += 1
                center_x, center_z = float(fields[11]), float(fields[13])
                if x_min <= center_x < x_max and z_min <= center_z < z_max:
                    owned_rows.append(" ".join(fields))
                    owned_count += 1
        (merged / f"{frame}.txt").write_text("\n".join(owned_rows) + ("\n" if owned_rows else ""))

    summary = {
        "status": "PASS",
        "stage": "PointRCNN true RPN proposals before RCNN refinement",
        "uses_gt": False,
        "checkpoint": str(TOOLS / "PointRCNN.pth"),
        "frames": len(frames),
        "source_workspace": str(source),
        "raw_slot_proposals": raw_count,
        "core_owned_proposals": owned_count,
        "mean_owned_proposals_per_frame": owned_count / len(frames),
        "score_semantics": "last KITTI column is raw RPN logit; selector applies sigmoid threshold",
        "merged_proposal_dir": str(merged),
        "slot_runs": records,
    }
    (output / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
