#!/usr/bin/env python3
"""Create ModelNet40 Downsampled50 dataset from original 1024-point .npy files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
DEFAULT_INPUT = PROJECT_ROOT / "datasets" / "modelnet40_original"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "step5_downsampled50_report.md"
DEFAULT_AUDIT_CSV = PROJECT_ROOT / "reports" / "step5_downsampled50_audit.csv"

ORIGINAL_POINTS = 1024
DOWNSAMPLED_POINTS = 512
GLOBAL_SEED = 42
SPLITS = ("train", "test")


@dataclass(frozen=True)
class DownsampleTask:
    split: str
    class_name: str
    shape_id: str
    label: int
    source_path: str
    output_path: str
    seed: int
    original_points: int
    downsampled_points: int


def per_file_seed(global_seed: int, split: str, class_name: str, shape_id: str) -> int:
    token = f"{global_seed}:{split}:{class_name}:{shape_id}".encode("utf-8")
    digest = hashlib.md5(token).hexdigest()
    return global_seed + (int(digest[:8], 16) % 1_000_000)


def discover_tasks(
    input_root: Path,
    output_root: Path,
    class_to_idx: dict[str, int],
    global_seed: int,
    original_points: int,
    downsampled_points: int,
) -> list[DownsampleTask]:
    tasks: list[DownsampleTask] = []
    for split in SPLITS:
        split_dir = input_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            class_name = class_dir.name
            if class_name not in class_to_idx:
                raise KeyError(f"Class {class_name} missing from class_to_idx.json")
            label = class_to_idx[class_name]
            for npy_path in sorted(class_dir.glob("*.npy")):
                shape_id = npy_path.stem
                out_path = output_root / split / class_name / f"{shape_id}.npy"
                seed = per_file_seed(global_seed, split, class_name, shape_id)
                tasks.append(
                    DownsampleTask(
                        split=split,
                        class_name=class_name,
                        shape_id=shape_id,
                        label=label,
                        source_path=str(npy_path),
                        output_path=str(out_path),
                        seed=seed,
                        original_points=original_points,
                        downsampled_points=downsampled_points,
                    )
                )
    return tasks


def process_one(task: DownsampleTask) -> dict:
    try:
        points = np.load(task.source_path).astype(np.float32)
        if points.shape != (task.original_points, 3):
            raise ValueError(f"Expected shape ({task.original_points}, 3), got {points.shape}")

        rng = np.random.default_rng(task.seed)
        indices = rng.choice(task.original_points, size=task.downsampled_points, replace=False)
        indices.sort()
        downsampled = points[indices].copy()

        if downsampled.shape != (task.downsampled_points, 3):
            raise ValueError(f"Output shape mismatch: {downsampled.shape}")
        if not np.isfinite(downsampled).all():
            raise ValueError("Non-finite values in downsampled output")

        out_path = Path(task.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, downsampled)

        return {
            "status": "ok",
            "split": task.split,
            "class_name": task.class_name,
            "shape_id": task.shape_id,
            "label": task.label,
            "source_path": task.source_path,
            "output_path": str(out_path),
            "original_points": task.original_points,
            "downsampled_points": task.downsampled_points,
            "seed": task.seed,
            "selected_indices_hash": int(hash(tuple(indices.tolist())) % (2**31 - 1)),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "split": task.split,
            "class_name": task.class_name,
            "shape_id": task.shape_id,
            "label": task.label,
            "source_path": task.source_path,
            "output_path": task.output_path,
            "original_points": task.original_points,
            "downsampled_points": task.downsampled_points,
            "seed": task.seed,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def copy_metadata(input_root: Path, output_root: Path) -> None:
    metadata_in = input_root / "metadata"
    metadata_out = output_root / "metadata"
    metadata_out.mkdir(parents=True, exist_ok=True)
    for name in ("class_to_idx.json", "idx_to_class.json"):
        shutil.copy2(metadata_in / name, metadata_out / name)


def write_manifests(output_root: Path, ok_rows: list[dict]) -> None:
    metadata_out = output_root / "metadata"
    metadata_out.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "class_name",
        "label",
        "shape_id",
        "source_path",
        "output_path",
        "original_points",
        "downsampled_points",
        "seed",
        "status",
    ]
    for split in SPLITS:
        split_rows = [row for row in ok_rows if row["split"] == split]
        manifest_path = metadata_out / f"{split}_manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in split_rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_failed_csv(output_root: Path, failed_rows: list[dict]) -> None:
    path = output_root / "metadata" / "failed_files.csv"
    fieldnames = ["split", "class_name", "shape_id", "source_path", "error"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in failed_rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_audit_csv(audit_path: Path, ok_rows: list[dict], failed_rows: list[dict]) -> None:
    per_split_class = defaultdict(lambda: Counter())
    per_split = Counter()
    for row in ok_rows:
        per_split[row["split"]] += 1
        per_split_class[row["split"]][row["class_name"]] += 1

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "key", "value"])
        writer.writerow(["summary", "train_count", per_split.get("train", 0)])
        writer.writerow(["summary", "test_count", per_split.get("test", 0)])
        writer.writerow(["summary", "total_count", len(ok_rows)])
        writer.writerow(["summary", "failed_count", len(failed_rows)])
        writer.writerow(["summary", "original_points", ORIGINAL_POINTS])
        writer.writerow(["summary", "downsampled_points", DOWNSAMPLED_POINTS])
        writer.writerow(["summary", "seed", GLOBAL_SEED])
        for split in SPLITS:
            for class_name in sorted(per_split_class[split]):
                writer.writerow(["per_class", f"{split}/{class_name}", per_split_class[split][class_name]])


def write_report(
    report_path: Path,
    input_root: Path,
    output_root: Path,
    ok_rows: list[dict],
    failed_rows: list[dict],
    elapsed_sec: float,
    seed: int,
) -> None:
    per_split_class = defaultdict(lambda: Counter())
    per_split = Counter()
    for row in ok_rows:
        per_split[row["split"]] += 1
        per_split_class[row["split"]][row["class_name"]] += 1

    shape_ok = 0
    nan_count = 0
    for row in ok_rows:
        arr = np.load(row["output_path"])
        if arr.shape == (DOWNSAMPLED_POINTS, 3):
            shape_ok += 1
        if not np.isfinite(arr).all():
            nan_count += 1

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# Step 5 — ModelNet40 Downsampled50 Report",
        "",
        f"- Generated at: {now}",
        f"- Input: `{input_root}`",
        f"- Output: `{output_root}`",
        f"- Random seed: {seed}",
        f"- Original points per sample: {ORIGINAL_POINTS}",
        f"- Downsampled points per sample: {DOWNSAMPLED_POINTS}",
        f"- Elapsed seconds: {elapsed_sec:.1f}",
        "",
        "## Summary",
        "",
        f"- Train samples: **{per_split.get('train', 0)}**",
        f"- Test samples: **{per_split.get('test', 0)}**",
        f"- Total successful: **{len(ok_rows)}**",
        f"- Failed samples: **{len(failed_rows)}**",
        f"- Class count: **40**",
        "",
        "## Audit",
        "",
        f"- Shape `(512, 3)` verified: {shape_ok}/{len(ok_rows)}",
        f"- Non-finite outputs: {nan_count}",
        f"- Status: {'PASS' if not failed_rows and shape_ok == len(ok_rows) and nan_count == 0 else 'FAIL'}",
        "",
        "## Per-split class counts",
        "",
    ]
    for split in SPLITS:
        lines.extend([f"### {split}", "", "| class | count |", "| --- | ---: |"])
        for class_name in sorted(per_split_class[split]):
            lines.append(f"| {class_name} | {per_split_class[split][class_name]} |")
        lines.append("")

    lines.extend(
        [
            "## Next step",
            "",
            "Wait for Step 4 Original baseline to finish, then proceed to Step 6: Downsampled50 baseline training.",
            "",
        ]
    )

    audit_json = {
        "generated_at": now,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": seed,
        "original_points": ORIGINAL_POINTS,
        "downsampled_points": DOWNSAMPLED_POINTS,
        "train_count": per_split.get("train", 0),
        "test_count": per_split.get("test", 0),
        "total_success": len(ok_rows),
        "failed_count": len(failed_rows),
        "per_split_class_counts": {split: dict(counter) for split, counter in per_split_class.items()},
    }
    with open(output_root / "metadata" / "dataset_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit_json, handle, indent=2, sort_keys=True)

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create ModelNet40 Downsampled50 dataset.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--original-points", type=int, default=ORIGINAL_POINTS)
    parser.add_argument("--downsampled-points", type=int, default=DOWNSAMPLED_POINTS)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()

    class_to_idx_path = input_root / "metadata" / "class_to_idx.json"
    if not class_to_idx_path.is_file():
        print(f"ERROR: missing {class_to_idx_path}", file=sys.stderr)
        return 1
    with open(class_to_idx_path, encoding="utf-8") as handle:
        class_to_idx = json.load(handle)

    tasks = discover_tasks(
        input_root,
        output_root,
        class_to_idx,
        args.seed,
        args.original_points,
        args.downsampled_points,
    )
    if not args.overwrite:
        tasks = [t for t in tasks if not Path(t.output_path).exists()]

    if not tasks:
        print("No tasks to process.")
        return 0

    print(f"Downsampling {len(tasks)} files with {args.num_workers} workers...")
    start = time.time()
    ok_rows: list[dict] = []
    failed_rows: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one, task): task for task in tasks}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result["status"] == "ok":
                ok_rows.append(result)
            else:
                failed_rows.append(result)
            if done % 500 == 0 or done == len(tasks):
                print(f"  progress: {done}/{len(tasks)} (ok={len(ok_rows)}, failed={len(failed_rows)})")

    elapsed = time.time() - start
    copy_metadata(input_root, output_root)
    write_manifests(output_root, ok_rows)
    write_failed_csv(output_root, failed_rows)
    write_audit_csv(args.audit_csv, ok_rows, failed_rows)
    write_report(args.report_path, input_root, output_root, ok_rows, failed_rows, elapsed, args.seed)

    print(f"Done. ok={len(ok_rows)} failed={len(failed_rows)} elapsed={elapsed:.1f}s")
    print(f"Report: {args.report_path}")
    return 0 if not failed_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
