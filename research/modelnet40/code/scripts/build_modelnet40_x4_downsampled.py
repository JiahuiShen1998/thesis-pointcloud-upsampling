#!/usr/bin/env python3
"""Build deterministic ModelNet40 downsampled ×4 dataset (N → N/4)."""

from __future__ import annotations

import argparse
import csv
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    GLOBAL_SEED,
    ORIGINAL_ROOT,
    UP_FACTOR,
    detect_original_point_count,
    farthest_point_sample,
    load_class_to_idx,
    reports_dir,
    resample_to_target,
    stable_seed,
)

SPLITS = ("train", "test")


@dataclass(frozen=True)
class Task:
    split: str
    class_name: str
    shape_id: str
    label: int
    source_path: str
    output_path: str
    seed: int
    original_points: int
    target_points: int
    method: str


def discover_tasks(
    input_root: Path,
    output_root: Path,
    class_to_idx: dict[str, int],
    original_points: int,
    target_points: int,
    downsample_method: str,
) -> list[Task]:
    tasks: list[Task] = []
    for split in SPLITS:
        split_dir = input_root / split
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            class_name = class_dir.name
            label = class_to_idx[class_name]
            for npy_path in sorted(class_dir.glob("*.npy")):
                shape_id = npy_path.stem
                out_path = output_root / split / class_name / f"{shape_id}.npy"
                seed = stable_seed(GLOBAL_SEED, "downsample_x4", split, class_name, shape_id)
                tasks.append(
                    Task(
                        split=split,
                        class_name=class_name,
                        shape_id=shape_id,
                        label=label,
                        source_path=str(npy_path),
                        output_path=str(out_path),
                        seed=seed,
                        original_points=original_points,
                        target_points=target_points,
                        method=downsample_method,
                    )
                )
    return tasks


def process_one(task: Task) -> dict:
    try:
        points = np.load(task.source_path).astype(np.float32)
        if points.shape[0] != task.original_points:
            raise ValueError(f"Expected {task.original_points} pts, got {points.shape[0]}")
        if task.method == "fps":
            out = farthest_point_sample(points, task.target_points, task.seed)
        else:
            out = resample_to_target(points, task.target_points, task.seed)
        if out.shape != (task.target_points, 3):
            raise ValueError(f"Bad output shape {out.shape}")
        if not np.isfinite(out).all():
            raise ValueError("Non-finite output")
        out_path = Path(task.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, out)
        return {
            "status": "ok",
            "split": task.split,
            "class_name": task.class_name,
            "shape_id": task.shape_id,
            "label": task.label,
            "source_path": task.source_path,
            "output_path": str(out_path),
            "original_points": task.original_points,
            "target_points": task.target_points,
            "downsample_method": task.method,
            "seed": task.seed,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "split": task.split,
            "class_name": task.class_name,
            "shape_id": task.shape_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def copy_metadata(input_root: Path, output_root: Path) -> None:
    meta_in = input_root / "metadata"
    meta_out = output_root / "metadata"
    meta_out.mkdir(parents=True, exist_ok=True)
    for name in ("class_to_idx.json", "idx_to_class.json"):
        shutil.copy2(meta_in / name, meta_out / name)


def write_manifests(output_root: Path, ok_rows: list[dict]) -> None:
    fields = [
        "split", "class_name", "label", "shape_id", "source_path", "output_path",
        "original_points", "target_points", "downsample_method", "seed", "status",
    ]
    meta = output_root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = [r for r in ok_rows if r["split"] == split]
        with open(meta / f"{split}_manifest.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=ORIGINAL_ROOT)
    parser.add_argument("--output-root", type=Path, default=DOWNSAMPLED_X4_ROOT)
    parser.add_argument("--method", choices=("fps", "random"), default="random",
                        help="fps=deterministic farthest-point; random=deterministic random (faster)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    n = detect_original_point_count(args.input_root)
    target = n // UP_FACTOR
    class_to_idx = load_class_to_idx()

    if args.skip_existing and args.output_root.is_dir():
        existing = sum(1 for _ in args.output_root.rglob("*.npy"))
        if existing >= 12311:
            print(f"Skip: {args.output_root} already has {existing} files")
            return 0

    tasks = discover_tasks(
        args.input_root, args.output_root, class_to_idx, n, target, args.method
    )
    ok_rows: list[dict] = []
    failed: list[dict] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            row = fut.result()
            if row["status"] == "ok":
                ok_rows.append(row)
            else:
                failed.append(row)

    copy_metadata(args.input_root, args.output_root)
    write_manifests(args.output_root, ok_rows)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rep = reports_dir()
    rep.mkdir(parents=True, exist_ok=True)
    audit_csv = rep / "modelnet40_x4_downsampled_build_audit.csv"
    with open(audit_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ok_rows[0].keys()) if ok_rows else ["status"])
        writer.writeheader()
        for row in ok_rows + failed:
            writer.writerow(row)

    md = rep / "modelnet40_x4_downsampled_build_report.md"
    md.write_text(
        "\n".join([
            "# ModelNet40 Downsampled ×4 Build Report",
            "",
            f"- Generated at: {ts}",
            f"- Input: `{args.input_root}`",
            f"- Output: `{args.output_root}`",
            f"- Original N: {n}",
            f"- Target N/4: {target}",
            f"- Downsample method: **{args.method}** (seed={GLOBAL_SEED})",
            f"- Success: {len(ok_rows)}",
            f"- Failed: {len(failed)}",
            "",
            "Note: deterministic random sampling used if method=random; documented in report.",
        ]),
        encoding="utf-8",
    )
    print(f"Built {len(ok_rows)} files, failed {len(failed)}, method={args.method}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
