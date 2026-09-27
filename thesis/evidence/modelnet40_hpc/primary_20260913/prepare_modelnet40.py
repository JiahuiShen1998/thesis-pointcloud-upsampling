#!/usr/bin/env python3
"""Prepare ModelNet40 for PointNet++ classification experiments.

Reads raw .off meshes from the official train/test directory layout,
samples 1024 points per shape (triangle area weighted when faces exist),
normalizes to unit sphere, and writes .npy files plus metadata manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_RAW_ROOT = (
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/"
    "datasets/raw_modelnet40/ModelNet40"
)
DEFAULT_OUTPUT_ROOT = (
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/"
    "datasets/modelnet40_original"
)
DEFAULT_REPORT_PATH = (
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling/"
    "reports/step2_modelnet40_prepare_report.md"
)
NUM_POINTS = 1024
SPLITS = ("train", "test")


@dataclass(frozen=True)
class OffMesh:
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3) int


@dataclass(frozen=True)
class ProcessTask:
    off_path: str
    class_name: str
    split: str
    shape_id: str
    output_path: str
    seed: int
    num_points: int


def parse_off_header(first_line: str, second_line: str | None) -> tuple[int, int, int, int]:
    """Parse OFF header supporting 'OFF' + next line counts or 'OFF n f e' on one line."""
    line = first_line.strip()
    if not line.upper().startswith("OFF"):
        raise ValueError(f"Not an OFF file: {first_line!r}")

    rest = line[3:].strip()
    if rest:
        parts = rest.split()
    else:
        if second_line is None:
            raise ValueError("Missing vertex/face counts after OFF header")
        parts = second_line.strip().split()

    if len(parts) < 2:
        raise ValueError(f"Invalid OFF counts: {parts}")

    n_vertices = int(parts[0])
    n_faces = int(parts[1])
    header_lines = 1 if rest else 2
    return n_vertices, n_faces, header_lines, len(parts)


def load_off_mesh(off_path: str) -> OffMesh:
    """Robustly load vertices and triangular faces from a ModelNet40 OFF file."""
    with open(off_path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    if not lines:
        raise ValueError("Empty OFF file")

    n_vertices, n_faces, header_lines, _ = parse_off_header(lines[0], lines[1] if len(lines) > 1 else None)
    start = header_lines

    if start + n_vertices > len(lines):
        raise ValueError(
            f"Expected {n_vertices} vertices starting at line {start + 1}, file has {len(lines)} lines"
        )

    vertices = np.empty((n_vertices, 3), dtype=np.float64)
    for i in range(n_vertices):
        parts = lines[start + i].split()
        if len(parts) < 3:
            raise ValueError(f"Bad vertex line {start + i + 1}: {lines[start + i]!r}")
        vertices[i, 0] = float(parts[0])
        vertices[i, 1] = float(parts[1])
        vertices[i, 2] = float(parts[2])

    face_start = start + n_vertices
    faces: list[list[int]] = []
    for i in range(n_faces):
        line_idx = face_start + i
        if line_idx >= len(lines):
            break
        parts = lines[line_idx].split()
        if not parts:
            continue
        face_size = int(parts[0])
        if face_size < 3:
            continue
        indices = [int(x) for x in parts[1 : 1 + face_size]]
        if face_size == 3:
            faces.append(indices)
        else:
            # Fan triangulation for n-gons.
            anchor = indices[0]
            for j in range(1, face_size - 1):
                faces.append([anchor, indices[j], indices[j + 1]])

    if faces:
        face_array = np.asarray(faces, dtype=np.int64)
    else:
        face_array = np.empty((0, 3), dtype=np.int64)

    return OffMesh(vertices=vertices, faces=face_array)


def normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    """Center point cloud at origin and scale into unit sphere."""
    centered = points - points.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(centered, axis=1)
    max_radius = float(radii.max()) if radii.size else 0.0
    if max_radius <= 0.0:
        raise ValueError("Degenerate point cloud after centering")
    return centered / max_radius


def sample_points_from_mesh(mesh: OffMesh, num_points: int, seed: int) -> np.ndarray:
    """Sample points; prefer triangle-area weighted surface sampling."""
    rng = np.random.default_rng(seed)
    vertices = mesh.vertices
    faces = mesh.faces

    if faces.size == 0:
        if vertices.shape[0] == 0:
            raise ValueError("Mesh has no vertices")
        if vertices.shape[0] >= num_points:
            idx = rng.choice(vertices.shape[0], size=num_points, replace=False)
            return vertices[idx].copy()
        idx = rng.choice(vertices.shape[0], size=num_points, replace=True)
        return vertices[idx].copy()

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    area_sum = float(areas.sum())
    if area_sum <= 0.0:
        if vertices.shape[0] >= num_points:
            idx = rng.choice(vertices.shape[0], size=num_points, replace=False)
            return vertices[idx].copy()
        idx = rng.choice(vertices.shape[0], size=num_points, replace=True)
        return vertices[idx].copy()

    probs = areas / area_sum
    chosen_faces = rng.choice(faces.shape[0], size=num_points, replace=True, p=probs)
    tri_v0 = vertices[faces[chosen_faces, 0]]
    tri_v1 = vertices[faces[chosen_faces, 1]]
    tri_v2 = vertices[faces[chosen_faces, 2]]

    r1 = rng.random(num_points)
    r2 = rng.random(num_points)
    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2
    return u[:, None] * tri_v0 + v[:, None] * tri_v1 + w[:, None] * tri_v2


def process_one(task: ProcessTask) -> dict:
    """Worker function: OFF -> sampled normalized npy."""
    try:
        mesh = load_off_mesh(task.off_path)
        points = sample_points_from_mesh(mesh, task.num_points, task.seed)
        points = normalize_unit_sphere(points)
        if points.shape != (task.num_points, 3):
            raise ValueError(f"Unexpected output shape {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("Non-finite values after normalization")

        output_path = Path(task.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, points.astype(np.float32))

        radius = float(np.linalg.norm(points, axis=1).max())
        return {
            "status": "ok",
            "class_name": task.class_name,
            "split": task.split,
            "shape_id": task.shape_id,
            "off_path": task.off_path,
            "output_path": str(output_path),
            "num_vertices": int(mesh.vertices.shape[0]),
            "num_faces": int(mesh.faces.shape[0]),
            "max_radius": radius,
        }
    except Exception as exc:  # noqa: BLE001 - collect per-file failures
        return {
            "status": "failed",
            "class_name": task.class_name,
            "split": task.split,
            "shape_id": task.shape_id,
            "off_path": task.off_path,
            "output_path": task.output_path,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def discover_tasks(raw_root: Path, output_root: Path, seed: int, num_points: int) -> list[ProcessTask]:
    tasks: list[ProcessTask] = []
    for class_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        class_name = class_dir.name
        for split in SPLITS:
            split_dir = class_dir / split
            if not split_dir.is_dir():
                continue
            for off_path in sorted(split_dir.glob("*.off")):
                shape_id = off_path.stem
                out_path = output_root / split / class_name / f"{shape_id}.npy"
                file_seed = seed + (hash((class_name, split, shape_id)) % 1_000_000)
                tasks.append(
                    ProcessTask(
                        off_path=str(off_path),
                        class_name=class_name,
                        split=split,
                        shape_id=shape_id,
                        output_path=str(out_path),
                        seed=file_seed,
                        num_points=num_points,
                    )
                )
    return tasks


def write_label_maps(output_root: Path, class_names: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    classes = sorted(set(class_names))
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_dir / "class_to_idx.json", "w", encoding="utf-8") as handle:
        json.dump(class_to_idx, handle, indent=2, sort_keys=True)
    with open(metadata_dir / "idx_to_class.json", "w", encoding="utf-8") as handle:
        json.dump({str(k): v for k, v in idx_to_class.items()}, handle, indent=2, sort_keys=True)
    return class_to_idx, idx_to_class


def write_manifest(output_root: Path, split: str, rows: list[dict], class_to_idx: dict[str, int]) -> None:
    manifest_path = output_root / "metadata" / f"{split}_manifest.csv"
    fieldnames = [
        "shape_id",
        "class_name",
        "label",
        "split",
        "source_off",
        "output_npy",
        "num_vertices",
        "num_faces",
        "max_radius",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "shape_id": row["shape_id"],
                    "class_name": row["class_name"],
                    "label": class_to_idx[row["class_name"]],
                    "split": split,
                    "source_off": row["off_path"],
                    "output_npy": row["output_path"],
                    "num_vertices": row.get("num_vertices", ""),
                    "num_faces": row.get("num_faces", ""),
                    "max_radius": row.get("max_radius", ""),
                }
            )


def write_failed_files(output_root: Path, failed_rows: list[dict]) -> None:
    failed_path = output_root / "metadata" / "failed_files.csv"
    fieldnames = ["class_name", "split", "shape_id", "off_path", "error"]
    with open(failed_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in failed_rows:
            writer.writerow(
                {
                    "class_name": row["class_name"],
                    "split": row["split"],
                    "shape_id": row["shape_id"],
                    "off_path": row["off_path"],
                    "error": row["error"],
                }
            )


def audit_outputs(output_root: Path, ok_rows: list[dict]) -> dict:
    """Post-run audit on saved npy files."""
    shape_issues: list[str] = []
    nan_issues: list[str] = []
    radius_values: list[float] = []

    for row in ok_rows:
        arr = np.load(row["output_path"])
        if arr.shape != (NUM_POINTS, 3):
            shape_issues.append(row["output_path"])
        if not np.isfinite(arr).all():
            nan_issues.append(row["output_path"])
        radius_values.append(float(np.linalg.norm(arr, axis=1).max()))

    return {
        "checked_files": len(ok_rows),
        "shape_mismatch_count": len(shape_issues),
        "non_finite_count": len(nan_issues),
        "radius_min": min(radius_values) if radius_values else None,
        "radius_max": max(radius_values) if radius_values else None,
        "radius_mean": float(np.mean(radius_values)) if radius_values else None,
        "shape_issue_examples": shape_issues[:5],
        "non_finite_examples": nan_issues[:5],
    }


def write_report(
    report_path: Path,
    raw_root: Path,
    output_root: Path,
    class_to_idx: dict[str, int],
    ok_rows: list[dict],
    failed_rows: list[dict],
    audit_summary: dict,
    elapsed_sec: float,
    num_workers: int,
    seed: int,
) -> None:
    per_split_class = defaultdict(lambda: Counter())
    per_split = Counter()
    for row in ok_rows:
        per_split[row["split"]] += 1
        per_split_class[row["split"]][row["class_name"]] += 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    lines = [
        "# Step 2 — ModelNet40 Prepare Report",
        "",
        f"- Generated at: {now}",
        f"- Raw ModelNet40 path: `{raw_root}`",
        f"- Output path: `{output_root}`",
        f"- Random seed: {seed}",
        f"- Sampling: {NUM_POINTS} points per shape (triangle area weighted when faces exist)",
        f"- Normalization: center + unit sphere",
        f"- Workers: {num_workers}",
        f"- Elapsed seconds: {elapsed_sec:.1f}",
        "",
        "## Summary",
        "",
        f"- Number of classes: **{len(class_to_idx)}**",
        f"- Train samples: **{per_split.get('train', 0)}**",
        f"- Test samples: **{per_split.get('test', 0)}**",
        f"- Total successful samples: **{len(ok_rows)}**",
        f"- Failed files: **{len(failed_rows)}**",
        "",
        "## Audit",
        "",
        f"- Checked `.npy` files: {audit_summary['checked_files']}",
        f"- Shape mismatches (expected `(1024, 3)`): {audit_summary['shape_mismatch_count']}",
        f"- Non-finite values: {audit_summary['non_finite_count']}",
        f"- Max radius min/mean/max: "
        f"{audit_summary['radius_min']:.6f} / {audit_summary['radius_mean']:.6f} / {audit_summary['radius_max']:.6f}",
        f"- Unit sphere check: {'PASS' if audit_summary['radius_max'] is not None and audit_summary['radius_max'] <= 1.000001 else 'FAIL'}",
        "",
        "## Per-split class counts",
        "",
    ]

    for split in SPLITS:
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| class | count |")
        lines.append("| --- | ---: |")
        for class_name in sorted(class_to_idx):
            lines.append(f"| {class_name} | {per_split_class[split].get(class_name, 0)} |")
        lines.append("")

    if failed_rows:
        lines.extend(
            [
                "## Failed files (first 20)",
                "",
                "| class | split | shape_id | error |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in failed_rows[:20]:
            err = row["error"].replace("|", "/")
            lines.append(f"| {row['class_name']} | {row['split']} | {row['shape_id']} | {err} |")
        lines.append("")

    lines.extend(
        [
            "## Next steps",
            "",
            "1. Run `scripts/check_modelnet40_original.py` to validate the processed dataset.",
            "2. Proceed to Step 3: clone and prepare PointNet++ classifier.",
            "3. Use `datasets/modelnet40_original/` as Line A Original baseline input.",
            "",
        ]
    )

    audit_json = {
        "generated_at": now,
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "num_classes": len(class_to_idx),
        "train_count": per_split.get("train", 0),
        "test_count": per_split.get("test", 0),
        "total_success": len(ok_rows),
        "failed_count": len(failed_rows),
        "per_split_class_counts": {split: dict(counter) for split, counter in per_split_class.items()},
        "audit": audit_summary,
        "seed": seed,
        "num_points": NUM_POINTS,
    }
    with open(output_root / "metadata" / "dataset_audit.json", "w", encoding="utf-8") as handle:
        json.dump(audit_json, handle, indent=2, sort_keys=True)

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ModelNet40 original dataset for PointNet++.")
    parser.add_argument("--raw-root", type=Path, default=Path(DEFAULT_RAW_ROOT))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--report-path", type=Path, default=Path(DEFAULT_REPORT_PATH))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    parser.add_argument("--max-files", type=int, default=0, help="Process only first N files (0 = all)")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess even if output exists")
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    report_path = args.report_path.resolve()

    if not raw_root.is_dir():
        print(f"ERROR: raw root not found: {raw_root}", file=sys.stderr)
        return 1

    tasks = discover_tasks(raw_root, output_root, args.seed, args.num_points)
    if args.max_files > 0:
        tasks = tasks[: args.max_files]

    if not args.overwrite:
        tasks = [t for t in tasks if not Path(t.output_path).exists()]

    if not tasks:
        print("No tasks to process (all outputs already exist). Use --overwrite to rebuild.")
        return 0

    print(f"Processing {len(tasks)} OFF files with {args.num_workers} workers...")
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
            if done % 100 == 0 or done == len(tasks):
                print(f"  progress: {done}/{len(tasks)} (ok={len(ok_rows)}, failed={len(failed_rows)})")

    elapsed = time.time() - start
    class_names = sorted({row["class_name"] for row in ok_rows})
    class_to_idx, _ = write_label_maps(output_root, class_names)

    for split in SPLITS:
        split_rows = [row for row in ok_rows if row["split"] == split]
        write_manifest(output_root, split, split_rows, class_to_idx)

    write_failed_files(output_root, failed_rows)
    audit_summary = audit_outputs(output_root, ok_rows)
    write_report(
        report_path=report_path,
        raw_root=raw_root,
        output_root=output_root,
        class_to_idx=class_to_idx,
        ok_rows=ok_rows,
        failed_rows=failed_rows,
        audit_summary=audit_summary,
        elapsed_sec=elapsed,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    print(f"Done. ok={len(ok_rows)} failed={len(failed_rows)} elapsed={elapsed:.1f}s")
    print(f"Report: {report_path}")
    return 0 if not failed_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
