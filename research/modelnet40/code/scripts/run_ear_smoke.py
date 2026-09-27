#!/usr/bin/env python3
"""Step 7a: EAR smoke upsampling on ModelNet40 smoke samples (CPU)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "step7_smoke_sample_manifest.csv"
DEFAULT_EAR_SCRIPT = Path(
    "/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/EAR/ear_upsampling.py"
)
DEFAULT_OUT_ORIGINAL = PROJECT_ROOT / "datasets" / "modelnet40_original_up" / "ear_smoke"
DEFAULT_OUT_DOWN = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear_smoke"
DEFAULT_RAW_DIR = PROJECT_ROOT / "outputs" / "ear_smoke" / "raw"
DEFAULT_AUDIT = PROJECT_ROOT / "reports" / "step7a_ear_smoke_audit.csv"
DEFAULT_LOG = PROJECT_ROOT / "logs" / "ear_smoke" / "run_ear_smoke.log"

TARGET_POINTS = 1024
GLOBAL_SEED = 42

VARIANTS = (
    {
        "source_variant": "original",
        "input_key": "smoke_original_path",
        "output_root": DEFAULT_OUT_ORIGINAL,
    },
    {
        "source_variant": "downsampled50",
        "input_key": "smoke_downsampled50_path",
        "output_root": DEFAULT_OUT_DOWN,
    },
)


def stable_seed(global_seed: int, *parts: str) -> int:
    token = ":".join([str(global_seed), *parts]).encode("utf-8")
    digest = hashlib.md5(token).hexdigest()
    return global_seed + (int(digest[:8], 16) % 1_000_000)


def resample_to_target(points: np.ndarray, target_n: int, seed: int) -> np.ndarray:
    count = points.shape[0]
    if count == target_n:
        return points.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    if count > target_n:
        indices = rng.choice(count, size=target_n, replace=False)
    else:
        indices = rng.choice(count, size=target_n, replace=True)
    return points[indices].astype(np.float32)


def load_ear_module(ear_script: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("ear_upsampling", ear_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import EAR module from {ear_script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ensure_metadata(output_root: Path, class_to_idx_src: Path) -> None:
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    dst = metadata_dir / "class_to_idx.json"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(class_to_idx_src.resolve())


def run_ear_on_xyz(
    ear_module,
    points_xyz: np.ndarray,
    target_n: int,
    seed: int,
    k_neighbors: int,
    edge_sensitivity: float,
    up_threshold: float,
    sigma_p: float,
    max_iter: int,
) -> np.ndarray:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) xyz, got {points_xyz.shape}")
    intensity = np.zeros((points_xyz.shape[0], 1), dtype=np.float32)
    points_xyzi = np.hstack([points_xyz.astype(np.float32), intensity])
    out_xyzi = ear_module.ear_upsample_kitti(
        points_xyzi,
        target_n=target_n,
        up_factor=2.0,
        k_neighbors=k_neighbors,
        edge_sensitivity=edge_sensitivity,
        up_threshold=up_threshold,
        sigma_p=sigma_p,
        max_iter=max_iter,
        seed=seed,
    )
    return out_xyzi[:, :3].astype(np.float32)


def process_sample(
    ear_module,
    row: dict,
    variant: dict,
    args,
) -> dict:
    split = row["split"]
    class_name = row["class_name"]
    shape_id = row["shape_id"]
    label = int(row["label"])
    source_variant = variant["source_variant"]
    input_path = Path(row[variant["input_key"]])
    output_root = variant["output_root"]
    final_output_path = output_root / split / class_name / f"{shape_id}.npy"
    raw_output_path = (
        args.raw_dir / source_variant / split / class_name / f"{shape_id}_ear_raw.npy"
    )

    record = {
        "split": split,
        "class_name": class_name,
        "label": label,
        "source_variant": source_variant,
        "input_path": str(input_path),
        "input_points": "",
        "ear_output_raw_path": str(raw_output_path),
        "final_output_path": str(final_output_path),
        "final_points": "",
        "ear_raw_points": "",
        "status": "failed",
        "error_message": "",
    }

    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing smoke input: {input_path}")

        points = np.load(input_path).astype(np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Invalid input shape {points.shape}, expected (N, 3)")
        if not np.isfinite(points).all():
            raise ValueError("Input contains NaN/Inf")

        record["input_points"] = points.shape[0]
        seed = stable_seed(args.seed, source_variant, split, class_name, shape_id, "ear")

        raw_xyz = run_ear_on_xyz(
            ear_module=ear_module,
            points_xyz=points,
            target_n=TARGET_POINTS,
            seed=seed,
            k_neighbors=args.k_neighbors,
            edge_sensitivity=args.edge_sensitivity,
            up_threshold=args.up_threshold,
            sigma_p=args.sigma_p,
            max_iter=args.max_iter,
        )
        record["ear_raw_points"] = raw_xyz.shape[0]

        raw_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_output_path, raw_xyz)

        norm_seed = stable_seed(args.seed, source_variant, split, class_name, shape_id, "to1024")
        final_xyz = resample_to_target(raw_xyz, TARGET_POINTS, norm_seed)
        if final_xyz.shape != (TARGET_POINTS, 3):
            raise ValueError(f"Normalization failed: {final_xyz.shape}")
        if not np.isfinite(final_xyz).all():
            raise ValueError("Final output contains NaN/Inf")

        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(final_output_path, final_xyz)

        record["final_points"] = final_xyz.shape[0]
        record["status"] = "success"
    except Exception as exc:
        record["error_message"] = f"{exc}\n{traceback.format_exc()}"

    return record


def write_manifests(output_root: Path, audit_rows: list[dict], variant_name: str) -> None:
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        manifest_path = metadata_dir / f"{split}_manifest.csv"
        split_rows = [
            r
            for r in audit_rows
            if r["source_variant"] == variant_name and r["split"] == split and r["status"] == "success"
        ]
        with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["split", "class_name", "label", "shape_id", "output_npy"],
            )
            writer.writeheader()
            for row in split_rows:
                out_path = Path(row["final_output_path"])
                writer.writerow(
                    {
                        "split": row["split"],
                        "class_name": row["class_name"],
                        "label": row["label"],
                        "shape_id": out_path.stem,
                        "output_npy": str(out_path),
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EAR smoke upsampling on ModelNet40 samples.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ear-script", type=Path, default=DEFAULT_EAR_SCRIPT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--k-neighbors", type=int, default=20)
    parser.add_argument("--edge-sensitivity", type=float, default=4.0)
    parser.add_argument("--up-threshold", type=float, default=0.93)
    parser.add_argument("--sigma-p", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Process only first N manifest rows")
    args = parser.parse_args()

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.manifest.is_file():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 1
    if not args.ear_script.is_file():
        print(f"ERROR: EAR script not found: {args.ear_script}", file=sys.stderr)
        return 1

    with open(args.manifest, newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if args.limit is not None:
        manifest_rows = manifest_rows[: args.limit]

    class_to_idx_src = PROJECT_ROOT / "datasets" / "modelnet40_original" / "metadata" / "class_to_idx.json"
    for variant in VARIANTS:
        ensure_metadata(variant["output_root"], class_to_idx_src)

    ear_module = load_ear_module(args.ear_script)
    audit_rows: list[dict] = []
    started = datetime.now(timezone.utc).isoformat()

    with open(args.log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"EAR smoke run started: {started}\n")
        log_handle.write(f"EAR script: {args.ear_script}\n")
        log_handle.write(f"Manifest rows: {len(manifest_rows)}\n\n")

        total_tasks = len(manifest_rows) * len(VARIANTS)
        task_idx = 0
        for row in manifest_rows:
            for variant in VARIANTS:
                task_idx += 1
                record = process_sample(ear_module, row, variant, args)
                audit_rows.append(record)
                status = record["status"]
                msg = (
                    f"[{task_idx}/{total_tasks}] {record['source_variant']} "
                    f"{record['split']}/{record['class_name']}/{Path(record['input_path']).name}: {status}"
                )
                print(msg, flush=True)
                log_handle.write(msg + "\n")
                if record["error_message"]:
                    log_handle.write(record["error_message"] + "\n")

    fieldnames = [
        "split",
        "class_name",
        "label",
        "source_variant",
        "input_path",
        "input_points",
        "ear_output_raw_path",
        "ear_raw_points",
        "final_output_path",
        "final_points",
        "status",
        "error_message",
    ]
    with open(args.audit_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    write_manifests(DEFAULT_OUT_ORIGINAL, audit_rows, "original")
    write_manifests(DEFAULT_OUT_DOWN, audit_rows, "downsampled50")

    success = sum(1 for r in audit_rows if r["status"] == "success")
    failed = len(audit_rows) - success
    print(f"Done. success={success} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
