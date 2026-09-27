#!/usr/bin/env python3
"""Step 7b: EAR full Line B upsampling for one manifest chunk (CPU)."""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ear_modelnet40_utils import (
    DEFAULT_EAR_SCRIPT,
    GLOBAL_SEED,
    TARGET_POINTS,
    is_valid_output_npy,
    load_ear_module,
    resample_to_target,
    run_ear_on_xyz,
    stable_seed,
)

PROJECT_ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear"
DEFAULT_CHUNK_AUDIT_DIR = PROJECT_ROOT / "logs" / "ear_full_lineB" / "chunk_audits"
SOURCE_VARIANT = "downsampled50"


def load_full_manifest(input_root: Path) -> list[dict]:
    rows: list[dict] = []
    metadata_dir = input_root / "metadata"
    for split in ("train", "test"):
        manifest_path = metadata_dir / f"{split}_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = dict(row)
                row["split"] = row.get("split") or split
                rows.append(row)
    rows.sort(key=lambda r: (r["split"], r["class_name"], r["shape_id"]))
    return rows


def chunk_rows(rows: list[dict], chunk_id: int, num_chunks: int) -> list[dict]:
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError(f"chunk_id={chunk_id} out of range for num_chunks={num_chunks}")
    chunk_size = (len(rows) + num_chunks - 1) // num_chunks
    start = chunk_id * chunk_size
    end = min(start + chunk_size, len(rows))
    return rows[start:end]


def ensure_metadata(output_root: Path) -> None:
    class_to_idx_src = PROJECT_ROOT / "datasets" / "modelnet40_original" / "metadata" / "class_to_idx.json"
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    dst = metadata_dir / "class_to_idx.json"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(class_to_idx_src.resolve())


def process_row(
    ear_module,
    row: dict,
    input_root: Path,
    output_root: Path,
    args,
) -> dict:
    split = row["split"]
    class_name = row["class_name"]
    shape_id = row["shape_id"]
    label = int(row["label"])

    input_path = Path(row.get("output_path") or "")
    if not input_path.is_file():
        input_path = input_root / split / class_name / f"{shape_id}.npy"

    final_output_path = output_root / split / class_name / f"{shape_id}.npy"
    record = {
        "chunk_id": args.chunk_id,
        "split": split,
        "class_name": class_name,
        "label": label,
        "shape_id": shape_id,
        "source_variant": SOURCE_VARIANT,
        "input_path": str(input_path),
        "input_points": "",
        "final_output_path": str(final_output_path),
        "final_points": "",
        "ear_raw_points": "",
        "status": "failed",
        "error_message": "",
    }

    if args.skip_existing and is_valid_output_npy(final_output_path):
        arr = np.load(final_output_path)
        record["input_points"] = int(np.load(input_path).shape[0]) if input_path.is_file() else ""
        record["final_points"] = arr.shape[0]
        record["ear_raw_points"] = arr.shape[0]
        record["status"] = "skipped_existing"
        return record

    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing input: {input_path}")

        points = np.load(input_path).astype(np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Invalid input shape {points.shape}, expected (N, 3)")
        if not np.isfinite(points).all():
            raise ValueError("Input contains NaN/Inf")

        record["input_points"] = points.shape[0]
        seed = stable_seed(args.seed, SOURCE_VARIANT, split, class_name, shape_id, "ear")

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

        norm_seed = stable_seed(args.seed, SOURCE_VARIANT, split, class_name, shape_id, "to1024")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="EAR full Line B upsampling chunk.")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--ear-script", type=Path, default=DEFAULT_EAR_SCRIPT)
    parser.add_argument("--chunk-audit-dir", type=Path, default=DEFAULT_CHUNK_AUDIT_DIR)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--k-neighbors", type=int, default=20)
    parser.add_argument("--edge-sensitivity", type=float, default=4.0)
    parser.add_argument("--up-threshold", type=float, default=0.93)
    parser.add_argument("--sigma-p", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    args = parser.parse_args()

    args.chunk_audit_dir.mkdir(parents=True, exist_ok=True)
    ensure_metadata(args.output_root)

    all_rows = load_full_manifest(args.input_root)
    chunk = chunk_rows(all_rows, args.chunk_id, args.num_chunks)
    if not chunk:
        print(f"Chunk {args.chunk_id} is empty, nothing to do.")
        return 0

    ear_module = load_ear_module(args.ear_script)
    audit_rows: list[dict] = []
    started = datetime.now(timezone.utc).isoformat()
    log_path = args.chunk_audit_dir / f"chunk_{args.chunk_id:04d}.log"
    audit_path = args.chunk_audit_dir / f"chunk_{args.chunk_id:04d}_audit.csv"

    with open(log_path, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"started: {started}\n")
        log_handle.write(f"chunk_id={args.chunk_id} num_chunks={args.num_chunks}\n")
        log_handle.write(f"rows_in_chunk={len(chunk)} total_rows={len(all_rows)}\n\n")

        for idx, row in enumerate(chunk, start=1):
            record = process_row(ear_module, row, args.input_root, args.output_root, args)
            audit_rows.append(record)
            msg = (
                f"[{idx}/{len(chunk)}] {record['split']}/{record['class_name']}/"
                f"{record['shape_id']}: {record['status']}"
            )
            print(msg, flush=True)
            log_handle.write(msg + "\n")
            if record["error_message"]:
                log_handle.write(record["error_message"] + "\n")

    fieldnames = [
        "chunk_id",
        "split",
        "class_name",
        "label",
        "shape_id",
        "source_variant",
        "input_path",
        "input_points",
        "ear_raw_points",
        "final_output_path",
        "final_points",
        "status",
        "error_message",
    ]
    with open(audit_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    success = sum(1 for r in audit_rows if r["status"] in ("success", "skipped_existing"))
    failed = sum(1 for r in audit_rows if r["status"] == "failed")
    print(f"Chunk {args.chunk_id} done. success_or_skip={success} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
