#!/usr/bin/env python3
"""EAR ×4 full upsampling chunk — Line A (1024→4096) or Line B (512→2048)."""

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
    INPUT_POINTS_DOWN,
    INPUT_POINTS_ORIGINAL,
    TARGET_POINTS_X4_DOWN,
    TARGET_POINTS_X4_ORIGINAL,
    UP_FACTOR_X4,
    ensure_metadata_link,
    is_valid_output_npy,
    load_ear_module,
    resample_to_target,
    run_ear_on_xyz,
    stable_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_TO_IDX = PROJECT_ROOT / "datasets" / "modelnet40_original" / "metadata" / "class_to_idx.json"

LINE_CONFIG = {
    "A": {
        "input_root": PROJECT_ROOT / "datasets" / "modelnet40_original",
        "output_root": PROJECT_ROOT / "datasets" / "modelnet40_original_up" / "ear_x4",
        "source_variant": "original",
        "expected_input": INPUT_POINTS_ORIGINAL,
        "target_points": TARGET_POINTS_X4_ORIGINAL,
        "chunk_audit_dir": PROJECT_ROOT / "logs" / "ear_x4_generation" / "lineA_chunk_audits",
    },
    "B": {
        "input_root": PROJECT_ROOT / "datasets" / "modelnet40_downsampled50",
        "output_root": PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear_x4",
        "source_variant": "downsampled50",
        "expected_input": INPUT_POINTS_DOWN,
        "target_points": TARGET_POINTS_X4_DOWN,
        "chunk_audit_dir": PROJECT_ROOT / "logs" / "ear_x4_generation" / "lineB_chunk_audits",
    },
}


def load_full_manifest(input_root: Path) -> list[dict]:
    rows: list[dict] = []
    metadata_dir = input_root / "metadata"
    for split in ("train", "test"):
        manifest_path = metadata_dir / f"{split}_manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing manifest: {manifest_path}")
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
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


def process_row(ear_module, row: dict, cfg: dict, args) -> dict:
    split = row["split"]
    class_name = row["class_name"]
    shape_id = row["shape_id"]
    label = int(row["label"])
    target_n = cfg["target_points"]
    variant = cfg["source_variant"]
    input_root = cfg["input_root"]
    output_root = cfg["output_root"]

    input_path = Path(row.get("output_path") or "")
    if not input_path.is_file():
        input_path = input_root / split / class_name / f"{shape_id}.npy"
    final_output_path = output_root / split / class_name / f"{shape_id}.npy"

    record = {
        "line": args.line,
        "chunk_id": args.chunk_id,
        "split": split,
        "class_name": class_name,
        "label": label,
        "shape_id": shape_id,
        "source_variant": variant,
        "input_path": str(input_path),
        "input_points": "",
        "target_points": target_n,
        "final_output_path": str(final_output_path),
        "final_points": "",
        "ear_raw_points": "",
        "status": "failed",
        "error_message": "",
    }

    if args.skip_existing and is_valid_output_npy(final_output_path, target_n):
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
        if points.shape[0] != cfg["expected_input"]:
            raise ValueError(f"Expected {cfg['expected_input']} pts, got {points.shape[0]}")
        if not np.isfinite(points).all():
            raise ValueError("Input NaN/Inf")

        record["input_points"] = points.shape[0]
        seed = stable_seed(args.seed, variant, split, class_name, shape_id, "ear_x4")
        raw_xyz = run_ear_on_xyz(
            ear_module,
            points,
            target_n=target_n,
            seed=seed,
            up_factor=UP_FACTOR_X4,
            k_neighbors=args.k_neighbors,
            edge_sensitivity=args.edge_sensitivity,
            up_threshold=args.up_threshold,
            sigma_p=args.sigma_p,
            max_iter=args.max_iter,
        )
        record["ear_raw_points"] = raw_xyz.shape[0]
        norm_seed = stable_seed(args.seed, variant, split, class_name, shape_id, "to_x4")
        final_xyz = resample_to_target(raw_xyz, target_n, norm_seed)
        if final_xyz.shape != (target_n, 3):
            raise ValueError(f"Bad shape {final_xyz.shape}")
        if not np.isfinite(final_xyz).all():
            raise ValueError("Output NaN/Inf")

        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(final_output_path, final_xyz)
        record["final_points"] = final_xyz.shape[0]
        record["status"] = "success"
    except Exception as exc:
        record["error_message"] = f"{exc}\n{traceback.format_exc()}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="EAR ×4 full upsampling chunk.")
    parser.add_argument("--line", choices=("A", "B"), required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--ear-script", type=Path, default=DEFAULT_EAR_SCRIPT)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--k-neighbors", type=int, default=20)
    parser.add_argument("--edge-sensitivity", type=float, default=4.0)
    parser.add_argument("--up-threshold", type=float, default=0.93)
    parser.add_argument("--sigma-p", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = LINE_CONFIG[args.line]
    if not args.ear_script.is_file():
        print(f"ERROR: EAR script missing: {args.ear_script}", file=sys.stderr)
        return 1

    ensure_metadata_link(cfg["output_root"], CLASS_TO_IDX)
    all_rows = load_full_manifest(cfg["input_root"])
    chunk = chunk_rows(all_rows, args.chunk_id, args.num_chunks)
    ear_module = load_ear_module(args.ear_script)

    audit_rows = [process_row(ear_module, row, cfg, args) for row in chunk]
    cfg["chunk_audit_dir"].mkdir(parents=True, exist_ok=True)
    audit_path = cfg["chunk_audit_dir"] / f"chunk_{args.chunk_id:03d}.csv"
    fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    with open(audit_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    failed = sum(1 for r in audit_rows if r["status"] == "failed")
    print(
        f"Line {args.line} chunk {args.chunk_id}: "
        f"{len(audit_rows)} rows, failed={failed}, audit={audit_path}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
