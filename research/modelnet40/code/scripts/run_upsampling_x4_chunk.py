#!/usr/bin/env python3
"""×4 full upsampling chunk for PDANS / PU-Net / PU-GCN."""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np

from upsampling_x4_common import (
    CLASS_TO_IDX,
    GLOBAL_SEED,
    chunk_rows,
    ensure_metadata_link,
    finalize_output,
    full_line_config,
    is_valid_output_npy,
    load_full_manifest,
    stable_seed,
)
from upsampling_x4_factory import METHODS, load_upsampler


def process_row(upsampler, row: dict, cfg: dict, args) -> dict:
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
        "method": cfg["method"],
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
        "method_raw_points": "",
        "status": "failed",
        "error_message": "",
    }

    if args.skip_existing and is_valid_output_npy(final_output_path, target_n):
        record["input_points"] = int(np.load(input_path).shape[0]) if input_path.is_file() else ""
        record["final_points"] = target_n
        record["method_raw_points"] = target_n
        record["status"] = "skipped_existing"
        return record

    try:
        if final_output_path.is_file() and not args.skip_existing:
            raise FileExistsError(f"Refusing to overwrite: {final_output_path}")
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing input: {input_path}")
        points = np.load(input_path).astype(np.float32)
        if points.shape[0] != cfg["expected_input"]:
            raise ValueError(f"Expected {cfg['expected_input']} pts, got {points.shape[0]}")
        if not np.isfinite(points).all():
            raise ValueError("Input NaN/Inf")

        record["input_points"] = points.shape[0]
        sample_name = f"{class_name}_{shape_id}"
        raw_xyz = upsampler.upsample_xyz(points, sample_name=sample_name)
        record["method_raw_points"] = raw_xyz.shape[0]
        norm_seed = stable_seed(args.seed, variant, split, class_name, shape_id, f"{cfg['method']}_x4")
        final_xyz = finalize_output(raw_xyz, points.shape[0], target_n, norm_seed)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(final_output_path, final_xyz)
        record["final_points"] = final_xyz.shape[0]
        record["status"] = "success"
    except Exception as exc:
        record["error_message"] = f"{exc}\n{traceback.format_exc()}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Main-method ×4 full upsampling chunk.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--line", choices=("A", "B"), required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    cfg = full_line_config(args.method, args.line)
    ensure_metadata_link(cfg["output_root"], CLASS_TO_IDX)
    all_rows = load_full_manifest(cfg["input_root"])
    chunk = chunk_rows(all_rows, args.chunk_id, args.num_chunks)
    upsampler = load_upsampler(args.method)

    audit_rows = [process_row(upsampler, row, cfg, args) for row in chunk]
    cfg["chunk_audit_dir"].mkdir(parents=True, exist_ok=True)
    audit_path = cfg["chunk_audit_dir"] / f"chunk_{args.chunk_id:03d}.csv"
    fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    with open(audit_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    failed = sum(1 for r in audit_rows if r["status"] == "failed")
    print(
        f"{args.method.upper()} line {args.line} chunk {args.chunk_id}: "
        f"{len(audit_rows)} rows, failed={failed}, audit={audit_path}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
