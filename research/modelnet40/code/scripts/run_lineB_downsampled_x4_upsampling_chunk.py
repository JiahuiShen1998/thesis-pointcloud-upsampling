#!/usr/bin/env python3
"""Line B upsampling chunk for new ×4 protocol (256→1024)."""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import numpy as np

from ear_modelnet40_utils import (
    DEFAULT_EAR_SCRIPT,
    UP_FACTOR_X4,
    is_valid_output_npy,
    load_ear_module,
    run_ear_on_xyz,
)
from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    GLOBAL_SEED,
    MAIN_METHODS,
    METHOD_ALIASES,
    PROJECT_ROOT,
    detect_original_point_count,
    lineB_paths,
    stable_seed,
)
from strict_normalize_point_count import normalize_file
from upsampling_x4_common import finalize_output
from upsampling_x4_factory import load_upsampler

LOG_ROOT = PROJECT_ROOT / "logs" / "lineB_downsampled_x4_up"


def load_manifest(input_root: Path) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "test"):
        mp = input_root / "metadata" / f"{split}_manifest.csv"
        with open(mp, newline="", encoding="utf-8") as handle:
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


def process_ear_row(
    ear_module,
    row: dict,
    n: int,
    down_n: int,
    args,
) -> dict:
    method_key = "ear"
    paths = lineB_paths(method_key)
    raw_root = paths["raw"]
    strict_root = paths["strict_N"]
    split, cls, sid = row["split"], row["class_name"], row["shape_id"]
    label = int(row["label"])
    inp = DOWNSAMPLED_X4_ROOT / split / cls / f"{sid}.npy"
    raw_out = raw_root / split / cls / f"{sid}.npy"
    strict_out = strict_root / split / cls / f"{sid}.npy"

    record = {
        "method": method_key,
        "chunk_id": args.chunk_id,
        "split": split,
        "class_name": cls,
        "label": label,
        "shape_id": sid,
        "input_path": str(inp),
        "input_points": "",
        "target_points": n,
        "raw_output_path": str(raw_out),
        "strict_output_path": str(strict_out),
        "method_raw_points": "",
        "final_points": "",
        "status": "failed",
        "error_message": "",
    }

    if args.skip_existing and is_valid_output_npy(strict_out, n):
        record["input_points"] = down_n
        record["method_raw_points"] = n
        record["final_points"] = n
        record["status"] = "skipped_existing"
        return record

    try:
        if not inp.is_file():
            raise FileNotFoundError(f"Missing input: {inp}")
        points = np.load(inp).astype(np.float32)
        if points.shape[0] != down_n:
            raise ValueError(f"Expected {down_n}, got {points.shape[0]}")
        if not np.isfinite(points).all():
            raise ValueError("Input NaN/Inf")

        record["input_points"] = points.shape[0]
        seed = stable_seed(args.seed, "lineB_ear", split, cls, sid)
        raw = run_ear_on_xyz(
            ear_module,
            points,
            target_n=n,
            seed=seed,
            up_factor=UP_FACTOR_X4,
        )
        record["method_raw_points"] = raw.shape[0]
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_out, raw)
        norm_seed = stable_seed(args.seed, "lineB_ear_strict", split, cls, sid)
        norm = normalize_file(raw_out, strict_out, n, norm_seed)
        record["final_points"] = norm["final_points"]
        record["status"] = "success"
    except Exception as exc:  # noqa: BLE001
        record["error_message"] = f"{exc}\n{traceback.format_exc()}"
    return record


def process_gpu_row(
    upsampler,
    row: dict,
    method_key: str,
    n: int,
    down_n: int,
    args,
) -> dict:
    paths = lineB_paths(method_key)
    split, cls, sid = row["split"], row["class_name"], row["shape_id"]
    label = int(row["label"])
    inp = DOWNSAMPLED_X4_ROOT / split / cls / f"{sid}.npy"
    raw_out = paths["raw"] / split / cls / f"{sid}.npy"
    strict_out = paths["strict_N"] / split / cls / f"{sid}.npy"

    record = {
        "method": method_key,
        "chunk_id": args.chunk_id,
        "split": split,
        "class_name": cls,
        "label": label,
        "shape_id": sid,
        "input_path": str(inp),
        "input_points": "",
        "target_points": n,
        "raw_output_path": str(raw_out),
        "strict_output_path": str(strict_out),
        "method_raw_points": "",
        "final_points": "",
        "status": "failed",
        "error_message": "",
    }

    if args.skip_existing and is_valid_output_npy(strict_out, n):
        record["input_points"] = down_n
        record["method_raw_points"] = n
        record["final_points"] = n
        record["status"] = "skipped_existing"
        return record

    try:
        if not inp.is_file():
            raise FileNotFoundError(f"Missing input: {inp}")
        points = np.load(inp).astype(np.float32)
        if points.shape[0] != down_n:
            raise ValueError(f"Expected {down_n}, got {points.shape[0]}")
        if not np.isfinite(points).all():
            raise ValueError("Input NaN/Inf")

        record["input_points"] = points.shape[0]
        sample_name = f"{cls}_{sid}"
        raw = upsampler.upsample_xyz(points, sample_name=sample_name)
        record["method_raw_points"] = raw.shape[0]
        norm_seed = stable_seed(args.seed, f"lineB_{method_key}", split, cls, sid)
        final = finalize_output(raw, points.shape[0], n, norm_seed)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        strict_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_out, raw)
        np.save(strict_out, final)
        record["final_points"] = final.shape[0]
        record["status"] = "success"
    except Exception as exc:  # noqa: BLE001
        record["error_message"] = f"{exc}\n{traceback.format_exc()}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Line B new protocol upsampling chunk (256→1024).")
    parser.add_argument("--method", required=True, choices=MAIN_METHODS)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--ear-script", type=Path, default=DEFAULT_EAR_SCRIPT)
    args = parser.parse_args()

    if args.method == "pu_gcn":
        print("PU-GCN is pending for new Line B protocol.", file=sys.stderr)
        return 1

    n = detect_original_point_count()
    down_n = n // 4
    rows = load_manifest(DOWNSAMPLED_X4_ROOT)
    chunk = chunk_rows(rows, args.chunk_id, args.num_chunks)

    if args.method == "ear":
        if not args.ear_script.is_file():
            print(f"EAR script missing: {args.ear_script}", file=sys.stderr)
            return 1
        ear_module = load_ear_module(args.ear_script)
        audit_rows = [process_ear_row(ear_module, row, n, down_n, args) for row in chunk]
    else:
        alias = METHOD_ALIASES[args.method]
        upsampler = load_upsampler(alias)
        audit_rows = [process_gpu_row(upsampler, row, args.method, n, down_n, args) for row in chunk]

    audit_dir = LOG_ROOT / args.method / "chunk_audits"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_path = audit_dir / f"chunk_{args.chunk_id:03d}.csv"
    fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    with open(audit_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    failed = sum(1 for r in audit_rows if r["status"] == "failed")
    print(
        f"Line B x4 chunk {args.chunk_id}/{args.num_chunks} "
        f"method={args.method}: {len(audit_rows)} rows, failed={failed}, audit={audit_path}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
