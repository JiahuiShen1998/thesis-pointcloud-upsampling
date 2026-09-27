#!/usr/bin/env python3
"""Resume Line B PU-Net upsampling for missing samples only (256→1024)."""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ear_modelnet40_utils import is_valid_output_npy
from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    GLOBAL_SEED,
    PROJECT_ROOT,
    detect_original_point_count,
    lineB_paths,
    reports_dir,
    stable_seed,
)
from upsampling_x4_common import finalize_output
from upsampling_x4_factory import load_upsampler

DEFAULT_MISSING = PROJECT_ROOT / "reports" / "modelnet40_lineB_punet_missing_samples.txt"
FAILED_CSV = PROJECT_ROOT / "reports" / "modelnet40_lineB_punet_resume_failed_samples.csv"
LOG_ROOT = PROJECT_ROOT / "logs" / "lineB_upsampling_resume"


def load_missing_rows(missing_txt: Path) -> list[dict]:
    rows: list[dict] = []
    for line in missing_txt.read_text(encoding="utf-8").splitlines():
        rel = line.strip()
        if not rel:
            continue
        parts = Path(rel).parts
        split = parts[0]
        cls = parts[1]
        sid = Path(rel).stem
        rows.append({"split": split, "class_name": cls, "shape_id": sid, "relative_path": rel})
    return rows


def chunk_rows(rows: list[dict], chunk_id: int, num_chunks: int) -> list[dict]:
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError(f"chunk_id={chunk_id} out of range for num_chunks={num_chunks}")
    chunk_size = (len(rows) + num_chunks - 1) // num_chunks
    start = chunk_id * chunk_size
    end = min(start + chunk_size, len(rows))
    return rows[start:end]


def process_row(upsampler, row: dict, n: int, down_n: int, seed: int) -> dict:
    split, cls, sid = row["split"], row["class_name"], row["shape_id"]
    paths = lineB_paths("pu_net")
    inp = DOWNSAMPLED_X4_ROOT / split / cls / f"{sid}.npy"
    raw_out = paths["raw"] / split / cls / f"{sid}.npy"
    strict_out = paths["strict_N"] / split / cls / f"{sid}.npy"

    record = {
        "split": split,
        "class_name": cls,
        "shape_id": sid,
        "input_path": str(inp),
        "strict_output_path": str(strict_out),
        "status": "failed",
        "error_message": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if is_valid_output_npy(strict_out, n):
        record["status"] = "skipped_existing"
        return record

    try:
        if not inp.is_file():
            raise FileNotFoundError(f"Missing input: {inp}")
        points = np.load(inp).astype(np.float32)
        if points.shape[0] != down_n:
            raise ValueError(f"Expected {down_n} pts, got {points.shape[0]}")
        if not np.isfinite(points).all():
            raise ValueError("Input NaN/Inf")

        sample_name = f"{cls}_{sid}"
        raw = upsampler.upsample_xyz(points, sample_name=sample_name)
        norm_seed = stable_seed(seed, "lineB_pu_net_resume", split, cls, sid)
        final = finalize_output(raw, points.shape[0], n, norm_seed)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        strict_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_out, raw)
        np.save(strict_out, final)
        record["status"] = "success"
    except Exception as exc:  # noqa: BLE001
        record["error_message"] = f"{exc}\n{traceback.format_exc()}"
    return record


def append_failed(record: dict) -> None:
    if record["status"] in ("success", "skipped_existing"):
        return
    FAILED_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(record.keys())
    write_header = not FAILED_CSV.is_file() or FAILED_CSV.stat().st_size == 0
    with open(FAILED_CSV, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume PU-Net Line B for missing samples only.")
    parser.add_argument("--missing-list", type=Path, default=DEFAULT_MISSING)
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    if not args.missing_list.is_file():
        print(f"Missing list not found: {args.missing_list}", file=sys.stderr)
        return 1

    n = detect_original_point_count()
    down_n = n // 4
    all_rows = load_missing_rows(args.missing_list)
    chunk = chunk_rows(all_rows, args.chunk_id, args.num_chunks)
    upsampler = load_upsampler("punet")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    audit_path = LOG_ROOT / f"chunk_{args.chunk_id:03d}_audit.csv"
    audit_rows = []

    for row in chunk:
        rec = process_row(upsampler, row, n, down_n, args.seed)
        audit_rows.append(rec)
        append_failed(rec)

    if audit_rows:
        with open(audit_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()))
            writer.writeheader()
            writer.writerows(audit_rows)

    failed = sum(1 for r in audit_rows if r["status"] == "failed")
    success = sum(1 for r in audit_rows if r["status"] == "success")
    skipped = sum(1 for r in audit_rows if r["status"] == "skipped_existing")
    print(
        f"Resume chunk {args.chunk_id}/{args.num_chunks}: "
        f"total={len(chunk)} success={success} skipped={skipped} failed={failed} "
        f"audit={audit_path}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
