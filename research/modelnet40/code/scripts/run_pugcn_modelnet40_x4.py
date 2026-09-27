#!/usr/bin/env python3
"""PU-GCN ModelNet40 ×4 two-line protocol runner (Line A / Line B)."""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from ear_modelnet40_utils import is_valid_output_npy
from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    GLOBAL_SEED,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    lineA_paths,
    lineB_paths,
    stable_seed,
)
from pugcn_modelnet40_utils import DEFAULT_RESTORE, load_pugcn_upsampler
from strict_normalize_point_count import normalize_file
from upsampling_x4_common import finalize_output, load_full_manifest

LINE_CONFIG = {
    "lineA": {
        "input_root": ORIGINAL_ROOT,
        "raw_output_root": lineA_paths("pu_gcn")["raw"],
        "strict_output_root": lineA_paths("pu_gcn")["strict_4N"],
        "expected_input_points": None,  # filled at runtime
        "expected_output_points": None,
    },
    "lineB": {
        "input_root": DOWNSAMPLED_X4_ROOT,
        "raw_output_root": lineB_paths("pu_gcn")["raw"],
        "strict_output_root": lineB_paths("pu_gcn")["strict_N"],
        "expected_input_points": None,
        "expected_output_points": None,
    },
}


def resolve_config(line: str, n: int) -> dict:
    cfg = dict(LINE_CONFIG[line])
    if line == "lineA":
        cfg["expected_input_points"] = n
        cfg["expected_output_points"] = n * 4
    else:
        cfg["expected_input_points"] = n // 4
        cfg["expected_output_points"] = n
    return cfg


def load_manifest(input_root: Path) -> list[dict]:
    return load_full_manifest(input_root)


def chunk_rows(rows: list[dict], chunk_id: int | None, num_chunks: int | None) -> list[dict]:
    if chunk_id is None or num_chunks is None:
        return rows
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError(f"chunk_id={chunk_id} out of range for num_chunks={num_chunks}")
    chunk_size = (len(rows) + num_chunks - 1) // num_chunks
    start = chunk_id * chunk_size
    end = min(start + chunk_size, len(rows))
    return rows[start:end]


def process_sample(
    upsampler,
    row: dict,
    cfg: dict,
    args,
) -> dict:
    split = row["split"]
    cls = row["class_name"]
    sid = row["shape_id"]
    label = int(row["label"])
    inp = cfg["input_root"] / split / cls / f"{sid}.npy"
    raw_out = cfg["raw_output_root"] / split / cls / f"{sid}.npy"
    strict_out = cfg["strict_output_root"] / split / cls / f"{sid}.npy"
    exp_in = cfg["expected_input_points"]
    exp_out = cfg["expected_output_points"]

    record = {
        "line": args.line,
        "split": split,
        "class_name": cls,
        "label": label,
        "shape_id": sid,
        "input_path": str(inp),
        "input_points": "",
        "raw_output_path": str(raw_out),
        "raw_output_shape": "",
        "strict_output_path": str(strict_out),
        "strict_output_shape": "",
        "target_points": exp_out,
        "status": "failed",
        "error_message": "",
        "elapsed_sec": "",
    }

    if args.resume and is_valid_output_npy(strict_out, exp_out):
        record["input_points"] = exp_in
        record["strict_output_shape"] = f"({exp_out}, 3)"
        record["status"] = "skipped_existing"
        return record

    t0 = time.perf_counter()
    try:
        if not inp.is_file():
            raise FileNotFoundError(f"Missing input: {inp}")
        points = np.load(inp).astype(np.float32)
        if points.shape != (exp_in, 3):
            raise ValueError(f"Expected input shape ({exp_in}, 3), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("Input NaN/Inf")

        record["input_points"] = exp_in
        sample_name = f"{cls}_{sid}"
        raw = upsampler.upsample_xyz(points, sample_name=sample_name)
        record["raw_output_shape"] = str(tuple(raw.shape))

        norm_seed = stable_seed(args.seed, args.line, "pu_gcn", split, cls, sid)
        strict = finalize_output(raw, exp_in, exp_out, norm_seed)

        raw_out.parent.mkdir(parents=True, exist_ok=True)
        strict_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_out, raw.astype(np.float32))
        np.save(strict_out, strict)

        record["strict_output_shape"] = str(tuple(strict.shape))
        record["status"] = "success"
    except Exception as exc:  # noqa: BLE001
        record["error_message"] = f"{exc}\n{traceback.format_exc()}"
    finally:
        record["elapsed_sec"] = f"{time.perf_counter() - t0:.3f}"

    return record


def append_failed(failed_csv: Path | None, record: dict) -> None:
    if failed_csv is None or record["status"] in ("success", "skipped_existing"):
        return
    failed_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not failed_csv.is_file()
    fields = [
        "line", "split", "class_name", "shape_id", "input_path",
        "status", "error_message",
    ]
    with open(failed_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="PU-GCN ModelNet40 ×4 upsampling (two-line protocol).")
    parser.add_argument("--line", required=True, choices=("lineA", "lineB"))
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--raw-output-root", type=Path, default=None)
    parser.add_argument("--strict-output-root", type=Path, default=None)
    parser.add_argument("--expected-input-points", type=int, default=None)
    parser.add_argument("--expected-output-points", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RESTORE)
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved; PU-GCN runs one sample per call")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--max-per-split", type=int, default=0, help="If >0, take this many per train/test split")
    parser.add_argument("--chunk-id", type=int, default=None)
    parser.add_argument("--num-chunks", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--failed-csv", type=Path, default=None)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    n = detect_original_point_count()
    cfg = resolve_config(args.line, n)
    if args.input_root:
        cfg["input_root"] = args.input_root
    if args.raw_output_root:
        cfg["raw_output_root"] = args.raw_output_root
    if args.strict_output_root:
        cfg["strict_output_root"] = args.strict_output_root
    if args.expected_input_points is not None:
        cfg["expected_input_points"] = args.expected_input_points
    if args.expected_output_points is not None:
        cfg["expected_output_points"] = args.expected_output_points

    if args.failed_csv is None:
        log_dir = PROJECT_ROOT / "logs" / "pugcn_x4" / args.line
        log_dir.mkdir(parents=True, exist_ok=True)
        args.failed_csv = log_dir / "failed_samples.csv"

    rows = load_manifest(cfg["input_root"])
    rows = chunk_rows(rows, args.chunk_id, args.num_chunks)
    if args.max_per_split > 0:
        counts = {"train": 0, "test": 0}
        filtered: list[dict] = []
        for row in rows:
            split = row["split"]
            if counts[split] >= args.max_per_split:
                continue
            filtered.append(row)
            counts[split] += 1
        rows = filtered
    elif args.max_samples > 0:
        rows = rows[: args.max_samples]

    if args.failed_csv and args.failed_csv.is_file() and (args.max_samples > 0 or args.max_per_split > 0):
        args.failed_csv.unlink()

    cfg["raw_output_root"].mkdir(parents=True, exist_ok=True)
    cfg["strict_output_root"].mkdir(parents=True, exist_ok=True)

    upsampler = load_pugcn_upsampler(restore_dir=args.checkpoint, gpu=args.gpu)

    total = len(rows)
    success = skipped = failed = 0
    t_start = time.perf_counter()

    for i, row in enumerate(rows, 1):
        record = process_sample(upsampler, row, cfg, args)
        append_failed(args.failed_csv, record)
        if record["status"] == "success":
            success += 1
        elif record["status"] == "skipped_existing":
            skipped += 1
        else:
            failed += 1

        if i % 10 == 0 or i == total:
            elapsed = time.perf_counter() - t_start
            print(
                f"[{args.line}] {i}/{total} success={success} skipped={skipped} "
                f"failed={failed} elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(
        f"Done {args.line}: total={total} success={success} skipped={skipped} "
        f"failed={failed} failed_csv={args.failed_csv}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
