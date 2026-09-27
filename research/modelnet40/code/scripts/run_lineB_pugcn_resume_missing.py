#!/usr/bin/env python3
"""Resume Line B PU-GCN upsampling for missing samples only (256→1024)."""

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
    PROJECT_ROOT,
    lineB_paths,
    stable_seed,
)
from pugcn_modelnet40_utils import DEFAULT_RESTORE, load_pugcn_upsampler
from upsampling_x4_common import finalize_output

DEFAULT_MISSING = PROJECT_ROOT / "reports" / "modelnet40_lineB_pugcn_missing_samples.txt"
DEFAULT_EXPECTED_INPUT_POINTS = 256
DEFAULT_EXPECTED_OUTPUT_POINTS = 1024


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


def process_row(
    upsampler,
    row: dict,
    paths: dict,
    seed: int,
    resume: bool,
    expected_input_points: int,
    expected_output_points: int,
) -> dict:
    split, cls, sid = row["split"], row["class_name"], row["shape_id"]
    inp = DOWNSAMPLED_X4_ROOT / split / cls / f"{sid}.npy"
    raw_out = paths["raw"] / split / cls / f"{sid}.npy"
    strict_out = paths["strict_N"] / split / cls / f"{sid}.npy"

    record = {
        "line": "lineB",
        "split": split,
        "class_name": cls,
        "shape_id": sid,
        "relative_path": row["relative_path"],
        "input_path": str(inp),
        "raw_output_path": str(raw_out),
        "strict_output_path": str(strict_out),
        "input_shape": "",
        "raw_output_shape": "",
        "strict_output_shape": "",
        "runtime_sec": "",
        "status": "failed",
        "error_message": "",
    }

    if resume and is_valid_output_npy(strict_out, expected_output_points):
        record["input_shape"] = f"({expected_input_points}, 3)"
        record["strict_output_shape"] = f"({expected_output_points}, 3)"
        record["status"] = "skipped_existing"
        return record

    t0 = time.perf_counter()
    try:
        if not inp.is_file():
            raise FileNotFoundError(f"Missing input: {inp}")
        points = np.load(inp).astype(np.float32)
        record["input_shape"] = str(tuple(points.shape))
        if points.shape != (expected_input_points, 3):
            raise ValueError(
                f"Expected input shape ({expected_input_points}, 3), got {points.shape}"
            )
        if not np.isfinite(points).all():
            raise ValueError("Input NaN/Inf")

        sample_name = f"{cls}_{sid}"
        raw = upsampler.upsample_xyz(points, sample_name=sample_name)
        record["raw_output_shape"] = str(tuple(raw.shape))

        norm_seed = stable_seed(seed, "lineB", "pu_gcn", split, cls, sid)
        strict = finalize_output(raw, expected_input_points, expected_output_points, norm_seed)
        record["strict_output_shape"] = str(tuple(strict.shape))

        raw_out.parent.mkdir(parents=True, exist_ok=True)
        strict_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_out, raw.astype(np.float32))
        np.save(strict_out, strict)
        record["status"] = "success"
    except Exception as exc:  # noqa: BLE001
        record["error_message"] = f"{exc}\n{traceback.format_exc()}"
    finally:
        record["runtime_sec"] = f"{time.perf_counter() - t0:.3f}"

    return record


def append_failed(failed_csv: Path, record: dict) -> None:
    if record["status"] in ("success", "skipped_existing"):
        return
    failed_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "line", "split", "class_name", "shape_id", "relative_path",
        "input_path", "raw_output_path", "strict_output_path",
        "input_shape", "raw_output_shape", "strict_output_shape",
        "runtime_sec", "status", "error_message",
    ]
    write_header = not failed_csv.is_file() or failed_csv.stat().st_size == 0
    with open(failed_csv, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume PU-GCN Line B for missing samples only.")
    parser.add_argument("--missing-list", type=Path, default=DEFAULT_MISSING)
    parser.add_argument("--input-root", type=Path, default=DOWNSAMPLED_X4_ROOT)
    parser.add_argument("--raw-output-root", type=Path, default=None)
    parser.add_argument("--strict-output-root", type=Path, default=None)
    parser.add_argument("--expected-input-points", type=int, default=DEFAULT_EXPECTED_INPUT_POINTS)
    parser.add_argument("--expected-output-points", type=int, default=DEFAULT_EXPECTED_OUTPUT_POINTS)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples in chunk")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_RESTORE)
    parser.add_argument("--array-id", type=int, default=None, help="Alias for --chunk-id")
    parser.add_argument("--array-count", type=int, default=None, help="Alias for --num-chunks")
    parser.add_argument("--chunk-id", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--failed-csv", type=Path, default=None)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    exp_in = args.expected_input_points
    exp_out = args.expected_output_points

    chunk_id = args.array_id if args.array_id is not None else args.chunk_id
    num_chunks = args.array_count if args.array_count is not None else args.num_chunks

    if not args.missing_list.is_file():
        print(f"Missing list not found: {args.missing_list}", file=sys.stderr)
        return 1

    paths = lineB_paths("pu_gcn")
    if args.raw_output_root:
        paths = dict(paths)
        paths["raw"] = args.raw_output_root
    if args.strict_output_root:
        paths = dict(paths)
        paths["strict_N"] = args.strict_output_root

    if args.failed_csv is None:
        args.failed_csv = (
            PROJECT_ROOT / "reports" / f"modelnet40_lineB_pugcn_resume_failed_samples_{chunk_id}.csv"
        )

    all_rows = load_missing_rows(args.missing_list)
    chunk = chunk_rows(all_rows, chunk_id, num_chunks)
    if args.max_samples > 0:
        chunk = chunk[: args.max_samples]
    upsampler = load_pugcn_upsampler(restore_dir=args.checkpoint, gpu=args.gpu)

    success = skipped = failed = 0
    t_start = time.perf_counter()

    for i, row in enumerate(chunk, 1):
        record = process_row(
            upsampler, row, paths, args.seed, args.resume, exp_in, exp_out
        )
        append_failed(args.failed_csv, record)
        if record["status"] == "success":
            success += 1
        elif record["status"] == "skipped_existing":
            skipped += 1
        else:
            failed += 1

        if i % 10 == 0 or i == len(chunk):
            elapsed = time.perf_counter() - t_start
            print(
                f"[lineB_pugcn_resume] chunk={chunk_id} {i}/{len(chunk)} "
                f"success={success} skipped={skipped} failed={failed} elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(
        f"Done lineB_pugcn_resume chunk={chunk_id}: total={len(chunk)} "
        f"success={success} skipped={skipped} failed={failed} failed_csv={args.failed_csv}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
