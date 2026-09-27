#!/usr/bin/env python3
"""×4 smoke upsampling for PDANS / PU-Net / PU-GCN."""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from upsampling_x4_common import (
    CLASS_TO_IDX,
    DEFAULT_MANIFEST,
    GLOBAL_SEED,
    UP_FACTOR_X4,
    ensure_metadata_link,
    finalize_output,
    load_smoke_manifest,
    smoke_variant_config,
    stable_seed,
)
from upsampling_x4_factory import METHODS, load_upsampler, normalize_method

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def process_sample(upsampler, row: dict, variant: dict, args) -> dict:
    split = row["split"]
    class_name = row["class_name"]
    shape_id = row["shape_id"]
    label = int(row["label"])
    input_path = Path(row[variant["input_key"]])
    target_n = variant["target_points"]
    output_path = variant["output_root"] / split / class_name / f"{shape_id}.npy"

    record = {
        "method": variant["method"],
        "line": variant["line"],
        "split": split,
        "class_name": class_name,
        "label": label,
        "shape_id": shape_id,
        "source_variant": variant["source_variant"],
        "input_path": str(input_path),
        "input_points": "",
        "target_points": target_n,
        "method_raw_points": "",
        "final_output_path": str(output_path),
        "final_points": "",
        "actual_ratio": "",
        "status": "failed",
        "error_message": "",
    }

    try:
        if output_path.is_file() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing input: {input_path}")
        points = np.load(input_path).astype(np.float32)
        if points.shape[0] != variant["expected_input"]:
            raise ValueError(f"Expected {variant['expected_input']} pts, got {points.shape[0]}")
        if not np.isfinite(points).all():
            raise ValueError("Input NaN/Inf")

        record["input_points"] = points.shape[0]
        sample_name = f"{class_name}_{shape_id}"
        raw_xyz = upsampler.upsample_xyz(points, sample_name=sample_name)
        record["method_raw_points"] = raw_xyz.shape[0]
        norm_seed = stable_seed(
            args.seed,
            variant["source_variant"],
            split,
            class_name,
            shape_id,
            f"{variant['method']}_x4",
        )
        final_xyz = finalize_output(raw_xyz, points.shape[0], target_n, norm_seed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, final_xyz)
        record["final_points"] = final_xyz.shape[0]
        record["actual_ratio"] = f"{final_xyz.shape[0] / points.shape[0]:.4f}"
        record["status"] = "success"
    except FileExistsError as exc:
        record["status"] = "skipped_existing"
        record["error_message"] = str(exc)
    except Exception as exc:
        record["error_message"] = f"{exc}\n{traceback.format_exc()}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Main-method ×4 smoke test.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--line", choices=("A", "B"), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--max-per-split", type=int, default=20)
    parser.add_argument("--smoke-tag", default="", help="Optional smoke output suffix, e.g. rerun")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    method = normalize_method(args.method)
    if method not in METHODS:
        print(f"ERROR: invalid method after normalize: {args.method!r} -> {method!r}", file=sys.stderr)
        return 1
    variant = smoke_variant_config(method, args.line, smoke_tag=args.smoke_tag)
    logs_dir = PROJECT_ROOT / "logs" / f"{method}_x4_generation"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{args.smoke_tag}" if args.smoke_tag else ""
    if args.audit is None:
        args.audit = PROJECT_ROOT / "reports" / f"{method}_x4_smoke_audit_{args.line}{tag_suffix}.csv"
    if args.log_file is None:
        args.log_file = logs_dir / f"smoke_line{args.line}{tag_suffix}_run.log"

    if not args.manifest.is_file():
        print(f"ERROR: manifest missing: {args.manifest}", file=sys.stderr)
        return 1

    ensure_metadata_link(variant["output_root"], CLASS_TO_IDX)
    manifest_rows = load_smoke_manifest(args.manifest)
    upsampler = load_upsampler(method)
    audit_rows: list[dict] = []

    with open(args.log_file, "w", encoding="utf-8") as log:
        log.write(
            f"{method.upper()} ×4 smoke line {args.line} started: "
            f"{datetime.now(timezone.utc).isoformat()}\n"
        )
        counts = {"train": 0, "test": 0}
        for row in manifest_rows:
            split = row["split"]
            if counts[split] >= args.max_per_split:
                continue
            record = process_sample(upsampler, row, variant, args)
            audit_rows.append(record)
            counts[split] += 1
            msg = (
                f"[{variant['line']}] {record['status']} "
                f"{split}/{record['class_name']}/{record['shape_id']}"
            )
            print(msg, flush=True)
            log.write(msg + "\n")

    fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with open(args.audit, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    failed = sum(
        1 for r in audit_rows if r["status"] not in ("success", "skipped_existing")
    )
    print(
        f"Smoke done ({method} line {args.line}): "
        f"{len(audit_rows)} samples, failed={failed}, up_factor={UP_FACTOR_X4}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
