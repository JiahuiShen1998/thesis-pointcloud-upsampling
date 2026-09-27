#!/usr/bin/env python3
"""EAR ×4 smoke upsampling — Line A (1024→4096) and Line B (512→2048)."""

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
    load_ear_module,
    resample_to_target,
    run_ear_on_xyz,
    stable_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "step7_smoke_sample_manifest.csv"
DEFAULT_AUDIT = PROJECT_ROOT / "reports" / "ear_x4_smoke_audit.csv"
DEFAULT_LOG = PROJECT_ROOT / "logs" / "ear_x4_generation" / "smoke_run.log"

VARIANTS = (
    {
        "line": "A",
        "source_variant": "original",
        "input_key": "smoke_original_path",
        "expected_input": INPUT_POINTS_ORIGINAL,
        "target_points": TARGET_POINTS_X4_ORIGINAL,
        "output_root": PROJECT_ROOT / "datasets" / "modelnet40_original_up" / "ear_x4_smoke",
    },
    {
        "line": "B",
        "source_variant": "downsampled50",
        "input_key": "smoke_downsampled50_path",
        "expected_input": INPUT_POINTS_DOWN,
        "target_points": TARGET_POINTS_X4_DOWN,
        "output_root": PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear_x4_smoke",
    },
)


def process_sample(ear_module, row: dict, variant: dict, args) -> dict:
    split = row["split"]
    class_name = row["class_name"]
    shape_id = row["shape_id"]
    label = int(row["label"])
    input_path = Path(row[variant["input_key"]])
    target_n = variant["target_points"]
    output_path = variant["output_root"] / split / class_name / f"{shape_id}.npy"

    record = {
        "line": variant["line"],
        "split": split,
        "class_name": class_name,
        "label": label,
        "shape_id": shape_id,
        "source_variant": variant["source_variant"],
        "input_path": str(input_path),
        "input_points": "",
        "target_points": target_n,
        "ear_raw_points": "",
        "final_output_path": str(output_path),
        "final_points": "",
        "actual_ratio": "",
        "status": "failed",
        "error_message": "",
    }

    try:
        if not input_path.is_file():
            raise FileNotFoundError(f"Missing input: {input_path}")
        points = np.load(input_path).astype(np.float32)
        if points.shape[0] != variant["expected_input"]:
            raise ValueError(f"Expected {variant['expected_input']} pts, got {points.shape[0]}")
        if not np.isfinite(points).all():
            raise ValueError("Input NaN/Inf")

        record["input_points"] = points.shape[0]
        seed = stable_seed(args.seed, variant["source_variant"], split, class_name, shape_id, "ear_x4")
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
        norm_seed = stable_seed(args.seed, variant["source_variant"], split, class_name, shape_id, "to_x4")
        final_xyz = resample_to_target(raw_xyz, target_n, norm_seed)
        if final_xyz.shape != (target_n, 3):
            raise ValueError(f"Bad shape {final_xyz.shape}, expected ({target_n}, 3)")
        if not np.isfinite(final_xyz).all():
            raise ValueError("Output NaN/Inf")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, final_xyz)
        record["final_points"] = final_xyz.shape[0]
        record["actual_ratio"] = f"{final_xyz.shape[0] / points.shape[0]:.4f}"
        record["status"] = "success"
    except Exception as exc:
        record["error_message"] = f"{exc}\n{traceback.format_exc()}"
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="EAR ×4 smoke test.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--ear-script", type=Path, default=DEFAULT_EAR_SCRIPT)
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--max-per-split", type=int, default=20, help="Max samples per split per line")
    parser.add_argument("--k-neighbors", type=int, default=20)
    parser.add_argument("--edge-sensitivity", type=float, default=4.0)
    parser.add_argument("--up-threshold", type=float, default=0.93)
    parser.add_argument("--sigma-p", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=5)
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"ERROR: manifest missing: {args.manifest}", file=sys.stderr)
        return 1
    if not args.ear_script.is_file():
        print(f"ERROR: EAR script missing: {args.ear_script}", file=sys.stderr)
        return 1

    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    class_to_idx_src = PROJECT_ROOT / "datasets" / "modelnet40_original" / "metadata" / "class_to_idx.json"

    with open(args.manifest, newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))

    ear_module = load_ear_module(args.ear_script)
    audit_rows: list[dict] = []

    with open(args.log_file, "w", encoding="utf-8") as log:
        log.write(f"EAR ×4 smoke started: {datetime.now(timezone.utc).isoformat()}\n")
        for variant in VARIANTS:
            ensure_metadata_link(variant["output_root"], class_to_idx_src)
            counts = {"train": 0, "test": 0}
            for row in manifest_rows:
                split = row["split"]
                if counts[split] >= args.max_per_split:
                    continue
                record = process_sample(ear_module, row, variant, args)
                audit_rows.append(record)
                counts[split] += 1
                msg = f"[{variant['line']}] {record['status']} {split}/{record['class_name']}/{record['shape_id']}"
                print(msg, flush=True)
                log.write(msg + "\n")

    fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    with open(args.audit, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    failed = sum(1 for r in audit_rows if r["status"] != "success")
    print(f"Smoke done: {len(audit_rows)} samples, failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
