#!/usr/bin/env python3
"""Smoke test for Line B new protocol upsampling (256→1024)."""

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

SMOKE_ROOT = PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up_smoke"


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


def smoke_output_roots(method: str, smoke_tag: str) -> tuple[Path, Path]:
    suffix = f"_{smoke_tag}" if smoke_tag else ""
    raw = SMOKE_ROOT / "raw" / f"{method}{suffix}"
    strict = SMOKE_ROOT / "strict_N" / f"{method}{suffix}"
    return raw, strict


def process_ear(row: dict, n: int, down_n: int, ear_module, raw_root: Path, strict_root: Path, seed: int) -> dict:
    split, cls, sid = row["split"], row["class_name"], row["shape_id"]
    label = int(row["label"])
    inp = DOWNSAMPLED_X4_ROOT / split / cls / f"{sid}.npy"
    raw_out = raw_root / split / cls / f"{sid}.npy"
    strict_out = strict_root / split / cls / f"{sid}.npy"
    record = {
        "method": "ear",
        "split": split,
        "class_name": cls,
        "label": label,
        "shape_id": sid,
        "input_path": str(inp),
        "input_points": "",
        "target_points": n,
        "final_output_path": str(strict_out),
        "final_points": "",
        "status": "failed",
        "error_message": "",
    }
    try:
        points = np.load(inp).astype(np.float32)
        if points.shape[0] != down_n:
            raise ValueError(f"Expected {down_n}, got {points.shape[0]}")
        record["input_points"] = points.shape[0]
        s = stable_seed(seed, "lineB_ear_smoke", split, cls, sid)
        raw = run_ear_on_xyz(ear_module, points, target_n=n, seed=s, up_factor=UP_FACTOR_X4)
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_out, raw)
        norm_seed = stable_seed(seed, "lineB_ear_smoke_strict", split, cls, sid)
        norm = normalize_file(raw_out, strict_out, n, norm_seed)
        record["final_points"] = norm["final_points"]
        record["status"] = "success"
    except Exception as exc:  # noqa: BLE001
        record["error_message"] = f"{exc}\n{traceback.format_exc()}"
    return record


def process_gpu(
    upsampler,
    method_key: str,
    row: dict,
    n: int,
    down_n: int,
    raw_root: Path,
    strict_root: Path,
    seed: int,
) -> dict:
    split, cls, sid = row["split"], row["class_name"], row["shape_id"]
    label = int(row["label"])
    inp = DOWNSAMPLED_X4_ROOT / split / cls / f"{sid}.npy"
    raw_out = raw_root / split / cls / f"{sid}.npy"
    strict_out = strict_root / split / cls / f"{sid}.npy"
    record = {
        "method": method_key,
        "split": split,
        "class_name": cls,
        "label": label,
        "shape_id": sid,
        "input_path": str(inp),
        "input_points": "",
        "target_points": n,
        "final_output_path": str(strict_out),
        "final_points": "",
        "status": "failed",
        "error_message": "",
    }
    try:
        points = np.load(inp).astype(np.float32)
        if points.shape[0] != down_n:
            raise ValueError(f"Expected {down_n}, got {points.shape[0]}")
        record["input_points"] = points.shape[0]
        sample_name = f"{cls}_{sid}"
        raw = upsampler.upsample_xyz(points, sample_name=sample_name)
        norm_seed = stable_seed(seed, f"lineB_{method_key}_smoke", split, cls, sid)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=MAIN_METHODS)
    parser.add_argument("--max-per-split", type=int, default=20)
    parser.add_argument("--smoke-tag", default="")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    parser.add_argument("--audit", type=Path, default=None)
    args = parser.parse_args()

    if args.method == "pu_gcn":
        print("PU-GCN is pending.", file=sys.stderr)
        return 1

    n = detect_original_point_count()
    down_n = n // 4
    rows = load_manifest(DOWNSAMPLED_X4_ROOT)
    raw_root, strict_root = smoke_output_roots(args.method, args.smoke_tag)

    logs_dir = PROJECT_ROOT / "logs" / "lineB_downsampled_x4_up" / args.method
    logs_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{args.smoke_tag}" if args.smoke_tag else ""
    if args.audit is None:
        args.audit = PROJECT_ROOT / "reports" / f"lineB_downsampled_x4_smoke_{args.method}{tag_suffix}.csv"

    counts = {"train": 0, "test": 0}
    audit_rows: list[dict] = []

    if args.method == "ear":
        ear_module = load_ear_module(DEFAULT_EAR_SCRIPT)
        for row in rows:
            split = row["split"]
            if counts[split] >= args.max_per_split:
                continue
            audit_rows.append(process_ear(row, n, down_n, ear_module, raw_root, strict_root, args.seed))
            counts[split] += 1
    else:
        alias = METHOD_ALIASES[args.method]
        upsampler = load_upsampler(alias)
        for row in rows:
            split = row["split"]
            if counts[split] >= args.max_per_split:
                continue
            audit_rows.append(
                process_gpu(upsampler, args.method, row, n, down_n, raw_root, strict_root, args.seed)
            )
            counts[split] += 1

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with open(args.audit, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()) if audit_rows else ["status"])
        writer.writeheader()
        writer.writerows(audit_rows)

    failed = sum(1 for r in audit_rows if r["status"] != "success")
    print(
        f"Smoke done method={args.method}: {len(audit_rows)} samples, failed={failed}, "
        f"input={down_n} target={n}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
