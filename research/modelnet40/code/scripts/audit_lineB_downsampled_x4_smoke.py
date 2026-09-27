#!/usr/bin/env python3
"""Audit Line B new protocol smoke outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    MAIN_METHODS,
    PROJECT_ROOT,
    detect_original_point_count,
    reports_dir,
)

SMOKE_ROOT = PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up_smoke"


def audit_method(method: str, smoke_tag: str, expected_train: int, expected_test: int) -> dict:
    n = detect_original_point_count()
    suffix = f"_{smoke_tag}" if smoke_tag else ""
    strict_root = SMOKE_ROOT / "strict_N" / f"{method}{suffix}"
    counts = {"train": 0, "test": 0}
    bad_shapes: list[str] = []
    nan_inf = False

    for split in ("train", "test"):
        split_dir = strict_root / split
        if not split_dir.is_dir():
            continue
        for fpath in sorted(split_dir.rglob("*.npy")):
            counts[split] += 1
            arr = np.load(fpath)
            if arr.shape != (n, 3):
                bad_shapes.append(f"{fpath.name}:{arr.shape}")
            if not np.isfinite(arr).all():
                nan_inf = True

    ok = (
        counts["train"] == expected_train
        and counts["test"] == expected_test
        and not bad_shapes
        and not nan_inf
    )
    return {
        "method": method,
        "smoke_tag": smoke_tag,
        "strict_root": str(strict_root),
        "expected_shape": f"({n}, 3)",
        "train_count": counts["train"],
        "test_count": counts["test"],
        "expected_train": expected_train,
        "expected_test": expected_test,
        "bad_shapes": ";".join(bad_shapes[:5]),
        "nan_inf": nan_inf,
        "status": "PASS" if ok else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=MAIN_METHODS)
    parser.add_argument("--smoke-tag", default="")
    parser.add_argument("--expected-train", type=int, default=20)
    parser.add_argument("--expected-test", type=int, default=20)
    args = parser.parse_args()

    if args.method == "pu_gcn":
        print("PU-GCN is pending.", file=sys.stderr)
        return 1

    row = audit_method(args.method, args.smoke_tag, args.expected_train, args.expected_test)
    rep = reports_dir()
    rep.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.smoke_tag}" if args.smoke_tag else ""
    csv_path = rep / f"lineB_downsampled_x4_smoke_audit_{args.method}{suffix}.csv"
    md_path = rep / f"lineB_downsampled_x4_smoke_audit_{args.method}{suffix}.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    md_path.write_text(
        "\n".join(
            [
                f"# Line B Smoke Audit — {args.method}",
                "",
                f"- Generated at: {ts}",
                f"- Status: **{row['status']}**",
                f"- strict root: `{row['strict_root']}`",
                f"- train/test: {row['train_count']}/{row['test_count']} "
                f"(expected {row['expected_train']}/{row['expected_test']})",
                f"- expected shape: {row['expected_shape']}",
                f"- bad shapes: {row['bad_shapes'] or 'none'}",
                f"- nan/inf: {row['nan_inf']}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Audit {row['status']}: {csv_path}")
    return 0 if row["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
