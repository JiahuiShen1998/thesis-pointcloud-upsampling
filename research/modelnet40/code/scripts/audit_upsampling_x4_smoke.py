#!/usr/bin/env python3
"""Post-smoke audit for main-method ×4 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from upsampling_x4_common import (
    LEGACY_EAR_X2,
    PROJECT_ROOT,
    load_class_to_idx,
    smoke_variant_config,
)
from upsampling_x4_factory import METHODS, normalize_method

EXPECTED_TRAIN = 20
EXPECTED_TEST = 20


def audit_variant(
    method: str,
    line: str,
    smoke_tag: str = "",
    expected_train: int = EXPECTED_TRAIN,
    expected_test: int = EXPECTED_TEST,
) -> dict:
    v = smoke_variant_config(method, line, smoke_tag=smoke_tag)
    class_to_idx = load_class_to_idx()

    issues: list[str] = []
    shape_ok = True
    nan_inf = False
    label_mismatch = 0
    counts = {"train": 0, "test": 0}
    bad_shapes: list[str] = []
    overwrite_hits = 0

    for split in ("train", "test"):
        split_dir = v["output_root"] / split
        if not split_dir.is_dir():
            issues.append(f"missing split dir: {split_dir}")
            continue
        for fpath in sorted(split_dir.rglob("*.npy")):
            counts[split] += 1
            arr = np.load(fpath)
            if arr.shape != (v["target_points"], 3):
                shape_ok = False
                bad_shapes.append(f"{fpath.name}:{arr.shape}")
            if not np.isfinite(arr).all():
                nan_inf = True
            class_name = fpath.parent.name
            if class_name not in class_to_idx:
                label_mismatch += 1

    legacy_count = sum(1 for _ in LEGACY_EAR_X2.rglob("*.npy")) if LEGACY_EAR_X2.is_dir() else 0
    legacy_intact = legacy_count > 0

    status = "PASS"
    if not shape_ok or nan_inf or label_mismatch:
        status = "FAIL"
    if counts["train"] == 0 or counts["test"] == 0:
        status = "FAIL"
        issues.append("empty train or test split")
    if counts["train"] != expected_train or counts["test"] != expected_test:
        status = "FAIL"
        issues.append(
            f"count mismatch: train={counts['train']} test={counts['test']} "
            f"(expected {expected_train}/{expected_test})"
        )
    if not legacy_intact:
        issues.append("warning: legacy ear x2 dir appears empty or missing")

    return {
        "method": method,
        "variant": f"Line {line}",
        "line": line,
        "output_root": str(v["output_root"].relative_to(PROJECT_ROOT)),
        "train_count": counts["train"],
        "test_count": counts["test"],
        "expected_train": expected_train,
        "expected_test": expected_test,
        "expected_shape": f"{v['target_points']}x3",
        "expected_input": v["expected_input"],
        "shape_ok": shape_ok,
        "bad_shapes": "; ".join(bad_shapes[:5]),
        "nan_inf": nan_inf,
        "class_count": len(class_to_idx),
        "label_mismatch_files": label_mismatch,
        "expected_ratio": 4.0,
        "legacy_ear_x2_intact": legacy_intact,
        "legacy_ear_x2_file_count": legacy_count,
        "overwrite_hits": overwrite_hits,
        "status": status,
        "issues": "; ".join(issues),
    }


def write_md(rows: list[dict], path: Path, method: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    overall = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    lines = [
        f"# {method.upper()} ×4 Smoke Audit",
        "",
        f"- Generated at: {ts}",
        f"- Overall: **{overall}**",
        "",
        "| Line | Train | Test | Shape | NaN/Inf | Classes | Legacy ×2 intact | Status |",
        "| --- | ---: | ---: | --- | --- | ---: | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['line']} | {r['train_count']} | {r['test_count']} | {r['expected_shape']} "
            f"| {r['nan_inf']} | {r['class_count']} | {r['legacy_ear_x2_intact']} | **{r['status']}** |"
        )
    lines.extend(["", "## Details", ""])
    for r in rows:
        lines.extend(
            [
                f"### Line {r['line']}",
                "",
                f"- Output: `{r['output_root']}`",
                f"- Shape OK: {r['shape_ok']}",
                f"- Issues: {r['issues'] or 'none'}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ×4 smoke outputs.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--smoke-tag", default="", help="Optional smoke output suffix, e.g. rerun")
    parser.add_argument("--expected-train", type=int, default=EXPECTED_TRAIN)
    parser.add_argument("--expected-test", type=int, default=EXPECTED_TEST)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    parser.add_argument("--line", choices=("A", "B"), default=None)
    args = parser.parse_args()

    method = normalize_method(args.method)
    if method not in METHODS:
        print(f"ERROR: invalid method: {args.method!r}", file=sys.stderr)
        return 1
    lines = [args.line] if args.line else ["A", "B"]
    rows = [
        audit_variant(
            method,
            line,
            smoke_tag=args.smoke_tag,
            expected_train=args.expected_train,
            expected_test=args.expected_test,
        )
        for line in lines
    ]

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{args.smoke_tag}" if args.smoke_tag else ""
    if args.csv_out and args.md_out:
        csv_path = args.csv_out
        md_path = args.md_out
    elif args.line:
        csv_path = reports / f"{method}_x4_smoke_audit_{args.line}{tag_suffix}.csv"
        md_path = reports / f"{method}_x4_smoke_audit_{args.line}{tag_suffix}.md"
    else:
        csv_path = reports / f"{method}_x4_smoke_audit.csv"
        md_path = reports / f"{method}_x4_smoke_audit.md"

    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_md(rows, md_path, method)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0 if all(r["status"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
