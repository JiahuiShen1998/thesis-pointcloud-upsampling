#!/usr/bin/env python3
"""Full-generation audit for main-method ×4 outputs."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from upsampling_x4_common import (
    LEGACY_EAR_X2,
    PROJECT_ROOT,
    full_line_config,
    load_class_to_idx,
)
from upsampling_x4_factory import METHODS

EXPECTED_TRAIN = 9843
EXPECTED_TEST = 2468


def audit_variant(method: str, line: str) -> dict:
    v = full_line_config(method, line)
    class_to_idx = load_class_to_idx()

    counts = {"train": 0, "test": 0}
    bad_shape = 0
    nan_inf = False
    ratios: list[float] = []

    for split in ("train", "test"):
        split_dir = v["output_root"] / split
        if not split_dir.is_dir():
            continue
        for fpath in split_dir.rglob("*.npy"):
            counts[split] += 1
            arr = np.load(fpath)
            if arr.shape != (v["target_points"], 3):
                bad_shape += 1
            if not np.isfinite(arr).all():
                nan_inf = True
            ratios.append(arr.shape[0] / v["expected_input"])

    legacy_count = sum(1 for _ in LEGACY_EAR_X2.rglob("*.npy")) if LEGACY_EAR_X2.is_dir() else 0
    count_ok = counts["train"] == EXPECTED_TRAIN and counts["test"] == EXPECTED_TEST
    shape_ok = bad_shape == 0 and counts["train"] + counts["test"] > 0
    ratio_ok = bool(ratios) and min(ratios) == max(ratios) == 4.0

    status = "PASS" if count_ok and shape_ok and not nan_inf and ratio_ok else "FAIL"
    issues = []
    if not count_ok:
        issues.append(f"counts {counts['train']}/{counts['test']} != {EXPECTED_TRAIN}/{EXPECTED_TEST}")
    if not shape_ok:
        issues.append(f"bad shapes: {bad_shape}")
    if nan_inf:
        issues.append("NaN/Inf")
    if not ratio_ok:
        issues.append("ratio mismatch")

    return {
        "method": method,
        "variant": f"Line {line}",
        "line": line,
        "output_root": str(v["output_root"].relative_to(PROJECT_ROOT)),
        "train_count": counts["train"],
        "test_count": counts["test"],
        "expected_train": EXPECTED_TRAIN,
        "expected_test": EXPECTED_TEST,
        "expected_shape": f"{v['target_points']}x3",
        "bad_shape_files": bad_shape,
        "nan_inf": nan_inf,
        "actual_ratio": f"{ratios[0]:.1f}" if ratios else "",
        "class_count": len(class_to_idx),
        "legacy_ear_x2_intact": legacy_count > 0,
        "status": status,
        "issues": "; ".join(issues),
    }


def write_md(rows: list[dict], path: Path, method: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    overall = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    lines = [
        f"# {method.upper()} ×4 Full Generation Audit",
        "",
        f"- Generated at: {ts}",
        f"- Overall: **{overall}**",
        "",
        "| Line | Train | Test | Shape | NaN/Inf | Ratio | Legacy ×2 | Status |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['line']} | {r['train_count']} | {r['test_count']} | {r['expected_shape']} "
            f"| {r['nan_inf']} | {r['actual_ratio']} | {r['legacy_ear_x2_intact']} | **{r['status']}** |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ×4 full generation outputs.")
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()

    method = args.method.lower()
    rows = [audit_variant(method, line) for line in ("A", "B")]
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / f"{method}_x4_full_generation_audit.csv"
    md_path = reports / f"{method}_x4_full_generation_audit.md"
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
