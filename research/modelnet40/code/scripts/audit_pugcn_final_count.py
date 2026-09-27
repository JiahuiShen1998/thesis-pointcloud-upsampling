#!/usr/bin/env python3
"""Final count audit for PU-GCN two-line protocol outputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    lineA_paths,
    lineB_paths,
    reports_dir,
)

EXPECTED_TOTAL = EXPECTED_TRAIN + EXPECTED_TEST


def load_manifest_keys(input_root: Path) -> set[str]:
    keys: set[str] = set()
    for split in ("train", "test"):
        mp = input_root / "metadata" / f"{split}_manifest.csv"
        with open(mp, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                keys.add(f"{split}/{row['class_name']}/{row['shape_id']}")
    return keys


def audit_line(line: str, n: int) -> dict:
    if line == "lineA":
        input_root = ORIGINAL_ROOT
        raw_root = lineA_paths("pu_gcn")["raw"]
        strict_root = lineA_paths("pu_gcn")["strict_4N"]
        exp_out = n * 4
    else:
        input_root = DOWNSAMPLED_X4_ROOT
        raw_root = lineB_paths("pu_gcn")["raw"]
        strict_root = lineB_paths("pu_gcn")["strict_N"]
        exp_out = n

    manifest = load_manifest_keys(input_root)
    per_split = defaultdict(lambda: {"raw": 0, "strict": 0})

    min_pts = max_pts = None
    exact = nan_c = inf_c = bad_shape = 0
    strict_keys: set[str] = set()

    for npy in strict_root.rglob("*.npy") if strict_root.is_dir() else []:
        rel = npy.relative_to(strict_root)
        key = f"{rel.parts[0]}/{rel.parts[1]}/{npy.stem}"
        strict_keys.add(key)
        split = rel.parts[0]
        per_split[split]["strict"] += 1
        arr = np.load(npy)
        pts = arr.shape[0]
        min_pts = pts if min_pts is None else min(min_pts, pts)
        max_pts = pts if max_pts is None else max(max_pts, pts)
        if arr.shape == (exp_out, 3):
            exact += 1
        else:
            bad_shape += 1
        if np.isnan(arr).any():
            nan_c += 1
        if np.isinf(arr).any():
            inf_c += 1

    for npy in raw_root.rglob("*.npy") if raw_root.is_dir() else []:
        per_split[npy.relative_to(raw_root).parts[0]]["raw"] += 1

    raw_count = sum(v["raw"] for v in per_split.values())
    strict_count = sum(v["strict"] for v in per_split.values())
    missing = len(manifest - strict_keys)

    status = "PASS"
    if strict_count != EXPECTED_TOTAL or bad_shape or nan_c or inf_c or missing:
        status = "FAIL"

    rows = []
    for split in ("train", "test"):
        exp = EXPECTED_TRAIN if split == "train" else EXPECTED_TEST
        rows.append({
            "method": "PU-GCN",
            "line": line,
            "split": split,
            "class": "all",
            "raw_count": per_split[split]["raw"],
            "strict_count": per_split[split]["strict"],
            "expected_count": exp,
            "min_points": min_pts if min_pts is not None else "",
            "max_points": max_pts if max_pts is not None else "",
            "exact_point_count_samples": exact if split == "all" else "",
            "nan_count": nan_c if split == "all" else "",
            "inf_count": inf_c if split == "all" else "",
            "missing_count": missing if split == "all" else "",
            "status": status if split == "train" else "",
        })

    rows.append({
        "method": "PU-GCN",
        "line": line,
        "split": "all",
        "class": "all",
        "raw_count": raw_count,
        "strict_count": strict_count,
        "expected_count": EXPECTED_TOTAL,
        "min_points": min_pts if min_pts is not None else "",
        "max_points": max_pts if max_pts is not None else "",
        "exact_point_count_samples": exact,
        "nan_count": nan_c,
        "inf_count": inf_c,
        "missing_count": missing,
        "status": status,
    })
    return {"rows": rows, "status": status, "strict_count": strict_count, "exp_out": exp_out}


def main() -> int:
    n = detect_original_point_count()
    results = [audit_line("lineA", n), audit_line("lineB", n)]
    all_rows = [r for res in results for r in res["rows"]]

    rep = reports_dir()
    csv_path = rep / "modelnet40_pugcn_final_count_audit.csv"
    md_path = rep / "modelnet40_pugcn_final_count_audit.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    md_lines = [
        "# PU-GCN Final Count Audit",
        "",
        f"- Generated at: {ts}",
        f"- Overall: **{overall}**",
        "",
        "| line | strict_count | expected | exact_shape | nan | inf | missing | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for res in results:
        row = res["rows"][-1]
        md_lines.append(
            f"| {row['line']} | {row['strict_count']} | {row['expected_count']} | "
            f"{row['exact_point_count_samples']} | {row['nan_count']} | {row['inf_count']} | "
            f"{row['missing_count']} | {row['status']} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"overall={overall}")
    print(f"Wrote {csv_path}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
