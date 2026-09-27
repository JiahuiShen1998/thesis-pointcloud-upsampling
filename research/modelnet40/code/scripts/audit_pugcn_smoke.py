#!/usr/bin/env python3
"""Smoke audit for PU-GCN two-line protocol outputs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    lineA_paths,
    lineB_paths,
    reports_dir,
)


def load_manifest_rows(input_root: Path) -> list[dict]:
    rows: list[dict] = []
    for split in ("train", "test"):
        mp = input_root / "metadata" / f"{split}_manifest.csv"
        with open(mp, newline="", encoding="utf-8") as handle:
            for mrow in csv.DictReader(handle):
                row = dict(mrow)
                row["split"] = row.get("split") or split
                rows.append(row)
    rows.sort(key=lambda r: (r["split"], r["class_name"], r["shape_id"]))
    return rows


def audit_line(
    line: str,
    n: int,
    max_samples: int,
) -> list[dict]:
    if line == "lineA":
        input_root = ORIGINAL_ROOT
        strict_root = lineA_paths("pu_gcn")["strict_4N"]
        raw_root = lineA_paths("pu_gcn")["raw"]
        exp_in = n
        exp_out = n * 4
    else:
        input_root = DOWNSAMPLED_X4_ROOT
        strict_root = lineB_paths("pu_gcn")["strict_N"]
        raw_root = lineB_paths("pu_gcn")["raw"]
        exp_in = n // 4
        exp_out = n

    per_split = max(1, max_samples // 2)
    manifest_rows = load_manifest_rows(input_root)
    selected: list[dict] = []
    counts = {"train": 0, "test": 0}
    for mrow in manifest_rows:
        split = mrow["split"]
        if counts[split] >= per_split:
            continue
        selected.append(mrow)
        counts[split] += 1

    rows: list[dict] = []
    for mrow in selected:
        split = mrow["split"]
        cls, sid = mrow["class_name"], mrow["shape_id"]
        inp = input_root / split / cls / f"{sid}.npy"
        raw_p = raw_root / split / cls / f"{sid}.npy"
        strict_p = strict_root / split / cls / f"{sid}.npy"

        rec = {
            "line": line,
            "sample": f"{split}/{cls}/{sid}",
            "input_path": str(inp),
            "input_shape": "",
            "raw_output_path": str(raw_p),
            "raw_output_shape": "",
            "strict_output_path": str(strict_p),
            "strict_output_shape": "",
            "nan": "no",
            "inf": "no",
            "status": "FAIL",
            "error_message": "",
        }
        try:
            if not inp.is_file():
                raise FileNotFoundError(f"missing input: {inp}")
            inp_arr = np.load(inp)
            rec["input_shape"] = str(tuple(inp_arr.shape))
            if inp_arr.shape != (exp_in, 3):
                raise ValueError(f"expected input ({exp_in},3), got {inp_arr.shape}")
            if not strict_p.is_file():
                raise FileNotFoundError(f"missing strict: {strict_p}")
            if raw_p.is_file():
                raw_arr = np.load(raw_p)
                rec["raw_output_shape"] = str(tuple(raw_arr.shape))
            strict_arr = np.load(strict_p)
            rec["strict_output_shape"] = str(tuple(strict_arr.shape))
            if strict_arr.shape != (exp_out, 3):
                raise ValueError(f"expected strict ({exp_out},3), got {strict_arr.shape}")
            if np.isnan(strict_arr).any():
                rec["nan"] = "yes"
                raise ValueError("NaN in strict output")
            if np.isinf(strict_arr).any():
                rec["inf"] = "yes"
                raise ValueError("Inf in strict output")
            rec["status"] = "PASS"
        except Exception as exc:  # noqa: BLE001
            rec["error_message"] = str(exc)
        rows.append(rec)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", choices=("lineA", "lineB"), default=None)
    parser.add_argument("--max-samples", type=int, default=10)
    args = parser.parse_args()

    n = detect_original_point_count()
    lines = [args.line] if args.line else ("lineA", "lineB")
    all_rows: list[dict] = []
    for line in lines:
        all_rows.extend(audit_line(line, n, args.max_samples))

    rep = reports_dir()
    csv_path = rep / "modelnet40_pugcn_smoke_audit.csv"
    md_path = rep / "modelnet40_pugcn_smoke_audit.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    fields = list(all_rows[0].keys()) if all_rows else ["status"]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    lineA_pass = all(r["status"] == "PASS" for r in all_rows if r["line"] == "lineA") if any(r["line"] == "lineA" for r in all_rows) else True
    lineB_pass = all(r["status"] == "PASS" for r in all_rows if r["line"] == "lineB") if any(r["line"] == "lineB" for r in all_rows) else True
    overall = "PASS" if lineA_pass and lineB_pass and all_rows else "FAIL"

    md_lines = [
        "# PU-GCN Smoke Audit",
        "",
        f"- Generated at: {ts}",
        f"- Overall: **{overall}**",
        f"- Line A: **{'PASS' if lineA_pass else 'FAIL'}**",
        f"- Line B: **{'PASS' if lineB_pass else 'FAIL'}**",
        "",
        "| line | sample | input_shape | strict_shape | nan | inf | status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in all_rows:
        md_lines.append(
            f"| {r['line']} | {r['sample']} | {r['input_shape']} | {r['strict_output_shape']} | "
            f"{r['nan']} | {r['inf']} | {r['status']} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"overall={overall} lineA={'PASS' if lineA_pass else 'FAIL'} lineB={'PASS' if lineB_pass else 'FAIL'}")
    print(f"Wrote {csv_path}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
