#!/usr/bin/env python3
"""Run PU-GCN smoke test for ModelNet40 Line A / Line B and write audit reports."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pugcn_paths import DEFAULT_PUGCN_PYTHON, DEFAULT_SEED, resolve_project_root  # noqa: E402


def audit_sample(line: str, input_path: Path, raw_path: Path, strict_path: Path, expected_in: int, expected_out: int) -> dict:
    row = {
        "line": line,
        "sample": str(input_path),
        "input_path": str(input_path),
        "input_shape": "",
        "raw_output_path": str(raw_path),
        "raw_output_shape": "",
        "strict_output_path": str(strict_path),
        "strict_output_shape": "",
        "nan": "",
        "inf": "",
        "status": "FAIL",
        "error_message": "",
    }
    try:
        inp = np.load(input_path)
        row["input_shape"] = str(tuple(inp.shape))
        if inp.shape != (expected_in, 3):
            raise ValueError(f"input shape {inp.shape} != ({expected_in}, 3)")
        if not raw_path.exists() or not strict_path.exists():
            raise FileNotFoundError("raw or strict output missing")
        raw = np.load(raw_path)
        strict = np.load(strict_path)
        row["raw_output_shape"] = str(tuple(raw.shape))
        row["strict_output_shape"] = str(tuple(strict.shape))
        row["nan"] = str(bool(not np.isfinite(raw).all() or not np.isfinite(strict).all()))
        row["inf"] = str(bool(np.isinf(raw).any() or np.isinf(strict).any()))
        if strict.shape != (expected_out, 3):
            raise ValueError(f"strict shape {strict.shape} != ({expected_out}, 3)")
        if row["nan"] == "True" or row["inf"] == "True":
            raise ValueError("NaN or Inf detected")
        row["status"] = "PASS"
    except Exception as exc:  # noqa: BLE001
        row["error_message"] = str(exc)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--python-bin", default=str(DEFAULT_PUGCN_PYTHON))
    args = parser.parse_args()

    project_root = resolve_project_root(Path(args.project_root) if args.project_root else None)
    runner = project_root / "scripts" / "run_pugcn_modelnet40_x4.py"
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    for line in ("lineA", "lineB"):
        cmd = [
            str(args.python_bin),
            str(runner),
            "--line",
            line,
            "--project-root",
            str(project_root),
            "--max-samples",
            str(args.max_samples),
            "--seed",
            str(args.seed),
            "--resume",
        ]
        print("Running:", " ".join(cmd))
        proc = subprocess.run(cmd, cwd=str(project_root))
        if proc.returncode != 0:
            print(f"WARNING: runner returned {proc.returncode} for {line}")

    rows: list[dict] = []
    cfg = {
        "lineA": {
            "input_root": project_root / "datasets" / "modelnet40_original_1024",
            "raw_root": project_root / "datasets" / "lineA_original_up" / "raw" / "pu_gcn",
            "strict_root": project_root / "datasets" / "lineA_original_up" / "strict_4N" / "pu_gcn",
            "expected_in": 1024,
            "expected_out": 4096,
        },
        "lineB": {
            "input_root": project_root / "datasets" / "modelnet40_downsampled_x4",
            "raw_root": project_root / "datasets" / "lineB_downsampled_x4_up" / "raw" / "pu_gcn",
            "strict_root": project_root / "datasets" / "lineB_downsampled_x4_up" / "strict_N" / "pu_gcn",
            "expected_in": 256,
            "expected_out": 1024,
        },
    }
    for line, c in cfg.items():
        input_root = c["input_root"]
        if not input_root.is_dir():
            continue
        count = 0
        for split in ("train", "test"):
            split_dir = input_root / split
            if not split_dir.is_dir():
                continue
            for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                for npy in sorted(class_dir.glob("*.npy")):
                    rel = npy.relative_to(input_root)
                    rows.append(
                        audit_sample(
                            line,
                            npy,
                            c["raw_root"] / rel,
                            c["strict_root"] / rel,
                            c["expected_in"],
                            c["expected_out"],
                        )
                    )
                    count += 1
                    if count >= args.max_samples:
                        break
                if count >= args.max_samples:
                    break
            if count >= args.max_samples:
                break

    csv_path = reports / "modelnet40_pugcn_smoke_audit.csv"
    md_path = reports / "modelnet40_pugcn_smoke_audit.md"
    fields = [
        "line",
        "sample",
        "input_path",
        "input_shape",
        "raw_output_path",
        "raw_output_shape",
        "strict_output_path",
        "strict_output_shape",
        "nan",
        "inf",
        "status",
        "error_message",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    pass_a = sum(1 for r in rows if r["line"] == "lineA" and r["status"] == "PASS")
    pass_b = sum(1 for r in rows if r["line"] == "lineB" and r["status"] == "PASS")
    total_a = sum(1 for r in rows if r["line"] == "lineA")
    total_b = sum(1 for r in rows if r["line"] == "lineB")
    smoke_pass = pass_a == total_a and pass_b == total_b and total_a > 0 and total_b > 0

    md = [
        "# ModelNet40 PU-GCN Smoke Audit",
        "",
        f"- project_root: `{project_root}`",
        f"- Line A PASS: `{pass_a}/{total_a}`",
        f"- Line B PASS: `{pass_b}/{total_b}`",
        f"- Overall smoke: **{'PASS' if smoke_pass else 'FAIL'}**",
        "",
        "## Criteria",
        "",
        "- Line A: input (1024,3), strict (4096,3), no NaN/Inf",
        "- Line B: input (256,3), strict (1024,3), no NaN/Inf",
        "",
        "## Notes",
        "",
        "- Smoke uses real PU-GCN inference via `run_pugcn_modelnet40_x4.py`.",
        "- Full 12311-sample jobs require HPC PROJECT_ROOT with complete protocol datasets.",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"smoke_pass={smoke_pass}")
    return 0 if smoke_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
