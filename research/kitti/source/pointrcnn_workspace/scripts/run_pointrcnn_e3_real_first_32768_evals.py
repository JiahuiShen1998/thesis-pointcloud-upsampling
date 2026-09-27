#!/usr/bin/env python3
"""Run and continuously summarize PointRCNN E3 real-first 32768 AP_R40."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
PYTHON = REPO / "venv_pointrcnn/bin/python"
RUNNER = REPO / "scripts/run_variant_eval_real_first_32768.py"
VAL = REPO / "data/KITTI/ImageSets/val.txt"
VARIANTS = (
    "original_baseline",
    "downsampled_x4_baseline",
    "original_x4_pdans",
    "downsampled_x4_pdans",
    "original_x4_pu_gcn",
    "downsampled_x4_pu_gcn",
    "original_x4_pu_edgeformer",
    "downsampled_x4_pu_edgeformer",
    "original_x4_pu_net",
    "downsampled_x4_pu_net",
)


def read_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_live_summary(workspace: Path) -> None:
    rows = []
    for variant in VARIANTS:
        input_dir = workspace / "inputs/e3_real_first_32768" / variant
        out_dir = workspace / "eval_outputs/e3_real_first_32768" / variant
        summary = read_summary(out_dir / "result_summary.json")
        metrics = summary.get("ap_r40_percent", {}) if summary else {}
        row = {
            "experiment": "E3_REAL_FIRST_32768",
            "variant": variant,
            "status": summary.get("status", "MISSING") if summary else "PENDING",
            "input_dir": str(input_dir),
            "runtime_seconds": summary.get("runtime_seconds", "") if summary else "",
            "prediction_files": summary.get("prediction_files", "") if summary else "",
            "empty_prediction_files": summary.get("empty_prediction_files", "") if summary else "",
        }
        for metric in ("bbox_ap", "bev_ap", "3d_ap"):
            for difficulty in ("easy", "moderate", "hard"):
                row[f"{metric}_{difficulty}"] = metrics.get(metric, {}).get(difficulty, "")
        rows.append(row)

    report_dir = workspace / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "e3_real_first_32768_live_ap_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PointRCNN E3 real-first 32768 live AP_R40 summary",
        "",
        "| Variant | Status | 3D Easy | 3D Moderate | 3D Hard |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: object) -> str:
            return f"{float(value):.2f}" if value != "" else "—"

        lines.append(
            f"| {row['variant']} | {row['status']} | {fmt(row['3d_ap_easy'])} | "
            f"{fmt(row['3d_ap_moderate'])} | {fmt(row['3d_ap_hard'])} |"
        )
    (report_dir / "e3_real_first_32768_live_ap_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--variant", action="append", help="Repeat to run selected variants")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    selected = set(args.variant or [])

    write_live_summary(workspace)
    for variant in VARIANTS:
        if selected and variant not in selected:
            continue
        input_dir = workspace / "inputs/e3_real_first_32768" / variant
        out_dir = workspace / "eval_outputs/e3_real_first_32768" / variant
        prior = read_summary(out_dir / "result_summary.json")
        if prior and prior.get("status") == "PASS":
            print(f"SKIP_PASS e3_{variant}", flush=True)
            continue
        if not input_dir.is_dir():
            raise FileNotFoundError(f"missing E3 input: {input_dir}")
        command = [
            str(PYTHON),
            str(RUNNER),
            "--name",
            f"e3_real_first_32768_{variant}",
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_dir),
            "--split-file",
            str(VAL),
        ]
        log_path = workspace / "logs/e3_real_first_32768" / f"{variant}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"START e3_{variant} log={log_path}", flush=True)
        env = os.environ.copy()
        env["POINT_RCNN_SKIP_PYTHON_AP"] = "1"
        with log_path.open("ab") as log:
            proc = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        write_live_summary(workspace)
        if proc.returncode:
            print(f"FAIL e3_{variant} returncode={proc.returncode}", flush=True)
            return proc.returncode
        print(f"PASS e3_{variant}", flush=True)

    write_live_summary(workspace)
    print(f"E3_EVAL_MATRIX_COMPLETE workspace={workspace}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
