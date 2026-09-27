#!/usr/bin/env python3
"""Run and continuously summarize the full PointRCNN E1/E2 AP_R40 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = REPO / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
PYTHON = REPO / "venv_pointrcnn/bin/python"
VALIDATED_RUNNER = (
    REPO
    / "results/kitti_unified_x4_pointrcnn_eval_line_a_sampler_safe_v2_20260717_235037"
    / "commands/run_variant_eval.py"
)
VAL = REPO / "data/KITTI/ImageSets/val.txt"

METHOD_VARIANTS = (
    "original_x4_pdans",
    "downsampled_x4_pdans",
    "original_x4_pu_gcn",
    "downsampled_x4_pu_gcn",
    "original_x4_pu_edgeformer",
    "downsampled_x4_pu_edgeformer",
    "original_x4_pu_net",
    "downsampled_x4_pu_net",
)
E2_BASELINES = ("original_baseline", "downsampled_x4_baseline")


def read_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def experiment_specs(workspace: Path, experiment: str) -> list[tuple[str, Path, Path]]:
    if experiment == "e1":
        names = METHOD_VARIANTS
        input_root = workspace / "inputs/e1_default_16384"
    elif experiment == "e2":
        names = E2_BASELINES + METHOD_VARIANTS
        input_root = workspace / "inputs/e2_canonical_16384"
    else:
        raise ValueError(experiment)
    return [
        (
            f"{experiment}_{name}",
            input_root / name,
            workspace / "eval_outputs" / experiment / name,
        )
        for name in names
    ]


def write_live_summary(workspace: Path) -> None:
    rows = []
    for experiment in ("e1", "e2"):
        for run_name, input_dir, out_dir in experiment_specs(workspace, experiment):
            variant = run_name[len(experiment) + 1 :]
            summary = read_summary(out_dir / "result_summary.json")
            row = {
                "experiment": experiment.upper(),
                "variant": variant,
                "status": summary.get("status", "MISSING") if summary else "PENDING",
                "input_dir": str(input_dir),
                "runtime_seconds": summary.get("runtime_seconds", "") if summary else "",
                "prediction_files": summary.get("prediction_files", "") if summary else "",
                "empty_prediction_files": summary.get("empty_prediction_files", "") if summary else "",
            }
            metrics = summary.get("ap_r40_percent", {}) if summary else {}
            for metric in ("bbox_ap", "bev_ap", "3d_ap"):
                for difficulty in ("easy", "moderate", "hard"):
                    row[f"{metric}_{difficulty}"] = metrics.get(metric, {}).get(difficulty, "")
            rows.append(row)

    report_dir = workspace / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "e1_e2_live_ap_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PointRCNN E1/E2 live AP_R40 summary",
        "",
        "| Experiment | Variant | Status | 3D Easy | 3D Moderate | 3D Hard |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: object) -> str:
            return f"{float(value):.2f}" if value != "" else "—"

        lines.append(
            f"| {row['experiment']} | {row['variant']} | {row['status']} | "
            f"{fmt(row['3d_ap_easy'])} | {fmt(row['3d_ap_moderate'])} | {fmt(row['3d_ap_hard'])} |"
        )
    (report_dir / "e1_e2_live_ap_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--experiment", choices=("e1", "e2", "all"), default="all")
    parser.add_argument("--variant", action="append", help="Repeat to run only selected base variant names")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    experiments = ("e1", "e2") if args.experiment == "all" else (args.experiment,)
    selected = set(args.variant or [])

    write_live_summary(workspace)
    for experiment in experiments:
        for run_name, input_dir, out_dir in experiment_specs(workspace, experiment):
            variant = run_name[len(experiment) + 1 :]
            if selected and variant not in selected:
                continue
            prior = read_summary(out_dir / "result_summary.json")
            if prior and prior.get("status") == "PASS":
                print(f"SKIP_PASS {run_name}", flush=True)
                continue
            if not input_dir.is_dir():
                raise FileNotFoundError(f"input directory is not prepared: {input_dir}")
            command = [
                str(PYTHON),
                str(VALIDATED_RUNNER),
                "--name",
                run_name,
                "--input-dir",
                str(input_dir),
                "--out-dir",
                str(out_dir),
                "--split-file",
                str(VAL),
            ]
            log_path = workspace / "logs" / experiment / f"{variant}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"START {run_name} log={log_path}", flush=True)
            env = os.environ.copy()
            env["POINT_RCNN_SKIP_PYTHON_AP"] = "1"
            with log_path.open("ab") as log:
                proc = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            write_live_summary(workspace)
            if proc.returncode:
                print(f"FAIL {run_name} returncode={proc.returncode}", flush=True)
                return proc.returncode
            print(f"PASS {run_name}", flush=True)

    write_live_summary(workspace)
    print(f"EVAL_MATRIX_COMPLETE workspace={workspace}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
