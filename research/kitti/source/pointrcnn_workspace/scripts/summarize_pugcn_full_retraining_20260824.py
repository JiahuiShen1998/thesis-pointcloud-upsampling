#!/usr/bin/env python3
"""Summarize full PU-GCN adaptation results for PointRCNN and CenterPoint."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results/pugcn_full_retrain_20260824"
OLD_PR = ROOT / "results/pointrcnn_finetune64_six_arms_20260824/evaluations"
OLD_CP = ROOT / "results/kitti_patch_punet_causal_ablation_v1_20260731/detectors/centerpoint"


def point_metrics(run_dir: Path) -> dict[str, float]:
    logs = sorted(run_dir.rglob("log_eval_one.txt"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"PointRCNN evaluation log missing under {run_dir}")
    text = logs[-1].read_text(errors="replace")
    match = re.search(
        r"Car AP@0\.70, 0\.70, 0\.70:\s*"
        r"bbox AP:([^\n]+)\s*bev\s+AP:([^\n]+)\s*3d\s+AP:([^\n]+)",
        text,
    )
    if not match:
        raise ValueError(f"Car AP block missing in {logs[-1]}")
    bev = [float(value.strip()) for value in match.group(2).split(",")]
    d3 = [float(value.strip()) for value in match.group(3).split(",")]
    return {
        "3d_easy": d3[0],
        "3d_moderate": d3[1],
        "3d_hard": d3[2],
        "bev_moderate": bev[1],
    }


def center_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text())
    car = payload["metrics_percent"]["Car"]
    return {
        "3d_easy": float(car["3d_ap_r40"]["easy"]),
        "3d_moderate": float(car["3d_ap_r40"]["moderate"]),
        "3d_hard": float(car["3d_ap_r40"]["hard"]),
        "bev_moderate": float(car["bev_ap_r40"]["moderate"]),
    }


def make_row(
    detector: str,
    line: str,
    input_name: str,
    weights: str,
    metrics: dict[str, float],
    baseline_moderate: float,
) -> dict[str, object]:
    return {
        "detector": detector,
        "line": line,
        "input": input_name,
        "weights": weights,
        **metrics,
        "line_baseline_3d_moderate": baseline_moderate,
        "delta_vs_line_baseline_3d_moderate": metrics["3d_moderate"]
        - baseline_moderate,
    }


def main() -> None:
    pr_a_base = point_metrics(OLD_PR / "line_a_baseline/pretrained")
    pr_a_pugcn = point_metrics(EXP / "evaluations/pointrcnn/line_a_pugcn")
    pr_b_base = point_metrics(EXP / "evaluations/pointrcnn/line_b_baseline")
    pr_b_pugcn = point_metrics(EXP / "evaluations/pointrcnn/line_b_pugcn")

    cp_a_base = center_metrics(OLD_CP / "original_baseline/result_summary.json")
    cp_a_pugcn = center_metrics(
        EXP / "evaluations/centerpoint/line_a_pugcn/result_summary.json"
    )
    cp_b_base = center_metrics(
        EXP / "evaluations/centerpoint/line_b_baseline/result_summary.json"
    )
    cp_b_pugcn = center_metrics(
        EXP / "evaluations/centerpoint/line_b_pugcn/result_summary.json"
    )

    rows = [
        make_row("PointRCNN", "A", "baseline", "official", pr_a_base, pr_a_base["3d_moderate"]),
        make_row("PointRCNN", "A", "PU-GCN", "full adapted", pr_a_pugcn, pr_a_base["3d_moderate"]),
        make_row("PointRCNN", "B", "baseline", "full adapted", pr_b_base, pr_b_base["3d_moderate"]),
        make_row("PointRCNN", "B", "PU-GCN", "full adapted", pr_b_pugcn, pr_b_base["3d_moderate"]),
        make_row("CenterPoint", "A", "baseline", "official", cp_a_base, cp_a_base["3d_moderate"]),
        make_row("CenterPoint", "A", "PU-GCN", "full adapted", cp_a_pugcn, cp_a_base["3d_moderate"]),
        make_row("CenterPoint", "B", "baseline", "full adapted", cp_b_base, cp_b_base["3d_moderate"]),
        make_row("CenterPoint", "B", "PU-GCN", "full adapted", cp_b_pugcn, cp_b_base["3d_moderate"]),
    ]

    report_dir = EXP / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "full_retraining_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "full_retraining_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )

    lines = [
        "# PU-GCN full-split detector adaptation",
        "",
        "All results use the fixed 256-frame evaluation split.",
        "",
        "| Detector | Line | Input | Weights | 3D Easy | 3D Moderate | 3D Hard | BEV Moderate | Delta vs line baseline |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {detector} | {line} | {input} | {weights} | {3d_easy:.4f} | "
            "{3d_moderate:.4f} | {3d_hard:.4f} | {bev_moderate:.4f} | "
            "{delta_vs_line_baseline_3d_moderate:+.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Line A uses the official full-resolution detector baseline. Line B uses a paired full-split downsampled-input adaptation baseline.",
        ]
    )
    (report_dir / "full_retraining_report.md").write_text("\n".join(lines) + "\n")
    print(report_dir / "full_retraining_report.md")


if __name__ == "__main__":
    main()
