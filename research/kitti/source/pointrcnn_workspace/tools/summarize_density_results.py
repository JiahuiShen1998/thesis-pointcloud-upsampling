#!/usr/bin/env python3
"""Summarize unified density-evaluation experiments.

The script is method-agnostic:
  - it scans experiment folders under ``experiments/density_baselines``,
  - reads the standard method summary CSV emitted by the evaluator,
  - merges point-cloud metrics with detector placeholders or detector AP data
    when available,
  - and produces a comparison table plus a Markdown report.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.check_pointrcnn_integrity import build_report as build_integrity_report
from tools.density_metric_schema import NA_VALUE, csv_fieldnames

EXP_ROOT = REPO_ROOT / "experiments" / "density_baselines"
SUMMARY_ROOT = EXP_ROOT / "summary"


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Summarize unified density-baseline results")
    parser.add_argument("--experiments-root", type=Path, default=EXP_ROOT)
    parser.add_argument("--summary-root", type=Path, default=SUMMARY_ROOT)
    return parser.parse_args()


def _read_csv_row(path: Path) -> Optional[Dict[str, str]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    return rows[0]


def _read_parsed_ap(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out: Dict[str, Dict[str, str]] = {}
        for row in reader:
            metric = str(row.get("metric", "")).strip().lower()
            if metric:
                out[metric] = row
        return out


def _scan_experiments(root: Path) -> List[Path]:
    candidates = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"summary", "_preprocessing_records"}:
            continue
        if child.name.startswith("."):
            continue
        if (child / "evaluation" / "method_summary.csv").exists():
            candidates.append(child)
    return candidates


def _to_float(value: object) -> Optional[float]:
    if value in (None, "", NA_VALUE):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _format_num(value: object) -> str:
    num = _to_float(value)
    if num is None:
        return NA_VALUE
    return f"{num:.2f}"


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, NA_VALUE) for field in csv_fieldnames()})


def _combine_rows(experiment_dir: Path) -> Dict[str, object]:
    method_row = _read_csv_row(experiment_dir / "evaluation" / "method_summary.csv") or {}
    ap_rows = _read_parsed_ap(experiment_dir / "evaluation" / "parsed_ap_results.csv")

    combined: Dict[str, object] = {field: NA_VALUE for field in csv_fieldnames()}
    for key, value in method_row.items():
        if key in combined:
            combined[key] = value

    combined["experiment_dir"] = str(experiment_dir.resolve())

    if ap_rows:
        for metric in ("bbox", "bev", "3d"):
            row = ap_rows.get(metric)
            if not row:
                continue
            combined[f"detector_{metric}_ap_easy"] = row.get("easy", NA_VALUE)
            combined[f"detector_{metric}_ap_moderate"] = row.get("moderate", NA_VALUE)
            combined[f"detector_{metric}_ap_hard"] = row.get("hard", NA_VALUE)

    return combined


def _delta_row(row: Dict[str, object], base: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    for field in [
        "input_points",
        "output_points",
        "added_points",
        "removed_points",
        "upsampling_ratio",
        "nn_mean_before",
        "nn_mean_after",
        "nn_median_before",
        "nn_median_after",
        "nn_std_before",
        "nn_std_after",
        "bev_density_mean_before",
        "bev_density_mean_after",
        "bev_density_median_before",
        "bev_density_median_after",
        "bev_density_std_before",
        "bev_density_std_after",
        "intensity_mean_before",
        "intensity_mean_after",
        "intensity_std_before",
        "intensity_std_after",
        "detector_bbox_ap_moderate",
        "detector_bev_ap_moderate",
        "detector_3d_ap_moderate",
    ]:
        cur = _to_float(row.get(field))
        ref = _to_float(base.get(field))
        if cur is None or ref is None:
            out[f"delta_{field}"] = NA_VALUE
        else:
            out[f"delta_{field}"] = round(cur - ref, 4)
    return out


def _markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    experiments_root = args.experiments_root.resolve()
    summary_root = args.summary_root.resolve()
    summary_root.mkdir(parents=True, exist_ok=True)

    exp_dirs = _scan_experiments(experiments_root)
    if not exp_dirs:
        raise RuntimeError(f"No experiment folders found under {experiments_root}")

    combined_rows = [_combine_rows(exp_dir) for exp_dir in exp_dirs]
    original = next((row for row in combined_rows if str(row.get("experiment_name", "")).lower() == "original"), None)
    if original is None:
        original = combined_rows[0]
    comparison_rows = [_delta_row(row, original) for row in combined_rows]

    _write_csv(summary_root / "all_ap_results.csv", combined_rows)

    md_rows = []
    headers = [
        "Experiment",
        "Method",
        "Type",
        "ref velodyne",
        "proc velodyne",
        "input pts",
        "output pts",
        "NN before",
        "NN after",
        "BEV before",
        "BEV after",
        "3D M",
        "Δ3D M",
    ]
    for row, delta_row in zip(combined_rows, comparison_rows):
        md_rows.append(
            [
                str(row.get("experiment_name", NA_VALUE)),
                str(row.get("method_name", NA_VALUE)),
                str(row.get("processing_type", NA_VALUE)),
                str(row.get("reference_velodyne", NA_VALUE)),
                str(row.get("processed_velodyne", NA_VALUE)),
                _format_num(row.get("input_points")),
                _format_num(row.get("output_points")),
                _format_num(row.get("nn_mean_before")),
                _format_num(row.get("nn_mean_after")),
                _format_num(row.get("bev_density_mean_before")),
                _format_num(row.get("bev_density_mean_after")),
                _format_num(row.get("detector_3d_ap_moderate")),
                _format_num(delta_row.get("delta_detector_3d_ap_moderate")),
            ]
        )
    (summary_root / "all_ap_results.md").write_text(_markdown_table(headers, md_rows), encoding="utf-8")

    report_lines = [
        "# Unified Density Experiment Summary",
        "",
        "## Protocol",
        "",
        "- The table schema is shared by all experiments.",
        "- Point-cloud metrics are always present.",
        "- Detector AP columns remain in the same schema and may show `N/A` until detector outputs exist.",
        "",
        "## Experiments",
        "",
    ]
    for row, delta_row in zip(combined_rows, comparison_rows):
        report_lines.extend(
            [
                f"### {row.get('experiment_name', NA_VALUE)}",
                "",
                f"- method_name: {row.get('method_name', NA_VALUE)}",
                f"- processing_type: {row.get('processing_type', NA_VALUE)}",
                f"- reference_velodyne: {row.get('reference_velodyne', NA_VALUE)}",
                f"- processed_velodyne: {row.get('processed_velodyne', NA_VALUE)}",
                f"- input_points: {_format_num(row.get('input_points'))}",
                f"- output_points: {_format_num(row.get('output_points'))}",
                f"- upsampling_ratio: {_format_num(row.get('upsampling_ratio'))}",
                f"- nn_mean_before: {_format_num(row.get('nn_mean_before'))}",
                f"- nn_mean_after: {_format_num(row.get('nn_mean_after'))}",
                f"- detector_3d_ap_moderate: {_format_num(row.get('detector_3d_ap_moderate'))}",
                f"- delta_detector_3d_ap_moderate_vs_original: {_format_num(delta_row.get('delta_detector_3d_ap_moderate'))}",
                "",
            ]
        )
    (summary_root / "experiment_summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    integrity_text, _ = build_integrity_report()
    (summary_root / "integrity_check.txt").write_text(integrity_text, encoding="utf-8")

    print(f"Wrote summary outputs to {summary_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

