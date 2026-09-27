#!/usr/bin/env python3
"""Finalize PU-EdgeFormer PointNet++ full-training audit, tables, and figures.

Run only after both Line A and Line B full jobs complete.
Does not start detector / KITTI AP / geometry recompute.
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
RESULTS = PROJECT / "pointnet2_results" / "x4_two_line_final"
FIG_ROOT = PROJECT / "figures" / "modelnet40" / "pointnet2_final_with_pu_edgeformer"
ACC_ABS = FIG_ROOT / "accuracy_absolute"
ACC_DELTA = FIG_ROOT / "accuracy_delta_methods_only"
FINAL_COMBINED = FIG_ROOT / "final_combined"

METHOD_COLORS = {
    "Original baseline": "#4C72B0",
    "Downsampled baseline": "#55A868",
    "EAR": "#C44E52",
    "PDANS": "#8172B3",
    "PU-Net": "#CCB974",
    "PU-GCN": "#64B5CD",
    "PU-EdgeFormer": "#B279A2",
}
METHODS_ONLY = ["EAR", "PDANS", "PU-Net", "PU-GCN", "PU-EdgeFormer"]

JOB_B = "1749441"
JOB_A = "1749442"

LINEA_METRICS_PATHS = {
    "Original baseline 1024": RESULTS / "lineA_original_baseline" / "metrics.json",
    "Original + EAR 4096": RESULTS / "lineA_original_up" / "ear" / "metrics.json",
    "Original + PDANS 4096": RESULTS / "lineA_original_up" / "pdans" / "metrics.json",
    "Original + PU-Net 4096": RESULTS / "lineA_original_up" / "pu_net" / "metrics.json",
    "Original + PU-GCN 4096": RESULTS / "lineA_original_up" / "pu_gcn" / "metrics.json",
}

LINEB_METRICS_PATHS = {
    "Downsampled x4 baseline": RESULTS / "lineB_downsampled_x4_baseline" / "metrics.json",
    "Downsampled x4 + EAR": RESULTS / "lineB_downsampled_x4_up" / "ear" / "metrics.json",
    "Downsampled x4 + PDANS": RESULTS / "lineB_downsampled_x4_up" / "pdans" / "metrics.json",
    "Downsampled x4 + PU-Net": RESULTS / "lineB_downsampled_x4_up" / "pu_net" / "metrics.json",
    "Downsampled x4 + PU-GCN": RESULTS / "lineB_downsampled_x4_up" / "pu_gcn" / "metrics.json",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def pp_str(pp: float) -> str:
    sign = "+" if pp >= 0 else ""
    return f"{sign}{pp:.2f} pp"


def to_pp(frac: float) -> float:
    return frac * 100.0


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sacct_info(job_id: str) -> dict:
    try:
        out = subprocess.check_output(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,State,ExitCode,Elapsed,NodeList,End",
                "-P",
                "-n",
            ],
            text=True,
        )
    except Exception as exc:  # noqa: BLE001
        return {"state": "UNKNOWN", "exit_code": "", "elapsed": "", "node": "", "error": str(exc)}
    for line in out.strip().splitlines():
        parts = line.split("|")
        if parts and parts[0] == job_id:
            return {
                "state": parts[1],
                "exit_code": parts[2],
                "elapsed": parts[3],
                "node": parts[4],
                "end": parts[5] if len(parts) > 5 else "",
            }
    return {"state": "UNKNOWN", "exit_code": "", "elapsed": "", "node": ""}


def parse_slurm_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    out = {
        "batch_shape": "",
        "points": "",
        "first_loss": "",
        "train_count": "",
        "test_count": "",
        "final_epoch_seen": "",
        "has_nan": bool(re.search(r"\bnan\b", text, re.I)),
        "has_oom": bool(re.search(r"out of memory|CUDA out of memory", text, re.I)),
        "has_traceback": "Traceback" in text,
    }
    m = re.search(
        r"First train batch tensor shape before device move: \(([^)]+)\).*points=(\d+)",
        text,
    )
    if m:
        out["batch_shape"] = f"({m.group(1)})"
        out["points"] = m.group(2)
    m = re.search(r"First train batch loss: ([0-9.eE+-]+)", text)
    if m:
        out["first_loss"] = m.group(1)
    m = re.search(r"Dataset sizes: train=(\d+) test=(\d+)", text)
    if m:
        out["train_count"], out["test_count"] = m.group(1), m.group(2)
    epochs = re.findall(r"Epoch (\d+)/\d+", text)
    if epochs:
        out["final_epoch_seen"] = epochs[-1]
    return out


def short_a(method: str) -> str:
    return {
        "Original baseline 1024": "Original baseline",
        "Original + EAR 4096": "EAR",
        "Original + PDANS 4096": "PDANS",
        "Original + PU-Net 4096": "PU-Net",
        "Original + PU-GCN 4096": "PU-GCN",
        "Original + PU-EdgeFormer 4096": "PU-EdgeFormer",
    }.get(method, method)


def short_b(method: str) -> str:
    return {
        "Downsampled x4 baseline": "Downsampled baseline",
        "Downsampled x4 baseline 256": "Downsampled baseline",
        "Downsampled x4 + EAR": "EAR",
        "Downsampled x4 + EAR 1024": "EAR",
        "Downsampled x4 + PDANS": "PDANS",
        "Downsampled x4 + PDANS 1024": "PDANS",
        "Downsampled x4 + PU-Net": "PU-Net",
        "Downsampled x4 + PU-Net 1024": "PU-Net",
        "Downsampled x4 + PU-GCN": "PU-GCN",
        "Downsampled x4 + PU-GCN 1024": "PU-GCN",
        "Downsampled x4 + PU-EdgeFormer": "PU-EdgeFormer",
        "Downsampled x4 + PU-EdgeFormer 1024": "PU-EdgeFormer",
    }.get(method, method)


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(ext))
    plt.close(fig)


def build_audit_and_rows() -> tuple[list[dict], list[dict], list[dict]]:
    """Return (audit_rows, linea_rows, lineb_rows). Accuracies as fractions; deltas as pp."""
    orig_path = RESULTS / "lineA_original_baseline" / "metrics.json"
    ds_path = RESULTS / "lineB_downsampled_x4_baseline" / "metrics.json"
    orig = load_metrics(orig_path)
    ds = load_metrics(ds_path)

    edge_b_path = RESULTS / "lineB_downsampled_x4_up" / "pu_edgeformer" / "metrics.json"
    edge_a_path = RESULTS / "lineA_original_up" / "pu_edgeformer" / "metrics.json"
    edge_b = load_metrics(edge_b_path) if edge_b_path.is_file() else {}
    edge_a = load_metrics(edge_a_path) if edge_a_path.is_file() else {}

    log_b = PROJECT / "logs" / "pointnet2_full" / "lineB_pu_edgeformer_1024" / f"slurm_{JOB_B}.out"
    log_a = PROJECT / "logs" / "pointnet2_full" / "lineA_pu_edgeformer_4096" / f"slurm_{JOB_A}.out"
    parsed_b, parsed_a = parse_slurm_log(log_b), parse_slurm_log(log_a)
    sacct_b, sacct_a = sacct_info(JOB_B), sacct_info(JOB_A)

    def audit_one(line, job_id, sacct, parsed, expected_pts, metrics, metrics_path, config, input_path):
        loss = parsed.get("first_loss") or ""
        loss_ok = bool(loss) and math.isfinite(float(loss))
        pts_ok = parsed.get("points") == str(expected_pts)
        metrics_ok = bool(metrics) and metrics_path.is_file()
        completed = sacct.get("state") == "COMPLETED" and sacct.get("exit_code") == "0:0"
        status = "PASS" if (
            completed
            and metrics_ok
            and pts_ok
            and loss_ok
            and not parsed.get("has_nan")
            and not parsed.get("has_oom")
            and not parsed.get("has_traceback")
        ) else "FAIL"
        return {
            "line": line,
            "method": "pu_edgeformer",
            "job_id": job_id,
            "state": sacct.get("state", ""),
            "exit_code": sacct.get("exit_code", ""),
            "node": sacct.get("node", ""),
            "runtime": sacct.get("elapsed", ""),
            "config_path": config,
            "input_path": input_path,
            "expected_points": expected_pts,
            "actual_batch_shape": parsed.get("batch_shape", ""),
            "actual_points": parsed.get("points", ""),
            "first_loss": loss,
            "final_epoch": metrics.get("final_epoch", parsed.get("final_epoch_seen", "")),
            "best_epoch": metrics.get("best_epoch", ""),
            "best_overall_accuracy": metrics.get("overall_accuracy", ""),
            "best_class_accuracy": metrics.get("class_accuracy", ""),
            "final_test_overall_accuracy": metrics.get("final_test_overall_accuracy", ""),
            "final_test_class_accuracy": metrics.get("final_test_class_accuracy", ""),
            "metrics_json_path": str(metrics_path),
            "log_path": str(
                PROJECT / "logs" / "pointnet2_full" /
                (f"lineB_pu_edgeformer_1024/slurm_{JOB_B}.out" if line == "B"
                 else f"lineA_pu_edgeformer_4096/slurm_{JOB_A}.out")
            ),
            "no_nan_loss": not parsed.get("has_nan") and loss_ok,
            "no_oom": not parsed.get("has_oom"),
            "no_cuda_error": not parsed.get("has_traceback"),
            "status": status,
        }

    audit = [
        audit_one(
            "B", JOB_B, sacct_b, parsed_b, 1024, edge_b, edge_b_path,
            "configs/pointnet2_x4_two_line/lineB_downsampled_x4_up_pu_edgeformer_1024.yaml",
            "pointnet2_inputs/lineB_downsampled_x4_up/pu_edgeformer",
        ),
        audit_one(
            "A", JOB_A, sacct_a, parsed_a, 4096, edge_a, edge_a_path,
            "configs/pointnet2_x4_two_line/lineA_original_up_pu_edgeformer_4096.yaml",
            "pointnet2_inputs/lineA_original_up/pu_edgeformer",
        ),
    ]

    # Build Line A rows from existing summary + EdgeFormer
    linea_src = read_csv(REPORTS / "modelnet40_pointnet2_lineA_classification_summary.csv")
    linea = []
    for r in linea_src:
        best = float(r["best_overall_accuracy"])
        final = float(r["final_test_overall_accuracy"])
        linea.append(
            {
                "line": "A",
                "method": r["method"],
                "point_count": int(r["point_count"]),
                "baseline_reference": "Original baseline 1024",
                "final_test_overall_accuracy": final,
                "final_test_class_accuracy": float(r["final_test_class_accuracy"]),
                "best_overall_accuracy": best,
                "best_class_accuracy": float(r["best_class_accuracy"]),
                "delta_accuracy_vs_original_baseline_pp": to_pp(best - float(orig["overall_accuracy"])),
                "delta_final_vs_original_baseline_pp": to_pp(final - float(orig["final_test_overall_accuracy"])),
                "status": "PASS",
                "metrics_json_path": str(LINEA_METRICS_PATHS.get(r["method"], "")),
            }
        )
    # drop any pre-existing EdgeFormer rows if summary was partially updated
    linea = [r for r in linea if "EdgeFormer" not in r["method"]]
    # EdgeFormer Line A
    linea.append(
        {
            "line": "A",
            "method": "Original + PU-EdgeFormer 4096",
            "point_count": 4096,
            "baseline_reference": "Original baseline 1024",
            "final_test_overall_accuracy": float(edge_a["final_test_overall_accuracy"]),
            "final_test_class_accuracy": float(edge_a["final_test_class_accuracy"]),
            "best_overall_accuracy": float(edge_a["overall_accuracy"]),
            "best_class_accuracy": float(edge_a["class_accuracy"]),
            "delta_accuracy_vs_original_baseline_pp": to_pp(
                float(edge_a["overall_accuracy"]) - float(orig["overall_accuracy"])
            ),
            "delta_final_vs_original_baseline_pp": to_pp(
                float(edge_a["final_test_overall_accuracy"]) - float(orig["final_test_overall_accuracy"])
            ),
            "status": "PASS",
            "metrics_json_path": str(edge_a_path),
        }
    )

    lineb_src = read_csv(REPORTS / "modelnet40_pointnet2_lineB_classification_summary.csv")
    # Keep non-edgeformer rows; recompute gaps vs original
    lineb = []
    for r in lineb_src:
        if "edgeformer" in r["method"].lower():
            continue
        method = r["method"]
        if method == "Downsampled x4 baseline":
            display = "Downsampled x4 baseline 256"
        elif not method.endswith("1024") and method.startswith("Downsampled x4 +"):
            display = f"{method} 1024"
        else:
            display = method
        best = float(r["best_overall_accuracy"])
        final = float(r["final_test_overall_accuracy"])
        lineb.append(
            {
                "line": "B",
                "method": display,
                "point_count": int(r["point_count"]),
                "baseline_reference_primary": "Downsampled x4 baseline 256",
                "baseline_reference_secondary": "Original baseline 1024",
                "final_test_overall_accuracy": final,
                "final_test_class_accuracy": float(r["final_test_class_accuracy"]),
                "best_overall_accuracy": best,
                "best_class_accuracy": float(r["best_class_accuracy"]),
                "delta_accuracy_vs_downsampled_baseline_pp": to_pp(best - float(ds["overall_accuracy"])),
                "delta_final_vs_downsampled_baseline_pp": to_pp(final - float(ds["final_test_overall_accuracy"])),
                "gap_accuracy_vs_original_baseline_pp": to_pp(best - float(orig["overall_accuracy"])),
                "gap_final_vs_original_baseline_pp": to_pp(final - float(orig["final_test_overall_accuracy"])),
                "status": "PASS",
                "metrics_json_path": r.get("metrics_json_path")
                or str(LINEB_METRICS_PATHS.get(r["method"], "")),
            }
        )
    lineb.append(
        {
            "line": "B",
            "method": "Downsampled x4 + PU-EdgeFormer 1024",
            "point_count": 1024,
            "baseline_reference_primary": "Downsampled x4 baseline 256",
            "baseline_reference_secondary": "Original baseline 1024",
            "final_test_overall_accuracy": float(edge_b["final_test_overall_accuracy"]),
            "final_test_class_accuracy": float(edge_b["final_test_class_accuracy"]),
            "best_overall_accuracy": float(edge_b["overall_accuracy"]),
            "best_class_accuracy": float(edge_b["class_accuracy"]),
            "delta_accuracy_vs_downsampled_baseline_pp": to_pp(
                float(edge_b["overall_accuracy"]) - float(ds["overall_accuracy"])
            ),
            "delta_final_vs_downsampled_baseline_pp": to_pp(
                float(edge_b["final_test_overall_accuracy"]) - float(ds["final_test_overall_accuracy"])
            ),
            "gap_accuracy_vs_original_baseline_pp": to_pp(
                float(edge_b["overall_accuracy"]) - float(orig["overall_accuracy"])
            ),
            "gap_final_vs_original_baseline_pp": to_pp(
                float(edge_b["final_test_overall_accuracy"]) - float(orig["final_test_overall_accuracy"])
            ),
            "status": "PASS",
            "metrics_json_path": str(edge_b_path),
        }
    )
    return audit, linea, lineb


def update_line_summaries(linea: list[dict], lineb: list[dict]) -> None:
    """Append EdgeFormer into existing lineA/lineB CSVs with dual comparison columns."""
    # Line A summary
    a_rows = []
    for r in linea:
        a_rows.append(
            {
                "line": "A",
                "method": r["method"],
                "point_count": r["point_count"],
                "final_test_overall_accuracy": r["final_test_overall_accuracy"],
                "final_test_class_accuracy": r["final_test_class_accuracy"],
                "best_overall_accuracy": r["best_overall_accuracy"],
                "best_class_accuracy": r["best_class_accuracy"],
                "delta_final_overall_vs_original_baseline": r["delta_final_vs_original_baseline_pp"] / 100.0,
                "delta_best_overall_vs_original_baseline": r["delta_accuracy_vs_original_baseline_pp"] / 100.0,
            }
        )
    write_csv(REPORTS / "modelnet40_pointnet2_lineA_classification_summary.csv", a_rows)

    # Line B with primary + secondary
    b_rows = []
    for r in lineb:
        method = r["method"]
        if method.endswith(" 1024"):
            method_csv = method[: -len(" 1024")]
        elif method.endswith(" 256"):
            method_csv = method[: -len(" 256")]
        else:
            method_csv = method
        b_rows.append(
            {
                "line": "B",
                "method": method_csv,
                "point_count": r["point_count"],
                "final_test_overall_accuracy": r["final_test_overall_accuracy"],
                "final_test_class_accuracy": r["final_test_class_accuracy"],
                "best_overall_accuracy": r["best_overall_accuracy"],
                "best_class_accuracy": r["best_class_accuracy"],
                "delta_final_overall_vs_downsampled_baseline": r["delta_final_vs_downsampled_baseline_pp"] / 100.0,
                "delta_final_class_vs_downsampled_baseline": "",
                "delta_best_overall_vs_downsampled_baseline": r["delta_accuracy_vs_downsampled_baseline_pp"] / 100.0,
                "delta_best_class_vs_downsampled_baseline": "",
                "gap_final_overall_vs_original_baseline": r["gap_final_vs_original_baseline_pp"] / 100.0,
                "gap_final_class_vs_original_baseline": "",
                "gap_best_overall_vs_original_baseline": r["gap_accuracy_vs_original_baseline_pp"] / 100.0,
                "gap_best_class_vs_original_baseline": "",
                "job_id": "",
                "status": "PASS",
                "metrics_json_path": r["metrics_json_path"],
            }
        )
    write_csv(REPORTS / "modelnet40_pointnet2_lineB_classification_summary.csv", b_rows)


def write_reports(audit: list[dict], linea: list[dict], lineb: list[dict]) -> None:
    # Audit
    write_csv(REPORTS / "modelnet40_pointnet2_pu_edgeformer_full_training_audit_20260716.csv", audit)
    all_pass = all(r["status"] == "PASS" for r in audit)
    md = [
        "# ModelNet40 PointNet++ PU-EdgeFormer Full Training Audit",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Job IDs: Line B=`{JOB_B}`, Line A=`{JOB_A}`",
        f"- Overall: **{'PASS' if all_pass else 'FAIL'}**",
        "",
        "## Safety",
        "",
        "- POINTNET_FULL_TRAINING_STARTED=YES",
        "- DETECTOR_EVAL_STARTED=NO",
        "- KITTI_AP_EVAL_STARTED=NO",
        "",
        "## Summary",
        "",
        "| Line | job | state | exit | runtime | node | batch shape | best overall | status |",
        "|---|---|---|---|---|---|---|---:|---|",
    ]
    for r in audit:
        best = r["best_overall_accuracy"]
        best_s = f"{float(best)*100:.2f}%" if best != "" else "—"
        md.append(
            f"| {r['line']} | {r['job_id']} | {r['state']} | {r['exit_code']} | {r['runtime']} | "
            f"{r['node']} | `{r['actual_batch_shape']}` | {best_s} | **{r['status']}** |"
        )
    for r in audit:
        md.extend(
            [
                "",
                f"## Line {r['line']} detail",
                "",
                f"- Config: `{r['config_path']}`",
                f"- Input: `{r['input_path']}`",
                f"- Expected points: {r['expected_points']}; actual points: {r['actual_points']}",
                f"- Batch shape: `{r['actual_batch_shape']}`",
                f"- First loss: {r['first_loss']}",
                f"- Final epoch: {r['final_epoch']}; best epoch: {r['best_epoch']}",
                f"- Best overall / class: {r['best_overall_accuracy']} / {r['best_class_accuracy']}",
                f"- Final test overall / class: {r['final_test_overall_accuracy']} / {r['final_test_class_accuracy']}",
                f"- metrics.json: `{r['metrics_json_path']}`",
                f"- Log: `{r['log_path']}`",
                f"- no NaN / OOM / CUDA error: {r['no_nan_loss']} / {r['no_oom']} / {r['no_cuda_error']}",
                f"- Status: **{r['status']}**",
            ]
        )
    (REPORTS / "modelnet40_pointnet2_pu_edgeformer_full_training_audit_20260716.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )

    # Final classification report with EdgeFormer
    report_rows = []
    for r in linea:
        report_rows.append(
            {
                "line": "A",
                "method": r["method"],
                "point_count": r["point_count"],
                "baseline_reference": r["baseline_reference"],
                "final_test_overall_accuracy": r["final_test_overall_accuracy"],
                "final_test_class_accuracy": r["final_test_class_accuracy"],
                "best_overall_accuracy": r["best_overall_accuracy"],
                "best_class_accuracy": r["best_class_accuracy"],
                "delta_accuracy_vs_original_baseline_pp": r["delta_accuracy_vs_original_baseline_pp"],
                "delta_final_vs_original_baseline_pp": r["delta_final_vs_original_baseline_pp"],
                "delta_accuracy_vs_downsampled_baseline_pp": "",
                "gap_accuracy_vs_original_baseline_pp": "",
                "status": r["status"],
                "metrics_json_path": r["metrics_json_path"],
            }
        )
    for r in lineb:
        report_rows.append(
            {
                "line": "B",
                "method": r["method"],
                "point_count": r["point_count"],
                "baseline_reference": r["baseline_reference_primary"],
                "final_test_overall_accuracy": r["final_test_overall_accuracy"],
                "final_test_class_accuracy": r["final_test_class_accuracy"],
                "best_overall_accuracy": r["best_overall_accuracy"],
                "best_class_accuracy": r["best_class_accuracy"],
                "delta_accuracy_vs_original_baseline_pp": "",
                "delta_final_vs_original_baseline_pp": "",
                "delta_accuracy_vs_downsampled_baseline_pp": r["delta_accuracy_vs_downsampled_baseline_pp"],
                "gap_accuracy_vs_original_baseline_pp": r["gap_accuracy_vs_original_baseline_pp"],
                "status": r["status"],
                "metrics_json_path": r["metrics_json_path"],
            }
        )
    write_csv(
        REPORTS / "modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.csv",
        report_rows,
    )

    def a_table() -> list[str]:
        lines = [
            "## Line A",
            "",
            "| Method | Points | Best Overall | Δ vs Original (pp) | Final Overall | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for r in linea:
            lines.append(
                f"| {r['method']} | {r['point_count']} | {pct(r['best_overall_accuracy'])} | "
                f"{pp_str(r['delta_accuracy_vs_original_baseline_pp'])} | "
                f"{pct(r['final_test_overall_accuracy'])} | {r['status']} |"
            )
        return lines

    def b_table() -> list[str]:
        lines = [
            "## Line B",
            "",
            "| Method | Points | Best Overall | Δ vs Downsampled (pp, primary) | gap vs Original (pp, secondary) | Final Overall | Status |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in lineb:
            lines.append(
                f"| {r['method']} | {r['point_count']} | {pct(r['best_overall_accuracy'])} | "
                f"{pp_str(r['delta_accuracy_vs_downsampled_baseline_pp'])} | "
                f"{pp_str(r['gap_accuracy_vs_original_baseline_pp'])} | "
                f"{pct(r['final_test_overall_accuracy'])} | {r['status']} |"
            )
        return lines

    report_md = [
        "# ModelNet40 PointNet++ Final Classification Report (with PU-EdgeFormer)",
        "",
        f"- Generated: `{utc_now()}`",
        "- Delta units: **percentage points (pp)**",
        "- Line A primary reference: Original baseline 1024",
        "- Line B primary reference: Downsampled ×4 baseline 256",
        "- Line B secondary gap: Original baseline 1024",
        "- Geometry deltas remain vs Original baseline (separate from classification)",
        "- See `reports/modelnet40_lineB_comparison_logic_geometry_vs_classification.md`",
        "",
    ]
    report_md.extend(a_table())
    report_md.append("")
    report_md.extend(b_table())
    report_md.append("")
    (REPORTS / "modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.md").write_text(
        "\n".join(report_md) + "\n", encoding="utf-8"
    )

    # Two-line summary
    two = []
    for r in linea:
        two.append(
            {
                "line": "A",
                "method": r["method"],
                "point_count": r["point_count"],
                "final_test_overall_accuracy": r["final_test_overall_accuracy"],
                "final_test_class_accuracy": r["final_test_class_accuracy"],
                "best_overall_accuracy": r["best_overall_accuracy"],
                "best_class_accuracy": r["best_class_accuracy"],
                "delta_accuracy_vs_original_baseline_pp": r["delta_accuracy_vs_original_baseline_pp"],
                "delta_accuracy_vs_downsampled_baseline_pp": "",
                "gap_accuracy_vs_original_baseline_pp": "",
            }
        )
    for r in lineb:
        two.append(
            {
                "line": "B",
                "method": r["method"],
                "point_count": r["point_count"],
                "final_test_overall_accuracy": r["final_test_overall_accuracy"],
                "final_test_class_accuracy": r["final_test_class_accuracy"],
                "best_overall_accuracy": r["best_overall_accuracy"],
                "best_class_accuracy": r["best_class_accuracy"],
                "delta_accuracy_vs_original_baseline_pp": "",
                "delta_accuracy_vs_downsampled_baseline_pp": r["delta_accuracy_vs_downsampled_baseline_pp"],
                "gap_accuracy_vs_original_baseline_pp": r["gap_accuracy_vs_original_baseline_pp"],
            }
        )
    write_csv(
        REPORTS / "modelnet40_pointnet2_final_two_line_classification_summary_with_pu_edgeformer.csv",
        two,
    )
    two_md = [
        "# Two-Line Classification Summary (with PU-EdgeFormer)",
        "",
        f"- Generated: `{utc_now()}`",
        "- Units for delta/gap columns: pp",
        "",
        "| Line | Method | Points | Best Overall | Primary Δ (pp) | Secondary gap (pp) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in two:
        if r["line"] == "A":
            primary = pp_str(float(r["delta_accuracy_vs_original_baseline_pp"]))
            secondary = "—"
        else:
            primary = pp_str(float(r["delta_accuracy_vs_downsampled_baseline_pp"]))
            secondary = pp_str(float(r["gap_accuracy_vs_original_baseline_pp"]))
        two_md.append(
            f"| {r['line']} | {r['method']} | {r['point_count']} | {pct(float(r['best_overall_accuracy']))} | "
            f"{primary} | {secondary} |"
        )
    (REPORTS / "modelnet40_pointnet2_final_two_line_classification_summary_with_pu_edgeformer.md").write_text(
        "\n".join(two_md) + "\n", encoding="utf-8"
    )

    # Interpretation
    a_ef = next(r for r in linea if "EdgeFormer" in r["method"])
    b_ef = next(r for r in lineb if "EdgeFormer" in r["method"])
    a_best = max(linea, key=lambda r: r["best_overall_accuracy"])
    b_up = [r for r in lineb if "baseline" not in r["method"].lower()]
    b_best_up = max(b_up, key=lambda r: r["best_overall_accuracy"])
    interp = [
        "# PointNet++ Final Classification Interpretation (with PU-EdgeFormer)",
        "",
        f"- Generated: `{utc_now()}`",
        "",
        "## Line A",
        "",
        f"- Strongest method: **{a_best['method']}** at {pct(a_best['best_overall_accuracy'])}.",
        f"- PU-EdgeFormer 4096: {pct(a_ef['best_overall_accuracy'])} "
        f"({pp_str(a_ef['delta_accuracy_vs_original_baseline_pp'])} vs Original baseline).",
        "- Line A deltas use Original baseline 1024 only.",
        "",
        "## Line B",
        "",
        f"- Best upsampling method on primary Δ: **{b_best_up['method']}** "
        f"({pct(b_best_up['best_overall_accuracy'])}, "
        f"{pp_str(b_best_up['delta_accuracy_vs_downsampled_baseline_pp'])} vs Downsampled baseline).",
        f"- PU-EdgeFormer 1024: {pct(b_ef['best_overall_accuracy'])}; "
        f"primary Δ vs Downsampled = {pp_str(b_ef['delta_accuracy_vs_downsampled_baseline_pp'])}; "
        f"secondary gap vs Original = {pp_str(b_ef['gap_accuracy_vs_original_baseline_pp'])}.",
        "- Primary ranking must use Downsampled ×4 baseline 256.",
        "- Secondary gap answers recovery toward Original baseline 1024.",
        "",
        "## Comparison logic reminder",
        "",
        "- Geometry Δ: vs Original baseline 1024.",
        "- Classification primary Δ (Line B): vs Downsampled ×4 baseline 256.",
        "- Classification secondary gap (Line B): vs Original baseline 1024.",
        "- Do not mix these references.",
        "",
        "## Safety",
        "",
        "- DETECTOR_EVAL_STARTED=NO",
        "- KITTI_AP_EVAL_STARTED=NO",
        "",
    ]
    (REPORTS / "modelnet40_pointnet2_final_classification_interpretation_with_pu_edgeformer.md").write_text(
        "\n".join(interp), encoding="utf-8"
    )


def plot_absolute(data: list[dict], baseline_idx: int, title: str, baseline_label: str, out: Path, short_fn) -> None:
    labels = [short_fn(d["method"]) for d in data]
    values = [d["best_overall_accuracy"] * 100 for d in data]
    colors = [METHOD_COLORS.get(l, "#888") for l in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(values[baseline_idx], color="#333", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Best Overall Accuracy (%)")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.25, f"{val:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(min(values) - 2, max(values) + 2.5)
    ax.legend(loc="lower right")
    save_fig(fig, out)


def plot_delta(data: list[dict], delta_key: str, title: str, ylabel: str, out: Path, short_fn) -> None:
    rows = [d for d in data if "baseline" not in d["method"].lower()]
    labels = [short_fn(d["method"]) for d in rows]
    deltas = [float(d[delta_key]) for d in rows]
    colors = ["#2CA02C" if d >= 0 else "#D62728" for d in deltas]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#333", linewidth=1.0, linestyle="--", label="baseline (y=0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, val in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.08 if val >= 0 else -0.08),
            pp_str(val),
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=8,
        )
    pad = max(abs(min(deltas)), abs(max(deltas))) * 0.35 + 0.3
    ax.set_ylim(min(deltas) - pad, max(deltas) + pad)
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, out)


def generate_figures(linea: list[dict], lineb: list[dict]) -> None:
    plot_absolute(
        linea, 0,
        "PointNet++ Line A Best Overall (with PU-EdgeFormer)",
        "Original baseline",
        ACC_ABS / "lineA_best_overall_accuracy_absolute_with_pu_edgeformer",
        short_a,
    )
    plot_absolute(
        lineb, 0,
        "PointNet++ Line B Best Overall (with PU-EdgeFormer)",
        "Downsampled baseline",
        ACC_ABS / "lineB_best_overall_accuracy_absolute_with_pu_edgeformer",
        short_b,
    )
    plot_delta(
        linea,
        "delta_accuracy_vs_original_baseline_pp",
        "Line A Δ Best Overall vs Original Baseline (methods only)",
        "Δ Best Overall vs Original baseline (pp)",
        ACC_DELTA / "lineA_delta_best_overall_vs_original_baseline_methods_only_with_pu_edgeformer",
        short_a,
    )
    plot_delta(
        lineb,
        "delta_accuracy_vs_downsampled_baseline_pp",
        "Line B Δ Best Overall vs Downsampled Baseline (methods only)",
        "Δ Best Overall vs Downsampled ×4 baseline (pp)",
        ACC_DELTA / "lineB_delta_best_overall_vs_downsampled_baseline_methods_only_with_pu_edgeformer",
        short_b,
    )
    plot_delta(
        lineb,
        "gap_accuracy_vs_original_baseline_pp",
        "Line B gap Best Overall vs Original Baseline (methods only, secondary)",
        "gap Best Overall vs Original baseline (pp)",
        ACC_DELTA / "lineB_gap_best_overall_vs_original_baseline_methods_only_with_pu_edgeformer",
        short_b,
    )

    # Geometry vs classification scatter
    geom_csv = REPORTS / "modelnet40_quality_metrics_thesis_table_with_pu_edgeformer.csv"
    geom_rows = {r["group"]: r for r in read_csv(geom_csv)}
    lineb_methods = {
        "EAR": "Downsampled x4 + EAR",
        "PDANS": "Downsampled x4 + PDANS",
        "PU-Net": "Downsampled x4 + PU-Net",
        "PU-GCN": "Downsampled x4 + PU-GCN",
        "PU-EdgeFormer": "Downsampled x4 + PU-EdgeFormer",
    }
    cls_by_short = {short_b(r["method"]): r for r in lineb if "baseline" not in r["method"].lower()}
    for metric_key, metric_label, stem in [
        ("delta_CD_vs_original", "CD", "lineB_delta_cd_vs_delta_accuracy_with_pu_edgeformer"),
        ("delta_HD_vs_original", "HD", "lineB_delta_hd_vs_delta_accuracy_with_pu_edgeformer"),
        ("delta_NUC_vs_original", "NUC", "lineB_delta_nuc_vs_delta_accuracy_with_pu_edgeformer"),
        ("delta_P2F_vs_original", "exact P2F", "lineB_delta_p2f_vs_delta_accuracy_with_pu_edgeformer"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for label, group in lineb_methods.items():
            g = geom_rows[group]
            c = cls_by_short[label]
            x = float(g[metric_key])
            y = float(c["delta_accuracy_vs_downsampled_baseline_pp"])
            ax.scatter(x, y, c=METHOD_COLORS.get(label, "#888"), s=110, edgecolors="black", linewidths=0.5)
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
        ax.axhline(0, color="#888", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="#888", linestyle="--", linewidth=0.8)
        ax.set_xlabel(f"Δ {metric_label} vs Original baseline")
        ax.set_ylabel("Δ Best Overall Accuracy vs Downsampled baseline (pp)")
        ax.set_title(f"Line B: Δ {metric_label} vs Δ Classification (with PU-EdgeFormer)")
        fig.text(
            0.01, 0.01,
            "Geometry Δ uses Original baseline; classification Δ uses Downsampled baseline.",
            fontsize=8, style="italic",
        )
        save_fig(fig, FINAL_COMBINED / stem)


def update_indexes() -> None:
    note = (
        "\n\n## PU-EdgeFormer PointNet++ classification update\n\n"
        "- PU-EdgeFormer PointNet++ classification has been added.\n"
        "- Line B classification uses Downsampled baseline as the primary reference.\n"
        "- Line B also reports gap to Original baseline.\n"
        "- Geometry delta remains relative to Original baseline.\n"
        "- Figures: `figures/modelnet40/pointnet2_final_with_pu_edgeformer/`\n"
        "- Reports: `reports/modelnet40_pointnet2_final_*_with_pu_edgeformer.*`\n"
        "- Audit: `reports/modelnet40_pointnet2_pu_edgeformer_full_training_audit_20260716.md`\n"
        "- DETECTOR_EVAL_STARTED=NO; KITTI_AP_EVAL_STARTED=NO\n"
    )
    for path in [
        PROJECT / "figures" / "modelnet40" / "figure_index.md",
        REPORTS / "modelnet40_thesis_figure_captions.md",
        REPORTS / "modelnet40_visualization_summary.md",
        REPORTS / "modelnet40_final_thesis_visualization_and_report_index.md",
    ]:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        marker = "## PU-EdgeFormer PointNet++ classification update"
        if marker in text:
            # replace existing block
            pre = text.split(marker)[0].rstrip()
            text = pre + note
        else:
            text = text.rstrip() + note
        path.write_text(text + "\n", encoding="utf-8")

    # Append figure table entries to figure_index
    fig_index = PROJECT / "figures" / "modelnet40" / "figure_index.md"
    extra = """
## PointNet++ final (with PU-EdgeFormer)

| Figure stem | Source | Description | Suggested thesis location |
|---|---|---|---|
| `pointnet2_final_with_pu_edgeformer/accuracy_absolute/lineA_best_overall_accuracy_absolute_with_pu_edgeformer` | final classification with EdgeFormer | Line A absolute best overall including PU-EdgeFormer. | Results — Line A classification |
| `pointnet2_final_with_pu_edgeformer/accuracy_absolute/lineB_best_overall_accuracy_absolute_with_pu_edgeformer` | final classification with EdgeFormer | Line B absolute best overall including PU-EdgeFormer. | Results — Line B classification |
| `pointnet2_final_with_pu_edgeformer/accuracy_delta_methods_only/lineA_delta_best_overall_vs_original_baseline_methods_only_with_pu_edgeformer` | final classification with EdgeFormer | Line A Δ vs Original baseline (methods only). | Results — Line A classification |
| `pointnet2_final_with_pu_edgeformer/accuracy_delta_methods_only/lineB_delta_best_overall_vs_downsampled_baseline_methods_only_with_pu_edgeformer` | final classification with EdgeFormer | Line B primary Δ vs Downsampled baseline (methods only). | Results — Line B classification |
| `pointnet2_final_with_pu_edgeformer/accuracy_delta_methods_only/lineB_gap_best_overall_vs_original_baseline_methods_only_with_pu_edgeformer` | final classification with EdgeFormer | Line B secondary gap vs Original baseline (methods only). | Results — Line B classification |
| `pointnet2_final_with_pu_edgeformer/final_combined/lineB_delta_*_vs_delta_accuracy_with_pu_edgeformer` | geometry + classification with EdgeFormer | Geometry Δ (vs Original) vs classification Δ (vs Downsampled). | Discussion — Geometry vs classification |
"""
    text = fig_index.read_text(encoding="utf-8")
    if "pointnet2_final_with_pu_edgeformer/accuracy_absolute" not in text:
        fig_index.write_text(text.rstrip() + "\n" + extra + "\n", encoding="utf-8")


def main() -> int:
    # Require metrics
    for p in [
        RESULTS / "lineB_downsampled_x4_up" / "pu_edgeformer" / "metrics.json",
        RESULTS / "lineA_original_up" / "pu_edgeformer" / "metrics.json",
    ]:
        if not p.is_file():
            print(f"MISSING metrics: {p}")
            return 2

    audit, linea, lineb = build_audit_and_rows()
    if any(r["status"] != "PASS" for r in audit):
        write_csv(REPORTS / "modelnet40_pointnet2_pu_edgeformer_full_training_audit_20260716.csv", audit)
        print("AUDIT FAIL — not updating tables/figures")
        for r in audit:
            print(r["line"], r["status"], r["state"], r["exit_code"])
        return 1

    update_line_summaries(linea, lineb)
    write_reports(audit, linea, lineb)
    generate_figures(linea, lineb)
    update_indexes()

    b_ef = next(r for r in lineb if "EdgeFormer" in r["method"])
    a_ef = next(r for r in linea if "EdgeFormer" in r["method"])
    print("OVERALL_PASS")
    print("LineB best", b_ef["best_overall_accuracy"], "delta_ds_pp", b_ef["delta_accuracy_vs_downsampled_baseline_pp"], "gap_orig_pp", b_ef["gap_accuracy_vs_original_baseline_pp"])
    print("LineA best", a_ef["best_overall_accuracy"], "delta_orig_pp", a_ef["delta_accuracy_vs_original_baseline_pp"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
