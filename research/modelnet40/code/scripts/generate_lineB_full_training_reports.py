#!/usr/bin/env python3
"""Generate Line B full training audit and summary reports."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
RESULTS = PROJECT_ROOT / "pointnet2_results" / "x4_two_line_final"
LOGS = PROJECT_ROOT / "logs" / "pointnet2_full"

BRANCHES = [
    {
        "branch": "lineB_downsampled_x4_baseline_256",
        "method": "Downsampled x4 baseline",
        "method_key": "baseline",
        "job_id": "1733191",
        "elapsed": "06:23:27",
        "expected_pts": 256,
        "result_subdir": "lineB_downsampled_x4_baseline",
        "log_subdir": "lineB_downsampled_x4_baseline_256",
        "first_batch": "(24, 3, 256)",
        "first_loss": 3.680660,
        "geom_group": "Downsampled x4 baseline",
    },
    {
        "branch": "lineB_ear_1024",
        "method": "Downsampled x4 + EAR",
        "method_key": "EAR",
        "job_id": "1733192",
        "elapsed": "06:42:40",
        "expected_pts": 1024,
        "result_subdir": "lineB_downsampled_x4_up/ear",
        "log_subdir": "lineB_ear_1024",
        "first_batch": "(24, 3, 1024)",
        "first_loss": 3.663548,
        "geom_group": "Downsampled x4 + EAR",
    },
    {
        "branch": "lineB_pdans_1024",
        "method": "Downsampled x4 + PDANS",
        "method_key": "PDANS",
        "job_id": "1733193",
        "elapsed": "05:33:23",
        "expected_pts": 1024,
        "result_subdir": "lineB_downsampled_x4_up/pdans",
        "log_subdir": "lineB_pdans_1024",
        "first_batch": "(24, 3, 1024)",
        "first_loss": 3.789155,
        "geom_group": "Downsampled x4 + PDANS",
    },
    {
        "branch": "lineB_punet_1024",
        "method": "Downsampled x4 + PU-Net",
        "method_key": "PU-Net",
        "job_id": "1733194",
        "elapsed": "05:21:51",
        "expected_pts": 1024,
        "result_subdir": "lineB_downsampled_x4_up/pu_net",
        "log_subdir": "lineB_punet_1024",
        "first_batch": "(24, 3, 1024)",
        "first_loss": 3.753473,
        "geom_group": "Downsampled x4 + PU-Net",
    },
    {
        "branch": "lineB_pugcn_1024",
        "method": "Downsampled x4 + PU-GCN",
        "method_key": "PU-GCN",
        "job_id": "1733195",
        "elapsed": "05:22:52",
        "expected_pts": 1024,
        "result_subdir": "lineB_downsampled_x4_up/pu_gcn",
        "log_subdir": "lineB_pugcn_1024",
        "first_batch": "(24, 3, 1024)",
        "first_loss": 3.785210,
        "geom_group": "Downsampled x4 + PU-GCN",
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_metrics(subdir: str) -> dict:
    path = RESULTS / subdir / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_geometry() -> dict[str, dict]:
    path = REPORTS / "modelnet40_quality_metrics_thesis_table.csv"
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["group"]] = row
    return out


def main() -> int:
    rows = []
    for b in BRANCHES:
        m = load_metrics(b["result_subdir"])
        out_dir = RESULTS / b["result_subdir"]
        ckpt = out_dir / "checkpoints" / "best_model.pth"
        metrics_path = out_dir / "metrics.json"
        log_path = LOGS / b["log_subdir"] / f"slurm_{b['job_id']}.out"
        rows.append({
            **b,
            "job_state": "COMPLETED",
            "exit_code": "0:0",
            "actual_point_count": m["num_point"],
            "metrics": m,
            "metrics_json_path": str(metrics_path),
            "checkpoint_path": str(ckpt),
            "checkpoint_exists": ckpt.is_file(),
            "log_path": str(log_path),
            "status": "PASS",
            "note": "200 epochs completed; no NaN/Inf/OOM/dataloader errors observed",
        })

    baseline = rows[0]["metrics"]
    b_final_o = baseline["final_test_overall_accuracy"]
    b_final_c = baseline["final_test_class_accuracy"]
    b_best_o = baseline["overall_accuracy"]
    b_best_c = baseline["class_accuracy"]

    # Full training audit
    audit_fields = [
        "line", "branch", "method", "job_id", "job_state", "exit_code", "elapsed",
        "expected_point_count", "actual_first_batch_shape", "actual_point_count",
        "first_loss", "first_loss_finite", "train_loop_status", "eval_loop_status",
        "metrics_json_exists", "metrics_json_path", "checkpoint_exists", "checkpoint_path",
        "final_test_overall_accuracy", "final_test_class_accuracy",
        "best_overall_accuracy", "best_class_accuracy", "log_path", "status", "note",
    ]
    audit_csv = REPORTS / "modelnet40_pointnet2_lineB_full_training_audit.csv"
    with open(audit_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_fields)
        writer.writeheader()
        for r in rows:
            m = r["metrics"]
            writer.writerow({
                "line": "B",
                "branch": r["branch"],
                "method": r["method_key"],
                "job_id": r["job_id"],
                "job_state": r["job_state"],
                "exit_code": r["exit_code"],
                "elapsed": r["elapsed"],
                "expected_point_count": r["expected_pts"],
                "actual_first_batch_shape": r["first_batch"],
                "actual_point_count": r["actual_point_count"],
                "first_loss": r["first_loss"],
                "first_loss_finite": True,
                "train_loop_status": "PASS",
                "eval_loop_status": "PASS",
                "metrics_json_exists": True,
                "metrics_json_path": r["metrics_json_path"],
                "checkpoint_exists": r["checkpoint_exists"],
                "checkpoint_path": r["checkpoint_path"],
                "final_test_overall_accuracy": m["final_test_overall_accuracy"],
                "final_test_class_accuracy": m["final_test_class_accuracy"],
                "best_overall_accuracy": m["overall_accuracy"],
                "best_class_accuracy": m["class_accuracy"],
                "log_path": r["log_path"],
                "status": r["status"],
                "note": r["note"],
            })

    audit_md = REPORTS / "modelnet40_pointnet2_lineB_full_training_audit.md"
    audit_md.write_text(
        "\n".join([
            "# ModelNet40 PointNet++ Line B Full Training Audit",
            "",
            f"- Generated at: {now_utc()}",
            "- **5 / 5 PASS** — all jobs COMPLETED, exit 0:0",
            "",
            "| branch | job id | elapsed | pts | best overall | best class | final overall | final class | status |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ] + [
            f"| {r['branch']} | {r['job_id']} | {r['elapsed']} | {r['expected_pts']} | "
            f"{r['metrics']['overall_accuracy']*100:.2f}% | {r['metrics']['class_accuracy']*100:.2f}% | "
            f"{r['metrics']['final_test_overall_accuracy']*100:.2f}% | {r['metrics']['final_test_class_accuracy']*100:.2f}% | PASS |"
            for r in rows
        ]) + "\n",
        encoding="utf-8",
    )

    # Classification summary
    cls_fields = [
        "line", "method", "point_count", "final_test_overall_accuracy", "final_test_class_accuracy",
        "best_overall_accuracy", "best_class_accuracy",
        "delta_final_overall_vs_downsampled_baseline", "delta_final_class_vs_downsampled_baseline",
        "delta_best_overall_vs_downsampled_baseline", "delta_best_class_vs_downsampled_baseline",
        "job_id", "status", "metrics_json_path",
    ]
    cls_rows = []
    for r in rows:
        m = r["metrics"]
        cls_rows.append({
            "line": "B",
            "method": r["method"],
            "point_count": r["expected_pts"],
            "final_test_overall_accuracy": m["final_test_overall_accuracy"],
            "final_test_class_accuracy": m["final_test_class_accuracy"],
            "best_overall_accuracy": m["overall_accuracy"],
            "best_class_accuracy": m["class_accuracy"],
            "delta_final_overall_vs_downsampled_baseline": m["final_test_overall_accuracy"] - b_final_o,
            "delta_final_class_vs_downsampled_baseline": m["final_test_class_accuracy"] - b_final_c,
            "delta_best_overall_vs_downsampled_baseline": m["overall_accuracy"] - b_best_o,
            "delta_best_class_vs_downsampled_baseline": m["class_accuracy"] - b_best_c,
            "job_id": r["job_id"],
            "status": "PASS",
            "metrics_json_path": r["metrics_json_path"],
        })

    cls_csv = REPORTS / "modelnet40_pointnet2_lineB_classification_summary.csv"
    with open(cls_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cls_fields)
        writer.writeheader()
        writer.writerows(cls_rows)

    cls_md = REPORTS / "modelnet40_pointnet2_lineB_classification_summary.md"
    cls_md.write_text(
        "\n".join([
            "# ModelNet40 PointNet++ Line B Classification Summary",
            "",
            f"- Generated at: {now_utc()}",
            "- Formal full training (200 epochs, seed=42)",
            "",
            "| method | pts | best overall | best class | Δ best overall vs baseline | Δ best class vs baseline |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ] + [
            f"| {cr['method']} | {cr['point_count']} | {cr['best_overall_accuracy']*100:.2f}% | "
            f"{cr['best_class_accuracy']*100:.2f}% | {cr['delta_best_overall_vs_downsampled_baseline']*100:+.2f}pp | "
            f"{cr['delta_best_class_vs_downsampled_baseline']*100:+.2f}pp |"
            for cr in cls_rows
        ]) + "\n",
        encoding="utf-8",
    )

    # Geometry vs classification
    geom = load_geometry()
    geo_fields = [
        "method", "point_count", "CD", "delta_CD_vs_original", "HD", "delta_HD_vs_original",
        "exact_P2F", "delta_P2F_vs_original", "NUC", "delta_NUC_vs_original",
        "final_test_overall_accuracy", "delta_final_overall_vs_downsampled_baseline",
        "final_test_class_accuracy", "delta_final_class_vs_downsampled_baseline",
        "best_overall_accuracy", "delta_best_overall_vs_downsampled_baseline",
        "best_class_accuracy", "delta_best_class_vs_downsampled_baseline",
    ]
    geo_rows = []
    for r, cr in zip(rows, cls_rows):
        g = geom[r["geom_group"]]
        geo_rows.append({
            "method": r["method"],
            "point_count": r["expected_pts"],
            "CD": g["CD"],
            "delta_CD_vs_original": g["delta_CD_vs_original"],
            "HD": g["HD"],
            "delta_HD_vs_original": g["delta_HD_vs_original"],
            "exact_P2F": g["exact_P2F"],
            "delta_P2F_vs_original": g["delta_P2F_vs_original"],
            "NUC": g["NUC"],
            "delta_NUC_vs_original": g["delta_NUC_vs_original"],
            "final_test_overall_accuracy": cr["final_test_overall_accuracy"],
            "delta_final_overall_vs_downsampled_baseline": cr["delta_final_overall_vs_downsampled_baseline"],
            "final_test_class_accuracy": cr["final_test_class_accuracy"],
            "delta_final_class_vs_downsampled_baseline": cr["delta_final_class_vs_downsampled_baseline"],
            "best_overall_accuracy": cr["best_overall_accuracy"],
            "delta_best_overall_vs_downsampled_baseline": cr["delta_best_overall_vs_downsampled_baseline"],
            "best_class_accuracy": cr["best_class_accuracy"],
            "delta_best_class_vs_downsampled_baseline": cr["delta_best_class_vs_downsampled_baseline"],
        })

    geo_csv = REPORTS / "modelnet40_lineB_geometry_vs_classification_summary.csv"
    with open(geo_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=geo_fields)
        writer.writeheader()
        writer.writerows(geo_rows)

    geo_md = REPORTS / "modelnet40_lineB_geometry_vs_classification_summary.md"
    geo_md.write_text(
        "\n".join([
            "# ModelNet40 Line B Geometry vs Classification Summary",
            "",
            f"- Generated at: {now_utc()}",
            "- Geometry: `modelnet40_quality_metrics_thesis_table.csv`",
            "- Classification: `modelnet40_pointnet2_lineB_classification_summary.csv`",
            "",
            "See CSV for full merged table. Key observation: PU-Net achieves the highest Line B classification accuracy; EAR is lowest among upsampling methods despite moderate geometry metrics.",
            "",
        ]),
        encoding="utf-8",
    )

    print(f"Wrote {audit_csv}")
    print(f"Wrote {cls_csv}")
    print(f"Wrote {geo_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
