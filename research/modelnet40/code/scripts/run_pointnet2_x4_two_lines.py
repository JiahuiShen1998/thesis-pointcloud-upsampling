#!/usr/bin/env python3
"""Launch or document PointNet++ classification for ×4 two-line protocol."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from modelnet40_x4_protocol import (
    MAIN_METHODS,
    PROJECT_ROOT,
    detect_original_point_count,
    pointnet2_input_paths,
    protocol_counts,
    reports_dir,
)

METHOD_LABELS = {"ear": "EAR", "pu_net": "PU-Net", "pu_gcn": "PU-GCN", "pdans": "PDANS"}


def load_existing_results() -> dict[str, dict]:
    """Map legacy completed results where protocol-compatible."""
    status_csv = PROJECT_ROOT / "reports" / "modelnet40_pointnet2_main_results_status.csv"
    out: dict[str, dict] = {}
    if not status_csv.is_file():
        return out
    with open(status_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[row["experiment"]] = row
    return out


def build_experiment_table(n: int) -> list[dict]:
    counts = protocol_counts(n)
    paths = pointnet2_input_paths()
    existing = load_existing_results()
    rows = []

    def add(line, branch, method, exp_key, data_root, num_point, note=""):
        legacy = existing.get(exp_key, {})
        rows.append({
            "line": line,
            "branch": branch,
            "method": method,
            "input_point_count": num_point,
            "upsampling_factor": 4 if branch == "upsampling" else 1,
            "data_root": str(data_root),
            "output_dir": str(PROJECT_ROOT / "pointnet2_results" / exp_key),
            "overall_accuracy": legacy.get("best_test_acc", ""),
            "class_avg_accuracy": "",
            "loss": "",
            "checkpoint": legacy.get("checkpoint", ""),
            "status": legacy.get("status", "pending"),
            "note": note,
        })

    add("A", "baseline", "original", "lineA_original_baseline",
        paths["lineA_baseline"], n, "Reuse original_baseline if completed")
    legacy_a = existing.get("original_baseline", {})
    if legacy_a:
        rows[0]["overall_accuracy"] = legacy_a.get("best_test_acc", "")
        rows[0]["checkpoint"] = legacy_a.get("checkpoint", "")
        rows[0]["status"] = legacy_a.get("status", "")

    for method in MAIN_METHODS:
        exp_a = f"lineA_original_up_{method}"
        add("A", "upsampling", METHOD_LABELS[method], exp_a,
            paths["lineA_up_root"] / method, counts["four_N"])
        legacy_key = f"original_{method.replace('pu_', 'pu')}_x4_pointnet2".replace("pu_net", "punet").replace("pu_gcn", "pugcn")
        legacy_key_map = {
            "ear": "original_ear_x4_pointnet2",
            "pu_net": "original_punet_x4_pointnet2",
            "pu_gcn": "original_pugcn_x4_pointnet2",
            "pdans": "original_pdans_x4_pointnet2",
        }
        lk = legacy_key_map[method]
        if lk in existing:
            rows[-1]["overall_accuracy"] = existing[lk].get("best_test_acc", "")
            rows[-1]["checkpoint"] = existing[lk].get("checkpoint", "")
            rows[-1]["status"] = existing[lk].get("status", "")

    add("B", "baseline", "downsampled_x4", "lineB_downsampled_x4_baseline",
        paths["lineB_baseline"], counts["N_div_4"], "NEW: 256-pt baseline; retrain required")
    legacy_b = existing.get("downsampled50_native512_pointnet2", {})
    rows[-1]["note"] = "Legacy downsampled50_native512 (512 pts) exists but is OLD protocol"

    for method in MAIN_METHODS:
        exp_b = f"lineB_downsampled_x4_up_{method}"
        add("B", "upsampling", METHOD_LABELS[method], exp_b,
            paths["lineB_up_root"] / method, n, "NEW: 256→1024; retrain required")
        lk = {
            "ear": "downsampled50_ear_x4_pointnet2",
            "pu_net": "downsampled50_punet_x4_pointnet2",
            "pu_gcn": "downsampled50_pugcn_x4_pointnet2",
            "pdans": "downsampled50_pdans_x4_pointnet2",
        }[method]
        if lk in existing:
            rows[-1]["note"] = f"Legacy {lk} (512→2048) exists — NOT valid for new Line B"

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    n = detect_original_point_count()
    rows = build_experiment_table(n)

    rep = reports_dir()
    csv_path = rep / "modelnet40_pointnet2_classification_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    results_root = PROJECT_ROOT / "pointnet2_results"
    results_root.mkdir(parents=True, exist_ok=True)

    md = rep / "modelnet40_pointnet2_classification_plan.md"
    md.write_text(
        "\n".join([
            "# PointNet++ Classification Plan (×4 Two-Line Protocol)",
            "",
            f"- Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "Train with `train_pointnet2.py --data-root <path> --num-point <N> --allow-resample false`.",
            "",
            f"Summary CSV: `{csv_path}`",
            f"Results root: `{results_root}`",
        ]),
        encoding="utf-8",
    )
    print(f"Wrote classification summary: {csv_path}")
    for r in rows:
        print(f"  Line {r['line']} {r['method']}: {r['status']} ({r['input_point_count']} pts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
