#!/usr/bin/env python3
"""Summarize the frozen 256-frame internal-proposal V3 ablation."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/detector_aware_pdans_v3_20260805/reports"
POINT_BASE = REPO / "results/pointrcnn_split_region_object_preserving256_v1_20260804/detector/original_baseline/evaluation/result_summary.json"
POINT_V1 = REPO / "results/pointrcnn_object_preserving_detector_aware_pdans256_v1_20260805/detector/full/evaluation/result_summary.json"
POINT_V2 = REPO / "results/pointrcnn_object_preserving_pdans_v2_20260805/detector"
POINT_V3 = REPO / "results/pointrcnn_object_preserving_pdans_v3_20260805/detector"
CENTER_BASE = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/detectors/centerpoint/original_baseline/result_summary.json"
CENTER_V1 = REPO / "results/centerpoint_detector_aware_pdans256_v1_20260805/full/result_summary.json"
CENTER_V2 = REPO / "results/centerpoint_detector_aware_pdans_v2_20260805"
CENTER_V3 = REPO / "results/centerpoint_detector_aware_pdans_v3_20260805"
DIFFICULTIES = ("easy", "moderate", "hard")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    point_sources = {
        "baseline": POINT_BASE, "v1_full": POINT_V1,
        "v2_voxel_only": POINT_V2 / "v2_voxel_only/evaluation/result_summary.json",
        "v2_proposal_only": POINT_V2 / "v2_proposal_pointrcnn/evaluation/result_summary.json",
        "v2_full": POINT_V2 / "v2_full_pointrcnn/evaluation/result_summary.json",
        "v3_internal_proposal_only": POINT_V3 / "v3_internal_proposal_pointrcnn/evaluation/result_summary.json",
        "v3_internal_full": POINT_V3 / "v3_internal_full_pointrcnn/evaluation/result_summary.json",
    }
    center_sources = {
        "baseline": CENTER_BASE, "v1_full": CENTER_V1,
        "v2_voxel_only": CENTER_V2 / "v2_voxel_only/result_summary.json",
        "v2_proposal_only": CENTER_V2 / "v2_proposal_centerpoint/result_summary.json",
        "v2_full": CENTER_V2 / "v2_full_centerpoint/result_summary.json",
        "v3_internal_proposal_only": CENTER_V3 / "v3_internal_proposal_centerpoint/result_summary.json",
        "v3_internal_full": CENTER_V3 / "v3_internal_full_centerpoint/result_summary.json",
    }
    rows = []
    point = {name: load(path)["ap_r40_percent"] for name, path in point_sources.items()}
    for variant, metrics in point.items():
        for metric, values in metrics.items():
            for difficulty in DIFFICULTIES:
                value = float(values[difficulty])
                rows.append({
                    "detector": "PointRCNN", "variant": variant, "class": "Car",
                    "metric": metric, "difficulty": difficulty, "ap_r40": value,
                    "delta_vs_baseline": value - float(point["baseline"][metric][difficulty]),
                    "delta_vs_v2_full": value - float(point["v2_full"][metric][difficulty]),
                })
    center = {name: load(path)["metrics_percent"] for name, path in center_sources.items()}
    for variant, classes in center.items():
        for class_name, metrics in classes.items():
            for metric, values in metrics.items():
                for difficulty in DIFFICULTIES:
                    value = float(values[difficulty])
                    rows.append({
                        "detector": "CenterPoint", "variant": variant, "class": class_name,
                        "metric": metric, "difficulty": difficulty, "ap_r40": value,
                        "delta_vs_baseline": value - float(center["baseline"][class_name][metric][difficulty]),
                        "delta_vs_v2_full": value - float(center["v2_full"][class_name][metric][difficulty]),
                    })
    write_csv(OUT / "all_ap_r40_with_deltas.csv", rows)

    def get(detector: str, variant: str, cls: str, metric: str, difficulty: str) -> dict:
        return next(row for row in rows if row["detector"] == detector and row["variant"] == variant
                    and row["class"] == cls and row["metric"] == metric and row["difficulty"] == difficulty)

    lines = [
        "# Detector-aware PDANS V3 — true internal-proposal 256-frame report", "",
        "V3 is GT-free. PointRCNN uses true pre-RCNN RPN proposals; CenterPoint uses decoded heatmap top-K boxes before NMS.", "",
        "## Moderate AP_R40 comparison", "",
        "| Detector | Variant | Class | 3D AP | Δ baseline | Δ V2 Full | BEV AP | Δ baseline | Δ V2 Full |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    variants = ("baseline", "v1_full", "v2_voxel_only", "v2_proposal_only", "v2_full",
                "v3_internal_proposal_only", "v3_internal_full")
    for detector, classes, metric3d, metricbev in (
        ("PointRCNN", ("Car",), "3d_ap", "bev_ap"),
        ("CenterPoint", ("Car", "Pedestrian", "Cyclist"), "3d_ap_r40", "bev_ap_r40"),
    ):
        for variant in variants:
            for cls in classes:
                d3 = get(detector, variant, cls, metric3d, "moderate")
                bev = get(detector, variant, cls, metricbev, "moderate")
                lines.append(
                    f"| {detector} | {variant} | {cls} | {d3['ap_r40']:.4f} | {d3['delta_vs_baseline']:+.4f} | "
                    f"{d3['delta_vs_v2_full']:+.4f} | {bev['ap_r40']:.4f} | {bev['delta_vs_baseline']:+.4f} | {bev['delta_vs_v2_full']:+.4f} |"
                )

    lines += ["", "## V3 Full: all reported AP metrics and difficulties", "",
              "| Detector | Class | Metric | Easy | Moderate | Hard |",
              "|---|---|---|---:|---:|---:|"]
    for metric in ("bbox_ap", "bev_ap", "3d_ap", "aos_ap"):
        values = [get("PointRCNN", "v3_internal_full", "Car", metric, difficulty)["ap_r40"] for difficulty in DIFFICULTIES]
        lines.append(f"| PointRCNN | Car | {metric} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    for cls in ("Car", "Pedestrian", "Cyclist"):
        for metric in ("bev_ap_r40", "3d_ap_r40"):
            values = [get("CenterPoint", "v3_internal_full", cls, metric, difficulty)["ap_r40"] for difficulty in DIFFICULTIES]
            lines.append(f"| CenterPoint | {cls} | {metric} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")

    lines += [
        "", "## Decision", "",
        "- V3 Full fails the expansion gate: neither detector improves its primary Car 3D and BEV metrics over baseline.",
        "- Replacing final detection boxes with true internal candidates changes V2 only marginally; detector-box bias was not the principal failure source.",
        "- CenterPoint still improves Pedestrian, but materially degrades Car and Cyclist, so the gain is not detector-wide.",
        "- Proposal-only performs worse than Full in both detectors, confirming that geometric/voxel quality filtering is necessary but not sufficient.",
        "- Do not expand V3 to 3,769 frames. The current evidence points to generated-point distribution/semantics, not proposal extraction correctness, as the remaining issue.", "",
        "The complete class × metric × difficulty table, including deltas against baseline and V2 Full, is in `all_ap_r40_with_deltas.csv`.", "",
    ]
    report = "\n".join(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v3_decision_report.md").write_text(report)
    (OUT / "metadata.json").write_text(json.dumps({
        "status": "PASS", "frames": 256, "gate": "FAIL", "expand_to_3769": False,
        "uses_gt_in_selector": False,
        "reason": "True internal-proposal V3 does not recover baseline Car detection in either frozen detector.",
    }, indent=2) + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
