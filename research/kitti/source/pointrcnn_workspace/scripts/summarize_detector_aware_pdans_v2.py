#!/usr/bin/env python3
"""Create the frozen 256-frame V2 dual-detector decision report."""

from __future__ import annotations

import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/detector_aware_pdans_v2_20260805/reports"
POINT_BASE = REPO / "results/pointrcnn_split_region_object_preserving256_v1_20260804/detector/original_baseline/evaluation/result_summary.json"
POINT_V1 = REPO / "results/pointrcnn_object_preserving_detector_aware_pdans256_v1_20260805/detector/full/evaluation/result_summary.json"
POINT_V2 = REPO / "results/pointrcnn_object_preserving_pdans_v2_20260805/detector"
CENTER_BASE = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/detectors/centerpoint/original_baseline/result_summary.json"
CENTER_V1 = REPO / "results/centerpoint_detector_aware_pdans256_v1_20260805/full/result_summary.json"
CENTER_V2 = REPO / "results/centerpoint_detector_aware_pdans_v2_20260805"
DIFFICULTIES = ("easy", "moderate", "hard")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    point_sources = {
        "baseline": POINT_BASE,
        "v1_full": POINT_V1,
        "v2_voxel_only": POINT_V2 / "v2_voxel_only/evaluation/result_summary.json",
        "v2_proposal_only": POINT_V2 / "v2_proposal_pointrcnn/evaluation/result_summary.json",
        "v2_full": POINT_V2 / "v2_full_pointrcnn/evaluation/result_summary.json",
    }
    center_sources = {
        "baseline": CENTER_BASE,
        "v1_full": CENTER_V1,
        "v2_voxel_only": CENTER_V2 / "v2_voxel_only/result_summary.json",
        "v2_proposal_only": CENTER_V2 / "v2_proposal_centerpoint/result_summary.json",
        "v2_full": CENTER_V2 / "v2_full_centerpoint/result_summary.json",
    }
    rows: list[dict] = []
    point_payload = {name: load(path)["ap_r40_percent"] for name, path in point_sources.items()}
    for name, metrics in point_payload.items():
        for metric, values in metrics.items():
            for difficulty in DIFFICULTIES:
                base = float(point_payload["baseline"][metric][difficulty])
                value = float(values[difficulty])
                rows.append({
                    "detector": "PointRCNN", "variant": name, "class": "Car",
                    "metric": metric, "difficulty": difficulty, "ap_r40": value,
                    "delta_vs_baseline": value - base,
                })
    center_payload = {name: load(path)["metrics_percent"] for name, path in center_sources.items()}
    for name, classes in center_payload.items():
        for class_name, metrics in classes.items():
            for metric, values in metrics.items():
                for difficulty in DIFFICULTIES:
                    base = float(center_payload["baseline"][class_name][metric][difficulty])
                    value = float(values[difficulty])
                    rows.append({
                        "detector": "CenterPoint", "variant": name, "class": class_name,
                        "metric": metric, "difficulty": difficulty, "ap_r40": value,
                        "delta_vs_baseline": value - base,
                    })
    write_csv(OUT / "all_ap_r40_with_deltas.csv", rows)

    def value(detector: str, variant: str, class_name: str, metric: str, difficulty: str = "moderate") -> tuple[float, float]:
        row = next(item for item in rows if item["detector"] == detector and item["variant"] == variant and item["class"] == class_name and item["metric"] == metric and item["difficulty"] == difficulty)
        return float(row["ap_r40"]), float(row["delta_vs_baseline"])

    lines = [
        "# Detector-aware PDANS V2 — frozen 256-frame decision report", "",
        "## Moderate AP_R40", "",
        "| Detector | Variant | Class | 3D AP | delta | BEV AP | delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for detector, classes in (("PointRCNN", ("Car",)), ("CenterPoint", ("Car", "Pedestrian", "Cyclist"))):
        for variant in ("baseline", "v1_full", "v2_voxel_only", "v2_proposal_only", "v2_full"):
            for class_name in classes:
                metric3d = "3d_ap" if detector == "PointRCNN" else "3d_ap_r40"
                metricbev = "bev_ap" if detector == "PointRCNN" else "bev_ap_r40"
                d3 = value(detector, variant, class_name, metric3d)
                bev = value(detector, variant, class_name, metricbev)
                lines.append(f"| {detector} | {variant} | {class_name} | {d3[0]:.4f} | {d3[1]:+.4f} | {bev[0]:.4f} | {bev[1]:+.4f} |")
    lines += [
        "", "## Gate decision", "",
        "- V2 voxel-only, proposal-only, and Full all fail the pre-registered cross-detector gate.",
        "- PointRCNN V1 Full remains the best PointRCNN 3D result, but its Moderate BEV AP is below baseline.",
        "- CenterPoint V2 Full improves Pedestrian but materially degrades Car and Cyclist; it is not a detector-wide improvement.",
        "- Final-box proposal gating is rejected: it strongly amplifies first-pass detector bias in both detectors.",
        "- Do not expand V2 or the current cross-detector method to 3,769 frames.",
        "", "## Valid next step", "",
        "Treat the current 256 frames as development/diagnostic data. Any further method change must be specified before AP evaluation and validated on a disjoint frame subset. The defensible paper result at this point is a detector-specific PointRCNN V1 result plus a negative cross-detector transfer finding, not a universal improvement claim.",
        "",
    ]
    report = "\n".join(lines)
    (OUT / "v2_decision_report.md").write_text(report, encoding="utf-8")
    (OUT / "metadata.json").write_text(json.dumps({
        "status": "PASS", "frames": 256, "gate": "FAIL",
        "expand_to_3769": False,
        "reason": "No V2 variant improves both frozen detectors without material class/metric regression.",
    }, indent=2) + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
