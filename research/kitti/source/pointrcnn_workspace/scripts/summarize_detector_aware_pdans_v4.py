#!/usr/bin/env python3
"""Create the frozen 256-frame V4 decision and voxel audit."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import analyze_detector_aware_centerpoint_voxels256 as vox  # noqa: E402
import analyze_detector_aware_transitions256 as trans  # noqa: E402


ROOT = REPO / "results/detector_aware_pdans_v4_20260805"
OUT = ROOT / "reports"
SPLIT = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/splits/pilot256.txt"
OBSERVED = REPO / "data/KITTI/object/training/velodyne_original_val"
KITTI = REPO / "data/KITTI/object/training"
V4_INPUT = ROOT / "inputs/v4_existing_voxel_centerpoint"
POINT_BASE = REPO / "results/pointrcnn_split_region_object_preserving256_v1_20260804/detector/original_baseline/evaluation/result_summary.json"
POINT_V4 = REPO / "results/pointrcnn_object_preserving_pdans_v4_20260805/detector/v4_existing_voxel_pointrcnn/evaluation/result_summary.json"
POINT_BASE_PRED = REPO / "results/pointrcnn_split_region_object_preserving256_v1_20260804/detector/original_baseline/merged_predictions"
POINT_V4_PRED = REPO / "results/pointrcnn_object_preserving_pdans_v4_20260805/detector/v4_existing_voxel_pointrcnn/merged_predictions"
CENTER_BASE = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/detectors/centerpoint/original_baseline/result_summary.json"
CENTER_V4 = REPO / "results/centerpoint_detector_aware_pdans_v4_20260805/v4_existing_voxel_centerpoint/result_summary.json"
CENTER_BASE_ROOT = CENTER_BASE.parent
CENTER_V4_ROOT = CENTER_V4.parent
DIFFICULTIES = ("easy", "moderate", "hard")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def center_results(root: Path) -> dict[str, dict]:
    matches = sorted(root.rglob("result.pkl"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one result.pkl under {root}")
    with matches[0].open("rb") as handle:
        return {str(item["frame_id"]): item for item in pickle.load(handle)}


def main() -> int:
    point_base = json.loads(POINT_BASE.read_text())["ap_r40_percent"]
    point_v4 = json.loads(POINT_V4.read_text())["ap_r40_percent"]
    center_base = json.loads(CENTER_BASE.read_text())["metrics_percent"]
    center_v4 = json.loads(CENTER_V4.read_text())["metrics_percent"]
    metric_rows = []
    for metric, values in point_v4.items():
        for difficulty in DIFFICULTIES:
            value = float(values[difficulty])
            metric_rows.append({"detector": "PointRCNN", "class": "Car", "metric": metric,
                                "difficulty": difficulty, "baseline": point_base[metric][difficulty],
                                "v4": value, "delta": value - float(point_base[metric][difficulty])})
    for cls, metrics in center_v4.items():
        for metric, values in metrics.items():
            for difficulty in DIFFICULTIES:
                value = float(values[difficulty])
                metric_rows.append({"detector": "CenterPoint", "class": cls, "metric": metric,
                                    "difficulty": difficulty, "baseline": center_base[cls][metric][difficulty],
                                    "v4": value, "delta": value - float(center_base[cls][metric][difficulty])})
    write_csv(OUT / "all_v4_ap_r40_vs_baseline.csv", metric_rows)

    frames = [line.strip() for line in SPLIT.read_text().splitlines() if line.strip()]
    center_old, center_new = center_results(CENTER_BASE_ROOT), center_results(CENTER_V4_ROOT)
    transition_objects, transition_frames, voxel_rows = [], [], []
    for position, frame in enumerate(frames, start=1):
        calib = trans.helper.det.parse_calib(KITTI / "calib" / f"{frame}.txt")
        gt = trans.helper.gt_for_frame(frame, calib)
        point_old = trans.helper.det.parse_kitti_boxes(POINT_BASE_PRED / f"{frame}.txt", calib, "baseline")
        point_new = trans.helper.det.parse_kitti_boxes(POINT_V4_PRED / f"{frame}.txt", calib, "v4")
        center_old_boxes = trans.helper.centerpoint_boxes(center_old[frame], "baseline")
        center_new_boxes = trans.helper.centerpoint_boxes(center_new[frame], "v4")
        for detector, old, new in (("pointrcnn", point_old, point_new),
                                   ("centerpoint", center_old_boxes, center_new_boxes)):
            objects, summary = trans.audit_pair(detector, "baseline", "v4", frame, gt, old, new)
            transition_objects.extend(objects)
            transition_frames.append(summary)

        observed = vox.read_bin(OBSERVED / f"{frame}.bin")
        cloud = vox.read_bin(V4_INPUT / f"{frame}.bin")
        base_points = observed[vox.centerpoint_mask(observed, frame, calib)]
        base_keys = vox.voxel_keys(base_points)
        baseline = {"_points": base_points, "_keys": base_keys, "_keep": vox.retained_mask(base_keys)}
        item = vox.analyze_cloud(frame, "v4", observed, cloud, calib, gt)
        vox.add_feature_shift(item, baseline)
        voxel_rows.append({key: value for key, value in item.items() if not key.startswith("_")})
        if position == 1 or position % 64 == 0 or position == len(frames):
            print(f"V4_AUDIT {position}/{len(frames)}", flush=True)
    transition_agg = trans.aggregate(transition_objects)
    write_csv(OUT / "transition_aggregate.csv", transition_agg)
    write_csv(OUT / "transition_per_frame.csv", transition_frames)
    write_csv(OUT / "centerpoint_voxel_per_frame.csv", voxel_rows)

    voxel_fields = (
        "generated_fov_range_points", "new_active_voxels_vs_baseline", "generated_only_voxel_fraction",
        "mixed_voxels", "generated_retention_fraction_max5", "generated_inside_gt_fraction",
        "changed_shared_voxels", "shared_voxel_xyz_mean_shift_mean_m",
    )
    voxel_summary = {"frames": len(voxel_rows)}
    for field in voxel_fields:
        values = np.asarray([float(row[field]) for row in voxel_rows])
        voxel_summary[f"{field}_mean"] = float(values.mean())
        voxel_summary[f"{field}_median"] = float(np.median(values))
        voxel_summary[f"{field}_p90"] = float(np.quantile(values, 0.9))
    (OUT / "centerpoint_voxel_summary.json").write_text(json.dumps(voxel_summary, indent=2) + "\n")

    def metric(detector: str, cls: str, name: str, difficulty: str = "moderate") -> dict:
        return next(row for row in metric_rows if row["detector"] == detector and row["class"] == cls
                    and row["metric"] == name and row["difficulty"] == difficulty)

    lines = ["# Detector-aware PDANS V4 — measured-voxel anchored report", "",
             "## Moderate AP_R40", "",
             "| Detector | Class | 3D baseline | 3D V4 | Δ | BEV baseline | BEV V4 | Δ |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for detector, classes, d3name, bevname in (
        ("PointRCNN", ("Car",), "3d_ap", "bev_ap"),
        ("CenterPoint", ("Car", "Pedestrian", "Cyclist"), "3d_ap_r40", "bev_ap_r40"),
    ):
        for cls in classes:
            d3, bev = metric(detector, cls, d3name), metric(detector, cls, bevname)
            lines.append(f"| {detector} | {cls} | {d3['baseline']:.4f} | {d3['v4']:.4f} | {d3['delta']:+.4f} | "
                         f"{bev['baseline']:.4f} | {bev['v4']:.4f} | {bev['delta']:+.4f} |")
    lines += ["", "## All V4 metrics", "",
              "| Detector | Class | Metric | Easy | Moderate | Hard |",
              "|---|---|---|---:|---:|---:|"]
    for detector, cls, names in (
        ("PointRCNN", "Car", ("bbox_ap", "bev_ap", "3d_ap", "aos_ap")),
        ("CenterPoint", "Car", ("bev_ap_r40", "3d_ap_r40")),
        ("CenterPoint", "Pedestrian", ("bev_ap_r40", "3d_ap_r40")),
        ("CenterPoint", "Cyclist", ("bev_ap_r40", "3d_ap_r40")),
    ):
        for name in names:
            vals = [metric(detector, cls, name, difficulty)["v4"] for difficulty in DIFFICULTIES]
            lines.append(f"| {detector} | {cls} | {name} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |")
    lines += ["", "## Voxel audit", "",
              f"- Median generated points in CenterPoint range: {voxel_summary['generated_fov_range_points_median']:.1f}.",
              f"- Median new active voxels: {voxel_summary['new_active_voxels_vs_baseline_median']:.1f}.",
              f"- Median generated-only voxel fraction: {100*voxel_summary['generated_only_voxel_fraction_median']:.2f}%.",
              f"- Median retention after native max-5: {100*voxel_summary['generated_retention_fraction_max5_median']:.2f}%.",
              "", "## Decision", "",
              "- V4 validates the causal diagnosis: eliminating generated-only voxel activation recovers the large V3 Car regression in CenterPoint and improves PointRCNN Car 3D/BEV.",
              "- The strict detector-wide expansion gate is not yet met because CenterPoint Pedestrian/Cyclist 3D and some Hard metrics still regress.",
              "- Do not expand this exact V4 to 3,769 frames yet; the next bounded test should use the predicted-class adaptive gate already specified in the method proposal, while keeping this measured-voxel anchor frozen.", ""]
    report = "\n".join(lines)
    (OUT / "v4_decision_report.md").write_text(report)
    (OUT / "metadata.json").write_text(json.dumps({
        "status": "PASS", "frames": 256, "causal_hypothesis_supported": True,
        "strict_expansion_gate": "FAIL", "expand_to_3769": False,
    }, indent=2) + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
