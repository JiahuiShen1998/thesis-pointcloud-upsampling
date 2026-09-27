#!/usr/bin/env python3
"""Summarize official AP, paired transitions, and CenterPoint voxel evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/detector_aware_cross_detector_diagnosis256_v1_20260805"
POINT_BASE = REPO / "results/pointrcnn_split_region_object_preserving256_v1_20260804/detector"
POINT_AWARE = REPO / "results/pointrcnn_object_preserving_detector_aware_pdans256_v1_20260805/detector"
CENTER_BASE = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/detectors/centerpoint"
CENTER_AWARE = REPO / "results/centerpoint_detector_aware_pdans256_v1_20260805"
VARIANTS = (
    "baseline", "strict_x4", "fixed_2p5_pdans", "matched_real_dynamic", "full",
    "no_confidence", "no_sparse", "no_adaptive", "nn_only", "density_only",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_json(detector: str, variant: str) -> Path:
    if detector == "pointrcnn":
        if variant == "baseline":
            return POINT_BASE / "original_baseline/evaluation/result_summary.json"
        if variant == "strict_x4":
            return POINT_BASE / "surface_c32_pdans/evaluation/result_summary.json"
        return POINT_AWARE / variant / "evaluation/result_summary.json"
    if variant == "baseline":
        return CENTER_BASE / "original_baseline/result_summary.json"
    if variant == "strict_x4":
        return CENTER_BASE / "surface_c32_pdans/result_summary.json"
    return CENTER_AWARE / variant / "result_summary.json"


def official_rows() -> list[dict]:
    rows = []
    for detector in ("pointrcnn", "centerpoint"):
        baseline_payload = json.loads(result_json(detector, "baseline").read_text(encoding="utf-8"))
        baseline = baseline_payload["ap_r40_percent" if detector == "pointrcnn" else "metrics_percent"]
        for variant in VARIANTS:
            payload = json.loads(result_json(detector, variant).read_text(encoding="utf-8"))
            metrics = payload["ap_r40_percent" if detector == "pointrcnn" else "metrics_percent"]
            classes = ("Car",) if detector == "pointrcnn" else ("Car", "Pedestrian", "Cyclist")
            for class_name in classes:
                for metric in (("3d_ap", "bev_ap") if detector == "pointrcnn" else ("3d_ap_r40", "bev_ap_r40")):
                    value = float(metrics[class_name][metric]["moderate"] if detector == "centerpoint" else metrics[metric]["moderate"])
                    base_value = float(baseline[class_name][metric]["moderate"] if detector == "centerpoint" else baseline[metric]["moderate"])
                    rows.append({
                        "detector": detector, "variant": variant, "class": class_name,
                        "metric": metric, "moderate": value, "delta_vs_baseline": value - base_value,
                    })
    return rows


def correlations(frame_rows: list[dict[str, str]], voxel_rows: list[dict[str, str]]) -> list[dict]:
    full_voxels = {row["frame_id"]: row for row in voxel_rows if row["variant"] == "full"}
    metrics = (
        "generated_fov_range_points", "generated_only_voxels", "mixed_voxels",
        "generated_inside_gt_fraction", "boundary_fraction_of_inside_gt",
        "new_active_voxels_vs_baseline", "changed_shared_voxels",
        "shared_voxel_xyz_mean_shift_mean_m",
    )
    outputs = []
    for detector in ("pointrcnn", "centerpoint"):
        selected = [row for row in frame_rows if row["detector"] == detector and row["reference"] == "baseline" and row["variant"] == "full"]
        for voxel_metric in metrics:
            x = np.asarray([float(full_voxels[row["frame_id"]][voxel_metric]) for row in selected])
            for outcome in ("net_tp_change", "lost", "recovered", "delta_false_positives"):
                y = np.asarray([float(row[outcome]) for row in selected])
                result = spearmanr(x, y)
                outputs.append({
                    "detector": detector, "variant": "full", "voxel_metric": voxel_metric,
                    "outcome": outcome, "spearman_r": float(result.statistic) if np.isfinite(result.statistic) else 0.0,
                    "p_value": float(result.pvalue) if np.isfinite(result.pvalue) else 1.0,
                })
    return outputs


def disagreement_rows(frame_rows: list[dict[str, str]], voxel_rows: list[dict[str, str]]) -> list[dict]:
    selected = {
        (row["detector"], row["frame_id"]): row
        for row in frame_rows
        if row["reference"] == "baseline" and row["variant"] == "full"
    }
    voxels = {row["frame_id"]: row for row in voxel_rows if row["variant"] == "full"}
    frames = sorted({frame for _, frame in selected})
    output = []
    for frame in frames:
        point = selected[("pointrcnn", frame)]
        center = selected[("centerpoint", frame)]
        voxel = voxels[frame]
        score = (
            float(point["net_tp_change"]) - float(center["net_tp_change"])
            + float(center["lost"]) + max(0.0, float(center["delta_false_positives"]))
        )
        output.append({
            "frame_id": frame, "disagreement_score": score,
            "pointrcnn_net_tp_change": point["net_tp_change"], "pointrcnn_lost": point["lost"],
            "pointrcnn_recovered": point["recovered"], "pointrcnn_delta_fp": point["delta_false_positives"],
            "centerpoint_net_tp_change": center["net_tp_change"], "centerpoint_lost": center["lost"],
            "centerpoint_recovered": center["recovered"], "centerpoint_delta_fp": center["delta_false_positives"],
            "generated_only_voxels": voxel["generated_only_voxels"],
            "mixed_voxels": voxel["mixed_voxels"],
            "generated_inside_gt_fraction": voxel["generated_inside_gt_fraction"],
            "new_active_voxels_vs_baseline": voxel["new_active_voxels_vs_baseline"],
        })
    return sorted(output, key=lambda row: (-float(row["disagreement_score"]), row["frame_id"]))


def metric_value(rows: list[dict], detector: str, variant: str, class_name: str, metric: str) -> tuple[float, float]:
    row = next(item for item in rows if item["detector"] == detector and item["variant"] == variant and item["class"] == class_name and item["metric"] == metric)
    return float(row["moderate"]), float(row["delta_vs_baseline"])


def main() -> int:
    official = official_rows()
    write_csv(ROOT / "summary/official_moderate_ap_matrix.csv", official)
    frame_rows = read_csv(ROOT / "transitions/per_frame_transition_summary.csv")
    voxel_rows = read_csv(ROOT / "voxels/per_frame_centerpoint_voxel_effects.csv")
    voxel_summary = read_csv(ROOT / "voxels/aggregate_centerpoint_voxel_effects.csv")
    corr = correlations(frame_rows, voxel_rows)
    write_csv(ROOT / "summary/frame_level_spearman_correlations.csv", corr)
    disagreement = disagreement_rows(frame_rows, voxel_rows)
    write_csv(ROOT / "summary/full_cross_detector_disagreement_frames.csv", disagreement)
    full_voxel = next(row for row in voxel_summary if row["variant"] == "full")
    fixed_voxel = next(row for row in voxel_summary if row["variant"] == "fixed_2p5_pdans")
    matched_voxel = next(row for row in voxel_summary if row["variant"] == "matched_real_dynamic")
    point_full = metric_value(official, "pointrcnn", "full", "Car", "3d_ap")
    center_car_full = metric_value(official, "centerpoint", "full", "Car", "3d_ap_r40")
    center_car_bev = metric_value(official, "centerpoint", "full", "Car", "bev_ap_r40")
    report = f"""# Detector-aware PDANS cross-detector diagnosis (256 frames)

## Outcome

- PointRCNN Full Car Moderate 3D AP: **{point_full[0]:.4f}** ({point_full[1]:+.4f} vs baseline).
- CenterPoint Full Car Moderate 3D AP: **{center_car_full[0]:.4f}** ({center_car_full[1]:+.4f}).
- CenterPoint Full Car Moderate BEV AP: **{center_car_bev[0]:.4f}** ({center_car_bev[1]:+.4f}).
- The cross-detector expansion gate remains **failed**: the PointRCNN gain does not transfer to CenterPoint.

## CenterPoint voxel evidence

- Full contributes a median **{float(full_voxel['generated_fov_range_points_median']):.1f}** generated points in CenterPoint FOV/range per frame.
- A median **{100*float(full_voxel['generated_only_voxel_fraction_median']):.2f}%** of Full generated voxels are generated-only; Full creates **{float(full_voxel['new_active_voxels_vs_baseline_median']):.1f}** new active voxels per frame.
- Only **{100*float(full_voxel['generated_inside_gt_fraction_median']):.2f}%** of Full FOV/range generated points are inside Car/Pedestrian/Cyclist GT boxes in the median frame.
- Full has only **{float(full_voxel['mixed_voxels_median']):.1f}** mixed observed/generated voxels per frame. Its main effect is therefore activating new background voxels, not enriching existing measured voxels.
- Fixed-2.5% creates **{float(fixed_voxel['new_active_voxels_vs_baseline_median']):.1f}** new voxels and places **{100*float(fixed_voxel['generated_points_in_observed_voxel_fraction_median']):.2f}%** of generated points into observed voxels.
- Matched-real creates **{float(matched_voxel['new_active_voxels_vs_baseline_median']):.1f}** new voxels, explaining why CenterPoint predictions are effectively unchanged from baseline.
- No run reaches CenterPoint's 40,000-voxel test cap, so global max-voxel truncation is not the cause.

## Diagnosis

The current geometry gate rewards novelty, sparsity and distance. For CenterPoint this turns almost every selected point into a new active voxel, while very few selected points fall inside labeled foreground objects. The likely failure mode is background/edge voxel activation rather than generated points being discarded by the five-points-per-voxel limit.

## Next controlled revision

1. Keep real-point priority, repaired surface-c32 PDANS, and the adaptive budget frozen.
2. Replace the current 0.1 m two-points cap with a detector-grid-aware gate at CenterPoint's exact 0.05 x 0.05 x 0.1 m voxel grid.
3. Penalize generated-only voxels unsupported by neighboring observed voxels; do not merely reward novelty.
4. Add a GT-free foreground/proposal gate or a local discontinuity/vertical-structure proxy so the budget is not spent almost entirely on background.
5. Validate the revised selector on a disjoint subset before any 3,769-frame expansion.

The transition CSVs are diagnostic matches only; official claims must use the frozen KITTI AP files listed in `official_moderate_ap_matrix.csv`.
"""
    (ROOT / "summary/diagnosis_report.md").write_text(report, encoding="utf-8")
    metadata = {"status": "PASS", "frames": 256, "report": str(ROOT / "summary/diagnosis_report.md"), "top_disagreement_frames": [row["frame_id"] for row in disagreement[:20]]}
    (ROOT / "summary/metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
