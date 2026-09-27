#!/usr/bin/env python3
"""Diagnose V3 detector regressions using frozen 256-frame artifacts.

This is a read-only post-hoc audit. GT is used only to explain already-frozen
AP changes; it is never used by the selector or to produce another input.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import analyze_detector_aware_centerpoint_voxels256 as vox  # noqa: E402
import analyze_detector_aware_transitions256 as trans  # noqa: E402


OUTPUT = REPO / "results/detector_aware_pdans_v3_20260805/failure_diagnosis"
SPLIT = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/splits/pilot256.txt"
OBSERVED = REPO / "data/KITTI/object/training/velodyne_original_val"
KITTI = REPO / "data/KITTI/object/training"
POINT_PATHS = {
    "baseline": REPO / "results/pointrcnn_split_region_object_preserving256_v1_20260804/detector/original_baseline/merged_predictions",
    "v2_full": REPO / "results/pointrcnn_object_preserving_pdans_v2_20260805/detector/v2_full_pointrcnn/merged_predictions",
    "v3_proposal": REPO / "results/pointrcnn_object_preserving_pdans_v3_20260805/detector/v3_internal_proposal_pointrcnn/merged_predictions",
    "v3_full": REPO / "results/pointrcnn_object_preserving_pdans_v3_20260805/detector/v3_internal_full_pointrcnn/merged_predictions",
}
CENTER_ROOTS = {
    "baseline": REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/detectors/centerpoint/original_baseline",
    "v2_full": REPO / "results/centerpoint_detector_aware_pdans_v2_20260805/v2_full_centerpoint",
    "v3_proposal": REPO / "results/centerpoint_detector_aware_pdans_v3_20260805/v3_internal_proposal_centerpoint",
    "v3_full": REPO / "results/centerpoint_detector_aware_pdans_v3_20260805/v3_internal_full_centerpoint",
}
VOXEL_INPUTS = {
    "v2_full": REPO / "results/detector_aware_pdans_v2_20260805/inputs/v2_full_centerpoint",
    "v3_proposal": REPO / "results/detector_aware_pdans_v3_20260805/inputs/v3_internal_proposal_centerpoint",
    "v3_full": REPO / "results/detector_aware_pdans_v3_20260805/inputs/v3_internal_full_centerpoint",
}
VARIANTS = ("v2_full", "v3_proposal", "v3_full")


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


def center_payload(root: Path) -> dict[str, dict]:
    matches = sorted(root.rglob("result.pkl"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one result.pkl under {root}, got {matches}")
    with matches[0].open("rb") as handle:
        return {str(item["frame_id"]): item for item in pickle.load(handle)}


def voxel_aggregate(rows: list[dict]) -> list[dict]:
    fields = (
        "supplement_points_raw", "generated_fov_range_points", "active_voxels",
        "generated_voxels", "generated_only_voxels", "mixed_voxels",
        "generated_points_in_observed_voxel_fraction", "generated_only_voxel_fraction",
        "generated_retention_fraction_max5", "generated_inside_gt_fraction",
        "boundary_fraction_of_inside_gt", "new_active_voxels_vs_baseline",
        "changed_shared_voxels", "shared_voxel_xyz_mean_shift_mean_m",
        "shared_voxel_intensity_mean_abs_shift",
    )
    output = []
    for variant in VARIANTS:
        chosen = [row for row in rows if row["variant"] == variant]
        item = {"variant": variant, "frames": len(chosen),
                "voxel_cap_hit_frames": sum(bool(row["voxel_cap_hit"]) for row in chosen)}
        for field in fields:
            values = np.asarray([float(row[field]) for row in chosen])
            item[f"{field}_mean"] = float(values.mean())
            item[f"{field}_median"] = float(np.median(values))
            item[f"{field}_p90"] = float(np.quantile(values, 0.90))
        output.append(item)
    return output


def correlations(frame_rows: list[dict], voxel_rows: list[dict]) -> list[dict]:
    lookup = {(row["variant"], row["frame_id"]): row for row in voxel_rows}
    metrics = (
        "generated_fov_range_points", "generated_only_voxels", "mixed_voxels",
        "generated_inside_gt_fraction", "new_active_voxels_vs_baseline",
        "changed_shared_voxels", "shared_voxel_xyz_mean_shift_mean_m",
    )
    output = []
    for detector in ("pointrcnn", "centerpoint"):
        for variant in VARIANTS:
            selected = [row for row in frame_rows if row["detector"] == detector and row["variant"] == variant]
            for metric in metrics:
                x = np.asarray([float(lookup[(variant, row["frame_id"])][metric]) for row in selected])
                for outcome in ("net_tp_change", "lost", "recovered", "delta_false_positives"):
                    y = np.asarray([float(row[outcome]) for row in selected])
                    result = spearmanr(x, y)
                    output.append({
                        "detector": detector, "variant": variant, "voxel_metric": metric,
                        "outcome": outcome,
                        "spearman_r": float(result.statistic) if np.isfinite(result.statistic) else 0.0,
                        "p_value": float(result.pvalue) if np.isfinite(result.pvalue) else 1.0,
                    })
    return output


def main() -> int:
    frames = [line.strip() for line in SPLIT.read_text().splitlines() if line.strip()]
    centers = {name: center_payload(root) for name, root in CENTER_ROOTS.items()}
    transition_rows = []
    frame_rows = []
    voxel_rows = []
    for position, frame in enumerate(frames, start=1):
        calib = trans.helper.det.parse_calib(KITTI / "calib" / f"{frame}.txt")
        gt = trans.helper.gt_for_frame(frame, calib)
        point_predictions = {
            name: trans.helper.det.parse_kitti_boxes(path / f"{frame}.txt", calib, name)
            for name, path in POINT_PATHS.items()
        }
        center_predictions = {
            name: trans.helper.centerpoint_boxes(payload[frame], name)
            for name, payload in centers.items()
        }
        for detector, predictions in (("pointrcnn", point_predictions), ("centerpoint", center_predictions)):
            for variant in VARIANTS:
                objects, summary = trans.audit_pair(
                    detector, "baseline", variant, frame, gt,
                    predictions["baseline"], predictions[variant],
                )
                transition_rows.extend(objects)
                frame_rows.append(summary)

        observed = vox.read_bin(OBSERVED / f"{frame}.bin")
        base_points = observed[vox.centerpoint_mask(observed, frame, calib)]
        base_keys = vox.voxel_keys(base_points)
        baseline = {"_points": base_points, "_keys": base_keys, "_keep": vox.retained_mask(base_keys)}
        for variant, input_root in VOXEL_INPUTS.items():
            cloud = vox.read_bin(input_root / f"{frame}.bin")
            item = vox.analyze_cloud(frame, variant, observed, cloud, calib, gt)
            vox.add_feature_shift(item, baseline)
            voxel_rows.append({key: value for key, value in item.items() if not key.startswith("_")})
        if position == 1 or position % 32 == 0 or position == len(frames):
            print(f"V3_DIAG {position}/{len(frames)}", flush=True)

    transition_agg = trans.aggregate(transition_rows)
    voxel_agg = voxel_aggregate(voxel_rows)
    corr = correlations(frame_rows, voxel_rows)
    write_csv(OUTPUT / "transitions/per_object.csv", transition_rows)
    write_csv(OUTPUT / "transitions/per_frame.csv", frame_rows)
    write_csv(OUTPUT / "transitions/aggregate.csv", transition_agg)
    write_csv(OUTPUT / "voxels/per_frame.csv", voxel_rows)
    write_csv(OUTPUT / "voxels/aggregate.csv", voxel_agg)
    write_csv(OUTPUT / "correlations.csv", corr)

    def class_row(detector: str, variant: str, class_name: str) -> dict:
        return next(row for row in transition_agg if row["detector"] == detector
                    and row["variant"] == variant and row["stratum_type"] == "class"
                    and row["stratum"] == class_name)

    def vox_row(variant: str) -> dict:
        return next(row for row in voxel_agg if row["variant"] == variant)

    lines = ["# PDANS V3 frozen failure diagnosis", "",
             "GT is used only for this post-hoc explanation, never by the selector.", "",
             "## GT-matched detection transitions", "",
             "| Detector | Variant | Class | Maintained | Conf. degraded | Loc. degraded | Lost | Recovered | Net TP |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for detector, classes in (("pointrcnn", ("Car",)), ("centerpoint", ("Car", "Pedestrian", "Cyclist"))):
        for variant in VARIANTS:
            for class_name in classes:
                row = class_row(detector, variant, class_name)
                lines.append(
                    f"| {detector} | {variant} | {class_name} | {row['maintained']} | "
                    f"{row['confidence_degraded']} | {row['localization_degraded']} | {row['lost']} | "
                    f"{row['recovered']} | {row['net_tp_change']} |"
                )
    lines += ["", "## CenterPoint voxel effects", "",
              "| Variant | Generated points | New active voxels | Generated-only voxel % | Mixed voxels | Inside GT % | Retained by max-5 % |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        row = vox_row(variant)
        lines.append(
            f"| {variant} | {row['generated_fov_range_points_median']:.1f} | "
            f"{row['new_active_voxels_vs_baseline_median']:.1f} | "
            f"{100*row['generated_only_voxel_fraction_median']:.2f} | {row['mixed_voxels_median']:.1f} | "
            f"{100*row['generated_inside_gt_fraction_median']:.2f} | "
            f"{100*row['generated_retention_fraction_max5_median']:.2f} |"
        )

    v2 = vox_row("v2_full")
    v3 = vox_row("v3_full")
    point_v3 = class_row("pointrcnn", "v3_full", "Car")
    center_car = class_row("centerpoint", "v3_full", "Car")
    lines += ["", "## Diagnosis", "",
              f"- V3 Full changes median new CenterPoint voxels from {v2['new_active_voxels_vs_baseline_median']:.1f} (V2) to {v3['new_active_voxels_vs_baseline_median']:.1f}.",
              f"- V3 Full generated-only voxel fraction remains {100*v3['generated_only_voxel_fraction_median']:.2f}%; internal proposals did not turn supplements into measured-surface densification.",
              f"- PointRCNN Car diagnostic net TP change is {point_v3['net_tp_change']}; CenterPoint Car is {center_car['net_tp_change']}.",
              "- If V2 and V3 voxel/transition profiles are nearly identical, proposal extraction is causally closed and the next revision must constrain voxel activation or generated-point semantics, not proposal thresholds.", ""]
    report = "\n".join(lines)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "diagnosis_report.md").write_text(report)
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "status": "PASS", "frames": len(frames), "uses_gt_only_posthoc": True,
        "selector_or_detector_outputs_modified": False,
    }, indent=2) + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
