#!/usr/bin/env python3
"""Paired GT-transition audit for the 256-frame detector-aware PDANS study.

This is a diagnostic audit, not the official KITTI AP implementation.  It
reuses frozen predictions and never changes detector or upsampling outputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "results/detector_aware_pdans_surface256_v1_20260804/protocol.json"
OUTPUT = REPO / "results/detector_aware_cross_detector_diagnosis256_v1_20260805"
POINT_BASE = REPO / "results/pointrcnn_split_region_object_preserving256_v1_20260804/detector"
POINT_AWARE = REPO / "results/pointrcnn_object_preserving_detector_aware_pdans256_v1_20260805/detector"
CENTER_BASE = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/detectors/centerpoint"
CENTER_AWARE = REPO / "results/centerpoint_detector_aware_pdans256_v1_20260805"
KITTI = REPO / "data/KITTI/object/training"

VARIANTS = (
    "strict_x4",
    "fixed_2p5_pdans",
    "matched_real_dynamic",
    "full",
    "no_confidence",
    "no_sparse",
    "no_adaptive",
    "nn_only",
    "density_only",
)
PAIRS = (("baseline", name) for name in VARIANTS)


def load_helpers():
    path = REPO / "scripts/analyze_dual_detector_frame_transitions.py"
    spec = importlib.util.spec_from_file_location("detector_transition_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


helper = load_helpers()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def point_path(variant: str) -> Path:
    if variant == "baseline":
        return POINT_BASE / "original_baseline/merged_predictions"
    if variant == "strict_x4":
        return POINT_BASE / "surface_c32_pdans/merged_predictions"
    return POINT_AWARE / variant / "merged_predictions"


def center_path(variant: str) -> Path:
    if variant == "baseline":
        root = CENTER_BASE / "original_baseline"
    elif variant == "strict_x4":
        root = CENTER_BASE / "surface_c32_pdans"
    else:
        root = CENTER_AWARE / variant
    matches = sorted(root.rglob("result.pkl"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one result.pkl under {root}, got {matches}")
    return matches[0]


def load_center(path: Path) -> dict[str, dict]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return {str(item["frame_id"]): item for item in payload}


def difficulty(gt: dict) -> str:
    height = float(gt.get("bbox_height", 0.0))
    occlusion = int(gt.get("occlusion", 99))
    truncation = float(gt.get("truncation", 99.0))
    if height >= 40 and occlusion <= 0 and truncation <= 0.15:
        return "easy"
    if height >= 25 and occlusion <= 1 and truncation <= 0.30:
        return "moderate"
    if height >= 25 and occlusion <= 2 and truncation <= 0.50:
        return "hard"
    return "ignored"


def distance_bin(value: float) -> str:
    if value < 20.0:
        return "0-20m"
    if value < 40.0:
        return "20-40m"
    return "40m+"


def classify(base_match: dict | None, new_match: dict | None) -> str:
    if base_match and new_match:
        if float(new_match["iou3d"]) - float(base_match["iou3d"]) <= -0.10:
            return "localization_degraded"
        if float(new_match["score"] or 0.0) - float(base_match["score"] or 0.0) <= -0.10:
            return "confidence_degraded"
        return "maintained"
    if base_match:
        return "lost"
    if new_match:
        return "recovered"
    return "missed_both"


def audit_pair(
    detector: str,
    reference: str,
    variant: str,
    frame: str,
    gt: list[dict],
    reference_predictions: list[dict],
    variant_predictions: list[dict],
) -> tuple[list[dict], dict]:
    classes = ("Car",) if detector == "pointrcnn" else ("Car", "Pedestrian", "Cyclist")
    old = helper.match_gt_predictions(gt, reference_predictions, classes)
    new = helper.match_gt_predictions(gt, variant_predictions, classes)
    rows: list[dict] = []
    counts = Counter()
    for index, gt_box in enumerate(old["gt"]):
        old_match = old["match_by_gt"].get(index)
        new_match = new["match_by_gt"].get(index)
        transition = classify(old_match, new_match)
        counts[transition] += 1
        center = helper.box_center(gt_box)
        rows.append(
            {
                "detector": detector,
                "reference": reference,
                "variant": variant,
                "frame_id": frame,
                "gt_index": index,
                "class": gt_box["class"],
                "difficulty": difficulty(gt_box),
                "distance_m": float(center[0]),
                "distance_bin": distance_bin(float(center[0])),
                "transition": transition,
                "reference_score": "" if old_match is None else old_match["score"],
                "variant_score": "" if new_match is None else new_match["score"],
                "reference_3d_iou": "" if old_match is None else old_match["iou3d"],
                "variant_3d_iou": "" if new_match is None else new_match["iou3d"],
                "delta_3d_iou": "" if old_match is None or new_match is None else float(new_match["iou3d"]) - float(old_match["iou3d"]),
            }
        )
    frame_row = {
        "detector": detector,
        "reference": reference,
        "variant": variant,
        "frame_id": frame,
        **{key: counts[key] for key in ("maintained", "confidence_degraded", "localization_degraded", "lost", "recovered", "missed_both")},
        "reference_false_positives": len(old["false_positive_pred"]),
        "variant_false_positives": len(new["false_positive_pred"]),
        "delta_false_positives": len(new["false_positive_pred"]) - len(old["false_positive_pred"]),
        "net_tp_change": counts["recovered"] - counts["lost"],
    }
    return rows, frame_row


def aggregate(rows: list[dict]) -> list[dict]:
    statuses = ("maintained", "confidence_degraded", "localization_degraded", "lost", "recovered", "missed_both")
    groups: dict[tuple, Counter] = defaultdict(Counter)
    for row in rows:
        for level, stratum in (
            ("class", row["class"]),
            ("difficulty", row["difficulty"]),
            ("distance", row["distance_bin"]),
            ("class_distance", f"{row['class']}:{row['distance_bin']}"),
        ):
            groups[(row["detector"], row["reference"], row["variant"], level, stratum)][row["transition"]] += 1
    output = []
    for key, counts in sorted(groups.items()):
        old_tp = sum(counts[item] for item in ("maintained", "confidence_degraded", "localization_degraded", "lost"))
        new_tp = sum(counts[item] for item in ("maintained", "confidence_degraded", "localization_degraded", "recovered"))
        output.append(
            {
                "detector": key[0], "reference": key[1], "variant": key[2], "stratum_type": key[3], "stratum": key[4],
                **{item: counts[item] for item in statuses},
                "reference_tp": old_tp, "variant_tp": new_tp, "net_tp_change": new_tp - old_tp,
                "lost_rate": counts["lost"] / max(old_tp, 1),
                "recovery_rate": counts["recovered"] / max(sum(counts.values()) - old_tp, 1),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    frames = [str(value) for value in protocol["frame_ids"]]
    if args.limit:
        frames = frames[: args.limit]
    all_variants = ("baseline",) + VARIANTS
    for variant in all_variants:
        if not point_path(variant).is_dir():
            raise FileNotFoundError(point_path(variant))
    center = {variant: load_center(center_path(variant)) for variant in all_variants}
    transition_rows: list[dict] = []
    frame_rows: list[dict] = []
    pair_list = [("baseline", variant) for variant in VARIANTS] + [("matched_real_dynamic", "full")]
    for position, frame in enumerate(frames, start=1):
        calib = helper.det.parse_calib(KITTI / "calib" / f"{frame}.txt")
        gt = helper.gt_for_frame(frame, calib)
        point_predictions = {
            variant: helper.det.parse_kitti_boxes(point_path(variant) / f"{frame}.txt", calib, variant)
            for variant in all_variants
        }
        center_predictions = {
            variant: helper.centerpoint_boxes(center[variant][frame], variant)
            for variant in all_variants
        }
        for detector, predictions in (("pointrcnn", point_predictions), ("centerpoint", center_predictions)):
            for reference, variant in pair_list:
                object_rows, frame_row = audit_pair(
                    detector, reference, variant, frame, gt, predictions[reference], predictions[variant]
                )
                transition_rows.extend(object_rows)
                frame_rows.append(frame_row)
        if position == 1 or position % 32 == 0:
            print(f"transitions {position}/{len(frames)}", flush=True)
    output = args.output.resolve()
    write_csv(output / "transitions/per_object_gt_transitions.csv", transition_rows)
    write_csv(output / "transitions/per_frame_transition_summary.csv", frame_rows)
    write_csv(output / "transitions/aggregate_transition_summary.csv", aggregate(transition_rows))
    metadata = {
        "status": "PASS", "frames": len(frames), "detectors": ["pointrcnn", "centerpoint"],
        "variants": list(VARIANTS), "comparisons": pair_list,
        "warning": "Diagnostic greedy GT matching; official AP remains the frozen KITTI evaluation.",
    }
    (output / "transitions/metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output / 'transitions'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
