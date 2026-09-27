#!/usr/bin/env python3
"""Summarize the paired 256-frame PU-Net normalization/patch 2x2 study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
GROUPS = {
    "A": "legacy_oldpatch",
    "B": "fixed_oldpatch",
    "C": "legacy_cover_knn_v3",
    "D": "fixed_cover_knn_v3",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise RuntimeError(f"detector summary is not PASS: {path}")
    return payload


def summary_path(root: Path, detector: str, code: str) -> Path:
    return (
        root
        / "detectors"
        / detector
        / f"punet_2x2_{code}_{GROUPS[code]}"
        / "result_summary.json"
    )


def baseline_path(root: Path, detector: str) -> Path:
    if detector == "pointrcnn":
        return (
            root
            / "detectors/pointrcnn/reference_pilot256/original_baseline"
            / "result_summary.json"
        )
    return root / "detectors/centerpoint/original_baseline/result_summary.json"


def box_summary_path(root: Path, detector: str, code: str) -> Path:
    return (
        root
        / "detectors/box_changes/punet_2x2"
        / f"{detector}_{code}_vs_original_baseline"
        / "box_change_summary.json"
    )


def flatten_pointrcnn(payload: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    metrics = payload["ap_r40_percent"]
    for source_key, metric_name in (("bev_ap", "bev_ap_r40"), ("3d_ap", "3d_ap_r40")):
        for difficulty, value in metrics[source_key].items():
            result[("Car", metric_name, difficulty)] = float(value)
    return result


def flatten_centerpoint(payload: dict[str, Any]) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    for class_name, class_metrics in payload["metrics_percent"].items():
        for metric_name, difficulties in class_metrics.items():
            for difficulty, value in difficulties.items():
                result[(class_name, metric_name, difficulty)] = float(value)
    return result


def effect_row(
    detector: str,
    key: tuple[str, str, str],
    values: dict[str, float],
    baseline: float,
) -> dict[str, Any]:
    a, b, c, d = (values[code] for code in "ABCD")
    norm_old = b - a
    norm_local = d - c
    patch_legacy = c - a
    patch_fixed = d - b
    return {
        "detector": detector,
        "class": key[0],
        "metric": key[1],
        "difficulty": key[2],
        "baseline": baseline,
        "A": a,
        "B": b,
        "C": c,
        "D": d,
        "normalization_at_old_patch_B_minus_A": norm_old,
        "normalization_at_local_patch_D_minus_C": norm_local,
        "patch_at_legacy_norm_C_minus_A": patch_legacy,
        "patch_at_fixed_norm_D_minus_B": patch_fixed,
        "normalization_main_effect": (norm_old + norm_local) / 2.0,
        "patch_main_effect": (patch_legacy + patch_fixed) / 2.0,
        "interaction_D_minus_C_minus_B_plus_A": d - c - b + a,
        "D_minus_A": d - a,
        "D_minus_baseline": d - baseline,
    }


def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_root.resolve()
    output = (args.output_dir or root / "punet256_2x2/reports").resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for detector, flatten in (
        ("pointrcnn", flatten_pointrcnn),
        ("centerpoint", flatten_centerpoint),
    ):
        group_metrics = {
            code: flatten(load_json(summary_path(root, detector, code)))
            for code in GROUPS
        }
        baseline_metrics = flatten(load_json(baseline_path(root, detector)))
        expected_keys = set(group_metrics["A"])
        if any(set(group_metrics[code]) != expected_keys for code in "BCD"):
            raise RuntimeError(f"metric-key mismatch across {detector} A-D summaries")
        if not expected_keys <= set(baseline_metrics):
            raise RuntimeError(f"baseline is missing {detector} metrics")
        for key in sorted(expected_keys):
            rows.append(
                effect_row(
                    detector,
                    key,
                    {code: group_metrics[code][key] for code in GROUPS},
                    baseline_metrics[key],
                )
            )

    csv_path = output / "punet_2x2_detector_effects.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    primary = [
        row
        for row in rows
        if row["class"] == "Car"
        and row["metric"] == "3d_ap_r40"
        and row["difficulty"] == "moderate"
    ]
    lines = [
        "# PU-Net 256-frame 2x2 detector effects",
        "",
        "A=legacy normalization+old patch; B=fixed normalization+old patch; "
        "C=legacy normalization+local patch; D=fixed normalization+local patch.",
        "",
        "## Primary Car Moderate 3D AP_R40",
        "",
        "| Detector | Baseline | A | B | C | D | D-A | D-Baseline | Interaction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| "
            + " | ".join(
                fmt(value)
                for value in (
                    row["detector"],
                    row["baseline"],
                    row["A"],
                    row["B"],
                    row["C"],
                    row["D"],
                    row["D_minus_A"],
                    row["D_minus_baseline"],
                    row["interaction_D_minus_C_minus_B_plus_A"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Primary causal decomposition (AP points)",
            "",
            "| Detector | Norm at old B-A | Norm at local D-C | Patch at legacy C-A | Patch at fixed D-B | Norm main | Patch main | Interaction |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in primary:
        lines.append(
            "| "
            + " | ".join(
                fmt(value)
                for value in (
                    row["detector"],
                    row["normalization_at_old_patch_B_minus_A"],
                    row["normalization_at_local_patch_D_minus_C"],
                    row["patch_at_legacy_norm_C_minus_A"],
                    row["patch_at_fixed_norm_D_minus_B"],
                    row["normalization_main_effect"],
                    row["patch_main_effect"],
                    row["interaction_D_minus_C_minus_B_plus_A"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The baseline is the paired 256-frame original point cloud because "
            "this 2x2 study uses Line A (`original_x4_up`).",
            "",
            "## All AP_R40 results",
            "",
            "| Detector | Class | Metric | Difficulty | Baseline | A | B | C | D | D-A | D-Baseline |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                fmt(value)
                for value in (
                    row["detector"],
                    row["class"],
                    row["metric"],
                    row["difficulty"],
                    row["baseline"],
                    row["A"],
                    row["B"],
                    row["C"],
                    row["D"],
                    row["D_minus_A"],
                    row["D_minus_baseline"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Prediction-box changes versus original baseline",
            "",
            "Same-class Hungarian matching with rotated BEV IoU >= 0.5.",
            "",
            "| Detector | Group | Class | Baseline boxes | Group boxes | Kept | Lost | Added | Kept / baseline | Added / group |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for detector in ("pointrcnn", "centerpoint"):
        for code in GROUPS:
            box_payload = load_json(box_summary_path(root, detector, code))
            for aggregate in box_payload["aggregates"]:
                lines.append(
                    "| "
                    + " | ".join(
                        fmt(value)
                        for value in (
                            detector,
                            code,
                            aggregate["class"],
                            aggregate["old_boxes"],
                            aggregate["new_boxes"],
                            aggregate["kept"],
                            aggregate["lost"],
                            aggregate["added"],
                            aggregate["kept_fraction_of_old"],
                            aggregate["added_fraction_of_new"],
                        )
                    )
                    + " |"
                )
    lines.extend(
        [
            "",
            "## 3769-frame expansion gate",
            "",
            "**Decision: do not expand the current PU-Net D configuration to "
            "3769 frames.** D materially improves over A, which confirms both "
            "the adapter normalization defect and its interaction with patch "
            "locality. However, D remains far below the paired original baseline "
            "for Car Moderate 3D AP_R40 in both detectors. CenterPoint Cyclist "
            "Moderate/Hard also does not improve consistently from B to D. The "
            "predeclared 'enough space and good performance' gate is therefore "
            "not satisfied.",
            "",
            "PointRCNN uses the frozen Car-only checkpoint/config; CenterPoint "
            "provides Car, Pedestrian, and Cyclist metrics.",
            "",
            f"Detailed effects: `{csv_path.name}`",
            "",
        ]
    )
    report_path = output / "PUNET_2X2_DETECTOR_EFFECTS.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(rows), "csv": str(csv_path), "report": str(report_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
