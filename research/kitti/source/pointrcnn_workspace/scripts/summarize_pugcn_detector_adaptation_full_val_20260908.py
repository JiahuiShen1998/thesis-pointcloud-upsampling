#!/usr/bin/env python3
"""Aggregate the audited 3,769-frame PU-GCN detector-adaptation matrix."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/pugcn_detector_adaptation_full_val_20260908"
EXPECTED_FRAMES = 3769


POINT_ARMS = (
    ("line_a_baseline_official", "A", "baseline N", "official"),
    ("line_a_pugcn_direct_official", "A", "PU-GCN direct 4N", "official"),
    ("line_a_pugcn_direct_adapted", "A", "PU-GCN direct 4N", "adapted"),
    ("line_a_pugcn_observed_first_official", "A", "observed N + predicted 3N", "official"),
    ("line_a_pugcn_observed_first_adapted", "A", "observed N + predicted 3N", "adapted"),
    ("line_b_baseline_official", "B", "baseline M", "official"),
    ("line_b_baseline_adapted", "B", "baseline M", "adapted"),
    ("line_b_pugcn_direct_official", "B", "PU-GCN direct 4M", "official"),
    ("line_b_pugcn_direct_adapted", "B", "PU-GCN direct 4M", "adapted"),
    ("line_b_pugcn_observed_first_official", "B", "observed M + predicted 3M", "official"),
    ("line_b_pugcn_observed_first_adapted", "B", "observed M + predicted 3M", "adapted"),
)

CENTER_ARMS = (
    ("line_a_baseline_official", "A", "baseline N", "official"),
    ("line_a_pugcn_direct_official", "A", "PU-GCN direct 4N", "official"),
    ("line_a_pugcn_observed_first_official", "A", "observed N + predicted 3N", "official"),
    ("line_a_pugcn_observed_first_adapted", "A", "observed N + predicted 3N", "adapted"),
    ("line_b_baseline_official", "B", "baseline M", "official"),
    ("line_b_baseline_adapted", "B", "baseline M", "adapted"),
    ("line_b_pugcn_direct_official", "B", "PU-GCN direct 4M", "official"),
    ("line_b_pugcn_observed_first_official", "B", "observed M + predicted 3M", "official"),
    ("line_b_pugcn_observed_first_adapted", "B", "observed M + predicted 3M", "adapted"),
)


def load_arm(detector: str, spec: tuple[str, str, str, str]) -> dict[str, object]:
    name, line, input_name, weights = spec
    if detector == "PointRCNN":
        path = RESULT / "evaluations/pointrcnn" / name / "run_complete.json"
        frame_key = "frame_count"
    else:
        path = RESULT / "evaluations/centerpoint" / ("fullval3769_20260908_" + name) / "result_summary.json"
        frame_key = "frames"
    row: dict[str, object] = {
        "detector": detector,
        "name": name,
        "line": line,
        "input": input_name,
        "weights": weights,
        "result_json": str(path),
        "status": "MISSING",
        "frame_count": 0,
        "bev_easy": "",
        "bev_moderate": "",
        "bev_hard": "",
        "3d_easy": "",
        "3d_moderate": "",
        "3d_hard": "",
        "reference": "",
        "delta_3d_moderate": "",
    }
    if not path.is_file():
        return row
    payload = json.loads(path.read_text(encoding="utf-8"))
    count = int(payload.get(frame_key, 0))
    status = str(payload.get("status", "UNKNOWN"))
    if status != "PASS" or count != EXPECTED_FRAMES:
        row.update(status="INVALID", frame_count=count)
        return row
    if detector == "PointRCNN":
        if payload.get("actual_consumed_frame_count") != EXPECTED_FRAMES or payload.get("ap_protocol") != "KITTI_AP_R40_41_precision_samples_exclude_recall_zero":
            row.update(status="INVALID_INPUT_OR_AP_PROTOCOL", frame_count=count)
            return row
    else:
        result_files = sorted((path.parent / "openpcdet_output").rglob("result.pkl"))
        if not result_files:
            row.update(status="MISSING_PREDICTIONS", frame_count=count)
            return row
        with result_files[-1].open("rb") as handle:
            predictions = pickle.load(handle)
        actual_ids = [str(item["frame_id"]) for item in predictions]
        expected_ids = [line.strip() for line in (ROOT / "data/KITTI/ImageSets/val.txt").read_text().splitlines() if line.strip()]
        if len(actual_ids) != EXPECTED_FRAMES or set(actual_ids) != set(expected_ids):
            row.update(status="INVALID_PREDICTION_COVERAGE", frame_count=len(actual_ids))
            return row
    car = payload["metrics_percent"]["Car"]
    for metric in ("bev_ap_r40", "3d_ap_r40"):
        for difficulty in ("easy", "moderate", "hard"):
            row[f"{metric.removesuffix('_ap_r40')}_{difficulty}"] = float(
                car[metric][difficulty]
            )
    row.update(status="PASS", frame_count=count)
    return row


def attach_deltas(rows: list[dict[str, object]]) -> None:
    by_key = {(row["detector"], row["name"]): row for row in rows}
    for row in rows:
        if row["status"] != "PASS":
            continue
        if row["line"] == "A":
            reference_name = "line_a_baseline_official"
        elif row["weights"] == "adapted":
            reference_name = "line_b_baseline_adapted"
        else:
            reference_name = "line_b_baseline_official"
        reference = by_key.get((row["detector"], reference_name))
        row["reference"] = reference_name
        if reference and reference["status"] == "PASS":
            row["delta_3d_moderate"] = round(
                float(row["3d_moderate"]) - float(reference["3d_moderate"]), 4
            )


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# PU-GCN detector adaptation: full KITTI validation",
        "",
        f"Required frame count per arm: **{EXPECTED_FRAMES}**.",
        "",
        "| Detector | Line | Input | Weights | Frames | Car 3D AP_R40 E/M/H | Δ moderate | Status |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["status"] == "PASS":
            triplet = "/".join(
                f"{float(row[key]):.4f}" for key in ("3d_easy", "3d_moderate", "3d_hard")
            )
            delta = f"{float(row['delta_3d_moderate']):+.4f}" if row["delta_3d_moderate"] != "" else "n/a"
        else:
            triplet = "pending"
            delta = "pending"
        lines.append(
            f"| {row['detector']} | {row['line']} | {row['input']} | {row['weights']} "
            f"| {row['frame_count']} | {triplet} | {delta} | {row['status']} |"
        )
    complete = sum(row["status"] == "PASS" for row in rows)
    lines.extend(
        [
            "",
            f"Complete arms: **{complete}/{len(rows)}**.",
            "",
            "For Line A adapted rows, no separately adapted original-N baseline was trained; "
            "their displayed delta therefore references the official Line A baseline and changes "
            "both input and weights. Other deltas use the same detector-weight family when available.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = [load_arm("PointRCNN", spec) for spec in POINT_ARMS]
    rows += [load_arm("CenterPoint", spec) for spec in CENTER_ARMS]
    attach_deltas(rows)
    by_name = {(row["detector"], row["name"]): row for row in rows}
    for row in rows:
        if row["detector"] != "PointRCNN" or row["weights"] != "adapted" or row["status"] != "PASS":
            continue
        official = by_name.get(("PointRCNN", str(row["name"]).replace("_adapted", "_official")))
        if official and official["status"] == "PASS":
            audit = Path(str(row["result_json"])).parent / "detector_frame_input_audit.json"
            reference_audit = Path(str(official["result_json"])).parent / "detector_frame_input_audit.json"
            if json.loads(audit.read_text()) != json.loads(reference_audit.read_text()):
                row["status"] = "INVALID_PAIRED_INPUT_MISMATCH"
    report_dir = RESULT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "full_val_detector_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = report_dir / "full_val_detector_matrix.json"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    write_markdown(rows, report_dir / "full_val_detector_matrix.md")
    long_rows = []
    for row in rows:
        if row["status"] != "PASS":
            continue
        payload = json.loads(Path(str(row["result_json"])).read_text())
        for category, category_metrics in payload["metrics_percent"].items():
            for metric, triplet in category_metrics.items():
                long_rows.append({
                    "detector": row["detector"], "arm": row["name"],
                    "frames": EXPECTED_FRAMES, "class": category, "metric": metric,
                    **triplet,
                })
    if long_rows:
        with (report_dir / "all_classes_ap_r40.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(long_rows[0]))
            writer.writeheader()
            writer.writerows(long_rows)
    complete = sum(row["status"] == "PASS" for row in rows)
    print(f"FULL_VAL_MATRIX_COMPLETE={complete}/{len(rows)}")
    return 0 if complete == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
