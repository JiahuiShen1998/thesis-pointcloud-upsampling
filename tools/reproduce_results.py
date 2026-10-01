#!/usr/bin/env python3
"""Rebuild result tables from archived evidence using only the Python standard library.

This recomputes table aggregates and deltas. It does not rerun training, predict
KITTI boxes, or reconstruct geometric distances from the omitted full datasets.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MN = Path("research/modelnet40/results")
KT = Path("research/kitti/results")


class EvidenceError(ValueError):
    """Archived evidence is missing, incomplete, or internally inconsistent."""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def close(actual, expected, context, tolerance=1e-9):
    require(
        math.isfinite(actual) and math.isfinite(expected),
        f"{context}: non-finite metric",
    )
    require(
        abs(actual - expected) <= tolerance,
        f"{context}: computed {actual}, recorded {expected}",
    )


def read_csv(root, relative, sources):
    path = root / relative
    sources.add(relative.as_posix())
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_json(root, relative, sources):
    sources.add(relative.as_posix())
    return json.loads((root / relative).read_text(encoding="utf-8"))


def modelnet_table(root, sources):
    runs = MN / "pointnet2_final_runs"
    metrics_paths = sorted((root / runs).rglob("metrics.json"))
    require(
        len(metrics_paths) == 14,
        f"Expected 14 final/control classifier runs, found {len(metrics_paths)}",
    )
    items = []
    for path in metrics_paths:
        relative = path.relative_to(root)
        m = read_json(root, relative, sources)
        log_path = path.with_name("train.log")
        sources.add(log_path.relative_to(root).as_posix())
        log = log_path.read_text(encoding="utf-8")
        epochs = re.findall(
            r"Epoch (\d+)/200 train_acc=([\d.]+) test_overall=([\d.]+) test_class=([\d.]+)",
            log,
        )
        require(
            [int(e[0]) for e in epochs] == list(range(1, 201)),
            f"{relative}: incomplete/duplicated epoch log",
        )
        require(
            m["final_epoch"] == 200 and m["allow_resample"] is False,
            f"{relative}: wrong final protocol",
        )
        for setting in ("seed=42", "epoch=200", "allow_resample=False"):
            require(setting in log, f"{relative}: missing recorded setting {setting}")
        close(
            float(epochs[-1][2]),
            m["final_test_overall_accuracy"],
            str(relative) + " final log OA",
            5.01e-5,
        )
        close(
            max(float(e[2]) for e in epochs),
            m["overall_accuracy"],
            str(relative) + " best log OA",
            5.01e-5,
        )
        require(1 <= m["best_epoch"] <= 200, f"{relative}: invalid best epoch")
        close(
            float(epochs[m["best_epoch"] - 1][2]),
            m["overall_accuracy"],
            str(relative) + " selected log OA",
            5.01e-5,
        )
        close(
            float(epochs[m["best_epoch"] - 1][3]),
            m["class_accuracy"],
            str(relative) + " selected log mAcc",
            5.01e-5,
        )
        close(
            float(epochs[-1][3]),
            m["final_test_class_accuracy"],
            str(relative) + " final log mAcc",
            5.01e-5,
        )
        run = path.parent.relative_to(root / runs).as_posix()
        line = (
            "A"
            if run.startswith("lineA")
            else "B"
            if run.startswith("lineB")
            else "control"
        )
        method = {
            "ear": "EAR",
            "pdans": "PDANS",
            "pu_net": "PU-Net",
            "pu_gcn": "PU-GCN",
            "pu_edgeformer": "PU-EdgeFormer",
        }.get(path.parent.name)
        method = method or (
            "Original"
            if line == "A"
            else "Sparse"
            if line == "B"
            else "Mesh-ref " + str(m["num_point"])
        )
        items.append(
            {
                "line": line,
                "method": method,
                "points": m["num_point"],
                "best_oa_pct": m["overall_accuracy"] * 100,
                "final_oa_pct": m["final_test_overall_accuracy"] * 100,
                "best_macc_pct": m["class_accuracy"] * 100,
                "best_epoch": m["best_epoch"],
                "seed": 42,
                "epochs": 200,
                "metrics_source": relative.as_posix(),
                "run": run,
            }
        )
    baseline = {
        row["line"]: row["best_oa_pct"]
        for row in items
        if row["method"] in {"Original", "Sparse"}
    }
    for row in items:
        reference = (
            "A"
            if row["line"] == "control" and row["points"] == 4096
            else "B"
            if row["line"] == "control"
            else row["line"]
        )
        row["delta_vs_reference_pp"] = row["best_oa_pct"] - baseline[reference]
        row["reference"] = "Original 1024" if reference == "A" else "Sparse 256"
    published = read_csv(
        root,
        MN
        / "reports/modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.csv",
        sources,
    )
    by_run = {row["run"]: row for row in items}
    for row in published:
        run = (
            row["metrics_json_path"]
            .split("/x4_two_line_final/")[1]
            .removesuffix("/metrics.json")
        )
        own = by_run[run]
        close(
            own["best_oa_pct"],
            float(row["best_overall_accuracy"]) * 100,
            run + " published best OA",
        )
        close(
            own["final_oa_pct"],
            float(row["final_test_overall_accuracy"]) * 100,
            run + " published final OA",
        )
        key = (
            "delta_accuracy_vs_original_baseline_pp"
            if row["line"] == "A"
            else "delta_accuracy_vs_downsampled_baseline_pp"
        )
        close(own["delta_vs_reference_pp"], float(row[key]), run + " published delta")
    require(len(published) == 12, "Expected 12 main classification rows")
    return items


def geometry_table(root, sources):
    per_sample = read_csv(
        root, MN / "reports/modelnet40_geometry_equal_n_per_sample.csv", sources
    )
    published = read_csv(
        root, MN / "reports/modelnet40_geometry_equal_n_summary.csv", sources
    )
    groups = collections.defaultdict(list)
    for row in per_sample:
        require(
            row["ok"] == "True" and row["split"] == "test",
            "Geometry evidence contains failed/non-test rows",
        )
        groups[(row["line"], row["method"])].append(row)
    require(len(groups) == 14, "Expected 14 geometry groups")
    keys = [(s["line"], s["method_key"]) for s in published]
    require(
        len(keys) == 14 and len(set(keys)) == 14 and set(keys) == set(groups),
        "Missing/duplicated geometry summary groups",
    )
    summaries = []
    for summary in published:
        group = groups[(summary["line"], summary["method_key"])]
        require(
            len(group) == 2468 and len({r["shape_id"] for r in group}) == 2468,
            "Missing/duplicated geometry shapes",
        )
        require(
            int(summary["valid_samples"]) == len(group),
            "Geometry sample count disagrees with summary",
        )
        require(
            all(
                int(r["point_count"]) == int(summary["points"])
                and r["reference"] == summary["reference"]
                for r in group
            ),
            "Geometry reference/count mismatch",
        )
        out = {
            "line": summary["line"],
            "method": summary["method"],
            "points": int(summary["points"]),
            "reference": summary["reference"],
            "split": "test",
            "samples": len(group),
        }
        for field, summary_field in [
            ("cd", "cd_vs_ref"),
            ("hd", "hd_vs_ref"),
            ("nuc", "nuc"),
        ]:
            value = statistics.fmean(float(r[field]) for r in group)
            close(
                value,
                float(summary[summary_field]),
                f"{out['line']} {out['method']} {field}",
            )
            out[field] = value
        summaries.append(out)
    return summaries


def kitti_table(root, sources):
    pr = read_json(root, KT / "convergence/final_comparison.json", sources)
    cp = read_json(
        root, KT / "convergence/centerpoint_convergence_summary.json", sources
    )
    cp_curve = read_csv(
        root, KT / "convergence/centerpoint_epoch_validation_curve.csv", sources
    )
    matrix = read_csv(
        root, KT / "latest_full_validation/full_val_detector_matrix.csv", sources
    )
    lookup = {(r["detector"], r["name"]): r for r in matrix}
    items = []
    for detector, records in [("PointRCNN", pr["phases"]), ("CenterPoint", cp["arms"])]:
        for line in ("A", "B"):
            arm = f"line_{line.lower()}_pugcn_observed_first"
            record = records[arm + "/rcnn"] if detector == "PointRCNN" else records[arm]
            if detector == "PointRCNN":
                curve = record["convergence_trace"]
                require(
                    [r["epoch"] for r in curve]
                    == list(range(1, record["expected_epochs"] + 1)),
                    "Incomplete PointRCNN curve",
                )
                frames = record["validation_frames_per_epoch"]
            else:
                curve = [
                    {"epoch": int(r["epoch"]), "ap": float(r["car_3d_ap_r40_moderate"])}
                    for r in cp_curve
                    if r["arm"] == arm
                ]
                require(
                    [r["epoch"] for r in curve] == list(range(1, 13)),
                    "Incomplete CenterPoint curve",
                )
                require(
                    all(int(r["frames"]) == 3769 for r in cp_curve if r["arm"] == arm),
                    "Wrong CenterPoint split",
                )
                frames = cp["validation_frames_per_epoch"]
            require(
                frames == 3769, "KITTI result does not cover the full validation set"
            )
            best = max(curve, key=lambda r: float(r["ap"]))
            close(
                float(best["ap"]),
                record["global_best_ap"],
                f"{detector} {line} curve maximum",
            )
            require(
                best["epoch"] == record["global_best_epoch"],
                "Best epoch disagrees with curve",
            )
            reference_name = f"line_{line.lower()}_baseline_" + (
                "official" if line == "A" else "adapted"
            )
            for name in (reference_name, arm + "_official"):
                require(
                    lookup[(detector, name)]["status"] == "PASS"
                    and int(lookup[(detector, name)]["frame_count"]) == 3769,
                    "KITTI baseline does not cover the full validation set",
                )
            baseline = float(lookup[(detector, reference_name)]["3d_moderate"])
            unadapted = float(lookup[(detector, arm + "_official")]["3d_moderate"])
            close(baseline, record["baseline_ap"], f"{detector} {line} baseline")
            close(
                unadapted,
                record["unadapted_same_input_ap"],
                f"{detector} {line} unadapted",
            )
            items.append(
                {
                    "detector": detector,
                    "line": line,
                    "metric": "Car 3D Moderate AP_R40",
                    "validation_frames": frames,
                    "unadapted_same_input_ap": unadapted,
                    "best_extended_ap": float(best["ap"]),
                    "best_epoch": best["epoch"],
                    "reference_baseline_ap": baseline,
                    "gain_vs_unadapted_ap_points": float(best["ap"]) - unadapted,
                    "gap_to_baseline_ap_points": float(best["ap"]) - baseline,
                    "reference": record["baseline_label"],
                }
            )
    return items


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(title for _, title in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = [
            f"{row[key]:.4f}" if isinstance(row[key], float) else str(row[key])
            for key, _ in columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def reproduce(root, output):
    sources = set()
    mn = modelnet_table(root, sources)
    geometry = geometry_table(root, sources)
    kt = kitti_table(root, sources)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("modelnet40_classification", mn),
        ("modelnet40_geometry", geometry),
        ("kitti_adaptation", kt),
    ]:
        write_csv(output / (name + ".csv"), rows)
    report = "# Rebuilt archive results\n\nComputed from archived run metrics, logs, per-shape geometry, and detector curves. No model was trained and no detector predictions or geometric distances were recomputed.\n\n"
    report += "## ModelNet40 classification\n\nOA and mAcc retain the archived batch-average implementation. Best checkpoints were selected on the test set, with one seed (42).\n\n"
    report += markdown_table(
        mn,
        [
            ("line", "Line"),
            ("method", "Method"),
            ("points", "Points"),
            ("best_oa_pct", "Best OA (%)"),
            ("delta_vs_reference_pp", "Delta (pp)"),
            ("reference", "Reference"),
        ],
    )
    report += "\n\n## KITTI detector adaptation\n\nFull 3,769-frame validation, Car 3D Moderate AP_R40. Gains compare the same upsampled input before and after adaptation. Reference-baseline training budgets differ.\n\n"
    report += (
        markdown_table(
            kt,
            [
                ("detector", "Detector"),
                ("line", "Line"),
                ("unadapted_same_input_ap", "Unadapted"),
                ("best_extended_ap", "Extended"),
                ("gain_vs_unadapted_ap_points", "Gain (AP points)"),
                ("gap_to_baseline_ap_points", "Baseline gap"),
            ],
        )
        + "\n"
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "scope": "reaggregation of archived evidence, not full experimental rerun",
        "modelnet40_classifier_runs": len(mn),
        "modelnet40_geometry_groups": len(geometry),
        "geometry_samples_per_group": 2468,
        "kitti_comparisons": len(kt),
        "checks": "all passed",
        "sources": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sorted(sources)
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/reproduced")
    args = parser.parse_args()
    try:
        result = reproduce(ROOT, args.output_dir.resolve())
    except (OSError, KeyError, ValueError) as error:
        print(f"Evidence validation failed: {error}", file=sys.stderr)
        return 1
    print(
        f"PASS: {result['modelnet40_classifier_runs']} classifier runs, {result['modelnet40_geometry_groups']} geometry groups, {result['kitti_comparisons']} detector comparisons"
    )
    print(f"Report: {args.output_dir.resolve() / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
