#!/usr/bin/env python3
"""Aggregate full-validation and selected-frame detector transition evidence."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO / "results/dual_detector_three_frame_root_cause_20260730"
)
STATUSES = (
    "maintained",
    "confidence_degraded",
    "localization_degraded",
    "lost_after_upsampling",
    "recovered_after_upsampling",
    "missed_both",
)
METHODS = ("PDANS", "PU-GCN", "PU-EdgeFormer", "PU-Net")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def distance_bin(depth: float) -> str:
    if depth < 20.0:
        return "0-20m"
    if depth < 40.0:
        return "20-40m"
    return "40m+"


def aggregate(rows: list[dict[str, str]], include_distance: bool) -> list[dict]:
    groups = defaultdict(Counter)
    for row in rows:
        key = (
            row["detector"],
            row["line"],
            row["method"],
            row["class"],
        )
        if include_distance:
            key += (distance_bin(float(row["gt_center_x_m"])),)
        groups[key][row["transition"]] += 1
    output = []
    for key, counts in sorted(groups.items()):
        total = sum(counts.values())
        baseline_tp = (
            counts["maintained"]
            + counts["confidence_degraded"]
            + counts["localization_degraded"]
            + counts["lost_after_upsampling"]
        )
        upsampled_tp = (
            counts["maintained"]
            + counts["confidence_degraded"]
            + counts["localization_degraded"]
            + counts["recovered_after_upsampling"]
        )
        row = {
            "detector": key[0],
            "line": key[1],
            "method": key[2],
            "class": key[3],
        }
        if include_distance:
            row["distance_bin"] = key[4]
        row.update({status: counts[status] for status in STATUSES})
        row.update(
            {
                "gt_opportunities": total,
                "baseline_tp": baseline_tp,
                "upsampled_tp": upsampled_tp,
                "net_tp_change": upsampled_tp - baseline_tp,
                "lost_rate_of_baseline_tp": (
                    counts["lost_after_upsampling"] / baseline_tp
                    if baseline_tp
                    else ""
                ),
                "recovery_rate_of_baseline_fn": (
                    counts["recovered_after_upsampling"] / (total - baseline_tp)
                    if total > baseline_tp
                    else ""
                ),
                "localization_degradation_rate_of_shared_tp": (
                    counts["localization_degraded"]
                    / max(
                        1,
                        counts["maintained"]
                        + counts["confidence_degraded"]
                        + counts["localization_degraded"],
                    )
                ),
            }
        )
        output.append(row)
    return output


def selected_case_summary(rows: list[dict[str, str]]) -> list[dict]:
    groups = defaultdict(Counter)
    for row in rows:
        key = (
            row["frame_id"],
            row["detector"],
            row["line"],
            row["method"],
            row["class"],
        )
        groups[key][row["transition"]] += 1
    output = []
    for key, counts in sorted(groups.items()):
        output.append(
            {
                "frame_id": key[0],
                "detector": key[1],
                "line": key[2],
                "method": key[3],
                "class": key[4],
                **{status: counts[status] for status in STATUSES},
                "lost_minus_recovered": (
                    counts["lost_after_upsampling"]
                    - counts["recovered_after_upsampling"]
                ),
            }
        )
    return output


def plot_lost_recovered(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for row_idx, detector in enumerate(("pointrcnn", "centerpoint")):
        for col_idx, line in enumerate(("A", "B")):
            ax = axes[row_idx, col_idx]
            classes = ["Car"] if detector == "pointrcnn" else ["Car", "Pedestrian", "Cyclist"]
            labels, lost, recovered = [], [], []
            for method in METHODS:
                for class_name in classes:
                    row = next(
                        item
                        for item in rows
                        if item["detector"] == detector
                        and item["line"] == line
                        and item["method"] == method
                        and item["class"] == class_name
                    )
                    labels.append(
                        method if len(classes) == 1 else f"{method}\n{class_name[:3]}"
                    )
                    lost.append(row["lost_after_upsampling"])
                    recovered.append(row["recovered_after_upsampling"])
            x = np.arange(len(labels))
            ax.bar(x, lost, color="#D62728", label="lost after upsampling")
            ax.bar(x, [-value for value in recovered], color="#00A6A6", label="recovered")
            ax.axhline(0, color="#444444", linewidth=0.7)
            ax.set_xticks(x, labels, rotation=25 if len(labels) > 4 else 10, fontsize=7)
            ax.set_ylabel("GT transitions across 3,769 frames")
            ax.set_title(f"{detector} — Line {line}")
            ax.grid(axis="y", linewidth=0.35, alpha=0.35)
            ax.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def plot_distance_loss(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    bins = ("0-20m", "20-40m", "40m+")
    for row_idx, detector in enumerate(("pointrcnn", "centerpoint")):
        for col_idx, line in enumerate(("A", "B")):
            ax = axes[row_idx, col_idx]
            for method in METHODS:
                method_rows = [
                    row
                    for row in rows
                    if row["detector"] == detector
                    and row["line"] == line
                    and row["method"] == method
                    and row["class"] == "Car"
                ]
                values = []
                for bin_name in bins:
                    match = next(
                        (row for row in method_rows if row["distance_bin"] == bin_name),
                        None,
                    )
                    values.append(
                        100.0 * float(match["lost_rate_of_baseline_tp"])
                        if match and match["lost_rate_of_baseline_tp"] != ""
                        else np.nan
                    )
                ax.plot(bins, values, marker="o", linewidth=1.4, label=method)
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Lost GT / baseline TP (%)")
            ax.set_title(f"{detector} — Line {line} Car")
            ax.grid(True, linewidth=0.35, alpha=0.35)
            ax.legend(frameon=False, fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    full_rows = read_csv(args.root / "analysis/all_frame_gt_transitions.csv")
    selected_rows = read_csv(args.root / "analysis/selected_frame_gt_transitions.csv")
    full_summary = aggregate(full_rows, include_distance=False)
    distance_summary = aggregate(full_rows, include_distance=True)
    selected_summary = selected_case_summary(selected_rows)
    write_csv(args.root / "analysis/transition_summary_full_val.csv", full_summary)
    write_csv(args.root / "analysis/transition_summary_by_distance.csv", distance_summary)
    write_csv(args.root / "analysis/selected_frame_case_summary.csv", selected_summary)
    plot_lost_recovered(
        full_summary, args.root / "figures/full_val_lost_vs_recovered_gt.png"
    )
    plot_distance_loss(
        distance_summary, args.root / "figures/full_val_car_loss_by_distance.png"
    )
    print(f"full_transition_groups={len(full_summary)}")
    print(f"distance_groups={len(distance_summary)}")
    print(f"selected_case_groups={len(selected_summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
