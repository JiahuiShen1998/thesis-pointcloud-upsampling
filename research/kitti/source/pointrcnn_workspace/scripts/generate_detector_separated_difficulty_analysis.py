#!/usr/bin/env python3
"""Build detector-separated, class-separated Easy/Moderate/Hard analyses.

This script is read-only with respect to all existing experiment outputs.  It
consumes the already-audited full-validation AP/transition tables plus the
PointRCNN 256-frame generated-ratio control experiment and writes a new report
package.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = REPO / "results/dual_detector_three_frame_root_cause_20260730"
OUTPUT = REPO / "results/detector_separated_difficulty_root_cause_20260730"

AP_SOURCE = SOURCE_PACKAGE / "analysis/all_detector_ap_r40_bbox_bev_3d.csv"
TRANSITION_SOURCE = SOURCE_PACKAGE / "analysis/transition_summary_full_val.csv"
DISTANCE_SOURCE = SOURCE_PACKAGE / "analysis/transition_summary_by_distance.csv"
RATIO_SOURCE = (
    REPO
    / "results/kitti_x4_detector_recovery_combined_analysis_20260725"
    / "combined_all_difficulty_metrics.csv"
)
CENTERPOINT_BEHAVIOR_SOURCE = (
    REPO
    / "results/centerpoint_input_mechanism_analysis_20260729"
    / "detector_behavior_summary.csv"
)
CENTERPOINT_VOXEL_SOURCE = (
    REPO
    / "results/centerpoint_input_mechanism_analysis_20260729"
    / "centerpoint_voxel_summary.csv"
)

DIFFICULTIES = ("easy", "moderate", "hard")
DIFFICULTY_LABELS = ("Easy", "Moderate", "Hard")
METRICS = ("bbox", "bev", "3d")
METRIC_LABELS = {"bbox": "BBox", "bev": "BEV", "3d": "3D", "aos": "AOS"}
METHODS = ("baseline", "PDANS", "PU-GCN", "PU-EdgeFormer", "PU-Net")
UPSAMPLERS = METHODS[1:]
METHOD_COLORS = {
    "baseline": "#4c4c4c",
    "PDANS": "#2878b5",
    "PU-GCN": "#39a96b",
    "PU-EdgeFormer": "#ef8a17",
    "PU-Net": "#c44e52",
    "Observed-fill control": "#7b61a8",
}
DIFFICULTY_COLORS = ("#4c78a8", "#f28e2b", "#59a14f")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return math.nan
    return float(value)


def normalize_method(value: str) -> str:
    mapping = {
        "baseline": "baseline",
        "pdans": "PDANS",
        "pu_gcn": "PU-GCN",
        "pu-gcn": "PU-GCN",
        "PU-GCN": "PU-GCN",
        "pu_edgeformer": "PU-EdgeFormer",
        "pu-edgeformer": "PU-EdgeFormer",
        "PU-EdgeFormer": "PU-EdgeFormer",
        "pu_net": "PU-Net",
        "pu-net": "PU-Net",
        "PU-Net": "PU-Net",
        "PDANS": "PDANS",
    }
    return mapping.get(value, value)


def method_rows(
    rows: list[dict[str, str]], detector: str, class_name: str, line: str, metric: str
) -> dict[str, dict[str, str]]:
    selected: dict[str, dict[str, str]] = {}
    baseline_line = f"baseline_{line}"
    for row in rows:
        if (
            row["detector"] != detector
            or row["class"] != class_name
            or row["metric"] != metric
        ):
            continue
        if row["line"] == line:
            selected[normalize_method(row["method"])] = row
        elif row["line"] == baseline_line:
            selected["baseline"] = row
    return selected


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def formal_ap_figure(
    ap_rows: list[dict[str, str]],
    detector: str,
    class_name: str,
    line: str,
    delta: bool,
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
    plot_methods = UPSAMPLERS if delta else METHODS
    x = np.arange(len(plot_methods), dtype=float)
    width = 0.23
    for ax, metric in zip(axes, METRICS):
        selected = method_rows(ap_rows, detector, class_name, line, metric)
        for idx, (difficulty, label, color) in enumerate(
            zip(DIFFICULTIES, DIFFICULTY_LABELS, DIFFICULTY_COLORS)
        ):
            key = f"delta_{difficulty}" if delta else difficulty
            values = [
                as_float(selected[method][key]) if method in selected else math.nan
                for method in plot_methods
            ]
            bars = ax.bar(
                x + (idx - 1) * width,
                values,
                width,
                label=label,
                color=color,
                alpha=0.9,
            )
            for bar, value in zip(bars, values):
                if math.isnan(value):
                    continue
                offset = 0.6 if value >= 0 else -0.8
                va = "bottom" if value >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + offset,
                    f"{value:.1f}",
                    ha="center",
                    va=va,
                    fontsize=7,
                    rotation=90,
                )
        ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x)
        labels = (
            ("PDANS", "PU-GCN", "PU-EdgeFormer", "PU-Net")
            if delta
            else ("Baseline", "PDANS", "PU-GCN", "PU-EdgeFormer", "PU-Net")
        )
        ax.set_xticklabels(labels, rotation=22, ha="right")
        ax.margins(y=0.16)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel("Δ AP_R40 (points)" if delta else "AP_R40 (%)")
    axes[0].legend(loc="best", frameon=False)
    detector_label = "PointRCNN" if detector == "pointrcnn" else "CenterPoint"
    mode = "AP change from line baseline" if delta else "absolute AP"
    fig.suptitle(f"{detector_label} · {class_name} · Line {line} · {mode}", fontsize=15)
    fig.tight_layout()
    save_figure(fig, output)


def transition_figure(
    rows: list[dict[str, str]],
    detector: str,
    class_name: str,
    line: str,
    output: Path,
) -> None:
    selected = {
        normalize_method(row["method"]): row
        for row in rows
        if row["detector"] == detector
        and row["class"] == class_name
        and row["line"] == line
    }
    methods = [method for method in UPSAMPLERS if method in selected]
    x = np.arange(len(methods), dtype=float)
    lost = [
        100 * as_float(selected[method]["lost_rate_of_baseline_tp"]) for method in methods
    ]
    recovered = [
        100 * as_float(selected[method]["recovery_rate_of_baseline_fn"])
        for method in methods
    ]
    net = [int(float(selected[method]["net_tp_change"])) for method in methods]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    width = 0.35
    axes[0].bar(x - width / 2, lost, width, label="Lost / baseline TP", color="#c44e52")
    axes[0].bar(
        x + width / 2,
        recovered,
        width,
        label="Recovered / baseline FN",
        color="#2a9d8f",
    )
    axes[0].set_ylabel("Rate (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_title("Lost and recovered rates")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    net_colors = ["#2a9d8f" if value >= 0 else "#c44e52" for value in net]
    bars = axes[1].bar(x, net, color=net_colors)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_ylabel("Upsampled TP − baseline TP")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_title("Net matched-GT change")
    axes[1].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, net):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:+d}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    detector_label = "PointRCNN" if detector == "pointrcnn" else "CenterPoint"
    fig.suptitle(
        f"{detector_label} · {class_name} · Line {line} · full-validation transitions",
        fontsize=14,
    )
    fig.tight_layout()
    save_figure(fig, output)


def distance_figure(
    rows: list[dict[str, str]],
    detector: str,
    class_name: str,
    line: str,
    output: Path,
) -> None:
    lookup = {
        (normalize_method(row["method"]), row["distance_bin"]): row
        for row in rows
        if row["detector"] == detector
        and row["class"] == class_name
        and row["line"] == line
    }
    bins = ("0-20m", "20-40m", "40m+")
    x = np.arange(len(bins), dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for method in UPSAMPLERS:
        values = [
            100 * as_float(lookup[(method, distance)]["lost_rate_of_baseline_tp"])
            if (method, distance) in lookup
            else math.nan
            for distance in bins
        ]
        ax.plot(
            x,
            values,
            marker="o",
            linewidth=2,
            label=method,
            color=METHOD_COLORS[method],
        )
        for xpos, value in zip(x, values):
            if not math.isnan(value):
                ax.text(xpos, value + 1.2, f"{value:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_ylabel("Lost / baseline TP (%)")
    ax.set_xlabel("GT distance")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    detector_label = "PointRCNN" if detector == "pointrcnn" else "CenterPoint"
    ax.set_title(f"{detector_label} · {class_name} · Line {line} · loss by distance")
    fig.tight_layout()
    save_figure(fig, output)


def ratio_metric_value(row: dict[str, str], metric: str, difficulty: str) -> float:
    return as_float(row[f"ap_{metric}_car_{difficulty}_r40"])


def positive_ratio_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if row["line"] != "original":
            continue
        ratio = as_float(row["ratio_percent"])
        if row["row_kind"] == "baseline_reference":
            selected.append(row)
        elif row["protocol"] == "fine_nested_e2_replacement" and ratio <= 10:
            selected.append(row)
    return selected


def positive_attribution(rows: list[dict[str, str]]) -> list[dict]:
    baseline = next(
        row
        for row in rows
        if row["line"] == "original" and row["row_kind"] == "baseline_reference"
    )
    pdans = next(
        row
        for row in rows
        if row["line"] == "original"
        and normalize_method(row["method"]) == "PDANS"
        and row["row_kind"] == "generated_ratio"
        and abs(as_float(row["ratio_percent"]) - 2.5) < 1e-6
    )
    control = next(
        row
        for row in rows
        if row["line"] == "original"
        and row["row_kind"] == "observed_fill_control"
        and abs(as_float(row["ratio_percent"]) - 2.5) < 1e-6
    )
    output = []
    for metric in ("bbox", "bev", "3d", "aos"):
        for difficulty in DIFFICULTIES:
            base_value = ratio_metric_value(baseline, metric, difficulty)
            control_value = ratio_metric_value(control, metric, difficulty)
            generated_value = ratio_metric_value(pdans, metric, difficulty)
            output.append(
                {
                    "class": "Car",
                    "metric": METRIC_LABELS[metric],
                    "difficulty": difficulty.capitalize(),
                    "baseline_ap": base_value,
                    "observed_fill_control_ap": control_value,
                    "pdans_g2p5_ap": generated_value,
                    "control_delta_vs_baseline": control_value - base_value,
                    "pdans_delta_vs_baseline": generated_value - base_value,
                    "pdans_delta_vs_control": generated_value - control_value,
                    "frames": 256,
                }
            )
    return output


def positive_attribution_figure(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    x = np.arange(3, dtype=float)
    width = 0.25
    labels = ("Baseline", "Observed control", "PDANS g2.5")
    colors = (
        METHOD_COLORS["baseline"],
        METHOD_COLORS["Observed-fill control"],
        METHOD_COLORS["PDANS"],
    )
    for ax, metric in zip(axes.ravel(), ("BBox", "BEV", "3D", "AOS")):
        metric_rows = [row for row in rows if row["metric"] == metric]
        for idx, (label, color) in enumerate(zip(labels, colors)):
            key = (
                "baseline_ap"
                if idx == 0
                else "observed_fill_control_ap"
                if idx == 1
                else "pdans_g2p5_ap"
            )
            values = [
                next(
                    float(row[key])
                    for row in metric_rows
                    if row["difficulty"] == difficulty
                )
                for difficulty in DIFFICULTY_LABELS
            ]
            bars = ax.bar(x + (idx - 1) * width, values, width, label=label, color=color)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.15,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(DIFFICULTY_LABELS)
        ax.set_ylabel("AP_R40 (%)")
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "PointRCNN · Car · Original line · 2.5% PDANS positive-control attribution",
        fontsize=15,
    )
    fig.tight_layout()
    save_figure(fig, output)


def fine_ratio_figure(
    rows: list[dict[str, str]], difficulty: str, output: Path
) -> None:
    selected = positive_ratio_rows(rows)
    baseline = next(row for row in selected if row["row_kind"] == "baseline_reference")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, metric in zip(axes, METRICS):
        baseline_value = ratio_metric_value(baseline, metric, difficulty)
        for method in UPSAMPLERS:
            method_key = method.lower().replace("-", "_")
            method_rows_ = [
                row
                for row in selected
                if normalize_method(row["method"]) == method
                and row["row_kind"] == "generated_ratio"
            ]
            method_rows_.sort(key=lambda row: as_float(row["ratio_percent"]))
            xs = [0.0] + [as_float(row["ratio_percent"]) for row in method_rows_]
            ys = [baseline_value] + [
                ratio_metric_value(row, metric, difficulty) for row in method_rows_
            ]
            ax.plot(
                xs,
                ys,
                marker="o",
                label=method,
                color=METHOD_COLORS[method],
                linewidth=2,
            )
        control_rows = [
            row for row in selected if row["row_kind"] == "observed_fill_control"
        ]
        control_rows.sort(key=lambda row: as_float(row["ratio_percent"]))
        control_x = [0.0] + [as_float(row["ratio_percent"]) for row in control_rows]
        control_y = [baseline_value] + [
            ratio_metric_value(row, metric, difficulty) for row in control_rows
        ]
        ax.plot(
            control_x,
            control_y,
            marker="s",
            linestyle="--",
            linewidth=1.8,
            label="Observed-fill control",
            color=METHOD_COLORS["Observed-fill control"],
        )
        ax.axhline(baseline_value, color="#555555", linewidth=0.8, linestyle=":")
        ax.set_xticks((0, 2.5, 5, 7.5, 10))
        ax.set_xlabel("Replaced slots with generated/control points (%)")
        ax.set_ylabel("AP_R40 (%)")
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        f"PointRCNN · Car · Original line · fine ratio · {difficulty.capitalize()}",
        fontsize=15,
    )
    fig.tight_layout()
    save_figure(fig, output)


def behavior_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["variant"]: row for row in rows}


def centerpoint_positive_rows(
    ap_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    output = []
    for row in ap_rows:
        if row["detector"] != "centerpoint" or row["line"] != "B":
            continue
        if row["class"] not in {"Pedestrian", "Cyclist"}:
            continue
        if any(as_float(row[f"delta_{difficulty}"]) > 0 for difficulty in DIFFICULTIES):
            output.append(row)
    return output


def centerpoint_positive_mechanism_figure(
    class_name: str,
    ap_rows: list[dict[str, str]],
    transition_rows: list[dict[str, str]],
    behavior_rows: list[dict[str, str]],
    output: Path,
) -> None:
    behavior = behavior_lookup(behavior_rows)
    baseline_behavior = behavior["downsampled_x4_baseline"]
    class_key = class_name.lower()
    methods = list(UPSAMPLERS)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # Panel 1: 3D delta for all difficulties.
    selected_3d = method_rows(ap_rows, "centerpoint", class_name, "B", "3d")
    x = np.arange(len(methods), dtype=float)
    width = 0.23
    for idx, (difficulty, label, color) in enumerate(
        zip(DIFFICULTIES, DIFFICULTY_LABELS, DIFFICULTY_COLORS)
    ):
        values = [
            as_float(selected_3d[method][f"delta_{difficulty}"]) for method in methods
        ]
        axes[0].bar(x + (idx - 1) * width, values, width, label=label, color=color)
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_ylabel("Δ 3D AP_R40")
    axes[0].set_title("3D AP change")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)
    # Panel 2: lost/recovered and net transitions.
    trans = {
        normalize_method(row["method"]): row
        for row in transition_rows
        if row["detector"] == "centerpoint"
        and row["line"] == "B"
        and row["class"] == class_name
    }
    lost = [int(float(trans[method]["lost_after_upsampling"])) for method in methods]
    recovered = [
        int(float(trans[method]["recovered_after_upsampling"])) for method in methods
    ]
    axes[1].bar(x - 0.18, lost, 0.36, label="Lost", color="#c44e52")
    axes[1].bar(x + 0.18, recovered, 0.36, label="Recovered", color="#2a9d8f")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=20, ha="right")
    axes[1].set_ylabel("Matched GT count")
    axes[1].set_title("Single-threshold GT transitions")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    # Panel 3: high-confidence prediction changes.
    baseline_high = int(
        float(baseline_behavior[f"{class_key}_score_ge_0p5_count"])
    )
    high_deltas = []
    for method in methods:
        variant = "downsampled_x4_" + method.lower().replace("-", "_")
        high = int(float(behavior[variant][f"{class_key}_score_ge_0p5_count"]))
        high_deltas.append(high - baseline_high)
    bars = axes[2].bar(
        x,
        high_deltas,
        color=["#2a9d8f" if value >= 0 else "#c44e52" for value in high_deltas],
    )
    axes[2].axhline(0, color="#555555", linewidth=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=20, ha="right")
    axes[2].set_ylabel("Δ predictions with score ≥ 0.5")
    axes[2].set_title("High-confidence prediction change")
    axes[2].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, high_deltas):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:+d}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    fig.suptitle(
        f"CenterPoint · {class_name} · Line B · positive-exception evidence",
        fontsize=15,
    )
    fig.tight_layout()
    save_figure(fig, output)


def format_triplet(row: dict[str, str], prefix: str = "") -> str:
    return " / ".join(f"{as_float(row[prefix + difficulty]):.2f}" for difficulty in DIFFICULTIES)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def formal_3d_table(
    ap_rows: list[dict[str, str]], detector: str, class_name: str, line: str
) -> str:
    selected = method_rows(ap_rows, detector, class_name, line, "3d")
    body = []
    for method in METHODS:
        if method not in selected:
            continue
        row = selected[method]
        deltas = "—" if method == "baseline" else format_triplet(row, "delta_")
        body.append([method.capitalize() if method == "baseline" else method, format_triplet(row), deltas])
    return markdown_table(
        ["Input", "3D AP_R40 Easy / Moderate / Hard", "Δ Easy / Moderate / Hard"],
        body,
    )


def formal_all_metric_tables(
    ap_rows: list[dict[str, str]], detector: str, class_name: str, line: str
) -> str:
    sections = []
    for metric in METRICS:
        selected = method_rows(ap_rows, detector, class_name, line, metric)
        body = []
        for method in METHODS:
            if method not in selected:
                continue
            row = selected[method]
            values = []
            for difficulty in DIFFICULTIES:
                value = as_float(row[difficulty])
                if method == "baseline":
                    values.append(f"{value:.2f}")
                else:
                    delta = as_float(row[f"delta_{difficulty}"])
                    values.append(f"{value:.2f} ({delta:+.2f})")
            body.append(
                [
                    method.capitalize() if method == "baseline" else method,
                    *values,
                ]
            )
        sections.append(
            f"#### {METRIC_LABELS[metric]}\n\n"
            + markdown_table(
                ["Input", "Easy AP (Δ)", "Moderate AP (Δ)", "Hard AP (Δ)"],
                body,
            )
        )
    return "\n\n".join(sections)


def point_report(
    ap_rows: list[dict[str, str]],
    attribution: list[dict],
) -> str:
    def attr(metric: str, difficulty: str) -> dict:
        return next(
            row
            for row in attribution
            if row["metric"] == metric and row["difficulty"] == difficulty
        )

    positive_rows = []
    for metric in ("BBox", "BEV", "3D", "AOS"):
        for difficulty in DIFFICULTY_LABELS:
            row = attr(metric, difficulty)
            positive_rows.append(
                [
                    metric,
                    difficulty,
                    f"{row['baseline_ap']:.2f}",
                    f"{row['observed_fill_control_ap']:.2f} ({row['control_delta_vs_baseline']:+.2f})",
                    f"{row['pdans_g2p5_ap']:.2f} ({row['pdans_delta_vs_baseline']:+.2f})",
                    f"{row['pdans_delta_vs_control']:+.2f}",
                ]
            )
    return f"""# PointRCNN  Independent analysis: Easy / Moderate / Hard,  It’s against the root cause.

## 1.  Analysis boundary

 The present report analyses only  PointRCNN,  Not with  CenterPoint  Common chart, common table or shared category conclusion. Current frozen  PointRCNN  Configure training and evaluation only  `Car`,  That’s why it doesn’t exist.  Pedestrian/Cyclist  Question of missing maps.

-  Official double-line results: KITTI val  All  3,769  frame, canonical E2 16,384  Point.
-  Optimization experiment: fixed  256  frame  Car  Subset, nested slot replacement; used to interpret mechanisms, cannot impersonates full validation set conclusion.
-  Indicators: BBox, BEV, 3D AP_R40,  Separate reports  Easy, Moderate, Hard.
-  According to the user  `diff/difficult`  Corresponding to the present report KITTI Official name  `Hard`.

## 2.  Official results: Car

### Line A: Original baseline → Original + x4 upsampling

{formal_all_metric_tables(ap_rows, "pointrcnn", "Car", "A")}

 Figure:

- `figures/car_line_a_absolute_ap.png`
- `figures/car_line_a_delta_ap.png`

 Difficulty conclusion:

1. PDANS  It’s...  3D  Down to  Easy `-7.66`, Moderate `-15.03`, Hard `-15.55`.  Medium / Difficult target ratio  Easy  Multiple losses approximately  7–8 AP,  Description that the remote, shielding and cut-off target is more sensitive to the error by adding neighbourhood.
2. PU-GCN  with  PU-EdgeFormer Yes.  Moderate/Hard  The decline is significantly greater than that.  Easy;  generated points was not the first to destroy a clearly near vehicle, but rather a difficult vehicle that had only point a small number of boundary.
3. PU-Net (a) All three difficulties are no longer valid; Easy It’s not going down much bigger.” Easy It’s the global geometric error caused by access deficiencies at the current coordinates normalization. AP Approaching floor limit.

### Line B: Downsampled-x4 baseline → Downsampled + x4 upsampling

{formal_all_metric_tables(ap_rows, "pointrcnn", "Car", "B")}

 Figure:

- `figures/car_line_b_absolute_ap.png`
- `figures/car_line_b_delta_ap.png`

Line B I don’t know. PDANS Even... Easy It’s coming down.  `-20.16`, Moderate/Hard Falling separately.  `-21.67/-22.21`; PU-GCN and EdgeFormer This means that the network no is reconstructing deleted real scanning evidence, instead replacing it with generated points PointRCNN fixed 16,384 Part of the point budget for reliable observations.

## 3.  Why was there a reasonable increase in the past?

 The positive results were:

`Original / PDANS / 2.5% generated slots / Car / 3D Moderate`

 It’s from  `82.66`  Increase to  `85.10`,  That’s...  `+2.43`.  However, three types of difficulty and four types of indicators must be fully developed:

{markdown_table(
    ["Metric", "Difficulty", "Baseline", "Observed control (Δ)", "PDANS g2.5 (Δ)", "PDANS − control"],
    positive_rows,
)}

 Key findings:

1.  It’s the main thing that really went up.  **3D Moderate**. 3D Easy Increase only  `+0.35`, 3D Hard It’s going down.  `-1.30`; BBox Moderate Decline  `-1.81`, BEV Moderate Increase only  `+0.17`.  So it’s not ”whole test performance is improved” but it’s difficult. / The most advantageous part of a given indicator.
2.  Compared to the real point of the slot. 3D Moderate Already from  `82.66` Raise to `84.42`,  Contribution `+1.75`; PDANS Overmatch only `+0.68`.  The majority of the upgrades came from sampling under the fixed-point budget, and cannot was all attributed to the generation model.
3. 2.5% As promised only 410 Generate a slot, about 97.5% It’s... PointRCNN Input is still a real observation. a small number of high quality PDANS Point may be added to individual car surfaces left out by sampling, but not sufficient to change neighbourhood statistics on a wide scale.
4.  That’s what I’m talking about. 5%–10%,  Positive gains disappear immediately; crude ratio 10%–50% All methods are reduced in a single way. This results in a dose of low-dose local evidence, high-dose contamination neighbourhood. — Response evidence.
5.  The results are from 256 frame Filter Collection, replace cannot 3,769 frame official results.  `positive mechanism screen`  or  `positive-control candidate`.

 Corresponding figures:

- `figures/car_pdans_g025_positive_attribution.png`
- `figures/car_fine_ratio_easy.png`
- `figures/car_fine_ratio_moderate.png`
- `figures/car_fine_ratio_hard.png`

## 4. PointRCNN It’s a unique root chain.

PointRCNN Directly at point foreground split, local characterization fusion and proposal Generating. fixed Enter point count means that the new generated points will not be added free of charge: they change the reserved real point, the ball neighbourhood members, the local density and proposal Return evidence.

 The full validation set frame migration is further proof that this is not the case AP The problem of resolution:

- Line A Car lost/baseline-TP: PDANS `23.1%`, PU-GCN `32.7%`, EdgeFormer `47.9%`, PU-Net `80.9%`.
- Line B: `35.4% / 55.4% / 67.0% / 76.5%`.
-  The rate of loss follows. 0–20m, 20–40m, 40m+ Total distance rises; see in particular  `figures/car_line_*_loss_by_distance.png`.
- sampler-safe fallback Impact only 3–12/3769 frame, ratio below 0.4%,  It is excluded as the primary cause.

 So the causal chain is:

` generated points geometric Error or redundancy  →  fixed Point Budget Point Real Point / generated points Composition Change  →  Local neighbourhood and foreground points  → proposal Confidence / Positioning degradation  → Moderate/Hard Drop frame rate up.  → AP Decline `

## 5.  Targeted PointRCNN Priority for improvement

1.  Put it. 2.5% As a safe starting point and complete 3,769 frame Upcheck PDANS contrast to match the real point; not directly use strict x4 High rate of generation.
2.  Preserve the real point; generated points fills only empty slots or has low-covered target neighbourhood and prohibits random replacement of reliable real point.
3.  Provides confidence for generated points by local surface consistency, range-image Neighborhood and filtration of disabilities.
4.  Use the target distance. / Original point count Self-adaptation ratio: close and complete vehicle approach 0%,  Rare-drive vehicles allow a small number of patches.
5.  Rehabilitation of common patch Locality and PU-Net normalization to compare model capabilities.
6.  If a higher rate of generation must be used, fine-tune the same mixed distribution PointRCNN,  instead of just changing the test input.

## 6.  Data subject to review

- `tables/formal_ap_all_difficulties.csv`
- `tables/transitions_full_validation.csv`
- `tables/transitions_by_distance.csv`
- `tables/positive_control_256_frame_all_metrics.csv`
- `tables/positive_control_attribution.csv`
"""


def center_report(
    ap_rows: list[dict[str, str]],
    transitions: list[dict[str, str]],
    behavior_rows: list[dict[str, str]],
) -> str:
    behavior = behavior_lookup(behavior_rows)
    base = behavior["downsampled_x4_baseline"]
    pdans = behavior["downsampled_x4_pdans"]
    pugcn = behavior["downsampled_x4_pu_gcn"]
    return f"""# CenterPoint  Independent analysis: by category  Easy / Moderate / Hard  With positive exception

## 1.  Analysis boundary

 The present report analyses only CenterPoint. Car, Pedestrian, Cyclist Splits into sections and diagrams, and does not place different categories in the same performance map. Each category is covered BBox, BEV, 3D and Easy, Moderate, Hard.

 The official results are all from KITTI val 3,769 frame. CenterPoint Use 0.05×0.05×0.1m voxel, maximum per voxel 5 Point. Most tested. 40,000 voxel and will be diluted 3D Characteristic Compression To BEV project the centre and frame parameters.

 According to the user  `diff/difficult`  Corresponding to the present report KITTI Official name  `Hard`.

## 2. Car

### Line A

{formal_all_metric_tables(ap_rows, "centerpoint", "Car", "A")}

### Line B

{formal_all_metric_tables(ap_rows, "centerpoint", "Car", "B")}

Car It’s down in two lines, three difficulties, four methods. Line B It’s... 3D Moderate Down to PDANS `-17.85`, PU-GCN `-25.53`, EdgeFormer `-39.85`, PU-Net `-52.88`. Car Use 0.7 IoU,  The centre, size, direction or a slight deviation from boundary will be converted. AP Loss; extra-false voxel is particularly detrimental to the full three-dimensional shell return.

Car Figure only includes Car:

- `figures/car_line_a_absolute_ap.png` with `car_line_a_delta_ap.png`
- `figures/car_line_b_absolute_ap.png` with `car_line_b_delta_ap.png`
- `figures/car_line_a_transitions.png` with `car_line_b_transitions.png`
- `figures/car_line_a_loss_by_distance.png` with `car_line_b_loss_by_distance.png`

## 3. Pedestrian

### Line A

{formal_all_metric_tables(ap_rows, "centerpoint", "Pedestrian", "A")}

Line A All three difficulties are down: the reference to the original scan voxel 100%,  Add a new point to change the voxel average and BEV Hot map, while no is really missing evidence for recovery.

### Line B

{formal_all_metric_tables(ap_rows, "centerpoint", "Pedestrian", "B")}

PDANS This is a positive exception for cross-difficult, cross-space indicators:

- BBox Easy/Moderate/Hard: `+0.19 / +0.35 / +0.43`
- BEV: `+3.19 / +2.22 / +2.27`
- 3D: `+4.23 / +3.61 / +3.08`

 It’s like... PointRCNN Single Moderate Good points are more credible because of three difficulties and BEV/3D In the same direction. The evidence of the act is: Pedestrian Total projection from  `{int(float(base["pedestrian_prediction_count"]))}`  Down to  `{int(float(pdans["pedestrian_prediction_count"]))}`,  But... score≥0.5 The projections are from  `{int(float(base["pedestrian_score_ge_0p5_count"]))}`  Increase to  `{int(float(pdans["pedestrian_score_ge_0p5_count"]))}`.  That means... PDANS A large number of low-scoring candidates were reduced, while some of the real small targets were reinforced by high-scoring voxel support.

PU-GCN It’s... Pedestrian 3D Easy/Moderate Just... `+0.55/+0.36`, Hard Yes. `-0.09`,  And... BBox and BEV Both dropped, so cannot was used as a steady rise.

Pedestrian Figure only includes Pedestrian,  In the light of the evidence:

- `figures/pedestrian_line_b_positive_mechanism.png`

## 4. Cyclist

### Line A

{formal_all_metric_tables(ap_rows, "centerpoint", "Cyclist", "A")}

### Line B

{formal_all_metric_tables(ap_rows, "centerpoint", "Cyclist", "B")}

Line B There are two defensible positive outcomes:

1. PDANS 3D Easy/Moderate/Hard Yes. `+2.32/+0.06/+0.40`;  full validation set Single threshold moving recovered `102`, lost `100`,  Net `+2`, score≥0.5 Projection from  `{int(float(base["cyclist_score_ge_0p5_count"]))}`  It’s on the way.  `{int(float(pdans["cyclist_score_ge_0p5_count"]))}`.
2. PU-GCN 3D Easy/Moderate/Hard Yes. `+5.17/+1.39/+1.70`; recovered `94`, lost `81`,  Net `+13`, score≥0.5 Projected increase to  `{int(float(pugcn["cyclist_score_ge_0p5_count"]))}`. BBox and BEV The three difficulties are also synchronized, so this is the most complete recovery evidence available.

Cyclist Use 0.5 IoU,  and downsampling post-baseline 3D Moderate Just... 15.46.  a small number of correctly generates voxel, which can provide a continuous and central response to a small number of riders; it also has a wider range of errors. Car It’s... 0.7 IoU.

Cyclist Figure only includes Cyclist,  In the light of the evidence:

- `figures/cyclist_line_b_positive_mechanism.png`

## 5.  Why are the targets rising? Car Still falling.

Line B no Any method to trigger 40k voxel cap, so no positive or negative result is due cap The geometric audit revealed that:

-  downsampling Baseline Reference 0.2m voxel RETURNED AS 45.8%.
- PDANS/PU-GCN Raised to 59.0%/57.3%,  But eventually, the extra voxel is still there. 45.5%/57.7%.
-  Yeah. Pedestrian/Cyclist,  A small number of correct additions to voxel may cross the minimum evidentiary threshold for “forming a detectable centre”.
-  Yeah. Car,  More complete and well-positioned casings, sizes and directional evidence are needed; the cumulative damage of the error voxel is greater than the recovery of the proceeds.
- EdgeFormer and PU-Net The additional voxel is larger and the reference accuracy is lower, so the three categories are degraded.

 Thus cannot is written as ”upsampling is valid for small targets”. The exact expression should be:

>  Yes. Downsampled Line B I don’t know. PDANS Yeah. Pedestrian, PU-GCN/PDANS Yeah. Cyclist Show specific categories of recovery; this benefit relies on low baseline density, 0.5 IoU threshold and a small number of correctly add voxel and do not migrate to Car or Line A.

## 6. Easy / Moderate / Hard Explanation

1. Easy Small targets are usually closer and less shielded, so the correct addition of voxel makes it easier to form a stabilization centre; Cyclist/PU-GCN It’s... Easy Maximum gain() `+5.17`).
2. Moderate Include more parts of the shield and medium distance target. PDANS/Pedestrian Still holding `+3.61`,  It shows that its new voxel quality is sufficient to improve some of the silt centres’ responses.
3. Hard (b) Effective recovery is reinforced with pseudo voxel contamination. PDANS/Pedestrian Still `+3.08`,  But... PU-GCN/Pedestrian It is close to zero, showing that the quality of the methodology determines whether the benefits can be sustained across the board.
4.  All Car It’s... Hard Still significant decline, indicating that the current point generation accuracy is not sufficient 0.7 IoU 3D frame returns.

## 7.  Targeted CenterPoint Priority for improvement

1.  No, I’m not. Line A Full scan is unconditional. x4;  Priority is given only to low-occupied voxel and small target candidate areas.
2.  Here. Pedestrian/PDANS, Cyclist/PU-GCN As a positive sample, learn “what generates voxel is retained” rather than keeping it in a uniform way only.
3.  Use it. 0.2m Reference Consistency, Local Level Discrepancies and range-image The next door sifts off the bridge. / Wandering point.
4.  voxel preserves the real point average for integration, generated points as a transit disability feature and avoids direct movement MeanVFE Centre.
5.  Selection of budget generation by category and distance: Car Tighter. Small target in. Line B Allows a higher but controlled partial ratio.
6.  Rehabilitation patch Locality and PU-Net The coordinates are changed before the same protocol check.

## 8.  Data subject to review

- `tables/formal_ap_all_difficulties.csv`
- `tables/transitions_full_validation.csv`
- `tables/transitions_by_distance.csv`
- `tables/positive_exception_ap.csv`
- `tables/detector_behavior_summary.csv`
- `tables/voxel_mechanism_summary.csv`
"""


def write_readme() -> None:
    text = """# Detector-separated Easy/Moderate/Hard analysis

 This catalogue is a structural reorganization of the previous version of the double detector evidence package. Two detectors no longer appear in the same performance map or in the same root cause report.

 of which  `diff/difficult`  Press KITTI Common writing for official indicator naming  `Hard`.

 Improvement of experimental design:

- `DETECTOR_AWARE_CONTROLLED_UPSAMPLING_PROPOSAL_EN.md`

## PointRCNN

- `pointrcnn/POINT_RCNN_SEPARATE_ANALYSIS_EN.md`
-  only Car,  Because frozen, the model configuration is... Car-only.
-  Official 3,769 frame Results and Before 256 frame PDANS 2.5% It’s being analysed at the same time.
- Easy, Moderate, Hard and BBox, BEV, 3D Separately.

## CenterPoint

- `centerpoint/CENTERPOINT_SEPARATE_ANALYSIS_EN.md`
- Car, Pedestrian, Cyclist is divided into sections.
-  Special analysis Line B Medium PDANS/Pedestrian with PDANS, PU-GCN/Cyclist The positive exception.

##  Original 3 frame frame visualization

 Full point cloud, frame, BEV And the target cropping is still:

`../dual_detector_three_frame_root_cause_20260730/frames/`

 This Catalogue no reproduces or modifies the results of old experiments, adding only statistics, charts and reports after separation.
"""
    (OUTPUT / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    setup_plot_style()
    ap_rows = read_rows(AP_SOURCE)
    transition_rows = read_rows(TRANSITION_SOURCE)
    distance_rows = read_rows(DISTANCE_SOURCE)
    ratio_rows = read_rows(RATIO_SOURCE)
    behavior_rows = read_rows(CENTERPOINT_BEHAVIOR_SOURCE)
    voxel_rows = read_rows(CENTERPOINT_VOXEL_SOURCE)

    point_dir = OUTPUT / "pointrcnn"
    center_dir = OUTPUT / "centerpoint"

    point_ap = [row for row in ap_rows if row["detector"] == "pointrcnn"]
    point_trans = [
        row for row in transition_rows if row["detector"] == "pointrcnn"
    ]
    point_dist = [row for row in distance_rows if row["detector"] == "pointrcnn"]
    point_ratio = positive_ratio_rows(ratio_rows)
    attribution = positive_attribution(ratio_rows)
    write_rows(point_dir / "tables/formal_ap_all_difficulties.csv", point_ap)
    write_rows(point_dir / "tables/transitions_full_validation.csv", point_trans)
    write_rows(point_dir / "tables/transitions_by_distance.csv", point_dist)
    write_rows(
        point_dir / "tables/positive_control_256_frame_all_metrics.csv", point_ratio
    )
    write_rows(point_dir / "tables/positive_control_attribution.csv", attribution)

    for line in ("A", "B"):
        formal_ap_figure(
            ap_rows,
            "pointrcnn",
            "Car",
            line,
            False,
            point_dir / f"figures/car_line_{line.lower()}_absolute_ap.png",
        )
        formal_ap_figure(
            ap_rows,
            "pointrcnn",
            "Car",
            line,
            True,
            point_dir / f"figures/car_line_{line.lower()}_delta_ap.png",
        )
        transition_figure(
            transition_rows,
            "pointrcnn",
            "Car",
            line,
            point_dir / f"figures/car_line_{line.lower()}_transitions.png",
        )
        distance_figure(
            distance_rows,
            "pointrcnn",
            "Car",
            line,
            point_dir / f"figures/car_line_{line.lower()}_loss_by_distance.png",
        )
    positive_attribution_figure(
        attribution, point_dir / "figures/car_pdans_g025_positive_attribution.png"
    )
    for difficulty in DIFFICULTIES:
        fine_ratio_figure(
            ratio_rows,
            difficulty,
            point_dir / f"figures/car_fine_ratio_{difficulty}.png",
        )
    (point_dir / "POINT_RCNN_SEPARATE_ANALYSIS_EN.md").write_text(
        point_report(ap_rows, attribution), encoding="utf-8"
    )

    center_ap = [row for row in ap_rows if row["detector"] == "centerpoint"]
    center_trans = [
        row for row in transition_rows if row["detector"] == "centerpoint"
    ]
    center_dist = [
        row for row in distance_rows if row["detector"] == "centerpoint"
    ]
    center_positive = centerpoint_positive_rows(ap_rows)
    write_rows(center_dir / "tables/formal_ap_all_difficulties.csv", center_ap)
    write_rows(center_dir / "tables/transitions_full_validation.csv", center_trans)
    write_rows(center_dir / "tables/transitions_by_distance.csv", center_dist)
    write_rows(center_dir / "tables/positive_exception_ap.csv", center_positive)
    write_rows(
        center_dir / "tables/detector_behavior_summary.csv",
        [row for row in behavior_rows if "tulip" not in row["variant"]],
    )
    write_rows(
        center_dir / "tables/voxel_mechanism_summary.csv",
        [row for row in voxel_rows if "tulip" not in row["variant"]],
    )

    for class_name in ("Car", "Pedestrian", "Cyclist"):
        class_slug = class_name.lower()
        for line in ("A", "B"):
            formal_ap_figure(
                ap_rows,
                "centerpoint",
                class_name,
                line,
                False,
                center_dir
                / f"figures/{class_slug}_line_{line.lower()}_absolute_ap.png",
            )
            formal_ap_figure(
                ap_rows,
                "centerpoint",
                class_name,
                line,
                True,
                center_dir / f"figures/{class_slug}_line_{line.lower()}_delta_ap.png",
            )
            transition_figure(
                transition_rows,
                "centerpoint",
                class_name,
                line,
                center_dir / f"figures/{class_slug}_line_{line.lower()}_transitions.png",
            )
            distance_figure(
                distance_rows,
                "centerpoint",
                class_name,
                line,
                center_dir
                / f"figures/{class_slug}_line_{line.lower()}_loss_by_distance.png",
            )
    for class_name in ("Pedestrian", "Cyclist"):
        centerpoint_positive_mechanism_figure(
            class_name,
            ap_rows,
            transition_rows,
            behavior_rows,
            center_dir
            / f"figures/{class_name.lower()}_line_b_positive_mechanism.png",
        )
    (center_dir / "CENTERPOINT_SEPARATE_ANALYSIS_EN.md").write_text(
        center_report(ap_rows, transition_rows, behavior_rows), encoding="utf-8"
    )

    write_readme()
    print(f"output={OUTPUT}")
    print(f"pointrcnn_figures={sum(1 for _ in (point_dir / 'figures').glob('*.png'))}")
    print(f"centerpoint_figures={sum(1 for _ in (center_dir / 'figures').glob('*.png'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
