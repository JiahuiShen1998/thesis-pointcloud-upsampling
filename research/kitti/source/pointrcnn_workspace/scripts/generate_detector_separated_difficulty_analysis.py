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
    return f"""# PointRCNN 独立分析：Easy / Moderate / Hard、正对照与根因

## 1. 分析边界

本报告只分析 PointRCNN，不与 CenterPoint 共图、共表或共享类别结论。当前冻结 PointRCNN 配置只训练和评估 `Car`，因此这里不存在 Pedestrian/Cyclist 缺图问题。

- 正式双线结果：KITTI val 全部 3,769 帧，canonical E2 16,384 点。
- 正对照比例实验：固定 256 帧 Car 子集，嵌套槽位替换；它用于解释机制，不能冒充全验证集结论。
- 指标：BBox、BEV、3D AP_R40，分别报告 Easy、Moderate、Hard。
- 用户所说的 `diff/difficult` 在本报告中对应KITTI官方命名 `Hard`。

## 2. 正式结果：Car

### Line A：Original baseline → Original + x4 upsampling

{formal_all_metric_tables(ap_rows, "pointrcnn", "Car", "A")}

图：

- `figures/car_line_a_absolute_ap.png`
- `figures/car_line_a_delta_ap.png`

难度结论：

1. PDANS 的 3D 下降为 Easy `-7.66`、Moderate `-15.03`、Hard `-15.55`。中/高难度目标比 Easy 多损失约 7–8 AP，说明远距离、遮挡和截断目标对错误新增邻域更敏感。
2. PU-GCN 与 PU-EdgeFormer在 Moderate/Hard 的下降显著大于 Easy；生成点首先破坏的不是明显近车，而是原本只有少量边界点的困难车辆。
3. PU-Net三种难度均失效；Easy下降更大不是“Easy更脆弱”，而是当前坐标归一化接入缺陷造成的全局几何错位，AP已接近下限。

### Line B：Downsampled-x4 baseline → Downsampled + x4 upsampling

{formal_all_metric_tables(ap_rows, "pointrcnn", "Car", "B")}

图：

- `figures/car_line_b_absolute_ap.png`
- `figures/car_line_b_delta_ap.png`

Line B中，PDANS即使是Easy也下降 `-20.16`，Moderate/Hard分别下降 `-21.67/-22.21`；PU-GCN和EdgeFormer下降更大。这说明网络没有重建被删掉的真实扫描证据，而是用生成点替换了PointRCNN固定16,384点预算中的一部分可靠观测。

## 3. 为什么以前会出现合理上升

此前的正结果是：

`Original / PDANS / 2.5% generated slots / Car / 3D Moderate`

它从 `82.66` 提高到 `85.10`，即 `+2.43`。但三种难度和四类指标必须完整展开：

{markdown_table(
    ["Metric", "Difficulty", "Baseline", "Observed control (Δ)", "PDANS g2.5 (Δ)", "PDANS − control"],
    positive_rows,
)}

关键判断：

1. 真正明显上升的主要是 **3D Moderate**。3D Easy只增加 `+0.35`，3D Hard反而下降 `-1.30`；BBox Moderate下降 `-1.81`，BEV Moderate只增加 `+0.17`。因此它不是“整体检测性能提高”，而是一个难度/指标特定的局部最优点。
2. 同槽位真实点对照的3D Moderate已经从 `82.66`升至`84.42`，贡献`+1.75`；PDANS超过匹配对照仅`+0.68`。大部分提升来自固定点预算下的采样覆盖变化，不能全部归因于生成模型。
3. 2.5%只对应约410个生成槽位，约97.5%的PointRCNN输入仍为真实观测。少量高质量PDANS点可能补到个别被采样遗漏的车体表面，同时不足以大范围改变邻域统计。
4. 当比例提高到5%–10%，正增益立即消失；粗比例10%–50%的所有方法总体单调下降。这形成“低剂量局部补证据、高剂量污染邻域”的剂量—反应证据。
5. 该结果来自256帧筛选集，尚不能取代3,769帧正式结果。论文中应称为 `positive mechanism screen` 或 `positive-control candidate`。

对应图：

- `figures/car_pdans_g025_positive_attribution.png`
- `figures/car_fine_ratio_easy.png`
- `figures/car_fine_ratio_moderate.png`
- `figures/car_fine_ratio_hard.png`

## 4. PointRCNN特有的根因链

PointRCNN直接在点上进行前景分割、局部特征聚合和proposal生成。固定输入点数意味着新增生成点不会免费加入：它们会改变被保留真实点、球邻域成员、局部密度和proposal回归证据。

全验证集框迁移进一步证明这不是AP解析问题：

- Line A Car lost/baseline-TP：PDANS `23.1%`、PU-GCN `32.7%`、EdgeFormer `47.9%`、PU-Net `80.9%`。
- Line B：`35.4% / 55.4% / 67.0% / 76.5%`。
- 丢失率随0–20m、20–40m、40m+距离整体上升；具体见 `figures/car_line_*_loss_by_distance.png`。
- sampler-safe fallback只影响3–12/3769帧，比例低于0.4%，排除其作为主因。

所以因果链是：

`生成点几何误差或冗余 → 固定点预算内真实点/生成点构成变化 → 局部邻域与前景得分变化 → proposal置信度/定位退化 → Moderate/Hard丢框率上升 → AP下降`

## 5. 针对PointRCNN的改进优先级

1. 把2.5%作为安全起点，并在完整3,769帧上复验PDANS与匹配真实点对照；不应直接使用严格x4高生成比例。
2. 保留真实点优先；生成点只填充空槽位或低覆盖目标邻域，禁止随机替换可靠真实点。
3. 为生成点提供置信度，按局部表面一致性、range-image邻接和法向残差过滤。
4. 用目标距离/原始点数自适应比例：近距离完整车辆接近0%，稀疏远车允许少量补点。
5. 修复共同patch局部性和PU-Net归一化后再比较模型能力。
6. 若必须使用较高生成比例，应以相同混合分布微调PointRCNN，而不是只改变测试输入。

## 6. 可复核数据

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
    return f"""# CenterPoint 独立分析：逐类别 Easy / Moderate / Hard 与正向例外

## 1. 分析边界

本报告只分析CenterPoint。Car、Pedestrian、Cyclist分别成节、分别成图，不把不同类别放在同一幅性能图中。每个类别均覆盖BBox、BEV、3D以及Easy、Moderate、Hard。

正式结果全部来自KITTI val 3,769帧。CenterPoint使用0.05×0.05×0.1m体素、每体素最多5点、测试最多40,000体素，并将稀疏3D特征压缩到BEV后预测中心与框参数。

用户所说的 `diff/difficult` 在本报告中对应KITTI官方命名 `Hard`。

## 2. Car

### Line A

{formal_all_metric_tables(ap_rows, "centerpoint", "Car", "A")}

### Line B

{formal_all_metric_tables(ap_rows, "centerpoint", "Car", "B")}

Car在两条线、三种难度、四种方法上全部下降。Line B的3D Moderate下降为PDANS `-17.85`、PU-GCN `-25.53`、EdgeFormer `-39.85`、PU-Net `-52.88`。Car使用0.7 IoU，中心、尺寸、朝向或边界的轻微偏差都会转成AP损失；额外伪体素对完整三维外壳回归尤其不利。

Car图仅包含Car：

- `figures/car_line_a_absolute_ap.png`与`car_line_a_delta_ap.png`
- `figures/car_line_b_absolute_ap.png`与`car_line_b_delta_ap.png`
- `figures/car_line_a_transitions.png`与`car_line_b_transitions.png`
- `figures/car_line_a_loss_by_distance.png`与`car_line_b_loss_by_distance.png`

## 3. Pedestrian

### Line A

{formal_all_metric_tables(ap_rows, "centerpoint", "Pedestrian", "A")}

Line A三种难度全部下降：原始扫描的参考体素召回已是100%，新增点主要改变小目标中心附近的体素均值和BEV热图，而没有真实缺失证据可恢复。

### Line B

{formal_all_metric_tables(ap_rows, "centerpoint", "Pedestrian", "B")}

PDANS是一个跨难度、跨空间指标一致的正向例外：

- BBox Easy/Moderate/Hard：`+0.19 / +0.35 / +0.43`
- BEV：`+3.19 / +2.22 / +2.27`
- 3D：`+4.23 / +3.61 / +3.08`

这比PointRCNN的单一Moderate正点更可信，因为三个难度和BEV/3D方向一致。行为证据是：Pedestrian总预测数从 `{int(float(base["pedestrian_prediction_count"]))}` 降到 `{int(float(pdans["pedestrian_prediction_count"]))}`，但score≥0.5的预测从 `{int(float(base["pedestrian_score_ge_0p5_count"]))}` 增到 `{int(float(pdans["pedestrian_score_ge_0p5_count"]))}`。这说明PDANS减少了大量低分候选，同时增强了一部分真正小目标的高分体素支持。

PU-GCN的Pedestrian 3D Easy/Moderate只有`+0.55/+0.36`，Hard为`-0.09`，而BBox和BEV均下降，因此不能当作稳定提升。

Pedestrian图仅包含Pedestrian，正向证据见：

- `figures/pedestrian_line_b_positive_mechanism.png`

## 4. Cyclist

### Line A

{formal_all_metric_tables(ap_rows, "centerpoint", "Cyclist", "A")}

### Line B

{formal_all_metric_tables(ap_rows, "centerpoint", "Cyclist", "B")}

Line B存在两个可解释的正向结果：

1. PDANS 3D Easy/Moderate/Hard为`+2.32/+0.06/+0.40`；全验证集单阈值迁移中recovered `102`、lost `100`，净`+2`，score≥0.5预测从 `{int(float(base["cyclist_score_ge_0p5_count"]))}` 墀至 `{int(float(pdans["cyclist_score_ge_0p5_count"]))}`。
2. PU-GCN 3D Easy/Moderate/Hard为`+5.17/+1.39/+1.70`；recovered `94`、lost `81`，净`+13`，score≥0.5预测增至 `{int(float(pugcn["cyclist_score_ge_0p5_count"]))}`。BBox和BEV三种难度也同步提高，因此这是当前最完整的小目标恢复证据。

Cyclist使用0.5 IoU，且下采样后基线3D Moderate只有15.46。少量正确生成体素可以让原本只有极少点的骑行者形成连续中心响应；其容错范围也大于Car的0.7 IoU。

Cyclist图仅包含Cyclist，正向证据见：

- `figures/cyclist_line_b_positive_mechanism.png`

## 5. 为什么小目标能涨、Car仍然跌

Line B没有任何方法触发40k体素上限，因此正负结果都不是由cap截断解释的。几何审计显示：

- 下采样基线参考0.2m体素召回为45.8%。
- PDANS/PU-GCN分别提高到59.0%/57.3%，但最终额外体素仍达到45.5%/57.7%。
- 对Pedestrian/Cyclist，少数正确新增体素可能跨过“形成可检测中心”的最低证据门槛。
- 对Car，需要更完整且位置准确的外壳、尺寸和朝向证据；错误体素的累计损害大于召回收益。
- EdgeFormer和PU-Net的额外体素更多、参考精度更低，所以三类都退化。

因此不能写成“上采样对小目标都有效”。准确表述应是：

> 在Downsampled Line B中，PDANS对Pedestrian、PU-GCN/PDANS对Cyclist表现出类别特定恢复；这种收益依赖低基线密度、0.5 IoU阈值和少量正确新增体素，并不迁移到Car或Line A。

## 6. Easy / Moderate / Hard的解释

1. Easy小目标通常更近、遮挡更少，所以正确新增体素更容易形成稳定中心；Cyclist/PU-GCN的Easy增益最大（`+5.17`）。
2. Moderate包含更多部分遮挡和中距离目标。PDANS/Pedestrian仍保持`+3.61`，说明其新增体素质量足以改善一部分稀疏中心响应。
3. Hard目标更远、更遮挡；有效恢复与伪体素污染同时增强。PDANS/Pedestrian仍为`+3.08`，但PU-GCN/Pedestrian已经接近零，显示方法质量决定收益是否能跨难度保持。
4. 所有Car的Hard仍明显下降，说明当前点生成精度不足以满足0.7 IoU三维框回归。

## 7. 针对CenterPoint的改进优先级

1. 不在Line A完整扫描上无条件x4；优先只处理低占用体素和小目标候选区域。
2. 以Pedestrian/PDANS、Cyclist/PU-GCN作为正样本，学习“哪些生成体素被保留”，而不是只按方法统一保留。
3. 用0.2m参考一致性、局部平面残差和range-image邻接筛掉桥接/漂移点。
4. 体素级融合时保留真实点均值，生成点作为带权残差特征，避免直接移动MeanVFE中心。
5. 按类别和距离选择生成预算：Car更严格，小目标在Line B允许较高但受控的局部比例。
6. 修复patch局部性和PU-Net坐标变换后再做同协议复验。

## 8. 可复核数据

- `tables/formal_ap_all_difficulties.csv`
- `tables/transitions_full_validation.csv`
- `tables/transitions_by_distance.csv`
- `tables/positive_exception_ap.csv`
- `tables/detector_behavior_summary.csv`
- `tables/voxel_mechanism_summary.csv`
"""


def write_readme() -> None:
    text = """# Detector-separated Easy/Moderate/Hard analysis

本目录是对上一版双检测器证据包的结构化重排。两个检测器不再出现在同一张性能图或同一份根因报告中。

其中 `diff/difficult` 按KITTI官方指标命名统一写作 `Hard`。

改进实验设计：

- `DETECTOR_AWARE_CONTROLLED_UPSAMPLING_PROPOSAL_ZH.md`

## PointRCNN

- `pointrcnn/POINT_RCNN_SEPARATE_ANALYSIS_ZH.md`
- 仅Car，因为冻结模型配置就是Car-only。
- 正式3,769帧结果和此前256帧PDANS 2.5%正对照同时分析。
- Easy、Moderate、Hard以及BBox、BEV、3D分别保留。

## CenterPoint

- `centerpoint/CENTERPOINT_SEPARATE_ANALYSIS_ZH.md`
- Car、Pedestrian、Cyclist分别成图成节。
- 特别分析Line B中PDANS/Pedestrian与PDANS、PU-GCN/Cyclist的正向例外。

## 原三帧框级可视化

完整点云、框、BEV和目标裁剪仍位于：

`../dual_detector_three_frame_root_cause_20260730/frames/`

本目录没有复制或修改旧实验结果，只新增分离后的统计、图表和报告。
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
    (point_dir / "POINT_RCNN_SEPARATE_ANALYSIS_ZH.md").write_text(
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
    (center_dir / "CENTERPOINT_SEPARATE_ANALYSIS_ZH.md").write_text(
        center_report(ap_rows, transition_rows, behavior_rows), encoding="utf-8"
    )

    write_readme()
    print(f"output={OUTPUT}")
    print(f"pointrcnn_figures={sum(1 for _ in (point_dir / 'figures').glob('*.png'))}")
    print(f"centerpoint_figures={sum(1 for _ in (center_dir / 'figures').glob('*.png'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
