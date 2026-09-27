#!/usr/bin/env python3
"""Generate thesis-ready PointNet++ final figures and reports from existing CSV/MD data."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
FIGURES = PROJECT_ROOT / "figures" / "modelnet40"
RESULTS = PROJECT_ROOT / "pointnet2_results" / "x4_two_line_final"

PN2_FINAL = FIGURES / "pointnet2_final"
ACC_ABS = PN2_FINAL / "accuracy_absolute"
ACC_DELTA = PN2_FINAL / "accuracy_delta_methods_only"
TWO_LINE = PN2_FINAL / "two_line_summary"
GEOM_DELTA = FIGURES / "geometry_delta_methods_only"
FINAL_COMBINED = FIGURES / "final_combined"

METHODS_ONLY = ["EAR", "PDANS", "PU-Net", "PU-GCN"]

METHOD_COLORS = {
    "Original baseline": "#4C72B0",
    "Downsampled baseline": "#55A868",
    "EAR": "#C44E52",
    "PDANS": "#8172B3",
    "PU-Net": "#CCB974",
    "PU-GCN": "#64B5CD",
}

LINEA_METRICS = {
    "Original baseline 1024": RESULTS / "lineA_original_baseline" / "metrics.json",
    "Original + EAR 4096": RESULTS / "lineA_original_up" / "ear" / "metrics.json",
    "Original + PDANS 4096": RESULTS / "lineA_original_up" / "pdans" / "metrics.json",
    "Original + PU-Net 4096": RESULTS / "lineA_original_up" / "pu_net" / "metrics.json",
    "Original + PU-GCN 4096": RESULTS / "lineA_original_up" / "pu_gcn" / "metrics.json",
}

GEOM_LINEB_GROUPS = [
    "Downsampled x4 + EAR",
    "Downsampled x4 + PDANS",
    "Downsampled x4 + PU-Net",
    "Downsampled x4 + PU-GCN",
]

GENERATED: list[dict] = []
FAILURES: list[str] = []

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp_str(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f} pp"


def short_label_linea(method: str) -> str:
    return {
        "Original baseline 1024": "Original baseline",
        "Original + EAR 4096": "EAR",
        "Original + PDANS 4096": "PDANS",
        "Original + PU-Net 4096": "PU-Net",
        "Original + PU-GCN 4096": "PU-GCN",
    }.get(method, method)


def short_label_lineb(method: str) -> str:
    return {
        "Downsampled x4 baseline": "Downsampled baseline",
        "Downsampled x4 baseline 256": "Downsampled baseline",
        "Downsampled x4 + EAR": "EAR",
        "Downsampled x4 + EAR 1024": "EAR",
        "Downsampled x4 + PDANS": "PDANS",
        "Downsampled x4 + PDANS 1024": "PDANS",
        "Downsampled x4 + PU-Net": "PU-Net",
        "Downsampled x4 + PU-Net 1024": "PU-Net",
        "Downsampled x4 + PU-GCN": "PU-GCN",
        "Downsampled x4 + PU-GCN 1024": "PU-GCN",
    }.get(method, method)


def save_fig(fig: plt.Figure, stem: Path, caption_note: str = "", *, close: bool = True) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(ext))
    GENERATED.append(
        {
            "path_png": str(stem.with_suffix(".png")),
            "path_pdf": str(stem.with_suffix(".pdf")),
            "caption_note": caption_note,
        }
    )
    if close:
        plt.close(fig)


def save_fig_multi(fig: plt.Figure, stems: list[Path], caption_note: str = "") -> None:
    for i, stem in enumerate(stems):
        save_fig(fig, stem, caption_note, close=(i == len(stems) - 1))


def load_linea_cls() -> list[dict]:
    rows = read_csv(REPORTS / "modelnet40_pointnet2_lineA_classification_summary.csv")
    out = []
    for r in rows:
        out.append(
            {
                "line": r["line"],
                "method": r["method"],
                "label": short_label_linea(r["method"]),
                "point_count": int(r["point_count"]),
                "final_test_overall_accuracy": float(r["final_test_overall_accuracy"]),
                "final_test_class_accuracy": float(r["final_test_class_accuracy"]),
                "best_overall_accuracy": float(r["best_overall_accuracy"]),
                "best_class_accuracy": float(r["best_class_accuracy"]),
                "delta_best": float(r["delta_best_overall_vs_original_baseline"]),
                "delta_final": float(r["delta_final_overall_vs_original_baseline"]),
                "baseline_reference": "Original baseline 1024",
                "metrics_json_path": str(LINEA_METRICS.get(r["method"], "")),
                "status": "PASS" if LINEA_METRICS.get(r["method"], Path()).exists() else "UNKNOWN",
            }
        )
    return out


def load_lineb_cls() -> list[dict]:
    rows = read_csv(REPORTS / "modelnet40_pointnet2_lineB_classification_summary.csv")
    out = []
    for r in rows:
        method = r["method"]
        if method == "Downsampled x4 baseline":
            method = "Downsampled x4 baseline 256"
        elif method.startswith("Downsampled x4 + "):
            method = f"{method} 1024"
        out.append(
            {
                "line": r["line"],
                "method": method,
                "label": short_label_lineb(r["method"]),
                "point_count": int(r["point_count"]),
                "final_test_overall_accuracy": float(r["final_test_overall_accuracy"]),
                "final_test_class_accuracy": float(r["final_test_class_accuracy"]),
                "best_overall_accuracy": float(r["best_overall_accuracy"]),
                "best_class_accuracy": float(r["best_class_accuracy"]),
                "delta_best": float(r["delta_best_overall_vs_downsampled_baseline"]),
                "delta_final": float(r["delta_final_overall_vs_downsampled_baseline"]),
                "baseline_reference": "Downsampled x4 baseline 256",
                "metrics_json_path": r.get("metrics_json_path", ""),
                "status": r.get("status", "PASS"),
            }
        )
    return out


def load_lineb_geometry_deltas() -> list[dict]:
    rows = read_csv(REPORTS / "modelnet40_quality_metrics_thesis_table.csv")
    out = []
    for group in GEOM_LINEB_GROUPS:
        for r in rows:
            if r["group"] == group:
                out.append(
                    {
                        "group": group,
                        "label": r["method"],
                        "delta_CD": float(r["delta_CD_vs_original"]),
                        "delta_HD": float(r["delta_HD_vs_original"]),
                        "delta_NUC": float(r["delta_NUC_vs_original"]),
                        "delta_P2F": float(r["delta_P2F_vs_original"]),
                    }
                )
                break
    return out


def load_geom_cls_lineb() -> list[dict]:
    rows = read_csv(REPORTS / "modelnet40_geometry_vs_classification_final_summary.csv")
    out = []
    for r in rows:
        if r["Line"] != "B" or "baseline" in r["Method"].lower():
            continue
        out.append(
            {
                "method": r["Method"],
                "label": short_label_lineb(r["Method"]),
                "CD": float(r["CD"]),
                "HD": float(r["HD"]),
                "NUC": float(r["NUC"]),
                "exact_P2F": float(r["exact_P2F"]),
                "delta_acc": float(r["Delta Best Overall vs line baseline"]) * 100,
            }
        )
    return out


def plot_accuracy_absolute(
    data: list[dict],
    baseline_idx: int,
    title: str,
    baseline_label: str,
    footnote: str,
    out_stem: Path,
) -> None:
    labels = [d["label"] for d in data]
    values = [d["best_overall_accuracy"] * 100 for d in data]
    baseline_val = values[baseline_idx]
    colors = [METHOD_COLORS.get(l, "#888888") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(
        baseline_val,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label=f"{baseline_label} ({baseline_val:.2f}%)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Best Overall Accuracy (%)")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(min(values) - 1.5, max(values) + 1.5)
    ax.legend(loc="lower right")
    fig.text(0.01, 0.01, footnote, fontsize=8, style="italic", wrap=True)
    save_fig(fig, out_stem, footnote)


def plot_delta_methods_only(
    data: list[dict],
    title: str,
    ylabel: str,
    out_stem: Path,
    caption: str = "",
) -> None:
    rows = [d for d in data if d["label"] in METHODS_ONLY]
    rows = sorted(rows, key=lambda d: METHODS_ONLY.index(d["label"]))
    labels = [d["label"] for d in rows]
    deltas = [d["delta_best"] * 100 for d in rows]
    colors = ["#2CA02C" if d >= 0 else "#D62728" for d in deltas]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, val in zip(bars, deltas):
        sign = "+" if val >= 0 else ""
        ypos = val + (0.06 if val >= 0 else -0.06)
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{sign}{val:.2f} pp",
            ha="center",
            va=va,
            fontsize=9,
        )
    pad = max(abs(min(deltas)), abs(max(deltas))) * 0.4 + 0.25
    ax.set_ylim(min(deltas) - pad, max(deltas) + pad)
    if caption:
        fig.text(0.01, 0.01, caption, fontsize=8, style="italic")
    save_fig(fig, out_stem, caption)


def plot_two_line_absolute(linea: list[dict], lineb: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, data, line_name, baseline_label in [
        (axes[0], linea, "Line A", "Original baseline"),
        (axes[1], lineb, "Line B", "Downsampled baseline"),
    ]:
        labels = [d["label"] for d in data]
        values = [d["best_overall_accuracy"] * 100 for d in data]
        baseline_val = values[0]
        colors = [METHOD_COLORS.get(l, "#888") for l in labels]
        x = np.arange(len(labels))
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.axhline(baseline_val, color="#333333", linestyle="--", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{line_name}: Best Overall Accuracy")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylim(min(values) - 1.5, max(values) + 1.5)

    axes[0].set_ylabel("Best Overall Accuracy (%)")
    fig.suptitle(
        "PointNet++ Two-Line Protocol: Best Overall Classification Accuracy",
        y=1.02,
    )
    caption = (
        "Line A baseline = Original baseline 1024; "
        "Line B baseline = Downsampled x4 baseline 256. "
        "Dashed lines mark each line's baseline reference."
    )
    fig.text(0.01, 0.01, caption, fontsize=8, style="italic")
    fig.tight_layout()
    save_fig_multi(
        fig,
        [
            ACC_ABS / "two_line_best_overall_accuracy_absolute",
            TWO_LINE / "two_line_best_overall_accuracy_absolute",
        ],
        caption,
    )


def plot_two_line_delta(linea: list[dict], lineb: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(METHODS_ONLY))
    width = 0.35

    def method_deltas(data: list[dict]) -> list[float]:
        by_label = {d["label"]: d["delta_best"] * 100 for d in data}
        return [by_label[m] for m in METHODS_ONLY]

    deltas_a = method_deltas(linea)
    deltas_b = method_deltas(lineb)

    bars_a = ax.bar(x - width / 2, deltas_a, width, label="Line A vs Original baseline", color="#4C72B0", edgecolor="white")
    bars_b = ax.bar(x + width / 2, deltas_b, width, label="Line B vs Downsampled baseline", color="#55A868", edgecolor="white")
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS_ONLY)
    ax.set_ylabel("Δ Best Overall Accuracy (pp)")
    ax.set_title("Two-Line Δ Best Overall Accuracy (Methods Only)")
    ax.legend(loc="lower right")

    for bars, deltas in [(bars_a, deltas_a), (bars_b, deltas_b)]:
        for bar, val in zip(bars, deltas):
            sign = "+" if val >= 0 else ""
            ypos = val + (0.05 if val >= 0 else -0.05)
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                ypos,
                f"{sign}{val:.2f}",
                ha="center",
                va=va,
                fontsize=7,
            )

    pad = max(max(abs(d) for d in deltas_a + deltas_b) * 0.4 + 0.3, 0.5)
    ax.set_ylim(min(deltas_a + deltas_b) - pad, max(deltas_a + deltas_b) + pad)
    caption = "Baselines are used only as zero references in this delta plot."
    fig.text(0.01, 0.01, caption, fontsize=8, style="italic")
    save_fig_multi(
        fig,
        [
            ACC_DELTA / "two_line_delta_best_overall_methods_only",
            TWO_LINE / "two_line_delta_best_overall_methods_only",
        ],
        caption,
    )


def plot_geometry_delta(geom: list[dict], metric_key: str, metric_label: str, out_stem: Path) -> None:
    labels = [g["label"] for g in geom]
    deltas = [g[metric_key] for g in geom]
    colors = ["#2CA02C" if d <= 0 else "#D62728" for d in deltas]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(f"Δ {metric_label} vs Original baseline")
    ax.set_title(f"Line B: Δ {metric_label} vs Original Baseline (Methods Only)")
    for bar, val in zip(bars, deltas):
        sign = "+" if val >= 0 else ""
        ypos = val + (0.002 if val >= 0 else -0.002)
        va = "bottom" if val >= 0 else "top"
        fmt = f"{sign}{val:.4f}"
        ax.text(bar.get_x() + bar.get_width() / 2, ypos, fmt, ha="center", va=va, fontsize=8)
    pad = max(abs(min(deltas)), abs(max(deltas))) * 0.25 + 0.005
    ax.set_ylim(min(deltas) - pad, max(deltas) + pad)
    caption = (
        "The Original baseline is used as the zero reference and is not plotted as a method bar. "
        "Absolute metrics were computed with respect to the dense mesh surface reference."
    )
    fig.text(0.01, 0.01, caption, fontsize=8, style="italic")
    save_fig(fig, out_stem, caption)


def plot_geom_vs_cls_scatter(
    geom_cls: list[dict],
    geom_deltas: list[dict],
    metric: str,
    delta_key: str,
    metric_label: str,
    out_stem: Path,
) -> None:
    delta_by_label = {g["label"]: g[delta_key] for g in geom_deltas}
    fig, ax = plt.subplots(figsize=(8, 6))

    for row in geom_cls:
        label = row["label"]
        if label not in METHODS_ONLY:
            continue
        x = delta_by_label[label]
        y = row["delta_acc"]
        color = METHOD_COLORS.get(label, "#888")
        ax.scatter(x, y, c=color, s=100, edgecolors="black", linewidths=0.5, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"Δ {metric_label} vs Original baseline")
    ax.set_ylabel("Δ Best Overall Accuracy vs Downsampled baseline (pp)")
    ax.set_title(f"Line B: Δ {metric_label} vs Δ Classification Accuracy")
    caption = (
        "This figure compares the deviation from Original geometric quality with the "
        "downstream classification improvement relative to the downsampled baseline."
    )
    fig.text(0.01, 0.01, caption, fontsize=8, style="italic")
    save_fig(fig, out_stem, caption)


def generate_accuracy_figures(linea: list[dict], lineb: list[dict]) -> None:
    plot_accuracy_absolute(
        linea,
        0,
        "PointNet++ Line A: Original baseline vs Original + Upsampling",
        "Original baseline",
        "The Original baseline is shown as a real training branch and used as the Line A reference.",
        ACC_ABS / "lineA_best_overall_accuracy_absolute",
    )
    plot_accuracy_absolute(
        lineb,
        0,
        "PointNet++ Line B: Downsampled baseline vs Downsampled + Upsampling",
        "Downsampled baseline",
        "PU-Net is the only method slightly above the downsampled baseline.",
        ACC_ABS / "lineB_best_overall_accuracy_absolute",
    )
    plot_two_line_absolute(linea, lineb)

    plot_delta_methods_only(
        linea,
        "PointNet++ Line A: Δ Best Overall vs Original Baseline (Methods Only)",
        "Δ Best Overall Accuracy vs Original baseline (pp)",
        ACC_DELTA / "lineA_delta_best_overall_vs_original_baseline_methods_only",
    )
    plot_delta_methods_only(
        lineb,
        "PointNet++ Line B: Δ Best Overall vs Downsampled Baseline (Methods Only)",
        "Δ Best Overall Accuracy vs Downsampled x4 baseline (pp)",
        ACC_DELTA / "lineB_delta_best_overall_vs_downsampled_baseline_methods_only",
    )
    plot_two_line_delta(linea, lineb)


def generate_geometry_delta_figures(geom: list[dict]) -> None:
    for metric_key, metric_label, stem in [
        ("delta_CD", "CD", "lineB_delta_cd_vs_original_baseline_methods_only"),
        ("delta_HD", "HD", "lineB_delta_hd_vs_original_baseline_methods_only"),
        ("delta_NUC", "NUC", "lineB_delta_nuc_vs_original_baseline_methods_only"),
        ("delta_P2F", "exact P2F", "lineB_delta_exact_p2f_vs_original_baseline_methods_only"),
    ]:
        plot_geometry_delta(geom, metric_key, metric_label, GEOM_DELTA / stem)


def generate_combined_figures(geom_cls: list[dict], geom_deltas: list[dict]) -> None:
    for metric_key, metric_label, stem in [
        ("delta_CD", "CD", "lineB_delta_cd_vs_delta_accuracy"),
        ("delta_HD", "HD", "lineB_delta_hd_vs_delta_accuracy"),
        ("delta_NUC", "NUC", "lineB_delta_nuc_vs_delta_accuracy"),
        ("delta_P2F", "exact P2F", "lineB_delta_p2f_vs_delta_accuracy"),
    ]:
        plot_geom_vs_cls_scatter(geom_cls, geom_deltas, metric_key, metric_key, metric_label, FINAL_COMBINED / stem)


def build_final_report_rows(linea: list[dict], lineb: list[dict]) -> list[dict]:
    rows = []
    for data in linea + lineb:
        rows.append(
            {
                "line": data["line"],
                "method": data["method"],
                "point_count": data["point_count"],
                "baseline_reference": data["baseline_reference"],
                "final_test_overall_accuracy": data["final_test_overall_accuracy"],
                "final_test_class_accuracy": data["final_test_class_accuracy"],
                "best_overall_accuracy": data["best_overall_accuracy"],
                "best_class_accuracy": data["best_class_accuracy"],
                "delta_best_overall_vs_line_baseline_pp": data["delta_best"],
                "delta_final_overall_vs_line_baseline_pp": data["delta_final"],
                "status": data["status"],
                "metrics_json_path": data["metrics_json_path"],
            }
        )
    return rows


def write_final_classification_report(rows: list[dict]) -> None:
    csv_path = REPORTS / "modelnet40_pointnet2_final_classification_report.csv"
    fieldnames = list(rows[0].keys())
    write_csv(csv_path, fieldnames, rows)

    linea_rows = [r for r in rows if r["line"] == "A"]
    lineb_rows = [r for r in rows if r["line"] == "B"]

    def table_section(title: str, section_rows: list[dict]) -> list[str]:
        lines = [
            f"## {title}",
            "",
            "| Method | Points | Best Overall | Best Class | Δ Best (pp) | Final Overall | Δ Final (pp) | Status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        for r in section_rows:
            lines.append(
                f"| {r['method']} | {r['point_count']} | "
                f"{pct(r['best_overall_accuracy'])} | {pct(r['best_class_accuracy'])} | "
                f"{pp_str(r['delta_best_overall_vs_line_baseline_pp'])} | "
                f"{pct(r['final_test_overall_accuracy'])} | "
                f"{pp_str(r['delta_final_overall_vs_line_baseline_pp'])} | {r['status']} |"
            )
        return lines

    def delta_summary(section_rows: list[dict], baseline_name: str) -> list[str]:
        upsampling = [r for r in section_rows if "baseline" not in r["method"].lower()]
        lines = [f"### Delta vs {baseline_name}", ""]
        for r in upsampling:
            lines.append(
                f"- {short_label_linea(r['method']) if r['line'] == 'A' else short_label_lineb(r['method'])}: "
                f"Δ best overall = {pp_str(r['delta_best_overall_vs_line_baseline_pp'])}"
            )
        return lines

    def ranking(section_rows: list[dict]) -> list[str]:
        ranked = sorted(section_rows, key=lambda r: r["best_overall_accuracy"], reverse=True)
        lines = ["### Ranking (best overall accuracy)", ""]
        for i, r in enumerate(ranked, 1):
            lines.append(
                f"{i}. {r['method']} — {pct(r['best_overall_accuracy'])} "
                f"({pp_str(r['delta_best_overall_vs_line_baseline_pp'])} vs baseline)"
            )
        return lines

    md_lines = [
        "# ModelNet40 PointNet++ Final Classification Report",
        "",
        f"- Generated at: {utc_now()}",
        "- Source: existing classification summaries (no retraining)",
        "- Delta units: percentage points (pp)",
        "",
    ]
    md_lines.extend(table_section("1. Line A Classification", linea_rows))
    md_lines.append("")
    md_lines.extend(delta_summary(linea_rows, "Original baseline 1024"))
    md_lines.append("")
    md_lines.extend(ranking(linea_rows))
    md_lines.append("")
    md_lines.extend(table_section("2. Line B Classification", lineb_rows))
    md_lines.append("")
    md_lines.extend(delta_summary(lineb_rows, "Downsampled x4 baseline 256"))
    md_lines.append("")
    md_lines.extend(ranking(lineb_rows))
    md_lines.append("")
    md_lines.extend(
        [
            "## 3. Combined Two-Line Classification",
            "",
            "| Line | Method | Points | Best Overall | Δ Best (pp) | Baseline Reference |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for r in rows:
        md_lines.append(
            f"| {r['line']} | {r['method']} | {r['point_count']} | "
            f"{pct(r['best_overall_accuracy'])} | "
            f"{pp_str(r['delta_best_overall_vs_line_baseline_pp'])} | {r['baseline_reference']} |"
        )
    md_lines.extend(
        [
            "",
            "## 4. Delta Accuracy Summary",
            "",
            "**Line A:** All upsampling methods (EAR, PDANS, PU-Net, PU-GCN) show negative Δ best overall vs Original baseline 1024.",
            "",
            "**Line B:** Only PU-Net shows positive Δ best overall (+0.42 pp) vs Downsampled x4 baseline 256. "
            "EAR, PDANS, and PU-GCN remain below baseline.",
            "",
            "## 5. Ranking Summary",
            "",
            f"- **Line A best:** {linea_rows[0]['method']} — {pct(linea_rows[0]['best_overall_accuracy'])}",
            f"- **Line B best (upsampling):** Downsampled x4 + PU-Net 1024 — "
            f"{pct(max(r['best_overall_accuracy'] for r in lineb_rows[1:]))}",
            f"- **Line B baseline:** {lineb_rows[0]['method']} — {pct(lineb_rows[0]['best_overall_accuracy'])}",
            "",
        ]
    )
    (REPORTS / "modelnet40_pointnet2_final_classification_report.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )


def write_interpretation() -> None:
    content = f"""# ModelNet40 PointNet++ Final Classification Interpretation

- Generated at: {utc_now()}
- Source: finalized two-line PointNet++ classification results (no retraining)

## 1. Line A

- **Original baseline 1024** achieves the highest best overall accuracy: **91.95%**.
- All **Original + Upsampling 4096** methods remain **below** this baseline on best overall accuracy:
  - EAR: 91.48% (−0.47 pp)
  - PDANS: 91.62% (−0.33 pp)
  - PU-Net: 90.95% (−1.00 pp)
  - PU-GCN: 91.63% (−0.32 pp)
- **Interpretation:** Increasing the input point count to 4096 via upsampling does **not** yield a stable classification benefit in Line A. The native 1024-point Original baseline remains strongest.

## 2. Line B

- **Downsampled x4 baseline 256** best overall accuracy: **90.85%**.
- **PU-Net** (Downsampled x4 + PU-Net 1024) best overall accuracy: **91.27%**.
- PU-Net is the **only** upsampling method slightly above the downsampled baseline, with a gain of **+0.42 percentage points**.
- EAR (−2.04 pp), PDANS (−0.51 pp), and PU-GCN (−0.79 pp) do **not** exceed the downsampled baseline on best overall accuracy.
- **Interpretation:** Recovering point count after ×4 downsampling can marginally help classification, but only PU-Net achieves this in our experiment.

## 3. Overall Conclusions

1. **Point cloud upsampling does not universally improve downstream classification.** Line A shows no benefit from 4096-point upsampling; Line B shows benefit only for PU-Net.
2. **The effect depends on the upsampling method and the downstream task.** Geometry-leading methods (e.g., PU-GCN on CD) do not necessarily yield the best classifier.
3. **PU-Net is the most promising method in Line B**, as the sole upsampling variant exceeding the downsampled baseline on best overall accuracy.
4. **Geometric quality and classification accuracy are not fully aligned.** Best CD (PU-GCN) ≠ best classification (PU-Net).

## 4. Thesis Usage

- Use **delta methods-only figures** in the main text to show relative improvement without baseline bars.
- Use **absolute accuracy figures** when showing baseline as a real evaluated training branch.
- Report absolute values with baseline rows in tables; interpret deltas in percentage points (pp).
"""
    (REPORTS / "modelnet40_pointnet2_final_classification_interpretation.md").write_text(
        content,
        encoding="utf-8",
    )


def write_baseline_reference_explanation() -> None:
    content = f"""# ModelNet40 Baseline and Reference Visualization Guide

- Generated at: {utc_now()}

## 1. Why delta figures no longer show baseline bars

In **delta (methods-only) figures**, the baseline is a **reference for comparison**, not an upsampling method. Including the baseline as a bar at y = 0 is visually redundant and can imply the baseline is one of the compared upsampling techniques. Instead:

- The baseline defines the **y = 0 reference line** (classification) or **zero deviation** (geometry).
- Only upsampling methods (EAR, PDANS, PU-Net, PU-GCN) appear as bars or scatter points.
- The caption states which baseline was used.

## 2. Why absolute classification figures can show baseline

In **absolute accuracy figures**, the Original baseline (Line A) and Downsampled x4 baseline (Line B) are **real training branches** with independently trained PointNet++ models. They are legitimate experimental conditions, not computed references. Showing them as bars alongside upsampling methods is appropriate.

A horizontal dashed line marks the baseline value for easy visual comparison.

## 3. Mesh reference vs experimental baseline

| Concept | Role | Used in |
| --- | --- | --- |
| **Dense mesh surface reference** (GT reference) | Ground truth for geometry metrics (CD, HD, exact P2F, NUC) | Geometry evaluation |
| **Original baseline 1024** | Line A classification baseline; geometry zero-reference for delta plots | Classification Line A; geometry deltas |
| **Downsampled x4 baseline 256** | Line B classification baseline | Classification Line B |

Do **not** call the mesh reference a "baseline" in figure captions. Use "dense mesh surface reference" or "mesh reference."

## 4. Absolute metric tables vs delta figures

- **Tables** report absolute metric values with a baseline row (delta = 0). Readers need exact numbers for thesis tables.
- **Delta figures** emphasize **relative change** and focus on method comparison. Baseline is implicit at zero.

## 5. Why baseline rows are kept in the final classification report

The report CSV/MD includes baseline rows with delta = 0 because:

- Tables must show the full experimental matrix.
- Readers can verify delta calculations from absolute values.
- Ranking and summary sections reference the baseline explicitly.

## 6. Recommended thesis figure usage

| Location | Recommended figures |
| --- | --- |
| **Main text** | Delta methods-only accuracy and geometry figures; final combined geometry-vs-classification scatter |
| **Tables** | Absolute values with baseline row (`modelnet40_pointnet2_final_classification_report`) |
| **Supplementary** | Absolute geometry bar plots; point cloud qualitative comparisons; interactive HTML viewers |
"""
    (REPORTS / "modelnet40_baseline_reference_visualization_fix.md").write_text(
        content,
        encoding="utf-8",
    )


FIGURE_INDEX_ENTRIES = [
    ("pointnet2_final/accuracy_absolute/lineA_best_overall_accuracy_absolute", "modelnet40_pointnet2_lineA_classification_summary.csv", "Line A absolute best overall accuracy; baseline shown as real branch with dashed reference.", "Results — Line A classification"),
    ("pointnet2_final/accuracy_absolute/lineB_best_overall_accuracy_absolute", "modelnet40_pointnet2_lineB_classification_summary.csv", "Line B absolute best overall accuracy; PU-Net only method above downsampled baseline.", "Results — Line B classification"),
    ("pointnet2_final/accuracy_absolute/two_line_best_overall_accuracy_absolute", "modelnet40_pointnet2_final_two_line_classification_summary.csv", "Grouped two-line absolute best overall accuracy.", "Results — Overview"),
    ("pointnet2_final/accuracy_delta_methods_only/lineA_delta_best_overall_vs_original_baseline_methods_only", "modelnet40_pointnet2_lineA_classification_summary.csv", "Line A Δ best overall vs Original baseline (pp); methods only, y=0 reference.", "Results — Line A classification"),
    ("pointnet2_final/accuracy_delta_methods_only/lineB_delta_best_overall_vs_downsampled_baseline_methods_only", "modelnet40_pointnet2_lineB_classification_summary.csv", "Line B Δ best overall vs downsampled baseline (pp); PU-Net +0.42 pp.", "Results — Line B classification"),
    ("pointnet2_final/accuracy_delta_methods_only/two_line_delta_best_overall_methods_only", "modelnet40_pointnet2_final_two_line_classification_summary.csv", "Two-line grouped Δ best overall (methods only).", "Results — Overview"),
    ("geometry_delta_methods_only/lineB_delta_cd_vs_original_baseline_methods_only", "modelnet40_quality_metrics_thesis_table.csv", "Line B Δ CD vs Original baseline (methods only).", "Results — Geometry quality"),
    ("geometry_delta_methods_only/lineB_delta_hd_vs_original_baseline_methods_only", "modelnet40_quality_metrics_thesis_table.csv", "Line B Δ HD vs Original baseline (methods only).", "Results — Geometry quality"),
    ("geometry_delta_methods_only/lineB_delta_nuc_vs_original_baseline_methods_only", "modelnet40_quality_metrics_thesis_table.csv", "Line B Δ NUC vs Original baseline (methods only).", "Results — Geometry quality"),
    ("geometry_delta_methods_only/lineB_delta_exact_p2f_vs_original_baseline_methods_only", "modelnet40_quality_metrics_thesis_table.csv", "Line B Δ exact P2F vs Original baseline (methods only).", "Results — Geometry quality"),
    ("final_combined/lineB_delta_cd_vs_delta_accuracy", "modelnet40_geometry_vs_classification_final_summary.csv", "Line B Δ CD vs Δ classification accuracy scatter.", "Discussion — Geometry vs classification"),
    ("final_combined/lineB_delta_hd_vs_delta_accuracy", "modelnet40_geometry_vs_classification_final_summary.csv", "Line B Δ HD vs Δ classification accuracy scatter.", "Discussion — Geometry vs classification"),
    ("final_combined/lineB_delta_nuc_vs_delta_accuracy", "modelnet40_geometry_vs_classification_final_summary.csv", "Line B Δ NUC vs Δ classification accuracy scatter.", "Discussion — Geometry vs classification"),
    ("final_combined/lineB_delta_p2f_vs_delta_accuracy", "modelnet40_geometry_vs_classification_final_summary.csv", "Line B Δ exact P2F vs Δ classification accuracy scatter.", "Discussion — Geometry vs classification"),
]


def write_figure_index() -> None:
    lines = [
        "# ModelNet40 Thesis Figure Index",
        "",
        f"- Generated at: {utc_now()}",
        f"- Root: `{FIGURES}`",
        "",
        "## PointNet++ Final Figures (thesis-ready)",
        "",
        "| Figure | Source data | Caption | Thesis section |",
        "| --- | --- | --- | --- |",
    ]
    for rel, source, caption, section in FIGURE_INDEX_ENTRIES:
        lines.append(f"| `{rel}.png` / `.pdf` | `{source}` | {caption} | {section} |")

    lines.extend(
        [
            "",
            "## Visualization conventions",
            "",
            "- **Absolute accuracy figures:** baseline is shown because it is a real evaluated branch.",
            "- **Delta figures (methods-only):** baseline is not plotted as a method bar; used only as y=0 reference.",
            "- **Geometry absolute values:** computed against the dense mesh surface reference.",
            "- **Delta geometry figures:** compare methods relative to the Original baseline.",
            "",
            "## Qualitative point cloud examples",
            "",
            "See `pointcloud_examples/` for Line A and Line B qualitative comparisons (airplane, chair, table, car, sofa).",
        ]
    )
    (FIGURES / "figure_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_thesis_captions() -> None:
    captions = f"""# ModelNet40 Thesis Figure Captions

- Generated at: {utc_now()}
- Style: concise, thesis-ready; classification deltas in percentage points (pp)

## PointNet++ absolute accuracy figures

**Figure: Line A absolute best overall accuracy** (`pointnet2_final/accuracy_absolute/lineA_best_overall_accuracy_absolute`)

Best overall test accuracy of PointNet++ on ModelNet40 under Line A (Original 1024 baseline vs Original + Upsampling at 4096 points). The Original baseline is shown as a real training branch and used as the Line A reference (dashed line at 91.95%). None of the upsampling methods exceed this reference.

**Figure: Line B absolute best overall accuracy** (`pointnet2_final/accuracy_absolute/lineB_best_overall_accuracy_absolute`)

Best overall test accuracy under Line B (Downsampled ×4 baseline at 256 points vs upsampling to 1024 points). The dashed line marks the downsampled baseline (90.85%). PU-Net is the only method slightly above the downsampled baseline.

**Figure: Two-line absolute accuracy** (`pointnet2_final/accuracy_absolute/two_line_best_overall_accuracy_absolute`)

Grouped comparison of best overall accuracy across Line A and Line B. Line A baseline = Original baseline 1024; Line B baseline = Downsampled x4 baseline 256.

## PointNet++ delta accuracy figures (methods only)

**Figure: Line A delta best overall** (`pointnet2_final/accuracy_delta_methods_only/lineA_delta_best_overall_vs_original_baseline_methods_only`)

Change in best overall accuracy relative to the Original 1024 baseline (percentage points). Only upsampling methods are plotted; the Original baseline is the y=0 reference, not a bar. All methods show negative deltas.

**Figure: Line B delta best overall** (`pointnet2_final/accuracy_delta_methods_only/lineB_delta_best_overall_vs_downsampled_baseline_methods_only`)

Change in best overall accuracy relative to the Downsampled ×4 baseline (percentage points). Only upsampling methods are plotted. PU-Net shows +0.42 pp; other methods are negative.

**Figure: Two-line delta best overall** (`pointnet2_final/accuracy_delta_methods_only/two_line_delta_best_overall_methods_only`)

Grouped comparison of Δ best overall for Line A (vs Original baseline) and Line B (vs downsampled baseline). Baselines are used only as zero references in this delta plot.

## Geometry delta figures (methods only)

**Figure: Line B geometry deltas** (`geometry_delta_methods_only/lineB_delta_*_vs_original_baseline_methods_only`)

Geometric quality deltas for Line B upsampling methods relative to the Original 1024 baseline. The Original baseline is used as the zero reference and is not plotted as a method bar. Absolute metrics were computed with respect to the dense mesh surface reference. PU-GCN and PU-Net have the smallest CD deltas; PU-GCN NUC is clearly elevated.

## Geometry vs classification (final combined)

**Figure: Line B geometry delta vs classification delta** (`final_combined/lineB_delta_*_vs_delta_accuracy`)

Scatter plots comparing deviation from Original geometric quality (x-axis) with downstream classification improvement relative to the downsampled baseline (y-axis, pp). Only Line B upsampling methods are shown. PU-GCN has the smallest CD delta but not the highest accuracy delta; PU-Net has the highest accuracy delta. Geometric quality and classification are not perfectly aligned.

## Qualitative point cloud examples

**Figure: Point cloud comparisons** (`pointcloud_examples/lineA_*`, `pointcloud_examples/lineB_*`)

Qualitative comparison of test-set point clouds under Line A and Line B protocols. Identical viewpoint and axis limits across methods.
"""
    (REPORTS / "modelnet40_thesis_figure_captions.md").write_text(captions, encoding="utf-8")


def write_visualization_summary() -> None:
    n_acc_abs = len(list(ACC_ABS.glob("*.png"))) if ACC_ABS.exists() else 0
    n_acc_delta = len(list(ACC_DELTA.glob("*.png"))) if ACC_DELTA.exists() else 0
    n_geom_delta = len(list(GEOM_DELTA.glob("*.png"))) if GEOM_DELTA.exists() else 0
    n_combined = len(list(FINAL_COMBINED.glob("*.png"))) if FINAL_COMBINED.exists() else 0
    n_pc = len(list((FIGURES / "pointcloud_examples").glob("line*.png"))) if (FIGURES / "pointcloud_examples").exists() else 0

    content = f"""# ModelNet40 Visualization Summary

- Generated at: {utc_now()}
- Project: `{PROJECT_ROOT}`

## 1. PointNet++ final absolute accuracy figures

- `{ACC_ABS}` — {n_acc_abs} PNG figures
- Line A, Line B, and two-line grouped absolute best overall accuracy
- Baseline shown as real training branch with horizontal reference line

## 2. PointNet++ final delta accuracy figures (methods only)

- `{ACC_DELTA}` — {n_acc_delta} PNG figures
- Line A, Line B, and two-line grouped Δ best overall (pp)
- Baseline not plotted as bar; y=0 reference only

## 3. Geometry delta figures (methods only)

- `{GEOM_DELTA}` — {n_geom_delta} PNG figures
- Line B Δ CD, HD, NUC, exact P2F vs Original baseline
- Original baseline as zero reference; metrics vs dense mesh surface reference

## 4. Final combined geometry-vs-classification figures

- `{FINAL_COMBINED}` — {n_combined} PNG figures
- Line B scatter: geometry delta vs classification delta
- Highlights misalignment between geometry and classification leaders

## 5. Point cloud qualitative figures

- `{FIGURES / "pointcloud_examples"}` — {n_pc} PNG comparison figures
- Classes: airplane, chair, table, car, sofa

## 6. Visualization conventions

- For **absolute accuracy figures**, baseline is shown because it is a real evaluated branch.
- For **delta figures**, baseline is not plotted as a method bar and is used only as the zero reference.
- **Geometry absolute values** are computed against the dense mesh surface reference.
- **Delta geometry figures** compare methods relative to the Original baseline.

## 7. Key observations

- **Line A:** Original baseline 1024 (91.95%) remains strongest; 4096-point upsampling does not improve classification.
- **Line B:** PU-Net (91.27%) is the only method slightly above the downsampled ×4 baseline (90.85%, +0.42 pp).
- **Geometry:** PU-GCN and PU-Net have the smallest CD delta vs Original; PU-GCN NUC is clearly elevated.
- **Alignment:** Geometry best ≠ classification best.

## 8. Failures

"""
    if FAILURES:
        content += "\n".join(f"- {f}" for f in FAILURES)
    else:
        content += "- None reported.\n"
    content += f"\n## 9. Total figures saved this run\n\n- {len(GENERATED)} figure artifacts (PNG/PDF)\n"
    (REPORTS / "modelnet40_visualization_summary.md").write_text(content, encoding="utf-8")


def write_final_index() -> None:
    content = f"""# ModelNet40 Final Thesis Visualization and Report Index

- Generated at: {utc_now()}
- Project root: `{PROJECT_ROOT}`

## 1. PointNet++ final classification report

- Markdown: `reports/modelnet40_pointnet2_final_classification_report.md`
- CSV: `reports/modelnet40_pointnet2_final_classification_report.csv`

## 2. PointNet++ final interpretation report

- `reports/modelnet40_pointnet2_final_classification_interpretation.md`

## 3. PointNet++ absolute accuracy figures

- Directory: `figures/modelnet40/pointnet2_final/accuracy_absolute/`
- `lineA_best_overall_accuracy_absolute.png` / `.pdf`
- `lineB_best_overall_accuracy_absolute.png` / `.pdf`
- `two_line_best_overall_accuracy_absolute.png` / `.pdf`

## 4. PointNet++ delta methods-only figures

- Directory: `figures/modelnet40/pointnet2_final/accuracy_delta_methods_only/`
- `lineA_delta_best_overall_vs_original_baseline_methods_only.png` / `.pdf`
- `lineB_delta_best_overall_vs_downsampled_baseline_methods_only.png` / `.pdf`
- `two_line_delta_best_overall_methods_only.png` / `.pdf`

## 5. Geometry delta methods-only figures

- Directory: `figures/modelnet40/geometry_delta_methods_only/`
- `lineB_delta_cd_vs_original_baseline_methods_only.png` / `.pdf`
- `lineB_delta_hd_vs_original_baseline_methods_only.png` / `.pdf`
- `lineB_delta_nuc_vs_original_baseline_methods_only.png` / `.pdf`
- `lineB_delta_exact_p2f_vs_original_baseline_methods_only.png` / `.pdf`

## 6. Geometry vs PointNet++ final combined figures

- Directory: `figures/modelnet40/final_combined/`
- `lineB_delta_cd_vs_delta_accuracy.png` / `.pdf`
- `lineB_delta_hd_vs_delta_accuracy.png` / `.pdf`
- `lineB_delta_nuc_vs_delta_accuracy.png` / `.pdf`
- `lineB_delta_p2f_vs_delta_accuracy.png` / `.pdf`

## 7. Existing point cloud qualitative figures

- Directory: `figures/modelnet40/pointcloud_examples/`
- Line A and Line B comparisons for airplane, chair, table, car, sofa

## 8. Captions

- `reports/modelnet40_thesis_figure_captions.md`

## 9. Figure index

- `figures/modelnet40/figure_index.md`

## 10. Visualization summary

- `reports/modelnet40_visualization_summary.md`

## 11. Baseline/reference visualization guide

- `reports/modelnet40_baseline_reference_visualization_fix.md`

## 12. Supporting classification tables

- `reports/modelnet40_thesis_table_classification_lineA.md`
- `reports/modelnet40_thesis_table_classification_lineB.md`
- `reports/modelnet40_thesis_table_classification_combined.md`
- `reports/modelnet40_pointnet2_final_two_line_classification_summary.csv`

## 13. Supporting geometry tables

- `reports/modelnet40_quality_metrics_thesis_table.csv`
- `reports/modelnet40_geometry_vs_classification_final_summary.csv`
"""
    (REPORTS / "modelnet40_final_thesis_visualization_and_report_index.md").write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    for d in (ACC_ABS, ACC_DELTA, TWO_LINE, GEOM_DELTA, FINAL_COMBINED):
        d.mkdir(parents=True, exist_ok=True)

    linea = load_linea_cls()
    lineb = load_lineb_cls()
    geom_deltas = load_lineb_geometry_deltas()
    geom_cls = load_geom_cls_lineb()

    generate_accuracy_figures(linea, lineb)
    generate_geometry_delta_figures(geom_deltas)
    generate_combined_figures(geom_cls, geom_deltas)

    report_rows = build_final_report_rows(linea, lineb)
    write_final_classification_report(report_rows)
    write_interpretation()
    write_baseline_reference_explanation()
    write_figure_index()
    write_thesis_captions()
    write_visualization_summary()
    write_final_index()

    print(f"Generated {len(GENERATED)} figure artifacts")
    print(f"Reports written to {REPORTS}")
    if FAILURES:
        print("Failures:")
        for f in FAILURES:
            print(" -", f)


if __name__ == "__main__":
    main()
