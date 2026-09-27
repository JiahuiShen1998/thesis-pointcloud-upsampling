#!/usr/bin/env python3
"""Generate thesis-ready figures from existing ModelNet40 + PointNet++ experiment results."""

from __future__ import annotations

import csv
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"
FIGURES = PROJECT_ROOT / "figures" / "modelnet40"

# Thesis styling
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

METHOD_COLORS = {
    "Original baseline": "#4C72B0",
    "Downsampled baseline": "#55A868",
    "EAR": "#C44E52",
    "PDANS": "#8172B3",
    "PU-Net": "#CCB974",
    "PU-GCN": "#64B5CD",
}

LINEA_SHORT = ["Original baseline", "EAR", "PDANS", "PU-Net", "PU-GCN"]
LINEB_SHORT = ["Downsampled baseline", "EAR", "PDANS", "PU-Net", "PU-GCN"]

GEOM_LINEB_GROUPS = [
    "Original baseline",
    "Downsampled x4 + EAR",
    "Downsampled x4 + PDANS",
    "Downsampled x4 + PU-Net",
    "Downsampled x4 + PU-GCN",
]
GEOM_LINEB_LABELS = ["Original baseline", "EAR", "PDANS", "PU-Net", "PU-GCN"]

REPRESENTATIVE_SAMPLES = [
    ("airplane", "airplane_0627"),
    ("chair", "chair_0890"),
    ("table", "table_0393"),
    ("car", "car_0198"),
    ("sofa", "sofa_0681"),
]

GENERATED: list[dict] = []
FAILURES: list[str] = []
HAS_PLOTLY = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_fig(fig: plt.Figure, stem: Path, caption_note: str = "") -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(ext))
    plt.close(fig)
    GENERATED.append(
        {
            "path_png": str(stem.with_suffix(".png")),
            "path_pdf": str(stem.with_suffix(".pdf")),
            "caption_note": caption_note,
        }
    )


def short_label_linea(method: str) -> str:
    mapping = {
        "Original baseline 1024": "Original baseline",
        "Original + EAR 4096": "EAR",
        "Original + PDANS 4096": "PDANS",
        "Original + PU-Net 4096": "PU-Net",
        "Original + PU-GCN 4096": "PU-GCN",
    }
    return mapping.get(method, method)


def short_label_lineb(method: str) -> str:
    mapping = {
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
    }
    return mapping.get(method, method)


def load_linea_cls() -> list[dict]:
    rows = read_csv(REPORTS / "modelnet40_pointnet2_lineA_classification_summary.csv")
    out = []
    for r in rows:
        out.append(
            {
                "label": short_label_linea(r["method"]),
                "best": float(r["best_overall_accuracy"]) * 100,
                "delta": float(r["delta_best_overall_vs_original_baseline"]) * 100,
            }
        )
    return out


def load_lineb_cls() -> list[dict]:
    rows = read_csv(REPORTS / "modelnet40_pointnet2_lineB_classification_summary.csv")
    out = []
    for r in rows:
        out.append(
            {
                "label": short_label_lineb(r["method"]),
                "best": float(r["best_overall_accuracy"]) * 100,
                "delta": float(r["delta_best_overall_vs_downsampled_baseline"]) * 100,
            }
        )
    return out


def bar_colors(labels: list[str], baseline_name: str) -> list[str]:
    return [METHOD_COLORS.get(l, "#888888") for l in labels]


def annotate_bars(ax, bars, values, fmt="{:.2f}%", offset=0.3):
    ymax = max(values) if values else 1
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return ymax


def plot_accuracy_bars(
    data: list[dict],
    baseline_idx: int,
    title: str,
    ylabel: str,
    footnote: str,
    out_stem: Path,
    baseline_label: str,
) -> None:
    labels = [d["label"] for d in data]
    values = [d["best"] for d in data]
    baseline_val = values[baseline_idx]
    colors = bar_colors(labels, labels[baseline_idx])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(baseline_val, color="#333333", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ymax = annotate_bars(ax, bars, values)
    ax.set_ylim(min(values) - 2, ymax + 2)
    ax.legend(loc="lower right")
    fig.text(0.01, 0.01, footnote, fontsize=8, style="italic", wrap=True)
    save_fig(fig, out_stem)


def plot_delta_bars(
    data: list[dict],
    title: str,
    out_stem: Path,
    skip_baseline: bool = True,
) -> None:
    rows = data[1:] if skip_baseline else data
    labels = [d["label"] for d in rows]
    deltas = [d["delta"] for d in rows]
    colors = ["#2CA02C" if d >= 0 else "#D62728" for d in deltas]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Δ Best Overall Accuracy vs line baseline (pp)")
    ax.set_title(title)
    for bar, val in zip(bars, deltas):
        sign = "+" if val >= 0 else ""
        ypos = val + (0.08 if val >= 0 else -0.08)
        va = "bottom" if val >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ypos,
            f"{sign}{val:.2f} pp",
            ha="center",
            va=va,
            fontsize=9,
        )
    pad = max(abs(min(deltas)), abs(max(deltas))) * 0.35 + 0.3
    ax.set_ylim(min(deltas) - pad, max(deltas) + pad)
    save_fig(fig, out_stem)


def plot_two_line_combined(linea: list[dict], lineb: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    labels_a = [f"A: {d['label']}" for d in linea]
    labels_b = [f"B: {d['label']}" for d in lineb]
    all_labels = labels_a + labels_b
    all_vals = [d["best"] for d in linea] + [d["best"] for d in lineb]
    y = np.arange(len(all_labels))
    colors = (
        [METHOD_COLORS.get(d["label"], "#888") for d in linea]
        + [METHOD_COLORS.get(d["label"], "#888") for d in lineb]
    )
    bars = ax.barh(y, all_vals, color=colors, edgecolor="white", height=0.7)
    ax.axvline(linea[0]["best"], color=METHOD_COLORS["Original baseline"], linestyle="--", linewidth=1.2, label="Line A baseline")
    ax.axvline(lineb[0]["best"], color=METHOD_COLORS["Downsampled baseline"], linestyle=":", linewidth=1.2, label="Line B baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(all_labels)
    ax.set_xlabel("Best Overall Accuracy (%)")
    ax.set_title("Two-Line Protocol: Best Overall Classification Accuracy")
    ax.invert_yaxis()
    for bar, val in zip(bars, all_vals):
        ax.text(val + 0.15, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=8)
    ax.legend(loc="lower right")
    ax.set_xlim(min(all_vals) - 1, max(all_vals) + 2)
    save_fig(
        fig,
        FIGURES / "accuracy" / "two_line_best_overall_accuracy",
        "Combined horizontal bar plot for Line A and Line B.",
    )


def load_lineb_geometry() -> list[dict]:
    rows = read_csv(REPORTS / "modelnet40_quality_metrics_thesis_table.csv")
    out = []
    for group in GEOM_LINEB_GROUPS:
        for r in rows:
            if r["group"] == group:
                out.append(
                    {
                        "group": group,
                        "label": GEOM_LINEB_LABELS[GEOM_LINEB_GROUPS.index(group)],
                        "CD": float(r["CD"]),
                        "HD": float(r["HD"]),
                        "NUC": float(r["NUC"]),
                        "exact_P2F": float(r["exact_P2F"]),
                    }
                )
                break
    return out


def plot_geometry_metric(geom: list[dict], metric: str, title: str, out_stem: Path, footnote: str) -> None:
    labels = [g["label"] for g in geom]
    values = [g[metric] for g in geom]
    baseline = values[0]
    colors = bar_colors(labels, labels[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(baseline, color="#333333", linestyle="--", linewidth=1.2, label="Original baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(metric if metric != "exact_P2F" else "exact P2F")
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(values) * 1.15)
    fig.text(0.01, 0.01, footnote, fontsize=8, style="italic")
    save_fig(fig, out_stem)


def plot_geometry_grouped(geom: list[dict]) -> None:
    metrics = ["CD", "HD", "NUC", "exact_P2F"]
    labels = [g["label"] for g in geom]
    x = np.arange(len(labels))
    width = 0.18
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        values = [g[metric] for g in geom]
        bars = ax.bar(x, values, color=[METHOD_COLORS.get(l, "#888") for l in labels], edgecolor="white")
        ax.axhline(values[0], color="#333333", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ylab = "exact P2F" if metric == "exact_P2F" else metric
        ax.set_title(f"{ylab} (lower is better)")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    fig.suptitle("Line B Geometry Metrics vs Original Baseline Reference", y=1.02)
    fig.text(0.01, 0.01, "Lower is better. exact P2F computed against dense mesh surfaces.", fontsize=8, style="italic")
    fig.tight_layout()
    save_fig(fig, FIGURES / "geometry" / "lineB_geometry_metrics_grouped")


def load_geom_cls() -> list[dict]:
    rows = read_csv(REPORTS / "modelnet40_geometry_vs_classification_final_summary.csv")
    out = []
    for r in rows:
        out.append(
            {
                "line": r["Line"],
                "method": r["Method"],
                "short": short_label_lineb(r["Method"]) if r["Line"] == "B" else short_label_linea(r["Method"]),
                "CD": float(r["CD"]),
                "HD": float(r["HD"]),
                "NUC": float(r["NUC"]),
                "exact_P2F": float(r["exact_P2F"]),
                "best_acc": float(r["Best Overall Accuracy"]) * 100,
            }
        )
    return out


def plot_scatter(geom_cls: list[dict], metric: str, out_stem: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    line_styles = {"A": "o", "B": "s"}
    for row in geom_cls:
        if row["line"] == "B" and "baseline" in row["method"].lower() and "Downsampled" in row["method"]:
            marker = "D"
            size = 90
            alpha = 1.0
        elif row["line"] == "A" and "Original baseline" in row["method"]:
            marker = "^"
            size = 90
            alpha = 0.85
        else:
            marker = line_styles.get(row["line"], "o")
            size = 70 if row["line"] == "B" else 55
            alpha = 1.0 if row["line"] == "B" else 0.65
        color = METHOD_COLORS.get(row["short"], "#888888")
        ax.scatter(
            row[metric],
            row["best_acc"],
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.4,
            zorder=3 if row["line"] == "B" else 2,
        )
        offset = (0.002, 0.15) if row["line"] == "B" else (-0.003, -0.35)
        ax.annotate(
            row["short"] + ("" if row["line"] == "B" else " (A)"),
            (row[metric], row["best_acc"]),
            textcoords="offset points",
            xytext=(offset[0] * 1000, offset[1] * 10),
            fontsize=8,
        )
    ylab = "exact P2F" if metric == "exact_P2F" else metric
    ax.set_xlabel(f"{ylab} (lower is better)")
    ax.set_ylabel("Best Overall Accuracy (%)")
    ax.set_title(f"{ylab} vs Best Overall Classification Accuracy")
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888", markersize=8, label="Line B upsampling"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markersize=7, alpha=0.6, label="Line A upsampling"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=METHOD_COLORS["Downsampled baseline"], markersize=8, label="Line B baseline"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=METHOD_COLORS["Original baseline"], markersize=8, label="Line A baseline"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
    fig.text(0.01, 0.01, "Geometric quality and classification accuracy are not perfectly aligned.", fontsize=8, style="italic")
    save_fig(fig, out_stem)


def point_size(n: int) -> float:
    if n <= 256:
        return 12.0
    if n <= 1024:
        return 4.0
    return 1.2


def load_points(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    return pts.astype(np.float32)


def project_points(pts: np.ndarray, elev: float = 20, azim: float = -60) -> np.ndarray:
    """Orthographic 3D→2D projection (matplotlib 3D unavailable on this host)."""
    elev_r = np.radians(elev)
    azim_r = np.radians(azim)
    cz, sz = np.cos(azim_r), np.sin(azim_r)
    ce, se = np.cos(elev_r), np.sin(elev_r)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=np.float32)
    rotated = pts @ (rx @ rz).T
    return rotated[:, :2]


def compute_limits_2d(point_sets: list[np.ndarray]) -> tuple[float, float, float, float]:
    all_pts = np.vstack(point_sets)
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    center = (mins + maxs) / 2
    radius = np.max(maxs - mins) / 2 * 1.05
    return center[0] - radius, center[0] + radius, center[1] - radius, center[1] + radius


def plot_pointcloud_row(
    panels: list[tuple[str, np.ndarray, int]],
    suptitle: str,
    out_stem: Path,
    elev: float = 20,
    azim: float = -60,
) -> bool:
    valid = [(t, p, n) for t, p, n in panels if p is not None]
    if len(valid) < len(panels):
        missing = [t for t, p, n in panels if p is None]
        FAILURES.append(f"Point cloud comparison {out_stem.name}: missing {missing}")
        if not valid:
            return False

    projected = [(t, project_points(p, elev, azim), p, n) for t, p, n in valid]
    limits = compute_limits_2d([pr for _, pr, _, _ in projected])
    fig, axes = plt.subplots(1, len(projected), figsize=(3.2 * len(projected), 3.5))
    if len(projected) == 1:
        axes = [axes]
    for ax, (title, pr, pts, n) in zip(axes, projected):
        ax.scatter(pr[:, 0], pr[:, 1], s=point_size(n), c=pts[:, 2], cmap="viridis", alpha=0.85, linewidths=0)
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(suptitle + " (orthographic 3D view)", fontsize=11, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_stem)
    return True


def lineb_panels(class_name: str, sample_id: str) -> list[tuple[str, np.ndarray | None, int]]:
    split = "test"
    base = PROJECT_ROOT / "datasets"
    return [
        (
            "Original 1024",
            load_points(base / "modelnet40_original" / split / class_name / f"{sample_id}.npy"),
            1024,
        ),
        (
            "Downsampled x4 256",
            load_points(base / "modelnet40_downsampled_x4" / split / class_name / f"{sample_id}.npy"),
            256,
        ),
        (
            "EAR 1024",
            load_points(base / "lineB_downsampled_x4_up" / "strict_N" / "ear" / split / class_name / f"{sample_id}.npy"),
            1024,
        ),
        (
            "PDANS 1024",
            load_points(base / "lineB_downsampled_x4_up" / "strict_N" / "pdans" / split / class_name / f"{sample_id}.npy"),
            1024,
        ),
        (
            "PU-Net 1024",
            load_points(base / "lineB_downsampled_x4_up" / "strict_N" / "pu_net" / split / class_name / f"{sample_id}.npy"),
            1024,
        ),
        (
            "PU-GCN 1024",
            load_points(base / "lineB_downsampled_x4_up" / "strict_N" / "pu_gcn" / split / class_name / f"{sample_id}.npy"),
            1024,
        ),
    ]


def linea_panels(class_name: str, sample_id: str) -> list[tuple[str, np.ndarray | None, int]]:
    split = "test"
    base = PROJECT_ROOT / "datasets"
    return [
        (
            "Original 1024",
            load_points(base / "modelnet40_original" / split / class_name / f"{sample_id}.npy"),
            1024,
        ),
        (
            "EAR 4096",
            load_points(base / "lineA_original_up" / "strict_4N" / "ear" / split / class_name / f"{sample_id}.npy"),
            4096,
        ),
        (
            "PDANS 4096",
            load_points(base / "lineA_original_up" / "strict_4N" / "pdans" / split / class_name / f"{sample_id}.npy"),
            4096,
        ),
        (
            "PU-Net 4096",
            load_points(base / "lineA_original_up" / "strict_4N" / "pu_net" / split / class_name / f"{sample_id}.npy"),
            4096,
        ),
        (
            "PU-GCN 4096",
            load_points(base / "lineA_original_up" / "strict_4N" / "pu_gcn" / split / class_name / f"{sample_id}.npy"),
            4096,
        ),
    ]


def plot_interactive_html(
    panels: list[tuple[str, np.ndarray | None, int]],
    title: str,
    out_path: Path,
) -> None:
    if not HAS_PLOTLY:
        return
    valid = [(t, p, n) for t, p, n in panels if p is not None]
    if not valid:
        return
    cols = len(valid)
    fig = make_subplots(
        rows=1,
        cols=cols,
        specs=[[{"type": "scatter3d"}] * cols],
        subplot_titles=[t for t, _, _ in valid],
    )
    for i, (t, pts, n) in enumerate(valid, start=1):
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=2 if n > 1024 else (4 if n > 256 else 6), color=pts[:, 2], colorscale="Viridis", opacity=0.85),
                name=t,
            ),
            row=1,
            col=i,
        )
    fig.update_layout(title=title, height=450, width=320 * cols, showlegend=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    GENERATED.append({"path_png": str(out_path), "path_pdf": "", "caption_note": "Interactive HTML viewer"})


def generate_accuracy_figures() -> None:
    linea = load_linea_cls()
    lineb = load_lineb_cls()
    plot_accuracy_bars(
        linea,
        0,
        "Line A: Original baseline vs Original + Upsampling",
        "Best Overall Accuracy (%)",
        "Original baseline 1024 is the reference.",
        FIGURES / "accuracy" / "lineA_best_overall_accuracy",
        "Original baseline (91.95%)",
    )
    plot_accuracy_bars(
        lineb,
        0,
        "Line B: Downsampled x4 baseline vs Upsampling",
        "Best Overall Accuracy (%)",
        "PU-Net is the only method slightly above the downsampled baseline.",
        FIGURES / "accuracy" / "lineB_best_overall_accuracy",
        "Downsampled x4 baseline (90.85%)",
    )
    plot_two_line_combined(linea, lineb)
    plot_delta_bars(
        linea,
        "Line A: Δ Best Overall Accuracy vs Original Baseline",
        FIGURES / "accuracy" / "lineA_delta_best_overall_vs_baseline",
    )
    plot_delta_bars(
        lineb,
        "Line B: Δ Best Overall Accuracy vs Downsampled x4 Baseline",
        FIGURES / "accuracy" / "lineB_delta_best_overall_vs_baseline",
    )


def generate_geometry_figures() -> None:
    geom = load_lineb_geometry()
    foot = "Lower is better."
    plot_geometry_metric(
        geom,
        "CD",
        "Line B: Chamfer Distance (CD) vs Original Baseline",
        FIGURES / "geometry" / "lineB_cd_comparison",
        foot + " PU-GCN and PU-Net are closest to Original baseline.",
    )
    plot_geometry_metric(
        geom,
        "HD",
        "Line B: Hausdorff Distance (HD) vs Original Baseline",
        FIGURES / "geometry" / "lineB_hd_comparison",
        foot + " PU-Net shows a more balanced HD.",
    )
    plot_geometry_metric(
        geom,
        "NUC",
        "Line B: Normalized Uniformity Coefficient (NUC) vs Original Baseline",
        FIGURES / "geometry" / "lineB_nuc_comparison",
        foot + " PU-GCN NUC is noticeably higher than other methods.",
    )
    plot_geometry_metric(
        geom,
        "exact_P2F",
        "Line B: exact Point-to-Surface (P2F) vs Original Baseline",
        FIGURES / "geometry" / "lineB_exact_p2f_comparison",
        "Lower is better. exact P2F is computed against dense mesh surfaces.",
    )
    plot_geometry_grouped(geom)


def generate_scatter_figures() -> None:
    geom_cls = load_geom_cls()
    for metric, stem in [
        ("CD", "cd_vs_best_accuracy"),
        ("HD", "hd_vs_best_accuracy"),
        ("NUC", "nuc_vs_best_accuracy"),
        ("exact_P2F", "p2f_vs_best_accuracy"),
    ]:
        plot_scatter(geom_cls, metric, FIGURES / "geometry_vs_classification" / stem)


def generate_pointcloud_figures() -> None:
    for class_name, sample_id in REPRESENTATIVE_SAMPLES:
        lineb = lineb_panels(class_name, sample_id)
        plot_pointcloud_row(
            lineb,
            f"Line B Point Cloud Comparison — {class_name} / {sample_id}",
            FIGURES / "pointcloud_examples" / f"lineB_{class_name}_{sample_id}_comparison",
        )
        linea = linea_panels(class_name, sample_id)
        plot_pointcloud_row(
            linea,
            f"Line A Point Cloud Comparison — {class_name} / {sample_id}",
            FIGURES / "pointcloud_examples" / f"lineA_{class_name}_{sample_id}_comparison",
        )
        if HAS_PLOTLY:
            plot_interactive_html(
                lineb,
                f"Line B — {class_name} / {sample_id}",
                FIGURES / "pointcloud_examples" / f"interactive_lineB_{class_name}_{sample_id}.html",
            )
            plot_interactive_html(
                linea,
                f"Line A — {class_name} / {sample_id}",
                FIGURES / "pointcloud_examples" / f"interactive_lineA_{class_name}_{sample_id}.html",
            )


FIGURE_INDEX_ENTRIES = [
    ("accuracy/lineA_best_overall_accuracy", "modelnet40_pointnet2_lineA_classification_summary.csv", "Line A best overall accuracy by upsampling method.", "Results — Line A classification"),
    ("accuracy/lineB_best_overall_accuracy", "modelnet40_pointnet2_lineB_classification_summary.csv", "Line B best overall accuracy; PU-Net only method above downsampled baseline.", "Results — Line B classification"),
    ("accuracy/two_line_best_overall_accuracy", "modelnet40_pointnet2_final_two_line_classification_summary.csv", "Combined two-line best overall accuracy comparison.", "Results — Overview"),
    ("accuracy/lineA_delta_best_overall_vs_baseline", "modelnet40_pointnet2_lineA_classification_summary.csv", "Line A delta best overall vs Original baseline (pp).", "Results — Line A classification"),
    ("accuracy/lineB_delta_best_overall_vs_baseline", "modelnet40_pointnet2_lineB_classification_summary.csv", "Line B delta best overall vs downsampled baseline (pp).", "Results — Line B classification"),
    ("geometry/lineB_cd_comparison", "modelnet40_quality_metrics_thesis_table.csv", "Line B CD vs Original baseline reference.", "Results — Geometry quality"),
    ("geometry/lineB_hd_comparison", "modelnet40_quality_metrics_thesis_table.csv", "Line B HD vs Original baseline reference.", "Results — Geometry quality"),
    ("geometry/lineB_nuc_comparison", "modelnet40_quality_metrics_thesis_table.csv", "Line B NUC; highlights elevated PU-GCN uniformity deviation.", "Results — Geometry quality"),
    ("geometry/lineB_exact_p2f_comparison", "modelnet40_quality_metrics_thesis_table.csv", "Line B exact P2F (mesh-surface distance).", "Results — Geometry quality"),
    ("geometry/lineB_geometry_metrics_grouped", "modelnet40_quality_metrics_thesis_table.csv", "Grouped Line B geometry metrics panel.", "Results — Geometry quality"),
    ("geometry_vs_classification/cd_vs_best_accuracy", "modelnet40_geometry_vs_classification_final_summary.csv", "CD vs best overall accuracy scatter.", "Discussion — Geometry vs classification"),
    ("geometry_vs_classification/hd_vs_best_accuracy", "modelnet40_geometry_vs_classification_final_summary.csv", "HD vs best overall accuracy scatter.", "Discussion — Geometry vs classification"),
    ("geometry_vs_classification/nuc_vs_best_accuracy", "modelnet40_geometry_vs_classification_final_summary.csv", "NUC vs best overall accuracy scatter.", "Discussion — Geometry vs classification"),
    ("geometry_vs_classification/p2f_vs_best_accuracy", "modelnet40_geometry_vs_classification_final_summary.csv", "exact P2F vs best overall accuracy scatter.", "Discussion — Geometry vs classification"),
]


def write_figure_index() -> None:
    lines = [
        "# ModelNet40 Thesis Figure Index",
        "",
        f"- Generated at: {utc_now()}",
        f"- Root: `{FIGURES}`",
        "",
        "| Figure | Source data | Caption | Thesis section |",
        "| --- | --- | --- | --- |",
    ]
    for rel, source, caption, section in FIGURE_INDEX_ENTRIES:
        lines.append(
            f"| `{rel}.png` / `.pdf` | `{source}` | {caption} | {section} |"
        )
    for cls, sid in REPRESENTATIVE_SAMPLES:
        lines.append(
            f"| `pointcloud_examples/lineB_{cls}_{sid}_comparison.png` | test split `.npy` point clouds | "
            f"Qualitative Line B comparison for {cls} ({sid}). | Qualitative examples |"
        )
        lines.append(
            f"| `pointcloud_examples/lineA_{cls}_{sid}_comparison.png` | test split `.npy` point clouds | "
            f"Qualitative Line A comparison for {cls} ({sid}). | Qualitative examples |"
        )
    if HAS_PLOTLY:
        for cls, sid in REPRESENTATIVE_SAMPLES:
            lines.append(
                f"| `pointcloud_examples/interactive_lineB_{cls}_{sid}.html` | test split `.npy` | "
                f"Interactive Line B viewer for {cls} ({sid}). | Supplementary |"
            )
    (FIGURES / "figure_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_visualization_summary() -> None:
    n_acc = len(list((FIGURES / "accuracy").glob("*.png"))) if (FIGURES / "accuracy").exists() else 0
    n_geom = len(list((FIGURES / "geometry").glob("*.png"))) if (FIGURES / "geometry").exists() else 0
    n_scatter = len(list((FIGURES / "geometry_vs_classification").glob("*.png"))) if (FIGURES / "geometry_vs_classification").exists() else 0
    n_pc = len(list((FIGURES / "pointcloud_examples").glob("line*.png"))) if (FIGURES / "pointcloud_examples").exists() else 0
    n_html = len(list((FIGURES / "pointcloud_examples").glob("interactive*.html"))) if (FIGURES / "pointcloud_examples").exists() else 0

    content = f"""# ModelNet40 Visualization Summary

- Generated at: {utc_now()}
- Project: `{PROJECT_ROOT}`

## 1. Accuracy plots generated

- `{FIGURES / "accuracy"}` — {n_acc} PNG figures
- Line A best overall accuracy bar plot
- Line B best overall accuracy bar plot
- Two-line combined horizontal bar plot
- Line A / Line B delta best overall vs baseline (percentage points)

## 2. Geometry plots generated

- `{FIGURES / "geometry"}` — {n_geom} PNG figures
- Individual CD, HD, NUC, exact P2F bar plots (Line B vs Original baseline reference)
- Grouped geometry metrics panel

## 3. Geometry-vs-classification scatter plots generated

- `{FIGURES / "geometry_vs_classification"}` — {n_scatter} PNG figures
- CD, HD, NUC, exact P2F vs best overall accuracy
- Line B methods emphasized; Line A shown with distinct markers

## 4. Point cloud qualitative figures generated

- `{FIGURES / "pointcloud_examples"}` — {n_pc} PNG comparison figures
- Classes: airplane, chair, table, car, sofa (test split first sample per class)
- Line A and Line B multi-method rows with matched camera view (orthographic 3D projection for static PNG/PDF)
- Interactive HTML viewers: {n_html} ({'plotly available; rotatable 3D' if HAS_PLOTLY else 'plotly not used'})

## 5. Key observations

- **Line A:** Original baseline 1024 (91.95% best overall) remains strongest; 4096-point upsampling does not improve classification.
- **Line B:** PU-Net (91.27%) is the only method slightly above the downsampled ×4 baseline (90.85%).
- **Geometry:** PU-GCN and PU-Net have the smallest CD delta vs Original; PU-GCN NUC is clearly elevated.
- **Balance:** PU-Net shows more balanced HD / NUC among Line B upsampling methods.
- **Alignment:** Geometry and classification are not perfectly aligned — best CD (PU-GCN) ≠ best classifier (PU-Net).

## 6. Failures

"""
    if FAILURES:
        content += "\n".join(f"- {f}" for f in FAILURES)
    else:
        content += "- None reported.\n"
    content += (
        "- Note: static point cloud PNG/PDF use orthographic 3D projection because "
        "matplotlib Axes3D is unavailable (mixed system/user matplotlib install). "
        "Interactive Plotly HTML provides rotatable 3D views.\n"
    )
    content += f"\n\n## 7. Total figures saved\n\n- {len(GENERATED)} figure artifacts (PNG/PDF/HTML)\n"
    (REPORTS / "modelnet40_visualization_summary.md").write_text(content, encoding="utf-8")


def write_thesis_captions() -> None:
    captions = f"""# ModelNet40 Thesis Figure Captions

- Generated at: {utc_now()}
- Style: concise, thesis-ready; deltas in percentage points (pp)

## Line A classification

**Figure: Line A best overall accuracy** (`accuracy/lineA_best_overall_accuracy`)

Best overall test accuracy of PointNet++ on ModelNet40 under Line A (Original 1024 baseline vs Original + Upsampling at 4096 points). The dashed line marks the Original baseline (91.95%). None of the upsampling methods exceed this reference.

## Line B classification

**Figure: Line B best overall accuracy** (`accuracy/lineB_best_overall_accuracy`)

Best overall test accuracy under Line B (Downsampled ×4 baseline at 256 points vs upsampling to 1024 points). The dashed line marks the downsampled baseline (90.85%). PU-Net (+0.42 pp) is the only upsampling method slightly above baseline.

## Two-line overview

**Figure: Two-line combined accuracy** (`accuracy/two_line_best_overall_accuracy`)

Horizontal comparison of best overall accuracy across Line A and Line B protocols. Dashed and dotted vertical lines indicate each line's respective baseline.

## Delta accuracy

**Figure: Line A delta best overall** (`accuracy/lineA_delta_best_overall_vs_baseline`)

Change in best overall accuracy relative to the Original 1024 baseline (percentage points). All upsampling methods show negative deltas.

**Figure: Line B delta best overall** (`accuracy/lineB_delta_best_overall_vs_baseline`)

Change in best overall accuracy relative to the Downsampled ×4 baseline (percentage points). Only PU-Net shows a positive delta (+0.42 pp).

## Geometry metrics

**Figure: Line B geometry comparison** (`geometry/lineB_cd_comparison`, `lineB_hd_comparison`, `lineB_nuc_comparison`, `lineB_exact_p2f_comparison`, `lineB_geometry_metrics_grouped`)

Geometric quality of Line B upsampling methods compared to the Original 1024 baseline reference (dashed line). CD, HD, NUC, and exact P2F are reported; lower is better. PU-GCN and PU-Net achieve the smallest CD deltas; PU-Net is more balanced on HD and NUC; PU-GCN shows elevated NUC. exact P2F is computed against dense mesh surfaces.

## Geometry vs classification

**Figure: Geometry vs classification scatter** (`geometry_vs_classification/cd_vs_best_accuracy`, etc.)

Scatter plots relating geometric metrics (x-axis; lower is better) to best overall PointNet++ accuracy (y-axis). Square markers: Line B; circles: Line A. PU-GCN achieves strong CD but not the highest accuracy; PU-Net is the best Line B classifier without dominating all geometry metrics. Geometric quality and classification accuracy are not perfectly aligned.

## Point cloud qualitative examples

**Figure: Line B point cloud comparison** (`pointcloud_examples/lineB_<class>_<id>_comparison`)

Qualitative comparison of test-set point clouds under Line B: Original 1024, Downsampled ×4 256, and four upsampling methods at 1024 points. Identical viewpoint and axis limits across methods; point size scaled by point count. Static figures use orthographic 3D projection; interactive HTML allows rotation.

**Figure: Line A point cloud comparison** (`pointcloud_examples/lineA_<class>_<id>_comparison`)

Qualitative comparison under Line A: Original 1024 vs four upsampling methods at 4096 points, shown with matched camera parameters for visual comparison of point distribution and surface coverage.
"""
    (REPORTS / "modelnet40_thesis_figure_captions.md").write_text(captions, encoding="utf-8")


def main() -> None:
    for sub in ("accuracy", "geometry", "geometry_vs_classification", "pointcloud_examples"):
        (FIGURES / sub).mkdir(parents=True, exist_ok=True)

    generate_accuracy_figures()
    generate_geometry_figures()
    generate_scatter_figures()
    generate_pointcloud_figures()
    write_figure_index()
    write_visualization_summary()
    write_thesis_captions()

    print(f"Generated {len(GENERATED)} figure artifacts under {FIGURES}")
    if FAILURES:
        print("Failures:")
        for f in FAILURES:
            print(" -", f)


if __name__ == "__main__":
    main()
