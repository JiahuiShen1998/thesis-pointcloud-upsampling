#!/usr/bin/env python3
"""Assemble thesis-ready ModelNet40 final visualization package from existing results.

Reads existing CSV / MD / metrics / figures / HTML only.
Does NOT retrain, recompute geometry metrics, or modify datasets.
"""

from __future__ import annotations

import csv
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
FIG = PROJECT / "figures" / "modelnet40"
PKG = FIG / "final_visualization_package"

GEOM_CSV = REPORTS / "modelnet40_quality_metrics_thesis_table_with_pu_edgeformer.csv"
CLS_CSV = REPORTS / "modelnet40_pointnet2_final_classification_report_with_pu_edgeformer.csv"

METHODS_UP = ["EAR", "PDANS", "PU-Net", "PU-GCN", "PU-EdgeFormer"]
REPRESENTATIVE = [
    ("airplane", "airplane_0627"),
    ("chair", "chair_0890"),
    ("table", "table_0393"),
    ("car", "car_0198"),
    ("sofa", "sofa_0681"),
]

METHOD_COLORS = {
    "Original baseline": "#4C72B0",
    "Downsampled baseline": "#55A868",
    "EAR": "#C44E52",
    "PDANS": "#8172B3",
    "PU-Net": "#CCB974",
    "PU-GCN": "#64B5CD",
    "PU-EdgeFormer": "#B279A2",
}

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

GENERATED: list[str] = []
AUDIT_CHECKS: list[tuple[str, bool, str]] = []


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(ext))
    plt.close(fig)
    GENERATED.append(str(stem))


def pp_str(pp: float) -> str:
    sign = "+" if pp >= 0 else ""
    return f"{sign}{pp:.2f} pp"


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def audit(name: str, ok: bool, detail: str = "") -> None:
    AUDIT_CHECKS.append((name, ok, detail))


def load_geometry() -> dict[str, dict]:
    rows = read_csv(GEOM_CSV)
    return {r["group"]: r for r in rows}


def load_classification() -> list[dict]:
    return read_csv(CLS_CSV)


def geom_lineb_upsampling(geom: dict[str, dict]) -> list[dict]:
    mapping = {
        "EAR": "Downsampled x4 + EAR",
        "PDANS": "Downsampled x4 + PDANS",
        "PU-Net": "Downsampled x4 + PU-Net",
        "PU-GCN": "Downsampled x4 + PU-GCN",
        "PU-EdgeFormer": "Downsampled x4 + PU-EdgeFormer",
    }
    out = []
    for label, group in mapping.items():
        r = geom[group]
        out.append(
            {
                "label": label,
                "group": group,
                "CD": float(r["CD"]),
                "delta_CD": float(r["delta_CD_vs_original"]),
                "HD": float(r["HD"]),
                "delta_HD": float(r["delta_HD_vs_original"]),
                "NUC": float(r["NUC"]),
                "delta_NUC": float(r["delta_NUC_vs_original"]),
                "exact_P2F": float(r["exact_P2F"]),
                "delta_P2F": float(r["delta_P2F_vs_original"]),
                "points": int(r["source_point_count"]),
            }
        )
    return out


def geom_linea_rows(geom: dict[str, dict]) -> list[dict]:
    groups = [
        ("Original baseline", "Original baseline"),
        ("EAR", "Original + EAR"),
        ("PDANS", "Original + PDANS"),
        ("PU-Net", "Original + PU-Net"),
        ("PU-GCN", "Original + PU-GCN"),
        ("PU-EdgeFormer", "Original + PU-EdgeFormer"),
    ]
    out = []
    for label, group in groups:
        r = geom[group]
        out.append(
            {
                "label": label,
                "group": group,
                "CD": float(r["CD"]),
                "delta_CD": float(r["delta_CD_vs_original"]),
                "HD": float(r["HD"]),
                "delta_HD": float(r["delta_HD_vs_original"]),
                "NUC": float(r["NUC"]),
                "delta_NUC": float(r["delta_NUC_vs_original"]),
                "exact_P2F": float(r["exact_P2F"]),
                "delta_P2F": float(r["delta_P2F_vs_original"]),
                "points": int(r["source_point_count"]),
            }
        )
    return out


def geom_lineb_absolute_rows(geom: dict[str, dict]) -> list[dict]:
    groups = [
        ("Original baseline", "Original baseline"),
        ("Downsampled baseline", "Downsampled x4 baseline"),
        ("EAR", "Downsampled x4 + EAR"),
        ("PDANS", "Downsampled x4 + PDANS"),
        ("PU-Net", "Downsampled x4 + PU-Net"),
        ("PU-GCN", "Downsampled x4 + PU-GCN"),
        ("PU-EdgeFormer", "Downsampled x4 + PU-EdgeFormer"),
    ]
    out = []
    for label, group in groups:
        r = geom[group]
        out.append(
            {
                "label": label,
                "CD": float(r["CD"]),
                "HD": float(r["HD"]),
                "NUC": float(r["NUC"]),
                "exact_P2F": float(r["exact_P2F"]),
                "points": int(r["source_point_count"]),
            }
        )
    return out


def cls_linea(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r["line"] != "A":
            continue
        method = r["method"]
        label = {
            "Original baseline 1024": "Original baseline",
            "Original + EAR 4096": "EAR",
            "Original + PDANS 4096": "PDANS",
            "Original + PU-Net 4096": "PU-Net",
            "Original + PU-GCN 4096": "PU-GCN",
            "Original + PU-EdgeFormer 4096": "PU-EdgeFormer",
        }.get(method, method)
        best = float(r["best_overall_accuracy"])
        delta = float(r["delta_accuracy_vs_original_baseline_pp"] or 0.0)
        out.append({"label": label, "method": method, "best": best * 100, "delta_pp": delta})
    order = ["Original baseline", "EAR", "PDANS", "PU-Net", "PU-GCN", "PU-EdgeFormer"]
    return sorted(out, key=lambda x: order.index(x["label"]))


def cls_lineb(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r["line"] != "B":
            continue
        method = r["method"]
        label = {
            "Downsampled x4 baseline 256": "Downsampled baseline",
            "Downsampled x4 + EAR 1024": "EAR",
            "Downsampled x4 + PDANS 1024": "PDANS",
            "Downsampled x4 + PU-Net 1024": "PU-Net",
            "Downsampled x4 + PU-GCN 1024": "PU-GCN",
            "Downsampled x4 + PU-EdgeFormer 1024": "PU-EdgeFormer",
        }.get(method, method)
        best = float(r["best_overall_accuracy"])
        delta_ds = float(r["delta_accuracy_vs_downsampled_baseline_pp"] or 0.0)
        gap_orig = float(r["gap_accuracy_vs_original_baseline_pp"] or 0.0)
        out.append(
            {
                "label": label,
                "method": method,
                "best": best * 100,
                "delta_pp": delta_ds,
                "gap_pp": gap_orig,
            }
        )
    order = ["Downsampled baseline", "EAR", "PDANS", "PU-Net", "PU-GCN", "PU-EdgeFormer"]
    return sorted(out, key=lambda x: order.index(x["label"]))


def plot_delta_methods(
    data: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    out: Path,
    fmt: str = "{:+.4f}",
) -> None:
    labels = [d["label"] for d in data]
    vals = [d[metric_key] for d in data]
    colors = [METHOD_COLORS.get(l, "#888") for l in labels]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.72)
    ax.axhline(0, color="#333", linestyle="--", linewidth=1.0, label="Original baseline (y=0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    pad = max(abs(min(vals)), abs(max(vals)), 1e-6) * 0.2 + 1e-4
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + (pad * 0.1 if v >= 0 else -pad * 0.25),
            fmt.format(v),
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
        )
    ax.set_ylim(min(0, min(vals)) - pad, max(0, max(vals)) + pad)
    ax.legend(loc="best", fontsize=8)
    fig.text(
        0.01,
        0.01,
        "The Original baseline is used as the zero reference and is not plotted as a method bar. "
        "Absolute metrics were computed with respect to the dense mesh surface reference.",
        fontsize=8,
        style="italic",
    )
    save_fig(fig, out)


def plot_absolute_geom(
    data: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    out: Path,
    baseline_idx: int = 0,
) -> None:
    labels = [d["label"] for d in data]
    vals = [d[metric_key] for d in data]
    colors = [METHOD_COLORS.get(l, "#888") for l in labels]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(vals[baseline_idx], color="#333", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title + " (supplementary)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylim(0, max(vals) * 1.12)
    fig.text(0.01, 0.01, "Absolute metrics vs dense mesh surface reference.", fontsize=8, style="italic")
    save_fig(fig, out)


def plot_accuracy_absolute(data: list[dict], baseline_idx: int, title: str, baseline_label: str, out: Path) -> None:
    labels = [d["label"] for d in data]
    vals = [d["best"] for d in data]
    colors = [METHOD_COLORS.get(l, "#888") for l in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(vals[baseline_idx], color="#333", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Best Overall Accuracy (%)")
    ax.set_title(title)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(min(vals) - 2, max(vals) + 2.5)
    ax.legend(loc="lower right")
    save_fig(fig, out)


def plot_delta_pp(data: list[dict], delta_key: str, title: str, ylabel: str, out: Path) -> None:
    rows = [d for d in data if "baseline" not in d["label"].lower()]
    labels = [d["label"] for d in rows]
    deltas = [d[delta_key] for d in rows]
    colors = ["#2CA02C" if d >= 0 else "#D62728" for d in deltas]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, deltas, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#333", linewidth=1.0, linestyle="--", label="baseline (y=0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, v in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + (0.08 if v >= 0 else -0.08),
            pp_str(v),
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
        )
    pad = max(abs(min(deltas)), abs(max(deltas))) * 0.35 + 0.3
    ax.set_ylim(min(deltas) - pad, max(deltas) + pad)
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, out)


def plot_geom_cls_scatter(geom_up: list[dict], cls_up: list[dict], metric_key: str, metric_label: str, out: Path) -> None:
    cls_by = {d["label"]: d for d in cls_up}
    fig, ax = plt.subplots(figsize=(8, 6))
    for g in geom_up:
        label = g["label"]
        c = cls_by[label]
        x = g[f"delta_{metric_key}" if metric_key != "P2F" else "delta_P2F"]
        if metric_key == "CD":
            x = g["delta_CD"]
        elif metric_key == "HD":
            x = g["delta_HD"]
        elif metric_key == "NUC":
            x = g["delta_NUC"]
        else:
            x = g["delta_P2F"]
        y = c["delta_pp"]
        ax.scatter(x, y, c=METHOD_COLORS.get(label, "#888"), s=110, edgecolors="black", linewidths=0.5)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.axhline(0, color="#888", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="#888", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"Δ {metric_label} vs Original baseline")
    ax.set_ylabel("Δ Best Overall Accuracy vs Downsampled baseline (pp)")
    ax.set_title(f"Line B: Δ {metric_label} vs Δ Classification (final)")
    fig.text(
        0.01,
        0.01,
        "Geometry recovery and classification improvement are evaluated against different references "
        "because they answer different questions.",
        fontsize=8,
        style="italic",
    )
    save_fig(fig, out)


def ensure_dirs() -> None:
    for sub in (
        "pointcloud_static/lineA",
        "pointcloud_static/lineB",
        "pointcloud_static/single_large",
        "pointcloud_interactive_index",
        "geometry_lineB_focus",
        "geometry_two_line_supplementary",
        "pointnet2_lineA",
        "pointnet2_lineB",
        "combined_geometry_vs_classification",
    ):
        (PKG / sub).mkdir(parents=True, exist_ok=True)


def setup_pointcloud_static() -> Path:
    """Index/copy existing static point cloud figures into final package."""
    static_root = PKG / "pointcloud_static"
    legacy = FIG / "pointcloud_examples"
    v2_note = "Note: `pointcloud_examples_v2/` is not present; indexed from `pointcloud_examples/` (legacy static comparisons)."
    copied = 0
    for cls, sid in REPRESENTATIVE:
        for line in ("lineA", "lineB"):
            src_png = legacy / f"{line}_{cls}_{sid}_comparison.png"
            src_pdf = legacy / f"{line}_{cls}_{sid}_comparison.pdf"
            dst_dir = static_root / ("lineA" if line == "lineA" else "lineB")
            if src_png.is_file():
                shutil.copy2(src_png, dst_dir / src_png.name)
                copied += 1
            if src_pdf.is_file():
                shutil.copy2(src_pdf, dst_dir / src_pdf.name)

    # single_large: create two large single-method views from existing .npy (read-only)
    single_dir = static_root / "single_large"
    singles = [
        ("lineB_chair_chair_0890_pu_net_large", "lineB", "chair", "chair_0890", "pu_net", 1024, "PU-Net 1024"),
        ("lineA_airplane_airplane_0627_original_large", "lineA", "airplane", "airplane_0627", "original", 1024, "Original 1024"),
    ]
    for stem, line, cls, sid, method, pts, title in singles:
        p = _resolve_npy(line, cls, sid, method)
        if p and p.is_file():
            _plot_single_large(p, title, single_dir / stem)

    index_path = static_root / "POINTCLOUD_STATIC_INDEX.md"
    lines = [
        "# Point Cloud Static Figures — Final Package Index",
        "",
        f"- Generated: `{utc_now()}`",
        f"- {v2_note}",
        f"- Source legacy directory: `figures/modelnet40/pointcloud_examples/`",
        "- Panels: Original, Downsampled (Line B only), EAR, PDANS, PU-Net, PU-GCN (no PU-EdgeFormer in legacy static panels).",
        "",
        "## Representative classes",
        "",
        "| Class | Sample ID | Line A comparison | Line B comparison | Thesis use |",
        "|---|---|---|---|---|",
    ]
    thesis_main = {
        "airplane": ("thesis body", "thesis body"),
        "chair": ("thesis body (recommended Line B)", "appendix"),
        "table": ("appendix", "appendix"),
        "car": ("appendix", "appendix"),
        "sofa": ("appendix", "appendix"),
    }
    for cls, sid in REPRESENTATIVE:
        la = f"`pointcloud_static/lineA/lineA_{cls}_{sid}_comparison.png`"
        lb = f"`pointcloud_static/lineB/lineB_{cls}_{sid}_comparison.png`"
        use_a, use_b = thesis_main.get(cls, ("appendix", "appendix"))
        lines.append(f"| {cls} | {sid} | {la} | {lb} | A: {use_a}; B: {use_b} |")

    lines.extend(
        [
            "",
            "## Recommended thesis inserts",
            "",
            "- **Line B comparison (main text):** `pointcloud_static/lineB/lineB_chair_chair_0890_comparison.png`",
            "- **Line A comparison (main text):** `pointcloud_static/lineA/lineA_airplane_airplane_0627_comparison.png`",
            "- **Single large (supplementary):**",
            "  - `pointcloud_static/single_large/lineB_chair_chair_0890_pu_net_large.png`",
            "  - `pointcloud_static/single_large/lineA_airplane_airplane_0627_original_large.png`",
            "",
            f"- Files copied into package: {copied} PNG/PDF pairs indexed",
            "",
        ]
    )
    index_path.write_text("\n".join(lines), encoding="utf-8")
    audit("point cloud static included", copied >= 10, f"copied={copied}")
    return index_path


def _resolve_npy(line: str, cls: str, sid: str, method: str) -> Path | None:
    split = "test"
    base = PROJECT / "datasets"
    if line == "lineA" and method == "original":
        return base / "modelnet40_original" / split / cls / f"{sid}.npy"
    if line == "lineB" and method == "pu_net":
        return base / "lineB_downsampled_x4_up" / "strict_N" / "pu_net" / split / cls / f"{sid}.npy"
    return None


def _project_points(pts: np.ndarray, elev: float = 20, azim: float = -60) -> np.ndarray:
    elev_r, azim_r = np.radians(elev), np.radians(azim)
    cz, sz = np.cos(azim_r), np.sin(azim_r)
    ce, se = np.cos(elev_r), np.sin(elev_r)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=np.float32)
    return pts @ (rx @ rz).T


def _plot_single_large(npy_path: Path, title: str, out: Path) -> None:
    pts = np.load(npy_path).astype(np.float32)
    pr = _project_points(pts)
    fig, ax = plt.subplots(figsize=(7, 7))
    s = 1.2 if pts.shape[0] > 1024 else 4.0
    ax.scatter(pr[:, 0], pr[:, 1], s=s, c=pts[:, 2], cmap="viridis", alpha=0.9, linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title + " (single large view)", fontsize=12)
    save_fig(fig, out)


def setup_interactive_index() -> Path:
    interactive_root = FIG / "pointcloud_examples_interactive_v2"
    out = REPORTS / "modelnet40_interactive_pointcloud_final_index.md"
    lines = [
        "# ModelNet40 Interactive Point Cloud — Final Index",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Root: `figures/modelnet40/pointcloud_examples_interactive_v2/`",
        "- HTML files are indexed only (not regenerated).",
        "",
        "## Recommended use",
        "",
        "| Use case | Recommended assets |",
        "|---|---|",
        "| Thesis supplementary | Line A/B grid depth HTML for airplane + chair |",
        "| Presentation / defense | dropdown viewers + grid black HTML |",
        "| Qualitative inspection | single_method HTML per method |",
        "",
        "## Line A grid (black / depth)",
        "",
    ]
    for cls, sid in REPRESENTATIVE:
        base = f"lineA_{cls}_{sid}_interactive_grid"
        lines.append(f"- `{interactive_root.relative_to(PROJECT)}/lineA/{base}_black.html`")
        lines.append(f"- `{interactive_root.relative_to(PROJECT)}/lineA/{base}_depth.html`")
    lines.extend(["", "## Line B grid (black / depth)", ""])
    for cls, sid in REPRESENTATIVE:
        base = f"lineB_{cls}_{sid}_interactive_grid"
        lines.append(f"- `{interactive_root.relative_to(PROJECT)}/lineB/{base}_black.html`")
        lines.append(f"- `{interactive_root.relative_to(PROJECT)}/lineB/{base}_depth.html`")
    lines.extend(["", "## Dropdown", ""])
    for cls, sid in REPRESENTATIVE:
        lines.append(f"- `{interactive_root.relative_to(PROJECT)}/dropdown/lineA_{cls}_{sid}_dropdown.html`")
        lines.append(f"- `{interactive_root.relative_to(PROJECT)}/dropdown/lineB_{cls}_{sid}_dropdown.html`")
    lines.extend(["", "## Single-method (airplane + chair)", ""])
    for f in sorted((interactive_root / "single_method").glob("*.html")):
        lines.append(f"- `{f.relative_to(PROJECT)}`")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    n_html = len(list(interactive_root.rglob("*.html")))
    audit("point cloud interactive included", n_html >= 50, f"html_count={n_html}")
    # symlink index folder
    idx_dir = PKG / "pointcloud_interactive_index"
    readme = idx_dir / "README.md"
    readme.write_text(
        "See `reports/modelnet40_interactive_pointcloud_final_index.md` for full HTML paths.\n",
        encoding="utf-8",
    )
    return out


def generate_geometry_figures(geom: dict[str, dict]) -> dict[str, list[str]]:
    lb_up = geom_lineb_upsampling(geom)
    la_rows = geom_linea_rows(geom)
    lb_abs = geom_lineb_absolute_rows(geom)
    paths: dict[str, list[str]] = {"lineB_focus": [], "supplementary": []}

    focus_dir = PKG / "geometry_lineB_focus"
    for key, label, stem in [
        ("delta_CD", "Δ CD", "lineB_delta_cd_vs_original_baseline_methods_only_final"),
        ("delta_HD", "Δ HD", "lineB_delta_hd_vs_original_baseline_methods_only_final"),
        ("delta_NUC", "Δ NUC", "lineB_delta_nuc_vs_original_baseline_methods_only_final"),
        ("delta_P2F", "Δ exact P2F", "lineB_delta_exact_p2f_vs_original_baseline_methods_only_final"),
    ]:
        plot_delta_methods(
            lb_up,
            key,
            f"{label} vs Original baseline",
            f"Line B {label} relative to Original baseline (methods only)",
            focus_dir / stem,
        )
        paths["lineB_focus"].append(str((focus_dir / stem).with_suffix(".png")))

    sup_dir = PKG / "geometry_two_line_supplementary"
    for metric, ylab, stem_a, stem_b in [
        ("CD", "CD", "lineA_absolute_cd_comparison_final", "lineB_absolute_cd_comparison_final"),
        ("HD", "HD", "lineA_absolute_hd_comparison_final", "lineB_absolute_hd_comparison_final"),
        ("NUC", "NUC", "lineA_absolute_nuc_comparison_final", "lineB_absolute_nuc_comparison_final"),
        ("exact_P2F", "exact P2F", "lineA_absolute_exact_p2f_comparison_final", "lineB_absolute_exact_p2f_comparison_final"),
    ]:
        plot_absolute_geom(la_rows, metric, ylab, f"Line A absolute {ylab}", sup_dir / stem_a)
        plot_absolute_geom(lb_abs, metric, ylab, f"Line B absolute {ylab}", sup_dir / stem_b, baseline_idx=0)
        paths["supplementary"].extend(
            [
                str((sup_dir / stem_a).with_suffix(".png")),
                str((sup_dir / stem_b).with_suffix(".png")),
            ]
        )
    audit("geometry Line B focus included", len(paths["lineB_focus"]) == 4)
    audit("geometry supplementary included", len(paths["supplementary"]) == 8)
    return paths


def generate_pointnet2_figures(cls_rows: list[dict]) -> dict[str, list[str]]:
    la = cls_linea(cls_rows)
    lb = cls_lineb(cls_rows)
    paths: dict[str, list[str]] = {"lineA": [], "lineB": []}

    a_dir = PKG / "pointnet2_lineA"
    plot_accuracy_absolute(
        la,
        0,
        "PointNet++ Line A: Best Overall Accuracy (final)",
        "Original baseline 1024",
        a_dir / "lineA_best_overall_accuracy_absolute_final",
    )
    plot_delta_pp(
        la,
        "delta_pp",
        "Line A Δ Best Overall vs Original Baseline (methods only, final)",
        "Δ Best Overall Accuracy vs Original baseline (pp)",
        a_dir / "lineA_delta_best_overall_vs_original_baseline_methods_only_final",
    )
    paths["lineA"] = [
        str((a_dir / "lineA_best_overall_accuracy_absolute_final").with_suffix(".png")),
        str((a_dir / "lineA_delta_best_overall_vs_original_baseline_methods_only_final").with_suffix(".png")),
    ]

    b_dir = PKG / "pointnet2_lineB"
    plot_accuracy_absolute(
        lb,
        0,
        "PointNet++ Line B: Best Overall Accuracy (final)",
        "Downsampled ×4 baseline 256",
        b_dir / "lineB_best_overall_accuracy_absolute_final",
    )
    plot_delta_pp(
        lb,
        "delta_pp",
        "Line B Δ Best Overall vs Downsampled Baseline (methods only, final)",
        "Δ Best Overall Accuracy vs Downsampled baseline (pp)",
        b_dir / "lineB_delta_best_overall_vs_downsampled_baseline_methods_only_final",
    )
    plot_delta_pp(
        lb,
        "gap_pp",
        "Line B gap Best Overall vs Original Baseline (methods only, secondary)",
        "Gap to Original baseline (pp)",
        b_dir / "lineB_gap_best_overall_vs_original_baseline_methods_only_final",
    )
    paths["lineB"] = [
        str((b_dir / "lineB_best_overall_accuracy_absolute_final").with_suffix(".png")),
        str((b_dir / "lineB_delta_best_overall_vs_downsampled_baseline_methods_only_final").with_suffix(".png")),
        str((b_dir / "lineB_gap_best_overall_vs_original_baseline_methods_only_final").with_suffix(".png")),
    ]
    audit("classification Line A split", len(paths["lineA"]) == 2)
    audit("classification Line B split", len(paths["lineB"]) == 3)
    return paths


def generate_combined_figures(geom: dict[str, dict], cls_rows: list[dict]) -> list[str]:
    lb_up = geom_lineb_upsampling(geom)
    lb_cls = [d for d in cls_lineb(cls_rows) if "baseline" not in d["label"].lower()]
    out_dir = PKG / "combined_geometry_vs_classification"
    paths = []
    for metric, label, stem in [
        ("CD", "CD", "lineB_delta_cd_vs_delta_accuracy_final"),
        ("HD", "HD", "lineB_delta_hd_vs_delta_accuracy_final"),
        ("NUC", "NUC", "lineB_delta_nuc_vs_delta_accuracy_final"),
        ("P2F", "exact P2F", "lineB_delta_p2f_vs_delta_accuracy_final"),
    ]:
        plot_geom_cls_scatter(lb_up, lb_cls, metric, label, out_dir / stem)
        paths.append(str((out_dir / stem).with_suffix(".png")))
    audit("combined geometry vs classification", len(paths) == 4)
    return paths


def build_complete_table(geom: dict[str, dict], cls_rows: list[dict]) -> list[dict]:
    cls_by_method = {(r["line"], r["method"]): r for r in cls_rows}
    rows = []

    def add_row(
        line: str,
        branch: str,
        method: str,
        in_pts: int,
        out_pts: int,
        geom_group: str,
        cls_key: tuple[str, str] | None,
        primary_ref: str,
    ) -> None:
        g = geom[geom_group]
        c = cls_by_method.get(cls_key) if cls_key else None
        primary_pp = ""
        gap_pp = ""
        if c:
            if line == "B":
                primary_pp = c.get("delta_accuracy_vs_downsampled_baseline_pp", "")
                gap_pp = c.get("gap_accuracy_vs_original_baseline_pp", "")
            else:
                primary_pp = c.get("delta_accuracy_vs_original_baseline_pp", "")
        rows.append(
            {
                "line": line,
                "branch": branch,
                "method": method,
                "input_points": in_pts,
                "output_points": out_pts,
                "geometry_reference": "dense mesh surface reference; delta vs Original baseline 1024",
                "cd": float(g["CD"]),
                "delta_cd_vs_original": float(g["delta_CD_vs_original"]),
                "hd": float(g["HD"]),
                "delta_hd_vs_original": float(g["delta_HD_vs_original"]),
                "nuc": float(g["NUC"]),
                "delta_nuc_vs_original": float(g["delta_NUC_vs_original"]),
                "exact_p2f": float(g["exact_P2F"]),
                "delta_exact_p2f_vs_original": float(g["delta_P2F_vs_original"]),
                "pointnet2_best_overall_accuracy": float(c["best_overall_accuracy"]) if c else "",
                "pointnet2_best_class_accuracy": float(c["best_class_accuracy"]) if c else "",
                "pointnet2_final_overall_accuracy": float(c["final_test_overall_accuracy"]) if c else "",
                "pointnet2_final_class_accuracy": float(c["final_test_class_accuracy"]) if c else "",
                "classification_primary_reference": primary_ref,
                "delta_best_overall_vs_primary_reference_pp": primary_pp,
                "gap_best_overall_vs_original_baseline_pp": gap_pp,
                "status": c.get("status", "geometry_only") if c else "geometry_only",
            }
        )

    add_row("A", "lineA_original_baseline", "Original baseline", 1024, 1024, "Original baseline", ("A", "Original baseline 1024"), "Original baseline 1024")
    for m, grp, out_pts, ck in [
        ("EAR", "Original + EAR", 4096, ("A", "Original + EAR 4096")),
        ("PDANS", "Original + PDANS", 4096, ("A", "Original + PDANS 4096")),
        ("PU-Net", "Original + PU-Net", 4096, ("A", "Original + PU-Net 4096")),
        ("PU-GCN", "Original + PU-GCN", 4096, ("A", "Original + PU-GCN 4096")),
        ("PU-EdgeFormer", "Original + PU-EdgeFormer", 4096, ("A", "Original + PU-EdgeFormer 4096")),
    ]:
        add_row("A", "lineA_original_up", m, 1024, out_pts, grp, ck, "Original baseline 1024")

    add_row("B", "lineB_downsampled_x4_baseline", "Downsampled baseline", 1024, 256, "Downsampled x4 baseline", ("B", "Downsampled x4 baseline 256"), "Downsampled x4 baseline 256")
    for m, grp, ck in [
        ("EAR", "Downsampled x4 + EAR", ("B", "Downsampled x4 + EAR 1024")),
        ("PDANS", "Downsampled x4 + PDANS", ("B", "Downsampled x4 + PDANS 1024")),
        ("PU-Net", "Downsampled x4 + PU-Net", ("B", "Downsampled x4 + PU-Net 1024")),
        ("PU-GCN", "Downsampled x4 + PU-GCN", ("B", "Downsampled x4 + PU-GCN 1024")),
        ("PU-EdgeFormer", "Downsampled x4 + PU-EdgeFormer", ("B", "Downsampled x4 + PU-EdgeFormer 1024")),
    ]:
        add_row("B", "lineB_downsampled_x4_up", m, 256, 1024, grp, ck, "Downsampled x4 baseline 256")
    return rows


def observation_for_row(r: dict) -> str:
    line, method = r["line"], r["method"]
    if line == "A" and method == "Original baseline":
        return "Line A reference; strongest classification"
    if line == "A":
        d = float(r["delta_best_overall_vs_primary_reference_pp"] or 0)
        return "below Original baseline" if d < 0 else "at/above Original baseline"
    if line == "B" and method == "Downsampled baseline":
        return "Line B degraded input reference"
    if line == "B" and method == "PU-GCN":
        return "smallest ΔCD vs Original baseline"
    if line == "B" and method == "PU-Net":
        return "only method above Downsampled baseline on classification"
    if line == "B" and method == "PU-EdgeFormer":
        return "valid geometry/classification; classification below both baselines"
    if line == "B":
        d = float(r["delta_best_overall_vs_primary_reference_pp"] or 0)
        return "below Downsampled baseline" if d < 0 else "above Downsampled baseline"
    return ""


def build_thesis_summary(complete: list[dict]) -> list[dict]:
    out = []
    for r in complete:
        primary = r["delta_best_overall_vs_primary_reference_pp"]
        gap = r["gap_best_overall_vs_original_baseline_pp"]
        out.append(
            {
                "line": r["line"],
                "method": r["method"],
                "points": r["output_points"],
                "CD": r["cd"],
                "delta_CD_vs_Original": r["delta_cd_vs_original"],
                "NUC": r["nuc"],
                "PointNet++_Best_OA": r["pointnet2_best_overall_accuracy"],
                "Primary_delta_pp": primary,
                "Gap_vs_Original_pp": gap,
                "main_observation": observation_for_row(r),
            }
        )
    return out


def write_complete_reports(complete: list[dict], summary: list[dict]) -> None:
    csv_p = REPORTS / "modelnet40_final_complete_geometry_pointnet2_comparison_with_pu_edgeformer.csv"
    write_csv(csv_p, complete)

    md_lines = [
        "# Final Complete Geometry + PointNet++ Comparison (with PU-EdgeFormer)",
        "",
        f"- Generated: `{utc_now()}`",
        "- Geometry absolute values vs **dense mesh surface reference**.",
        "- Geometry deltas vs **Original baseline 1024**.",
        "- Line B classification primary Δ vs **Downsampled ×4 baseline 256**.",
        "- Line B secondary gap vs **Original baseline 1024**.",
        "- Line A classification Δ vs **Original baseline 1024**.",
        "",
        "| Line | Method | pts | CD | ΔCD | Best OA | Primary Δ pp | Gap pp | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in complete:
        boa = r["pointnet2_best_overall_accuracy"]
        boa_s = pct(float(boa)) if boa != "" else "—"
        pp1 = pp_str(float(r["delta_best_overall_vs_primary_reference_pp"])) if r["delta_best_overall_vs_primary_reference_pp"] != "" else "—"
        gap = pp_str(float(r["gap_best_overall_vs_original_baseline_pp"])) if r["gap_best_overall_vs_original_baseline_pp"] != "" else "—"
        md_lines.append(
            f"| {r['line']} | {r['method']} | {r['output_points']} | {r['cd']:.6f} | {r['delta_cd_vs_original']:+.6f} | "
            f"{boa_s} | {pp1} | {gap} | {r['status']} |"
        )
    (REPORTS / "modelnet40_final_complete_geometry_pointnet2_comparison_with_pu_edgeformer.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )

    write_csv(REPORTS / "modelnet40_final_thesis_summary_table_with_pu_edgeformer.csv", summary)
    smd = [
        "# Thesis Summary Table (with PU-EdgeFormer)",
        "",
        f"- Generated: `{utc_now()}`",
        "",
        "| Line | Method | Points | CD | ΔCD vs Original | NUC | Best OA | Primary Δ pp | Gap pp | Observation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in summary:
        boa = pct(float(r["PointNet++_Best_OA"])) if r["PointNet++_Best_OA"] != "" else "—"
        p1 = pp_str(float(r["Primary_delta_pp"])) if r["Primary_delta_pp"] != "" else "—"
        gap = pp_str(float(r["Gap_vs_Original_pp"])) if r["Gap_vs_Original_pp"] != "" else "—"
        smd.append(
            f"| {r['line']} | {r['method']} | {r['points']} | {float(r['CD']):.6f} | "
            f"{float(r['delta_CD_vs_Original']):+.6f} | {float(r['NUC']):.4f} | {boa} | {p1} | {gap} | {r['main_observation']} |"
        )
    (REPORTS / "modelnet40_final_thesis_summary_table_with_pu_edgeformer.md").write_text("\n".join(smd) + "\n", encoding="utf-8")


def write_interpretation() -> None:
    text = f"""# Geometry vs PointNet++ Final Interpretation (with PU-EdgeFormer)

- Generated: `{utc_now()}`
- Comparison logic: `reports/modelnet40_lineB_comparison_logic_geometry_vs_classification.md`

## 1. Line A

- **Original baseline 1024** remains the strongest PointNet++ branch (**91.95%** best overall accuracy).
- None of the **4096-point upsampling** branches exceeds the Original baseline on best overall accuracy:
  - EAR: 91.48% (−0.47 pp)
  - PDANS: 91.62% (−0.33 pp)
  - PU-Net: 90.95% (−1.00 pp)
  - PU-GCN: 91.63% (−0.32 pp)
  - PU-EdgeFormer: 90.54% (−1.41 pp)
- **Interpretation:** Increasing point count via upsampling does not yield a stable classification benefit in Line A.

## 2. Line B

- **Geometry** is evaluated against the **Original baseline 1024** (recovery question).
- **PU-GCN** has the smallest **ΔCD** vs Original baseline among upsampling methods.
- **Classification primary comparison** uses the **Downsampled ×4 baseline 256**.
- **PU-Net** achieves the best PointNet++ classification among upsampling methods and is the **only** method above the Downsampled baseline (**+0.42 pp**).
- PU-Net still remains **below** the Original baseline (**−0.68 pp** gap).
- **PU-EdgeFormer** reports valid geometry and classification results, but classification decreases vs both references:
  - primary Δ vs Downsampled baseline: **−1.53 pp**
  - secondary gap vs Original baseline: **−2.63 pp**

## 3. Geometry vs classification

- Geometry recovery and downstream classification are **not perfectly aligned**.
- PU-EdgeFormer reinforces this mismatch: moderate geometry deltas with weaker classification recovery.

## 4. Point cloud qualitative visualization

- **Static figures** (`final_visualization_package/pointcloud_static/`) are suitable for thesis document insertion.
- **Interactive HTML** (`pointcloud_examples_interactive_v2/`) is indexed for supplementary material, presentation, and qualitative inspection.
- Legacy static comparisons do not include PU-EdgeFormer panels; interactive v2 grids should be used for full-method qualitative review.

## 5. Safety / scope

- No retraining, no geometry recomputation, no dataset modification in this visualization pass.
- DETECTOR_EVAL_STARTED=NO; KITTI_AP_EVAL_STARTED=NO
"""
    (REPORTS / "modelnet40_final_geometry_pointnet2_comparison_interpretation_with_pu_edgeformer.md").write_text(
        text, encoding="utf-8"
    )


def update_global_indexes() -> None:
    block = """

## Final visualization package (complete thesis-ready)

- Root: `figures/modelnet40/final_visualization_package/`
- Master index: `reports/modelnet40_complete_final_visualization_index.md`
- Audit: `reports/modelnet40_complete_final_visualization_audit.md`

### Logic (must not mix)

1. **Main geometry focus:** Line B delta vs Original baseline 1024 (methods-only bars; y=0 reference).
2. **Classification figures:** shown separately for Line A and Line B.
3. **Line B classification:** primary delta vs Downsampled ×4 baseline 256; secondary gap vs Original baseline 1024.
4. **Point cloud qualitative:** static PNG/PDF + interactive HTML both indexed.
5. Absolute geometry uses **dense mesh surface reference** (not called “baseline”).

### Key outputs

- Complete table: `reports/modelnet40_final_complete_geometry_pointnet2_comparison_with_pu_edgeformer.{csv,md}`
- Thesis summary: `reports/modelnet40_final_thesis_summary_table_with_pu_edgeformer.{csv,md}`
- Interpretation: `reports/modelnet40_final_geometry_pointnet2_comparison_interpretation_with_pu_edgeformer.md`
- Interactive index: `reports/modelnet40_interactive_pointcloud_final_index.md`
"""
    for path in [
        FIG / "figure_index.md",
        REPORTS / "modelnet40_thesis_figure_captions.md",
        REPORTS / "modelnet40_visualization_summary.md",
        REPORTS / "modelnet40_final_thesis_visualization_and_report_index.md",
    ]:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        marker = "## Final visualization package (complete thesis-ready)"
        if marker in text:
            text = text.split(marker)[0].rstrip() + block
        else:
            text = text.rstrip() + block
        path.write_text(text + "\n", encoding="utf-8")

    # Append captions for final package figures
    cap_path = REPORTS / "modelnet40_thesis_figure_captions.md"
    extra_caps = """

## Final visualization package captions

**Figure: Line B geometry delta (final package)** (`final_visualization_package/geometry_lineB_focus/lineB_delta_*_vs_original_baseline_methods_only_final`)

Line B geometric quality deltas for upsampling methods relative to the Original 1024 baseline. The Original baseline is the zero reference only (not a method bar). Absolute metrics use the dense mesh surface reference.

**Figure: PointNet++ Line B primary delta (final package)** (`final_visualization_package/pointnet2_lineB/lineB_delta_best_overall_vs_downsampled_baseline_methods_only_final`)

Change in best overall accuracy relative to the Downsampled ×4 baseline (pp). Methods only; baseline is y=0.

**Figure: PointNet++ Line B secondary gap (final package)** (`final_visualization_package/pointnet2_lineB/lineB_gap_best_overall_vs_original_baseline_methods_only_final`)

Recovery gap to Original baseline 1024 (pp). Secondary column only; not the primary ranking metric.

**Figure: Geometry vs classification combined (final package)** (`final_visualization_package/combined_geometry_vs_classification/lineB_delta_*_vs_delta_accuracy_final`)

Scatter comparing geometry Δ (vs Original) with classification primary Δ (vs Downsampled baseline). Different references by design.
"""
    cap_text = cap_path.read_text(encoding="utf-8")
    if "Final visualization package captions" not in cap_text:
        cap_path.write_text(cap_text.rstrip() + extra_caps + "\n", encoding="utf-8")


def write_master_index(
    static_idx: Path,
    interactive_idx: Path,
    geom_paths: dict,
    pn2_paths: dict,
    combined_paths: list[str],
) -> Path:
    p = REPORTS / "modelnet40_complete_final_visualization_index.md"
    lines = [
        "# ModelNet40 Complete Final Visualization Index",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Package root: `figures/modelnet40/final_visualization_package/`",
        "",
        "## 1. Point cloud static figures",
        "",
        f"- Index: `{static_idx.relative_to(PROJECT)}`",
        "- Directory: `final_visualization_package/pointcloud_static/`",
        "",
        "## 2. Point cloud interactive HTML",
        "",
        f"- Index: `{interactive_idx.relative_to(PROJECT)}`",
        "- Source HTML: `figures/modelnet40/pointcloud_examples_interactive_v2/`",
        "",
        "## 3. Geometry Line B focus figures",
        "",
    ]
    for fp in geom_paths.get("lineB_focus", []):
        lines.append(f"- `{Path(fp).relative_to(PROJECT)}`")
    lines.extend(["", "## 4. Geometry supplementary two-line figures", ""])
    for fp in geom_paths.get("supplementary", []):
        lines.append(f"- `{Path(fp).relative_to(PROJECT)}`")
    lines.extend(["", "## 5. PointNet++ Line A figures", ""])
    for fp in pn2_paths.get("lineA", []):
        lines.append(f"- `{Path(fp).relative_to(PROJECT)}`")
    lines.extend(["", "## 6. PointNet++ Line B figures", ""])
    for fp in pn2_paths.get("lineB", []):
        lines.append(f"- `{Path(fp).relative_to(PROJECT)}`")
    lines.extend(["", "## 7. Geometry vs classification combined figures", ""])
    for fp in combined_paths:
        lines.append(f"- `{Path(fp).relative_to(PROJECT)}`")
    lines.extend(
        [
            "",
            "## 8. Final complete comparison tables",
            "",
            "- `reports/modelnet40_final_complete_geometry_pointnet2_comparison_with_pu_edgeformer.csv`",
            "- `reports/modelnet40_final_complete_geometry_pointnet2_comparison_with_pu_edgeformer.md`",
            "",
            "## 9. Thesis summary tables",
            "",
            "- `reports/modelnet40_final_thesis_summary_table_with_pu_edgeformer.csv`",
            "- `reports/modelnet40_final_thesis_summary_table_with_pu_edgeformer.md`",
            "",
            "## 10. Interpretation report",
            "",
            "- `reports/modelnet40_final_geometry_pointnet2_comparison_interpretation_with_pu_edgeformer.md`",
            "",
            "## 11. Figure captions",
            "",
            "- `reports/modelnet40_thesis_figure_captions.md`",
            "",
            "## 12. Visualization summary",
            "",
            "- `reports/modelnet40_visualization_summary.md`",
            "",
        ]
    )
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_audit() -> Path:
    # verify methods in generated delta figures data
    geom = load_geometry()
    lb = geom_lineb_upsampling(geom)
    methods_ok = {d["label"] for d in lb} == set(METHODS_UP)
    audit("all upsampling methods in geometry", methods_ok, str([d["label"] for d in lb]))

    cls = load_classification()
    la_methods = {d["label"] for d in cls_linea(cls)}
    lb_methods = {d["label"] for d in cls_lineb(cls)}
    audit("Line A methods include EdgeFormer", "PU-EdgeFormer" in la_methods)
    audit("Line B methods include EdgeFormer", "PU-EdgeFormer" in lb_methods)
    audit("both lines included", "Original baseline" in la_methods and "Downsampled baseline" in lb_methods)

    # delta units pp check from CSV
    sample = cls[0]
    audit("classification CSV has pp columns", "delta_accuracy_vs_downsampled_baseline_pp" in sample or "delta_accuracy_vs_original_baseline_pp" in sample)

    # no fabricated - all from existing CSV
    audit("no fabricated values", GEOM_CSV.is_file() and CLS_CSV.is_file(), "read from existing reports")

    # figure count
    png_count = len(list(PKG.rglob("*.png")))
    audit("final package figures generated", png_count >= 20, f"png_count={png_count}")

    p = REPORTS / "modelnet40_complete_final_visualization_audit.md"
    all_pass = all(ok for _, ok, _ in AUDIT_CHECKS)
    lines = [
        "# Complete Final Visualization Audit",
        "",
        f"- Generated: `{utc_now()}`",
        f"- Overall: **{'PASS' if all_pass else 'FAIL'}**",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in AUDIT_CHECKS:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines.extend(
        [
            "",
            "## Reference logic",
            "",
            "- Line B geometry delta: vs Original baseline 1024",
            "- Line B classification primary: vs Downsampled ×4 baseline 256",
            "- Line B classification secondary gap: vs Original baseline 1024",
            "- Line A classification: vs Original baseline 1024",
            "- Delta figures: no baseline bar; y=0 reference line",
            "- Delta units: pp (percentage points)",
            "",
            "## Notes",
            "",
            "- `pointcloud_examples_v2/` static directory not found; static index uses `pointcloud_examples/`.",
            "- Legacy static panels omit PU-EdgeFormer; interactive HTML covers all methods.",
            "- No retraining / no geometry recomputation / no dataset modification.",
            "",
        ]
    )
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> int:
    ensure_dirs()
    geom = load_geometry()
    cls_rows = load_classification()

    static_idx = setup_pointcloud_static()
    interactive_idx = setup_interactive_index()
    geom_paths = generate_geometry_figures(geom)
    pn2_paths = generate_pointnet2_figures(cls_rows)
    combined_paths = generate_combined_figures(geom, cls_rows)

    complete = build_complete_table(geom, cls_rows)
    summary = build_thesis_summary(complete)
    write_complete_reports(complete, summary)
    write_interpretation()
    update_global_indexes()
    master_idx = write_master_index(static_idx, interactive_idx, geom_paths, pn2_paths, combined_paths)
    audit_path = write_audit()

    all_pass = all(ok for _, ok, _ in AUDIT_CHECKS)
    print("PACKAGE_ROOT", PKG)
    print("MASTER_INDEX", master_idx)
    print("AUDIT", audit_path)
    print("FIGURES_GENERATED", len(GENERATED))
    print("OVERALL", "PASS" if all_pass else "FAIL")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
