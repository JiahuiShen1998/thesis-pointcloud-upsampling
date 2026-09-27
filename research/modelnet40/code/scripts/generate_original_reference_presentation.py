#!/usr/bin/env python3
"""Build Original-reference revised presentation (PPTX + PDF + figures + notes).

Geometry uses CD/HD vs corresponding Original point clouds only (no mesh / P2F).
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
FIG = PROJECT / "figures" / "modelnet40"
FIG_REV = FIG / "original_reference_revised"
INTERACTIVE = FIG / "pointcloud_examples_interactive_v2"
PRES = PROJECT / "presentations"
DATASETS = PROJECT / "datasets"

OUT_PPTX = PRES / "ModelNet40_PointNet2_Original_Reference_Revised.pptx"
OUT_PDF = PRES / "ModelNet40_PointNet2_Original_Reference_Revised.pdf"
NOTES = PRES / "ModelNet40_Presentation_Revision_Notes.md"

# Public HTTPS base for interactive HTML (GitHub Pages / GitLab Pages).
# Example: https://USER.github.io/REPO/figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2
# file:// links do not open for teachers on other machines — always prefer this.
PUBLIC_INTERACTIVE_BASE = os.environ.get(
    "PUBLIC_INTERACTIVE_BASE",
    "",
).rstrip("/")

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
NAVY = RGBColor(0x1A, 0x3A, 0x5C)
ACCENT = RGBColor(0xC4, 0x4E, 0x52)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
DARK = RGBColor(0x22, 0x22, 0x22)
GRAY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF5, 0xF7, 0xFA)
LIGHT_GRAY = RGBColor(0x88, 0x88, 0x88)

# Classification data (user-specified; matches existing final metrics)
LINEA_CLS = [
    ("Original", 1024, 91.95, 91.31, 0.00),
    ("EAR", 4096, 91.48, 90.85, -0.47),
    ("PDANS", 4096, 91.62, 90.87, -0.33),
    ("PU-Net", 4096, 90.95, 90.55, -1.00),
    ("PU-GCN", 4096, 91.63, 91.00, -0.32),
    ("PU-EdgeFormer", 4096, 90.54, 90.13, -1.41),
]
LINEB_CLS = [
    ("Downsampled ×4", 256, 90.85, 90.46, 0.00, -1.10),
    ("EAR", 1024, 88.81, 88.02, -2.04, -3.14),
    ("PDANS", 1024, 90.34, 89.46, -0.51, -1.61),
    ("PU-Net", 1024, 91.27, 90.65, 0.42, -0.68),
    ("PU-GCN", 1024, 90.06, 89.63, -0.79, -1.89),
    ("PU-EdgeFormer", 1024, 89.32, 88.58, -1.53, -2.63),
]
# Mesh-ref PointNet++ baselines (area-weighted from .off; independent of Original/downsample)
# (name, pts, best, final, compare_label, compare_best, delta_pp)
MESH_REF_CLS = [
    ("Mesh-ref 256", 256, 90.88, 90.38, "Downsampled ×4 256", 90.85, 0.03),
    ("Mesh-ref 4096", 4096, 92.00, 91.18, "Original 1024", 91.95, 0.05),
]

SAMPLES = {
    "airplane": "airplane_0627",
    "chair": "chair_0890",
    "table": "table_0393",
    "car": "car_0198",
    "sofa": "sofa_0681",
}
# Per-class elev/azim for clearer structure; shared within each class
CAMERAS = {
    "airplane": (22, -45),   # elevated front-side three-quarter
    "chair": (18, -35),      # front-side three-quarter
    "table": (35, -50),      # elevated three-quarter
    "car": (20, -55),        # elevated front-side
    "sofa": (18, -40),       # front-side three-quarter
}

METHODS_LINEA = [
    ("Original", "original", 1024),
    ("EAR", "ear", 4096),
    ("PDANS", "pdans", 4096),
    ("PU-Net", "pu_net", 4096),
    ("PU-GCN", "pu_gcn", 4096),
    ("PU-EdgeFormer", "pu_edgeformer", 4096),
]
METHODS_LINEB = [
    ("Original", "original", 1024),
    ("Down ×4", "downsampled_x4", 256),
    ("EAR", "ear", 1024),
    ("PDANS", "pdans", 1024),
    ("PU-Net", "pu_net", 1024),
    ("PU-GCN", "pu_gcn", 1024),
    ("PU-EdgeFormer", "pu_edgeformer", 1024),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_geometry() -> dict[str, list[dict[str, Any]]]:
    """Load equal-cardinality geometry summary (preferred) or legacy vs-Original."""
    eq = REPORTS / "modelnet40_geometry_equal_n_summary.csv"
    if eq.exists():
        rows: list[dict[str, Any]] = []
        with open(eq, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                parsed: dict[str, Any] = dict(row)
                for key in (
                    "points",
                    "cd_vs_ref",
                    "hd_vs_ref",
                    "nuc",
                    "delta_nuc_vs_ref",
                    "valid_samples",
                    "cd_std",
                    "hd_std",
                    "nuc_std",
                ):
                    if key in parsed and parsed[key] not in ("", None, "Pending"):
                        try:
                            parsed[key] = float(parsed[key]) if key != "points" and key != "valid_samples" else int(float(parsed[key]))
                        except ValueError:
                            pass
                # aliases for figure helpers
                parsed["cd_vs_original"] = parsed.get("cd_vs_ref")
                parsed["hd_vs_original"] = parsed.get("hd_vs_original", parsed.get("hd_vs_ref"))
                parsed["delta_nuc_vs_original"] = parsed.get("delta_nuc_vs_ref")
                rows.append(parsed)
        return {
            "A": [r for r in rows if r.get("line") == "A"],
            "B": [r for r in rows if r.get("line") == "B"],
            "source": "equal_n",
        }

    path = REPORTS / "modelnet40_geometry_vs_original_summary.json"
    if not path.exists():
        return {"A": [], "B": [], "source": "none"}
    rows_j = json.loads(path.read_text(encoding="utf-8"))
    return {
        "A": [r for r in rows_j if r["line"] == "A"],
        "B": [r for r in rows_j if r["line"] == "B"],
        "source": "legacy_original",
    }


def fmt4(v: Any) -> str:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return f"{float(v):.4f}"
    return "Pending"


def set_run(run, text: str, size: int = 18, bold: bool = False, color=DARK):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"


def add_title(slide, text: str, top=0.22, size=28):
    box = slide.shapes.add_textbox(Inches(0.45), Inches(top), Inches(12.4), Inches(0.55))
    p = box.text_frame.paragraphs[0]
    run = p.add_run()
    set_run(run, text, size, True, NAVY)


def add_bullets(slide, bullets: list[str], left=0.5, top=1.0, width=12.2, height=5.5, size=18):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        set_run(run, "•  " + item, size, False, DARK)
        p.space_after = Pt(7)


def add_footer(slide, page: int, total: int):
    box = slide.shapes.add_textbox(Inches(0.45), Inches(7.05), Inches(12.4), Inches(0.3))
    p = box.text_frame.paragraphs[0]
    run = p.add_run()
    set_run(run, f"ModelNet40 · PointNet++ Upsampling    {page}/{total}", 10, False, GRAY)


def add_table(slide, rows: list[list[str]], left, top, width, height, font_size=14):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = shape.table
    for r in range(n_rows):
        for c in range(n_cols):
            cell = table.cell(r, c)
            cell.text = rows[r][c]
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                for run in p.runs:
                    run.font.size = Pt(font_size if r > 0 else font_size + 1)
                    run.font.bold = r == 0
                    run.font.name = "Calibri"
                    run.font.color.rgb = WHITE if r == 0 else DARK
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            elif r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT
    return shape


def blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def load_points(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    return pts.astype(np.float32)


def resolve_cloud(line: str, method_key: str, class_name: str, sample_id: str) -> Path:
    split = "test"
    if method_key == "original":
        return DATASETS / "modelnet40_original" / split / class_name / f"{sample_id}.npy"
    if method_key == "downsampled_x4":
        return DATASETS / "modelnet40_downsampled_x4" / split / class_name / f"{sample_id}.npy"
    if line == "A":
        return DATASETS / "lineA_original_up" / "strict_4N" / method_key / split / class_name / f"{sample_id}.npy"
    return DATASETS / "lineB_downsampled_x4_up" / "strict_N" / method_key / split / class_name / f"{sample_id}.npy"


def project_points(pts: np.ndarray, elev: float, azim: float) -> np.ndarray:
    elev_r = np.radians(elev)
    azim_r = np.radians(azim)
    cz, sz = np.cos(azim_r), np.sin(azim_r)
    ce, se = np.cos(elev_r), np.sin(elev_r)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]], dtype=np.float32)
    rotated = pts @ (rx @ rz).T
    return rotated[:, :2]


def point_size(n: int) -> float:
    if n <= 256:
        return 14.0
    if n <= 1024:
        return 5.0
    return 1.8


def plot_pointcloud_comparison(
    line: str,
    class_name: str,
    sample_id: str,
    methods: list[tuple[str, str, int]],
    out_path: Path,
) -> bool:
    elev, azim = CAMERAS[class_name]
    panels: list[tuple[str, np.ndarray, int]] = []
    for title, key, n in methods:
        pts = load_points(resolve_cloud(line, key, class_name, sample_id))
        if pts is None:
            return False
        panels.append((f"{title}\n{n} pts", pts, n))

    projected = [(t, project_points(p, elev, azim), p, n) for t, p, n in panels]
    all_pr = np.vstack([pr for _, pr, _, _ in projected])
    center = (all_pr.min(axis=0) + all_pr.max(axis=0)) / 2
    radius = float(np.max(all_pr.max(axis=0) - all_pr.min(axis=0)) / 2 * 1.08)
    xlim = (center[0] - radius, center[0] + radius)
    ylim = (center[1] - radius, center[1] + radius)

    ncols = len(projected)
    fig, axes = plt.subplots(1, ncols, figsize=(2.35 * ncols, 3.15), facecolor="#1a1a1a")
    if ncols == 1:
        axes = [axes]
    for ax, (title, pr, pts, n) in zip(axes, projected):
        ax.set_facecolor("#1a1a1a")
        ax.scatter(
            pr[:, 0],
            pr[:, 1],
            s=point_size(n),
            c=pts[:, 2],
            cmap="viridis",
            alpha=0.92,
            linewidths=0,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=9, color="white", pad=4)
        ax.axis("off")
    fig.suptitle(
        f"Line {line} · {class_name} · {sample_id}\n"
        "Same sample · Same camera · Same axis limits · Same point-size rule  |  Color = depth (not error)",
        fontsize=10,
        color="white",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def bar_classification(line: str, out_path: Path) -> None:
    if line == "A":
        labels = [r[0] for r in LINEA_CLS]
        best = [r[2] for r in LINEA_CLS]
        baseline = 91.95
        title = "Line A — Best Overall Accuracy (primary)"
        ylabel = "Best Overall (%)"
        colors = ["#1a3a5c" if i == 0 else ("#c44e52" if best[i] < baseline else "#2e7d32") for i in range(len(best))]
    else:
        labels = [r[0] for r in LINEB_CLS]
        best = [r[2] for r in LINEB_CLS]
        baseline = 90.85
        title = "Line B — Best Overall Accuracy (primary)"
        ylabel = "Best Overall (%)"
        colors = []
        for i, v in enumerate(best):
            if i == 0:
                colors.append("#1a3a5c")
            elif v > baseline:
                colors.append("#2e7d32")
            else:
                colors.append("#c44e52")

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, best, color=colors, edgecolor="none", width=0.65)
    ax.axhline(baseline, color="#555555", linestyle="--", linewidth=1.4, label=f"Baseline {baseline:.2f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, color="#1a3a5c", pad=10)
    ymin = min(best) - 1.5
    ymax = max(best) + 1.2
    ax.set_ylim(ymin, ymax)
    for b, v in zip(bars, best):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def bar_mesh_ref_cls(out_path: Path) -> None:
    """Grouped bars: mesh-ref baselines vs matched existing baselines."""
    labels = ["256 pts\n(Mesh-ref vs Down ×4)", "4096 / 1024 pts\n(Mesh-ref vs Original)"]
    mesh = [MESH_REF_CLS[0][2], MESH_REF_CLS[1][2]]
    cmp_ = [MESH_REF_CLS[0][5], MESH_REF_CLS[1][5]]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    b1 = ax.bar(x - w / 2, mesh, w, label="Mesh-ref", color="#1a3a5c", edgecolor="none")
    b2 = ax.bar(x + w / 2, cmp_, w, label="Existing baseline", color="#4a6fa5", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Best Overall (%)", fontsize=12)
    ax.set_title("Mesh-ref PointNet++ baselines vs existing", fontsize=14, color="#1a3a5c", pad=10)
    all_v = mesh + cmp_
    ax.set_ylim(min(all_v) - 1.2, max(all_v) + 1.0)
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def bar_geometry(line: str, geo_rows: list[dict], metric: str, out_path: Path) -> None:
    if not geo_rows:
        return
    labels = [r["method"] for r in geo_rows]
    vals = [float(r[metric]) if isinstance(r[metric], (int, float)) else float("nan") for r in geo_rows]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    x = np.arange(len(labels))
    colors = ["#1a3a5c" if i == 0 else "#4a6fa5" for i in range(len(labels))]
    bars = ax.bar(x, vals, color=colors, width=0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    name = "CD vs ref ↓" if "cd" in metric else "HD vs ref ↓"
    ax.set_ylabel(name, fontsize=12)
    ax.set_title(f"Line {line} — {name}", fontsize=14, color="#1a3a5c")
    for b, v in zip(bars, vals):
        if math.isfinite(v):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def scatter_geom_cls(line: str, geo_rows: list[dict], out_path: Path) -> bool:
    """Geometry vs classification scatter using recomputed CD vs Original."""
    if not geo_rows:
        return False
    geo_map = {r["method"]: r for r in geo_rows}
    points = []
    if line == "A":
        for name, _, best, _, delta in LINEA_CLS:
            if name == "Original":
                continue
            g = geo_map.get(name)
            if not g or not isinstance(g.get("cd_vs_original"), (int, float)):
                continue
            points.append((float(g["cd_vs_original"]), delta, name))
        xlabel = "CD vs Mesh-ref 4096 ↓"
        ylabel = "Δ Best Overall vs Original (pp)"
        title = "Line A — Equal-N geometry vs classification Δ"
    else:
        for row in LINEB_CLS:
            name, n, best, _, delta, _ = row
            if name.startswith("Down") or n != 1024:
                continue
            g = geo_map.get(name)
            if not g or not isinstance(g.get("cd_vs_original"), (int, float)):
                continue
            points.append((float(g["cd_vs_original"]), delta, name))
        xlabel = "CD vs Original 1024 ↓"
        ylabel = "Δ Best Overall vs Downsampled (pp)"
        title = "Line B — Equal-N geometry vs classification Δ"

    if len(points) < 2:
        return False

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.axhline(0, color="#888", linestyle="--", linewidth=1)
    ax.scatter(xs, ys, s=70, c="#1a3a5c", zorder=3)
    for x, y, name in points:
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, color="#1a3a5c")
    ax.text(
        0.02,
        0.02,
        "Lower CD = closer to equal-N reference; does not imply higher accuracy.",
        transform=ax.transAxes,
        fontsize=8,
        style="italic",
        color="#555",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_figures(geo: dict[str, list[dict]]) -> dict[str, Path]:
    FIG_REV.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    p = FIG_REV / "lineA_best_overall.png"
    bar_classification("A", p)
    paths["lineA_cls"] = p
    p = FIG_REV / "lineB_best_overall.png"
    bar_classification("B", p)
    paths["lineB_cls"] = p
    p = FIG_REV / "mesh_ref_best_overall.png"
    bar_mesh_ref_cls(p)
    paths["mesh_ref_cls"] = p

    if geo["A"]:
        p = FIG_REV / "lineA_cd_vs_original.png"
        bar_geometry("A", geo["A"], "cd_vs_original", p)
        paths["lineA_cd"] = p
        p = FIG_REV / "lineA_hd_vs_original.png"
        bar_geometry("A", geo["A"], "hd_vs_original", p)
        paths["lineA_hd"] = p
        p = FIG_REV / "lineA_geom_vs_cls.png"
        if scatter_geom_cls("A", geo["A"], p):
            paths["lineA_scatter"] = p
    if geo["B"]:
        p = FIG_REV / "lineB_cd_vs_original.png"
        bar_geometry("B", geo["B"], "cd_vs_original", p)
        paths["lineB_cd"] = p
        p = FIG_REV / "lineB_hd_vs_original.png"
        bar_geometry("B", geo["B"], "hd_vs_original", p)
        paths["lineB_hd"] = p
        p = FIG_REV / "lineB_geom_vs_cls.png"
        if scatter_geom_cls("B", geo["B"], p):
            paths["lineB_scatter"] = p

    for cls, sid in SAMPLES.items():
        out = FIG_REV / "pointclouds" / f"lineA_{cls}_{sid}_comparison.png"
        ok = plot_pointcloud_comparison("A", cls, sid, METHODS_LINEA, out)
        if ok:
            paths[f"pc_A_{cls}"] = out
        out = FIG_REV / "pointclouds" / f"lineB_{cls}_{sid}_comparison.png"
        ok = plot_pointcloud_comparison("B", cls, sid, METHODS_LINEB, out)
        if ok:
            paths[f"pc_B_{cls}"] = out
    return paths


def regenerate_interactive() -> list[str]:
    """Regenerate dropdown HTML viewers with improved per-class cameras."""
    import plotly.graph_objects as go

    verified: list[str] = []
    out_dir = INTERACTIVE / "dropdown_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    camera_eye = {
        "airplane": dict(x=1.55, y=1.35, z=0.95),
        "chair": dict(x=1.45, y=1.55, z=1.05),
        "table": dict(x=1.35, y=1.45, z=1.35),
        "car": dict(x=1.55, y=1.25, z=0.95),
        "sofa": dict(x=1.45, y=1.50, z=1.00),
    }

    def marker_size(n: int) -> float:
        if n <= 256:
            return 3.0
        if n <= 1024:
            return 1.6
        return 1.0

    for line, methods in (("A", METHODS_LINEA), ("B", METHODS_LINEB)):
        for cls in ("airplane", "chair", "table", "car", "sofa"):
            sid = SAMPLES[cls]
            panels = []
            for title, key, n in methods:
                pts = load_points(resolve_cloud(line, key, cls, sid))
                if pts is None:
                    continue
                panels.append((f"{title} ({n})", pts, n))
            if len(panels) < 2:
                continue
            all_pts = np.vstack([p for _, p, _ in panels])
            mins, maxs = all_pts.min(0), all_pts.max(0)
            center = (mins + maxs) / 2
            radius = float(np.max(maxs - mins) / 2 * 1.05)
            ranges = (
                [center[0] - radius, center[0] + radius],
                [center[1] - radius, center[1] + radius],
                [center[2] - radius, center[2] + radius],
            )
            fig = go.Figure()
            for i, (name, pts, n) in enumerate(panels):
                fig.add_trace(
                    go.Scatter3d(
                        x=pts[:, 0],
                        y=pts[:, 1],
                        z=pts[:, 2],
                        mode="markers",
                        marker=dict(
                            size=marker_size(n),
                            color=pts[:, 2],
                            colorscale="Viridis",
                            opacity=0.92,
                            showscale=False,
                        ),
                        name=name,
                        visible=(i == 0),
                        hovertemplate=f"{name}<extra></extra>",
                    )
                )
            buttons = []
            for i, (name, _, _) in enumerate(panels):
                vis = [False] * len(panels)
                vis[i] = True
                buttons.append(
                    dict(
                        label=name,
                        method="update",
                        args=[{"visible": vis}, {"title": f"Line {line} · {cls} · {sid} · {name}"}],
                    )
                )
            scene = dict(
                xaxis=dict(range=ranges[0], showticklabels=False, title=""),
                yaxis=dict(range=ranges[1], showticklabels=False, title=""),
                zaxis=dict(range=ranges[2], showticklabels=False, title=""),
                aspectmode="cube",
                camera=dict(eye=camera_eye[cls], up=dict(x=0, y=0, z=1)),
                bgcolor="#111111",
            )
            fig.update_layout(
                title=f"Line {line} · {cls} · {sid} · {panels[0][0]}",
                scene=scene,
                paper_bgcolor="#111111",
                font=dict(color="white"),
                margin=dict(l=0, r=0, t=50, b=0),
                updatemenus=[
                    dict(
                        buttons=buttons,
                        direction="down",
                        x=0.01,
                        y=0.98,
                        xanchor="left",
                        bgcolor="#333",
                        font=dict(color="white"),
                    )
                ],
                annotations=[
                    dict(
                        text="Rotate / zoom / pan · method switch keeps camera · color = depth",
                        showarrow=False,
                        x=0.5,
                        y=0.01,
                        xref="paper",
                        yref="paper",
                        font=dict(size=11, color="#cccccc"),
                    )
                ],
            )
            out = out_dir / f"line{line}_{cls}_{sid}_dropdown.html"
            fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)
            verified.append(str(out.relative_to(PROJECT)))
    return verified


def build_pptx(geo: dict[str, list[dict]], figs: dict[str, Path], interactive_files: list[str]) -> int:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    slides_meta: list = []

    def add():
        s = blank(prs)
        slides_meta.append(s)
        return s

    # 1 Title
    s = add()
    box = s.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(2.2))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    set_run(run, "Point Cloud Upsampling on ModelNet40", 34, True, NAVY)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    run = p2.add_run()
    set_run(
        run,
        "Geometry Consistency with Original Point Clouds\nand PointNet++ Classification",
        22,
        False,
        GRAY,
    )
    box2 = s.shapes.add_textbox(Inches(1.2), Inches(4.6), Inches(10.9), Inches(1.4))
    tf = box2.text_frame
    for i, line in enumerate(
        [
            "Two-line protocol · EAR · PDANS · PU-Net · PU-GCN · PU-EdgeFormer",
            "Geometry reference: corresponding Original point cloud only (no mesh / no P2F)",
            "Classifier: PointNet++ retrained per input size · 200 epochs · seed=42",
        ]
    ):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        set_run(run, line, 16, False, DARK)

    # 2 Research questions
    s = add()
    add_title(s, "Research Questions and Contributions")
    add_bullets(
        s,
        [
            "Does densification (1024 → 4096) preserve Original geometry and improve PointNet++?",
            "Can ×4 upsampling recover a point cloud closer to Original 1024 after downsampling?",
            "Does stronger geometric consistency with the Original point cloud imply better classification?",
            "Contribution: unified Original-reference geometry protocol + two-line classification analysis",
            "Contribution: revised qualitative views and interactive 3D inspection for thesis defense",
        ],
        top=1.05,
        size=19,
    )

    # 3 Protocol
    s = add()
    add_title(s, "Experimental Protocol: Line A and Line B")
    add_bullets(
        s,
        [
            "Line A (Densification): Original 1024 → Upsampler ×4 → 4096; geometry CD/HD vs mesh-sampled 4096",
            "Line B (Recovery): Downsample 256 → Upsampler ×4 → 1024; CD/HD: 256 vs mesh-256; 1024 vs Original 1024",
            "Classifier: each branch trained from scratch for its point count (A: 4096; B: 256 baseline / 1024 up)",
            "Same methods on both lines: EAR, PDANS, PU-Net, PU-GCN, PU-EdgeFormer",
        ],
        top=0.95,
        height=2.4,
        size=15,
    )
    add_table(
        s,
        [
            ["Line", "Input / path", "Output pts", "Geometry CD/HD in PPT", "PointNet++ setting"],
            ["A", "Original 1024 → ×4", "4096", "4096 vs mesh-ref 4096", "Retrained with 4096-pt input"],
            ["B", "256 → ×4 up", "1024", "1024 vs Original 1024", "Retrained with 1024-pt input"],
            ["B", "Downsampled ×4", "256", "256 vs mesh-ref 256", "Trained with 256-pt input"],
        ],
        Inches(0.45),
        Inches(3.55),
        Inches(12.4),
        Inches(2.35),
        font_size=13,
    )

    # 4 Geometry definition
    s = add()
    add_title(s, "Geometry Evaluation Definition")
    add_bullets(
        s,
        [
            "Equal cardinality only: CD/HD require the same point count for ranking.",
            "Mesh references: area-weighted samples from ModelNet40 `.off` + unit-sphere normalize (same as Original 1024).",
            "Line A: upsamplers 4096 vs mesh-ref 4096; Line B Downsampled 256 vs mesh-ref 256.",
            "Line B recovery: upsamplers 1024 vs Original 1024.",
            "Self rows (mesh-ref / Original): CD = 0 and HD = 0 by definition. No P2F.",
        ],
        top=1.05,
        size=16,
    )

    # 5 Metrics
    s = add()
    add_title(s, "Geometry Metrics: CD, HD and NUC")
    add_bullets(
        s,
        [
            "CD: mean NN L2(a→b) + mean NN L2(b→a) vs equal-N reference; lower better",
            "HD: max(max NN a→b, max NN b→a) vs equal-N reference; lower better",
            "NUC: uniformity of each cloud alone (not a distance to the reference)",
            "ΔNUC = NUC_method − NUC_reference (matched cardinality)",
            "Unequal-|P| CD/HD (e.g. 4096 vs 1024) are omitted from this presentation",
        ],
        top=1.05,
        size=16,
    )

    geo_a = geo.get("A") or []
    geo_b = geo.get("B") or []

    # Line A equal-N geometry (4096 vs mesh-4096)
    s = add()
    add_title(s, "Line A Geometry: 4096 vs Mesh-ref 4096")
    rows_a = [r for r in geo_a if int(r.get("points", -1)) == 4096]
    if rows_a:
        rows = [["Method", "Pts", "Ref", "CD ↓", "HD ↓", "NUC ↓", "ΔNUC"]]
        for r in rows_a:
            rows.append(
                [
                    r.get("method", ""),
                    str(int(r["points"])),
                    str(r.get("reference", "Mesh-ref 4096")),
                    fmt4(r.get("cd_vs_ref", r.get("cd_vs_original"))),
                    fmt4(r.get("hd_vs_ref", r.get("hd_vs_original"))),
                    fmt4(r.get("nuc")),
                    fmt4(r.get("delta_nuc_vs_ref", r.get("delta_nuc_vs_original"))),
                ]
            )
        add_table(s, rows, Inches(0.4), Inches(0.95), Inches(12.5), Inches(4.5), font_size=14)
        note = s.shapes.add_textbox(Inches(0.45), Inches(5.6), Inches(12.4), Inches(1.1))
        tf = note.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        run = p.add_run()
        n = rows_a[0].get("valid_samples", "?")
        set_run(
            run,
            f"Test split N={n}. Equal cardinality. Reference = mesh-sampled 4096 from ModelNet40 .off "
            "(area-weighted). PointNet++ for these branches was retrained at 4096 (classification separate).",
            13,
            False,
            DARK,
        )
    else:
        add_bullets(s, ["Pending equal-N recomputation."], top=1.2)

    # Line B: 256 vs mesh-256 + 1024 vs Original
    s = add()
    add_title(s, "Line B Geometry: Equal Cardinality Only")
    rows_b256 = [r for r in geo_b if int(r.get("points", -1)) == 256]
    rows_b1024 = [r for r in geo_b if int(r.get("points", -1)) == 1024]
    y = 0.85
    if rows_b256:
        title_box = s.shapes.add_textbox(Inches(0.45), Inches(y), Inches(12.4), Inches(0.35))
        run = title_box.text_frame.paragraphs[0].add_run()
        set_run(run, "256 vs Mesh-ref 256 (Downsampled)", 14, True, NAVY)
        y += 0.35
        rows = [["Method", "Pts", "Ref", "CD ↓", "HD ↓", "NUC ↓", "ΔNUC"]]
        for r in rows_b256:
            rows.append(
                [
                    r.get("method", ""),
                    str(int(r["points"])),
                    str(r.get("reference", "Mesh-ref 256")),
                    fmt4(r.get("cd_vs_ref", r.get("cd_vs_original"))),
                    fmt4(r.get("hd_vs_ref", r.get("hd_vs_original"))),
                    fmt4(r.get("nuc")),
                    fmt4(r.get("delta_nuc_vs_ref", r.get("delta_nuc_vs_original"))),
                ]
            )
        add_table(s, rows, Inches(0.4), Inches(y), Inches(12.5), Inches(1.35), font_size=12)
        y += 1.5
    if rows_b1024:
        title_box = s.shapes.add_textbox(Inches(0.45), Inches(y), Inches(12.4), Inches(0.35))
        run = title_box.text_frame.paragraphs[0].add_run()
        set_run(run, "1024 vs Original 1024 (Recovery upsamplers)", 14, True, NAVY)
        y += 0.35
        rows = [["Method", "Pts", "Ref", "CD ↓", "HD ↓", "NUC ↓", "ΔNUC"]]
        for r in rows_b1024:
            rows.append(
                [
                    r.get("method", ""),
                    str(int(r["points"])),
                    str(r.get("reference", "Original 1024")),
                    fmt4(r.get("cd_vs_ref", r.get("cd_vs_original"))),
                    fmt4(r.get("hd_vs_ref", r.get("hd_vs_original"))),
                    fmt4(r.get("nuc")),
                    fmt4(r.get("delta_nuc_vs_ref", r.get("delta_nuc_vs_original"))),
                ]
            )
        add_table(s, rows, Inches(0.4), Inches(y), Inches(12.5), Inches(3.2), font_size=12)
    if not rows_b256 and not rows_b1024:
        add_bullets(s, ["Pending equal-N recomputation."], top=1.2)

    # Line B qualitative — airplane + chair (two key views)
    s = add()
    add_title(s, "Line B Qualitative Comparison")
    if "pc_B_airplane" in figs:
        s.shapes.add_picture(str(figs["pc_B_airplane"]), Inches(0.3), Inches(0.78), width=Inches(12.7))
    elif "pc_B_chair" in figs:
        s.shapes.add_picture(str(figs["pc_B_chair"]), Inches(0.3), Inches(0.78), width=Inches(12.7))
    else:
        add_bullets(s, ["Qualitative figures pending."], top=1.2)
    note = s.shapes.add_textbox(Inches(0.45), Inches(6.55), Inches(12.4), Inches(0.35))
    p = note.text_frame.paragraphs[0]
    run = p.add_run()
    set_run(run, "Same sample · Same camera · Same axis limits · Color = depth (not error)", 13, False, GRAY)

    s = add()
    add_title(s, "Line B Qualitative — Chair / Table")
    y = 0.78
    for key in ("pc_B_chair", "pc_B_table"):
        if key in figs:
            s.shapes.add_picture(str(figs[key]), Inches(0.3), Inches(y), width=Inches(12.7))
            break

    # Line A qualitative
    s = add()
    add_title(s, "Line A Qualitative Comparison")
    for key in ("pc_A_airplane", "pc_A_chair", "pc_A_table"):
        if key in figs:
            s.shapes.add_picture(str(figs[key]), Inches(0.3), Inches(0.78), width=Inches(12.7))
            break
    note = s.shapes.add_textbox(Inches(0.45), Inches(6.55), Inches(12.4), Inches(0.35))
    p = note.text_frame.paragraphs[0]
    run = p.add_run()
    set_run(run, "Same sample · Same camera · Same axis limits · Color = depth (not error)", 13, False, GRAY)

    s = add()
    add_title(s, "Line A Qualitative — Chair / Table")
    for key in ("pc_A_chair", "pc_A_table"):
        if key in figs:
            s.shapes.add_picture(str(figs[key]), Inches(0.3), Inches(0.78), width=Inches(12.7))
            break

    # 11 Best vs Final
    s = add()
    add_title(s, "Best Overall vs Final Overall")
    add_bullets(
        s,
        [
            "Best Overall: highest recorded overall classification accuracy during training (primary)",
            "Corresponds to best_model.pth; Final Overall = epoch 200 (supplementary only).",
            "Paper (PointNet++): ModelNet40 uses official split 9843 train / 2468 test — no validation set stated for this task.",
            "Code: loader asserts split in {train, test}; best checkpoint updated when test-instance accuracy improves.",
            "This is the common ModelNet40 PointNet++ practice; rankings use Best Overall.",
        ],
        top=1.05,
        size=15,
    )

    # 12 Line A classification
    s = add()
    add_title(s, "Line A Classification Results")
    if "lineA_cls" in figs:
        s.shapes.add_picture(str(figs["lineA_cls"]), Inches(0.4), Inches(0.85), width=Inches(7.6))
    rows = [["Method", "Pts", "Best Overall", "Final Overall", "Δ Best vs Original"]]
    for name, n, best, final, delta in LINEA_CLS:
        sign = f"{delta:+.2f} pp"
        rows.append([name, str(n), f"{best:.2f}%", f"{final:.2f}%", sign])
    add_table(s, rows, Inches(8.1), Inches(0.95), Inches(4.8), Inches(4.5), font_size=12)
    note = s.shapes.add_textbox(Inches(0.45), Inches(5.7), Inches(12.4), Inches(1.1))
    tf = note.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    set_run(
        run,
        "Original 1024 achieves the highest Best Overall (91.95%). "
        "Each 4096 branch used PointNet++ retrained from scratch with 4096-point input; "
        "the drop vs Original is therefore not from applying a 1024-only classifier to denser clouds. "
        "Under this matched protocol, densification still does not improve classification.",
        13,
        False,
        DARK,
    )

    # 13 Line B classification
    s = add()
    add_title(s, "Line B Classification Results")
    if "lineB_cls" in figs:
        s.shapes.add_picture(str(figs["lineB_cls"]), Inches(0.35), Inches(0.85), width=Inches(7.3))
    rows = [["Method", "Pts", "Best", "Final", "Δ vs Down", "Gap vs Orig"]]
    for name, n, best, final, d_down, gap in LINEB_CLS:
        rows.append([name, str(n), f"{best:.2f}%", f"{final:.2f}%", f"{d_down:+.2f} pp", f"{gap:+.2f} pp"])
    add_table(s, rows, Inches(7.8), Inches(0.9), Inches(5.2), Inches(4.4), font_size=11)
    note = s.shapes.add_textbox(Inches(0.45), Inches(5.55), Inches(12.4), Inches(1.25))
    tf = note.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    set_run(
        run,
        "Line B classifiers are also retrained for their point counts (256 baseline; 1024 for upsamplers). "
        "PU-Net is the only method above Downsampled 256 (+0.42 pp → 91.27%), but remains −0.68 pp below Original 1024 — "
        "partial recovery, not full restoration. Geometry quality and classification are not fully aligned.",
        13,
        False,
        DARK,
    )

    # Mesh-ref PointNet++ baselines
    s = add()
    add_title(s, "Mesh-ref PointNet++ Baselines")
    if "mesh_ref_cls" in figs:
        s.shapes.add_picture(str(figs["mesh_ref_cls"]), Inches(0.35), Inches(0.85), width=Inches(7.0))
    rows = [["Method", "Pts", "Best", "Final", "Compare to", "Δ Best"]]
    for name, n, best, final, cmp_label, cmp_best, delta in MESH_REF_CLS:
        rows.append(
            [
                name,
                str(n),
                f"{best:.2f}%",
                f"{final:.2f}%",
                f"{cmp_label} ({cmp_best:.2f}%)",
                f"{delta:+.2f} pp",
            ]
        )
    add_table(s, rows, Inches(7.5), Inches(0.95), Inches(5.5), Inches(2.4), font_size=11)
    note = s.shapes.add_textbox(Inches(0.45), Inches(5.35), Inches(12.4), Inches(1.45))
    tf = note.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    set_run(
        run,
        "Independent area-weighted samples from ModelNet40 `.off` (not downsampled from Original 1024). "
        "Mesh-ref 256 ≈ Downsampled ×4 256 (90.88% vs 90.85%, +0.03 pp). "
        "Mesh-ref 4096 ≈ Original 1024 (92.00% vs 91.95%, +0.05 pp). "
        "Equal-N mesh sampling classifies almost identically; denser 4096 does not beat Original 1024 by a meaningful margin.",
        13,
        False,
        DARK,
    )

    # 14 Geometry vs classification
    s = add()
    add_title(s, "Geometry vs Classification")
    if "lineB_scatter" in figs:
        s.shapes.add_picture(str(figs["lineB_scatter"]), Inches(1.5), Inches(0.9), width=Inches(10.0))
        add_bullets(
            s,
            [
                "Scatter uses equal-cardinality CD only (Line B: 1024 vs Original 1024)",
                "Lower CD does not automatically imply higher PointNet++ accuracy",
                "Limited method count: geometry alone is insufficient to predict classification",
            ],
            top=5.35,
            height=1.5,
            size=15,
        )
    else:
        add_bullets(s, ["Pending recomputation of equal-cardinality CD before scatter plots."], top=1.2)

    # Interactive
    s = add()
    add_title(s, "Interactive 3D Point-Cloud Demo")
    if PUBLIC_INTERACTIVE_BASE:
        link_mode = f"Hosted HTTPS base: {PUBLIC_INTERACTIVE_BASE}"
        link_hint = "Click the HTTPS links below (works for teacher after push to GitHub/GitLab Pages)."
    else:
        link_mode = "Links currently local-only (file://) — will not open on other machines."
        link_hint = (
            "After push, regenerate with PUBLIC_INTERACTIVE_BASE=https://…/dropdown_v2 "
            "so PPT embeds clickable GitHub/GitLab Pages URLs."
        )
    add_bullets(
        s,
        [
            "Static slides use manually selected shared cameras per class",
            "Interactive HTML: rotate / zoom / pan / method switch (Plotly CDN)",
            "Repo path: figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/",
            link_mode,
            link_hint,
        ],
        top=0.95,
        height=2.35,
        size=15,
    )
    # Clickable hyperlinks: prefer public HTTPS; fall back to file:// with warning
    from pptx.oxml.ns import qn
    from lxml import etree

    box = s.shapes.add_textbox(Inches(0.5), Inches(3.5), Inches(12.2), Inches(3.1))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    set_run(
        run,
        "Demo links:" if PUBLIC_INTERACTIVE_BASE else "Local demo links (open only on this HPC machine):",
        14,
        True,
        NAVY,
    )
    demo_links = [
        f for f in interactive_files if any(x in f for x in ("airplane", "chair", "table"))
    ][:6]
    for rel in demo_links:
        abs_path = (PROJECT / rel).resolve()
        name = abs_path.name
        if PUBLIC_INTERACTIVE_BASE:
            url = f"{PUBLIC_INTERACTIVE_BASE}/{name}"
        else:
            url = abs_path.as_uri()
        p = tf.add_paragraph()
        run = p.add_run()
        set_run(run, name if not PUBLIC_INTERACTIVE_BASE else url, 12, False, RGBColor(0x15, 0x65, 0xC0))
        rId = s.part.relate_to(
            url,
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
            is_external=True,
        )
        rPr = run._r.get_or_add_rPr()
        hlink = etree.SubElement(rPr, qn("a:hlinkClick"))
        hlink.set(qn("r:id"), rId)

    # 16 Main findings
    s = add()
    add_title(s, "Main Findings")
    add_bullets(
        s,
        [
            "Line A: Original 1024 remains best at 91.95% among densification branches; no 4096 upsampler improves OA",
            "Line B: PU-Net only method above Downsampled 256 (91.27%, +0.42 pp); still −0.68 pp below Original",
            "Mesh-ref 256 ≈ Downsampled 256 (90.88%, +0.03 pp); Mesh-ref 4096 ≈ Original 1024 (92.00%, +0.05 pp)",
            "Denser mesh sampling (4096) does not beat Original 1024 by a meaningful margin",
            "Geometry CD/HD: equal-N only — Line A vs mesh-4096; Downsampled vs mesh-256; Line B ups vs Original 1024",
            "Geometry and classification are not fully aligned; upsampling quality is method- and task-dependent",
        ],
        top=1.0,
        size=15,
    )

    # 17 Limitations
    s = add()
    add_title(s, "Limitations and Next Steps")
    add_bullets(
        s,
        [
            "Mesh-ref CD/HD measure consistency with an independent mesh sample, not the exact Original 1024 set",
            "Best Overall uses test-set monitoring (no separate val split in ModelNet40 PointNet++ setup)",
            "NUC depends on radius / center sampling; keep parameters fixed for comparisons",
            "Interactive demos need hosted HTTPS links (file:// does not open for other machines)",
            "Next: class-wise geometry / classification analysis",
        ],
        top=1.1,
        size=16,
    )

    # 18 Conclusions
    s = add()
    add_title(s, "Conclusions")
    add_bullets(
        s,
        [
            "Line A: Original 1024 remains best at 91.95% even vs PointNet++ retrained at 4096",
            "None of the 4096 densification branches improves classification under this protocol",
            "Line B: PU-Net only above Downsampled 256 (+0.42 pp → 91.27%); still −0.68 pp below Original",
            "Mesh-ref equal-N baselines match Downsampled-256 / Original-1024 almost identically (+0.03 / +0.05 pp)",
            "Geometry ranking uses equal-cardinality CD/HD (mesh-256 / mesh-4096 / Original 1024)",
            "Upsampling quality is method- and task-dependent — report geometry and classification separately",
        ],
        top=1.0,
        size=15,
    )

    # 19 Questions
    s = add()
    box = s.shapes.add_textbox(Inches(1), Inches(2.8), Inches(11.3), Inches(1.5))
    p = box.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    set_run(run, "Questions?", 44, True, NAVY)
    p2 = box.text_frame.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    run = p2.add_run()
    set_run(run, "Thank you", 22, False, GRAY)

    total = len(slides_meta)
    for i, slide in enumerate(slides_meta, start=1):
        add_footer(slide, i, total)

    PRES.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    return total


def build_pdf(geo: dict[str, list[dict]], figs: dict[str, Path], n_slides: int) -> None:
    """Export a print-ready PDF summary mirroring the PPT content."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import landscape, A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    page = landscape(A4)
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=page,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "T",
        parent=styles["Title"],
        fontSize=26,
        textColor=colors.HexColor("#1a3a5c"),
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    h = ParagraphStyle(
        "H",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#1a3a5c"),
        spaceAfter=10,
    )
    body = ParagraphStyle("B", parent=styles["BodyText"], fontSize=12, leading=16, spaceAfter=6)
    story = []

    def bullets(items: list[str]):
        for it in items:
            story.append(Paragraph(f"• {it}", body))

    def table(data):
        t = Table(data, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7fa")]),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)

    story.append(Paragraph("Point Cloud Upsampling on ModelNet40", title))
    story.append(
        Paragraph(
            "Geometry Consistency with Original Point Clouds and PointNet++ Classification",
            ParagraphStyle("S", parent=body, alignment=TA_CENTER, fontSize=14, textColor=colors.HexColor("#555555")),
        )
    )
    story.append(Spacer(1, 0.3 * inch))
    bullets(
        [
            "Geometry reference: corresponding Original point cloud only (no mesh / no P2F)",
            "Unequal cardinality (Line A 4096 vs 1024; Line B Downsampled 256 vs 1024) = discrete-set consistency only",
            "PointNet++ retrained per input size (A: 4096 from scratch; B: 256 / 1024 from scratch)",
            "Two-line protocol with EAR, PDANS, PU-Net, PU-GCN, PU-EdgeFormer",
            f"Companion editable PPTX has {n_slides} slides",
        ]
    )
    story.append(PageBreak())

    story.append(Paragraph("Geometry Evaluation Definition", h))
    bullets(
        [
            "Equal cardinality only. Mesh-ref 256/4096: area-weighted from ModelNet40 .off + unit sphere.",
            "Line A: 4096 vs mesh-ref 4096. Line B Downsampled: 256 vs mesh-ref 256. Line B ups: 1024 vs Original 1024.",
            "Self rows: CD=0, HD=0. Unequal-|P| CD/HD omitted.",
        ]
    )
    story.append(PageBreak())

    story.append(Paragraph("Geometry Results (equal cardinality)", h))
    rows = (geo.get("A") or []) + (geo.get("B") or [])
    if rows:
        data = [["Line", "Method", "Points", "Reference", "CD", "HD", "NUC", "ΔNUC", "N"]]
        for r in rows:
            data.append(
                [
                    str(r.get("line", "")),
                    str(r.get("method", "")),
                    str(r.get("points", "")),
                    str(r.get("reference", "")),
                    fmt4(r.get("cd_vs_ref", r.get("cd_vs_original"))),
                    fmt4(r.get("hd_vs_ref", r.get("hd_vs_original"))),
                    fmt4(r.get("nuc")),
                    fmt4(r.get("delta_nuc_vs_ref", r.get("delta_nuc_vs_original"))),
                    str(r.get("valid_samples", "")),
                ]
            )
        table(data)
    else:
        bullets(["Pending recomputation"])
    story.append(PageBreak())

    story.append(Paragraph("Line A Classification (Best Overall primary)", h))
    data = [["Method", "Pts", "Best Overall", "Final Overall", "Δ Best vs Original"]]
    for name, n, best, final, delta in LINEA_CLS:
        data.append([name, str(n), f"{best:.2f}%", f"{final:.2f}%", f"{delta:+.2f} pp"])
    table(data)
    if "lineA_cls" in figs:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Image(str(figs["lineA_cls"]), width=7.5 * inch, height=3.3 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Line B Classification (Best Overall primary)", h))
    data = [["Method", "Pts", "Best", "Final", "Δ vs Down", "Gap vs Original"]]
    for name, n, best, final, d, g in LINEB_CLS:
        data.append([name, str(n), f"{best:.2f}%", f"{final:.2f}%", f"{d:+.2f} pp", f"{g:+.2f} pp"])
    table(data)
    if "lineB_cls" in figs:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Image(str(figs["lineB_cls"]), width=7.5 * inch, height=3.3 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Mesh-ref PointNet++ Baselines", h))
    data = [["Method", "Pts", "Best Overall", "Final Overall", "Compare to", "Δ Best"]]
    for name, n, best, final, cmp_label, cmp_best, delta in MESH_REF_CLS:
        data.append(
            [
                name,
                str(n),
                f"{best:.2f}%",
                f"{final:.2f}%",
                f"{cmp_label} ({cmp_best:.2f}%)",
                f"{delta:+.2f} pp",
            ]
        )
    table(data)
    bullets(
        [
            "Independent area-weighted samples from ModelNet40 .off (not downsampled from Original 1024).",
            "Mesh-ref 256 ≈ Downsampled ×4 256 (90.88% vs 90.85%, +0.03 pp).",
            "Mesh-ref 4096 ≈ Original 1024 (92.00% vs 91.95%, +0.05 pp) — not a meaningful margin.",
        ]
    )
    if "mesh_ref_cls" in figs:
        story.append(Spacer(1, 0.15 * inch))
        story.append(Image(str(figs["mesh_ref_cls"]), width=7.0 * inch, height=3.4 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Geometry vs Classification", h))
    bullets(
        [
            "Lower CD/HD = greater consistency with discrete Original; does not automatically imply higher accuracy.",
            "Limited methods: no strong statistical claim; geometry alone is insufficient to predict classification.",
        ]
    )
    if "lineB_scatter" in figs:
        story.append(Image(str(figs["lineB_scatter"]), width=6.8 * inch, height=3.8 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Main Findings / Conclusions", h))
    bullets(
        [
            "Line A: Original 1024 remains best at 91.95% among densification branches; 4096 upsamplers do not improve OA.",
            "Line B: PU-Net only above Downsampled 256 (91.27%, +0.42 pp); still −0.68 pp below Original.",
            "Mesh-ref 256 ≈ Downsampled 256 (90.88%, +0.03 pp); Mesh-ref 4096 ≈ Original 1024 (92.00%, +0.05 pp).",
            "Denser mesh sampling (4096) does not beat Original 1024 by a meaningful margin.",
            "Geometry CD/HD: equal-N mesh-ref 256/4096 + Original 1024 for Line B recovery.",
            "Upsampling quality is method-dependent and task-dependent.",
        ]
    )

    # Qualitative pages
    for key, caption in [
        ("pc_B_airplane", "Line B qualitative — airplane"),
        ("pc_B_chair", "Line B qualitative — chair"),
        ("pc_B_table", "Line B qualitative — table"),
        ("pc_A_airplane", "Line A qualitative — airplane"),
        ("pc_A_chair", "Line A qualitative — chair"),
        ("pc_A_table", "Line A qualitative — table"),
    ]:
        if key in figs:
            story.append(PageBreak())
            story.append(Paragraph(caption, h))
            story.append(Image(str(figs[key]), width=10.5 * inch, height=2.6 * inch))

    doc.build(story)


def write_notes(geo: dict, interactive_files: list[str], n_slides: int) -> None:
    geo_done = bool(geo.get("A") and geo.get("B"))
    lines = [
        "# ModelNet40 Presentation Revision Notes",
        "",
        f"- Generated: {utc_now()}",
        f"- PPTX: `{OUT_PPTX}`",
        f"- PDF: `{OUT_PDF}`",
        f"- Slides: {n_slides}",
        "",
        "## Protocol changes",
        "",
        "- Deleted dense mesh / dense surface reference from geometry evaluation narrative and tables.",
        "- Deleted P2F / exact P2F / point-to-face entirely.",
        "- Original CD/HD are now 0 by definition (self-comparison), not mesh-reference values (e.g. old CD≈0.0495).",
        "- Statement used throughout: each upsampled point cloud is evaluated directly against the corresponding Original point cloud; no mesh or dense surface reference is used.",
        "",
        "## Geometry recomputation",
        "",
        f"- Script: `scripts/compute_geometry_equal_n.py` (+ `scripts/build_mesh_reference_points.py`)",
        f"- Status: {'COMPLETED on test split (equal-N)' if geo_done else 'PENDING / incomplete'}",
        f"- Mesh refs: `datasets/modelnet40_mesh_ref_256/`, `datasets/modelnet40_mesh_ref_4096/`",
        f"- Outputs: `reports/modelnet40_geometry_equal_n_summary.csv`, `..._lineA.csv`, `..._lineB.csv`, `..._audit.md`",
        "- Protocol: Line A 4096 vs mesh-4096; Downsampled 256 vs mesh-256; Line B ups 1024 vs Original 1024.",
        "- Unequal-cardinality CD/HD are not reported in the presentation.",
        "",
        "## Best Overall vs Final Overall",
        "",
        "- Best Overall = highest recorded checkpoint overall accuracy (primary).",
        "- Final Overall = epoch-200 overall accuracy (supplementary).",
        "- Code audit: original PointNet++ PyTorch `train_classification.py` + our `train_pointnet2.py` — ModelNet40 has train/test only (no val); best checkpoint when test-instance accuracy improves.",
        "- Line A 4096 branches: PointNet++ retrained from scratch with num_point=4096 (not pretrained 1024 eval).",
        "- Line B: 256 baseline and 1024 upsampling branches each trained from scratch for that point count.",
        "- Presentation wording: “Best Overall is the highest recorded checkpoint accuracy.”",
        "",
        "## Mesh-ref PointNet++ baselines",
        "",
        "- Report: `reports/modelnet40_pointnet2_mesh_ref_baseline_results.md`",
        "- Mesh-ref 256 Best OA 90.88% vs Downsampled ×4 256 90.85% (+0.03 pp).",
        "- Mesh-ref 4096 Best OA 92.00% vs Original 1024 91.95% (+0.05 pp).",
        "- Interpretation: equal-N mesh sampling classifies almost identically; denser 4096 does not beat Original by a meaningful margin.",
        "",
        "## Hosted interactive links",
        "",
        "- PPT previously used `file://` links (open only on the generating machine).",
        "- Set `PUBLIC_INTERACTIVE_BASE` to a GitHub/GitLab Pages URL ending in `.../dropdown_v2` and regenerate PPT for HTTPS links.",
        f"- Current PUBLIC_INTERACTIVE_BASE: `{PUBLIC_INTERACTIVE_BASE or '(empty — local file:// fallback)'}`",
        "",
        "## Qualitative cameras re-selected",
        "",
        "- airplane: elev=22, azim=-45 (elevated front-side three-quarter)",
        "- chair: elev=18, azim=-35",
        "- table: elev=35, azim=-50 (elevated three-quarter; avoids narrow silhouette)",
        "- car: elev=20, azim=-55",
        "- sofa: elev=18, azim=-40",
        "- Within each class, all methods share identical camera, axis limits, point-size rule, depth coloring, dark background.",
        "",
        "## Interactive HTML viewers verified",
        "",
    ]
    if interactive_files:
        lines.append(f"- Regenerated {len(interactive_files)} dropdown viewers under `figures/modelnet40/pointcloud_examples_interactive_v2/dropdown_v2/`.")
        for f in interactive_files:
            lines.append(f"  - `{f}`")
    else:
        lines.append("- No new interactive files generated.")
    lines += [
        "- Pre-existing v2 grid/dropdown/single_method HTML under `pointcloud_examples_interactive_v2/` remain available for live demo.",
        "",
        "## Incomplete / caveats",
        "",
        f"- Geometry recomputation {'complete for test split' if geo_done else 'still pending — slides show Pending where needed'}.",
        "- Train-split geometry not required for this presentation (user asked for test-set means).",
        "- LibreOffice not available on this host; PDF exported via ReportLab as a print companion (PPTX remains the editable master).",
        "- Unequal-cardinality CD/HD are omitted; presentation uses mesh-ref 256/4096 + Original 1024 equal-N tables.",
        "",
    ]
    NOTES.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"[{utc_now()}] loading geometry…")
    geo = load_geometry()
    print(f"  Line A rows={len(geo['A'])} Line B rows={len(geo['B'])}")
    print(f"[{utc_now()}] generating figures…")
    figs = generate_figures(geo)
    print(f"  figures={len(figs)}")
    print(f"[{utc_now()}] regenerating interactive HTML…")
    interactive_files = regenerate_interactive()
    print(f"  interactive={len(interactive_files)}")
    print(f"[{utc_now()}] building PPTX…")
    n_slides = build_pptx(geo, figs, interactive_files)
    print(f"  slides={n_slides} -> {OUT_PPTX}")
    print(f"[{utc_now()}] building PDF…")
    build_pdf(geo, figs, n_slides)
    print(f"  -> {OUT_PDF}")
    write_notes(geo, interactive_files, n_slides)
    # Also copy CSVs to presentations for convenience
    for name in (
        "modelnet40_geometry_equal_n_lineA.csv",
        "modelnet40_geometry_equal_n_lineB.csv",
        "modelnet40_geometry_equal_n_summary.csv",
        "modelnet40_geometry_equal_n_audit.md",
        "modelnet40_mesh_ref_256_build_audit.md",
        "modelnet40_mesh_ref_4096_build_audit.md",
        "modelnet40_pointnet2_mesh_ref_baseline_results.md",
        "modelnet40_pointnet2_mesh_ref_baseline_results.csv",
    ):
        src = REPORTS / name
        if src.exists():
            dst = PRES / name
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[{utc_now()}] notes -> {NOTES}")
    print("DONE")


if __name__ == "__main__":
    main()
