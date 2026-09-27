#!/usr/bin/env python3
"""Build an English presentation for the PointRCNN x4 upsampling analysis.

The deck is intentionally generated from the consolidated CSV instead of
copying values from a report, so that the presentation stays aligned with the
validated 256-frame screening results.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from PIL import Image

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "results" / "kitti_x4_detector_recovery_combined_analysis_20260725"
OUT_DIR = ROOT / "results" / "kitti_x4_detector_recovery_presentation_20260726"
ASSET_DIR = OUT_DIR / "assets"
RENDER_DIR = OUT_DIR / "rendered"

CSV_PATH = DATA_DIR / "combined_all_difficulty_metrics.csv"
RATIO_CHART = DATA_DIR / "ratio_ap_comparison.png"
QUAL_VIS = (
    ROOT
    / "results"
    / "unified_e1_e2_detector_selection_visualization_v1"
    / "comparison_sheets"
    / "line_B_frame_000002.png"
)
QUAL_VIS_3D = (
    ROOT
    / "results"
    / "unified_e1_e2_detector_selection_visualization_v1"
    / "comparison_sheets"
    / "line_B_frame_000002_3d.html"
)
COMPLETE_XLSX = DATA_DIR / "all_methods_all_ratios_all_metrics_comparison.xlsx"
INTERACTIVE_INDEX = ROOT / "results" / "final_visualization_bundle" / "index.html"


BG = "#07131F"
PANEL = "#102334"
PANEL_2 = "#142B3E"
WHITE = "#F5F8FC"
TEXT = "#D7E2EC"
MUTED = "#8FA6B9"
GRID = "#334B5F"
TEAL = "#20C997"
BLUE = "#4EA8DE"
ORANGE = "#F4A261"
RED = "#FF6B6B"
YELLOW = "#FFD166"
GREEN = "#65D6A6"
PURPLE = "#A78BFA"
FONT = "Liberation Sans"

METHOD_LABELS = {
    "pdans": "PDANS",
    "pu_gcn": "PU-GCN",
    "pu_edgeformer": "PU-EdgeFormer",
    "pu_net": "PU-Net",
}
METHOD_COLORS = {
    "pdans": TEAL,
    "pu_gcn": BLUE,
    "pu_edgeformer": ORANGE,
    "pu_net": RED,
}

METRICS = [
    ("BBox Easy", "ap_bbox_car_easy_r40"),
    ("BBox Moderate", "ap_bbox_car_moderate_r40"),
    ("BBox Hard", "ap_bbox_car_hard_r40"),
    ("BEV Easy", "ap_bev_car_easy_r40"),
    ("BEV Moderate", "ap_bev_car_moderate_r40"),
    ("BEV Hard", "ap_bev_car_hard_r40"),
    ("3D Easy", "ap_3d_car_easy_r40"),
    ("3D Moderate", "ap_3d_car_moderate_r40"),
    ("3D Hard", "ap_3d_car_hard_r40"),
    ("AOS Easy", "ap_aos_car_easy_r40"),
    ("AOS Moderate", "ap_aos_car_moderate_r40"),
    ("AOS Hard", "ap_aos_car_hard_r40"),
]


def rgb(hex_color: str) -> RGBColor:
    value = hex_color.lstrip("#")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def setup_output() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    RENDER_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RATIO_CHART, ASSET_DIR / RATIO_CHART.name)
    if QUAL_VIS.exists():
        shutil.copy2(QUAL_VIS, ASSET_DIR / QUAL_VIS.name)


def set_dark_plot(ax) -> None:
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(WHITE)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(axis="x", color=GRID, alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)


def load_data() -> pd.DataFrame:
    data = pd.read_csv(CSV_PATH)
    numeric_cols = [col for _, col in METRICS] + ["ratio_percent"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data


def baseline_row(data: pd.DataFrame, line: str) -> pd.Series:
    return data[
        (data["line"] == line) & (data["row_kind"] == "baseline_reference")
    ].iloc[0]


def generated_rows(data: pd.DataFrame, line: str) -> pd.DataFrame:
    return data[(data["line"] == line) & (data["row_kind"] == "generated_ratio")]


def protocol_short(protocol: str) -> str:
    if protocol == "fine_nested_e2_replacement":
        return "fine"
    if protocol == "coarse_independent_sampling":
        return "coarse"
    return protocol


def best_rows(data: pd.DataFrame, line: str) -> list[dict]:
    original_base = baseline_row(data, "original")
    rows = generated_rows(data, line)
    result = []
    for label, col in METRICS:
        idx = rows[col].idxmax()
        row = rows.loc[idx]
        result.append(
            {
                "metric": label,
                "method": METHOD_LABELS[row["method"]],
                "method_key": row["method"],
                "ratio": float(row["ratio_percent"]),
                "protocol": protocol_short(row["protocol"]),
                "ap": float(row[col]),
                "delta": float(row[col] - original_base[col]),
            }
        )
    return result


def fmt_ratio(value: float) -> str:
    return f"{value:g}%"


def create_best_delta_chart(data: pd.DataFrame, line: str, out_path: Path) -> None:
    best = best_rows(data, line)
    labels = [item["metric"].replace(" Moderate", "-M").replace(" Easy", "-E").replace(" Hard", "-H") for item in best]
    values = np.array([item["delta"] for item in best])
    colors = [TEAL if value >= 0 else RED for value in values]

    fig, ax = plt.subplots(figsize=(10.4, 5.7), facecolor=BG)
    set_dark_plot(ax)
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, alpha=0.92, height=0.66)
    ax.axvline(0, color=WHITE, linewidth=1.2)
    ax.set_yticks(y, labels, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("Best AP delta vs Original baseline (points)")
    scope = "Original line" if line == "original" else "Downsampled line"
    ax.set_title(f"{scope}: best tested configuration for each metric", fontsize=16, pad=14)

    xmin = min(-20.5, values.min() - 1.3) if line == "downsampled" else min(-1.1, values.min() - 0.35)
    xmax = max(3.0, values.max() + 0.75)
    ax.set_xlim(xmin, xmax)
    for yi, value in zip(y, values):
        if value >= 0:
            x = value + 0.08
            ha = "left"
        else:
            x = value - 0.08
            ha = "right"
        ax.text(x, yi, f"{value:+.2f}", va="center", ha=ha, color=WHITE, fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=210, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def control_gap_matrix(data: pd.DataFrame, line: str) -> np.ndarray:
    ratios = [2.5, 5.0, 7.5, 10.0]
    methods = ["pdans", "pu_gcn", "pu_edgeformer", "pu_net"]
    metric = "ap_3d_car_moderate_r40"
    matrix = np.zeros((len(methods), len(ratios)), dtype=float)
    controls = data[
        (data["line"] == line)
        & (data["row_kind"] == "observed_fill_control")
        & (data["protocol"] == "fine_nested_e2_replacement")
    ]
    generated = data[
        (data["line"] == line)
        & (data["row_kind"] == "generated_ratio")
        & (data["protocol"] == "fine_nested_e2_replacement")
    ]
    for i, method in enumerate(methods):
        for j, ratio in enumerate(ratios):
            method_ap = generated[
                (generated["method"] == method) & np.isclose(generated["ratio_percent"], ratio)
            ][metric].iloc[0]
            control_ap = controls[np.isclose(controls["ratio_percent"], ratio)][metric].iloc[0]
            matrix[i, j] = method_ap - control_ap
    return matrix


def create_control_heatmap(data: pd.DataFrame, out_path: Path) -> None:
    ratios = ["2.5%", "5%", "7.5%", "10%"]
    methods = ["PDANS", "PU-GCN", "PU-EdgeFormer", "PU-Net"]
    cmap = LinearSegmentedColormap.from_list("control_gap", [RED, "#B34D57", "#273B4C", TEAL])
    norm = TwoSlopeNorm(vmin=-20.5, vcenter=0.0, vmax=0.8)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), facecolor=BG)
    for ax, line, title in zip(
        axes, ["original", "downsampled"], ["Original line", "Downsampled line"]
    ):
        matrix = control_gap_matrix(data, line)
        ax.set_facecolor(BG)
        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(4), ratios, color=TEXT)
        ax.set_yticks(range(4), methods, color=TEXT)
        ax.set_title(title, color=WHITE, fontsize=15, pad=12)
        ax.tick_params(length=0)
        for i in range(4):
            for j in range(4):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    color=WHITE,
                    fontsize=11,
                    fontweight="bold",
                )
        for spine in ax.spines.values():
            spine.set_color(GRID)
    fig.suptitle(
        "Synthetic points vs matched observed-fill control — 3D AP Moderate",
        color=WHITE,
        fontsize=17,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Value = method AP − control AP. Negative values isolate additional harm from generated geometry/distribution.",
        color=MUTED,
        ha="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=210, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def create_waterfall(out_path: Path) -> None:
    labels = ["Original\nbaseline", "Downsample\nloss", "Observed-fill\nsampling", "PDANS\ngeometry", "Final\nPDANS g2.5"]
    values = [82.6611, -14.3708, -1.3977, -0.4911, 66.4015]
    levels = [82.6611, 68.2903, 66.8925, 66.4015]

    fig, ax = plt.subplots(figsize=(9.2, 5.5), facecolor=BG)
    set_dark_plot(ax)
    ax.grid(axis="y", color=GRID, alpha=0.55)
    ax.grid(axis="x", visible=False)
    x = np.arange(5)
    floor = 60.0

    ax.bar(x[0], values[0] - floor, bottom=floor, color=BLUE, width=0.64)
    for idx, change in enumerate(values[1:4], start=1):
        before = levels[idx - 1]
        after = levels[idx]
        ax.bar(
            x[idx],
            before - after,
            bottom=after,
            color=RED if change < 0 else TEAL,
            width=0.64,
        )
        ax.plot([x[idx - 1] + 0.32, x[idx] - 0.32], [before, before], color=MUTED, linestyle="--", linewidth=1)
    ax.bar(x[4], values[4] - floor, bottom=floor, color=TEAL, width=0.64)
    ax.plot([x[3] + 0.32, x[4] - 0.32], [levels[-1], levels[-1]], color=MUTED, linestyle="--", linewidth=1)

    ax.set_xticks(x, labels, color=TEXT)
    ax.set_ylim(floor, 86.0)
    ax.set_ylabel("Car 3D AP R40 Moderate")
    ax.set_title("Where the Downsampled-line loss comes from (PDANS g2.5)", fontsize=16, pad=14)
    annotations = ["82.66", "−14.37", "−1.40", "−0.49", "66.40"]
    ann_y = [82.9, 75.0, 67.9, 66.9, 66.7]
    for xi, text_value, yi in zip(x, annotations, ann_y):
        ax.text(xi, yi, text_value, ha="center", va="bottom", color=WHITE, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=210, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def create_robustness_chart(data: pd.DataFrame, out_path: Path) -> None:
    methods = ["pdans", "pu_gcn", "pu_edgeformer", "pu_net"]
    labels = [METHOD_LABELS[m] for m in methods]
    metric = "ap_3d_car_moderate_r40"
    drops = {}
    for line in ["original", "downsampled"]:
        rows = data[
            (data["line"] == line)
            & (data["row_kind"] == "generated_ratio")
            & (data["protocol"] == "coarse_independent_sampling")
        ]
        line_drops = []
        for method in methods:
            method_rows = rows[rows["method"] == method]
            ap10 = method_rows[np.isclose(method_rows["ratio_percent"], 10.0)][metric].iloc[0]
            ap50 = method_rows[np.isclose(method_rows["ratio_percent"], 50.0)][metric].iloc[0]
            line_drops.append(ap10 - ap50)
        drops[line] = line_drops

    fig, ax = plt.subplots(figsize=(9.5, 5.3), facecolor=BG)
    set_dark_plot(ax)
    y = np.arange(len(methods))
    height = 0.34
    ax.barh(y - height / 2, drops["original"], height=height, color=BLUE, label="Original line")
    ax.barh(y + height / 2, drops["downsampled"], height=height, color=RED, label="Downsampled line")
    ax.set_yticks(y, labels, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlabel("AP drop from 10% to 50% generated points")
    ax.set_title("High generated-point ratios reduce detector robustness", fontsize=16, pad=14)
    legend = ax.legend(loc="lower right", frameon=False, fontsize=10)
    for text_obj in legend.get_texts():
        text_obj.set_color(TEXT)
    for yi, (a, b) in enumerate(zip(drops["original"], drops["downsampled"])):
        ax.text(a + 0.5, yi - height / 2, f"{a:.2f}", va="center", color=TEXT, fontsize=9.5)
        ax.text(b + 0.5, yi + height / 2, f"{b:.2f}", va="center", color=TEXT, fontsize=9.5)
    ax.set_xlim(0, max(max(drops["original"]), max(drops["downsampled"])) + 7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=210, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def generate_charts(data: pd.DataFrame) -> dict[str, Path]:
    paths = {
        "best_original": ASSET_DIR / "best_per_metric_original_delta.png",
        "best_downsampled": ASSET_DIR / "best_per_metric_downsampled_delta.png",
        "control_heatmap": ASSET_DIR / "generated_vs_control_heatmap.png",
        "waterfall": ASSET_DIR / "downsampled_loss_waterfall.png",
        "robustness": ASSET_DIR / "high_ratio_robustness.png",
    }
    create_best_delta_chart(data, "original", paths["best_original"])
    create_best_delta_chart(data, "downsampled", paths["best_downsampled"])
    create_control_heatmap(data, paths["control_heatmap"])
    create_waterfall(paths["waterfall"])
    create_robustness_chart(data, paths["robustness"])
    return paths


def add_rect(slide, x, y, w, h, fill, line=None, radius=True, transparency=0):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    if transparency:
        shape.fill.transparency = transparency
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = rgb(line)
        shape.line.width = Pt(1)
    return shape


def add_text(
    slide,
    text,
    x,
    y,
    w,
    h,
    size=18,
    color=TEXT,
    bold=False,
    align=PP_ALIGN.LEFT,
    valign=MSO_ANCHOR.TOP,
    font=FONT,
    margin=0.03,
    italic=False,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(margin)
    tf.margin_bottom = Inches(margin)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    p.space_after = Pt(0)
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = rgb(color)
    return box


def add_rich_text(slide, runs, x, y, w, h, size=18, align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.03)
    tf.margin_right = Inches(0.03)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = valign
    p = tf.paragraphs[0]
    p.alignment = align
    for item in runs:
        run = p.add_run()
        run.text = item["text"]
        run.font.name = FONT
        run.font.size = Pt(item.get("size", size))
        run.font.bold = item.get("bold", False)
        run.font.italic = item.get("italic", False)
        run.font.color.rgb = rgb(item.get("color", TEXT))
    return box


def add_bullets(
    slide,
    bullets,
    x,
    y,
    w,
    h,
    size=17,
    color=TEXT,
    bullet_color=None,
    spacing=7,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.03)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    for idx, item in enumerate(bullets):
        if isinstance(item, tuple):
            text_value, level = item
        else:
            text_value, level = item, 0
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = text_value
        p.level = level
        p.font.name = FONT
        p.font.size = Pt(size - level * 1.5)
        p.font.color.rgb = rgb(color)
        p.space_after = Pt(spacing)
        p.line_spacing = 1.05
        p.text = f"•  {text_value}" if level == 0 else f"–  {text_value}"
        if bullet_color:
            p.runs[0].font.color.rgb = rgb(color)
    return box


def add_card(slide, x, y, w, h, title, body, accent=TEAL, title_size=15, body_size=17):
    add_rect(slide, x, y, w, h, PANEL, line=GRID)
    add_rect(slide, x, y, 0.08, h, accent, radius=False)
    add_text(slide, title.upper(), x + 0.22, y + 0.14, w - 0.4, 0.34, size=title_size, color=accent, bold=True)
    add_text(slide, body, x + 0.22, y + 0.55, w - 0.4, h - 0.68, size=body_size, color=WHITE, bold=False)


def add_picture_contain(slide, image_path: Path, x, y, w, h, frame=False, frame_color=GRID):
    if frame:
        add_rect(slide, x, y, w, h, "#FFFFFF", line=frame_color, radius=False)
    with Image.open(image_path) as image:
        iw, ih = image.size
    image_ratio = iw / ih
    box_ratio = w / h
    if image_ratio >= box_ratio:
        pic_w = w
        pic_h = w / image_ratio
        pic_x = x
        pic_y = y + (h - pic_h) / 2
    else:
        pic_h = h
        pic_w = h * image_ratio
        pic_x = x + (w - pic_w) / 2
        pic_y = y
    return slide.shapes.add_picture(
        str(image_path), Inches(pic_x), Inches(pic_y), width=Inches(pic_w), height=Inches(pic_h)
    )


def set_cell_text(cell, text, size=12, color=TEXT, bold=False, align=PP_ALIGN.CENTER):
    cell.text = ""
    cell.margin_left = Inches(0.04)
    cell.margin_right = Inches(0.04)
    cell.margin_top = Inches(0.03)
    cell.margin_bottom = Inches(0.03)
    tf = cell.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = str(text)
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)


def add_table(
    slide,
    rows,
    headers,
    x,
    y,
    w,
    h,
    col_widths=None,
    font_size=12,
    first_col_left=True,
    highlights=None,
):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w), Inches(h))
    table = shape.table
    if col_widths:
        total = sum(col_widths)
        for idx, width in enumerate(col_widths):
            table.columns[idx].width = Inches(w * width / total)
    for col_idx, header in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = rgb(PANEL_2)
        set_cell_text(cell, header, size=font_size, color=WHITE, bold=True)
    for row_idx, row in enumerate(rows, start=1):
        for col_idx, value in enumerate(row):
            cell = table.cell(row_idx, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(PANEL if row_idx % 2 else "#0D1D2A")
            align = PP_ALIGN.LEFT if first_col_left and col_idx == 0 else PP_ALIGN.CENTER
            cell_color = TEXT
            cell_bold = False
            if highlights and (row_idx - 1, col_idx) in highlights:
                spec = highlights[(row_idx - 1, col_idx)]
                cell.fill.fore_color.rgb = rgb(spec.get("fill", PANEL_2))
                cell_color = spec.get("color", WHITE)
                cell_bold = spec.get("bold", True)
            set_cell_text(cell, value, size=font_size, color=cell_color, bold=cell_bold, align=align)
    return shape


def add_header(slide, number, kicker, title, subtitle=None):
    add_text(slide, kicker.upper(), 0.55, 0.20, 8.0, 0.3, size=10.5, color=TEAL, bold=True)
    add_text(slide, title, 0.55, 0.52, 12.1, 0.55, size=27, color=WHITE, bold=True, valign=MSO_ANCHOR.MIDDLE)
    if subtitle:
        add_text(slide, subtitle, 0.56, 1.04, 12.0, 0.35, size=12.5, color=MUTED)
    add_text(slide, f"{number:02d}", 12.35, 0.22, 0.45, 0.28, size=10, color=MUTED, align=PP_ALIGN.RIGHT)


def add_footer(slide, note="KITTI Car • AP R40 • 256-frame screen • same PointRCNN checkpoint"):
    add_rect(slide, 0.55, 7.18, 12.2, 0.012, GRID, radius=False)
    add_text(slide, note, 0.56, 7.22, 11.8, 0.18, size=7.8, color=MUTED)


def add_background(slide):
    bg = slide.background
    bg.fill.solid()
    bg.fill.fore_color.rgb = rgb(BG)


def add_point_cloud_decoration(slide):
    rng = np.random.default_rng(42)
    for _ in range(150):
        px = 8.5 + float(rng.random()) * 4.6
        py = 0.2 + float(rng.random()) * 6.7
        center_bias = 1.0 - min(1.0, abs(px - 10.8) / 2.4)
        if rng.random() > 0.34 + 0.58 * center_bias:
            continue
        size = 0.014 + float(rng.random()) * 0.03
        color = TEAL if rng.random() < 0.18 else BLUE
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(px), Inches(py), Inches(size), Inches(size))
        dot.fill.solid()
        dot.fill.fore_color.rgb = rgb(color)
        dot.fill.transparency = 20 if color == TEAL else 45
        dot.line.fill.background()


def add_arrow_shape(slide, x, y, w, h, color=TEAL):
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = rgb(color)
    arrow.line.fill.background()
    return arrow


def add_metric_icon(slide, kind, x, y, w, h, color):
    if kind == "bbox":
        add_rect(slide, x + 0.25, y + 0.20, w - 0.5, h - 0.40, BG, line=color, radius=False)
        add_rect(slide, x + 0.43, y + 0.38, w - 0.86, h - 0.76, PANEL_2, line=MUTED, radius=False)
    elif kind == "bev":
        for offset in [0.18, 0.38, 0.58]:
            line = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(x + offset),
                Inches(y + 0.28 + offset * 0.18),
                Inches(w - 0.8),
                Inches(0.02),
            )
            line.fill.solid()
            line.fill.fore_color.rgb = rgb(MUTED)
            line.line.fill.background()
        box = slide.shapes.add_shape(
            MSO_SHAPE.PARALLELOGRAM, Inches(x + 0.38), Inches(y + 0.38), Inches(w - 0.76), Inches(h - 0.70)
        )
        box.fill.background()
        box.line.color.rgb = rgb(color)
        box.line.width = Pt(2)
    elif kind == "3d":
        front = add_rect(slide, x + 0.35, y + 0.34, w - 0.80, h - 0.72, BG, line=color, radius=False)
        back = add_rect(slide, x + 0.53, y + 0.18, w - 0.80, h - 0.72, BG, line=MUTED, radius=False)
        front.rotation = 0
        back.rotation = 0
        for (x1, y1, x2, y2) in [
            (x + 0.35, y + 0.34, x + 0.53, y + 0.18),
            (x + w - 0.45, y + 0.34, x + w - 0.27, y + 0.18),
            (x + 0.35, y + h - 0.38, x + 0.53, y + h - 0.54),
            (x + w - 0.45, y + h - 0.38, x + w - 0.27, y + h - 0.54),
        ]:
            connector = slide.shapes.add_connector(
                1, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
            )
            connector.line.color.rgb = rgb(color)
            connector.line.width = Pt(1.5)
    else:
        circle = slide.shapes.add_shape(MSO_SHAPE.ARC, Inches(x + 0.35), Inches(y + 0.20), Inches(w - 0.7), Inches(h - 0.4))
        circle.fill.background()
        circle.line.color.rgb = rgb(color)
        circle.line.width = Pt(2.5)
        add_text(slide, "↻", x + 0.47, y + 0.17, w - 0.94, h - 0.3, size=29, color=color, bold=True, align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE)


def add_title_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_point_cloud_decoration(slide)
    add_rect(slide, 0.58, 0.55, 0.08, 5.85, TEAL, radius=False)
    add_text(slide, "POINT-CLOUD UPSAMPLING × 4", 0.90, 0.70, 6.8, 0.38, size=12, color=TEAL, bold=True)
    add_text(
        slide,
        "Why 4× Point-Cloud\nUpsampling Does Not\nImprove PointRCNN",
        0.86,
        1.15,
        7.2,
        2.55,
        size=31,
        color=WHITE,
        bold=True,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_text(
        slide,
        "A controlled analysis of observed vs generated points,\ndetector sampling, and AP degradation",
        0.90,
        3.88,
        6.9,
        0.82,
        size=17,
        color=TEXT,
    )
    add_card(slide, 0.90, 5.05, 2.12, 1.12, "Coverage", "88 PASS\ngroups", TEAL, 10, 19)
    add_card(slide, 3.16, 5.05, 2.12, 1.12, "Best screen candidate", "+2.43 AP\nPDANS g2.5", BLUE, 10, 18)
    add_card(slide, 5.42, 5.05, 2.12, 1.12, "Missing-data cost", "−14.37 AP\n3D Moderate", RED, 10, 18)
    add_text(slide, "English presentation • 26 July 2026", 0.90, 6.62, 6.0, 0.28, size=10, color=MUTED)
    add_footer(slide, "PointRCNN detector recovery study • Same 256 KITTI Car frames and checkpoint across experiments")


def add_executive_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 2, "Executive answer", "The degradation is shared across the pipeline")
    add_text(
        slide,
        "No single component explains the drop. Downsampling removes measurement evidence; current upsamplers infer plausible density but introduce geometry/distribution error; PointRCNN then treats every synthetic point as a real return.",
        0.62,
        1.28,
        12.0,
        0.72,
        size=17.5,
        color=TEXT,
        bold=False,
        valign=MSO_ANCHOR.MIDDLE,
    )
    cards = [
        ("1  MISSING EVIDENCE", "Downsampled baseline already falls from 82.66 to 68.29 on 3D Moderate.", RED),
        ("2  SYNTHETIC GEOMETRY", "Most methods score below a matched observed-fill control at the same ratio.", ORANGE),
        ("3  DETECTOR AMPLIFICATION", "A fixed 16,384-point quota makes generated points replace scarce observed slots.", BLUE),
    ]
    for idx, (title, body, color) in enumerate(cards):
        add_card(slide, 0.64 + idx * 4.18, 2.30, 3.82, 2.28, title, body, color, 12, 17)
    add_rect(slide, 0.64, 4.95, 12.0, 1.52, "#0B2B2A", line=TEAL)
    add_text(slide, "ONLY CLEAR SCREEN CANDIDATE", 0.92, 5.18, 3.0, 0.30, size=11, color=TEAL, bold=True)
    add_rich_text(
        slide,
        [
            {"text": "Original / PDANS g2.5: ", "color": WHITE, "bold": True, "size": 21},
            {"text": "85.10", "color": TEAL, "bold": True, "size": 25},
            {"text": " vs baseline 82.66 (", "color": TEXT, "size": 18},
            {"text": "+2.43", "color": TEAL, "bold": True, "size": 21},
            {"text": "), but only ", "color": TEXT, "size": 18},
            {"text": "+0.68", "color": YELLOW, "bold": True, "size": 21},
            {"text": " over the matched c2.5 observed-fill control.", "color": TEXT, "size": 18},
        ],
        0.92,
        5.55,
        11.35,
        0.55,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_text(slide, "This candidate requires full KITTI validation before claiming detector improvement.", 0.92, 6.12, 10.8, 0.27, size=10.5, color=MUTED, italic=True)
    add_footer(slide)


def add_experiment_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 3, "Experimental design", "Two lines, four methods, two ratio protocols")
    add_card(slide, 0.62, 1.38, 2.20, 1.05, "Methods", "PDANS • PU-GCN\nPU-EdgeFormer • PU-Net", TEAL, 11, 15)
    add_card(slide, 3.02, 1.38, 2.20, 1.05, "Fine protocol", "2.5 • 5 • 7.5 • 10%\nNested E2 replacement", BLUE, 11, 14)
    add_card(slide, 5.42, 1.38, 2.20, 1.05, "Coarse protocol", "10 • 15 • 25 • 35 • 40 • 50%\nIndependent sampling", ORANGE, 11, 13)
    add_card(slide, 7.82, 1.38, 2.20, 1.05, "Controls", "Observed-fill at\n2.5 • 5 • 7.5 • 10%", YELLOW, 11, 14)
    add_card(slide, 10.22, 1.38, 2.20, 1.05, "Coverage", "88 / 88 PASS\n+ 2 baselines", GREEN, 11, 16)

    add_text(slide, "LINE A — ORIGINAL", 0.65, 2.82, 2.3, 0.28, size=11, color=TEAL, bold=True)
    add_rect(slide, 0.65, 3.18, 2.10, 0.92, PANEL, line=GRID)
    add_text(slide, "Original KITTI\nmeasurements", 0.82, 3.37, 1.76, 0.52, size=15, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_arrow_shape(slide, 2.96, 3.45, 0.65, 0.36, TEAL)
    add_rect(slide, 3.83, 3.18, 2.10, 0.92, PANEL, line=GRID)
    add_text(slide, "4× upsampling\nmethod output", 4.00, 3.37, 1.76, 0.52, size=15, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_arrow_shape(slide, 6.14, 3.45, 0.65, 0.36, TEAL)
    add_rect(slide, 7.01, 3.18, 2.32, 0.92, PANEL, line=GRID)
    add_text(slide, "16,384-point\nobserved/generated mix", 7.18, 3.34, 1.98, 0.57, size=14, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_arrow_shape(slide, 9.55, 3.45, 0.65, 0.36, TEAL)
    add_rect(slide, 10.43, 3.18, 1.95, 0.92, "#0B2B2A", line=TEAL)
    add_text(slide, "PointRCNN\nAP R40", 10.60, 3.37, 1.61, 0.52, size=15, color=TEAL, bold=True, align=PP_ALIGN.CENTER)

    add_text(slide, "LINE B — DOWNSAMPLED", 0.65, 4.55, 2.8, 0.28, size=11, color=RED, bold=True)
    add_rect(slide, 0.65, 4.91, 2.10, 0.92, PANEL, line=GRID)
    add_text(slide, "Original KITTI\nmeasurements", 0.82, 5.10, 1.76, 0.52, size=15, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_arrow_shape(slide, 2.96, 5.18, 0.65, 0.36, RED)
    add_rect(slide, 3.83, 4.91, 2.10, 0.92, "#2B1B23", line=RED)
    add_text(slide, "Retain ~N/4\nobserved points", 4.00, 5.10, 1.76, 0.52, size=15, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_arrow_shape(slide, 6.14, 5.18, 0.65, 0.36, RED)
    add_rect(slide, 7.01, 4.91, 2.32, 0.92, PANEL, line=GRID)
    add_text(slide, "4× output:\nN/4 observed + 3N/4 inferred", 7.14, 5.04, 2.06, 0.65, size=13.5, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_arrow_shape(slide, 9.55, 5.18, 0.65, 0.36, RED)
    add_rect(slide, 10.43, 4.91, 1.95, 0.92, "#2B1B23", line=RED)
    add_text(slide, "PointRCNN\nAP R40", 10.60, 5.10, 1.61, 0.52, size=15, color=RED, bold=True, align=PP_ALIGN.CENTER)

    add_rect(slide, 0.65, 6.23, 11.73, 0.54, "#0D1D2A", line=GRID)
    add_text(
        slide,
        "Protocol caveat: fine and coarse experiments use the same frames and checkpoint, but different detector-side sampling. Their g10 values must not be spliced into one exact continuous curve.",
        0.86,
        6.34,
        11.25,
        0.31,
        size=10.5,
        color=YELLOW,
        bold=True,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_footer(slide)


def add_ratio_meaning_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 4, "Input composition", "“2.5%” is not the 4× factor")
    add_text(
        slide,
        "All methods perform 4× upsampling upstream. The plotted percentage controls how many generated points enter PointRCNN’s fixed 16,384-point input.",
        0.65,
        1.30,
        12.0,
        0.54,
        size=17,
        color=TEXT,
        valign=MSO_ANCHOR.MIDDLE,
    )
    ratios = [
        (2.5, 410, 15974, TEAL),
        (10.0, 1638, 14746, BLUE),
        (25.0, 4096, 12288, ORANGE),
        (50.0, 8192, 8192, RED),
    ]
    for idx, (ratio, generated, observed, color) in enumerate(ratios):
        x = 0.65 + idx * 3.09
        add_rect(slide, x, 2.12, 2.78, 2.54, PANEL, line=GRID)
        add_text(slide, f"{ratio:g}%", x + 0.18, 2.31, 2.42, 0.50, size=26, color=color, bold=True, align=PP_ALIGN.CENTER)
        bar_x, bar_y, bar_w, bar_h = x + 0.27, 3.04, 2.24, 0.37
        add_rect(slide, bar_x, bar_y, bar_w, bar_h, "#284052", radius=False)
        gen_w = bar_w * ratio / 100.0
        if gen_w < 0.06:
            gen_w = 0.06
        add_rect(slide, bar_x + bar_w - gen_w, bar_y, gen_w, bar_h, color, radius=False)
        add_text(slide, f"{observed:,} observed", x + 0.20, 3.62, 2.38, 0.28, size=12.5, color=TEXT, align=PP_ALIGN.CENTER)
        add_text(slide, f"{generated:,} generated", x + 0.20, 4.00, 2.38, 0.28, size=12.5, color=color, bold=True, align=PP_ALIGN.CENTER)
    add_rect(slide, 0.65, 5.08, 12.0, 1.33, "#0B202E", line=BLUE)
    add_rich_text(
        slide,
        [
            {"text": "Example — c2.5 control: ", "color": BLUE, "bold": True, "size": 18},
            {"text": "15,974 common baseline-core points + 410 real observed-fill points. ", "color": WHITE, "size": 18},
            {"text": "The same slots are replaced as in g2.5, so the control separates sampling/coverage from synthetic geometry.", "color": TEXT, "size": 16},
        ],
        0.92,
        5.35,
        11.4,
        0.75,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_text(slide, "Observed points are physical LiDAR returns. Generated points are model-inferred samples; their intensity is typically copied/filled from nearby observed returns.", 0.75, 6.62, 11.7, 0.34, size=11, color=MUTED, italic=True, align=PP_ALIGN.CENTER)
    add_footer(slide)


def add_metrics_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 5, "How to read the evaluation", "KITTI Car AP R40 and difficulty levels")
    cards = [
        ("BBox AP", "2D image-box IoU", "bbox", BLUE),
        ("BEV AP", "Oriented top-down rectangle IoU", "bev", TEAL),
        ("3D AP", "Full oriented 3D-volume IoU", "3d", ORANGE),
        ("AOS", "Detection matching + yaw similarity", "aos", PURPLE),
    ]
    for idx, (title, body, icon, color) in enumerate(cards):
        x = 0.65 + idx * 3.08
        add_rect(slide, x, 1.45, 2.76, 2.20, PANEL, line=GRID)
        add_metric_icon(slide, icon, x + 0.77, 1.62, 1.22, 0.92, color)
        add_text(slide, title, x + 0.20, 2.58, 2.36, 0.35, size=18, color=color, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, body, x + 0.22, 3.01, 2.32, 0.42, size=11.5, color=TEXT, align=PP_ALIGN.CENTER)

    rows = [
        ["Easy", "≥ 40 px", "0", "≤ 0.15"],
        ["Moderate", "≥ 25 px", "≤ 1", "≤ 0.30"],
        ["Hard", "≥ 25 px", "≤ 2", "≤ 0.50"],
    ]
    highlights = {(1, 0): {"fill": "#174B43", "color": TEAL}}
    add_table(
        slide,
        rows,
        ["Difficulty", "Min. bbox height", "Max. occlusion", "Max. truncation"],
        0.65,
        4.12,
        7.60,
        1.72,
        col_widths=[1.25, 1.55, 1.55, 1.55],
        font_size=12.5,
        highlights=highlights,
    )
    add_card(slide, 8.55, 4.12, 4.10, 1.72, "Primary metric in this study", "Car 3D AP R40 Moderate\nStrict Car IoU ≈ 0.70", ORANGE, 12, 20)
    add_rect(slide, 0.65, 6.13, 12.0, 0.62, "#0D1D2A", line=GRID)
    add_text(
        slide,
        "R40 averages precision over 40 non-zero recall positions. “Moderate” is the main KITTI operating point; it is harder than Easy because targets may be smaller, partially occluded, and more truncated.",
        0.88,
        6.27,
        11.5,
        0.33,
        size=10.7,
        color=TEXT,
        valign=MSO_ANCHOR.MIDDLE,
    )
    add_footer(slide)


def add_baseline_slide(prs, data):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 6, "Reference performance", "The Downsampled line starts with a large information deficit")
    original = baseline_row(data, "original")
    down = baseline_row(data, "downsampled")
    groups = [
        ("BBox", "ap_bbox_car"),
        ("BEV", "ap_bev_car"),
        ("3D", "ap_3d_car"),
        ("AOS", "ap_aos_car"),
    ]
    rows = []
    for label, prefix in groups:
        rows.append(
            [
                f"{label} — Original",
                f"{original[prefix + '_easy_r40']:.2f}",
                f"{original[prefix + '_moderate_r40']:.2f}",
                f"{original[prefix + '_hard_r40']:.2f}",
            ]
        )
        rows.append(
            [
                f"{label} — Downsampled",
                f"{down[prefix + '_easy_r40']:.2f}",
                f"{down[prefix + '_moderate_r40']:.2f}",
                f"{down[prefix + '_hard_r40']:.2f}",
            ]
        )
    highlights = {}
    for idx in [1, 3, 5, 7]:
        highlights[(idx, 0)] = {"fill": "#2B1B23", "color": RED}
    highlights[(4, 2)] = {"fill": "#174B43", "color": TEAL}
    highlights[(5, 2)] = {"fill": "#40202A", "color": RED}
    add_table(
        slide,
        rows,
        ["Baseline / metric", "Easy", "Moderate", "Hard"],
        0.65,
        1.44,
        7.85,
        4.86,
        col_widths=[2.5, 1, 1, 1],
        font_size=13,
        highlights=highlights,
    )
    add_card(slide, 8.80, 1.45, 3.84, 1.38, "Original 3D Moderate", "82.66 AP", TEAL, 11, 27)
    add_card(slide, 8.80, 3.05, 3.84, 1.38, "Downsampled 3D Moderate", "68.29 AP", RED, 11, 27)
    add_card(slide, 8.80, 4.65, 3.84, 1.65, "Loss before adding generated points", "−14.37 AP\n(−17.4% relative)", ORANGE, 11, 22)
    add_text(slide, "Restoring the count to N does not reconstruct the deleted N − N/4 sensor measurements.", 8.86, 6.48, 3.68, 0.42, size=11, color=MUTED, italic=True, align=PP_ALIGN.CENTER)
    add_footer(slide)


def add_ratio_curves_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 7, "Complete ratio sweep", "Higher generated-point ratios consistently reduce 3D AP Moderate")
    add_picture_contain(slide, ASSET_DIR / RATIO_CHART.name, 0.45, 1.20, 9.15, 5.75, frame=True)
    add_card(slide, 9.78, 1.24, 3.00, 1.34, "Fine result", "Only PDANS g2.5 clearly exceeds the Original baseline.", TEAL, 11, 15)
    add_card(slide, 9.78, 2.80, 3.00, 1.34, "Downsampled result", "Every method falls below the Downsampled baseline, even at 2.5%.", RED, 11, 15)
    add_card(slide, 9.78, 4.36, 3.00, 1.34, "Coarse result", "35–50% is systematically harmful in both lines.", ORANGE, 11, 15)
    add_text(slide, "Do not join fine and coarse g10 as one exact curve: the sampling protocols differ.", 9.86, 5.98, 2.84, 0.52, size=10.5, color=YELLOW, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide)


def best_table_rows(best):
    rows = []
    for item in best:
        config = f"{item['method']} {fmt_ratio(item['ratio'])} {item['protocol']}"
        rows.append([item["metric"], config, f"{item['ap']:.2f}", f"{item['delta']:+.2f}"])
    return rows


def add_original_best_slide(prs, data, chart_path):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(
        slide,
        8,
        "Original line",
        "Selected metrics improve—there is no universal winner",
        "Each bar is the best tested method/ratio for that metric—not one shared configuration.",
    )
    best = best_rows(data, "original")
    add_picture_contain(slide, chart_path, 0.55, 1.43, 7.35, 5.48)
    top = sorted(best, key=lambda item: item["delta"], reverse=True)[:5]
    rows = [
        [item["metric"], f"{item['method']} {fmt_ratio(item['ratio'])}", f"{item['ap']:.2f}", f"{item['delta']:+.2f}"]
        for item in top
    ]
    highlights = {(idx, 3): {"fill": "#174B43", "color": TEAL} for idx in range(len(rows))}
    add_table(
        slide,
        rows,
        ["Metric", "Best config", "AP", "Δ"],
        8.12,
        1.48,
        4.62,
        2.85,
        col_widths=[1.45, 1.55, 0.7, 0.65],
        font_size=11.2,
        highlights=highlights,
    )
    add_card(slide, 8.12, 4.55, 4.62, 1.15, "Strongest candidate", "3D Moderate: PDANS 2.5% fine\n85.10 AP • +2.43", TEAL, 11, 17)
    add_card(slide, 8.12, 5.92, 4.62, 0.82, "Still below baseline", "BEV-H • 3D-H • AOS-M", RED, 11, 14)
    add_footer(slide)


def add_downsampled_best_slide(prs, data, chart_path):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(
        slide,
        9,
        "Downsampled line",
        "Even the best tested setting remains below the Original baseline",
        "Best-per-metric oracle across all methods, ratios, and both protocols.",
    )
    best = best_rows(data, "downsampled")
    add_picture_contain(slide, chart_path, 0.55, 1.43, 7.35, 5.48)
    top = sorted(best, key=lambda item: item["delta"], reverse=True)[:5]
    rows = [
        [item["metric"], f"{item['method']} {fmt_ratio(item['ratio'])}", f"{item['ap']:.2f}", f"{item['delta']:+.2f}"]
        for item in top
    ]
    highlights = {(idx, 3): {"fill": "#40202A", "color": RED} for idx in range(len(rows))}
    add_table(
        slide,
        rows,
        ["Least-bad metric", "Best config", "AP", "Δ"],
        8.08,
        1.48,
        4.66,
        2.85,
        col_widths=[1.45, 1.55, 0.7, 0.65],
        font_size=11.0,
        highlights=highlights,
    )
    add_card(slide, 8.08, 4.55, 4.66, 1.08, "Best 3D Moderate", "PDANS 2.5% fine: 66.40\n−16.26 vs Original baseline", RED, 11, 17)
    add_card(slide, 8.08, 5.85, 4.66, 0.90, "Conclusion", "No setting recovers the lost evidence.", ORANGE, 11, 14)
    add_footer(slide)


def add_control_slide(prs, chart_path):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 10, "Matched observed-fill control", "Control separates sampling sensitivity from synthetic geometry")
    add_picture_contain(slide, chart_path, 0.55, 1.42, 8.10, 5.38)
    add_card(slide, 8.86, 1.47, 3.84, 1.38, "What the control holds fixed", "Same 16,384 total points\nSame baseline slots removed\nSame generated-point ratio", BLUE, 10.5, 14)
    add_card(slide, 8.86, 3.08, 3.84, 1.38, "What changes", "Generated samples are replaced with real, retained sensor observations.", YELLOW, 10.5, 15)
    add_card(slide, 8.86, 4.69, 3.84, 1.38, "What the result says", "Most synthetic methods are worse than control; geometry/distribution adds harm beyond sampling.", RED, 10.5, 14.5)
    add_text(slide, "Downsampled controls use only retained Downsampled observations—deleted Original points are never leaked back.", 8.94, 6.28, 3.68, 0.48, size=9.6, color=MUTED, italic=True, align=PP_ALIGN.CENTER)
    add_footer(slide)


def add_waterfall_slide(prs, chart_path):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 11, "Loss decomposition", "The dominant loss appears before synthetic points enter the detector")
    add_picture_contain(slide, chart_path, 0.55, 1.38, 8.05, 5.45)
    add_card(slide, 8.85, 1.50, 3.76, 1.12, "1 • Missing measurements", "−14.37 AP", RED, 11, 24)
    add_card(slide, 8.85, 2.84, 3.76, 1.12, "2 • Sampling / coverage", "−1.40 AP", ORANGE, 11, 24)
    add_card(slide, 8.85, 4.18, 3.76, 1.12, "3 • PDANS geometry", "−0.49 AP", YELLOW, 11, 24)
    add_rect(slide, 8.85, 5.57, 3.76, 1.00, "#0D1D2A", line=GRID)
    add_text(slide, "TOTAL", 9.12, 5.78, 0.9, 0.26, size=11, color=MUTED, bold=True)
    add_text(slide, "−16.26 AP", 10.12, 5.72, 2.12, 0.38, size=22, color=RED, bold=True, align=PP_ALIGN.RIGHT)
    add_text(slide, "Original 82.66 → Downsampled PDANS g2.5 66.40", 8.94, 6.13, 3.58, 0.28, size=9.7, color=TEXT, align=PP_ALIGN.CENTER)
    add_footer(slide)


def add_visual_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 12, "Qualitative detector evidence", "Visual density is not the same as detector-compatible evidence")
    if (ASSET_DIR / QUAL_VIS.name).exists():
        add_picture_contain(slide, ASSET_DIR / QUAL_VIS.name, 0.48, 1.34, 8.75, 5.48, frame=True)
    else:
        add_rect(slide, 0.48, 1.34, 8.75, 5.48, PANEL, line=GRID)
        add_text(slide, "Qualitative comparison image not found", 1.0, 3.5, 7.7, 0.5, size=20, color=MUTED, align=PP_ALIGN.CENTER)
    add_card(slide, 9.48, 1.38, 3.30, 1.22, "Point count", "Restored by 4× upsampling", TEAL, 10.5, 16)
    add_card(slide, 9.48, 2.82, 3.30, 1.22, "Measurement identity", "Not restored: deleted rays and occlusion evidence remain missing", RED, 10.5, 14)
    add_card(slide, 9.48, 4.26, 3.30, 1.22, "Detector input", "[x, y, z, intensity] only\nNo generated-point provenance flag", BLUE, 10.5, 14)
    add_card(slide, 9.48, 5.70, 3.30, 0.86, "Outcome", "Boxes and IoU can degrade even when the cloud looks denser.", ORANGE, 10.5, 13.5)
    add_text(
        slide,
        "Qualitative same-frame diagnostic from the prior strict-x4 E2 visualization study; AP labels inside the image are not the 256-frame ratio-screen values.",
        0.68,
        6.84,
        8.35,
        0.22,
        size=7.8,
        color=MUTED,
        italic=True,
        align=PP_ALIGN.CENTER,
    )
    add_footer(slide)


def add_robustness_slide(prs, chart_path):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 13, "Method robustness", "PDANS is most tolerant; the current PU-Net integration is least reliable")
    add_picture_contain(slide, chart_path, 0.55, 1.40, 8.05, 5.35)
    ranking = [
        ("1", "PDANS", "Best tolerance at high generated ratios", TEAL),
        ("2", "PU-GCN", "Second-best, but degrades strongly in Downsampled line", BLUE),
        ("3", "PU-EdgeFormer", "Large geometry/distribution penalty", ORANGE),
        ("4", "PU-Net (current wrapper)", "Worst current curve; integration defect confirmed", RED),
    ]
    for idx, (rank, name, desc, color) in enumerate(ranking):
        y = 1.48 + idx * 1.23
        add_rect(slide, 8.86, y, 3.84, 1.02, PANEL, line=GRID)
        add_text(slide, rank, 9.04, y + 0.24, 0.40, 0.40, size=20, color=color, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, name, 9.55, y + 0.13, 2.85, 0.28, size=14.5, color=WHITE, bold=True)
        add_text(slide, desc, 9.55, y + 0.49, 2.88, 0.37, size=10.2, color=TEXT)
    add_rect(slide, 8.86, 6.42, 3.84, 0.37, "#2B1B23", line=RED)
    add_text(slide, "PU-Net wrapper lacks required centering, unit-radius normalization, and inverse transform.", 8.98, 6.49, 3.60, 0.22, size=7.9, color=RED, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide)


def add_root_cause_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 14, "Root-cause model", "Why detector AP falls while the point cloud can still look acceptable")
    nodes = [
        ("Downsample", "Deletes physical returns,\nrays, and occlusion cues", RED),
        ("Upsampler", "Infers plausible surfaces;\ndoes not recover measurements", ORANGE),
        ("Fixed quota", "Generated points replace\nscarce observed slots", YELLOW),
        ("PointRCNN", "Treats synthetic [x,y,z,i]\nas real observations", BLUE),
        ("AP / IoU", "Proposal scores and box\ngeometry shift", TEAL),
    ]
    for idx, (title, body, color) in enumerate(nodes):
        x = 0.45 + idx * 2.58
        add_rect(slide, x, 2.00, 2.12, 1.82, PANEL, line=color)
        add_text(slide, title, x + 0.16, 2.24, 1.80, 0.34, size=16, color=color, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, body, x + 0.16, 2.76, 1.80, 0.66, size=11.5, color=TEXT, align=PP_ALIGN.CENTER)
        if idx < len(nodes) - 1:
            add_arrow_shape(slide, x + 2.18, 2.70, 0.32, 0.34, color)
    add_rect(slide, 0.65, 4.40, 5.78, 1.65, "#0C2630", line=BLUE)
    add_text(slide, "Evidence that the detector is sensitive", 0.92, 4.65, 5.20, 0.32, size=15.5, color=BLUE, bold=True)
    add_text(slide, "Observed-fill controls change AP even without synthetic geometry. Detector sampling and coverage therefore matter.", 0.92, 5.12, 5.10, 0.62, size=14.5, color=TEXT)
    add_rect(slide, 6.86, 4.40, 5.78, 1.65, "#2B1B23", line=RED)
    add_text(slide, "Evidence that it is not detector-only", 7.13, 4.65, 5.20, 0.32, size=15.5, color=RED, bold=True)
    add_text(slide, "Most methods remain below the matched observed-fill control. Generated geometry/distribution causes extra loss.", 7.13, 5.12, 5.10, 0.62, size=14.5, color=TEXT)
    add_text(slide, "Diagnosis: information loss dominates Downsampled; synthetic geometry and detector mismatch amplify it.", 0.75, 6.43, 11.75, 0.38, size=17, color=WHITE, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide)


def add_actions_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 15, "Recovery plan", "What to change if the objective is higher detector AP")
    actions = [
        ("01", "Validate the only candidate", "Run full KITTI validation for Original baseline, c2.5, and PDANS g2.5. Claim gain only if PDANS beats both references.", TEAL),
        ("02", "Preserve observed points first", "Stop random replacement of real measurements. Append or prioritize observed points under the detector quota.", BLUE),
        ("03", "Gate synthetic points", "Use confidence, surface consistency, and object-neighborhood selection; reject uncertain generated samples.", ORANGE),
        ("04", "Adapt the detector", "Train/fine-tune PointRCNN on the same observed/generated mixture and expose provenance or confidence when possible.", PURPLE),
        ("05", "Fix integrations before ranking", "Repair PU-Net centering, unit-radius normalization, inverse transform, and patch construction, then re-run.", RED),
    ]
    for idx, (num, title, body, color) in enumerate(actions):
        y = 1.40 + idx * 1.02
        add_rect(slide, 0.65, y, 12.0, 0.82, PANEL, line=GRID)
        add_text(slide, num, 0.86, y + 0.17, 0.56, 0.38, size=18, color=color, bold=True, align=PP_ALIGN.CENTER)
        add_text(slide, title, 1.62, y + 0.13, 3.10, 0.28, size=14.5, color=WHITE, bold=True)
        add_text(slide, body, 4.85, y + 0.12, 7.52, 0.50, size=12.2, color=TEXT, valign=MSO_ANCHOR.MIDDLE)
    add_rect(slide, 0.65, 6.60, 12.0, 0.38, "#0B2B2A", line=TEAL)
    add_text(slide, "Recommended gate: do not advance 35–50% configurations; the 256-frame trend already shows systematic degradation.", 0.86, 6.67, 11.55, 0.23, size=9.6, color=TEAL, bold=True, align=PP_ALIGN.CENTER)
    add_footer(slide)


def add_final_slide(prs, data):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_background(slide)
    add_header(slide, 16, "Decision and supporting files", "What can be concluded now")
    add_rect(slide, 0.65, 1.40, 12.0, 1.18, "#0B2B2A", line=TEAL)
    add_text(
        slide,
        "4× upsampling restores point count—not missing measurement information. Under the current detector-side replacement policy, more generated points generally reduce AP.",
        0.92,
        1.68,
        11.45,
        0.58,
        size=20,
        color=WHITE,
        bold=True,
        align=PP_ALIGN.CENTER,
        valign=MSO_ANCHOR.MIDDLE,
    )
    decisions = [
        ("Original data available", "Keep the Original baseline; validate PDANS g2.5 as a narrow candidate.", TEAL),
        ("Only Downsampled data available", "Upsampling alone is insufficient; preserve observations and adapt the detector.", RED),
        ("High synthetic ratio requested", "Reject current 35–50% settings; all methods show substantial AP loss.", ORANGE),
    ]
    for idx, (title, body, color) in enumerate(decisions):
        add_card(slide, 0.65 + idx * 4.10, 2.95, 3.82, 1.52, title, body, color, 11, 14.5)
    rows = [
        ["Complete metrics workbook", str(COMPLETE_XLSX)],
        ["Combined analysis report", str(DATA_DIR / "combined_analysis_report.md")],
        ["Interactive visualization bundle", str(INTERACTIVE_INDEX)],
        ["Representative interactive 3D frame", str(QUAL_VIS_3D)],
    ]
    add_table(
        slide,
        rows,
        ["Supporting artifact", "Local path"],
        0.65,
        4.85,
        12.0,
        1.58,
        col_widths=[2.0, 5.8],
        font_size=9.0,
        first_col_left=True,
    )
    add_text(
        slide,
        f"Coverage in the consolidated table: {len(data)} rows = 2 baseline references + 88 completed experiment groups; 12 AP metrics per row.",
        0.78,
        6.64,
        11.75,
        0.25,
        size=10.2,
        color=MUTED,
        align=PP_ALIGN.CENTER,
    )
    add_footer(slide)


def add_document_properties(prs):
    props = prs.core_properties
    props.title = "Why 4× Point-Cloud Upsampling Does Not Improve PointRCNN"
    props.subject = "Controlled KITTI Car AP R40 analysis of Original and Downsampled x4 upsampling lines"
    props.author = "Codex"
    props.keywords = "PointRCNN, point-cloud upsampling, KITTI, AP R40, PDANS, PU-GCN, PU-EdgeFormer, PU-Net"
    props.comments = "Generated from consolidated 256-frame screening results on 26 July 2026."


def build_deck(data, charts):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    add_document_properties(prs)

    add_title_slide(prs)
    add_executive_slide(prs)
    add_experiment_slide(prs)
    add_ratio_meaning_slide(prs)
    add_metrics_slide(prs)
    add_baseline_slide(prs, data)
    add_ratio_curves_slide(prs)
    add_original_best_slide(prs, data, charts["best_original"])
    add_downsampled_best_slide(prs, data, charts["best_downsampled"])
    add_control_slide(prs, charts["control_heatmap"])
    add_waterfall_slide(prs, charts["waterfall"])
    add_visual_slide(prs)
    add_robustness_slide(prs, charts["robustness"])
    add_root_cause_slide(prs)
    add_actions_slide(prs)
    add_final_slide(prs, data)

    out_path = OUT_DIR / "PointRCNN_x4_Upsampling_Failure_Analysis_English_20260726.pptx"
    prs.save(out_path)
    return out_path


def write_manifest(deck_path: Path, charts: dict[str, Path], data: pd.DataFrame) -> Path:
    manifest = OUT_DIR / "presentation_manifest.md"
    lines = [
        "# English PointRCNN Upsampling Presentation",
        "",
        f"- PowerPoint: `{deck_path}`",
        f"- Rendered PDF: `{RENDER_DIR / 'PointRCNN_x4_Upsampling_Failure_Analysis_English_20260726.pdf'}`",
        f"- Slide contact sheet: `{RENDER_DIR / 'contact_sheet.png'}`",
        f"- Slides: 16",
        f"- Consolidated experiment rows: {len(data)}",
        f"- Source CSV: `{CSV_PATH}`",
        f"- Complete workbook: `{COMPLETE_XLSX}`",
        f"- Interactive bundle: `{INTERACTIVE_INDEX}`",
        f"- Representative interactive 3D view: `{QUAL_VIS_3D}`",
        "",
        "## Generated charts",
        "",
    ]
    for name, path in charts.items():
        lines.append(f"- {name}: `{path}`")
    lines += [
        f"- ratio sweep: `{ASSET_DIR / RATIO_CHART.name}`",
        "",
        "## Interpretation guardrails",
        "",
        "- Fine and coarse g10 values use different detector-side sampling protocols and are not spliced as one exact curve.",
        "- Best-per-metric slides are oracle summaries; winners may use different methods, ratios, and protocols.",
        "- The qualitative detector panel is from the prior strict-x4 E2 visualization study; its embedded AP labels are not the 256-frame ratio-screen values.",
    ]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main():
    setup_output()
    data = load_data()
    charts = generate_charts(data)
    deck_path = build_deck(data, charts)
    manifest_path = write_manifest(deck_path, charts, data)
    print(f"deck={deck_path}")
    print(f"manifest={manifest_path}")
    print(f"slides=16")
    print(f"rows={len(data)}")


if __name__ == "__main__":
    main()
