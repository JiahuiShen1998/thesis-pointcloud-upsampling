#!/usr/bin/env python3
"""Build a thesis-ready PowerPoint summarizing ModelNet40 + PointNet++ results."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

PROJECT = Path(__file__).resolve().parents[1]
FIG = PROJECT / "figures" / "modelnet40"
PKG = FIG / "final_visualization_package"
OUT = PROJECT / "presentations" / "ModelNet40_PointNet2_Results_Presentation.pptx"

# Prefer EdgeFormer-complete figures when available
F_LINEA_ABS = FIG / "pointnet2_final_with_pu_edgeformer" / "accuracy_absolute" / "lineA_best_overall_accuracy_absolute_with_pu_edgeformer.png"
F_LINEB_ABS = FIG / "pointnet2_final_with_pu_edgeformer" / "accuracy_absolute" / "lineB_best_overall_accuracy_absolute_with_pu_edgeformer.png"
F_LINEA_DELTA = FIG / "pointnet2_final_with_pu_edgeformer" / "accuracy_delta_methods_only" / "lineA_delta_best_overall_vs_original_baseline_methods_only_with_pu_edgeformer.png"
F_LINEB_DELTA = FIG / "pointnet2_final_with_pu_edgeformer" / "accuracy_delta_methods_only" / "lineB_delta_best_overall_vs_downsampled_baseline_methods_only_with_pu_edgeformer.png"
F_LINEB_GAP = FIG / "pointnet2_final_with_pu_edgeformer" / "accuracy_delta_methods_only" / "lineB_gap_best_overall_vs_original_baseline_methods_only_with_pu_edgeformer.png"

F_CD = FIG / "geometry_delta_methods_only_with_pu_edgeformer" / "lineB_delta_cd_vs_original_baseline_methods_only_with_pu_edgeformer.png"
F_HD = FIG / "geometry_delta_methods_only_with_pu_edgeformer" / "lineB_delta_hd_vs_original_baseline_methods_only_with_pu_edgeformer.png"
F_NUC = FIG / "geometry_delta_methods_only_with_pu_edgeformer" / "lineB_delta_nuc_vs_original_baseline_methods_only_with_pu_edgeformer.png"
F_P2F = FIG / "geometry_delta_methods_only_with_pu_edgeformer" / "lineB_delta_exact_p2f_vs_original_baseline_methods_only_with_pu_edgeformer.png"

F_SCATTER_CD = FIG / "pointnet2_final_with_pu_edgeformer" / "final_combined" / "lineB_delta_cd_vs_delta_accuracy_with_pu_edgeformer.png"
F_SCATTER_HD = FIG / "pointnet2_final_with_pu_edgeformer" / "final_combined" / "lineB_delta_hd_vs_delta_accuracy_with_pu_edgeformer.png"
F_SCATTER_NUC = FIG / "pointnet2_final_with_pu_edgeformer" / "final_combined" / "lineB_delta_nuc_vs_delta_accuracy_with_pu_edgeformer.png"
F_SCATTER_P2F = FIG / "pointnet2_final_with_pu_edgeformer" / "final_combined" / "lineB_delta_p2f_vs_delta_accuracy_with_pu_edgeformer.png"

# Fallbacks without EdgeFormer
FALLBACKS = {
    F_LINEA_ABS: PKG / "pointnet2_lineA" / "lineA_best_overall_accuracy_absolute_final.png",
    F_LINEB_ABS: PKG / "pointnet2_lineB" / "lineB_best_overall_accuracy_absolute_final.png",
    F_LINEA_DELTA: PKG / "pointnet2_lineA" / "lineA_delta_best_overall_vs_original_baseline_methods_only_final.png",
    F_LINEB_DELTA: PKG / "pointnet2_lineB" / "lineB_delta_best_overall_vs_downsampled_baseline_methods_only_final.png",
    F_LINEB_GAP: PKG / "pointnet2_lineB" / "lineB_gap_best_overall_vs_original_baseline_methods_only_final.png",
    F_CD: PKG / "geometry_lineB_focus" / "lineB_delta_cd_vs_original_baseline_methods_only_final.png",
    F_HD: PKG / "geometry_lineB_focus" / "lineB_delta_hd_vs_original_baseline_methods_only_final.png",
    F_NUC: PKG / "geometry_lineB_focus" / "lineB_delta_nuc_vs_original_baseline_methods_only_final.png",
    F_P2F: PKG / "geometry_lineB_focus" / "lineB_delta_exact_p2f_vs_original_baseline_methods_only_final.png",
    F_SCATTER_CD: PKG / "combined_geometry_vs_classification" / "lineB_delta_cd_vs_delta_accuracy_final.png",
    F_SCATTER_HD: PKG / "combined_geometry_vs_classification" / "lineB_delta_hd_vs_delta_accuracy_final.png",
    F_SCATTER_NUC: PKG / "combined_geometry_vs_classification" / "lineB_delta_nuc_vs_delta_accuracy_final.png",
    F_SCATTER_P2F: PKG / "combined_geometry_vs_classification" / "lineB_delta_p2f_vs_delta_accuracy_final.png",
}

PC_LINEB = [
    PKG / "pointcloud_static" / "lineB" / "lineB_airplane_airplane_0627_comparison.png",
    PKG / "pointcloud_static" / "lineB" / "lineB_chair_chair_0890_comparison.png",
    PKG / "pointcloud_static" / "lineB" / "lineB_table_table_0393_comparison.png",
    PKG / "pointcloud_static" / "lineB" / "lineB_car_car_0198_comparison.png",
    PKG / "pointcloud_static" / "lineB" / "lineB_sofa_sofa_0681_comparison.png",
]
PC_LINEA = [
    PKG / "pointcloud_static" / "lineA" / "lineA_airplane_airplane_0627_comparison.png",
    PKG / "pointcloud_static" / "lineA" / "lineA_chair_chair_0890_comparison.png",
    PKG / "pointcloud_static" / "lineA" / "lineA_table_table_0393_comparison.png",
]

# Older static fallbacks
PC_LINEB_FALLBACK = [
    FIG / "pointcloud_examples" / "lineB_airplane_airplane_0627_comparison.png",
    FIG / "pointcloud_examples" / "lineB_chair_chair_0890_comparison.png",
    FIG / "pointcloud_examples" / "lineB_table_table_0393_comparison.png",
    FIG / "pointcloud_examples" / "lineB_car_car_0198_comparison.png",
    FIG / "pointcloud_examples" / "lineB_sofa_sofa_0681_comparison.png",
]
PC_LINEA_FALLBACK = [
    FIG / "pointcloud_examples" / "lineA_airplane_airplane_0627_comparison.png",
    FIG / "pointcloud_examples" / "lineA_chair_chair_0890_comparison.png",
    FIG / "pointcloud_examples" / "lineA_table_table_0393_comparison.png",
]

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

NAVY = RGBColor(0x1A, 0x3A, 0x5C)
ACCENT = RGBColor(0xC4, 0x4E, 0x52)
DARK = RGBColor(0x22, 0x22, 0x22)
GRAY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF5, 0xF7, 0xFA)


def resolve(path: Path) -> Path | None:
    if path.exists():
        return path
    fb = FALLBACKS.get(path)
    if fb and fb.exists():
        return fb
    return path if path.exists() else None


def set_run(run, text: str, size: int = 18, bold: bool = False, color=DARK):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"


def add_title_simple(slide, text: str, top=0.25, size=28):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(top), Inches(12.3), Inches(0.55))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    set_run(run, text, size, True, NAVY)
    return box


def add_bullets(slide, bullets: list[str], left=0.5, top=1.0, width=12.0, height=5.5, size=18):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = 0
        run = p.add_run()
        set_run(run, "•  " + item, size, False, DARK)
        p.space_after = Pt(8)
    return box


def add_footer(slide, page: str):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(7.1), Inches(12.3), Inches(0.3))
    tf = box.text_frame
    p = tf.paragraphs[0]
    run = p.add_run()
    set_run(run, f"ModelNet40 · PointNet++ Upsampling · {page}", 10, False, GRAY)


def add_picture_fit(slide, path: Path | None, left, top, width, height):
    if path is None or not Path(path).exists():
        box = slide.shapes.add_textbox(left, top, width, height)
        tf = box.text_frame
        p = tf.paragraphs[0]
        run = p.add_run()
        set_run(run, f"[Missing figure]\n{path}", 12, False, ACCENT)
        return
    slide.shapes.add_picture(str(path), left, top, width=width, height=height)


def add_table(slide, rows: list[list[str]], left, top, width, height, col_widths=None):
    n_rows, n_cols = len(rows), len(rows[0])
    table_shape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = table_shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w
    for r in range(n_rows):
        for c in range(n_cols):
            cell = table.cell(r, c)
            cell.text = rows[r][c]
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.CENTER if c > 0 else PP_ALIGN.LEFT
                for run in p.runs:
                    run.font.size = Pt(11 if r > 0 else 12)
                    run.font.bold = r == 0
                    run.font.name = "Calibri"
                    run.font.color.rgb = WHITE if r == 0 else DARK
            if r == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            elif r % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT
    return table_shape


def blank_slide(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])  # blank


def build():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    page = 0

    def next_page():
        nonlocal page
        page += 1
        return str(page)

    # ---- 1 Title ----
    s = blank_slide(prs)
    box = s.shapes.add_textbox(Inches(1), Inches(2.0), Inches(11.3), Inches(1.2))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    set_run(run, "Point Cloud Upsampling on ModelNet40", 36, True, NAVY)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    run = p2.add_run()
    set_run(run, "Geometry Metrics · Qualitative Comparison · PointNet++ Classification", 20, False, GRAY)
    box2 = s.shapes.add_textbox(Inches(1), Inches(4.0), Inches(11.3), Inches(1.5))
    tf = box2.text_frame
    lines = [
        "Two-line protocol (Line A: Original + Upsampling · Line B: Downsampled ×4 + Upsampling)",
        "Methods: EAR · PDANS · PU-Net · PU-GCN · PU-EdgeFormer",
        "Downstream: PointNet++ classification (200 epochs, seed=42)",
    ]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        set_run(run, line, 16, False, DARK)
    add_footer(s, next_page())

    # ---- 2 Agenda ----
    s = blank_slide(prs)
    add_title_simple(s, "Agenda")
    add_bullets(
        s,
        [
            "Experimental setup: two-line protocol & methods",
            "Geometry quality metrics (CD, HD, NUC, exact P2F)",
            "Qualitative point cloud comparisons (Line A / Line B)",
            "PointNet++ Line A classification results",
            "PointNet++ Line B classification results",
            "Geometry vs classification: are they aligned?",
            "Key takeaways for the thesis / presentation",
        ],
        top=1.1,
        size=20,
    )
    add_footer(s, next_page())

    # ---- 3 Protocol ----
    s = blank_slide(prs)
    add_title_simple(s, "Experimental Protocol: Two Lines")
    add_bullets(
        s,
        [
            "Dataset: ModelNet40 (40 classes); classifier: PointNet++",
            "Line A — densification: Original 1024 → upsample to 4096 (×4)",
            "Line B — recovery: Downsample ×4 (256) → upsample back to 1024",
            "Upsampling methods: EAR, PDANS, PU-Net, PU-GCN, PU-EdgeFormer",
            "Geometry: CD / HD / NUC / exact P2F vs dense mesh surface reference",
            "Classification deltas in percentage points (pp); baselines are real training branches",
        ],
        top=1.0,
        size=18,
    )
    add_table(
        s,
        [
            ["Line", "Baseline", "Upsampled input", "Question"],
            ["A", "Original 1024", "Original + Up @ 4096", "Does denser input help?"],
            ["B", "Downsampled ×4 @ 256", "Downsampled + Up @ 1024", "Can upsampling recover accuracy?"],
        ],
        Inches(0.8),
        Inches(4.6),
        Inches(11.5),
        Inches(1.6),
        [Inches(1.2), Inches(3.2), Inches(3.5), Inches(3.6)],
    )
    add_footer(s, next_page())

    # ---- 4 Geometry overview ----
    s = blank_slide(prs)
    add_title_simple(s, "Geometry Metrics (Line B vs Original baseline)")
    add_bullets(
        s,
        [
            "Lower is better for CD, HD, NUC, exact P2F",
            "Absolute metrics measured against dense mesh surface reference (GT)",
            "Delta figures: Original baseline = zero reference (not a method bar)",
            "PU-GCN: best CD among Line B upsamplers, but elevated NUC",
            "PU-Net: more balanced HD / NUC; competitive CD",
        ],
        top=1.0,
        height=2.2,
        size=17,
    )
    add_table(
        s,
        [
            ["Method", "CD", "ΔCD", "HD", "ΔHD", "NUC", "ΔNUC", "P2F"],
            ["Original 1024", "0.0495", "0", "0.1197", "0", "1.062", "0", "0.0825"],
            ["Down ×4 256", "0.0771", "+0.028", "0.2143", "+0.095", "2.049", "+0.987", "0.0825"],
            ["EAR 1024", "0.0814", "+0.032", "0.2088", "+0.089", "0.958", "−0.104", "0.0812"],
            ["PDANS 1024", "0.0631", "+0.014", "0.2059", "+0.086", "1.090", "+0.028", "0.0796"],
            ["PU-Net 1024", "0.0571", "+0.008", "0.1377", "+0.018", "1.058", "−0.004", "0.0801"],
            ["PU-GCN 1024", "0.0548", "+0.005", "0.1652", "+0.045", "1.486", "+0.425", "0.0838"],
            ["PU-EdgeFormer", "0.0630", "+0.013", "0.1935", "+0.074", "1.098", "+0.036", "0.0842"],
        ],
        Inches(0.4),
        Inches(3.3),
        Inches(12.5),
        Inches(3.5),
    )
    add_footer(s, next_page())

    # ---- 5 Geometry figures 2x2 ----
    s = blank_slide(prs)
    add_title_simple(s, "Line B Geometry Deltas (methods only)", size=26)
    imgs = [resolve(F_CD), resolve(F_HD), resolve(F_NUC), resolve(F_P2F)]
    positions = [
        (Inches(0.3), Inches(0.9)),
        (Inches(6.8), Inches(0.9)),
        (Inches(0.3), Inches(4.0)),
        (Inches(6.8), Inches(4.0)),
    ]
    for img, (l, t) in zip(imgs, positions):
        add_picture_fit(s, img, l, t, Inches(6.2), Inches(2.9))
    add_footer(s, next_page())

    # ---- 6 Point clouds Line B intro ----
    s = blank_slide(prs)
    add_title_simple(s, "Qualitative Point Clouds — Line B")
    add_bullets(
        s,
        [
            "Panels (left→right): Original 1024 · Downsampled ×4 256 · EAR · PDANS · PU-Net · PU-GCN",
            "Same sample, shared viewpoint / axis limits for fair visual comparison",
            "256-point inputs look sparse by design — no artificial densification in the plot",
            "Interactive 3D HTML viewers also available (rotate / zoom / pan)",
            "Representative classes: airplane, chair, table, car, sofa",
        ],
        top=1.1,
        size=18,
    )
    add_footer(s, next_page())

    # ---- 7-11 Line B point clouds ----
    for i, path in enumerate(PC_LINEB):
        pth = path if path.exists() else PC_LINEB_FALLBACK[i]
        names = ["Airplane", "Chair", "Table", "Car", "Sofa"]
        s = blank_slide(prs)
        add_title_simple(s, f"Line B Point Cloud — {names[i]}", size=24)
        add_picture_fit(s, pth, Inches(0.3), Inches(0.9), Inches(12.7), Inches(5.8))
        add_footer(s, next_page())

    # ---- 12 Line A point clouds intro + samples ----
    s = blank_slide(prs)
    add_title_simple(s, "Qualitative Point Clouds — Line A")
    add_bullets(
        s,
        [
            "Panels: Original 1024 · EAR 4096 · PDANS 4096 · PU-Net 4096 · PU-GCN 4096",
            "Question: does denser (4096) upsampled input preserve / improve structure?",
            "Visually denser clouds ≠ better PointNet++ accuracy (see classification)",
        ],
        top=1.1,
        size=18,
    )
    add_footer(s, next_page())

    for i, path in enumerate(PC_LINEA):
        pth = path if path.exists() else PC_LINEA_FALLBACK[i]
        names = ["Airplane", "Chair", "Table"]
        s = blank_slide(prs)
        add_title_simple(s, f"Line A Point Cloud — {names[i]}", size=24)
        add_picture_fit(s, pth, Inches(0.3), Inches(0.9), Inches(12.7), Inches(5.8))
        add_footer(s, next_page())

    # ---- Classification Line A table ----
    s = blank_slide(prs)
    add_title_simple(s, "PointNet++ Line A — Classification Table")
    add_bullets(
        s,
        [
            "Reference: Original baseline 1024 (best overall = 91.95%)",
            "All Original + Upsampling @ 4096 remain below baseline",
            "4096-point densification does not yield a stable classification gain",
        ],
        top=0.95,
        height=1.5,
        size=16,
    )
    add_table(
        s,
        [
            ["Method", "Pts", "Best Overall", "Δ Best (pp)", "Final Overall", "Δ Final (pp)"],
            ["Original baseline", "1024", "91.95%", "+0.00", "91.31%", "+0.00"],
            ["+ EAR", "4096", "91.48%", "−0.47", "90.85%", "−0.46"],
            ["+ PDANS", "4096", "91.62%", "−0.33", "90.87%", "−0.44"],
            ["+ PU-Net", "4096", "90.95%", "−1.00", "90.55%", "−0.76"],
            ["+ PU-GCN", "4096", "91.63%", "−0.32", "91.00%", "−0.32"],
            ["+ PU-EdgeFormer", "4096", "90.54%", "−1.41", "90.13%", "−1.18"],
        ],
        Inches(0.6),
        Inches(2.6),
        Inches(12.0),
        Inches(4.2),
    )
    add_footer(s, next_page())

    # ---- Line A figures ----
    s = blank_slide(prs)
    add_title_simple(s, "Line A: Absolute Accuracy & Δ vs Original", size=24)
    add_picture_fit(s, resolve(F_LINEA_ABS), Inches(0.3), Inches(0.9), Inches(6.3), Inches(5.6))
    add_picture_fit(s, resolve(F_LINEA_DELTA), Inches(6.7), Inches(0.9), Inches(6.3), Inches(5.6))
    add_footer(s, next_page())

    # ---- Line B table ----
    s = blank_slide(prs)
    add_title_simple(s, "PointNet++ Line B — Classification Table")
    add_bullets(
        s,
        [
            "Primary reference: Downsampled ×4 baseline 256 (best overall = 90.85%)",
            "PU-Net is the only method above downsampled baseline: +0.42 pp → 91.27%",
            "Secondary gap vs Original 1024 still negative for all Line B branches",
        ],
        top=0.95,
        height=1.5,
        size=16,
    )
    add_table(
        s,
        [
            ["Method", "Pts", "Best Overall", "Δ vs Down (pp)", "gap vs Orig (pp)", "Final Overall"],
            ["Down ×4 baseline", "256", "90.85%", "+0.00", "−1.10", "90.46%"],
            ["+ EAR", "1024", "88.81%", "−2.04", "−3.14", "88.02%"],
            ["+ PDANS", "1024", "90.34%", "−0.51", "−1.61", "89.46%"],
            ["+ PU-Net", "1024", "91.27%", "+0.42", "−0.68", "90.65%"],
            ["+ PU-GCN", "1024", "90.06%", "−0.79", "−1.89", "89.63%"],
            ["+ PU-EdgeFormer", "1024", "89.32%", "−1.53", "−2.63", "88.58%"],
        ],
        Inches(0.5),
        Inches(2.6),
        Inches(12.2),
        Inches(4.2),
    )
    add_footer(s, next_page())

    # ---- Line B absolute + delta ----
    s = blank_slide(prs)
    add_title_simple(s, "Line B: Absolute Accuracy & Δ vs Downsampled", size=24)
    add_picture_fit(s, resolve(F_LINEB_ABS), Inches(0.3), Inches(0.9), Inches(6.3), Inches(5.6))
    add_picture_fit(s, resolve(F_LINEB_DELTA), Inches(6.7), Inches(0.9), Inches(6.3), Inches(5.6))
    add_footer(s, next_page())

    # ---- Line B gap vs original ----
    s = blank_slide(prs)
    add_title_simple(s, "Line B: Gap vs Original Baseline (secondary)", size=24)
    add_picture_fit(s, resolve(F_LINEB_GAP), Inches(1.5), Inches(1.0), Inches(10.3), Inches(5.5))
    add_footer(s, next_page())

    # ---- Three comparisons summary ----
    s = blank_slide(prs)
    add_title_simple(s, "Three Core Comparisons")
    add_table(
        s,
        [
            ["Comparison", "What we ask", "Main finding"],
            [
                "1. Geometry quality",
                "Which upsampler is closest to Original?",
                "PU-GCN / PU-Net best CD; PU-Net more balanced HD/NUC",
            ],
            [
                "2. Line A classification",
                "Does 4096 densification help PointNet++?",
                "No — Original 1024 remains best (91.95%)",
            ],
            [
                "3. Line B classification",
                "Can upsampling recover after ×4 downsample?",
                "Only PU-Net slightly above baseline (+0.42 pp)",
            ],
        ],
        Inches(0.5),
        Inches(1.2),
        Inches(12.3),
        Inches(3.5),
        [Inches(2.8), Inches(4.5), Inches(5.0)],
    )
    add_bullets(
        s,
        [
            "Geometry best ≠ classification best (PU-GCN CD vs PU-Net accuracy)",
            "Upsampling is method- and task-dependent — not a universal upgrade",
        ],
        top=5.0,
        height=1.5,
        size=17,
    )
    add_footer(s, next_page())

    # ---- Geometry vs classification ----
    s = blank_slide(prs)
    add_title_simple(s, "Geometry Δ vs Classification Δ (Line B)", size=24)
    add_picture_fit(s, resolve(F_SCATTER_CD), Inches(0.3), Inches(0.9), Inches(6.3), Inches(5.6))
    add_picture_fit(s, resolve(F_SCATTER_HD), Inches(6.7), Inches(0.9), Inches(6.3), Inches(5.6))
    add_footer(s, next_page())

    s = blank_slide(prs)
    add_title_simple(s, "Geometry Δ vs Classification Δ (NUC / P2F)", size=24)
    add_picture_fit(s, resolve(F_SCATTER_NUC), Inches(0.3), Inches(0.9), Inches(6.3), Inches(5.6))
    add_picture_fit(s, resolve(F_SCATTER_P2F), Inches(6.7), Inches(0.9), Inches(6.3), Inches(5.6))
    add_footer(s, next_page())

    # ---- Method ranking ----
    s = blank_slide(prs)
    add_title_simple(s, "Method Ranking Snapshot")
    add_table(
        s,
        [
            ["Rank focus", "Winner", "Evidence"],
            ["Line A best overall", "Original baseline 1024", "91.95% (all ups. negative)"],
            ["Line B best overall", "PU-Net 1024", "91.27% (+0.42 pp vs Down 256)"],
            ["Line B best CD", "PU-GCN", "ΔCD ≈ +0.005 vs Original"],
            ["Line B balanced geometry", "PU-Net", "Competitive CD + better HD/NUC"],
            ["Worst Line B classifier", "EAR", "−2.04 pp vs Downsampled"],
            ["PU-EdgeFormer (cls)", "Below baselines", "Line A −1.41 pp; Line B −1.53 pp"],
        ],
        Inches(0.5),
        Inches(1.2),
        Inches(12.3),
        Inches(5.2),
        [Inches(3.2), Inches(3.5), Inches(5.6)],
    )
    add_footer(s, next_page())

    # ---- Interactive note ----
    s = blank_slide(prs)
    add_title_simple(s, "Interactive 3D Viewers (for live demo)")
    add_bullets(
        s,
        [
            "Folder: figures/modelnet40/pointcloud_examples_interactive_v2/",
            "Line A / Line B multi-panel grids (black + depth coloring)",
            "Dropdown pages: switch method while keeping camera stable",
            "Single-method large views for airplane & chair detail inspection",
            "Open any .html in a browser → rotate / zoom / pan",
            "Static PNGs (this deck) for thesis PDF; HTML for presentation demo",
        ],
        top=1.1,
        size=18,
    )
    add_footer(s, next_page())

    # ---- Conclusions ----
    s = blank_slide(prs)
    add_title_simple(s, "Conclusions")
    add_bullets(
        s,
        [
            "Point cloud upsampling does not universally improve PointNet++ classification.",
            "Line A: denser 4096-point inputs never beat Original 1024 (91.95%).",
            "Line B: only PU-Net slightly exceeds the downsampled baseline (+0.42 pp).",
            "Geometry quality and classification are not fully aligned.",
            "PU-GCN can win on CD while PU-Net wins on downstream accuracy.",
            "Report both absolute tables (with baseline rows) and methods-only delta figures.",
            "Effect depends on the upsampling method and the downstream task.",
        ],
        top=1.1,
        size=18,
    )
    add_footer(s, next_page())

    # ---- Thank you ----
    s = blank_slide(prs)
    box = s.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11.3), Inches(2))
    tf = box.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    set_run(run, "Thank you — Questions?", 40, True, NAVY)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    run = p2.add_run()
    set_run(run, "Reports & figures under thesis_pointcloud/modelnet40_pointnet2_upsampling/", 14, False, GRAY)
    add_footer(s, next_page())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"Wrote {OUT}")
    print(f"Slides: {len(prs.slides)}")
    return OUT


if __name__ == "__main__":
    build()
