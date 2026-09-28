#!/usr/bin/env python3
"""Generate a detailed full-thesis PPTX + PDF for ModelNet40 upsampling + PointNet++.

Loads OA from pointnet2_results/**/metrics.json (authoritative).
Reuses equal-N geometry CSV and figure helpers from the Original-reference deck.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
PRES = PROJECT / "presentations"
RESULTS = PROJECT / "pointnet2_results" / "x4_two_line_final"
FIG_REV = PROJECT / "figures" / "modelnet40" / "original_reference_revised"

OUT_PPTX = PRES / "ModelNet40_PointNet2_Full_Report.pptx"
OUT_PDF = PRES / "ModelNet40_PointNet2_Full_Report.pdf"
OUT_NOTES = PRES / "ModelNet40_PointNet2_Full_Report_Notes.md"
OUT_STATUS = REPORTS / "modelnet40_pointnet2_full_presentation_status.md"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
NAVY = RGBColor(0x1A, 0x3A, 0x5C)
ACCENT = RGBColor(0xC4, 0x4E, 0x52)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
DARK = RGBColor(0x22, 0x22, 0x22)
GRAY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xF5, 0xF7, 0xFA)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def pct(x: float, digits: int = 2) -> float:
    return round(float(x) * 100.0, digits)


def delta_pp(a: float, b: float, digits: int = 2) -> float:
    """Percentage-point delta of raw accuracies, rounded after *100."""
    return round(float(a) * 100.0 - float(b) * 100.0, digits)


def load_metrics(rel: str) -> dict[str, Any]:
    path = RESULTS / rel / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_cls() -> dict[str, Any]:
    """Authoritative classification numbers from metrics.json."""
    m = {
        "original": load_metrics("lineA_original_baseline"),
        "ear_a": load_metrics("lineA_original_up/ear"),
        "pdans_a": load_metrics("lineA_original_up/pdans"),
        "punet_a": load_metrics("lineA_original_up/pu_net"),
        "pugcn_a": load_metrics("lineA_original_up/pu_gcn"),
        "puef_a": load_metrics("lineA_original_up/pu_edgeformer"),
        "down": load_metrics("lineB_downsampled_x4_baseline"),
        "ear_b": load_metrics("lineB_downsampled_x4_up/ear"),
        "pdans_b": load_metrics("lineB_downsampled_x4_up/pdans"),
        "punet_b": load_metrics("lineB_downsampled_x4_up/pu_net"),
        "pugcn_b": load_metrics("lineB_downsampled_x4_up/pu_gcn"),
        "puef_b": load_metrics("lineB_downsampled_x4_up/pu_edgeformer"),
        "mesh256": load_metrics("mesh_ref_baseline_256"),
        "mesh4096": load_metrics("mesh_ref_baseline_4096"),
    }

    def row(key: str, name: str) -> dict[str, Any]:
        d = m[key]
        return {
            "name": name,
            "num_point": int(d["num_point"]),
            "best": pct(d["overall_accuracy"]),
            "final": pct(d["final_test_overall_accuracy"]),
            "best_epoch": int(d["best_epoch"]),
            "raw_best": float(d["overall_accuracy"]),
            "raw_final": float(d["final_test_overall_accuracy"]),
        }

    orig = row("original", "Original 1024")
    down = row("down", "Downsampled ×4")
    mesh256 = row("mesh256", "Mesh-ref 256")
    mesh4096 = row("mesh4096", "Mesh-ref 4096")

    line_a = [
        row("ear_a", "EAR"),
        row("pdans_a", "PDANS"),
        row("punet_a", "PU-Net"),
        row("pugcn_a", "PU-GCN"),
        row("puef_a", "PU-EdgeFormer"),
    ]
    # Sort Line A by best OA descending for fair table; keep mesh-ref comparison order later
    line_a_vs_mesh = sorted(line_a, key=lambda r: -r["best"])
    for r in line_a:
        r["delta_vs_orig"] = delta_pp(r["raw_best"], orig["raw_best"])
        r["delta_vs_mesh4096"] = delta_pp(r["raw_best"], mesh4096["raw_best"])

    line_b = [
        row("ear_b", "EAR"),
        row("pdans_b", "PDANS"),
        row("punet_b", "PU-Net"),
        row("pugcn_b", "PU-GCN"),
        row("puef_b", "PU-EdgeFormer"),
    ]
    for r in line_b:
        r["delta_vs_down"] = delta_pp(r["raw_best"], down["raw_best"])
        r["gap_vs_orig"] = delta_pp(r["raw_best"], orig["raw_best"])

    mesh256["delta_vs_down"] = delta_pp(mesh256["raw_best"], down["raw_best"])
    mesh4096["delta_vs_orig"] = delta_pp(mesh4096["raw_best"], orig["raw_best"])

    return {
        "original": orig,
        "down": down,
        "mesh256": mesh256,
        "mesh4096": mesh4096,
        "line_a": line_a,
        "line_a_vs_mesh": line_a_vs_mesh,
        "line_b": line_b,
    }


def import_ref_module():
    path = PROJECT / "scripts" / "generate_original_reference_presentation.py"
    spec = importlib.util.spec_from_file_location("orig_ref_pres", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orig_ref_pres"] = mod
    spec.loader.exec_module(mod)
    return mod


def set_run(run, text: str, size: int = 18, bold: bool = False, color=DARK):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"


def add_title(slide, text: str, top=0.22, size=26):
    box = slide.shapes.add_textbox(Inches(0.45), Inches(top), Inches(12.4), Inches(0.5))
    p = box.text_frame.paragraphs[0]
    run = p.add_run()
    set_run(run, text, size, True, NAVY)


def add_section_banner(slide, section: str, title: str):
    """Small section tag + title."""
    tag = slide.shapes.add_textbox(Inches(0.45), Inches(0.12), Inches(12.4), Inches(0.28))
    p = tag.text_frame.paragraphs[0]
    run = p.add_run()
    set_run(run, section.upper(), 11, True, ACCENT)
    add_title(slide, title, top=0.38, size=24)


def add_bullets(slide, bullets: list[str], left=0.5, top=1.0, width=12.2, height=5.5, size=16):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        set_run(run, "•  " + item, size, False, DARK)
        p.space_after = Pt(6)


def add_footer(slide, page: int, total: int):
    box = slide.shapes.add_textbox(Inches(0.45), Inches(7.05), Inches(12.4), Inches(0.3))
    p = box.text_frame.paragraphs[0]
    run = p.add_run()
    set_run(run, f"ModelNet40 · PointNet++ Upsampling Full Report    {page}/{total}", 10, False, GRAY)


def add_table(slide, rows: list[list[str]], left, top, width, height, font_size=13):
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


def add_note(slide, text: str, top=5.7, size=12):
    box = slide.shapes.add_textbox(Inches(0.45), Inches(top), Inches(12.4), Inches(1.2))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    set_run(run, text, size, False, DARK)


def blank(prs: Presentation):
    return prs.slides.add_slide(prs.slide_layouts[6])


def fmt4(v: Any) -> str:
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return f"{float(v):.4f}"
    return "—"


def ensure_figures(ref_mod) -> tuple[dict, dict[str, Path]]:
    geo = ref_mod.load_geometry()
    # Prefer existing figures; regenerate if missing key charts
    need = ["lineA_cls", "lineB_cls", "mesh_ref_cls"]
    figs: dict[str, Path] = {}
    mapping = {
        "lineA_cls": FIG_REV / "lineA_best_overall.png",
        "lineB_cls": FIG_REV / "lineB_best_overall.png",
        "mesh_ref_cls": FIG_REV / "mesh_ref_best_overall.png",
        "lineA_cd": FIG_REV / "lineA_cd_vs_original.png",
        "lineA_hd": FIG_REV / "lineA_hd_vs_original.png",
        "lineB_cd": FIG_REV / "lineB_cd_vs_original.png",
        "lineB_hd": FIG_REV / "lineB_hd_vs_original.png",
        "lineA_scatter": FIG_REV / "lineA_geom_vs_cls.png",
        "lineB_scatter": FIG_REV / "lineB_geom_vs_cls.png",
    }
    for cls in ("airplane", "chair", "table", "car", "sofa"):
        for line in ("A", "B"):
            key = f"pc_{line}_{cls}"
            sid = ref_mod.SAMPLES[cls]
            mapping[key] = FIG_REV / "pointclouds" / f"line{line}_{cls}_{sid}_comparison.png"

    missing = [k for k in need if not mapping[k].exists()]
    if missing or not all(mapping[k].exists() for k in ("lineA_cd", "lineB_cd") if geo.get("A") or geo.get("B")):
        figs = ref_mod.generate_figures(geo)
    else:
        for k, p in mapping.items():
            if p.exists():
                figs[k] = p
        # Fill any remaining via regenerate if needed
        if len(figs) < 8:
            figs = ref_mod.generate_figures(geo)
    return geo, figs


def build_pptx(cls: dict[str, Any], geo: dict, figs: dict[str, Path]) -> int:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    slides: list = []

    def add():
        s = blank(prs)
        slides.append(s)
        return s

    orig = cls["original"]
    down = cls["down"]
    mesh256 = cls["mesh256"]
    mesh4096 = cls["mesh4096"]
    line_a = cls["line_a"]
    line_b = cls["line_b"]
    geo_a = geo.get("A") or []
    geo_b = geo.get("B") or []

    # ---------- A. Setup & Motivation ----------
    # 1 Title
    s = add()
    box = s.shapes.add_textbox(Inches(0.7), Inches(1.6), Inches(12.0), Inches(2.4))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    set_run(run, "ModelNet40 Point-Cloud Upsampling", 32, True, NAVY)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    run = p2.add_run()
    set_run(run, "Equal-N Geometry + PointNet++ Classification", 22, False, GRAY)
    p3 = tf.add_paragraph()
    p3.alignment = PP_ALIGN.CENTER
    run = p3.add_run()
    set_run(run, "Full Thesis Experiment Report", 18, False, DARK)
    box2 = s.shapes.add_textbox(Inches(1.0), Inches(4.5), Inches(11.3), Inches(1.8))
    tf = box2.text_frame
    for i, line in enumerate(
        [
            "Line A densification (→4096)  ·  Line B recovery (256→1024)",
            "Upsamplers: EAR · PDANS · PU-Net · PU-GCN · PU-EdgeFormer",
            "Downstream: PointNet++ SSG (pointnet2_cls_ssg), XYZ-only, 40 classes",
            "Classifier: retrain from scratch per variant (matched num_point)",
        ]
    ):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        set_run(run, line, 15, False, DARK)

    # 2 Agenda
    s = add()
    add_section_banner(s, "A · Setup", "Agenda")
    add_bullets(
        s,
        [
            "A. Setup & Motivation — research question, two lines, methods, classifier",
            "B. Data & Protocols — ModelNet40 split, baselines, equal-N geometry rules",
            "C. Retrain Method — WHY PointNet++ SSG from scratch (supervisor Q&A)",
            "D. Classification Results — OA tables (baselines, Line A, Line B)",
            "E. Geometry Equal-N Highlights — CD/HD vs mesh-ref / Original",
            "F. Findings & Conclusions — honest limitations",
            "G. Appendix — report paths, metrics sources",
        ],
        top=1.05,
        size=17,
    )

    # 3 Research question
    s = add()
    add_section_banner(s, "A · Setup", "Research Question")
    add_bullets(
        s,
        [
            "Core question: Does point-cloud upsampling help 3D object classification on ModelNet40?",
            "Line A (Densification): start from Official Original 1024 → upsample ×4 → 4096; ask if denser clouds improve PointNet++ OA",
            "Line B (Recovery): downsample Original to 256 → upsample ×4 → 1024; ask if upsampling can recover the lost signal",
            "Two evaluation axes: (1) geometry consistency under equal cardinality; (2) classification OA after retrain",
            "Hypothesis under test is empirical — results may show geometry gains without OA gains",
        ],
        top=1.05,
        size=16,
    )

    # 4 Two lines
    s = add()
    add_section_banner(s, "A · Setup", "Two Experimental Lines")
    add_table(
        s,
        [
            ["Line", "Role", "Pipeline", "Output pts", "Classifier num_point"],
            ["A", "Densification", "Original 1024 → Upsampler ×4", "4096", "4096 (retrain)"],
            ["B", "Recovery", "Downsample → 256 → Upsampler ×4", "1024", "1024 (retrain)"],
            ["B base", "Sparse baseline", "Original 1024 → Downsample ×4", "256", "256 (train)"],
            ["Mesh-ref", "Fair equal-N ref", "Area-weighted sample from .off", "256 / 4096", "matched (retrain)"],
        ],
        Inches(0.4),
        Inches(1.1),
        Inches(12.5),
        Inches(3.4),
        font_size=13,
    )
    add_note(
        s,
        "Same five upsamplers on both lines. Upsamplers only generate clouds; classification is a separate PointNet++ stage.",
        top=4.8,
        size=14,
    )

    # 5 Methods
    s = add()
    add_section_banner(s, "A · Setup", "Upsampling Methods & Downstream Classifier")
    add_bullets(
        s,
        [
            "EAR — Edge-Aware Resampling (classical / geometric densification)",
            "PDANS — learning-based upsampling (project pipeline, ×4)",
            "PU-Net — pioneering learning-based point upsampling network",
            "PU-GCN — graph-convolutional upsampling",
            "PU-EdgeFormer — edge-aware transformer-style upsampler (included in final tables)",
            "Downstream classifier: PointNet++ SSG (`pointnet2_cls_ssg`), XYZ only, 40 ModelNet40 classes",
            "No normals / features beyond XYZ in the classification input used here",
        ],
        top=1.05,
        size=15,
    )

    # ---------- B. Data & Protocols ----------
    # 6 Dataset
    s = add()
    add_section_banner(s, "B · Data & Protocols", "ModelNet40 Data Split")
    add_bullets(
        s,
        [
            "Official ModelNet40 classification split: 9843 train / 2468 test",
            "No validation split for this task — paper and code assert split ∈ {train, test}",
            "Original 1024: mesh-sampled baseline used throughout the project (unit-sphere normalized)",
            "Mesh-ref 256 / Mesh-ref 4096: independent area-weighted samples from `.off` at equal cardinality",
            "Mesh-ref is NOT downsampled from Original 1024 — different sampling process, same mesh source family",
            "Downsampled ×4 (256): FPS/protocol downsample from Original 1024 (Line B sparse baseline)",
        ],
        top=1.05,
        size=15,
    )

    # 7 Baselines overview
    s = add()
    add_section_banner(s, "B · Data & Protocols", "Classification Baselines (Exact OA)")
    add_table(
        s,
        [
            ["Baseline", "Pts", "Best OA", "Final OA", "Best epoch", "Role"],
            [
                "Original 1024",
                "1024",
                f"{orig['best']:.2f}%",
                f"{orig['final']:.2f}%",
                str(orig["best_epoch"]),
                "Primary Line A / recovery target",
            ],
            [
                "Downsampled ×4",
                "256",
                f"{down['best']:.2f}%",
                f"{down['final']:.2f}%",
                str(down["best_epoch"]),
                "Line B sparse baseline",
            ],
            [
                "Mesh-ref 256",
                "256",
                f"{mesh256['best']:.2f}%",
                f"{mesh256['final']:.2f}%",
                str(mesh256["best_epoch"]),
                "Equal-N 256 mesh sample",
            ],
            [
                "Mesh-ref 4096",
                "4096",
                f"{mesh4096['best']:.2f}%",
                f"{mesh4096['final']:.2f}%",
                str(mesh4096["best_epoch"]),
                "Equal-N 4096 mesh sample",
            ],
        ],
        Inches(0.4),
        Inches(1.1),
        Inches(12.5),
        Inches(3.6),
        font_size=13,
    )
    add_note(
        s,
        f"Mesh-ref 256 vs Downsampled: {mesh256['delta_vs_down']:+.2f} pp. "
        f"Mesh-ref 4096 vs Original: {mesh4096['delta_vs_orig']:+.2f} pp. "
        "Source: pointnet2_results/.../metrics.json",
        top=5.0,
        size=13,
    )

    # 8 Geometry protocols
    s = add()
    add_section_banner(s, "B · Data & Protocols", "Geometry Evaluation Protocols")
    add_bullets(
        s,
        [
            "Legacy (unfair when |P| differs): CD/HD vs Original 1024 for all clouds — unequal cardinality confounds ranking",
            "Equal-N (used in this report): compare only at the same point count",
            "  · Line A: upsamplers 4096 vs Mesh-ref 4096",
            "  · Downsampled 256 vs Mesh-ref 256",
            "  · Line B recovery: upsamplers 1024 vs Original 1024",
            "CD: mean NN L2(a→b) + mean NN L2(b→a); HD: max directed Hausdorff; lower is better",
            "Self-comparison rows (mesh-ref / Original): CD = 0, HD = 0 by definition. No P2F.",
        ],
        top=1.05,
        size=15,
    )

    # ---------- C. Retrain WHY ----------
    # 9 Why retrain — EN
    s = add()
    add_section_banner(s, "C · Retrain Protocol", "Why Retrain? — Definition")
    add_bullets(
        s,
        [
            "Network: PointNet++ SSG (`pointnet2_cls_ssg`), XYZ only, 40 classes",
            "Retrain = train the classifier FROM SCRATCH on each point-cloud variant",
            "Matched `--num-point` to the cloud cardinality (256 / 1024 / 4096)",
            "NOT finetune from the Original-1024 checkpoint",
            "NOT training / updating the upsampler in this stage — upsamplers only produce clouds",
            "Each variant gets its own independent best_model.pth under identical hparams",
            "Training: 200 epochs, Adam lr=0.001, seed=42; best by test instance accuracy",
        ],
        top=1.05,
        size=15,
    )

    # 10 Why retrain — reasons
    s = add()
    add_section_banner(s, "C · Retrain Protocol", "Why This Method? (5 Reasons)")
    add_bullets(
        s,
        [
            "1. Isolates the effect of the point-cloud representation (density / distribution) under a fixed backbone",
            "2. Matching num_point avoids input-size mismatch artifacts (e.g. forcing 4096 into a 1024 model)",
            "3. From-scratch avoids carrying features tuned to another density/distribution",
            "4. PointNet++ SSG is the standard ModelNet40 backbone in this project — keeps comparisons comparable",
            "5. Upsamplers generate clouds only; “retrain” always means the classifier, never the upsampler here",
            "Fairness note: same architecture + same hparams + independent checkpoints per variant",
            "Stricter alternative (not done): one frozen Original-1024 classifier evaluating all clouds — future / limitation",
        ],
        top=1.05,
        size=14,
    )

    # 11 Why retrain — Chinese bilingual for supervisor Q&A
    s = add()
    add_section_banner(s, "C · Retrain Protocol", "Why Retrain? /  Why retrain classifier? ")
    add_bullets(
        s,
        [
            " Network: PointNet++ SSG (pointnet2_cls_ssg),  only  XYZ, 40  Category ",
            "Retrain =  Train the classifier from scratch on each point-cloud variant ( matched num_point),  It’s not true.  Original-1024  Weighted  finetune",
            " And not at this stage, the upsampling network. —— upsampler is only responsible for generated points clouds ",
            " Purpose: In equity, standards  backbone  under these conditions, isolate the point-cloud representation (density / Impact on classification ",
            " Match point count to avoid input size mismatch; avoid sending another density feature to the current variant from zero training ",
            " Training: 200 epochs, Adam 0.001, seed=42;  Press  test instance accuracy  Save  best_model ( None  val  Synopsis, need to be honest) ",
            " Current protocol: Independent of the same structure and supersync, per variant  checkpoint;  Tighter protocol = frozen Single classifier Assessment of all clouds (not done) ",
        ],
        top=1.05,
        size=14,
    )

    # 12 Training setup detail
    s = add()
    add_section_banner(s, "C · Retrain Protocol", "Training Setup & Best vs Final")
    add_table(
        s,
        [
            ["Item", "Setting"],
            ["Model", "pointnet2_cls_ssg (PointNet++ SSG)"],
            ["Input", "XYZ only · 40 classes"],
            ["Optimizer / lr", "Adam · 0.001"],
            ["Epochs / seed", "200 · 42"],
            ["Best checkpoint", "Highest test instance (overall) accuracy"],
            ["Final Overall", "Epoch-200 test OA (supplementary)"],
            ["Val split", "None — ModelNet40 train|test only (paper + code)"],
        ],
        Inches(1.5),
        Inches(1.1),
        Inches(10.2),
        Inches(4.6),
        font_size=14,
    )

    # ---------- D. Classification ----------
    # 13 Line A vs Original
    s = add()
    add_section_banner(s, "D · Classification", "Line A vs Original 1024 (Best OA)")
    if "lineA_cls" in figs:
        s.shapes.add_picture(str(figs["lineA_cls"]), Inches(0.3), Inches(0.95), width=Inches(7.2))
    rows = [["Method", "Pts", "Best OA", "Final OA", "Δ vs Original"]]
    rows.append(
        [
            "Original",
            str(orig["num_point"]),
            f"{orig['best']:.2f}%",
            f"{orig['final']:.2f}%",
            "+0.00 pp",
        ]
    )
    for r in line_a:
        rows.append(
            [
                r["name"],
                str(r["num_point"]),
                f"{r['best']:.2f}%",
                f"{r['final']:.2f}%",
                f"{r['delta_vs_orig']:+.2f} pp",
            ]
        )
    add_table(s, rows, Inches(7.7), Inches(0.95), Inches(5.2), Inches(4.8), font_size=11)
    add_note(
        s,
        "All 4096 branches: PointNet++ retrained from scratch at num_point=4096. "
        f"No densification method exceeds Original Best OA {orig['best']:.2f}%.",
        top=6.0,
        size=12,
    )

    # 14 Line A vs Mesh-ref 4096 (CRITICAL table from user)
    s = add()
    add_section_banner(s, "D · Classification", "Line A 4096 vs Mesh-ref 4096 (Fair Equal-N OA)")
    rows = [["Method", "Pts", "Best OA", "Δ vs Mesh-ref 4096", "Final OA"]]
    rows.append(
        [
            "Mesh-ref 4096",
            "4096",
            f"{mesh4096['best']:.2f}%",
            "+0.00 pp",
            f"{mesh4096['final']:.2f}%",
        ]
    )
    for r in sorted(line_a, key=lambda x: -x["best"]):
        rows.append(
            [
                r["name"],
                "4096",
                f"{r['best']:.2f}%",
                f"{r['delta_vs_mesh4096']:+.2f} pp",
                f"{r['final']:.2f}%",
            ]
        )
    add_table(s, rows, Inches(1.8), Inches(1.1), Inches(9.7), Inches(4.4), font_size=14)
    add_note(
        s,
        "No Line A upsampler beats Mesh-ref 4096 on Best OA. "
        "PU-GCN is closest (−0.36 pp); PU-EdgeFormer farthest (−1.46 pp). "
        "Numbers from metrics.json (raw Δ rounded to 2 decimals).",
        top=5.75,
        size=13,
    )

    # 15 Line B
    s = add()
    add_section_banner(s, "D · Classification", "Line B Recovery Classification")
    if "lineB_cls" in figs:
        s.shapes.add_picture(str(figs["lineB_cls"]), Inches(0.25), Inches(0.95), width=Inches(7.0))
    rows = [["Method", "Pts", "Best OA", "Final OA", "Δ vs Down", "Gap vs Orig"]]
    rows.append(
        [
            "Downsampled ×4",
            "256",
            f"{down['best']:.2f}%",
            f"{down['final']:.2f}%",
            "+0.00 pp",
            f"{delta_pp(down['raw_best'], orig['raw_best']):+.2f} pp",
        ]
    )
    for r in line_b:
        rows.append(
            [
                r["name"],
                str(r["num_point"]),
                f"{r['best']:.2f}%",
                f"{r['final']:.2f}%",
                f"{r['delta_vs_down']:+.2f} pp",
                f"{r['gap_vs_orig']:+.2f} pp",
            ]
        )
    add_table(s, rows, Inches(7.4), Inches(0.95), Inches(5.6), Inches(4.6), font_size=10)
    punet_b = next(r for r in line_b if r["name"] == "PU-Net")
    add_note(
        s,
        f"PU-Net Best OA {punet_b['best']:.2f}% vs Downsampled {down['best']:.2f}% "
        f"({punet_b['delta_vs_down']:+.2f} pp), but still {punet_b['gap_vs_orig']:+.2f} pp below Original "
        f"{orig['best']:.2f}%. Partial recovery only.",
        top=5.85,
        size=12,
    )

    # 16 Mesh-ref baselines slide
    s = add()
    add_section_banner(s, "D · Classification", "Mesh-ref vs Existing Baselines")
    if "mesh_ref_cls" in figs:
        s.shapes.add_picture(str(figs["mesh_ref_cls"]), Inches(0.35), Inches(1.0), width=Inches(7.0))
    rows = [
        ["Comparison", "Best OA A", "Best OA B", "Δ"],
        [
            "Mesh-ref 256 vs Downsampled 256",
            f"{mesh256['best']:.2f}%",
            f"{down['best']:.2f}%",
            f"{mesh256['delta_vs_down']:+.2f} pp",
        ],
        [
            "Mesh-ref 4096 vs Original 1024",
            f"{mesh4096['best']:.2f}%",
            f"{orig['best']:.2f}%",
            f"{mesh4096['delta_vs_orig']:+.2f} pp",
        ],
    ]
    add_table(s, rows, Inches(7.5), Inches(1.2), Inches(5.4), Inches(2.2), font_size=12)
    add_bullets(
        s,
        [
            "Equal-N mesh sampling classifies almost identically to Downsampled / Original counterparts",
            "Denser Mesh-ref 4096 does not beat Original 1024 by a meaningful margin (+0.05 pp)",
            f"Final OA also reported: Mesh-ref 256 {mesh256['final']:.2f}%; Mesh-ref 4096 {mesh4096['final']:.2f}%",
        ],
        left=7.5,
        top=3.7,
        width=5.4,
        height=2.8,
        size=13,
    )

    # 17 Full OA summary table
    s = add()
    add_section_banner(s, "D · Classification", "Full Best OA Summary (All Variants)")
    rows = [["Group", "Method", "Pts", "Best OA", "Final OA"]]
    rows.append(["Base", "Original", "1024", f"{orig['best']:.2f}%", f"{orig['final']:.2f}%"])
    rows.append(["Base", "Downsampled ×4", "256", f"{down['best']:.2f}%", f"{down['final']:.2f}%"])
    rows.append(["Base", "Mesh-ref 256", "256", f"{mesh256['best']:.2f}%", f"{mesh256['final']:.2f}%"])
    rows.append(["Base", "Mesh-ref 4096", "4096", f"{mesh4096['best']:.2f}%", f"{mesh4096['final']:.2f}%"])
    for r in line_a:
        rows.append(["Line A", r["name"], "4096", f"{r['best']:.2f}%", f"{r['final']:.2f}%"])
    for r in line_b:
        rows.append(["Line B", r["name"], "1024", f"{r['best']:.2f}%", f"{r['final']:.2f}%"])
    add_table(s, rows, Inches(1.6), Inches(0.95), Inches(10.1), Inches(5.7), font_size=11)

    # ---------- E. Geometry ----------
    # 18 Line A geometry
    s = add()
    add_section_banner(s, "E · Geometry", "Line A Equal-N Geometry (vs Mesh-ref 4096)")
    rows_a = [r for r in geo_a if int(r.get("points", -1)) == 4096]
    if rows_a:
        rows = [["Method", "CD ↓", "HD ↓", "NUC", "ΔNUC"]]
        for r in rows_a:
            rows.append(
                [
                    str(r.get("method", "")),
                    fmt4(r.get("cd_vs_ref", r.get("cd_vs_original"))),
                    fmt4(r.get("hd_vs_ref", r.get("hd_vs_original"))),
                    fmt4(r.get("nuc")),
                    fmt4(r.get("delta_nuc_vs_ref", r.get("delta_nuc_vs_original"))),
                ]
            )
        add_table(s, rows, Inches(0.5), Inches(1.0), Inches(7.2), Inches(4.5), font_size=12)
    if "lineA_cd" in figs:
        s.shapes.add_picture(str(figs["lineA_cd"]), Inches(7.9), Inches(1.1), width=Inches(5.0))
    add_note(
        s,
        "PU-GCN has the lowest CD among Line A upsamplers vs Mesh-ref 4096 (best geometry among methods), "
        "yet still trails Mesh-ref on OA (−0.36 pp). Better geometry ≠ automatic OA gains.",
        top=5.8,
        size=13,
    )

    # 19 Line B geometry
    s = add()
    add_section_banner(s, "E · Geometry", "Line B Equal-N Geometry")
    rows_b256 = [r for r in geo_b if int(r.get("points", -1)) == 256]
    rows_b1024 = [r for r in geo_b if int(r.get("points", -1)) == 1024]
    y = 0.95
    if rows_b256:
        title_box = s.shapes.add_textbox(Inches(0.45), Inches(y), Inches(12.4), Inches(0.3))
        run = title_box.text_frame.paragraphs[0].add_run()
        set_run(run, "256 vs Mesh-ref 256", 13, True, NAVY)
        y += 0.3
        rows = [["Method", "CD ↓", "HD ↓", "NUC", "ΔNUC"]]
        for r in rows_b256:
            rows.append(
                [
                    str(r.get("method", "")),
                    fmt4(r.get("cd_vs_ref", r.get("cd_vs_original"))),
                    fmt4(r.get("hd_vs_ref", r.get("hd_vs_original"))),
                    fmt4(r.get("nuc")),
                    fmt4(r.get("delta_nuc_vs_ref", r.get("delta_nuc_vs_original"))),
                ]
            )
        add_table(s, rows, Inches(0.4), Inches(y), Inches(12.5), Inches(1.25), font_size=11)
        y += 1.4
    if rows_b1024:
        title_box = s.shapes.add_textbox(Inches(0.45), Inches(y), Inches(12.4), Inches(0.3))
        run = title_box.text_frame.paragraphs[0].add_run()
        set_run(run, "1024 vs Original 1024 (recovery)", 13, True, NAVY)
        y += 0.3
        rows = [["Method", "CD ↓", "HD ↓", "NUC", "ΔNUC"]]
        for r in rows_b1024:
            rows.append(
                [
                    str(r.get("method", "")),
                    fmt4(r.get("cd_vs_ref", r.get("cd_vs_original"))),
                    fmt4(r.get("hd_vs_ref", r.get("hd_vs_original"))),
                    fmt4(r.get("nuc")),
                    fmt4(r.get("delta_nuc_vs_ref", r.get("delta_nuc_vs_original"))),
                ]
            )
        add_table(s, rows, Inches(0.4), Inches(y), Inches(12.5), Inches(3.0), font_size=11)

    # 20 Geometry vs classification
    s = add()
    add_section_banner(s, "E · Geometry", "Geometry vs Classification — Key Message")
    if "lineB_scatter" in figs:
        s.shapes.add_picture(str(figs["lineB_scatter"]), Inches(0.4), Inches(1.0), width=Inches(6.8))
    if "lineA_scatter" in figs:
        s.shapes.add_picture(str(figs["lineA_scatter"]), Inches(7.3), Inches(1.0), width=Inches(5.6))
    add_bullets(
        s,
        [
            "Lower equal-N CD does not automatically imply higher PointNet++ Best OA",
            "Line A: PU-GCN best geometry among upsamplers, still below Mesh-ref 4096 on OA",
            "Line B: PU-Net best OA recovery; geometry ranking does not fully match OA ranking",
            "Report geometry and classification as separate axes — do not conflate them",
        ],
        top=4.9,
        height=1.8,
        size=14,
    )

    # Qualitative
    s = add()
    add_section_banner(s, "E · Geometry", "Qualitative — Line A (Densification)")
    for key in ("pc_A_airplane", "pc_A_chair"):
        if key in figs:
            s.shapes.add_picture(str(figs[key]), Inches(0.3), Inches(0.95), width=Inches(12.7))
            break
    add_note(s, "Same sample · same camera · same axis limits · color = depth (not error)", top=6.5, size=12)

    s = add()
    add_section_banner(s, "E · Geometry", "Qualitative — Line B (Recovery)")
    for key in ("pc_B_airplane", "pc_B_chair"):
        if key in figs:
            s.shapes.add_picture(str(figs[key]), Inches(0.3), Inches(0.95), width=Inches(12.7))
            break
    add_note(s, "Same sample · same camera · same axis limits · color = depth (not error)", top=6.5, size=12)

    # ---------- F. Findings ----------
    s = add()
    add_section_banner(s, "F · Findings", "Main Findings")
    add_bullets(
        s,
        [
            f"No Line A upsampler beats Mesh-ref 4096 Best OA ({mesh4096['best']:.2f}%)",
            f"No Line A upsampler beats Original 1024 Best OA ({orig['best']:.2f}%) under matched 4096 retrain",
            f"Mesh-ref ≈ counterparts: 256 {mesh256['best']:.2f}% vs Down {down['best']:.2f}% "
            f"({mesh256['delta_vs_down']:+.2f} pp); 4096 {mesh4096['best']:.2f}% vs Orig {orig['best']:.2f}% "
            f"({mesh4096['delta_vs_orig']:+.2f} pp)",
            f"Line B: PU-Net {punet_b['best']:.2f}% = only method above Downsampled "
            f"({punet_b['delta_vs_down']:+.2f} pp), still {punet_b['gap_vs_orig']:+.2f} pp below Original",
            "Upsampling can improve / alter geometry; classification benefit is limited and method-unstable",
            "Better equal-N geometry does not guarantee better OA",
        ],
        top=1.05,
        size=15,
    )

    s = add()
    add_section_banner(s, "F · Findings", "Honest Limitations")
    add_bullets(
        s,
        [
            "Best Overall selected on the test set — ModelNet40 has no val split in this PointNet++ setup",
            "Independent retrain per variant (fair matched protocol) ≠ single frozen classifier on all clouds",
            "Stricter protocol (frozen Original-1024 classifier) was not run — optional future work",
            "Mesh-ref CD/HD measures consistency with an independent equal-N mesh sample, not identity with Original",
            "NUC depends on radius / sampling parameters — keep fixed when comparing",
            "Results are specific to PointNet++ SSG + these upsamplers on ModelNet40",
        ],
        top=1.05,
        size=15,
    )

    s = add()
    add_section_banner(s, "F · Findings", "Conclusions")
    add_bullets(
        s,
        [
            "Under a fair from-scratch PointNet++ SSG retrain, densification (Line A) does not improve OA vs mesh-ref 4096 or Original 1024",
            "Recovery (Line B) is partial at best: only PU-Net exceeds the sparse baseline, and not enough to restore Original OA",
            "Mesh sampling at equal N nearly ties existing baselines — density alone is not a free classification win",
            "Geometry and classification must be reported separately; upsampling quality is method- and task-dependent",
            "Retrain protocol deliberately isolates representation effects — that is why we use PointNet++ SSG from scratch",
        ],
        top=1.05,
        size=15,
    )

    # ---------- G. Appendix ----------
    s = add()
    add_section_banner(s, "G · Appendix", "Sources — Reports & Metrics")
    add_bullets(
        s,
        [
            "OA metrics: pointnet2_results/x4_two_line_final/**/metrics.json",
            "Mesh-ref baselines: reports/modelnet40_pointnet2_mesh_ref_baseline_results.md",
            "Two-line summary: reports/modelnet40_pointnet2_final_two_line_classification_summary_with_pu_edgeformer.md",
            "Equal-N geometry: reports/modelnet40_geometry_equal_n_audit.md (+ summary CSV)",
            "Equal-N protocol: reports/modelnet40_geometry_equal_n_protocol.md",
            "Generator: scripts/generate_full_thesis_presentation.py (reproducible)",
            "Companion revised deck: presentations/ModelNet40_PointNet2_Original_Reference_Revised.pptx",
        ],
        top=1.05,
        size=14,
    )

    s = add()
    add_section_banner(s, "G · Appendix", "High-Level Pipeline Commands / Jobs")
    add_bullets(
        s,
        [
            "Build mesh refs: scripts/build_mesh_reference_points.py → datasets/modelnet40_mesh_ref_{256,4096}/",
            "Equal-N geometry: scripts/compute_geometry_equal_n.py → reports/modelnet40_geometry_equal_n_*.csv",
            "PointNet++ train: scripts/train_pointnet2.py / run_pointnet2_x4_two_lines.py (per-variant from scratch)",
            "Mesh-ref PointNet++: scripts/prepare_mesh_ref_pointnet2_baselines.py + collect_mesh_ref_pointnet2_results.py",
            "Line A/B upsampling generation: run_lineA_original_x4_upsampling.py / run_lineB_downsampled_x4_upsampling.py",
            "This deck: python scripts/generate_full_thesis_presentation.py",
        ],
        top=1.05,
        size=14,
    )

    # Closing
    s = add()
    box = s.shapes.add_textbox(Inches(1), Inches(2.5), Inches(11.3), Inches(2.0))
    p = box.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    set_run(run, "Questions?", 44, True, NAVY)
    p2 = box.text_frame.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    run = p2.add_run()
    set_run(run, "Thank you", 22, False, GRAY)
    p3 = box.text_frame.add_paragraph()
    p3.alignment = PP_ALIGN.CENTER
    run = p3.add_run()
    set_run(
        run,
        f"Original {orig['best']:.2f}% · Mesh-ref 4096 {mesh4096['best']:.2f}% · "
        f"PU-Net Line B {punet_b['best']:.2f}%",
        14,
        False,
        DARK,
    )

    total = len(slides)
    for i, slide in enumerate(slides, start=1):
        add_footer(slide, i, total)

    PRES.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    return total


def build_pdf(cls: dict[str, Any], geo: dict, figs: dict[str, Path], n_slides: int) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
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

    orig = cls["original"]
    down = cls["down"]
    mesh256 = cls["mesh256"]
    mesh4096 = cls["mesh4096"]
    line_a = cls["line_a"]
    line_b = cls["line_b"]
    punet_b = next(r for r in line_b if r["name"] == "PU-Net")

    page = landscape(A4)
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=page,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.4 * inch,
        bottomMargin=0.4 * inch,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "T", parent=styles["Title"], fontSize=24, textColor=colors.HexColor("#1a3a5c"), alignment=TA_CENTER, spaceAfter=10
    )
    h = ParagraphStyle("H", parent=styles["Heading1"], fontSize=16, textColor=colors.HexColor("#1a3a5c"), spaceAfter=8)
    body = ParagraphStyle("B", parent=styles["BodyText"], fontSize=11, leading=15, spaceAfter=5)
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
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(t)

    story.append(Paragraph("ModelNet40 Point-Cloud Upsampling — Full Thesis Report", title))
    story.append(
        Paragraph(
            "Equal-N Geometry + PointNet++ SSG Retrain Classification",
            ParagraphStyle("S", parent=body, alignment=TA_CENTER, fontSize=13, textColor=colors.HexColor("#555555")),
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    bullets(
        [
            f"Companion PPTX has {n_slides} slides: {OUT_PPTX.name}",
            "Classifier: pointnet2_cls_ssg from scratch per variant · 200 epochs · Adam 0.001 · seed 42",
            "NOT finetune of Original-1024 checkpoint; NOT training the upsampler in classification stage",
            f"Generated: {utc_now()}",
        ]
    )
    story.append(PageBreak())

    story.append(Paragraph("Why Retrain (English)", h))
    bullets(
        [
            "Retrain = train PointNet++ SSG from scratch on each cloud variant with matched num_point",
            "Isolates representation (density/distribution) under a fair fixed backbone",
            "Matching num_point avoids input-size mismatch; from-scratch avoids transfer from another density",
            "PointNet++ SSG is the project standard ModelNet40 backbone",
            "Upsamplers only generate clouds; retrain refers to the classifier only",
            "Fair protocol: same arch/hparams, independent checkpoints; stricter frozen-classifier protocol not done",
        ]
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(" Why retrain classifier?  ( Chinese )", h))
    bullets(
        [
            " Training from zero on every point cloud variant  PointNet++ SSG ( Match  num_point),  It’s not.  finetune Original-1024",
            " And it’s not training upsampling. —— upsampler Only generated points clouds ",
            " Purpose: In standards  backbone  point cloud for impact on classification ",
            " No validation set; press  test instance accuracy  Save  best_model ( (To be honest) ",
        ]
    )
    story.append(PageBreak())

    story.append(Paragraph("Baselines — Best / Final OA", h))
    table(
        [
            ["Baseline", "Pts", "Best OA", "Final OA"],
            ["Original 1024", "1024", f"{orig['best']:.2f}%", f"{orig['final']:.2f}%"],
            ["Downsampled ×4", "256", f"{down['best']:.2f}%", f"{down['final']:.2f}%"],
            ["Mesh-ref 256", "256", f"{mesh256['best']:.2f}%", f"{mesh256['final']:.2f}%"],
            ["Mesh-ref 4096", "4096", f"{mesh4096['best']:.2f}%", f"{mesh4096['final']:.2f}%"],
        ]
    )
    story.append(PageBreak())

    story.append(Paragraph("Line A vs Mesh-ref 4096", h))
    data = [["Method", "Best OA", "Δ vs Mesh-ref 4096", "Final OA"]]
    data.append(["Mesh-ref 4096", f"{mesh4096['best']:.2f}%", "+0.00 pp", f"{mesh4096['final']:.2f}%"])
    for r in sorted(line_a, key=lambda x: -x["best"]):
        data.append([r["name"], f"{r['best']:.2f}%", f"{r['delta_vs_mesh4096']:+.2f} pp", f"{r['final']:.2f}%"])
    table(data)
    if "lineA_cls" in figs:
        story.append(Spacer(1, 0.1 * inch))
        story.append(Image(str(figs["lineA_cls"]), width=7.2 * inch, height=3.2 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Line B Recovery", h))
    data = [["Method", "Pts", "Best OA", "Δ vs Down", "Gap vs Orig"]]
    data.append(
        [
            "Downsampled ×4",
            "256",
            f"{down['best']:.2f}%",
            "+0.00 pp",
            f"{delta_pp(down['raw_best'], orig['raw_best']):+.2f} pp",
        ]
    )
    for r in line_b:
        data.append(
            [
                r["name"],
                "1024",
                f"{r['best']:.2f}%",
                f"{r['delta_vs_down']:+.2f} pp",
                f"{r['gap_vs_orig']:+.2f} pp",
            ]
        )
    table(data)
    bullets(
        [
            f"PU-Net {punet_b['best']:.2f}% vs Down {down['best']:.2f}% ({punet_b['delta_vs_down']:+.2f} pp); "
            f"still {punet_b['gap_vs_orig']:+.2f} pp below Original {orig['best']:.2f}%",
        ]
    )
    if "lineB_cls" in figs:
        story.append(Image(str(figs["lineB_cls"]), width=7.2 * inch, height=3.2 * inch))
    story.append(PageBreak())

    story.append(Paragraph("Equal-N Geometry Summary", h))
    rows = (geo.get("A") or []) + (geo.get("B") or [])
    if rows:
        data = [["Line", "Method", "Pts", "Ref", "CD", "HD", "NUC", "ΔNUC"]]
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
                ]
            )
        table(data)
    story.append(PageBreak())

    story.append(Paragraph("Findings & Conclusions", h))
    bullets(
        [
            f"No Line A upsampler beats Mesh-ref 4096 ({mesh4096['best']:.2f}%) or Original ({orig['best']:.2f}%)",
            f"Mesh-ref ≈ Down/Original (+{mesh256['delta_vs_down']:.2f} / +{mesh4096['delta_vs_orig']:.2f} pp)",
            f"Line B PU-Net partial recovery only ({punet_b['delta_vs_down']:+.2f} vs Down; {punet_b['gap_vs_orig']:+.2f} vs Orig)",
            "Better geometry ≠ automatic OA gains",
            "Limitations: best on test (no val); independent retrain ≠ frozen single classifier",
        ]
    )

    for key, caption in [
        ("pc_A_airplane", "Line A qualitative — airplane"),
        ("pc_B_airplane", "Line B qualitative — airplane"),
    ]:
        if key in figs:
            story.append(PageBreak())
            story.append(Paragraph(caption, h))
            story.append(Image(str(figs[key]), width=10.5 * inch, height=2.6 * inch))

    doc.build(story)


def write_notes(cls: dict, n_slides: int) -> None:
    orig = cls["original"]
    mesh4096 = cls["mesh4096"]
    line_a = cls["line_a"]
    lines = [
        "# ModelNet40 PointNet2 Full Report — Notes",
        "",
        f"- Generated: {utc_now()}",
        f"- PPTX: `{OUT_PPTX}`",
        f"- PDF: `{OUT_PDF}`",
        f"- Slides: {n_slides}",
        "",
        "## Numbers verified from metrics.json",
        "",
        f"- Original Best OA: {orig['best']:.2f}%",
        f"- Mesh-ref 4096 Best OA: {mesh4096['best']:.2f}%",
        "- Line A Δ vs Mesh-ref 4096:",
    ]
    for r in sorted(line_a, key=lambda x: -x["best"]):
        lines.append(f"  - {r['name']}: {r['best']:.2f}% ({r['delta_vs_mesh4096']:+.2f} pp)")
    lines += [
        "",
        "## Why retrain (included)",
        "",
        "- Dedicated English + Chinese slides explaining PointNet++ SSG from-scratch retrain",
        "- Clarifies: not finetune; not training upsampler; matched num_point; independent checkpoints",
        "",
        "## Regenerate",
        "",
        "```bash",
        "python scripts/generate_full_thesis_presentation.py",
        "```",
        "",
    ]
    OUT_NOTES.write_text("\n".join(lines), encoding="utf-8")
    OUT_STATUS.write_text(
        "\n".join(
            [
                "# Full thesis presentation status",
                "",
                f"- Generated: {utc_now()}",
                f"- PPTX: `{OUT_PPTX}`",
                f"- PDF: `{OUT_PDF}`",
                f"- Slides: {n_slides}",
                "- OA source: `pointnet2_results/x4_two_line_final/**/metrics.json`",
                "- Geometry: equal-N reports under `reports/modelnet40_geometry_equal_n_*`",
                "- Status: DONE",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    print(f"[{utc_now()}] loading classification metrics…")
    cls = load_all_cls()
    print(
        f"  Original={cls['original']['best']:.2f}%  "
        f"Mesh4096={cls['mesh4096']['best']:.2f}%  "
        f"Down={cls['down']['best']:.2f}%"
    )
    for r in sorted(cls["line_a"], key=lambda x: -x["best"]):
        print(f"  LineA {r['name']}: {r['best']:.2f}% ({r['delta_vs_mesh4096']:+.2f} vs mesh4096)")

    print(f"[{utc_now()}] importing figure helpers…")
    ref_mod = import_ref_module()
    print(f"[{utc_now()}] ensuring figures / geometry…")
    geo, figs = ensure_figures(ref_mod)
    print(f"  geo A={len(geo.get('A') or [])} B={len(geo.get('B') or [])} figs={len(figs)}")

    print(f"[{utc_now()}] building PPTX…")
    n = build_pptx(cls, geo, figs)
    print(f"  slides={n} -> {OUT_PPTX}")

    print(f"[{utc_now()}] building PDF…")
    build_pdf(cls, geo, figs, n)
    print(f"  -> {OUT_PDF}")

    write_notes(cls, n)
    print(f"[{utc_now()}] notes -> {OUT_NOTES}")
    print("DONE")


if __name__ == "__main__":
    main()
