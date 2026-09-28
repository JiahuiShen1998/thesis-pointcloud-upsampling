#!/usr/bin/env python3
"""Generate the evidence-complete English PU-GCN detector-adaptation deck."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/home/ra87racy/projects/baseline_detectors/PointRCNN")
CN_SCRIPT = ROOT / "results/pugcn_detector_adaptation_full_val_presentation_20260913/generate_deck.py"
spec = importlib.util.spec_from_file_location("deck_data", CN_SCRIPT)
cn = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cn)

b = cn.b
W, H = cn.W, cn.H
WHITE, INK, NAVY, BLUE = cn.WHITE, cn.INK, cn.NAVY, cn.BLUE
MID, LINE, PALE, PALE_BLUE = cn.MID, cn.LINE, cn.PALE, cn.PALE_BLUE
RED, GREEN = cn.RED, cn.GREEN
GOLD, TEAL = cn.GOLD, cn.TEAL
PALE_RED, PALE_GREEN, PALE_GOLD = cn.PALE_RED, cn.PALE_GREEN, cn.PALE_GOLD
MATRIX, ALL_CLASSES, CONVERGENCE = cn.MATRIX, cn.ALL_CLASSES, cn.CONVERGENCE

OUT = ROOT / "results/pugcn_detector_adaptation_full_val_presentation_en_20260913"
SLIDES = OUT / "slides"
FODP = OUT / "PU_GCN_Full_Validation_Detector_Adaptation_Evidence_20260913.fodp"
PPTX = OUT / "PU_GCN_Full_Validation_Detector_Adaptation_Evidence_20260913.pptx"
PDF = OUT / "PU_GCN_Full_Validation_Detector_Adaptation_Evidence_20260913.pdf"
b.OUT, b.SLIDES, b.FODP = OUT, SLIDES, FODP

MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def t(draw, xy, value, size=28, color=INK, bold=False, max_width=None, spacing=10, anchor=None):
    return b.text(draw, xy, value, size, color, bold, max_width, spacing, anchor)


def row(detector: str, name: str):
    return next(item for item in MATRIX if item["detector"] == detector and item["name"] == name)


def class_row(arm: str, class_name: str):
    return next(
        item for item in ALL_CLASSES
        if item["detector"] == "CenterPoint"
        and item["arm"] == arm
        and item["class"] == class_name
        and item["metric"] == "3d_ap_r40"
    )


def base(title: str, section: str, page: int, source: str | None = None):
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    t(d, (100, 52), "PU-GCN × KITTI FULL VALIDATION", 20, NAVY, True)
    t(d, (1815, 52), section.upper(), 18, MID, True, anchor="ra")
    t(d, (100, 105), title, 44, INK, True, max_width=1650, spacing=5)
    d.line((100, 178, 1820, 178), fill=LINE, width=2)
    d.line((100, 1032, 1820, 1032), fill=LINE, width=2)
    if source:
        t(d, (100, 1042), source, 14, MID, max_width=1570, spacing=2)
    t(d, (1818, 1045), f"{page:02d}", 18, MID, True, anchor="ra")
    return im, d


def save(im, index: int, name: str):
    path = SLIDES / f"{index:02d}_{name}.png"
    im.save(path)
    return path


def metric_card(d, box, value, label, detail, color=NAVY, fill=WHITE):
    b.panel(d, box, fill=fill)
    x0, y0, x1, _ = box
    t(d, (x0 + 28, y0 + 28), value, 48, color, True)
    t(d, (x0 + 28, y0 + 100), label, 23, INK, True, max_width=x1 - x0 - 56, spacing=5)
    t(d, (x0 + 28, y0 + 164), detail, 19, MID, max_width=x1 - x0 - 56, spacing=5)


def vertical_bars(d, box, labels, values, colors, ymax=100):
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 85, x1 - 35, y0 + 42, y1 - 130
    for tick in range(0, ymax + 1, 20):
        yy = bottom - tick / ymax * (bottom - top)
        d.line((left, yy, right, yy), fill=LINE, width=1)
        t(d, (left - 13, yy), str(tick), 16, MID, anchor="rm")
    d.line((left, top, left, bottom), fill=INK, width=2)
    d.line((left, bottom, right, bottom), fill=INK, width=2)
    slot = (right - left) / len(values)
    width = min(112, slot * 0.56)
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        cx = left + (i + 0.5) * slot
        yy = bottom - value / ymax * (bottom - top)
        d.rounded_rectangle((cx - width / 2, yy, cx + width / 2, bottom), radius=6, fill=color)
        t(d, (cx, yy - 13), f"{value:.2f}", 19, INK, True, anchor="ma")
        t(d, (cx, bottom + 21), label, 17, INK, True, anchor="ma")


def paired_comparison(d, box, items):
    """Dumbbell plot explicitly comparing official/unadapted and 3-epoch adapted detectors."""
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 385, x1 - 155, y0 + 72, y1 - 70
    xmin, xmax = 20, 85
    for tick in range(20, 86, 10):
        xx = left + (tick - xmin) / (xmax - xmin) * (right - left)
        d.line((xx, top, xx, bottom), fill=LINE, width=1)
        t(d, (xx, bottom + 20), str(tick), 16, MID, anchor="ma")
    step = (bottom - top) / len(items)
    for i, (label, before, after) in enumerate(items):
        yy = top + (i + 0.5) * step
        xb = left + (before - xmin) / (xmax - xmin) * (right - left)
        xa = left + (after - xmin) / (xmax - xmin) * (right - left)
        t(d, (left - 22, yy), label, 19, INK, True, anchor="rm")
        d.line((xb, yy, xa, yy), fill=GREEN, width=6)
        d.ellipse((xb - 10, yy - 10, xb + 10, yy + 10), fill=WHITE, outline=NAVY, width=4)
        d.ellipse((xa - 10, yy - 10, xa + 10, yy + 10), fill=GREEN)
        t(d, (xb - 10, yy - 23), f"{before:.2f}", 17, NAVY, True, anchor="ra")
        t(d, (xa + 10, yy - 23), f"{after:.2f}", 17, GREEN, True)
        t(d, (right + 18, yy), f"+{after-before:.2f}", 20, GREEN, True, anchor="lm")
    d.ellipse((left, y0 + 13, left + 18, y0 + 31), fill=WHITE, outline=NAVY, width=3)
    t(d, (left + 31, y0 + 22), "Original official detector — no adaptation", 18, NAVY, True, anchor="lm")
    d.ellipse((left + 500, y0 + 12, left + 520, y0 + 32), fill=GREEN)
    t(d, (left + 533, y0 + 22), "Retrained/adapted detector — 3 epochs", 18, GREEN, True, anchor="lm")


def delta_bars(d, box, items):
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 420, x1 - 105, y0 + 45, y1 - 70
    xmin, xmax = -25, 0
    for tick in (-25, -20, -15, -10, -5, 0):
        xx = left + (tick - xmin) / (xmax - xmin) * (right - left)
        d.line((xx, top, xx, bottom), fill=MID if tick == 0 else LINE, width=2 if tick == 0 else 1)
        t(d, (xx, bottom + 18), str(tick), 16, MID, anchor="ma")
    step = (bottom - top) / len(items)
    for i, (label, value, caveat) in enumerate(items):
        yy = top + (i + 0.5) * step
        xv = left + (value - xmin) / (xmax - xmin) * (right - left)
        t(d, (left - 22, yy), label, 19, INK, True, anchor="rm")
        d.rounded_rectangle((xv, yy - 13, right, yy + 13), radius=6, fill=RED)
        t(d, (xv - 12, yy), f"{value:.2f}", 18, RED, True, anchor="rm")
        if caveat:
            t(d, (right + 15, yy), caveat, 15, GOLD, True, anchor="lm")


def percent_bars(d, box, items):
    x0, y0, x1, y1 = box
    left, right, top, bottom = x0 + 350, x1 - 100, y0 + 35, y1 - 65
    xmin, xmax = -30, 0
    for tick in (-30, -20, -10, 0):
        xx = left + (tick - xmin) / (xmax - xmin) * (right - left)
        d.line((xx, top, xx, bottom), fill=MID if tick == 0 else LINE, width=2 if tick == 0 else 1)
        t(d, (xx, bottom + 18), f"{tick}%", 16, MID, anchor="ma")
    step = (bottom - top) / len(items)
    for i, (label, value) in enumerate(items):
        yy = top + (i + 0.5) * step
        xv = left + (value - xmin) / (xmax - xmin) * (right - left)
        t(d, (left - 18, yy), label, 18, INK, True, anchor="rm")
        d.rounded_rectangle((xv, yy - 11, right, yy + 11), radius=5, fill=GREEN)
        t(d, (xv - 10, yy), f"{value:.2f}%", 17, GREEN, True, anchor="rm")


def code_block(d, box, lines, title=None, highlights=None, size=20):
    x0, y0, x1, y1 = box
    d.rounded_rectangle(box, radius=9, fill="#111827", outline="#334155", width=2)
    if title:
        t(d, (x0 + 24, y0 + 18), title, 19, "#93C5FD", True)
        start_y = y0 + 65
    else:
        start_y = y0 + 24
    regular = ImageFont.truetype(MONO, size)
    bold = ImageFont.truetype(MONO_BOLD, size)
    line_h = size + 12
    for i, line in enumerate(lines):
        yy = start_y + i * line_h
        active = highlights and i in highlights
        if active:
            d.rounded_rectangle((x0 + 14, yy - 4, x1 - 14, yy + line_h - 4), radius=4, fill="#243047")
        d.text((x0 + 24, yy), line, font=bold if active else regular, fill="#FBBF24" if active else "#E5E7EB")


def flow_box(d, box, number, title, body, fill=WHITE, accent=BLUE):
    x0, y0, x1, _ = box
    b.panel(d, box, fill=fill)
    d.rounded_rectangle((x0 + 22, y0 + 20, x0 + 70, y0 + 68), radius=7, fill=accent)
    t(d, (x0 + 46, y0 + 44), str(number), 22, WHITE, True, anchor="mm")
    t(d, (x0 + 87, y0 + 23), title, 23, NAVY, True, max_width=x1 - x0 - 112, spacing=4)
    t(d, (x0 + 24, y0 + 94), body, 20, INK, max_width=x1 - x0 - 48, spacing=6)


def arrow(d, start, end, color=BLUE):
    d.line((start[0], start[1], end[0], end[1]), fill=color, width=5)
    ex, ey = end
    d.polygon(((ex, ey), (ex - 17, ey - 10), (ex - 17, ey + 10)), fill=color)


def input_label(item):
    names = {
        "baseline N": "baseline N",
        "baseline M": "baseline M",
        "PU-GCN direct 4N": "direct 4N",
        "PU-GCN direct 4M": "direct 4M",
        "observed N + predicted 3N": "observed N + predicted 3N",
        "observed M + predicted 3M": "observed M + predicted 3M",
    }
    weights = "unadapted" if item["weights"] == "official" else "adapted"
    return f"{names[item['input']]} / {weights}"


def build_slides():
    SLIDES.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1. Cover
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 28, H), fill=NAVY)
    t(d, (110, 82), "PU-GCN × KITTI | COMPLETED RERUN", 23, NAVY, True)
    t(d, (110, 205), "Full-Validation Detector\nAdaptation Results", 66, INK, True, max_width=1600, spacing=14)
    t(d, (110, 440), "Original unadapted detectors vs 3-epoch adapted detectors", 32, MID, max_width=1600)
    d.line((110, 565, 1790, 565), fill=LINE, width=2)
    t(d, (110, 625), "Primary result", 23, NAVY, True)
    t(d, (110, 674), "Adaptation recovers a substantial part of the PU-GCN loss,\nbut no tested adapted PU-GCN arm reaches its reference baseline.", 40, INK, True, max_width=1570, spacing=11)
    t(d, (110, 930), "KITTI validation: 3,769 frames | Car 3D AP_R40 | PointRCNN + CenterPoint", 22, MID)
    t(d, (110, 995), "Evidence freeze: 10 September 2026 | Deck: 13 September 2026", 18, MID)
    paths.append(save(im, 1, "cover"))

    # 2. Terminology and direct answer
    im, d = base("Terminology first: PU-GCN was not retrained on KITTI", "Interpretation", 2, "Evidence: reports/protocol_and_code_en.md §§0,1,7,8; adaptation_protocol.json files")
    b.panel(d, (100, 235, 900, 920), "What 'retrained' means in this deck", fill=PALE_GREEN)
    b.bullets(d, 145, 325, [
        "PU-GCN remains frozen at the released PU1K model-100 checkpoint.",
        "PointRCNN and CenterPoint are adapted from their official pretrained checkpoints.",
        "The comparison is therefore: original official detector weights vs detector weights adapted for 3 epochs on the matching PU-GCN input.",
        "Line B also includes schedule-matched adapted baseline detectors.",
    ], 700, 25, 22)
    b.panel(d, (950, 235, 1815, 920), "Direct answer", fill=PALE_BLUE)
    t(d, (995, 320), "Did adaptation help?", 24, NAVY, True)
    t(d, (995, 365), "Yes — all 6 paired PU-GCN arms improved.", 31, GREEN, True, max_width=730)
    t(d, (995, 500), "Did the adapted models beat baseline?", 24, NAVY, True)
    t(d, (995, 545), "No — the remaining gaps are 4.58–21.72 AP.", 31, RED, True, max_width=730)
    t(d, (995, 700), "Did 3 epochs converge?", 24, NAVY, True)
    t(d, (995, 745), "Training loss decreased; validation convergence was not demonstrated.", 29, GOLD, True, max_width=730, spacing=8)
    paths.append(save(im, 2, "terminology"))

    # 3. Scope
    im, d = base("Completed scope and acceptance criteria", "Scope", 3, "Evidence: current_progress.json; full_val_detector_matrix.csv; strict-4× audits")
    metric_card(d, (100, 235, 500, 505), "3,712", "Adaptation frames", "Full KITTI train split for every trained arm", NAVY, PALE_BLUE)
    metric_card(d, (535, 235, 935, 505), "3,769", "Validation frames", "Complete Line A and Line B generation", BLUE, PALE_BLUE)
    metric_card(d, (970, 235, 1370, 505), "20 / 20", "Evaluation arms", "Every arm: PASS and frame_count=3,769", GREEN, PALE_GREEN)
    metric_card(d, (1405, 235, 1815, 505), "AP_R40", "Common metric", "Car 3D/BEV; CenterPoint also has 3 classes", NAVY, PALE_BLUE)
    b.panel(d, (100, 570, 1815, 920), "Matrix coverage", fill=WHITE)
    b.table(d, (135, 650, 1780, 870), ["Detector", "Arms", "Official / unadapted", "Adapted", "Classes"], [
        ["PointRCNN", "11", "6", "5", "Car"],
        ["CenterPoint", "9", "6", "3", "Car / Pedestrian / Cyclist"],
        ["Total", "20", "12", "8", "All completed outputs"],
    ], [0.23,0.14,0.22,0.14,0.27], row_h=55, size=22, align_right={1,2,3}, highlights={2: PALE_GREEN})
    paths.append(save(im, 3, "scope"))

    # 4. Explicit original vs adapted comparison
    pairs = [
        ("PointRCNN A · direct", row("PointRCNN", "line_a_pugcn_direct_official")["3d_moderate"], row("PointRCNN", "line_a_pugcn_direct_adapted")["3d_moderate"]),
        ("PointRCNN A · observed-first", row("PointRCNN", "line_a_pugcn_observed_first_official")["3d_moderate"], row("PointRCNN", "line_a_pugcn_observed_first_adapted")["3d_moderate"]),
        ("PointRCNN B · direct", row("PointRCNN", "line_b_pugcn_direct_official")["3d_moderate"], row("PointRCNN", "line_b_pugcn_direct_adapted")["3d_moderate"]),
        ("PointRCNN B · observed-first", row("PointRCNN", "line_b_pugcn_observed_first_official")["3d_moderate"], row("PointRCNN", "line_b_pugcn_observed_first_adapted")["3d_moderate"]),
        ("CenterPoint A · observed-first", row("CenterPoint", "line_a_pugcn_observed_first_official")["3d_moderate"], row("CenterPoint", "line_a_pugcn_observed_first_adapted")["3d_moderate"]),
        ("CenterPoint B · observed-first", row("CenterPoint", "line_b_pugcn_observed_first_official")["3d_moderate"], row("CenterPoint", "line_b_pugcn_observed_first_adapted")["3d_moderate"]),
    ]
    im, d = base("Direct comparison: original unadapted vs retrained/adapted", "Main Comparison", 4, "Same detector + line + PU-GCN input; only detector weights change. Car 3D Moderate AP_R40.")
    paired_comparison(d, (100, 225, 1815, 920), pairs)
    paths.append(save(im, 4, "unadapted_vs_adapted"))

    # 5. PointRCNN A
    names = ["line_a_baseline_official", "line_a_pugcn_direct_official", "line_a_pugcn_direct_adapted", "line_a_pugcn_observed_first_official", "line_a_pugcn_observed_first_adapted"]
    vals = [row("PointRCNN", name)["3d_moderate"] for name in names]
    im, d = base("PointRCNN · Line A: adaptation narrows the loss", "Results", 5, "Car 3D Moderate AP_R40; source: full_val_detector_matrix.csv")
    vertical_bars(d, (100, 230, 1300, 915), ["Baseline\nofficial", "Direct\nunadapted", "Direct\nadapted", "Obs-first\nunadapted", "Obs-first\nadapted"], vals, [NAVY,"#D08C84",RED,"#7AB29A",GREEN])
    b.panel(d, (1350, 240, 1815, 525), "Adaptation gain", fill=PALE_GREEN)
    t(d, (1390, 335), f"Direct: +{vals[2]-vals[1]:.2f} AP", 29, GREEN, True)
    t(d, (1390, 405), f"Observed-first: +{vals[4]-vals[3]:.2f} AP", 29, GREEN, True)
    b.panel(d, (1350, 575, 1815, 915), "Reference-baseline gap", fill=PALE_GOLD)
    t(d, (1390, 655), f"Official baseline: {vals[0]:.2f}", 27, NAVY, True)
    t(d, (1390, 720), f"Best PU-GCN: {vals[4]:.2f}", 27, GREEN, True)
    t(d, (1390, 785), f"Gap: {vals[4]-vals[0]:.2f} AP", 30, RED, True)
    t(d, (1390, 850), "No schedule-matched adapted original-N baseline exists for Line A; this gap changes both input and weights.", 18, GOLD, max_width=370, spacing=4)
    paths.append(save(im, 5, "pointrcnn_a"))

    # 6. PointRCNN B
    names = ["line_b_baseline_official", "line_b_baseline_adapted", "line_b_pugcn_direct_official", "line_b_pugcn_direct_adapted", "line_b_pugcn_observed_first_official", "line_b_pugcn_observed_first_adapted"]
    vals = [row("PointRCNN", name)["3d_moderate"] for name in names]
    im, d = base("PointRCNN · Line B: the largest recovery is observed-first", "Results", 6, "Car 3D Moderate AP_R40; source: full_val_detector_matrix.csv")
    vertical_bars(d, (100, 230, 1370, 915), ["Baseline\nunadapted", "Baseline\nadapted", "Direct\nunadapted", "Direct\nadapted", "Obs-first\nunadapted", "Obs-first\nadapted"], vals, [NAVY,BLUE,"#D08C84",RED,"#7AB29A",GREEN])
    b.panel(d, (1410, 240, 1815, 530), "Adaptation gain", fill=PALE_GREEN)
    t(d, (1450, 325), f"Direct: +{vals[3]-vals[2]:.2f}", 29, GREEN, True)
    t(d, (1450, 395), f"Observed-first: +{vals[5]-vals[4]:.2f}", 29, GREEN, True)
    b.panel(d, (1410, 580, 1815, 915), "Schedule-matched baseline", fill=PALE_GOLD)
    t(d, (1450, 660), f"Adapted baseline: {vals[1]:.2f}", 25, NAVY, True)
    t(d, (1450, 725), f"Best PU-GCN: {vals[5]:.2f}", 25, GREEN, True)
    t(d, (1450, 790), f"Gap: {vals[5]-vals[1]:.2f} AP", 29, RED, True)
    paths.append(save(im, 6, "pointrcnn_b"))

    # 7. CenterPoint A
    names = ["line_a_baseline_official", "line_a_pugcn_direct_official", "line_a_pugcn_observed_first_official", "line_a_pugcn_observed_first_adapted"]
    vals = [row("CenterPoint", name)["3d_moderate"] for name in names]
    im, d = base("CenterPoint · Line A: observed-first approaches baseline", "Results", 7, "Car 3D Moderate AP_R40; source: full_val_detector_matrix.csv")
    vertical_bars(d, (100, 230, 1290, 915), ["Baseline\nofficial", "Direct\nunadapted", "Obs-first\nunadapted", "Obs-first\nadapted"], vals, [NAVY,RED,"#7AB29A",GREEN])
    b.panel(d, (1340, 240, 1815, 530), "Observed-first adaptation", fill=PALE_GREEN)
    t(d, (1380, 325), f"{vals[2]:.2f} → {vals[3]:.2f}", 33, INK, True)
    t(d, (1380, 405), f"Gain: +{vals[3]-vals[2]:.2f} AP", 28, GREEN, True)
    b.panel(d, (1340, 580, 1815, 915), "Remaining gap", fill=PALE_GOLD)
    t(d, (1380, 665), f"Official baseline: {vals[0]:.2f}", 27, NAVY, True)
    t(d, (1380, 730), f"Adapted: {vals[3]:.2f}", 27, GREEN, True)
    t(d, (1380, 795), f"Gap: {vals[3]-vals[0]:.2f} AP", 29, RED, True)
    t(d, (1380, 850), "CenterPoint direct-4N adapted was not run and is not imputed.", 18, GOLD, max_width=380)
    paths.append(save(im, 7, "centerpoint_a"))

    # 8. CenterPoint B
    names = ["line_b_baseline_official", "line_b_baseline_adapted", "line_b_pugcn_direct_official", "line_b_pugcn_observed_first_official", "line_b_pugcn_observed_first_adapted"]
    vals = [row("CenterPoint", name)["3d_moderate"] for name in names]
    im, d = base("CenterPoint · Line B: adaptation recovers most of the loss", "Results", 8, "Car 3D Moderate AP_R40; source: full_val_detector_matrix.csv")
    vertical_bars(d, (100, 230, 1320, 915), ["Baseline\nunadapted", "Baseline\nadapted", "Direct\nunadapted", "Obs-first\nunadapted", "Obs-first\nadapted"], vals, [NAVY,BLUE,RED,"#7AB29A",GREEN])
    b.panel(d, (1370, 240, 1815, 530), "Observed-first adaptation", fill=PALE_GREEN)
    t(d, (1410, 325), f"{vals[3]:.2f} → {vals[4]:.2f}", 31, INK, True)
    t(d, (1410, 405), f"Gain: +{vals[4]-vals[3]:.2f} AP", 27, GREEN, True)
    b.panel(d, (1370, 580, 1815, 915), "Schedule-matched baseline", fill=PALE_GOLD)
    t(d, (1410, 665), f"Adapted baseline: {vals[1]:.2f}", 25, NAVY, True)
    t(d, (1410, 730), f"Adapted PU-GCN: {vals[4]:.2f}", 25, GREEN, True)
    t(d, (1410, 795), f"Gap: {vals[4]-vals[1]:.2f} AP", 29, RED, True)
    paths.append(save(im, 8, "centerpoint_b"))

    # 9. Gaps
    gap_items = [
        ("PointRCNN A · direct adapted", row("PointRCNN", "line_a_pugcn_direct_adapted")["delta_3d_moderate"], "A*"),
        ("PointRCNN A · observed-first adapted", row("PointRCNN", "line_a_pugcn_observed_first_adapted")["delta_3d_moderate"], "A*"),
        ("PointRCNN B · direct adapted", row("PointRCNN", "line_b_pugcn_direct_adapted")["delta_3d_moderate"], ""),
        ("PointRCNN B · observed-first adapted", row("PointRCNN", "line_b_pugcn_observed_first_adapted")["delta_3d_moderate"], ""),
        ("CenterPoint A · observed-first adapted", row("CenterPoint", "line_a_pugcn_observed_first_adapted")["delta_3d_moderate"], "A*"),
        ("CenterPoint B · observed-first adapted", row("CenterPoint", "line_b_pugcn_observed_first_adapted")["delta_3d_moderate"], ""),
    ]
    im, d = base("Adaptation helps, but 4.58–21.72 AP remains", "Baseline Gap", 9, "Δ Car 3D Moderate AP_R40 vs same-weight-family baseline, except Line A rows marked A*.")
    delta_bars(d, (100, 235, 1815, 865), gap_items)
    b.panel(d, (100, 875, 1815, 930), fill=PALE_GOLD)
    t(d, (130, 891), "A*: no schedule-matched adapted original-N baseline was trained; the displayed Line A delta uses the official baseline and changes both input and weights.", 18, GOLD, True, max_width=1640)
    paths.append(save(im, 9, "remaining_gaps"))

    # 10. What was trained
    im, d = base("What was frozen and what was adapted", "Training Design", 10, "Evidence: PU-GCN checkpoint file; run_*_full_train.py; adaptation_protocol.json")
    boxes = [(100, 285, 430, 700), (500, 285, 830, 700), (900, 285, 1230, 700), (1300, 285, 1815, 700)]
    flow_box(d, boxes[0], 1, "Frozen PU-GCN", "Released PU1K model-100. No KITTI PU-GCN optimization in this experiment.", PALE_BLUE)
    flow_box(d, boxes[1], 2, "Matching inputs", "Direct 4N/4M or observed-first N+3N/M+3M over all 3,712 train frames.", WHITE)
    flow_box(d, boxes[2], 3, "Detector init", "Official PointRCNN checkpoint or official 80-epoch CenterPoint KITTI checkpoint.", WHITE)
    flow_box(d, boxes[3], 4, "3-epoch adaptation", "PointRCNN: RPN 3 epochs then offline RCNN 3 epochs. CenterPoint: 3 epochs. Only epoch 3 saved.", PALE_GREEN, GREEN)
    for i in range(3):
        arrow(d, (boxes[i][2] + 8, 490), (boxes[i + 1][0] - 8, 490))
    b.panel(d, (100, 770, 1815, 925), fill=PALE_GOLD)
    t(d, (140, 810), "Interpretation boundary", 23, GOLD, True)
    t(d, (140, 855), "These are short detector-adaptation runs from pretrained checkpoints, not from-scratch training and not evidence that PU-GCN itself was retrained.", 26, INK, True, max_width=1600)
    paths.append(save(im, 10, "what_was_trained"))

    # 11. Why 3 epochs
    im, d = base("Why exactly 3 epochs? The recorded evidence and its limit", "Epoch Choice", 11, "Evidence: script defaults, 13 adaptation_protocol.json files, convergence_audit.md §Interpretation")
    b.panel(d, (100, 240, 645, 920), "Recorded facts", fill=PALE_GREEN)
    b.bullets(d, 140, 330, [
        "Both training wrappers set the default epoch count to 3.",
        "Every executed adaptation protocol records epochs=3.",
        "The same short schedule was applied across arms for controlled comparison.",
        "Only epoch 3 was checkpointed and evaluated.",
    ], 455, 23, 21)
    b.panel(d, (690, 240, 1235, 920), "Operational interpretation", fill=PALE_BLUE)
    b.bullets(d, 730, 330, [
        "The detectors start from strong pretrained checkpoints, so the run is adaptation rather than full retraining.",
        "Three epochs function as a fixed, compute-bounded adaptation budget.",
        "This is a reasonable interpretation of the implementation, not a documented empirical selection study.",
    ], 455, 23, 21)
    b.panel(d, (1280, 240, 1815, 920), "What cannot be claimed", fill=PALE_GOLD)
    b.bullets(d, 1320, 330, [
        "No experiment record explains why 3 was chosen instead of 6 or 10.",
        "No validation curve selected epoch 3.",
        "No early-stopping rule or plateau criterion was used.",
        "Therefore 3 epochs is a pre-fixed schedule, not a demonstrated optimum.",
    ], 445, 23, 21)
    paths.append(save(im, 11, "why_three_epochs"))

    # 12. Epoch code evidence
    im, d = base("Code evidence: 3 epochs and epoch-3-only checkpointing", "Code", 12, "Exact workspace lines: run_pointrcnn_full_train_stage.py:31,151–165; run_centerpoint_full_train.py:29,106–124")
    code_block(d, (100, 245, 930, 865), [
        "# PointRCNN wrapper",
        "parser.add_argument(\"--epochs\",",
        "                    type=int, default=3)",
        "...",
        "\"--epochs\", str(args.epochs),",
        "\"--ckpt_save_interval\",",
        "str(args.epochs),",
    ], "scripts/run_pointrcnn_full_train_stage.py", {2,4,5,6}, 20)
    code_block(d, (990, 245, 1815, 865), [
        "# CenterPoint wrapper",
        "parser.add_argument(\"--epochs\",",
        "                    type=int, default=3)",
        "...",
        "\"--epochs\", str(args.epochs),",
        "\"--ckpt_save_interval\",",
        "str(args.epochs),",
        "\"--num_epochs_to_eval\", \"0\",",
    ], "scripts/run_centerpoint_full_train.py", {2,4,5,6,7}, 19)
    t(d, (100, 900), "Observed artifacts: 10 PointRCNN epoch-3 checkpoints + 3 CenterPoint epoch-3 checkpoints; no epoch-1/2 checkpoints exist in these runs.", 21, INK, True, max_width=1700)
    paths.append(save(im, 12, "epoch_code"))

    # 13. PointRCNN convergence table
    im, d = base("PointRCNN: complete 3-epoch optimization audit", "Convergence", 13, "Median loss is used because RCNN logs contain extreme spikes; outliers are counts >100× epoch median.")
    labels = {"line_a_pugcn":"A direct", "line_a_pugcn_observed_first":"A obs-first", "line_b_baseline":"B baseline", "line_b_pugcn":"B direct", "line_b_pugcn_observed_first":"B obs-first"}
    table_rows = []
    for arm in ("line_a_pugcn", "line_a_pugcn_observed_first", "line_b_baseline", "line_b_pugcn", "line_b_pugcn_observed_first"):
        for stage in ("rpn", "rcnn"):
            epochs = CONVERGENCE["PointRCNN"][arm][stage]["epochs"]
            vals = [float(item["loss_median"]) for item in epochs]
            change = (vals[-1] / vals[0] - 1) * 100
            outliers = sum(int(item["extreme_outlier_count_gt_100x_median"]) for item in epochs)
            table_rows.append([labels[arm], stage.upper(), f"{vals[0]:.4f}", f"{vals[1]:.4f}", f"{vals[2]:.4f}", f"{change:.2f}%", str(outliers)])
    b.table(d, (100, 230, 1820, 900), ["Arm", "Stage", "Epoch 1", "Epoch 2", "Epoch 3", "E1→E3", "Outliers"], table_rows,
            [0.25,0.12,0.14,0.14,0.14,0.14,0.07], row_h=59, size=19, align_right={2,3,4,5,6})
    paths.append(save(im, 13, "pointrcnn_convergence"))

    # 14. CenterPoint convergence
    im, d = base("CenterPoint: complete 3-epoch optimization audit", "Convergence", 14, "Mean training loss from convergence_audit.json; final learning rate is recorded for every arm.")
    labels = {"line_a_pugcn":"A observed-first", "line_b_pugcn":"B observed-first", "line_b_baseline":"B baseline"}
    cp_rows = []
    reduction_items = []
    for arm in ("line_a_pugcn", "line_b_pugcn", "line_b_baseline"):
        epochs = CONVERGENCE["CenterPoint"][arm]["epochs"]
        vals = [float(item["loss_mean"]) for item in epochs]
        change = (vals[-1] / vals[0] - 1) * 100
        cp_rows.append([labels[arm], f"{vals[0]:.2f}", f"{vals[1]:.2f}", f"{vals[2]:.2f}", f"{change:.2f}%", f"{float(epochs[-1]['learning_rate']):.3e}"])
        reduction_items.append((labels[arm], change))
    b.table(d, (100, 255, 1040, 565), ["Arm", "E1", "E2", "E3", "E1→E3", "Final LR"], cp_rows,
            [0.34,0.12,0.12,0.12,0.15,0.15], row_h=62, size=20, align_right={1,2,3,4,5})
    b.panel(d, (1100, 255, 1815, 565), "Loss reduction", fill=PALE_GREEN)
    percent_bars(d, (1115, 300, 1790, 545), reduction_items)
    b.panel(d, (100, 660, 1815, 920), fill=PALE_GOLD)
    t(d, (145, 710), "Result", 23, GOLD, True)
    t(d, (145, 760), "All three mean-loss series decrease from epoch 1 to epoch 3.", 31, GREEN, True)
    t(d, (145, 830), "This proves optimization progress; it does not prove that validation AP has plateaued or that epoch 3 is optimal.", 27, INK, True, max_width=1580)
    paths.append(save(im, 14, "centerpoint_convergence"))

    # 15. Convergence verdict
    all_reductions = []
    for arm, parts in CONVERGENCE["PointRCNN"].items():
        for stage, data in parts.items():
            vals = [float(item["loss_median"]) for item in data["epochs"]]
            all_reductions.append((f"{arm.replace('line_', '').replace('_pugcn', '').replace('_', ' ')} · {stage}", (vals[-1]/vals[0]-1)*100))
    for arm, data in CONVERGENCE["CenterPoint"].items():
        vals = [float(item["loss_mean"]) for item in data["epochs"]]
        all_reductions.append((f"CP · {arm.replace('line_', '').replace('_pugcn', '').replace('_', ' ')}", (vals[-1]/vals[0]-1)*100))
    im, d = base("Convergence verdict: all losses fell; validation convergence is unproven", "Convergence", 15, "Evidence: convergence_audit.json + checkpoint inventory + wrapper checkpoint/evaluation settings")
    metric_card(d, (100, 235, 500, 505), "13 / 13", "Training-loss series decreased", "10 PointRCNN stages + 3 CenterPoint arms", GREEN, PALE_GREEN)
    metric_card(d, (535, 235, 935, 505), "13 / 13", "Only epoch-3 checkpoints", "No epoch-1 or epoch-2 checkpoints retained", GOLD, PALE_GOLD)
    metric_card(d, (970, 235, 1370, 505), "0", "Per-epoch validation AP points", "No AP curve exists for epoch selection", RED, PALE_RED)
    metric_card(d, (1405, 235, 1815, 505), "NO", "Convergence demonstrated?", "Schedule completed; optimum/plateau not established", RED, PALE_RED)
    b.panel(d, (100, 575, 1815, 925), "The only defensible statement", fill=WHITE)
    t(d, (145, 655), "The three-epoch adaptation schedule completed successfully and every recorded training-loss series decreased.", 31, INK, True, max_width=1580, spacing=9)
    t(d, (145, 785), "Because only epoch 3 was saved and there is no per-epoch validation AP, the current records cannot establish convergence or select epoch 3 as the optimum.", 31, RED, True, max_width=1580, spacing=9)
    paths.append(save(im, 15, "convergence_verdict"))

    # 16. 3N principle
    im, d = base("What '3N' means: observed-first composition", "3N Principle", 16, "Evidence: prepare_centerpoint_observed_first_train.py:52–75; protocol_and_code_en.md §§2–3")
    boxes = [(100, 290, 430, 700), (500, 290, 850, 700), (920, 290, 1270, 700), (1340, 290, 1815, 700)]
    flow_box(d, boxes[0], 1, "Observed cloud", "Line A: N original rows.\nLine B: M=floor(N/4) downsampled rows.", PALE_BLUE)
    flow_box(d, boxes[1], 2, "Strict PU-GCN cloud", "PU-GCN patch outputs are merged, then uniformly sampled without replacement to exactly 4N or 4M rows.", WHITE)
    flow_box(d, boxes[2], 3, "Select generated rows", "Choose exactly 3N or 3M rows from the strict PU-GCN cloud using a deterministic per-frame random seed.", PALE_GOLD, GOLD)
    flow_box(d, boxes[3], 4, "Concatenate", "Final detector input = all observed rows first + selected generated rows. Total = 4N or 4M.", PALE_GREEN, GREEN)
    for i in range(3):
        arrow(d, (boxes[i][2] + 8, 495), (boxes[i+1][0] - 8, 495))
    b.panel(d, (100, 775, 1815, 925), fill=PALE_RED)
    t(d, (140, 815), "Important", 23, RED, True)
    t(d, (140, 855), "3N is not 'the three new children per input anchor.' It is a second frame-level sample of 3N rows from the already constructed strict 4N PU-GCN output.", 27, INK, True, max_width=1600)
    paths.append(save(im, 16, "three_n_principle"))

    # 17. 3N code
    im, d = base("Exact 3N/3M selection code", "Code", 17, "scripts/prepare_centerpoint_observed_first_train.py:14–21, 52–75; SHA-256 in source_manifest.json")
    code_block(d, (100, 230, 1815, 850), [
        "BASE_SEED = 20260718",
        "",
        "def stable_seed(*parts):",
        "    digest = hashlib.sha256(\"|\".join(str(p) for p in parts)",
        "                            .encode(\"utf-8\")).digest()",
        "    return int.from_bytes(digest[:8], \"little\") & 0xFFFFFFFF",
        "",
        "expected = 4 * observed.shape[0]",
        "assert predicted.shape[0] == expected",
        "seed = stable_seed(BASE_SEED, \"e1\", reference_token, frame)",
        "rng = np.random.default_rng(seed)",
        "selected = rng.choice(expected, size=3 * observed.shape[0],",
        "                      replace=False)",
        "final = np.concatenate((observed, predicted[selected]), axis=0)",
    ], "Deterministic uniform sampling without replacement", {0,5,7,8,9,10,11,12,13}, 18)
    t(d, (100, 885), "Line A reference_token: original_x4_pu_gcn | Line B: downsampled_x4_pu_gcn", 21, NAVY, True)
    t(d, (100, 930), "The SHA-256-derived seed makes selection repeatable per frame and distinct across lines/reference tokens.", 21, INK)
    paths.append(save(im, 17, "three_n_code"))

    # 18. What selection is not
    im, d = base("Selection rule: what it uses and what it does not use", "3N Semantics", 18, "Direct code inspection: rng.choice(..., replace=False); no geometry/score ranking in the selection path")
    b.panel(d, (100, 245, 900, 920), "The actual rule", fill=PALE_GREEN)
    b.bullets(d, 145, 335, [
        "Candidate set: every row in the strict 4N/4M PU-GCN output.",
        "Sample size: exactly 3N/3M.",
        "Distribution: uniform over row indices.",
        "Replacement: false.",
        "Determinism: stable hash of base seed, token, and frame ID.",
        "Output order: all observed rows, then selected generated rows.",
    ], 700, 24, 19)
    b.panel(d, (950, 245, 1815, 920), "Not used by this rule", fill=PALE_RED)
    b.bullets(d, 995, 335, [
        "No confidence score or uncertainty.",
        "No distance-to-observed ranking.",
        "No FPS, curvature, voxel quota, or foreground weighting.",
        "No coordinate deduplication.",
        "No guarantee that selected generated coordinates are geometrically distinct from observed coordinates.",
        "'Generated' is a provenance label, not a set-difference operation.",
    ], 760, 24, 19)
    paths.append(save(im, 18, "three_n_semantics"))

    # 19. 3N audit evidence
    im, d = base("3N/3M audit evidence: every validation frame passes", "Evidence", 19, "Sources: inputs/line_[ab]_pugcn_observed_first_manifest.csv")
    metric_card(d, (100, 235, 500, 505), "3,769 / 3,769", "Line A rows audited", "PASS; predicted=4N, selected=3N, output=4N", GREEN, PALE_GREEN)
    metric_card(d, (535, 235, 935, 505), "3,769 / 3,769", "Line B rows audited", "PASS; predicted=4M, selected=3M, output=4M", GREEN, PALE_GREEN)
    metric_card(d, (970, 235, 1370, 505), "TRUE", "Observed prefix exact", "All frames, both lines", NAVY, PALE_BLUE)
    metric_card(d, (1405, 235, 1815, 505), "TRUE", "Generated suffix exact", "All frames, both lines", NAVY, PALE_BLUE)
    b.table(d, (100, 600, 1820, 805), ["Example", "Observed", "Predicted", "Selected", "Output", "Seed", "Status"], [
        ["A · frame 000001", "120,268", "481,072", "360,804", "481,072", "1091044475", "PASS"],
        ["B · frame 000001", "30,067", "120,268", "90,201", "120,268", "2918483960", "PASS"],
    ], [0.24,0.13,0.13,0.13,0.13,0.16,0.08], row_h=62, size=20, align_right={1,2,3,4,5})
    t(d, (100, 865), "Each manifest row also stores selected_indices_sha256 and output_float32_sha256, enabling byte-level reproduction checks.", 22, INK, True, max_width=1700)
    paths.append(save(im, 19, "three_n_audit"))

    # 20. PU-GCN and strict principle
    im, d = base("From a PU-GCN patch to strict frame-level 4×", "Mechanism", 20, "Code: generator.py:228–266; model.py:251–258; strict_x4_from_merged_raw.py:118–130,194–220")
    b.panel(d, (100, 230, 800, 900), "Network-level principle", fill=PALE_BLUE)
    b.bullets(d, 145, 315, [
        "Each 2,048-point patch is centered and normalized to a unit sphere.",
        "The graph network extracts features and NodeShuffle upsamples them.",
        "A coordinate head predicts 3D residuals.",
        "Each input anchor is tiled four times and the residuals are added, producing 8,192 rows per patch.",
        "Predictions are transformed back to metric coordinates.",
    ], 605, 23, 20)
    code_block(d, (850, 230, 1815, 560), [
        "outputs = tf.squeeze(coord, [2])",
        "outputs += tf.reshape(tf.tile(",
        "    tf.expand_dims(inputs, 2),",
        "    [1, 1, self.up_ratio, 1]), ...)",
    ], "PU-GCN residual output", {0,1,2,3}, 18)
    code_block(d, (850, 600, 1815, 900), [
        "target_points = input_count * 4",
        "idx = rng.choice(raw_count,",
        "                 size=target_points,",
        "                 replace=False)",
        "strict_xyz = raw_xyz[idx]",
        "intensity = nearest_observed_intensity(strict_xyz)",
    ], "Strict frame-level 4× adapter", {0,1,2,3,4,5}, 18)
    paths.append(save(im, 20, "pugcn_mechanism"))

    # 21. Complete PointRCNN
    im, d = base("Complete PointRCNN Car results", "Complete Results", 21, "3,769 frames; all 11 arms PASS; 3D/BEV AP_R40")
    rows_pr = [item for item in MATRIX if item["detector"] == "PointRCNN"]
    data = [[item["line"], input_label(item), f"{item['3d_easy']:.2f}", f"{item['3d_moderate']:.2f}", f"{item['3d_hard']:.2f}", f"{item['bev_moderate']:.2f}", "PASS"] for item in rows_pr]
    b.table(d, (100, 225, 1820, 920), ["Line", "Input / weights", "3D E", "3D M", "3D H", "BEV M", "Status"], data,
            [0.07,0.37,0.115,0.115,0.115,0.115,0.095], row_h=53, size=18, align_right={2,3,4,5}, highlights={0:PALE_BLUE,4:PALE_GREEN,5:PALE_BLUE,6:PALE_BLUE,10:PALE_GREEN})
    paths.append(save(im, 21, "pointrcnn_complete"))

    # 22. Complete CenterPoint Car
    im, d = base("Complete CenterPoint Car results", "Complete Results", 22, "3,769 frames; all 9 executed arms PASS; no unexecuted arm is imputed")
    rows_cp = [item for item in MATRIX if item["detector"] == "CenterPoint"]
    data = [[item["line"], input_label(item), f"{item['3d_easy']:.2f}", f"{item['3d_moderate']:.2f}", f"{item['3d_hard']:.2f}", f"{item['bev_moderate']:.2f}", "PASS"] for item in rows_cp]
    b.table(d, (100, 245, 1820, 850), ["Line", "Input / weights", "3D E", "3D M", "3D H", "BEV M", "Status"], data,
            [0.07,0.37,0.115,0.115,0.115,0.115,0.095], row_h=55, size=18, align_right={2,3,4,5}, highlights={0:PALE_BLUE,3:PALE_GREEN,4:PALE_BLUE,5:PALE_BLUE,8:PALE_GREEN})
    b.panel(d, (100, 885, 1820, 940), fill=PALE_GOLD)
    t(d, (130, 900), "CenterPoint direct-adapted was not executed; it is intentionally absent from results and charts.", 19, GOLD, True)
    paths.append(save(im, 22, "centerpoint_complete"))

    # 23. Three classes
    im, d = base("CenterPoint: complete Moderate 3D AP_R40 for all classes", "Complete Results", 23, "Source: all_classes_ap_r40.csv; PointRCNN has no Pedestrian/Cyclist outputs in this matrix")
    data = []
    for item in rows_cp:
        data.append([item["line"], input_label(item), f"{class_row(item['name'],'Car')['moderate']:.2f}", f"{class_row(item['name'],'Pedestrian')['moderate']:.2f}", f"{class_row(item['name'],'Cyclist')['moderate']:.2f}"])
    b.table(d, (100, 245, 1820, 850), ["Line", "Input / weights", "Car", "Pedestrian", "Cyclist"], data,
            [0.08,0.44,0.16,0.16,0.16], row_h=55, size=19, align_right={2,3,4}, highlights={0:PALE_BLUE,3:PALE_GREEN,4:PALE_BLUE,5:PALE_BLUE,8:PALE_GREEN})
    b.panel(d, (100, 885, 1820, 940), fill=PALE_BLUE)
    t(d, (130, 900), "The class-level conclusion is restricted to CenterPoint and is not extrapolated to PointRCNN.", 19, NAVY, True)
    paths.append(save(im, 23, "all_classes"))

    # 24. Hashes and source provenance
    im, d = base("Code provenance: exact files and SHA-256 evidence", "Provenance", 24, "source_manifest.json snapshots 151 source/config/split files; hashes below match the current workspace")
    b.table(d, (100, 235, 1820, 690), ["Evidence item", "Path", "SHA-256"], [
        ["3N selector", "scripts/prepare_centerpoint_observed_first_train.py", "f5e521219f8d49b8fed8d9d40192749378b02a0f54888b92e949d041db1f4f42"],
        ["PointRCNN trainer", "scripts/run_pointrcnn_full_train_stage.py", "4b177d0dd27769a5f562ec45060e09537654965ab94936d9bd2df8f43b43828a"],
        ["CenterPoint trainer", "scripts/run_centerpoint_full_train.py", "a71cbac6817271e53c19828cefbcdbc5ae1ad9d9c9bf5d4e665d16f211801ee6"],
        ["Convergence audit", "reports/convergence_audit.json", "3326d80c8916023936b9be7ad2b27f8469a8c9f32079fe8107ed60e8fadeacca"],
        ["Full result matrix", "reports/full_val_detector_matrix.csv", "68606b1062b93c8ae7ac1965f62e53221df2851eb533883e69ecb3632f3c63bd"],
    ], [0.20,0.37,0.43], row_h=72, size=16)
    b.panel(d, (100, 770, 1820, 925), "Snapshot package", fill=PALE_BLUE)
    t(d, (140, 815), "reports/pugcn_full_val_sources.tar.gz", 23, NAVY, True)
    t(d, (590, 815), "reports/source_manifest.json", 23, NAVY, True)
    t(d, (1010, 815), "reports/experiment_source_full.md", 23, NAVY, True)
    t(d, (140, 865), "The package records byte-exact local code used by the experiment; it does not imply that every file equals the upstream author repository.", 20, INK, max_width=1600)
    paths.append(save(im, 24, "code_provenance"))

    # 25. Reproduction and conclusion
    im, d = base("Evidence index, reproduction entry point, and final verdict", "Conclusion", 25, "Only completed full-validation artifacts are used as final results; pilot256 metrics are excluded.")
    b.panel(d, (100, 235, 930, 900), "Reproduction entry point", fill=PALE_BLUE)
    code_block(d, (140, 320, 890, 480), [
        "cd /home/ra87racy/projects/baseline_detectors/PointRCNN",
        "bash scripts/run_pugcn_detector_adaptation_full_val_20260908.sh all",
    ], None, {0,1}, 17)
    t(d, (140, 545), "Primary evidence", 23, NAVY, True)
    b.bullets(d, 140, 595, [
        "reports/full_val_detector_matrix.csv",
        "reports/all_classes_ap_r40.csv",
        "reports/convergence_audit.json",
        "inputs/line_[ab]_pugcn_observed_first_manifest.csv",
        "reports/protocol_and_code_en.md",
    ], 700, 20, 14)
    b.panel(d, (990, 235, 1815, 900), "Final verdict", fill=WHITE)
    t(d, (1035, 330), "Adaptation effect", 22, NAVY, True)
    t(d, (1035, 375), "Positive in all 6 paired PU-GCN arms", 29, GREEN, True, max_width=690)
    t(d, (1035, 500), "Baseline recovery", 22, NAVY, True)
    t(d, (1035, 545), "Incomplete: every tested adapted PU-GCN arm remains below baseline", 29, RED, True, max_width=690, spacing=8)
    t(d, (1035, 700), "Convergence", 22, NAVY, True)
    t(d, (1035, 745), "Optimization progressed; validation convergence was not demonstrated", 29, GOLD, True, max_width=690, spacing=8)
    paths.append(save(im, 25, "conclusion"))

    return paths


def set_metadata():
    tree = ET.parse(FODP)
    root = tree.getroot()
    ns = {"office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0"}
    meta = root.find("office:meta", ns)
    dc = "http://purl.org/dc/elements/1.1/"
    mn = "urn:oasis:names:tc:opendocument:xmlns:meta:1.0"
    if meta is not None:
        values = {
            f"{{{dc}}}title": "PU-GCN Full-Validation Detector Adaptation Evidence",
            f"{{{dc}}}subject": "Original unadapted vs 3-epoch adapted detectors, convergence, 3N selection and code evidence",
            f"{{{dc}}}creator": "Jiahuai",
            f"{{{dc}}}description": "Evidence-complete English deck for the completed 3,769-frame KITTI rerun.",
            f"{{{mn}}}initial-creator": "Jiahuai",
            f"{{{mn}}}creation-date": "2026-09-13T20:00:00+02:00",
            f"{{{dc}}}date": "2026-09-13T20:00:00+02:00",
        }
        for tag, value in values.items():
            element = meta.find(tag)
            if element is not None:
                element.text = value
        keyword_tag = f"{{{mn}}}keyword"
        for element in list(meta.findall(keyword_tag)):
            meta.remove(element)
        for keyword in ("PU-GCN", "KITTI", "PointRCNN", "CenterPoint", "AP_R40", "detector adaptation", "3N selection", "convergence"):
            ET.SubElement(meta, keyword_tag).text = keyword
    tree.write(FODP, encoding="utf-8", xml_declaration=True)


def main():
    paths = build_slides()
    b.assemble_fodp(paths)
    set_metadata()
    print(FODP)


if __name__ == "__main__":
    main()
