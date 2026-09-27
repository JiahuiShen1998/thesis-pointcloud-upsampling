#!/usr/bin/env python3
"""Generate the integrated upsampling failure explanation package.

This exporter intentionally builds on the two existing visualization packages:

* results/interactive_object_crop_visualization_improved_v6
* results/pointrcnn_detection_box_visualization_v1

It does not recompute official KITTI AP. Instead it combines the detector
matching audit, AP provenance audit, and object-crop point-cloud evidence into
thesis-ready dashboards and summary tables.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DET_ROOT = ROOT / "results/pointrcnn_detection_box_visualization_v1"
PC_ROOT = ROOT / "results/interactive_object_crop_visualization_improved_v6"
OUT = ROOT / "results/upsampling_detection_failure_explanation_v2"

METHOD_ORDER = ["EAR", "PU-Net", "PU-GCN", "PDANS", "TULIP", "SPU-PMD"]
FAILURE_ORDER = [
    "Recall degradation",
    "Precision degradation",
    "Localization degradation",
    "Confidence degradation",
    "Mixed failure",
    "No clear degradation",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def rel(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return str(path)


def rel_from_out(path: Path | str, start: Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return rel(p, start)


def h(s) -> str:
    return html.escape("" if s is None else str(s), quote=True)


def fnum(value, default=0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value, default=0) -> int:
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value, digits=3) -> str:
    if value in ("", None):
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "n/a"
    return f"{v:.{digits}f}"


def pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "n/a"


def badge(label: str, kind: str = "neutral") -> str:
    return f'<span class="badge {h(kind)}">{h(label)}</span>'


def page(title: str, body: str, extra_head: str = "") -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{h(title)}</title>
  <style>
    :root {{
      --ink: #18212f;
      --muted: #5b6676;
      --line: #d9dee7;
      --panel: #ffffff;
      --soft: #f5f7fa;
      --accent: #1e6b7a;
      --warn: #a45d00;
      --bad: #b42318;
      --good: #16784c;
      --violet: #5b4aa0;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #eef1f5;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 36px 22px;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
    }}
    main {{ padding: 24px 36px 44px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 20px; letter-spacing: 0; }}
    h3 {{ margin: 18px 0 8px; font-size: 16px; letter-spacing: 0; }}
    p {{ margin: 8px 0; }}
    a {{ color: #0b6272; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ color: #313b4c; background: #f7f9fb; position: sticky; top: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    .two {{ display: grid; grid-template-columns: minmax(280px, 0.9fr) minmax(360px, 1.5fr); gap: 16px; align-items: start; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }}
    .metric {{ display: grid; gap: 4px; }}
    .metric b {{ font-size: 22px; }}
    .muted {{ color: var(--muted); }}
    .badge {{ display: inline-flex; align-items: center; gap: 4px; border-radius: 999px; padding: 3px 8px; font-size: 12px; font-weight: 650; border: 1px solid var(--line); background: #fff; margin: 2px 4px 2px 0; }}
    .ok {{ color: var(--good); border-color: #b9dccb; background: #eef8f3; }}
    .warning {{ color: var(--warn); border-color: #efd199; background: #fff6e7; }}
    .error, .bad {{ color: var(--bad); border-color: #f2b8b5; background: #fff0ef; }}
    .qual {{ color: var(--violet); border-color: #d2caee; background: #f3f0ff; }}
    .neutral {{ color: #445064; background: #f7f9fb; }}
    .callout {{ border-left: 4px solid var(--accent); background: #f0f8fa; padding: 14px 16px; border-radius: 6px; }}
    .warnbox {{ border-left: 4px solid var(--warn); background: #fff8ed; padding: 14px 16px; border-radius: 6px; }}
    .badbox {{ border-left: 4px solid var(--bad); background: #fff2f1; padding: 14px 16px; border-radius: 6px; }}
    .viewer {{ width: 100%; height: 760px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    .links a {{ display: inline-block; margin: 3px 10px 3px 0; }}
    .annotation-list li {{ margin: 8px 0; }}
    .bar {{ height: 12px; background: #dde5ed; border-radius: 999px; overflow: hidden; }}
    .bar span {{ display: block; height: 100%; background: var(--accent); }}
    @media (max-width: 900px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .two {{ grid-template-columns: 1fr; }}
      .viewer {{ height: 560px; }}
    }}
  </style>
  {extra_head}
</head>
<body>
  {body}
</body>
</html>
"""


def table(rows: list[dict], cols: list[tuple[str, str]], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    head = "".join(f"<th>{h(label)}</th>" for _, label in cols)
    body = []
    for row in shown:
        body.append("<tr>" + "".join(f"<td>{row.get(key, '')}</td>" for key, _ in cols) + "</tr>")
    if not body:
        body.append(f"<tr><td colspan='{len(cols)}' class='muted'>No rows available.</td></tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def index_by_key(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], dict]:
    out = {}
    for row in rows:
        out[tuple(row.get(k, "") for k in keys)] = row
    return out


def group_by_key(rows: list[dict], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict]]:
    out = defaultdict(list)
    for row in rows:
        out[tuple(row.get(k, "") for k in keys)].append(row)
    return out


def line_label(line: str) -> str:
    if line.startswith("line_a"):
        return "Line A: original baseline vs original + upsampling"
    if line.startswith("line_b"):
        return "Line B: downsampled baseline vs downsampled + upsampling"
    return line


def baseline_input(line: str) -> str:
    return "original KITTI point cloud" if line.startswith("line_a") else "50% downsampled point cloud"


def upsampled_input(method: str, line: str) -> str:
    base = "original" if line.startswith("line_a") else "downsampled"
    return f"{base} + {method} upsampling"


def provenance_badges(verdict: str) -> tuple[str, str]:
    v = (verdict or "UNKNOWN").upper()
    if v == "OK":
        return "AP-consistent", badge("AP-consistent", "ok")
    if v == "WARNING":
        return "qualitative-only", badge("qualitative-only", "warning") + badge("AP provenance warning", "warning")
    if v == "ERROR":
        return "missing provenance", badge("missing provenance", "error")
    return "provenance unclear", badge("provenance unclear", "qual")


def row_deltas(row: dict) -> dict[str, float]:
    btp = inum(row.get("baseline_gt_matched_prediction_count", row.get("baseline_tp")))
    bfp = inum(row.get("baseline_gt_false_positive_count", row.get("baseline_fp")))
    bfn = inum(row.get("baseline_gt_missed_gt_count", row.get("baseline_fn")))
    utp = inum(row.get("upsampled_gt_matched_prediction_count", row.get("upsampled_tp")))
    ufp = inum(row.get("upsampled_gt_false_positive_count", row.get("upsampled_fp")))
    ufn = inum(row.get("upsampled_gt_missed_gt_count", row.get("upsampled_fn")))
    return {
        "tp_delta": utp - btp,
        "fp_delta": ufp - bfp,
        "fn_delta": ufn - bfn,
        "iou_delta": fnum(row.get("upsampled_gt_mean_iou")) - fnum(row.get("baseline_gt_mean_iou")),
        "score_delta": fnum(row.get("upsampled_gt_mean_tp_score")) - fnum(row.get("baseline_gt_mean_tp_score")),
        "precision_delta": fnum(row.get("upsampled_gt_approx_precision")) - fnum(row.get("baseline_gt_approx_precision")),
        "recall_delta": fnum(row.get("upsampled_gt_approx_recall")) - fnum(row.get("baseline_gt_approx_recall")),
    }


def classify(row: dict, pc_rows: list[dict]) -> tuple[list[str], str]:
    d = row_deltas(row)
    categories = []
    if d["fn_delta"] > 0 or d["tp_delta"] < 0:
        categories.append("Recall degradation")
    if d["fp_delta"] > 0 or d["precision_delta"] < -0.02:
        categories.append("Precision degradation")
    if d["iou_delta"] < -0.02:
        categories.append("Localization degradation")
    if d["score_delta"] < -0.2:
        categories.append("Confidence degradation")
    if len(categories) > 1:
        categories.insert(0, "Mixed failure")
    if not categories:
        categories.append("No clear degradation")

    pc_evidence = []
    if pc_rows:
        avg_inside = sum(fnum(r.get("new_points_inside_bbox_ratio")) for r in pc_rows) / len(pc_rows)
        avg_outside = sum(fnum(r.get("outside_bbox_ratio")) for r in pc_rows) / len(pc_rows)
        avg_count_ratio = sum(fnum(r.get("new_points_ratio")) for r in pc_rows) / len(pc_rows)
        if avg_outside > avg_inside:
            pc_evidence.append("added or changed points are more often outside the labeled object box than inside it")
        if avg_count_ratio < 0.05:
            pc_evidence.append("object-crop density changes are small for the selected anchors")
        if any(fnum(r.get("upsampled_crop_points")) < fnum(r.get("base_crop_points")) for r in pc_rows):
            pc_evidence.append("some upsampled crops contain fewer points than the corresponding baseline crop")
    else:
        pc_evidence.append("no matching v6 crop metrics were available for this frame")
    return categories, "; ".join(pc_evidence)


def conclusion(row: dict, categories: list[str], pc_evidence: str) -> str:
    d = row_deltas(row)
    fragments = []
    if d["fn_delta"] > 0:
        fragments.append(f"FN increases from {inum(row.get('baseline_gt_missed_gt_count'))} to {inum(row.get('upsampled_gt_missed_gt_count'))}")
    if d["fp_delta"] > 0:
        fragments.append(f"FP increases from {inum(row.get('baseline_gt_false_positive_count'))} to {inum(row.get('upsampled_gt_false_positive_count'))}")
    if d["iou_delta"] < -0.005:
        fragments.append(f"mean IoU changes from {fmt(row.get('baseline_gt_mean_iou'), 3)} to {fmt(row.get('upsampled_gt_mean_iou'), 3)}")
    if d["score_delta"] < -0.05:
        fragments.append(f"mean TP score changes from {fmt(row.get('baseline_gt_mean_tp_score'), 3)} to {fmt(row.get('upsampled_gt_mean_tp_score'), 3)}")
    if not fragments:
        fragments.append("TP/FP/FN, IoU, and score are similar at the frame level")

    if "No clear degradation" in categories:
        return (
            "This frame does not show a clear detector-level degradation after upsampling: "
            + ", ".join(fragments)
            + ". The point-cloud evidence should be treated as qualitative context rather than a direct AP explanation."
        )
    return (
        "Upsampling does not improve detection in this frame because "
        + ", ".join(fragments)
        + ". Point-cloud metrics suggest that "
        + pc_evidence
        + ", which is consistent with added density not becoming reliable object-level geometry for PointRCNN."
    )


def source_frame_dir(row: dict) -> Path:
    return DET_ROOT / row["method"] / row["line"] / f"frame_{row['frame_id']}"


def frame_dir(row: dict) -> Path:
    return OUT / row["method"] / row["line"] / f"frame_{row['frame_id']}"


def pc_link_candidates(pc_rows: list[dict], start: Path) -> str:
    links = []
    for r in pc_rows:
        if r.get("object_crop_html_path"):
            links.append(f'<a href="{h(rel_from_out(r["object_crop_html_path"], start))}">v6 crop object {h(r.get("object_id"))}</a>')
        if r.get("full_frame_html_path"):
            links.append(f'<a href="{h(rel_from_out(r["full_frame_html_path"], start))}">v6 full frame object {h(r.get("object_id"))}</a>')
    return " ".join(links) or '<span class="muted">No matching v6 crop page.</span>'


def existing_link(path: Path, label: str, start: Path) -> str:
    if path.exists():
        return f'<a href="{h(rel(path, start))}">{h(label)}</a>'
    return f'<span class="muted">{h(label)} missing</span>'


def metric_cards(row: dict, pc_rows: list[dict]) -> str:
    d = row_deltas(row)
    avg_new_ratio = sum(fnum(r.get("new_points_ratio")) for r in pc_rows) / len(pc_rows) if pc_rows else 0
    avg_inside = sum(fnum(r.get("new_points_inside_bbox_ratio")) for r in pc_rows) / len(pc_rows) if pc_rows else 0
    avg_outside = sum(fnum(r.get("outside_bbox_ratio")) for r in pc_rows) / len(pc_rows) if pc_rows else 0
    metrics = [
        ("GT Cars", row.get("gt_car_count"), "Total labeled Car objects"),
        ("TP change", f"{d['tp_delta']:+.0f}", "upsampled TP minus baseline TP"),
        ("FP change", f"{d['fp_delta']:+.0f}", "upsampled FP minus baseline FP"),
        ("FN change", f"{d['fn_delta']:+.0f}", "upsampled FN minus baseline FN"),
        ("IoU change", f"{d['iou_delta']:+.3f}", "mean matched approximate BEV IoU"),
        ("Score change", f"{d['score_delta']:+.3f}", "mean matched TP score"),
        ("Added-point ratio", pct(avg_new_ratio), "mean v6 selected-crop added/changed point ratio"),
        ("Outside-box ratio", pct(avg_outside), f"inside-box ratio {pct(avg_inside)}"),
    ]
    return '<div class="grid">' + "".join(
        f'<div class="panel metric"><span class="muted">{h(label)}</span><b>{h(value)}</b><small>{h(note)}</small></div>'
        for label, value, note in metrics
    ) + "</div>"


def object_metrics_from_pc(pc_rows: list[dict]) -> dict[str, str]:
    if not pc_rows:
        return {
            "point_metric_status": "missing",
            "base_crop_points": "",
            "upsampled_crop_points": "",
            "new_points_ratio": "",
            "new_points_inside_bbox_ratio": "",
            "outside_bbox_ratio": "",
            "object_density_change_vs_shell": "not available",
        }
    r = pc_rows[0]
    base = fnum(r.get("base_crop_points"))
    up = fnum(r.get("upsampled_crop_points"))
    outside = fnum(r.get("outside_bbox_ratio"))
    inside = fnum(r.get("new_points_inside_bbox_ratio"))
    return {
        "point_metric_status": "available from selected v6 anchor crop",
        "base_crop_points": r.get("base_crop_points", ""),
        "upsampled_crop_points": r.get("upsampled_crop_points", ""),
        "new_points_ratio": r.get("new_points_ratio", ""),
        "new_points_inside_bbox_ratio": r.get("new_points_inside_bbox_ratio", ""),
        "outside_bbox_ratio": r.get("outside_bbox_ratio", ""),
        "object_density_change_vs_shell": (
            "shell/background change dominates selected new points"
            if outside > inside
            else "selected new points are not dominated by shell/background"
        ),
        "point_count_ratio": f"{(up / base):.4f}" if base else "",
    }


def object_page_body(title: str, obj: dict, row: dict, pc_rows: list[dict], src_links: str) -> str:
    metrics = object_metrics_from_pc(pc_rows)
    status = obj.get("failure_type", "Object")
    detector_evidence = obj.get("detector_evidence", "")
    point_evidence = obj.get("point_cloud_evidence", "")
    return f"""
<header>
  <h1>{h(title)}</h1>
  <p class="muted">{h(row['method'])} | {h(line_label(row['line']))} | frame {h(row['frame_id'])}</p>
  {badge(status, 'bad' if status != 'TP' else 'ok')}
</header>
<main>
  <section class="two">
    <div class="panel">
      <h2>Interpretation</h2>
      <div class="callout">{h(obj.get('explanation', 'This object page combines detector matching evidence with available point-crop metrics.'))}</div>
      <h3>Detector Evidence</h3>
      <p>{h(detector_evidence)}</p>
      <h3>Point-Cloud Evidence</h3>
      <p>{h(point_evidence)}</p>
      <h3>Metrics</h3>
      <table>
        <tbody>
          <tr><th>Object id</th><td>{h(obj.get('object_id'))}</td></tr>
          <tr><th>GT index</th><td>{h(obj.get('gt_idx'))}</td></tr>
          <tr><th>Baseline prediction index</th><td>{h(obj.get('baseline_pred_idx'))}</td></tr>
          <tr><th>Upsampled prediction index</th><td>{h(obj.get('upsampled_pred_idx'))}</td></tr>
          <tr><th>Baseline IoU</th><td>{h(fmt(obj.get('baseline_iou')))}</td></tr>
          <tr><th>Upsampled IoU</th><td>{h(fmt(obj.get('upsampled_iou')))}</td></tr>
          <tr><th>Baseline score</th><td>{h(fmt(obj.get('baseline_score')))}</td></tr>
          <tr><th>Upsampled score</th><td>{h(fmt(obj.get('upsampled_score')))}</td></tr>
          <tr><th>Base crop points</th><td>{h(metrics.get('base_crop_points'))}</td></tr>
          <tr><th>Upsampled crop points</th><td>{h(metrics.get('upsampled_crop_points'))}</td></tr>
          <tr><th>Point count ratio</th><td>{h(metrics.get('point_count_ratio', ''))}</td></tr>
          <tr><th>New/changed point ratio</th><td>{h(metrics.get('new_points_ratio'))}</td></tr>
          <tr><th>New points inside GT ratio</th><td>{h(metrics.get('new_points_inside_bbox_ratio'))}</td></tr>
          <tr><th>Shell/background outside ratio</th><td>{h(metrics.get('outside_bbox_ratio'))}</td></tr>
          <tr><th>Centroid and NN metrics</th><td>Not present in the source v1/v6 metadata for this crop; see limitations.</td></tr>
        </tbody>
      </table>
    </div>
    <div class="panel">
      <h2>Linked Visual Evidence</h2>
      <div class="links">{src_links}</div>
      <h3>Original v6 Point-Cloud Views</h3>
      <div class="links">{pc_link_candidates(pc_rows, frame_dir(row) / 'objects')}</div>
    </div>
  </section>
</main>
"""


def make_frame_objects(row: dict, meta: dict, pc_rows: list[dict]) -> list[dict]:
    objects = []
    b = meta.get("baseline_gt_prediction_audit", {})
    u = meta.get("upsampled_gt_prediction_audit", {})
    b_matches = {m.get("gt_idx"): m for m in b.get("matches", [])}
    u_matches = {m.get("gt_idx"): m for m in u.get("matches", [])}
    gt_indices = sorted(set(b_matches) | set(u_matches) | set(b.get("fn_gt_indices", [])) | set(u.get("fn_gt_indices", [])))

    for gt_idx in gt_indices:
        bm = b_matches.get(gt_idx)
        um = u_matches.get(gt_idx)
        if bm and um:
            iou_delta = fnum(um.get("bev_iou")) - fnum(bm.get("bev_iou"))
            score_delta = fnum(um.get("score")) - fnum(bm.get("score"))
            failure_type = "TP"
            if iou_delta < -0.02:
                failure_type = "IOU_DROP"
            elif score_delta < -0.2:
                failure_type = "SCORE_DROP"
            objects.append({
                "failure_type": failure_type,
                "object_id": gt_idx,
                "gt_idx": gt_idx,
                "baseline_pred_idx": bm.get("pred_idx"),
                "upsampled_pred_idx": um.get("pred_idx"),
                "baseline_iou": bm.get("bev_iou"),
                "upsampled_iou": um.get("bev_iou"),
                "baseline_score": bm.get("score"),
                "upsampled_score": um.get("score"),
                "detector_evidence": f"GT {gt_idx} is matched in both runs; IoU delta {iou_delta:+.3f}, score delta {score_delta:+.3f}.",
                "point_cloud_evidence": "Available v6 crop metrics are listed below; use the crop pages to inspect object interior and shell/background points.",
                "explanation": (
                    "The object remains detected after upsampling, but localization or confidence may degrade. "
                    "This is consistent with a local point distribution that changes without adding stable object-surface structure."
                    if failure_type != "TP"
                    else "The object remains a true positive in both runs. It is included as a control case for comparing point-density changes against stable detector behavior."
                ),
            })
        elif bm and not um:
            objects.append({
                "failure_type": "FN",
                "object_id": gt_idx,
                "gt_idx": gt_idx,
                "baseline_pred_idx": bm.get("pred_idx"),
                "upsampled_pred_idx": "",
                "baseline_iou": bm.get("bev_iou"),
                "upsampled_iou": "",
                "baseline_score": bm.get("score"),
                "upsampled_score": "",
                "detector_evidence": f"Detected before upsampling, missed after upsampling for GT {gt_idx}.",
                "point_cloud_evidence": "Inspect whether object density increased inside the GT box or mainly around the background shell.",
                "explanation": "This missed GT may contribute to recall degradation: the upsampled cloud did not produce a matched prediction where the baseline did.",
            })
        elif um and not bm:
            objects.append({
                "failure_type": "TP",
                "object_id": gt_idx,
                "gt_idx": gt_idx,
                "baseline_pred_idx": "",
                "upsampled_pred_idx": um.get("pred_idx"),
                "baseline_iou": "",
                "upsampled_iou": um.get("bev_iou"),
                "baseline_score": "",
                "upsampled_score": um.get("score"),
                "detector_evidence": f"GT {gt_idx} is matched only after upsampling.",
                "point_cloud_evidence": "This positive case is useful for contrast with failure cases.",
                "explanation": "This object improves locally after upsampling, but method-level AP still depends on aggregate FP, FN, IoU, and score changes.",
            })
        else:
            objects.append({
                "failure_type": "FN",
                "object_id": gt_idx,
                "gt_idx": gt_idx,
                "baseline_pred_idx": "",
                "upsampled_pred_idx": "",
                "baseline_iou": "",
                "upsampled_iou": "",
                "baseline_score": "",
                "upsampled_score": "",
                "detector_evidence": f"GT {gt_idx} is missed by both baseline and upsampled detections.",
                "point_cloud_evidence": "The crop evidence can show whether added points still fail to create a detectable object pattern.",
                "explanation": "This is a persistent missed object rather than an upsampling-specific regression.",
            })

    for pred_idx in u.get("fp_pred_indices", []):
        objects.append({
            "failure_type": "FP",
            "object_id": f"U{pred_idx}",
            "gt_idx": "",
            "baseline_pred_idx": "",
            "upsampled_pred_idx": pred_idx,
            "baseline_iou": "",
            "upsampled_iou": "",
            "baseline_score": "",
            "upsampled_score": "",
            "detector_evidence": f"Upsampled prediction {pred_idx} has no matched GT under the v1 matching policy.",
            "point_cloud_evidence": "If the matching crop shows high shell/background growth, this is consistent with background amplification producing object-like evidence.",
            "explanation": "This false positive may contribute to precision degradation after upsampling.",
        })
    return objects


def generate_annotation_page(kind: str, row: dict, pc_rows: list[dict], categories: list[str], pc_evidence: str, target: Path | None) -> str:
    start = frame_dir(row)
    target_html = ""
    if target and target.exists():
        target_html = f'<iframe class="viewer" src="{h(rel(target, start))}" title="{h(kind)}"></iframe>'
    else:
        target_html = '<div class="panel muted">Source visualization is missing for this frame.</div>'

    annotations = {
        "evaluation_fov_annotated": [
            "TP: matched GT-prediction pair under the v1 same-class center-distance policy.",
            "FP: prediction without GT; a new FP after upsampling indicates precision degradation.",
            "FN: missed GT; a GT detected before upsampling and missed after upsampling indicates recall degradation.",
            "IoU drop: matched object remains detected but approximate BEV IoU decreases.",
            "Score drop: matched TP confidence decreases after upsampling.",
            "Localization shift: prediction center changes between baseline and upsampled outputs.",
        ],
        "pointcloud_change_view": [
            "Object interior points are interpreted from the GT crop metrics where available.",
            "Background/shell points are approximated by new or changed points outside the labeled object box.",
            "Object density not improved: selected crop point ratio is small or decreases.",
            "Background density increased: outside-box ratio exceeds inside-box ratio.",
            "Boundary scatter/noise is a qualitative interpretation from crop overlays, not an official AP metric.",
        ],
        "detector_change_view": [
            "Baseline and upsampled detection statuses are compared using TP, FP, FN, IoU, and score.",
            "Matched boxes with lower IoU are flagged as localization degradation.",
            "Matched boxes with lower TP score are flagged as confidence degradation.",
        ],
        "full_lidar_context_annotated": [
            "This is full Velodyne context, not the official KITTI evaluation region.",
            "KITTI labels and PointRCNN detections are camera-FOV based.",
            "Do not interpret boxes outside the camera FOV as missing detector failures.",
        ],
    }.get(kind, [])

    return page(kind.replace("_", " ").title(), f"""
<header>
  <h1>{h(kind.replace('_', ' ').title())}</h1>
  <p class="muted">{h(row['method'])} | {h(line_label(row['line']))} | frame {h(row['frame_id'])}</p>
  {''.join(badge(c, 'bad' if c != 'No clear degradation' else 'neutral') for c in categories)}
</header>
<main>
  <section class="two">
    <div class="panel">
      <h2>Precise Annotations</h2>
      <ul class="annotation-list">{''.join(f'<li>{h(x)}</li>' for x in annotations)}</ul>
      <h2>Evidence Link</h2>
      <p>{h(pc_evidence)}</p>
      <div class="links">{pc_link_candidates(pc_rows, start)}</div>
    </div>
    <div>{target_html}</div>
  </section>
</main>
""")


def generate_dashboard(row: dict, pc_rows: list[dict], categories: list[str], pc_evidence: str) -> str:
    verdict = row.get("ap_consistency_verdict") or row.get("verdict") or "UNKNOWN"
    status, badges = provenance_badges(verdict)
    start = frame_dir(row)
    src = source_frame_dir(row)
    warning = row.get("ap_consistency_warnings", "")
    summary_rows = [
        {"k": "Method", "v": h(row["method"])},
        {"k": "Line", "v": h(line_label(row["line"]))},
        {"k": "Frame", "v": h(row["frame_id"])},
        {"k": "Baseline input", "v": h(baseline_input(row["line"]))},
        {"k": "Upsampled input", "v": h(upsampled_input(row["method"], row["line"]))},
        {"k": "AP provenance status", "v": badges},
        {"k": "Use in final AP-consistent explanation", "v": h(status)},
        {"k": "Warnings", "v": h(warning or "none")},
    ]
    eval_rows = [
        ("GT Car count", row.get("gt_car_count")),
        ("Easy / Moderate / Hard GT", f"{row.get('gt_car_easy_count','')} / {row.get('gt_car_moderate_count','')} / {row.get('gt_car_hard_count','')}"),
        ("Baseline prediction count", row.get("baseline_detection_count")),
        ("Upsampled prediction count", row.get("upsampled_detection_count")),
        ("Baseline TP / FP / FN", f"{row.get('baseline_gt_matched_prediction_count')} / {row.get('baseline_gt_false_positive_count')} / {row.get('baseline_gt_missed_gt_count')}"),
        ("Upsampled TP / FP / FN", f"{row.get('upsampled_gt_matched_prediction_count')} / {row.get('upsampled_gt_false_positive_count')} / {row.get('upsampled_gt_missed_gt_count')}"),
        ("Mean IoU baseline / upsampled", f"{fmt(row.get('baseline_gt_mean_iou'))} / {fmt(row.get('upsampled_gt_mean_iou'))}"),
        ("Mean TP score baseline / upsampled", f"{fmt(row.get('baseline_gt_mean_tp_score'))} / {fmt(row.get('upsampled_gt_mean_tp_score'))}"),
        ("Approx precision baseline / upsampled", f"{fmt(row.get('baseline_gt_approx_precision'))} / {fmt(row.get('upsampled_gt_approx_precision'))}"),
        ("Approx recall baseline / upsampled", f"{fmt(row.get('baseline_gt_approx_recall'))} / {fmt(row.get('upsampled_gt_approx_recall'))}"),
    ]
    nav = [
        ("Annotated FOV", "evaluation_fov_annotated.html"),
        ("Point-cloud change", "pointcloud_change_view.html"),
        ("Detector change", "detector_change_view.html"),
        ("Full LiDAR context", "full_lidar_context_annotated.html"),
        ("Root-cause summary", "root_cause_summary.html"),
        ("Original v1 dashboard", rel(src / "combined_dashboard.html", start)),
    ]
    return page("Analysis Dashboard", f"""
<header>
  <h1>Integrated Analysis Dashboard</h1>
  <p class="muted">{h(row['method'])} | {h(line_label(row['line']))} | frame {h(row['frame_id'])}</p>
  {badges}{''.join(badge(c, 'bad' if c != 'No clear degradation' else 'neutral') for c in categories)}
</header>
<main>
  <section class="callout"><b>Main conclusion.</b> {h(conclusion(row, categories, pc_evidence))}</section>
  <h2>Experiment Identity</h2>
  <table><tbody>{''.join(f'<tr><th>{h(r["k"])}</th><td>{r["v"]}</td></tr>' for r in summary_rows)}</tbody></table>
  <h2>Evaluation Summary</h2>
  {metric_cards(row, pc_rows)}
  <table><tbody>{''.join(f'<tr><th>{h(k)}</th><td>{h(v)}</td></tr>' for k, v in eval_rows)}</tbody></table>
  <h2>Navigation</h2>
  <div class="panel links">{''.join(f'<a href="{h(url)}">{h(label)}</a>' for label, url in nav)}</div>
  <h2>Point-Cloud Evidence</h2>
  <div class="panel">
    <p>{h(pc_evidence)}</p>
    <div class="links">{pc_link_candidates(pc_rows, start)}</div>
  </div>
</main>
""")


def generate_root_cause(row: dict, pc_rows: list[dict], categories: list[str], pc_evidence: str) -> str:
    detector = row.get("evaluation_interpretation") or "See frame-level TP/FP/FN, IoU, and score deltas."
    items = []
    for cat in categories:
        if cat == "Mixed failure":
            continue
        items.append(f"""
<div class="panel">
  <h3>{h(cat)}</h3>
  <p><b>Detector evidence:</b> {h(detector)}</p>
  <p><b>Point-cloud evidence:</b> {h(pc_evidence)}</p>
  <p><b>Explanation:</b> {h(root_cause_explanation(cat))}</p>
  <p><a href="evaluation_fov_annotated.html">Annotated detector view</a> | <a href="pointcloud_change_view.html">Point-cloud change view</a></p>
</div>
""")
    return page("Root Cause Summary", f"""
<header>
  <h1>Root-Cause Summary</h1>
  <p class="muted">{h(row['method'])} | {h(line_label(row['line']))} | frame {h(row['frame_id'])}</p>
</header>
<main>
  <section class="callout">{h(conclusion(row, categories, pc_evidence))}</section>
  <div class="grid">{''.join(items)}</div>
</main>
""")


def root_cause_explanation(cat: str) -> str:
    return {
        "Recall degradation": "Upsampling is not strengthening useful object geometry enough to preserve a matched prediction.",
        "Precision degradation": "Background or shell-region density may become more object-like, increasing unmatched predictions.",
        "Localization degradation": "Local point distributions may shift around object boundaries, reducing overlap with GT boxes.",
        "Confidence degradation": "The detector features may become less consistent with the statistics learned from original LiDAR points.",
        "No clear degradation": "Frame-level detector metrics do not isolate one dominant failure mode.",
    }.get(cat, "Multiple detector and point-cloud changes occur together.")


def build_package() -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    det_summary = read_csv(DET_ROOT / "summary_all.csv")
    matching = index_by_key(read_csv(DET_ROOT / "gt_prediction_matching_audit.csv"), ("method", "line", "frame_id"))
    ap_audit = index_by_key(read_csv(DET_ROOT / "ap_consistency_audit.csv"), ("method", "line", "frame_id"))
    pc_summary = read_csv(PC_ROOT / "summary_all.csv")
    pc_by_frame = group_by_key(pc_summary, ("method", "line", "frame_id"))

    # Prefer v1 summary rows with usable matching metrics.
    cases = []
    seen = set()
    for row in det_summary:
        key = (row.get("method", ""), row.get("line", ""), row.get("frame_id", ""))
        if key in seen:
            continue
        seen.add(key)
        if not row.get("baseline_gt_matched_prediction_count") and key in matching:
            row.update(matching[key])
        if key in ap_audit:
            row.setdefault("ap_consistency_verdict", ap_audit[key].get("verdict", ""))
            row.setdefault("ap_consistency_warnings", ap_audit[key].get("warnings", ""))
        if row.get("baseline_gt_matched_prediction_count") or row.get("upsampled_gt_matched_prediction_count"):
            cases.append(row)

    cases.sort(key=lambda r: (METHOD_ORDER.index(r["method"]) if r["method"] in METHOD_ORDER else 99, r["line"], r["frame_id"]))

    case_rows = []
    object_rows = []
    provenance_rows = []
    method_counts = defaultdict(Counter)
    generated = Counter()
    missing_links = []

    for row in cases:
        key = (row["method"], row["line"], row["frame_id"])
        pc_rows = pc_by_frame.get(key, [])
        categories, pc_evidence = classify(row, pc_rows)
        outdir = frame_dir(row)
        objects_dir = outdir / "objects"
        objects_dir.mkdir(parents=True, exist_ok=True)

        meta = load_json(source_frame_dir(row) / "metadata.json")
        objects = make_frame_objects(row, meta, pc_rows)

        write_text(outdir / "analysis_dashboard.html", generate_dashboard(row, pc_rows, categories, pc_evidence))
        generated["dashboards"] += 1

        src = source_frame_dir(row)
        annotation_targets = {
            "evaluation_fov_annotated.html": src / "overlay_fov_frame.html",
            "pointcloud_change_view.html": Path(pc_rows[0]["object_crop_html_path"]) if pc_rows and pc_rows[0].get("object_crop_html_path") else src / "full_frame.html",
            "detector_change_view.html": src / "combined_dashboard.html",
            "full_lidar_context_annotated.html": src / "full_frame.html",
        }
        for filename, target in annotation_targets.items():
            write_text(outdir / filename, generate_annotation_page(filename[:-5], row, pc_rows, categories, pc_evidence, target))
            generated[filename] += 1
            if not target.exists():
                missing_links.append(str(target))
        write_text(outdir / "root_cause_summary.html", generate_root_cause(row, pc_rows, categories, pc_evidence))

        for obj in objects:
            obj_prefix = obj["failure_type"]
            obj_name = f"object_{obj_prefix}_{obj['object_id']}.html".replace("/", "_")
            src_links = [
                existing_link(src / "combined_dashboard.html", "v1 combined dashboard", objects_dir),
                existing_link(src / "overlay_fov_frame.html", "v1 overlay FOV", objects_dir),
                existing_link(src / "full_frame.html", "v1 full frame", objects_dir),
            ]
            if obj.get("upsampled_pred_idx") != "":
                idx = inum(obj.get("upsampled_pred_idx"))
                src_links.append(existing_link(src / f"eval_object_upsampled_{idx:03d}_{'fp' if obj_prefix == 'FP' else 'tp'}.html", "v1 upsampled object page", objects_dir))
            if obj.get("baseline_pred_idx") != "":
                idx = inum(obj.get("baseline_pred_idx"))
                src_links.append(existing_link(src / f"eval_object_baseline_{idx:03d}_{'fp' if obj_prefix == 'FP' else 'tp'}.html", "v1 baseline object page", objects_dir))

            write_text(objects_dir / obj_name, page(f"Object {obj_prefix}", object_page_body(f"Object {obj_prefix}", obj, row, pc_rows, " ".join(src_links))))
            generated["object_pages"] += 1

            metrics = object_metrics_from_pc(pc_rows)
            object_rows.append({
                "method": row["method"],
                "line": row["line"],
                "frame_id": row["frame_id"],
                "failure_type": obj_prefix,
                "object_id": obj.get("object_id", ""),
                "gt_idx": obj.get("gt_idx", ""),
                "baseline_pred_idx": obj.get("baseline_pred_idx", ""),
                "upsampled_pred_idx": obj.get("upsampled_pred_idx", ""),
                "baseline_iou": obj.get("baseline_iou", ""),
                "upsampled_iou": obj.get("upsampled_iou", ""),
                "iou_change": fnum(obj.get("upsampled_iou")) - fnum(obj.get("baseline_iou")) if obj.get("baseline_iou") != "" and obj.get("upsampled_iou") != "" else "",
                "baseline_score": obj.get("baseline_score", ""),
                "upsampled_score": obj.get("upsampled_score", ""),
                "score_change": fnum(obj.get("upsampled_score")) - fnum(obj.get("baseline_score")) if obj.get("baseline_score") != "" and obj.get("upsampled_score") != "" else "",
                "base_crop_points": metrics.get("base_crop_points", ""),
                "upsampled_crop_points": metrics.get("upsampled_crop_points", ""),
                "new_points_ratio": metrics.get("new_points_ratio", ""),
                "new_points_inside_bbox_ratio": metrics.get("new_points_inside_bbox_ratio", ""),
                "outside_bbox_ratio": metrics.get("outside_bbox_ratio", ""),
                "object_density_change_vs_shell": metrics.get("object_density_change_vs_shell", ""),
                "page": rel(objects_dir / obj_name, OUT),
                "detector_evidence": obj.get("detector_evidence", ""),
                "point_cloud_evidence": obj.get("point_cloud_evidence", ""),
            })

        d = row_deltas(row)
        case = {
            "method": row["method"],
            "line": row["line"],
            "frame_id": row["frame_id"],
            "root_cause_categories": "; ".join(categories),
            "dominant_failure_mode": categories[0],
            "ap_provenance_status": provenance_badges(row.get("ap_consistency_verdict", ""))[0],
            "ap_consistency_verdict": row.get("ap_consistency_verdict", ""),
            "ap_consistency_warnings": row.get("ap_consistency_warnings", ""),
            "gt_car_count": row.get("gt_car_count", ""),
            "baseline_prediction_count": row.get("baseline_detection_count", ""),
            "upsampled_prediction_count": row.get("upsampled_detection_count", ""),
            "baseline_tp": row.get("baseline_gt_matched_prediction_count", ""),
            "baseline_fp": row.get("baseline_gt_false_positive_count", ""),
            "baseline_fn": row.get("baseline_gt_missed_gt_count", ""),
            "upsampled_tp": row.get("upsampled_gt_matched_prediction_count", ""),
            "upsampled_fp": row.get("upsampled_gt_false_positive_count", ""),
            "upsampled_fn": row.get("upsampled_gt_missed_gt_count", ""),
            "tp_change": d["tp_delta"],
            "fp_change": d["fp_delta"],
            "fn_change": d["fn_delta"],
            "baseline_mean_iou": row.get("baseline_gt_mean_iou", ""),
            "upsampled_mean_iou": row.get("upsampled_gt_mean_iou", ""),
            "mean_iou_change": d["iou_delta"],
            "baseline_mean_tp_score": row.get("baseline_gt_mean_tp_score", ""),
            "upsampled_mean_tp_score": row.get("upsampled_gt_mean_tp_score", ""),
            "mean_tp_score_change": d["score_delta"],
            "baseline_precision": row.get("baseline_gt_approx_precision", ""),
            "upsampled_precision": row.get("upsampled_gt_approx_precision", ""),
            "baseline_recall": row.get("baseline_gt_approx_recall", ""),
            "upsampled_recall": row.get("upsampled_gt_approx_recall", ""),
            "point_cloud_evidence": pc_evidence,
            "dashboard": rel(outdir / "analysis_dashboard.html", OUT),
            "evaluation_fov_annotated": rel(outdir / "evaluation_fov_annotated.html", OUT),
            "pointcloud_change_view": rel(outdir / "pointcloud_change_view.html", OUT),
            "detector_change_view": rel(outdir / "detector_change_view.html", OUT),
        }
        case_rows.append(case)
        for cat in categories:
            method_counts[(row["method"], row["line"])][cat] += 1
            generated[f"failure_{cat}"] += 1
        generated["ap_consistent" if case["ap_provenance_status"] == "AP-consistent" else "qualitative"] += 1
        provenance_rows.append({
            "method": row["method"],
            "line": row["line"],
            "frame_id": row["frame_id"],
            "ap_consistency_verdict": row.get("ap_consistency_verdict", ""),
            "ap_consistency_warnings": row.get("ap_consistency_warnings", ""),
            "baseline_detection_txt_path": row.get("baseline_detection_txt_path", ""),
            "upsampled_detection_txt_path": row.get("upsampled_detection_txt_path", ""),
            "baseline_pointcloud_path": pc_rows[0].get("base_pointcloud_path", "") if pc_rows else meta.get("paths", {}).get("baseline_point_cloud", ""),
            "upsampled_pointcloud_path": pc_rows[0].get("upsampled_pointcloud_path", "") if pc_rows else meta.get("paths", {}).get("upsampled_point_cloud", ""),
            "v1_metadata": rel(source_frame_dir(row) / "metadata.json", OUT),
            "v6_metadata": rel_from_out(pc_rows[0]["metadata_json_path"], OUT) if pc_rows and pc_rows[0].get("metadata_json_path") else "",
            "safe_for_ap_consistent_explanation": "yes" if row.get("ap_consistency_verdict") == "OK" else "no",
        })

        metadata = {
            "method": row["method"],
            "line": row["line"],
            "frame_id": row["frame_id"],
            "categories": categories,
            "source_detector_metadata": rel(source_frame_dir(row) / "metadata.json", outdir),
            "source_pointcloud_metadata": [rel_from_out(r.get("metadata_json_path", ""), outdir) for r in pc_rows if r.get("metadata_json_path")],
            "case_summary": case,
        }
        write_text(outdir / "metadata.json", json.dumps(metadata, indent=2))

    method_rows = []
    for (method, line), counts in sorted(method_counts.items(), key=lambda x: (METHOD_ORDER.index(x[0][0]) if x[0][0] in METHOD_ORDER else 99, x[0][1])):
        relevant = [r for r in case_rows if r["method"] == method and r["line"] == line]
        dominant = counts.most_common(1)[0][0] if counts else "No clear degradation"
        method_rows.append({
            "method": method,
            "line": line,
            "selected_cases": len(relevant),
            "fn_increase_cases": sum(1 for r in relevant if fnum(r["fn_change"]) > 0),
            "fp_increase_cases": sum(1 for r in relevant if fnum(r["fp_change"]) > 0),
            "iou_decrease_cases": sum(1 for r in relevant if fnum(r["mean_iou_change"]) < 0),
            "tp_score_decrease_cases": sum(1 for r in relevant if fnum(r["mean_tp_score_change"]) < 0),
            "dominant_failure_mode": dominant,
            "avg_tp_change": sum(fnum(r["tp_change"]) for r in relevant) / len(relevant),
            "avg_fp_change": sum(fnum(r["fp_change"]) for r in relevant) / len(relevant),
            "avg_fn_change": sum(fnum(r["fn_change"]) for r in relevant) / len(relevant),
            "avg_iou_change": sum(fnum(r["mean_iou_change"]) for r in relevant) / len(relevant),
            "avg_score_change": sum(fnum(r["mean_tp_score_change"]) for r in relevant) / len(relevant),
            "line_question": "Does upsampling improve detection relative to the original baseline?" if line.startswith("line_a") else "Does upsampling improve detection relative to the downsampled baseline?",
            "interpretation": method_line_interpretation(method, line, relevant),
        })

    write_csv(OUT / "case_level_failure_analysis.csv", case_rows, list(case_rows[0].keys()) if case_rows else [])
    write_csv(OUT / "object_level_failure_analysis.csv", object_rows, list(object_rows[0].keys()) if object_rows else [])
    write_csv(OUT / "method_line_summary.csv", method_rows, list(method_rows[0].keys()) if method_rows else [])
    write_csv(OUT / "input_and_provenance_audit.csv", provenance_rows, list(provenance_rows[0].keys()) if provenance_rows else [])

    write_text(OUT / "failure_analysis_summary.md", render_summary_md(method_rows, case_rows))
    write_text(OUT / "failure_analysis_summary.html", render_summary_html(method_rows, case_rows))
    write_text(OUT / "thesis_ready_findings.md", render_thesis_findings(method_rows, case_rows))
    examples = select_examples(case_rows, object_rows)
    write_text(OUT / "meeting_ready_examples.md", render_examples_md(examples, object_rows))
    write_text(OUT / "input_and_provenance_audit.md", render_provenance_md(provenance_rows))
    write_text(OUT / "README.md", render_readme(generated, method_rows))
    write_text(OUT / "index.html", render_index(method_rows, case_rows, examples))

    validation = validate_outputs(case_rows, object_rows, method_rows, missing_links)
    generated.update(validation)
    write_text(OUT / "validation_report.json", json.dumps(dict(generated), indent=2, sort_keys=True))
    return dict(generated)


def method_line_interpretation(method: str, line: str, rows: list[dict]) -> str:
    fp = sum(1 for r in rows if fnum(r["fp_change"]) > 0)
    fn = sum(1 for r in rows if fnum(r["fn_change"]) > 0)
    iou = sum(1 for r in rows if fnum(r["mean_iou_change"]) < 0)
    score = sum(1 for r in rows if fnum(r["mean_tp_score_change"]) < 0)
    bits = []
    if fn:
        bits.append(f"{fn} cases increase FN")
    if fp:
        bits.append(f"{fp} cases increase FP")
    if iou:
        bits.append(f"{iou} cases decrease IoU")
    if score:
        bits.append(f"{score} cases decrease TP score")
    if not bits:
        bits.append("selected cases do not show a consistent frame-level degradation")
    question = "Line A" if line.startswith("line_a") else "Line B"
    return f"For {method} {question}, " + ", ".join(bits) + "; this does not provide consistent evidence that upsampling improves PointRCNN detection."


def render_summary_md(method_rows: list[dict], case_rows: list[dict]) -> str:
    lines = ["# Failure Analysis Summary", ""]
    lines.append("This report summarizes selected frame-level evidence connecting point-cloud changes to detector outcomes. Line A compares the original baseline with original + upsampling. Line B compares the downsampled baseline with downsampled + upsampling; it is not described as recovery to original performance.")
    lines.append("")
    lines.append("| Method | Line | Cases | FN inc | FP inc | IoU dec | Score dec | Dominant mode | Interpretation |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|---|")
    for r in method_rows:
        lines.append(f"| {r['method']} | {r['line']} | {r['selected_cases']} | {r['fn_increase_cases']} | {r['fp_increase_cases']} | {r['iou_decrease_cases']} | {r['tp_score_decrease_cases']} | {r['dominant_failure_mode']} | {r['interpretation']} |")
    lines.append("")
    counts = Counter()
    for r in case_rows:
        for cat in r["root_cause_categories"].split("; "):
            counts[cat] += 1
    lines.append("## Failure Mode Counts")
    for cat in FAILURE_ORDER:
        if counts.get(cat):
            lines.append(f"- {cat}: {counts[cat]}")
    return "\n".join(lines) + "\n"


def render_summary_html(method_rows: list[dict], case_rows: list[dict]) -> str:
    rows = []
    for r in method_rows:
        rows.append({
            "method": h(r["method"]),
            "line": h(r["line"]),
            "cases": h(r["selected_cases"]),
            "fn": h(r["fn_increase_cases"]),
            "fp": h(r["fp_increase_cases"]),
            "iou": h(r["iou_decrease_cases"]),
            "score": h(r["tp_score_decrease_cases"]),
            "mode": h(r["dominant_failure_mode"]),
            "avg": h(f"TP {r['avg_tp_change']:+.2f}, FP {r['avg_fp_change']:+.2f}, FN {r['avg_fn_change']:+.2f}, IoU {r['avg_iou_change']:+.3f}, score {r['avg_score_change']:+.3f}"),
        })
    counts = Counter()
    for r in case_rows:
        for cat in r["root_cause_categories"].split("; "):
            counts[cat] += 1
    bars = "".join(
        f'<p><b>{h(cat)}</b> {counts[cat]}<div class="bar"><span style="width:{min(100, counts[cat] * 100 / max(1, len(case_rows))):.1f}%"></span></div></p>'
        for cat in FAILURE_ORDER if counts.get(cat)
    )
    return page("Failure Analysis Summary", f"""
<header><h1>Failure Analysis Summary</h1><p class="muted">Method and line-level explanation of why upsampling does not consistently improve PointRCNN.</p></header>
<main>
  <section class="callout">Line A compares original baseline vs original + upsampling. Line B compares downsampled baseline vs downsampled + upsampling, not recovery to original performance.</section>
  <h2>Method-Line Summary</h2>
  {table(rows, [('method','Method'),('line','Line'),('cases','Cases'),('fn','FN inc'),('fp','FP inc'),('iou','IoU dec'),('score','Score dec'),('mode','Dominant mode'),('avg','Average deltas')])}
  <h2>Failure Mode Counts</h2>
  <div class="panel">{bars}</div>
</main>
""")


def render_thesis_findings(method_rows: list[dict], case_rows: list[dict]) -> str:
    total = len(case_rows)
    ap_ok = sum(1 for r in case_rows if r["ap_provenance_status"] == "AP-consistent")
    counts = Counter()
    for r in case_rows:
        for cat in r["root_cause_categories"].split("; "):
            counts[cat] += 1
    return f"""# Thesis-Ready Findings

## Analysis Method

This package integrates two levels of evidence: point-cloud crop changes from `interactive_object_crop_visualization_improved_v6` and PointRCNN detection-box changes from `pointrcnn_detection_box_visualization_v1`. The analysis is organized by method, comparison line, and frame.

Line A compares the original KITTI baseline against original + upsampling. Line B compares the downsampled baseline against downsampled + upsampling. Line B is interpreted only as an upsampling effect relative to the downsampled baseline, not as recovery to original performance.

## GT and Prediction Matching

Ground-truth Car boxes and PointRCNN predictions are matched using the existing v1 audit policy: same-class matching with a camera-center distance threshold fallback and approximate BEV IoU reporting. TP, FP, and FN counts in this package are therefore analysis metrics for selected cases, not a replacement for official KITTI AP.

## Main Finding

Although upsampling increases point density, the additional points do not necessarily provide useful object-level geometry for PointRCNN. In several evaluated frames, upsampling increases missed detections, introduces false positives, reduces localization IoU, or lowers confidence scores. The object-level crop analysis suggests that the added points may be scattered around object boundaries or background regions rather than improving the structure of labeled objects. Therefore, point-cloud upsampling does not consistently translate into better detection performance.

## Evidence Summary

- Total integrated cases: {total}
- AP-consistent cases: {ap_ok}
- Qualitative/provenance-warning cases: {total - ap_ok}
- Recall degradation labels: {counts.get('Recall degradation', 0)}
- Precision degradation labels: {counts.get('Precision degradation', 0)}
- Localization degradation labels: {counts.get('Localization degradation', 0)}
- Confidence degradation labels: {counts.get('Confidence degradation', 0)}

## Interpretation Across Methods

{chr(10).join(f'- {r["method"]} {r["line"]}: {r["interpretation"]}' for r in method_rows)}

## Limitations

The detector matching in this visualization package is approximate and designed for explanation. Official AP should be taken from the validated evaluation pipeline. Several point-cloud metrics, such as exact generated-point nearest-neighbor distance, object centroid shift, and full shell density, are only available where the source v6 metadata exposes enough information. When AP provenance is marked WARNING or ERROR, the case is retained for qualitative explanation but not treated as final AP-consistent evidence.

## Causality Wording

The evidence supports cautious statements such as "suggests", "indicates", and "is consistent with". The visualizations show correlation between point-cloud changes and detector behavior; they do not prove a unique causal mechanism for every individual prediction.
"""


def select_examples(case_rows: list[dict], object_rows: list[dict]) -> list[dict]:
    def first(pred):
        return next((r for r in case_rows if pred(r)), None)

    specs = [
        ("FN-increase case", lambda r: fnum(r["fn_change"]) > 0),
        ("FP-increase case", lambda r: fnum(r["fp_change"]) > 0),
        ("IoU-drop case", lambda r: fnum(r["mean_iou_change"]) < -0.02),
        ("Score-drop case", lambda r: fnum(r["mean_tp_score_change"]) < -0.2),
        ("Line A example", lambda r: r["line"].startswith("line_a")),
        ("Line B example", lambda r: r["line"].startswith("line_b")),
        ("PDANS example", lambda r: r["method"] == "PDANS"),
        ("Denser point cloud without detector improvement", lambda r: "outside" in r["point_cloud_evidence"] or fnum(r["fp_change"]) > 0 or fnum(r["fn_change"]) > 0),
    ]
    examples = []
    used = set()
    for label, pred in specs:
        row = first(lambda r, p=pred: p(r) and (r["method"], r["line"], r["frame_id"], label) not in used)
        if row:
            used.add((row["method"], row["line"], row["frame_id"], label))
            examples.append({"label": label, **row})
    return examples


def render_examples_md(examples: list[dict], object_rows: list[dict]) -> str:
    lines = ["# Meeting-Ready Examples", ""]
    for ex in examples:
        objs = [o for o in object_rows if o["method"] == ex["method"] and o["line"] == ex["line"] and o["frame_id"] == ex["frame_id"]]
        obj = next((o for o in objs if o["failure_type"] in ("FN", "FP", "IOU_DROP", "SCORE_DROP")), objs[0] if objs else None)
        lines.extend([
            f"## {ex['label']}",
            "",
            f"- Method: {ex['method']}",
            f"- Line: {ex['line']}",
            f"- Frame: {ex['frame_id']}",
            f"- Failure type: {ex['root_cause_categories']}",
            f"- Detector evidence: TP change {ex['tp_change']}, FP change {ex['fp_change']}, FN change {ex['fn_change']}, IoU change {fmt(ex['mean_iou_change'])}, score change {fmt(ex['mean_tp_score_change'])}",
            f"- Point-cloud evidence: {ex['point_cloud_evidence']}",
            f"- Dashboard: [{ex['dashboard']}]({ex['dashboard']})",
            f"- Annotated FOV: [{ex['evaluation_fov_annotated']}]({ex['evaluation_fov_annotated']})",
            f"- Point-cloud change: [{ex['pointcloud_change_view']}]({ex['pointcloud_change_view']})",
        ])
        if obj:
            lines.append(f"- Object-level page: [{obj['page']}]({obj['page']})")
        lines.extend([
            "",
            "Presentation script: This example shows that increasing or changing local point density is not sufficient for PointRCNN improvement. The detector evidence changes in TP, FP, FN, IoU, or score, while the point-cloud evidence suggests that added density may remain outside the object surface or may not strengthen the object interior. This supports a careful interpretation: upsampling can change the input distribution without reliably adding the geometry that the detector needs.",
            "",
        ])
    return "\n".join(lines)


def render_provenance_md(rows: list[dict]) -> str:
    lines = ["# Input and Provenance Audit", "", "| Method | Line | Frame | Verdict | Safe for AP evidence | Warning |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['line']} | {r['frame_id']} | {r['ap_consistency_verdict']} | {r['safe_for_ap_consistent_explanation']} | {r['ap_consistency_warnings']} |")
    return "\n".join(lines) + "\n"


def render_readme(generated: Counter, method_rows: list[dict]) -> str:
    return f"""# Upsampling Detection Failure Explanation v2

This package explains why point-cloud upsampling does not consistently improve PointRCNN detection performance. It connects point-cloud-level crop changes, detector-level GT/prediction matching, TP/FP/FN changes, IoU, confidence, and AP provenance.

Start with `index.html`, then open `failure_analysis_summary.html`, `thesis_ready_findings.md`, and `meeting_ready_examples.md`.

Generated content:

- Integrated dashboards: {generated.get('dashboards', 0)}
- Annotated FOV views: {generated.get('evaluation_fov_annotated.html', 0)}
- Point-cloud change views: {generated.get('pointcloud_change_view.html', 0)}
- Object-level pages: {generated.get('object_pages', 0)}
- Method-line summaries: {len(method_rows)}

AP provenance rule: cases marked AP-consistent can be used as final AP-consistent explanatory evidence. WARNING/ERROR cases are retained for qualitative inspection only.
"""


def render_index(method_rows: list[dict], case_rows: list[dict], examples: list[dict]) -> str:
    cards = []
    for r in method_rows:
        subset = [c for c in case_rows if c["method"] == r["method"] and c["line"] == r["line"]]
        links = " ".join(f'<a href="{h(c["dashboard"])}">frame {h(c["frame_id"])}</a>' for c in subset)
        cards.append(f"""
<div class="panel">
  <h3>{h(r['method'])} | {h(r['line'])}</h3>
  <p>{h(r['line_question'])}</p>
  <p>{badge(str(r['selected_cases']) + ' cases')} {badge(h(r['dominant_failure_mode']), 'bad' if r['dominant_failure_mode'] != 'No clear degradation' else 'neutral')}</p>
  <p>{h(r['interpretation'])}</p>
  <div class="links">{links}</div>
</div>
""")
    ex_links = "".join(f'<li><a href="{h(ex["dashboard"])}">{h(ex["label"])}: {h(ex["method"])} {h(ex["line"])} frame {h(ex["frame_id"])}</a></li>' for ex in examples)
    return page("Upsampling Detection Failure Explanation v2", f"""
<header>
  <h1>Upsampling Detection Failure Explanation v2</h1>
  <p class="muted">Integrated visual and quantitative explanation of why point-cloud upsampling does not consistently improve PointRCNN detection.</p>
  {badge('AP-consistent', 'ok')}{badge('qualitative-only', 'warning')}{badge('missing provenance', 'error')}
</header>
<main>
  <section class="callout">This package connects point-cloud changes, GT-vs-prediction matching, TP/FP/FN, IoU, confidence, and cautious interpretation for thesis and supervisor presentation.</section>
  <h2>Summary Reports</h2>
  <div class="panel links">
    <a href="failure_analysis_summary.html">HTML summary</a>
    <a href="failure_analysis_summary.md">Markdown summary</a>
    <a href="thesis_ready_findings.md">Thesis-ready findings</a>
    <a href="meeting_ready_examples.md">Meeting-ready examples</a>
    <a href="input_and_provenance_audit.md">Provenance audit</a>
    <a href="case_level_failure_analysis.csv">Case CSV</a>
    <a href="object_level_failure_analysis.csv">Object CSV</a>
    <a href="method_line_summary.csv">Method-line CSV</a>
  </div>
  <h2>Meeting-Ready Examples</h2>
  <div class="panel"><ul>{ex_links}</ul></div>
  <h2>Method and Line Navigation</h2>
  <div class="grid">{''.join(cards)}</div>
</main>
""")


def validate_outputs(case_rows: list[dict], object_rows: list[dict], method_rows: list[dict], missing_links: list[str]) -> dict:
    counts = Counter()
    for r in case_rows:
        outdir = OUT / r["method"] / r["line"] / f"frame_{r['frame_id']}"
        for fn in ["analysis_dashboard.html", "evaluation_fov_annotated.html", "pointcloud_change_view.html", "detector_change_view.html", "full_lidar_context_annotated.html", "root_cause_summary.html", "metadata.json"]:
            if not (outdir / fn).exists():
                counts["missing_required_files"] += 1
        if inum(r["baseline_tp"]) + inum(r["baseline_fp"]) != inum(r["baseline_prediction_count"]):
            counts["baseline_count_mismatch"] += 1
        if inum(r["upsampled_tp"]) + inum(r["upsampled_fp"]) != inum(r["upsampled_prediction_count"]):
            counts["upsampled_count_mismatch"] += 1
        if inum(r["baseline_tp"]) + inum(r["baseline_fn"]) != inum(r["gt_car_count"]):
            counts["baseline_gt_count_mismatch"] += 1
        if inum(r["upsampled_tp"]) + inum(r["upsampled_fn"]) != inum(r["gt_car_count"]):
            counts["upsampled_gt_count_mismatch"] += 1
    counts["missing_local_links"] = len(set(missing_links))
    counts["method_summaries"] = len(method_rows)
    counts["object_rows"] = len(object_rows)
    counts["package_size_bytes"] = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    return counts


def main() -> None:
    counts = build_package()
    print(json.dumps(counts, indent=2, sort_keys=True))
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
