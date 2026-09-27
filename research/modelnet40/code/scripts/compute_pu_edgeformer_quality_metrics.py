#!/usr/bin/env python3
"""Compute ModelNet40 geometry metrics for PU-EdgeFormer only (Line A + Line B).

Reuses compute_modelnet40_x4_quality_metrics primitives vs dense mesh GT.
Does not recompute other methods. Does not start PointNet++ / detector.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from compute_modelnet40_x4_quality_metrics import (  # noqa: E402
    SourceGroup,
    compute_group_metrics,
    load_manifest_rows,
    rel_key,
    summarize_rows,
    write_csv,
)
from modelnet40_x4_protocol import (  # noqa: E402
    detect_original_point_count,
    lineA_paths,
    lineB_paths,
    reports_dir,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def edgeformer_groups(n: int) -> list[SourceGroup]:
    return [
        SourceGroup(
            "downsampled_x4_up_pu_edgeformer_vs_gt",
            "lineB",
            "pu_edgeformer",
            lineB_paths("pu_edgeformer")["strict_N"],
            n,
            "main",
        ),
        SourceGroup(
            "original_up_pu_edgeformer_vs_gt",
            "lineA",
            "pu_edgeformer",
            lineA_paths("pu_edgeformer")["strict_4N"],
            n * 4,
            "supplementary",
        ),
    ]


def build_thesis_rows_with_edgeformer(ef_sum: list[dict]) -> list[dict]:
    """Merge existing thesis table with EdgeFormer absolute + delta-vs-original."""
    rep = reports_dir()
    base_csv = rep / "modelnet40_quality_metrics_thesis_table.csv"
    with open(base_csv, newline="", encoding="utf-8") as handle:
        existing = list(csv.DictReader(handle))

    # Drop any previous EdgeFormer rows if re-run
    existing = [
        r
        for r in existing
        if "PU-EdgeFormer" not in r.get("group", "") and "PU-EdgeFormer" not in r.get("method", "")
    ]

    by_group = {r["group"]: r for r in existing}
    original = by_group["Original baseline"]
    o_cd = float(original["CD"])
    o_hd = float(original["HD"])
    o_p2f = float(original["exact_P2F"])
    o_nuc = float(original["NUC"])

    ef_by = {r["group"]: r for r in ef_sum}

    def make_row(label: str, method: str, src_pts: int, gkey: str) -> dict:
        s = ef_by[gkey]
        cd = float(s["cd_mean"])
        hd = float(s["hd_mean"])
        p2f = float(s["p2f_mean"])
        nuc = float(s["nuc_mean"])
        return {
            "group": label,
            "method": method,
            "source_point_count": src_pts,
            "CD": cd,
            "delta_CD_vs_original": cd - o_cd,
            "HD": hd,
            "delta_HD_vs_original": hd - o_hd,
            "exact_P2F": p2f,
            "delta_P2F_vs_original": p2f - o_p2f,
            "NUC": nuc,
            "delta_NUC_vs_original": nuc - o_nuc,
            "valid_samples": int(float(s["valid_samples"])),
        }

    new_rows = list(existing)
    # Insert Line B EdgeFormer after PU-GCN Line B row
    lineb_ef = make_row(
        "Downsampled x4 + PU-EdgeFormer",
        "PU-EdgeFormer",
        1024,
        "downsampled_x4_up_pu_edgeformer_vs_gt",
    )
    linea_ef = make_row(
        "Original + PU-EdgeFormer",
        "PU-EdgeFormer",
        4096,
        "original_up_pu_edgeformer_vs_gt",
    )

    out: list[dict] = []
    inserted_b = inserted_a = False
    for r in new_rows:
        out.append(r)
        if r["group"] == "Downsampled x4 + PU-GCN" and not inserted_b:
            out.append(lineb_ef)
            inserted_b = True
        if r["group"] == "Original + PU-GCN" and not inserted_a:
            out.append(linea_ef)
            inserted_a = True
    if not inserted_b:
        out.append(lineb_ef)
    if not inserted_a:
        out.append(linea_ef)
    return out


def write_md_table(path: Path, rows: list[dict], title: str) -> None:
    cols = list(rows[0].keys())
    lines = [f"# {title}", "", f"- Generated: `{utc_now()}`", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_extended_summary(rows: list[dict]) -> None:
    rep = reports_dir()
    csv_path = rep / "modelnet40_quality_metrics_extended_comparison_summary_with_pu_edgeformer.csv"
    md_path = rep / "modelnet40_quality_metrics_extended_comparison_summary_with_pu_edgeformer.md"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# ModelNet40 Geometry Extended Comparison (with PU-EdgeFormer)",
        "",
        f"- Generated: `{utc_now()}`",
        "- Absolute metrics computed vs **dense mesh surface reference** (mesh / GT reference).",
        "- **Geometry deltas** = method_metric − **Original baseline** metric (Line B geometry recovery).",
        "- Original baseline is **not** the mesh GT, and is **not** the primary Line B classification reference.",
        "- Line B classification primary delta (when available) uses **Downsampled ×4 baseline 256**;",
        "  secondary `gap_accuracy_vs_original_baseline` is separate.",
        "- See `reports/modelnet40_lineB_comparison_logic_geometry_vs_classification.md`.",
        "",
        "| group | method | pts | CD | ΔCD | HD | ΔHD | exact P2F | ΔP2F | NUC | ΔNUC | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['group']} | {r['method']} | {r['source_point_count']} | "
            f"{float(r['CD']):.6f} | {float(r['delta_CD_vs_original']):.6f} | "
            f"{float(r['HD']):.6f} | {float(r['delta_HD_vs_original']):.6f} | "
            f"{float(r['exact_P2F']):.6f} | {float(r['delta_P2F_vs_original']):.6f} | "
            f"{float(r['NUC']):.6f} | {float(r['delta_NUC_vs_original']):.6f} | "
            f"{r['valid_samples']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_geom_vs_clf_pending(rows: list[dict]) -> None:
    rep = reports_dir()
    path = rep / "modelnet40_geometry_vs_classification_with_pu_edgeformer_pending_classification.md"
    lineb = [r for r in rows if r["group"].startswith("Downsampled x4 + ")]
    lines = [
        "# Geometry vs Classification (PU-EdgeFormer classification pending)",
        "",
        f"- Generated: `{utc_now()}`",
        "- Comparison logic: `reports/modelnet40_lineB_comparison_logic_geometry_vs_classification.md`",
        "",
        "## Comparison rules (do not mix)",
        "",
        "1. **Geometry delta (Line B):** vs **Original baseline 1024**",
        "2. **Classification primary delta (Line B):** vs **Downsampled ×4 baseline 256**",
        "   (`delta_accuracy_vs_downsampled_baseline`)",
        "3. **Classification secondary gap (Line B):** vs **Original baseline 1024**",
        "   (`gap_accuracy_vs_original_baseline`)",
        "",
        "Absolute geometry metrics use dense mesh surface / mesh / GT reference.",
        "",
        "## Line B geometry deltas vs Original baseline 1024",
        "",
        "| Method | ΔCD vs Original | ΔHD vs Original | ΔNUC vs Original | Δ exact P2F vs Original | Geometry status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in lineb:
        lines.append(
            f"| {r['method']} | {float(r['delta_CD_vs_original']):.6f} | "
            f"{float(r['delta_HD_vs_original']):.6f} | {float(r['delta_NUC_vs_original']):.6f} | "
            f"{float(r['delta_P2F_vs_original']):.6f} | available |"
        )
    lines.extend(
        [
            "",
            "## Line B classification columns (schema reserved; PU-EdgeFormer pending)",
            "",
            "| Method | Acc | delta_accuracy_vs_downsampled_baseline (primary) | gap_accuracy_vs_original_baseline (secondary) | Classification status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for r in lineb:
        if r["method"] == "PU-EdgeFormer":
            lines.append("| PU-EdgeFormer | — | pending | pending | **pending** (PointNet++ not started) |")
        else:
            lines.append(
                f"| {r['method']} | available | available | available | existing run |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "- Primary Line B classification ranking must use `delta_accuracy_vs_downsampled_baseline`.",
            "- `gap_accuracy_vs_original_baseline` is an additional recovery-gap column only.",
            "- Do not invent PU-EdgeFormer accuracy.",
            "- POINTNET_CLASSIFIER_STARTED=NO",
            "- DETECTOR_EVAL_STARTED=NO",
            "- KITTI_AP_EVAL_STARTED=NO",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p2f", choices=("exact", "approx", "skip"), default="exact")
    parser.add_argument("--metrics-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    n = detect_original_point_count()
    groups = edgeformer_groups(n)
    for g in groups:
        if not g.source_root.is_dir():
            print(f"Missing source root: {g.source_root}")
            return 1

    off_map = {
        rel_key(r["split"], r["class_name"], r["shape_id"]): r["source_off"]
        for r in load_manifest_rows()
    }
    max_samples = 20 if args.smoke else args.max_samples
    p2f_mode = "skip" if args.p2f == "skip" else args.p2f

    all_rows: list[dict] = []
    for group in groups:
        print(f"Computing {group.group} ...", flush=True)
        rows = compute_group_metrics(
            group,
            off_map,
            max_samples,
            p2f_mode,
            workers=1 if args.smoke else args.metrics_workers,
        )
        print(f"  {len(rows)} samples", flush=True)
        all_rows.extend(rows)

    rep = reports_dir()
    per_csv = rep / "modelnet40_pu_edgeformer_quality_metrics_per_sample.csv"
    sum_csv = rep / "modelnet40_pu_edgeformer_quality_metrics_summary.csv"
    write_csv(per_csv, all_rows)
    summaries = summarize_rows(all_rows)
    write_csv(sum_csv, summaries)

    thesis_rows = build_thesis_rows_with_edgeformer(summaries)
    thesis_csv = rep / "modelnet40_quality_metrics_thesis_table_with_pu_edgeformer.csv"
    thesis_md = rep / "modelnet40_quality_metrics_thesis_table_with_pu_edgeformer.md"
    with open(thesis_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(thesis_rows[0].keys()))
        writer.writeheader()
        writer.writerows(thesis_rows)
    write_md_table(thesis_md, thesis_rows, "ModelNet40 Quality Metrics Thesis Table (with PU-EdgeFormer)")
    write_extended_summary(thesis_rows)
    write_geom_vs_clf_pending(thesis_rows)

    print(f"Wrote {per_csv}")
    print(f"Wrote {sum_csv}")
    print(f"Wrote {thesis_csv}")
    print(f"GEOMETRY_METRICS_STARTED=YES (PU-EdgeFormer only)")
    print("POINTNET_CLASSIFIER_STARTED=NO")
    print("DETECTOR_EVAL_STARTED=NO")
    print(f"Finished {utc_now()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
