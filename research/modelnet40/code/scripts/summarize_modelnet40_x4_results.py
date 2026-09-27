#!/usr/bin/env python3
"""Summarize ModelNet40 ×4 two-line protocol results into final report."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    protocol_counts,
    reports_dir,
)


def read_csv_if_exists(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    n = detect_original_point_count()
    counts = protocol_counts(n)
    rep = reports_dir()

    audit_md = rep / "modelnet40_x4_data_audit.md"
    lineA_audit = read_csv_if_exists(rep / "modelnet40_lineA_original_up_count_audit.csv")
    lineB_audit = read_csv_if_exists(rep / "modelnet40_lineB_downsampled_x4_up_count_audit.csv")
    quality_summary = read_csv_if_exists(rep / "modelnet40_lineB_quality_metrics_summary.csv")
    classification = read_csv_if_exists(rep / "modelnet40_pointnet2_classification_summary.csv")

    downsampled_reused = DOWNSAMPLED_X4_ROOT.is_dir() and sum(1 for _ in DOWNSAMPLED_X4_ROOT.rglob("*.npy")) >= 12311

    md_path = rep / "modelnet40_x4_final_protocol_report.md"
    lines = [
        "# ModelNet40 ×4 Final Protocol Report",
        "",
        f"- Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- PROJECT_ROOT: `{PROJECT_ROOT}`",
        "",
        "## 1. Protocol overview",
        "",
        "This experiment retains **two main classification lines** plus an independent **quality metrics module**.",
        "",
        "| Line | Baseline | Upsampling branch |",
        "| --- | --- | --- |",
        f"| **Line A** | Original baseline ({n} pts) | Original + Upsampling → {counts['four_N']} pts |",
        f"| **Line B** | Downsampled ×4 baseline ({counts['N_div_4']} pts) | Downsampled ×4 + Upsampling → {n} pts |",
        "",
        "CD / HD / P2F / NUC are **not** a third classification line. They evaluate Line B intermediate",
        "geometry (upsampled point cloud vs original point cloud / CAD mesh) **before** PointNet++.",
        "",
        "## 2. Point counts (detected from data)",
        "",
        f"- N (original baseline): **{n}**",
        f"- N/4 (downsampled ×4): **{counts['N_div_4']}**",
        f"- 4N (Line A upsampling output): **{counts['four_N']}**",
        f"- N (Line B upsampling output): **{n}**",
        "",
        "## 3. Downsampled ×4 database",
        "",
        f"- Path: `{DOWNSAMPLED_X4_ROOT}`",
        f"- Reused existing: **{downsampled_reused}**",
        "",
        "Legacy `modelnet40_downsampled50` (512 pts, N/2) is preserved but **not** used as x4 downsample.",
        "",
        "## 4. Line A method status",
        "",
        "| method | train | test | expected | status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for r in lineA_audit:
        lines.append(
            f"| {r.get('method','')} | {r.get('train_count','')} | {r.get('test_count','')} | "
            f"{r.get('expected_output','')} | {r.get('status','')} |"
        )

    lines += [
        "",
        "## 5. Line B method status",
        "",
        "| method | train | test | expected | status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for r in lineB_audit:
        lines.append(
            f"| {r.get('method','')} | {r.get('train_count','')} | {r.get('test_count','')} | "
            f"{r.get('expected_output','')} | {r.get('status','')} |"
        )

    lines += [
        "",
        "## 6. Quality metrics (Line B intermediate)",
        "",
        "### Definitions",
        "",
        "- **CD**: Chamfer distance (forward + backward mean squared min-distance)",
        "- **HD**: Hausdorff distance (symmetric max-min distance)",
        "- **P2F**: Mean/median/95th/max distance from upsampled points to original CAD .off mesh surface",
        "- **NUC**: Neighborhood Uniformity Coefficient — CV of neighbor counts at radii (0.02, 0.05, 0.10); lower = more uniform",
        "",
        "### Summary",
        "",
    ]
    if quality_summary:
        lines.append("| group | method | split | source_pts | target_pts | cd_mean | hd_mean | p2f_mean | nuc_mean |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in quality_summary:
            lines.append(
                f"| {r.get('comparison_group','')} | {r.get('method','')} | {r.get('split','')} | "
                f"{r.get('source_point_count','')} | {r.get('target_point_count','')} | "
                f"{r.get('cd_mean','')} | {r.get('hd_mean','')} | {r.get('p2f_mean','')} | {r.get('nuc_mean','')} |"
            )
    else:
        lines.append("*Quality metrics pending — run `compute_lineB_quality_metrics.py`*")

    lines += [
        "",
        "## 7. PointNet++ classification summary",
        "",
    ]
    if classification:
        lines.append("| line | branch | method | input_pts | upsampling | accuracy | status |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
        for r in classification:
            lines.append(
                f"| {r.get('line','')} | {r.get('branch','')} | {r.get('method','')} | "
                f"{r.get('input_point_count','')} | {r.get('upsampling_factor','')} | "
                f"{r.get('overall_accuracy','')} | {r.get('status','')} |"
            )
    else:
        lines.append("*Classification summary pending*")

    lines += [
        "",
        "## 8. Difference from old protocol",
        "",
        "| Aspect | Old protocol | New protocol |",
        "| --- | --- | --- |",
        f"| Line B downsample | N/2 = 512 | N/4 = {counts['N_div_4']} |",
        f"| Line B upsampling output | 2048 | {n} |",
        f"| Line A upsampling output | 4096 | {counts['four_N']} (unchanged) |",
        "| Quality metrics | None | CD/HD/P2F/NUC on Line B intermediate |",
        "",
        "## 9. Failures / missing",
        "",
    ]
    lineB_pending = [r for r in lineB_audit if r.get("status") != "complete"]
    if lineB_pending:
        lines.append("Line B upsampling pending for: " + ", ".join(r.get("method", "") for r in lineB_pending))
    pugcn_a = next((r for r in lineA_audit if "GCN" in r.get("method", "")), None)
    if pugcn_a and pugcn_a.get("status") != "reused_legacy":
        lines.append("- Line A PU-GCN: not fully generated")
    lines.append("- Line B PointNet++: requires retraining on new 256/1024 protocol (legacy 512/2048 results preserved)")

    lines += [
        "",
        "## 10. Thesis conclusion draft",
        "",
        f"We evaluate point cloud upsampling on ModelNet40 with two parallel lines. "
        f"Line A compares native {n}-point inputs against {counts['four_N']}-point upsampled clouds. "
        f"Line B first downsamples to {counts['N_div_4']} points (×4 reduction), then upsamples back to {n} points. "
        f"Geometric fidelity of Line B upsampling is quantified independently via Chamfer/Hausdorff distances, "
        f"point-to-surface error against CAD meshes, and neighborhood uniformity (NUC). "
        f"Classification with PointNet++ is reported separately for each line and method.",
        "",
        "## File index",
        "",
        f"- Data audit: `{rep / 'modelnet40_x4_data_audit.md'}`",
        f"- Line A count audit: `{rep / 'modelnet40_lineA_original_up_count_audit.md'}`",
        f"- Line B count audit: `{rep / 'modelnet40_lineB_downsampled_x4_up_count_audit.md'}`",
        f"- Quality per-sample: `{rep / 'modelnet40_lineB_quality_metrics_per_sample.csv'}`",
        f"- Quality summary: `{rep / 'modelnet40_lineB_quality_metrics_summary.csv'}`",
        f"- Classification summary: `{rep / 'modelnet40_pointnet2_classification_summary.csv'}`",
        f"- PointNet++ inputs: `{PROJECT_ROOT / 'pointnet2_inputs'}`",
        f"- PointNet++ results: `{PROJECT_ROOT / 'pointnet2_results'}`",
    ]

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
