#!/usr/bin/env python3
"""Collect mesh-ref PointNet++ baseline metrics and compare to existing branches."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
RESULTS = PROJECT / "pointnet2_results" / "x4_two_line_final"

MESH_BRANCHES = {
    "mesh_ref_baseline_256": RESULTS / "mesh_ref_baseline_256",
    "mesh_ref_baseline_4096": RESULTS / "mesh_ref_baseline_4096",
}

COMPARE = [
    {
        "name": "Mesh-ref 256",
        "points": 256,
        "metrics": RESULTS / "mesh_ref_baseline_256" / "metrics.json",
        "compare_to": "Downsampled ×4 (from Original 1024)",
        "compare_metrics": RESULTS / "lineB_downsampled_x4_baseline" / "metrics.json",
        "compare_points": 256,
    },
    {
        "name": "Mesh-ref 4096",
        "points": 4096,
        "metrics": RESULTS / "mesh_ref_baseline_4096" / "metrics.json",
        "compare_to": "Original 1024 baseline",
        "compare_metrics": RESULTS / "lineA_original_baseline" / "metrics.json",
        "compare_points": 1024,
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_metrics(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float | None) -> str:
    if x is None:
        return "PENDING"
    return f"{100.0 * float(x):.2f}%"


def pp(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "PENDING"
    return f"{100.0 * (float(a) - float(b)):.2f} pp"


def main() -> None:
    rows = []
    pending = []
    for item in COMPARE:
        m = load_metrics(item["metrics"])
        c = load_metrics(item["compare_metrics"])
        if m is None:
            pending.append(item["name"])
        row = {
            "method": item["name"],
            "points": item["points"],
            "best_overall": None if m is None else float(m.get("overall_accuracy", m.get("best_overall_accuracy", 0))),
            "final_overall": None
            if m is None
            else float(m.get("final_test_overall_accuracy", m.get("final_overall_accuracy", 0))),
            "best_epoch": None if m is None else m.get("best_epoch", ""),
            "compare_to": item["compare_to"],
            "compare_points": item["compare_points"],
            "compare_best_overall": None
            if c is None
            else float(c.get("overall_accuracy", c.get("best_overall_accuracy", 0))),
            "delta_best_vs_compare_pp": None
            if m is None or c is None
            else 100.0
            * (
                float(m.get("overall_accuracy", m.get("best_overall_accuracy", 0)))
                - float(c.get("overall_accuracy", c.get("best_overall_accuracy", 0)))
            ),
            "metrics_path": str(item["metrics"]),
            "status": "PENDING" if m is None else "DONE",
        }
        rows.append(row)

    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / "modelnet40_pointnet2_mesh_ref_baseline_results.csv"
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Mesh-ref PointNet++ baseline results",
        "",
        f"- Generated: {utc_now()}",
        "- Model: `pointnet2_cls_ssg`, 200 epochs, Adam lr=0.001, seed=42, from scratch",
        "",
        "| Method | Pts | Best Overall | Final Overall | Best epoch | Compare to | Compare Best | Δ Best | Status |",
        "|---|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['method']} | {r['points']} | {pct(r['best_overall'])} | {pct(r['final_overall'])} | "
            f"{r['best_epoch']} | {r['compare_to']} | {pct(r['compare_best_overall'])} | "
            f"{'PENDING' if r['delta_best_vs_compare_pp'] is None else f'{r['delta_best_vs_compare_pp']:+.2f} pp'} | "
            f"{r['status']} |"
        )
    md += [
        "",
        "## Interpretation notes",
        "",
        "- Mesh-ref 256/4096 are independent area-weighted samples from `.off`, not downsampled from Original 1024.",
        "- Compare Mesh-ref 256 vs existing Downsampled×4 256: same count, different sampling process.",
        "- Compare Mesh-ref 4096 vs Original 1024: different counts; useful as denser mesh-sample baseline for Line A.",
        "",
    ]
    if pending:
        md.append(f"- Still pending: {', '.join(pending)}")
        md.append("")
    md_path = REPORTS / "modelnet40_pointnet2_mesh_ref_baseline_results.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print("pending=", pending)
    if pending:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
