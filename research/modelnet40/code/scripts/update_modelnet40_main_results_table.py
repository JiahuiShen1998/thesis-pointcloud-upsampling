#!/usr/bin/env python3
"""Update main ModelNet40 PointNet++ results table from per-experiment summaries."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COLLECT_SCRIPT = PROJECT_ROOT / "scripts" / "collect_pointnet2_training_result.py"
CSV_PATH = PROJECT_ROOT / "reports" / "modelnet40_pointnet2_main_results_status.csv"
MD_PATH = PROJECT_ROOT / "reports" / "modelnet40_pointnet2_main_results_status.md"


@dataclass
class ExperimentSpec:
    name: str
    line: str
    branch: str
    protocol_role: str
    num_points: int | str
    baseline_name: str | None = None
    baseline_acc: float | None = None
    notes_default: str = ""


EXPERIMENTS: list[ExperimentSpec] = [
    ExperimentSpec(
        "original_baseline", "A", "baseline", "main Line A baseline", 1024,
        notes_default="PointNet++ on native 1024 pts",
    ),
    ExperimentSpec(
        "downsampled50_native512_pointnet2", "B", "baseline", "main Line B baseline", 512,
        notes_default="allow_resample=false",
    ),
    ExperimentSpec(
        "downsampled50_baseline", "B", "baseline", "ablation only", 1024,
        notes_default="loader naive resample 512→1024; not main baseline",
    ),
    ExperimentSpec(
        "downsampled50_ear_pointnet2", "B", "upsampling", "ablation/preliminary only", 1024,
        notes_default="legacy EAR ×2 → 1024; superseded by ×4 protocol",
    ),
    ExperimentSpec(
        "original_ear_x4_pointnet2", "A", "upsampling", "main Line A upsampling", 4096,
        baseline_name="original_baseline", baseline_acc=91.95,
        notes_default="EAR ×4 1024→4096",
    ),
    ExperimentSpec(
        "downsampled50_ear_x4_pointnet2", "B", "upsampling", "main Line B upsampling", 2048,
        baseline_name="downsampled50_native512_pointnet2", baseline_acc=91.26,
        notes_default="EAR ×4 512→2048",
    ),
    ExperimentSpec("original_punet_x4_pointnet2", "A", "upsampling", "main Line A upsampling", "", notes_default="blocked"),
    ExperimentSpec("downsampled50_punet_x4_pointnet2", "B", "upsampling", "main Line B upsampling", "", notes_default="blocked"),
    ExperimentSpec("original_pugcn_x4_pointnet2", "A", "upsampling", "main Line A upsampling", "", notes_default="blocked"),
    ExperimentSpec("downsampled50_pugcn_x4_pointnet2", "B", "upsampling", "main Line B upsampling", "", notes_default="blocked"),
    ExperimentSpec("original_pdans_x4_pointnet2", "A", "upsampling", "main Line A upsampling", "", notes_default="blocked"),
    ExperimentSpec("downsampled50_pdans_x4_pointnet2", "B", "upsampling", "main Line B upsampling", "", notes_default="blocked"),
]


def reports_dir_for(experiment: str) -> Path:
    overrides = {
        "downsampled50_ear_pointnet2": PROJECT_ROOT / "reports" / "step8_downsampled50_ear_pointnet2",
    }
    return overrides.get(experiment, PROJECT_ROOT / "reports" / experiment)


def summary_path(experiment: str) -> Path:
    return reports_dir_for(experiment) / "result_summary.csv"


def read_summary(experiment: str) -> dict[str, str] | None:
    path = summary_path(experiment)
    if not path.is_file():
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[0] if rows else None


def run_collector(spec: ExperimentSpec, refresh: bool) -> dict[str, str] | None:
    if not refresh and summary_path(spec.name).is_file():
        return read_summary(spec.name)

    cmd = [sys.executable, str(COLLECT_SCRIPT), spec.name]
    if spec.baseline_name and spec.baseline_acc is not None:
        cmd.extend(["--baseline-name", spec.baseline_name, "--baseline-acc", str(spec.baseline_acc)])

    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"collect failed for {spec.name}: {proc.stderr or proc.stdout}", file=sys.stderr)
        return read_summary(spec.name)
    return read_summary(spec.name)


def fmt_acc(value: str | float | None) -> str:
    if value is None or value == "":
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{v * 100:.4f}" if v <= 1.0 else f"{v:.4f}"


def build_row(spec: ExperimentSpec, summary: dict[str, str] | None) -> dict[str, str]:
    if summary:
        status = summary.get("status", "unknown")
        if status == "running":
            mapped_status = "running"
        elif status == "completed":
            mapped_status = "completed"
        else:
            mapped_status = status
        return {
            "experiment": spec.name,
            "line": spec.line,
            "branch": spec.branch,
            "protocol_role": spec.protocol_role,
            "status": mapped_status,
            "num_points": str(summary.get("num_point") or spec.num_points),
            "best_epoch": summary.get("best_epoch", ""),
            "best_test_acc": fmt_acc(summary.get("best_test_overall_acc")),
            "final_epoch": summary.get("final_epoch", ""),
            "final_test_acc": fmt_acc(summary.get("final_test_overall_acc")),
            "checkpoint": summary.get("checkpoint_path", f"outputs/{spec.name}/checkpoints/best_model.pth"),
            "notes": spec.notes_default,
        }

    # No summary — preserve blocked/pending defaults from existing CSV if present
    existing = _read_existing_csv()
    if spec.name in existing:
        return existing[spec.name]

    status = "blocked" if "blocked" in spec.notes_default else "pending"
    return {
        "experiment": spec.name,
        "line": spec.line,
        "branch": spec.branch,
        "protocol_role": spec.protocol_role,
        "status": status,
        "num_points": str(spec.num_points),
        "best_epoch": "",
        "best_test_acc": "",
        "final_epoch": "",
        "final_test_acc": "",
        "checkpoint": f"outputs/{spec.name}/checkpoints/best_model.pth",
        "notes": spec.notes_default,
    }


def _read_existing_csv() -> dict[str, dict[str, str]]:
    if not CSV_PATH.is_file():
        return {}
    with open(CSV_PATH, newline="", encoding="utf-8") as handle:
        return {row["experiment"]: row for row in csv.DictReader(handle)}


def write_csv(rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "experiment", "line", "branch", "protocol_role", "status", "num_points",
        "best_epoch", "best_test_acc", "final_epoch", "final_test_acc", "checkpoint", "notes",
    ]
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict[str, str]]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    completed = [r for r in rows if r["status"] == "completed"]
    running = [r for r in rows if r["status"] == "running"]
    submitted = [r for r in rows if r["status"] == "submitted"]
    blocked = [r for r in rows if r["status"] == "blocked"]

    lines = [
        "# ModelNet40 PointNet++ — Main Results Status",
        "",
        f"- Updated: {now}",
        f"- CSV: `reports/modelnet40_pointnet2_main_results_status.csv`",
        "",
        "## Completed",
        "",
        "| Experiment | Line | Best acc | Final acc | `num_point` |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in completed:
        best = f"{float(r['best_test_acc'])*100:.2f}%" if r["best_test_acc"] and float(r["best_test_acc"]) <= 1 else (
            f"{r['best_test_acc']}%" if r["best_test_acc"] else "—"
        )
        final = f"{float(r['final_test_acc'])*100:.2f}%" if r["final_test_acc"] and float(r["final_test_acc"]) <= 1 else (
            f"{r['final_test_acc']}%" if r["final_test_acc"] else "—"
        )
        if r["best_test_acc"] and float(r["best_test_acc"]) > 1:
            best = f"{float(r['best_test_acc']):.2f}%"
        if r["final_test_acc"] and float(r["final_test_acc"]) > 1:
            final = f"{float(r['final_test_acc']):.2f}%"
        lines.append(
            f"| {r['experiment']} | {r['line']} | {best} | {final} | {r['num_points']} |"
        )

    if running:
        lines.extend(["", "## Running", ""])
        for r in running:
            lines.append(f"- **{r['experiment']}** (`num_point={r['num_points']}`)")

    if submitted:
        lines.extend(["", "## Submitted / pending start", ""])
        for r in submitted:
            lines.append(f"- **{r['experiment']}**")

    if blocked:
        lines.extend(["", "## Blocked", ""])
        for r in blocked:
            lines.append(f"- **{r['experiment']}**: {r['notes']}")

    lines.extend(
        [
            "",
            "## Main protocol ×4",
            "",
            "| Line | Baseline | Upsampling (EAR ×4) |",
            "| --- | --- | --- |",
        ]
    )
    for line in ("A", "B"):
        base = next((r for r in rows if r["line"] == line and r["branch"] == "baseline" and "ablation" not in r["protocol_role"]), None)
        ups = next((r for r in rows if r["line"] == line and "ear_x4" in r["experiment"]), None)
        b_status = base["status"] if base else "—"
        u_status = ups["status"] if ups else "—"
        lines.append(f"| {line} | {base['experiment'] if base else '—'} ({b_status}) | {ups['experiment'] if ups else '—'} ({u_status}) |")

    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Update main PointNet++ results table.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-run collectors for experiments with baselines / known logs",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Subset of experiment names to refresh (default: all with logs or summaries)",
    )
    args = parser.parse_args()

    specs = EXPERIMENTS
    if args.experiments:
        names = set(args.experiments)
        specs = [s for s in EXPERIMENTS if s.name in names]

    rows: list[dict[str, str]] = []
    for spec in specs:
        summary = None
        if args.refresh or summary_path(spec.name).is_file():
            summary = run_collector(spec, refresh=args.refresh)
        rows.append(build_row(spec, summary))

    # Merge with experiments not in refresh subset
    if args.experiments:
        existing_names = {s.name for s in specs}
        for spec in EXPERIMENTS:
            if spec.name not in existing_names:
                rows.append(build_row(spec, read_summary(spec.name)))

    rows.sort(key=lambda r: EXPERIMENTS.index(next(s for s in EXPERIMENTS if s.name == r["experiment"])))

    write_csv(rows)
    write_md(rows)
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
