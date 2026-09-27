#!/usr/bin/env python3
"""Compare generated-only and observed-first full PointRCNN adaptation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results/pugcn_full_retrain_20260824"
OLD_PR = ROOT / "results/pointrcnn_finetune64_six_arms_20260824/evaluations"


def metrics(run_dir: Path) -> dict[str, float]:
    logs = sorted(run_dir.rglob("log_eval_one.txt"), key=lambda path: path.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"PointRCNN evaluation log missing under {run_dir}")
    text = logs[-1].read_text(errors="replace")
    match = re.search(
        r"Car AP@0\.70, 0\.70, 0\.70:\s*"
        r"bbox AP:([^\n]+)\s*bev\s+AP:([^\n]+)\s*3d\s+AP:([^\n]+)",
        text,
    )
    if not match:
        raise ValueError(f"Car AP block missing in {logs[-1]}")
    bev = [float(value.strip()) for value in match.group(2).split(",")]
    d3 = [float(value.strip()) for value in match.group(3).split(",")]
    return {
        "3d_easy": d3[0],
        "3d_moderate": d3[1],
        "3d_hard": d3[2],
        "bev_moderate": bev[1],
    }


def main() -> None:
    sources = {
        ("A", "baseline"): OLD_PR / "line_a_baseline/pretrained",
        ("A", "generated-only"): EXP / "evaluations/pointrcnn/line_a_pugcn",
        ("A", "observed-first"): EXP
        / "evaluations/pointrcnn_observed_first/line_a_pugcn_observed_first",
        ("B", "baseline"): EXP / "evaluations/pointrcnn/line_b_baseline",
        ("B", "generated-only"): EXP / "evaluations/pointrcnn/line_b_pugcn",
        ("B", "observed-first"): EXP
        / "evaluations/pointrcnn_observed_first/line_b_pugcn_observed_first",
    }
    values = {key: metrics(path) for key, path in sources.items()}
    rows: list[dict[str, object]] = []
    for line in ("A", "B"):
        baseline = values[(line, "baseline")]["3d_moderate"]
        generated = values[(line, "generated-only")]["3d_moderate"]
        for input_name in ("baseline", "generated-only", "observed-first"):
            item = values[(line, input_name)]
            rows.append(
                {
                    "line": line,
                    "input": input_name,
                    **item,
                    "delta_vs_baseline_3d_moderate": item["3d_moderate"] - baseline,
                    "delta_vs_generated_only_3d_moderate": (
                        item["3d_moderate"] - generated
                        if input_name == "observed-first"
                        else 0.0
                    ),
                }
            )

    report_dir = EXP / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "pointrcnn_observed_first_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (report_dir / "pointrcnn_observed_first_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# PointRCNN PU-GCN observed-first adaptation",
        "",
        "All detector adaptations use all 3712 training frames. Evaluation uses the fixed 256-frame split.",
        "",
        "| Line | Input | 3D Easy | 3D Moderate | 3D Hard | BEV Moderate | Delta vs baseline | Delta vs generated-only |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {line} | {input} | {3d_easy:.4f} | {3d_moderate:.4f} | "
            "{3d_hard:.4f} | {bev_moderate:.4f} | "
            "{delta_vs_baseline_3d_moderate:+.4f} | "
            "{delta_vs_generated_only_3d_moderate:+.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Observed-first keeps every observed point, then deterministically selects generated points to preserve the same strict-4x point budget.",
        ]
    )
    report = report_dir / "pointrcnn_observed_first_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
