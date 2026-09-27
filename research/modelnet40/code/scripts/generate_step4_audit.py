#!/usr/bin/env python3
"""Generate Step 4 original baseline audit from training logs and result files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = PROJECT_ROOT / "logs" / "original_baseline.log"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "original_baseline"
DEFAULT_RESULT_CSV = PROJECT_ROOT / "reports" / "original_baseline_result.csv"
DEFAULT_RESULT_MD = PROJECT_ROOT / "reports" / "original_baseline_result.md"
DEFAULT_AUDIT = PROJECT_ROOT / "reports" / "step4_original_baseline_audit.md"


def parse_epochs(log_path: Path) -> list[dict]:
    pattern = re.compile(
        r"Epoch (\d+)/(\d+) train_acc=([\d.]+) test_overall=([\d.]+) test_class=([\d.]+)"
    )
    rows: list[dict] = []
    if not log_path.is_file():
        return rows
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            rows.append(
                {
                    "epoch": int(match.group(1)),
                    "total_epochs": int(match.group(2)),
                    "train_acc": float(match.group(3)),
                    "test_overall": float(match.group(4)),
                    "test_class": float(match.group(5)),
                }
            )
    return rows


def read_result_csv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    result: dict[str, str] = {}
    for row in rows:
        if len(row) >= 2 and row[0] and row[0] != "metric":
            result[row[0]] = row[1]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-csv", type=Path, default=DEFAULT_RESULT_CSV)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args()

    epochs = parse_epochs(args.log_file)
    result_csv = read_result_csv(args.result_csv)
    checkpoint = args.output_dir / "checkpoints" / "best_model.pth"
    metrics_json = args.output_dir / "metrics.json"

    best_epoch_log = max(epochs, key=lambda x: x["test_overall"]) if epochs else None
    final_epoch_log = epochs[-1] if epochs else None
    total_target = final_epoch_log["total_epochs"] if final_epoch_log else 200

    if metrics_json.is_file():
        with open(metrics_json, encoding="utf-8") as handle:
            metrics = json.load(handle)
    else:
        metrics = {}

    errors: list[str] = []
    if args.log_file.is_file():
        text = args.log_file.read_text(encoding="utf-8", errors="replace").lower()
        for token in ("traceback", "cuda out of memory", "nan", "error"):
            if token in text and "no error" not in text:
                if token == "error" and "0.0000" in text:
                    continue
                errors.append(token)

    training_finished = "Training finished" in args.log_file.read_text(encoding="utf-8", errors="replace") if args.log_file.is_file() else False
    completed_epochs = len(epochs)
    status = "COMPLETED" if training_finished and completed_epochs >= total_target else "IN_PROGRESS"

    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# Step 4 — Original Baseline Audit",
        "",
        f"- Generated at: {now}",
        f"- Experiment: `original_baseline`",
        f"- Training status: **{status}**",
        "",
        "## Experiment Config",
        "",
        "| Field | Value |",
        "| --- | --- |",
        "| dataset | `datasets/modelnet40_original` |",
        "| model | `pointnet2_cls_ssg` |",
        "| input points | 1024 |",
        "| train samples | 9843 |",
        "| test samples | 2468 |",
        "| batch size | 24 |",
        "| epochs | 200 |",
        "| optimizer | Adam |",
        "| learning rate | 0.001 |",
        "| seed | 42 |",
        "",
        "## Progress",
        "",
        f"- Completed epochs (logged): **{completed_epochs} / {total_target}**",
        f"- Checkpoint exists: **{checkpoint.is_file()}**",
        f"- Result CSV exists: **{args.result_csv.is_file()}**",
        f"- Result MD exists: **{DEFAULT_RESULT_MD.is_file()}**",
        "",
    ]

    if best_epoch_log:
        lines.extend(
            [
                "## Best epoch (from training log)",
                "",
                f"- best epoch: **{best_epoch_log['epoch']}**",
                f"- best overall accuracy: **{best_epoch_log['test_overall'] * 100:.2f}%**",
                f"- best class accuracy: **{best_epoch_log['test_class'] * 100:.2f}%**",
                "",
            ]
        )

    if final_epoch_log:
        lines.extend(
            [
                "## Latest logged epoch",
                "",
                f"- epoch: **{final_epoch_log['epoch']}**",
                f"- train accuracy: **{final_epoch_log['train_acc'] * 100:.2f}%**",
                f"- test overall accuracy: **{final_epoch_log['test_overall'] * 100:.2f}%**",
                f"- test class accuracy: **{final_epoch_log['test_class'] * 100:.2f}%**",
                "",
            ]
        )

    if result_csv:
        lines.extend(["## Saved result CSV", "", "| metric | value |", "| --- | --- |"])
        for key, value in result_csv.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    lines.extend(
        [
            "## Paths",
            "",
            f"- checkpoint: `{checkpoint}`",
            f"- training log: `{args.log_file}`",
            f"- slurm log: `{PROJECT_ROOT / 'logs'}` (original_baseline_*.log)",
            f"- result csv: `{args.result_csv}`",
            f"- result md: `{DEFAULT_RESULT_MD}`",
            "",
            "## Issues",
            "",
        ]
    )
    if errors:
        for err in errors:
            lines.append(f"- detected in log: `{err}`")
    else:
        lines.append("- No OOM / NaN / traceback detected so far.")

    lines.extend(
        [
            "",
            "## Next step",
            "",
            "Wait for training to finish, then re-run:",
            "",
            "```bash",
            "python scripts/generate_step4_audit.py",
            "```",
            "",
        ]
    )
    args.audit_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit written to: {args.audit_path}")
    print(f"Status: {status} ({completed_epochs}/{total_target} epochs logged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
