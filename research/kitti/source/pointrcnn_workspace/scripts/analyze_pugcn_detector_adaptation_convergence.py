#!/usr/bin/env python3
"""Audit whether the three-epoch PU-GCN detector adaptation actually converged.

PointRCNN scalars are read from its TensorBoard event files.  CenterPoint's
logger prints a reset-per-epoch average in parentheses, so its epoch summaries
are parsed from the final iteration of each epoch.  The report deliberately
distinguishes optimization completion from validation convergence.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = PROJECT_ROOT / "results/pugcn_full_retrain_20260824"


def point_scalars(stage_dir: Path, tag: str = "train_loss"):
    # A few arms were resumed.  For duplicate steps, the value with the latest
    # event wall time is the effective final record.
    by_step = {}
    for event_file in sorted((stage_dir / "tensorboard").glob("events.out.tfevents.*")):
        try:
            for event in summary_iterator(str(event_file)):
                for value in event.summary.value:
                    if value.tag == tag:
                        previous = by_step.get(int(event.step))
                        current = (float(event.wall_time), float(value.simple_value), str(event_file))
                        if previous is None or current[0] >= previous[0]:
                            by_step[int(event.step)] = current
        except Exception as error:  # preserve usable completed event files
            print("EVENT_WARNING %s %r" % (event_file, error))
    return sorted((step, item[1]) for step, item in by_step.items())


def summarize_equal_epochs(values, epochs=3):
    if not values:
        return []
    steps = np.asarray([item[0] for item in values], dtype=np.int64)
    losses = np.asarray([item[1] for item in values], dtype=np.float64)
    first = int(steps.min())
    last = int(steps.max())
    boundaries = np.linspace(first - 1, last, epochs + 1)
    rows = []
    for epoch in range(1, epochs + 1):
        if epoch == 1:
            mask = (steps >= first) & (steps <= boundaries[epoch])
        else:
            mask = (steps > boundaries[epoch - 1]) & (steps <= boundaries[epoch])
        selected = losses[mask]
        selected_steps = steps[mask]
        if not selected.size:
            continue
        tenth = max(1, int(np.ceil(selected.size * 0.1)))
        x = np.arange(selected.size, dtype=np.float64)
        slope = float(np.polyfit(x, selected, 1)[0]) if selected.size > 1 else 0.0
        p99 = float(np.quantile(selected, 0.99))
        trimmed = selected[selected <= p99]
        median = float(np.median(selected))
        rows.append(
            {
                "epoch": epoch,
                "step_first": int(selected_steps[0]),
                "step_last": int(selected_steps[-1]),
                "samples": int(selected.size),
                "loss_mean": float(selected.mean()),
                "loss_median": median,
                "loss_p99_trimmed_mean": float(trimmed.mean()),
                "extreme_outlier_count_gt_100x_median": int(np.count_nonzero(selected > 100.0 * median)),
                "loss_first_10pct_mean": float(selected[:tenth].mean()),
                "loss_last_10pct_mean": float(selected[-tenth:].mean()),
                "loss_linear_slope_per_step": slope,
            }
        )
    return rows


CP_PATTERN = re.compile(
    r"Train:\s+(\d+)/3.*\[(\d+)/(\d+).*Loss:\s+([0-9.eE+-]+)\s+\(([0-9.eE+-]+)\).*LR:\s+([0-9.eE+-]+)"
)


def centerpoint_epochs(log_path: Path):
    last_by_epoch = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = CP_PATTERN.search(line)
        if not match:
            continue
        epoch, iteration, total, instant, average, lr = match.groups()
        epoch = int(epoch)
        row = {
            "epoch": epoch,
            "iteration": int(iteration),
            "iterations": int(total),
            "instant_loss": float(instant),
            "loss_mean": float(average),
            "learning_rate": float(lr),
        }
        if epoch not in last_by_epoch or row["iteration"] >= last_by_epoch[epoch]["iteration"]:
            last_by_epoch[epoch] = row
    return [last_by_epoch[key] for key in sorted(last_by_epoch)]


def relative_change(first: float, last: float) -> float:
    return float((last - first) / first) if first else 0.0


def audit(experiment: Path):
    point_arms = (
        "line_a_pugcn",
        "line_b_pugcn",
        "line_b_baseline",
        "line_a_pugcn_observed_first",
        "line_b_pugcn_observed_first",
    )
    center_arms = ("line_a_pugcn", "line_b_pugcn", "line_b_baseline")
    payload = {"PointRCNN": {}, "CenterPoint": {}}

    for arm in point_arms:
        payload["PointRCNN"][arm] = {}
        for stage in ("rpn", "rcnn"):
            stage_dir = experiment / "pointrcnn" / arm / stage
            values = point_scalars(stage_dir)
            epochs = summarize_equal_epochs(values)
            payload["PointRCNN"][arm][stage] = {
                "event_value_count": len(values),
                "epochs": epochs,
                "median_loss_relative_change_epoch1_to_epoch3": (
                    relative_change(epochs[0]["loss_median"], epochs[-1]["loss_median"])
                    if len(epochs) == 3
                    else None
                ),
            }

    for arm in center_arms:
        output = experiment / "centerpoint" / arm / "openpcdet_output"
        logs = sorted(output.glob("train_*.log"), key=lambda path: path.stat().st_mtime)
        if not logs:
            payload["CenterPoint"][arm] = {"epochs": [], "error": "no training log"}
            continue
        # Prefer the latest log containing a completed third epoch.
        candidates = [(path, centerpoint_epochs(path)) for path in logs]
        completed = [(path, rows) for path, rows in candidates if len(rows) == 3 and rows[-1]["epoch"] == 3]
        log_path, rows = completed[-1] if completed else candidates[-1]
        payload["CenterPoint"][arm] = {
            "log": str(log_path.resolve()),
            "epochs": rows,
            "mean_loss_relative_change_epoch1_to_epoch3": (
                relative_change(rows[0]["loss_mean"], rows[-1]["loss_mean"])
                if len(rows) == 3
                else None
            ),
        }
    return payload


def write_markdown(path: Path, payload) -> None:
    lines = [
        "# Three-epoch detector-adaptation convergence audit",
        "",
        "The three epochs are detector adaptation, not PU-GCN training. PU-GCN uses the fixed PU1K model-100 checkpoint.",
        "",
        "## PointRCNN training losses",
        "",
        "| Arm | Stage | Epoch 1 median | Epoch 2 median | Epoch 3 median | E1→E3 | Outliers (>100×median) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm, stages in payload["PointRCNN"].items():
        for stage, item in stages.items():
            rows = item["epochs"]
            means = [row["loss_median"] for row in rows]
            if len(means) == 3:
                change = item["median_loss_relative_change_epoch1_to_epoch3"] * 100.0
                outliers = sum(row["extreme_outlier_count_gt_100x_median"] for row in rows)
                lines.append("| %s | %s | %.6f | %.6f | %.6f | %+.2f%% | %d |" % (arm, stage, means[0], means[1], means[2], change, outliers))
            else:
                lines.append("| %s | %s | incomplete | incomplete | incomplete | n/a | n/a |" % (arm, stage))
    lines.extend(
        [
            "",
            "## CenterPoint training losses",
            "",
            "| Arm | Epoch 1 mean | Epoch 2 mean | Epoch 3 mean | Final LR | E1→E3 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, item in payload["CenterPoint"].items():
        rows = item.get("epochs", [])
        if len(rows) == 3:
            change = item["mean_loss_relative_change_epoch1_to_epoch3"] * 100.0
            lines.append("| %s | %.6f | %.6f | %.6f | %.3e | %+.2f%% |" % (arm, rows[0]["loss_mean"], rows[1]["loss_mean"], rows[2]["loss_mean"], rows[2]["learning_rate"], change))
        else:
            lines.append("| %s | incomplete | incomplete | incomplete | n/a | n/a |" % arm)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A completed schedule or a decreasing training loss is not proof of validation convergence. Only epoch 3 was checkpointed and there was no per-epoch validation AP, so the present records cannot establish that epoch 3 is the optimum or that AP had plateaued. The supported conclusion is: the three-epoch optimization schedule completed and training losses decreased, but validation convergence was not demonstrated. A convergence claim requires saving/evaluating every epoch, an explicitly declared stopping rule and separation of checkpoint selection from the final evaluation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.experiment.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output_md, payload)
    print("CONVERGENCE_AUDIT_PASS %s" % args.output_md.resolve())


if __name__ == "__main__":
    main()
