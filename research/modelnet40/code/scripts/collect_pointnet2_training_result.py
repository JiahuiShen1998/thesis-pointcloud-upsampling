#!/usr/bin/env python3
"""Collect PointNet++ training results for a single experiment."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+) train_acc=([\d.]+) test_overall=([\d.]+) test_class=([\d.]+)"
)
FINISHED_RE = re.compile(
    r"Training finished\. best_overall=([\d.]+) best_class=([\d.]+)"
)
ARGS_EPOCH_RE = re.compile(r"epoch=(\d+)")
LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ - ")


@dataclass
class ExperimentLayout:
    experiment: str
    log_paths: list[Path]
    output_dir: Path
    reports_dir: Path
    config_paths: list[Path]
    checkpoint_path: Path


# Legacy / non-standard directory layouts.
LAYOUT_OVERRIDES: dict[str, dict[str, Any]] = {
    "original_baseline": {
        "log_paths": ["logs/original_baseline.log", "logs/original_baseline/train.log"],
        "reports_dir": "reports/original_baseline",
    },
    "downsampled50_baseline": {
        "log_paths": ["logs/downsampled50_baseline.log", "logs/downsampled50_baseline/train.log"],
        "reports_dir": "reports/downsampled50_baseline",
    },
    "downsampled50_ear_pointnet2": {
        "log_dir": "logs/step8_downsampled50_ear_pointnet2",
        "output_dir": "outputs/step8_downsampled50_ear_pointnet2",
        "reports_dir": "reports/step8_downsampled50_ear_pointnet2",
        "config_paths": ["configs/downsampled50_ear_pointnet2.yaml"],
    },
}


@dataclass
class EpochMetrics:
    epoch: int
    total_epochs: int
    train_acc: float
    test_overall: float
    test_class: float


@dataclass
class ParsedLog:
    epochs: list[EpochMetrics] = field(default_factory=list)
    target_epochs: int = 200
    training_finished: bool = False
    finished_best_overall: float | None = None
    finished_best_class: float | None = None
    start_time: str = ""
    end_time: str = ""
    num_point: int | None = None
    allow_resample: bool | None = None


def resolve_layout(experiment: str) -> ExperimentLayout:
    override = LAYOUT_OVERRIDES.get(experiment, {})
    log_dir = PROJECT_ROOT / override.get("log_dir", f"logs/{experiment}")
    output_dir = PROJECT_ROOT / override.get("output_dir", f"outputs/{experiment}")
    reports_dir = PROJECT_ROOT / override.get("reports_dir", f"reports/{experiment}")
    checkpoint_path = output_dir / "checkpoints" / "best_model.pth"

    log_paths: list[Path] = []
    for rel in override.get("log_paths", []):
        log_paths.append(PROJECT_ROOT / rel)
    standard = log_dir / "train.log"
    if standard not in log_paths:
        log_paths.insert(0, standard)
    for candidate in sorted(log_dir.glob("train_*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        if candidate not in log_paths:
            log_paths.append(candidate)

    config_paths: list[Path] = []
    for rel in override.get("config_paths", [f"configs/{experiment}.yaml"]):
        path = PROJECT_ROOT / rel
        if path.is_file():
            config_paths.append(path)
    default_cfg = PROJECT_ROOT / "configs" / f"{experiment}.yaml"
    if default_cfg.is_file() and default_cfg not in config_paths:
        config_paths.append(default_cfg)

    return ExperimentLayout(
        experiment=experiment,
        log_paths=log_paths,
        output_dir=output_dir,
        reports_dir=reports_dir,
        config_paths=config_paths,
        checkpoint_path=checkpoint_path,
    )


def pick_log_file(layout: ExperimentLayout) -> Path | None:
    best: Path | None = None
    best_score = -1
    for path in layout.log_paths:
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        epoch_count = len(EPOCH_RE.findall(text))
        score = epoch_count * 1_000_000 + path.stat().st_size
        if score > best_score:
            best_score = score
            best = path
    return best


def parse_log(path: Path) -> ParsedLog:
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = ParsedLog()

    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            parsed.epochs.append(
                EpochMetrics(
                    epoch=int(m.group(1)),
                    total_epochs=int(m.group(2)),
                    train_acc=float(m.group(3)),
                    test_overall=float(m.group(4)),
                    test_class=float(m.group(5)),
                )
            )
            parsed.target_epochs = int(m.group(2))
            ts = LOG_TS_RE.match(line)
            if ts:
                if not parsed.start_time:
                    parsed.start_time = ts.group(1)
                parsed.end_time = ts.group(1)

        if "num_point=" in line and parsed.num_point is None:
            np_m = re.search(r"num_point=(\d+)", line)
            if np_m:
                parsed.num_point = int(np_m.group(1))
            ar_m = re.search(r"allow_resample=(True|False)", line)
            if ar_m:
                parsed.allow_resample = ar_m.group(1) == "True"

        if "epoch=" in line and "Namespace" in line:
            em = ARGS_EPOCH_RE.search(line)
            if em:
                parsed.target_epochs = int(em.group(1))

        fm = FINISHED_RE.search(line)
        if fm:
            parsed.training_finished = True
            parsed.finished_best_overall = float(fm.group(1))
            parsed.finished_best_class = float(fm.group(2))
            ts = LOG_TS_RE.match(line)
            if ts:
                parsed.end_time = ts.group(1)

    return parsed


def load_config(config_paths: list[Path]) -> dict[str, Any]:
    if not config_paths:
        return {}
    with open(config_paths[0], encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def normalize_acc(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 100.0 if value <= 1.0 else value


def compute_best_final(parsed: ParsedLog) -> dict[str, Any]:
    if not parsed.epochs:
        return {
            "best_epoch": None,
            "best_test_overall_acc": None,
            "best_test_class_acc": None,
            "final_epoch": None,
            "final_test_overall_acc": None,
            "final_test_class_acc": None,
        }

    best = max(parsed.epochs, key=lambda e: e.test_overall)
    final = parsed.epochs[-1]
    return {
        "best_epoch": best.epoch,
        "best_test_overall_acc": best.test_overall,
        "best_test_class_acc": best.test_class,
        "final_epoch": final.epoch,
        "final_test_overall_acc": final.test_overall,
        "final_test_class_acc": final.test_class,
    }


def is_complete(parsed: ParsedLog) -> bool:
    if parsed.training_finished:
        return True
    if not parsed.epochs:
        return False
    return parsed.epochs[-1].epoch >= parsed.target_epochs


def pct_str(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{normalize_acc(value):.2f}%"


def write_monitoring_snapshot(
    layout: ExperimentLayout,
    log_path: Path,
    parsed: ParsedLog,
    metrics: dict[str, Any],
    config: dict[str, Any],
    baseline_name: str | None,
    baseline_acc: float | None,
) -> Path:
    layout.reports_dir.mkdir(parents=True, exist_ok=True)
    out = layout.reports_dir / "monitoring_snapshot.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    last = parsed.epochs[-1] if parsed.epochs else None

    lines = [
        f"# {layout.experiment} — Monitoring Snapshot",
        "",
        f"- Generated at: {now}",
        f"- Status: **running**",
        "",
        "## Progress",
        "",
        f"| Field | Value |",
        f"| --- | --- |",
        f"| target epochs | {parsed.target_epochs} |",
        f"| current epoch | {last.epoch if last else 0} |",
        f"| best epoch (so far) | {metrics.get('best_epoch', '—')} |",
        f"| best test overall (so far) | {pct_str(metrics.get('best_test_overall_acc'))} |",
        f"| best test class (so far) | {pct_str(metrics.get('best_test_class_acc'))} |",
        f"| latest test overall | {pct_str(metrics.get('final_test_overall_acc'))} |",
        f"| log | `{log_path.relative_to(PROJECT_ROOT)}` |",
        f"| checkpoint exists | {layout.checkpoint_path.is_file()} |",
        "",
        "## Note",
        "",
        "Training not complete — `final_report.md` not generated.",
        "Re-run after epoch "
        f"{parsed.target_epochs} or when log contains `Training finished`.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_final_report(
    layout: ExperimentLayout,
    log_path: Path,
    parsed: ParsedLog,
    metrics: dict[str, Any],
    config: dict[str, Any],
    baseline_name: str | None,
    baseline_acc: float | None,
    delta_pct: float | None,
) -> Path:
    layout.reports_dir.mkdir(parents=True, exist_ok=True)
    out = layout.reports_dir / "final_report.md"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    cfg_path = layout.config_paths[0] if layout.config_paths else None

    lines = [
        f"# {layout.experiment} — Final Report",
        "",
        f"- Generated at: {now}",
        f"- Experiment: **{layout.experiment}**",
        f"- Status: **completed**",
        "",
        "## Training results",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Best test overall accuracy | **{pct_str(metrics['best_test_overall_acc'])}** (epoch {metrics['best_epoch']}) |",
        f"| Best test class accuracy | {pct_str(metrics['best_test_class_acc'])} |",
        f"| Final test overall accuracy | {pct_str(metrics['final_test_overall_acc'])} (epoch {metrics['final_epoch']}) |",
        f"| Final test class accuracy | {pct_str(metrics['final_test_class_acc'])} |",
        f"| Target epochs | {parsed.target_epochs} |",
        f"| Training start | {parsed.start_time or '—'} |",
        f"| Training end | {parsed.end_time or '—'} |",
        "",
        "## Artifacts",
        "",
        "| Type | Path |",
        "| --- | --- |",
        f"| Log | `{log_path}` |",
        f"| Checkpoint | `{layout.checkpoint_path}` |",
        f"| Config | `{cfg_path}` |" if cfg_path else "| Config | — |",
        f"| Output dir | `{layout.output_dir}` |",
        "",
    ]

    if baseline_name and baseline_acc is not None:
        lines.extend(
            [
                "## Baseline comparison",
                "",
                f"| Field | Value |",
                f"| --- | --- |",
                f"| Baseline | `{baseline_name}` |",
                f"| Baseline best acc | {baseline_acc:.2f}% |",
                f"| This experiment best acc | {pct_str(metrics['best_test_overall_acc'])} |",
                f"| Delta vs baseline | {delta_pct:+.2f} pp |" if delta_pct is not None else "",
                "",
            ]
        )

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_result_summary_csv(
    layout: ExperimentLayout,
    log_path: Path,
    status: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
    baseline_name: str | None,
    baseline_acc: float | None,
    delta_pct: float | None,
) -> Path:
    layout.reports_dir.mkdir(parents=True, exist_ok=True)
    out = layout.reports_dir / "result_summary.csv"
    cfg_path = layout.config_paths[0] if layout.config_paths else ""

    row = {
        "experiment": layout.experiment,
        "status": status,
        "best_test_overall_acc": metrics.get("best_test_overall_acc"),
        "best_test_class_acc": metrics.get("best_test_class_acc"),
        "best_epoch": metrics.get("best_epoch"),
        "final_epoch": metrics.get("final_epoch"),
        "final_test_overall_acc": metrics.get("final_test_overall_acc"),
        "final_test_class_acc": metrics.get("final_test_class_acc"),
        "checkpoint_path": str(layout.checkpoint_path),
        "log_path": str(log_path),
        "config_path": str(cfg_path),
        "num_point": config.get("num_point") or metrics.get("num_point"),
        "allow_resample": config.get("allow_resample", metrics.get("allow_resample")),
        "training_start": metrics.get("training_start", ""),
        "training_end": metrics.get("training_end", ""),
        "baseline_name": baseline_name or "",
        "baseline_acc_pct": baseline_acc if baseline_acc is not None else "",
        "delta_vs_baseline_pp": delta_pct if delta_pct is not None else "",
    }

    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return out


def parse_baseline_acc(value: float) -> float:
    """Accept 0.9195 or 91.95, return percent."""
    return value if value > 1.0 else value * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect PointNet++ training result.")
    parser.add_argument("experiment", help="Experiment name, e.g. original_ear_x4_pointnet2")
    parser.add_argument("--baseline-name", default=None)
    parser.add_argument("--baseline-acc", type=float, default=None, help="e.g. 91.95 or 0.9195")
    args = parser.parse_args()

    layout = resolve_layout(args.experiment)
    log_path = pick_log_file(layout)
    if log_path is None:
        print(f"ERROR: no log file found for {args.experiment}", flush=True)
        for candidate in layout.log_paths:
            print(f"  checked: {candidate}", flush=True)
        return 1

    parsed = parse_log(log_path)
    metrics = compute_best_final(parsed)
    metrics["num_point"] = parsed.num_point
    metrics["allow_resample"] = parsed.allow_resample
    metrics["training_start"] = parsed.start_time
    metrics["training_end"] = parsed.end_time
    config = load_config(layout.config_paths)

    baseline_name = args.baseline_name
    baseline_acc = parse_baseline_acc(args.baseline_acc) if args.baseline_acc is not None else None
    delta_pct = None
    if baseline_acc is not None and metrics.get("best_test_overall_acc") is not None:
        delta_pct = normalize_acc(metrics["best_test_overall_acc"]) - baseline_acc

    complete = is_complete(parsed)
    if not complete:
        snap = write_monitoring_snapshot(
            layout, log_path, parsed, metrics, config, baseline_name, baseline_acc
        )
        print(f"status=running current_epoch={metrics.get('final_epoch', 0)}")
        print(f"Wrote {snap}")
        print("final_report.md not generated (training incomplete)")
        return 0

    report = write_final_report(
        layout, log_path, parsed, metrics, config, baseline_name, baseline_acc, delta_pct
    )
    csv_path = write_result_summary_csv(
        layout, log_path, "completed", metrics, config, baseline_name, baseline_acc, delta_pct
    )
    print(f"status=completed")
    print(f"Wrote {report}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
