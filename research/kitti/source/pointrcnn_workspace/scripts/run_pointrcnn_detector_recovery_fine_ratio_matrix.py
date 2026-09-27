#!/usr/bin/env python3
"""Run the isolated nested fine-ratio PointRCNN recovery experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO / "results/kitti_x4_detector_recovery_fine_ratio_v2_20260725"
SOURCE = REPO / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
V1 = REPO / "results/kitti_x4_detector_recovery_ratio_v1_20260723"
PYTHON = REPO / "venv_pointrcnn/bin/python"
PREPARER = REPO / "scripts/prepare_pointrcnn_detector_recovery_fine_ratio_inputs.py"
RUNNER = REPO / "scripts/run_variant_eval_detector_recovery.py"
SCORER = REPO / "scripts/score_existing_predictions_detector_recovery.py"
VAL = REPO / "data/KITTI/ImageSets/val.txt"
SCREEN_SOURCE = V1 / "splits/val_detector_recovery_screen256_v1.txt"
SCREEN_NAME = "val_detector_recovery_fine_ratio_screen256_v2"
METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
VARIANTS = tuple(
    [f"original_x4_{method}" for method in METHODS]
    + [f"downsampled_x4_{method}" for method in METHODS]
)
POLICIES = (
    "g025_o975_16384",
    "g050_o950_16384",
    "g075_o925_16384",
    "g100_o900_16384",
)
POLICY_RATIO = {
    "g025_o975_16384": 0.025,
    "g050_o950_16384": 0.05,
    "g075_o925_16384": 0.075,
    "g100_o900_16384": 0.10,
}
CONTROL_POLICIES = (
    "c025_o975_16384",
    "c050_o950_16384",
    "c075_o925_16384",
    "c100_o900_16384",
)
CONTROL_REPLACEMENT_RATIO = {
    "c025_o975_16384": 0.025,
    "c050_o950_16384": 0.05,
    "c075_o925_16384": 0.075,
    "c100_o900_16384": 0.10,
}
CONTROL_VARIANTS = ("original_observed_fill", "downsampled_observed_fill")
BASELINES = ("original_baseline", "downsampled_x4_baseline")


def frame_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_summary(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def run_command(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"RUN {' '.join(command)} log={log_path}", flush=True)
    env = os.environ.copy()
    env["POINT_RCNN_SKIP_PYTHON_AP"] = "1"
    with log_path.open("ab") as log:
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"command failed with return code {proc.returncode}: {' '.join(command)}")


def create_screen_split(workspace: Path) -> Path:
    if not SCREEN_SOURCE.is_file():
        raise FileNotFoundError(f"missing validated screen split: {SCREEN_SOURCE}")
    text = SCREEN_SOURCE.read_text(encoding="utf-8")
    split = workspace / "splits" / f"{SCREEN_NAME}.txt"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text(text, encoding="utf-8")
    (REPO / "data/KITTI/ImageSets" / f"{SCREEN_NAME}.txt").write_text(text, encoding="utf-8")
    return split


def prepare_screen(workspace: Path, split: Path) -> None:
    controls = [
        workspace / "manifests" / policy / f"{variant}.csv"
        for policy in CONTROL_POLICIES
        for variant in CONTROL_VARIANTS
    ]
    expected = len(frame_ids(split))
    missing_variants = []
    for variant in VARIANTS:
        marker = workspace / "manifests/g100_o900_16384" / f"{variant}.csv"
        output = workspace / "inputs/g100_o900_16384" / variant
        if not marker.is_file() or len(list(output.glob("*.bin"))) != expected:
            missing_variants.append(variant)
    methods_complete = not missing_variants
    controls_complete = all(path.is_file() for path in controls)
    if methods_complete and controls_complete:
        print("SKIP fine-ratio screen inputs already complete", flush=True)
        return
    if not methods_complete:
        command = [
            str(PYTHON),
            str(PREPARER),
            "--workspace",
            str(workspace),
            "--split-file",
            str(split),
            "--skip-controls",
        ]
        for variant in missing_variants:
            command.extend(("--variant", variant))
        run_command(
            command,
            workspace / "logs/preparation/screen_missing_methods.log",
        )
    if not controls_complete:
        run_command(
            [
                str(PYTHON),
                str(PREPARER),
                "--workspace",
                str(workspace),
                "--split-file",
                str(split),
                "--controls-only",
            ],
            workspace / "logs/preparation/screen_controls.log",
        )


def score_baseline_references(workspace: Path, split: Path) -> None:
    for baseline in BASELINES:
        out_dir = workspace / "eval_outputs/screen_reference" / baseline
        prior = read_summary(out_dir / "result_summary.json")
        if prior and prior.get("status") == "PASS":
            print(f"SKIP_PASS screen reference {baseline}", flush=True)
            continue
        pred_dir = (
            SOURCE
            / "eval_outputs/e2"
            / baseline
            / "inference/eval/epoch_no_number/val/final_result/data"
        )
        run_command(
            [
                str(PYTHON),
                str(SCORER),
                "--name",
                f"fine_ratio_screen_reference_{baseline}",
                "--pred-dir",
                str(pred_dir),
                "--out-dir",
                str(out_dir),
                "--split-file",
                str(split),
            ],
            workspace / "logs/screen_reference" / f"{baseline}.log",
        )


def evaluate_one(
    workspace: Path,
    phase: str,
    split: Path,
    policy: str,
    variant: str,
) -> None:
    input_dir = workspace / "inputs" / policy / variant
    out_dir = workspace / "eval_outputs" / phase / policy / variant
    prior = read_summary(out_dir / "result_summary.json")
    if prior and prior.get("status") == "PASS":
        print(f"SKIP_PASS {phase} {policy} {variant}", flush=True)
        return
    run_command(
        [
            str(PYTHON),
            str(RUNNER),
            "--name",
            f"fine_ratio_{phase}_{policy}_{variant}",
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_dir),
            "--split-file",
            str(split),
        ],
        workspace / "logs" / phase / policy / f"{variant}.log",
    )


def result_row(
    phase: str,
    policy: str,
    ratio: float | str,
    variant: str,
    path: Path,
    row_kind: str,
) -> dict:
    summary = read_summary(path)
    metrics = summary.get("ap_r40_percent", {}) if summary else {}
    ap = metrics.get("3d_ap", {})
    return {
        "phase": phase,
        "row_kind": row_kind,
        "policy": policy,
        "generated_ratio": ratio,
        "variant": variant,
        "status": summary.get("status", "PENDING") if summary else "PENDING",
        "runtime_seconds": summary.get("runtime_seconds", "") if summary else "",
        "prediction_files": summary.get("prediction_files", "") if summary else "",
        "3d_ap_easy": ap.get("easy", ""),
        "3d_ap_moderate": ap.get("moderate", ""),
        "3d_ap_hard": ap.get("hard", ""),
        "summary_path": str(path),
    }


def collect_screen(workspace: Path) -> list[dict]:
    rows = []
    for baseline in BASELINES:
        rows.append(
            result_row(
                "screen",
                "g000_o1000_16384",
                0.0,
                baseline,
                workspace / "eval_outputs/screen_reference" / baseline / "result_summary.json",
                "baseline_reference",
            )
        )
    for policy in POLICIES:
        for variant in VARIANTS:
            rows.append(
                result_row(
                    "screen",
                    policy,
                    POLICY_RATIO[policy],
                    variant,
                    workspace / "eval_outputs/screen" / policy / variant / "result_summary.json",
                    "generated_ratio",
                )
            )
    for policy in CONTROL_POLICIES:
        for variant in CONTROL_VARIANTS:
            rows.append(
                result_row(
                    "screen",
                    policy,
                    CONTROL_REPLACEMENT_RATIO[policy],
                    variant,
                    workspace
                    / "eval_outputs/screen"
                    / policy
                    / variant
                    / "result_summary.json",
                    "observed_fill_control",
                )
            )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_best(rows: list[dict]) -> dict[str, str]:
    best: dict[str, str] = {}
    for variant in VARIANTS:
        candidates = [
            row
            for row in rows
            if row["variant"] == variant
            and row["row_kind"] == "generated_ratio"
            and row["status"] == "PASS"
            and row["3d_ap_moderate"] != ""
        ]
        if candidates:
            winner = max(candidates, key=lambda row: float(row["3d_ap_moderate"]))
            best[variant] = str(winner["policy"])
    return best


def write_screen_report(workspace: Path, rows: list[dict], best: dict[str, str]) -> None:
    lookup = {(row["policy"], row["variant"]): row for row in rows}
    lines = [
        "# PointRCNN fine generated-ratio screen",
        "",
        "The strict-x4 upstream clouds are unchanged. Detector inputs replace a nested,",
        "method-independent subset of the exact E2 baseline's 16,384 observed slots.",
        "",
        "| Variant | Baseline g0 | g2.5 | g5 | g7.5 | g10 | Best generated policy |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        baseline = "original_baseline" if variant.startswith("original_") else "downsampled_x4_baseline"

        def value(policy: str, name: str) -> str:
            row = lookup.get((policy, name))
            raw = row.get("3d_ap_moderate", "") if row else ""
            return f"{float(raw):.2f}" if raw != "" else "—"

        lines.append(
            f"| {variant} | {value('g000_o1000_16384', baseline)} | "
            f"{value(POLICIES[0], variant)} | {value(POLICIES[1], variant)} | "
            f"{value(POLICIES[2], variant)} | {value(POLICIES[3], variant)} | "
            f"{best.get(variant, 'pending')} |"
        )
    lines.extend(
        [
            "",
            "## Observed-fill controls",
            "",
            "| Line | c2.5 | c5 | c7.5 | c10 |",
            "|---|---:|---:|---:|---:|",
            "| Original | "
            + " | ".join(
                value(policy, "original_observed_fill") for policy in CONTROL_POLICIES
            )
            + " |",
            "| Downsampled | "
            + " | ".join(
                value(policy, "downsampled_observed_fill") for policy in CONTROL_POLICIES
            )
            + " |",
            "",
        ]
    )
    report = workspace / "reports/fine_ratio_screen_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    (workspace / "reports/best_generated_policy.json").write_text(
        json.dumps(best, indent=2) + "\n", encoding="utf-8"
    )


def prepare_full_best(workspace: Path, best: dict[str, str]) -> None:
    expected = len(frame_ids(VAL))
    for variant in VARIANTS:
        policy = best[variant]
        output = workspace / "inputs" / policy / variant
        if output.is_dir() and len(list(output.glob("*.bin"))) == expected:
            print(f"SKIP full fine-ratio input {policy} {variant}", flush=True)
            continue
        run_command(
            [
                str(PYTHON),
                str(PREPARER),
                "--workspace",
                str(workspace),
                "--split-file",
                str(VAL),
                "--ratio",
                str(POLICY_RATIO[policy]),
                "--variant",
                variant,
                "--skip-controls",
            ],
            workspace / "logs/preparation" / f"full_{policy}_{variant}.log",
        )


def collect_full(workspace: Path, best: dict[str, str]) -> list[dict]:
    rows = []
    for baseline in BASELINES:
        rows.append(
            result_row(
                "full",
                "g000_o1000_16384",
                0.0,
                baseline,
                SOURCE / "eval_outputs/e2" / baseline / "result_summary.json",
                "baseline_reference",
            )
        )
    for variant in VARIANTS:
        policy = best[variant]
        rows.append(
            result_row(
                "full",
                policy,
                POLICY_RATIO[policy],
                variant,
                workspace / "eval_outputs/full" / policy / variant / "result_summary.json",
                "generated_ratio",
            )
        )
    return rows


def write_full_report(workspace: Path, rows: list[dict]) -> None:
    lines = [
        "# PointRCNN fine generated-ratio full-validation result",
        "",
        "| Variant | Policy | Easy | Moderate | Hard | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = row[key]
            return f"{float(value):.2f}" if value != "" else "—"

        lines.append(
            f"| {row['variant']} | {row['policy']} | {fmt('3d_ap_easy')} | "
            f"{fmt('3d_ap_moderate')} | {fmt('3d_ap_hard')} | {row['status']} |"
        )
    report = workspace / "reports/fine_ratio_full_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--phase", choices=("prepare", "screen", "full", "all"), default="all")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace in (SOURCE.resolve(), V1.resolve()):
        raise ValueError("refusing to write fine-ratio results into an existing experiment")
    workspace.mkdir(parents=True, exist_ok=True)
    split = create_screen_split(workspace)

    prepare_screen(workspace, split)
    score_baseline_references(workspace, split)
    if args.phase == "prepare":
        print(f"FINE_RATIO_PREPARE_COMPLETE workspace={workspace}", flush=True)
        return 0

    if args.phase in ("screen", "all"):
        for policy in POLICIES:
            for variant in VARIANTS:
                evaluate_one(workspace, "screen", split, policy, variant)
        for policy in CONTROL_POLICIES:
            for control in CONTROL_VARIANTS:
                evaluate_one(workspace, "screen", split, policy, control)

    screen_rows = collect_screen(workspace)
    write_rows(workspace / "reports/fine_ratio_screen_ap_summary.csv", screen_rows)
    best = choose_best(screen_rows)
    write_screen_report(workspace, screen_rows, best)
    if args.phase == "screen":
        print(f"FINE_RATIO_SCREEN_COMPLETE workspace={workspace}", flush=True)
        return 0
    if len(best) != len(VARIANTS):
        raise RuntimeError(f"full phase needs {len(VARIANTS)} screened variants, found {len(best)}")

    if args.phase in ("full", "all"):
        prepare_full_best(workspace, best)
        for variant in VARIANTS:
            evaluate_one(workspace, "full", VAL, best[variant], variant)
        full_rows = collect_full(workspace, best)
        write_rows(workspace / "reports/fine_ratio_full_ap_summary.csv", full_rows)
        write_full_report(workspace, full_rows)

    print(f"FINE_RATIO_MATRIX_COMPLETE workspace={workspace}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
