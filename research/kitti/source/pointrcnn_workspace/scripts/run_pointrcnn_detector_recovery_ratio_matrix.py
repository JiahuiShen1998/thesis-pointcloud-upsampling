#!/usr/bin/env python3
"""Prepare, screen, and fully evaluate isolated ratio-recovery variants."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO / "results/kitti_x4_detector_recovery_ratio_v1_20260723"
SOURCE = REPO / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
PYTHON_INPUT = REPO / "venv_pointrcnn/bin/python"
PYTHON_PREP = REPO / "venv_pointrcnn/bin/python"
PREPARER = REPO / "scripts/prepare_pointrcnn_detector_recovery_ratio_inputs.py"
RUNNER = REPO / "scripts/run_variant_eval_detector_recovery.py"
SCORER = REPO / "scripts/score_existing_predictions_detector_recovery.py"
VAL = REPO / "data/KITTI/ImageSets/val.txt"
SCREEN_NAME = "val_detector_recovery_screen256_v1"
SCREEN_COPY = REPO / "data/KITTI/ImageSets" / f"{SCREEN_NAME}.txt"
METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
VARIANTS = tuple(
    [f"original_x4_{method}" for method in METHODS]
    + [f"downsampled_x4_{method}" for method in METHODS]
)
POLICIES = (
    "g10_o90_16384",
    "g15_o85_16384",
    "g25_o75_16384",
    "g35_o65_16384",
    "g40_o60_16384",
    "g50_o50_16384",
)
REFERENCE_VARIANTS = (
    "original_baseline",
    "downsampled_x4_baseline",
    *VARIANTS,
)
POLICY_RATIO = {
    "g10_o90_16384": 0.10,
    "g15_o85_16384": 0.15,
    "g25_o75_16384": 0.25,
    "g35_o65_16384": 0.35,
    "g40_o60_16384": 0.40,
    "g50_o50_16384": 0.50,
}


def frame_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create_screen_split(workspace: Path, count: int = 256) -> Path:
    candidates = []
    for frame in frame_ids(VAL):
        label = REPO / "data/KITTI/object/training/label_2" / f"{frame}.txt"
        if label.is_file() and any(
            line.startswith("Car ") for line in label.read_text(encoding="utf-8").splitlines()
        ):
            candidates.append(frame)
    if len(candidates) < count:
        raise RuntimeError(f"only {len(candidates)} Car frames available for screen count {count}")
    positions = [round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)]
    selected = [candidates[position] for position in positions]
    if len(set(selected)) != count:
        raise RuntimeError("screen split selection produced duplicate frames")
    text = "\n".join(selected) + "\n"
    split = workspace / "splits" / f"{SCREEN_NAME}.txt"
    split.parent.mkdir(parents=True, exist_ok=True)
    split.write_text(text, encoding="utf-8")
    SCREEN_COPY.write_text(text, encoding="utf-8")
    return split


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


def prepare_screen(workspace: Path, split: Path) -> None:
    expected_frames = frame_ids(split)
    missing_policies = []
    for policy in POLICIES:
        complete = True
        for variant in VARIANTS:
            marker = workspace / "manifests" / policy / f"{variant}.csv"
            output = workspace / "inputs" / policy / variant
            if not marker.is_file() or not all(
                (output / f"{frame}.bin").is_file() for frame in expected_frames
            ):
                complete = False
                break
        if not complete:
            missing_policies.append(policy)
    if not missing_policies:
        print("SKIP screen inputs already complete", flush=True)
        return
    command = [str(PYTHON_PREP), str(PREPARER), "--workspace", str(workspace), "--split-file", str(split)]
    for policy in missing_policies:
        command.extend(("--ratio", str(POLICY_RATIO[policy])))
    run_command(command, workspace / "logs/preparation/screen_inputs.log")


def score_screen_references(workspace: Path, split: Path) -> None:
    for variant in REFERENCE_VARIANTS:
        pred_dir = (
            SOURCE / "eval_outputs/e2" / variant
            / "inference/eval/epoch_no_number/val/final_result/data"
        )
        out_dir = workspace / "eval_outputs/screen_reference/e2_current" / variant
        prior = read_summary(out_dir / "result_summary.json")
        if prior and prior.get("status") == "PASS":
            print(f"SKIP_PASS screen reference {variant}", flush=True)
            continue
        command = [
            str(PYTHON_INPUT), str(SCORER), "--name", f"screen_reference_e2_{variant}",
            "--pred-dir", str(pred_dir), "--out-dir", str(out_dir), "--split-file", str(split),
        ]
        run_command(command, workspace / "logs/screen_reference" / f"{variant}.log")


def evaluate_one(workspace: Path, phase: str, split: Path, policy: str, variant: str) -> None:
    input_dir = workspace / "inputs" / policy / variant
    out_dir = workspace / "eval_outputs" / phase / policy / variant
    prior = read_summary(out_dir / "result_summary.json")
    if prior and prior.get("status") == "PASS":
        print(f"SKIP_PASS {phase} {policy} {variant}", flush=True)
        return
    command = [
        str(PYTHON_INPUT),
        str(RUNNER),
        "--name",
        f"recovery_{phase}_{policy}_{variant}",
        "--input-dir",
        str(input_dir),
        "--out-dir",
        str(out_dir),
        "--split-file",
        str(split),
    ]
    run_command(command, workspace / "logs" / phase / policy / f"{variant}.log")


def collect(workspace: Path, phase: str) -> list[dict]:
    rows = []
    for policy in POLICIES:
        for variant in VARIANTS:
            path = workspace / "eval_outputs" / phase / policy / variant / "result_summary.json"
            summary = read_summary(path)
            metrics = summary.get("ap_r40_percent", {}) if summary else {}
            rows.append(
                {
                    "phase": phase,
                    "policy": policy,
                    "generated_ratio": POLICY_RATIO[policy],
                    "variant": variant,
                    "status": summary.get("status", "PENDING") if summary else "PENDING",
                    "runtime_seconds": summary.get("runtime_seconds", "") if summary else "",
                    "prediction_files": summary.get("prediction_files", "") if summary else "",
                    "3d_ap_easy": metrics.get("3d_ap", {}).get("easy", ""),
                    "3d_ap_moderate": metrics.get("3d_ap", {}).get("moderate", ""),
                    "3d_ap_hard": metrics.get("3d_ap", {}).get("hard", ""),
                    "summary_path": str(path),
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_best(screen_rows: list[dict]) -> dict[str, str]:
    best = {}
    for variant in VARIANTS:
        candidates = [
            row for row in screen_rows
            if row["variant"] == variant and row["status"] == "PASS" and row["3d_ap_moderate"] != ""
        ]
        if not candidates:
            continue
        selected = max(candidates, key=lambda row: float(row["3d_ap_moderate"]))
        best[variant] = str(selected["policy"])
    return best


def prepare_full_best(workspace: Path, best: dict[str, str]) -> None:
    for variant, policy in best.items():
        ratio = POLICY_RATIO[policy]
        expected = len(frame_ids(VAL))
        output = workspace / "inputs" / policy / variant
        if output.is_dir() and len(list(output.glob("*.bin"))) == expected:
            print(f"SKIP full input {policy} {variant}", flush=True)
            continue
        command = [
            str(PYTHON_PREP), str(PREPARER), "--workspace", str(workspace),
            "--split-file", str(VAL), "--ratio", str(ratio), "--variant", variant,
        ]
        run_command(command, workspace / "logs/preparation" / f"full_{policy}_{variant}.log")


def write_report(workspace: Path, screen_rows: list[dict], best: dict[str, str]) -> None:
    report = workspace / "reports/recovery_ratio_status.md"
    lines = [
        "# PointRCNN detector recovery ratio experiment",
        "",
        "Original E1/E2/E3 inputs and evaluations are read-only. All new files live in this recovery workspace.",
        "",
        "| Variant | g10 | g15 | g25 | g35 | g40 | g50 | Selected policy |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in VARIANTS:
        values = {}
        for row in screen_rows:
            if row["variant"] == variant:
                values[row["policy"]] = row["3d_ap_moderate"]
        def fmt(policy: str) -> str:
            value = values.get(policy, "")
            return f"{float(value):.2f}" if value != "" else "—"
        lines.append(
            f"| {variant} | "
            + " | ".join(fmt(policy) for policy in POLICIES)
            + f" | {best.get(variant, 'pending')} |"
        )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (workspace / "reports/best_screen_policy.json").write_text(
        json.dumps(best, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--phase", choices=("prepare", "screen", "full", "all"), default="all")
    parser.add_argument("--variant", action="append")
    parser.add_argument("--policy", action="append", choices=POLICIES)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace == SOURCE.resolve():
        raise ValueError("refusing to write recovery results into original E1/E2 workspace")
    workspace.mkdir(parents=True, exist_ok=True)
    split = create_screen_split(workspace)
    prepare_screen(workspace, split)
    score_screen_references(workspace, split)
    if args.phase == "prepare":
        print(f"RECOVERY_PREPARE_COMPLETE workspace={workspace}", flush=True)
        return 0

    selected_variants = tuple(args.variant or VARIANTS)
    selected_policies = tuple(args.policy or POLICIES)
    if args.phase in ("screen", "all"):
        for policy in selected_policies:
            for variant in selected_variants:
                evaluate_one(workspace, "screen", split, policy, variant)

    screen_rows = collect(workspace, "screen")
    write_rows(workspace / "reports/screen_ap_summary.csv", screen_rows)
    best = choose_best(screen_rows)
    write_report(workspace, screen_rows, best)
    if args.phase == "screen":
        print(f"RECOVERY_SCREEN_COMPLETE workspace={workspace}", flush=True)
        return 0
    if len(best) != len(VARIANTS):
        raise RuntimeError(f"full phase requires {len(VARIANTS)} screened variants, found {len(best)}")

    if args.phase in ("full", "all"):
        prepare_full_best(workspace, best)
        for variant in selected_variants:
            evaluate_one(workspace, "full", VAL, best[variant], variant)
        full_rows = collect(workspace, "full")
        write_rows(workspace / "reports/full_ap_summary.csv", full_rows)
    print(f"RECOVERY_MATRIX_COMPLETE workspace={workspace}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
