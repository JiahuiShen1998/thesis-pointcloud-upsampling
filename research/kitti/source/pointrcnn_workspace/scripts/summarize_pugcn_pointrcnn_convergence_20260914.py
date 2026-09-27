#!/usr/bin/env python3
"""Summarize stage-wise PointRCNN convergence on fixed PU-GCN inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARMS = ("line_a_pugcn_observed_first", "line_b_pugcn_observed_first")
PHASES = ("rpn", "rcnn")
OLD_EVAL = PROJECT_ROOT / "results/pugcn_detector_adaptation_full_val_20260908/evaluations/pointrcnn"
REFERENCES = {
    "line_a_pugcn_observed_first": {
        "old_epoch3": OLD_EVAL / "line_a_pugcn_observed_first_adapted/run_complete.json",
        "unadapted_same_input": OLD_EVAL / "line_a_pugcn_observed_first_official/run_complete.json",
        "baseline": OLD_EVAL / "line_a_baseline_official/run_complete.json",
        "baseline_label": "official PointRCNN on original-N Line A input",
    },
    "line_b_pugcn_observed_first": {
        "old_epoch3": OLD_EVAL / "line_b_pugcn_observed_first_adapted/run_complete.json",
        "unadapted_same_input": OLD_EVAL / "line_b_pugcn_observed_first_official/run_complete.json",
        "baseline": OLD_EVAL / "line_b_baseline_adapted/run_complete.json",
        "baseline_label": "existing 3-epoch adapted sparse Line B baseline",
    },
}
PU_GCN_CHECKPOINT_FILES = (
    PROJECT_ROOT / "external/PU-GCN/pretrained/pu1k-pugcn/model-100.data-00000-of-00001",
    PROJECT_ROOT / "external/PU-GCN/pretrained/pu1k-pugcn/model-100.index",
    PROJECT_ROOT / "external/PU-GCN/pretrained/pu1k-pugcn/model-100.meta",
)
INPUT_PROTOCOLS = {
    "line_a_pugcn_observed_first": {
        "train": PROJECT_ROOT / "results/pugcn_full_retrain_20260824/generation/train3712_linea/pu_gcn_linea_surface_pr1_c32/protocol.json",
        "validation": PROJECT_ROOT / "results/pugcn_detector_adaptation_full_val_20260908/generation/val3769_linea/pu_gcn_linea_surface_pr1_c32/protocol.json",
        "strict_x4_audit": PROJECT_ROOT / "results/pugcn_detector_adaptation_full_val_20260908/reports/line_a_strict4x_audit.json",
    },
    "line_b_pugcn_observed_first": {
        "train": PROJECT_ROOT / "results/pugcn_full_retrain_20260824/generation/train3712_lineb/pu_gcn_lineb_c2048_r4/protocol.json",
        "validation": PROJECT_ROOT / "results/pugcn_detector_adaptation_full_val_20260908/generation/val3769_lineb/pu_gcn_lineb_c2048_r4/protocol.json",
        "strict_x4_audit": PROJECT_ROOT / "results/pugcn_detector_adaptation_full_val_20260908/reports/line_b_strict4x_audit.json",
    },
}
CODE_FILES = (
    PROJECT_ROOT / "scripts/run_pugcn_pointrcnn_convergence_20260914.sh",
    PROJECT_ROOT / "scripts/run_pointrcnn_full_train_stage.py",
    PROJECT_ROOT / "scripts/run_pointrcnn_rpn_feature_export.py",
    PROJECT_ROOT / "scripts/combine_pointrcnn_rpn_with_official_rcnn.py",
    PROJECT_ROOT / "scripts/run_pointrcnn_checkpoint_split_eval.py",
    PROJECT_ROOT / "scripts/evaluate_kitti_r40_openpcdet_fast.py",
    Path(__file__).resolve(),
)
TRAINING_SNAPSHOT_FILENAMES = (
    "train_rcnn.py",
    "train_functions.py",
    "kitti_rcnn_dataset.py",
    "point_rcnn.py",
    "rpn.py",
    "rcnn_net.py",
    "pointnet2_msg.py",
)


def file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def extension_protocol_record(path: Path, expected_epochs: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predecessor = payload.get("predecessor", {})
    extension = payload.get("extension", {})
    criterion = payload.get("criterion", {})
    predecessor_summary = Path(str(predecessor.get("line_a_rpn_summary", "")))
    predecessor_record = file_record(predecessor_summary)
    predecessor_payload = json.loads(predecessor_summary.read_text(encoding="utf-8"))
    expected = {
        "manifest_status": "EXTENSION_LAUNCHED",
        "metric": "Car 3D Moderate AP_R40",
        "validation_frames_per_epoch": 3769,
        "min_delta_ap_points": 0.2,
        "patience_epochs": 3,
        "predecessor_status": "NOT_YET_CONVERGED",
        "predecessor_terminal_plateau_reached": False,
        "predecessor_summary_sha256": predecessor_record["sha256"],
        "extension_schedule_epochs": expected_epochs,
        "extension_run_root": str(path.parent.resolve()),
        "initialization_sha256": file_record(PROJECT_ROOT / "tools/PointRCNN.pth")["sha256"],
        "resume_ckpt": None,
    }
    actual = {
        "manifest_status": payload.get("status"),
        "metric": criterion.get("metric"),
        "validation_frames_per_epoch": criterion.get("validation_frames_per_epoch"),
        "min_delta_ap_points": criterion.get("min_delta_ap_points"),
        "patience_epochs": criterion.get("patience_epochs"),
        "predecessor_status": predecessor_payload.get("status"),
        "predecessor_terminal_plateau_reached": predecessor_payload.get(
            "terminal_plateau_reached"
        ),
        "predecessor_summary_sha256": predecessor.get("line_a_rpn_summary_sha256"),
        "extension_schedule_epochs": extension.get("schedule_epochs"),
        "extension_run_root": str(Path(str(extension.get("run_root", ""))).resolve()),
        "initialization_sha256": extension.get("initialization_sha256"),
        "resume_ckpt": extension.get("resume_ckpt"),
    }
    mismatches = {
        key: {"expected": value, "actual": actual.get(key)}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"invalid schedule extension protocol {path}: {mismatches}")
    return {
        **file_record(path),
        "status": payload["status"],
        "criterion": criterion,
        "predecessor": predecessor,
        "extension": extension,
        "predecessor_summary_evidence": predecessor_record,
    }


def input_protocol_record(path: Path, expected_frames: int) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frame_count = len(payload.get("frame_ids", []))
    if payload.get("status") != "PASS" or frame_count != expected_frames:
        raise RuntimeError(f"invalid fixed-input protocol: {path}")
    return {
        **file_record(path),
        "status": payload["status"],
        "frame_count": frame_count,
        "method": payload.get("method"),
        "variant": payload.get("variant"),
        "strict_up_ratio": payload.get("strict_up_ratio"),
        "patch_selection": payload.get("patch_selection"),
    }


def strict_x4_audit_record(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS"
        or int(payload.get("frames_expected", 0)) != 3769
        or int(payload.get("frames_checked", 0)) != 3769
        or int(payload.get("frames_pass", 0)) != 3769
        or int(payload.get("frames_fail", -1)) != 0
    ):
        raise RuntimeError(f"invalid strict-x4 audit: {path}")
    return {
        **file_record(path),
        "status": payload["status"],
        "frames_checked": payload["frames_checked"],
        "frames_pass": payload["frames_pass"],
        "frames_fail": payload["frames_fail"],
    }


def training_protocol_record(
    path: Path, expected_mode: str, expected_epochs: int
) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_optimization = {
        "name": "adam_onecycle",
        "max_learning_rate": 0.0002,
        "initial_learning_rate": 0.00002,
        "final_learning_rate": 0.000000002,
        "adam_betas": [0.9, 0.99],
        "weight_decay": 0.001,
        "momentums": [0.95, 0.85],
        "division_factor": 10.0,
        "pct_start": 0.4,
        "gradient_norm_clip": 1.0,
        "lr_warmup": False,
    }
    expected_batch_norm_schedule = {
        "initial_momentum": 0.1,
        "decay": 0.5,
        "minimum_momentum": 0.01,
        "decay_epoch_steps": [2],
    }
    expected = {
        "mode": expected_mode,
        "frame_count": 3712,
        "epochs": expected_epochs,
        "ckpt_save_interval": 1,
        "workers": 0,
        "batch_size": 1,
        "gt_database_augmentation": False,
        "optimization": expected_optimization,
        "batch_norm_schedule": expected_batch_norm_schedule,
        "rpn_loc_xz_fine": False,
    }
    mismatches = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"invalid stage training protocol {path}: {mismatches}")
    resume_ckpt = payload.get("resume_ckpt")
    if resume_ckpt is not None:
        resume_path = Path(resume_ckpt).resolve()
        expected_ckpt_dir = (path.parent / "ckpt").resolve()
        if resume_path.parent != expected_ckpt_dir or not resume_path.name.startswith(
            "checkpoint_epoch_"
        ):
            raise RuntimeError(
                f"resume checkpoint is not from the same clean schedule: {resume_path}"
            )
    import torch

    checkpoint_records = []
    previous_iteration = -1
    for epoch in range(1, expected_epochs + 1):
        checkpoint_path = path.parent / "ckpt" / f"checkpoint_epoch_{epoch}.pth"
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        internal_epoch = int(checkpoint_payload.get("epoch", -1))
        internal_iteration = int(checkpoint_payload.get("it", -1))
        model_state = checkpoint_payload.get("model_state", {})
        optimizer_state = checkpoint_payload.get("optimizer_state", {})
        if internal_epoch != epoch:
            raise RuntimeError(
                f"checkpoint internal epoch mismatch: {checkpoint_path}: {internal_epoch}"
            )
        if internal_iteration <= previous_iteration:
            raise RuntimeError(
                f"checkpoint iteration is not increasing: {checkpoint_path}: {internal_iteration}"
            )
        if not model_state or not optimizer_state.get("state"):
            raise RuntimeError(f"incomplete checkpoint state: {checkpoint_path}")
        checkpoint_records.append(
            {
                **file_record(checkpoint_path),
                "internal_epoch": internal_epoch,
                "internal_iteration": internal_iteration,
                "model_tensor_count": len(model_state),
                "optimizer_state_entry_count": len(optimizer_state["state"]),
                "optimizer_param_group_count": len(optimizer_state.get("param_groups", [])),
            }
        )
        previous_iteration = internal_iteration
        del checkpoint_payload
    execution_code_snapshot = [
        file_record(path.parent / "backup_files" / filename)
        for filename in TRAINING_SNAPSHOT_FILENAMES
    ]
    return {
        **file_record(path),
        **expected,
        "seed": payload.get("seed"),
        "lidar_dir": payload.get("lidar_dir"),
        "init_ckpt": payload.get("init_ckpt"),
        "resume_ckpt": resume_ckpt,
        "optimization": expected_optimization,
        "batch_norm_schedule": expected_batch_norm_schedule,
        "rpn_loc_xz_fine": False,
        "training_log": file_record(path.parent / "runner_stdout.log"),
        "checkpoint_files": checkpoint_records,
        "execution_code_snapshot": execution_code_snapshot,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage = subparsers.add_parser("stage", help="summarize one arm and one training phase")
    stage.add_argument("--run-root", type=Path, required=True)
    stage.add_argument("--arm", choices=ARMS, required=True)
    stage.add_argument("--phase", choices=PHASES, required=True)
    stage.add_argument("--expected-epochs", type=int, required=True)
    stage.add_argument("--min-delta", type=float, default=0.2)
    stage.add_argument("--patience", type=int, default=3)

    aggregate = subparsers.add_parser("aggregate", help="combine four converged stage summaries")
    aggregate.add_argument("--run-root", type=Path, required=True)
    aggregate.add_argument("--expected-epochs", type=int, required=True)
    return parser


def load_reference(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS"
        or int(payload.get("frame_count", 0)) != 3769
        or int(payload.get("actual_consumed_frame_count", 0)) != 3769
    ):
        raise RuntimeError(f"invalid reference result: {path}")
    return float(payload["metrics_percent"]["Car"]["3d_ap_r40"]["moderate"])


def early_stop(rows: list[dict[str, object]], min_delta: float, patience: int) -> dict[str, object]:
    significant_best = float("-inf")
    significant_best_epoch = -1
    bad_epochs = 0
    first_stop_epoch = None
    significant_best_at_first_stop = None
    significant_best_epoch_at_first_stop = None
    trace = []
    for row in rows:
        value = float(row["car_3d_ap_r40_moderate"])
        epoch = int(row["epoch"])
        significant_best_before = None if significant_best == float("-inf") else significant_best
        threshold_to_reset = (
            None if significant_best_before is None else significant_best_before + min_delta
        )
        significant_improvement = threshold_to_reset is None or value > threshold_to_reset
        if significant_improvement:
            significant_best = value
            significant_best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if first_stop_epoch is None and bad_epochs >= patience:
                first_stop_epoch = epoch
                significant_best_at_first_stop = significant_best
                significant_best_epoch_at_first_stop = significant_best_epoch
        trace.append(
            {
                "epoch": epoch,
                "ap": value,
                "significant_best_before_epoch": significant_best_before,
                "threshold_to_reset_ap": threshold_to_reset,
                "significant_improvement": significant_improvement,
                "significant_best_ap_after_epoch": significant_best,
                "significant_best_epoch_after_epoch": significant_best_epoch,
                "consecutive_non_improving_epochs": bad_epochs,
                "patience_reached": bad_epochs >= patience,
            }
        )
    global_best = max(rows, key=lambda row: float(row["car_3d_ap_r40_moderate"]))
    return {
        "criterion": "Car 3D Moderate AP_R40; improvement must exceed min_delta",
        "min_delta_ap_points": min_delta,
        "patience_epochs": patience,
        "early_stop_triggered": first_stop_epoch is not None,
        "terminal_plateau_reached": bad_epochs >= patience,
        "terminal_non_improving_epochs": bad_epochs,
        "first_stop_epoch": first_stop_epoch,
        "significant_best_epoch_at_first_stop": significant_best_epoch_at_first_stop,
        "significant_best_ap_at_first_stop": significant_best_at_first_stop,
        "global_best_epoch": int(global_best["epoch"]),
        "global_best_ap": float(global_best["car_3d_ap_r40_moderate"]),
        "global_best_checkpoint": str(global_best["checkpoint"]),
        "convergence_trace": trace,
    }


def load_stage_rows(run_root: Path, arm: str, phase: str, expected_epochs: int) -> list[dict[str, object]]:
    rows = []
    for epoch in range(1, expected_epochs + 1):
        path = (
            run_root
            / "evaluations/pointrcnn"
            / f"{arm}_{phase}_epoch_{epoch}"
            / "run_complete.json"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"evaluation status is not PASS: {path}")
        for key in ("frame_count", "prediction_count", "actual_consumed_frame_count"):
            if int(payload.get(key, 0)) != 3769:
                raise RuntimeError(f"{key} is not 3769: {path}")
        if payload.get("ap_protocol") != "KITTI_AP_R40_41_precision_samples_exclude_recall_zero":
            raise RuntimeError(f"unexpected AP protocol: {path}")
        checkpoint = Path(payload["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        marker_record = file_record(path)
        checkpoint_record = file_record(checkpoint)
        metrics = payload["metrics_percent"]["Car"]
        rows.append(
            {
                "arm": arm,
                "phase": phase,
                "epoch": epoch,
                "car_3d_ap_r40_easy": metrics["3d_ap_r40"]["easy"],
                "car_3d_ap_r40_moderate": metrics["3d_ap_r40"]["moderate"],
                "car_3d_ap_r40_hard": metrics["3d_ap_r40"]["hard"],
                "car_bev_ap_r40_moderate": metrics["bev_ap_r40"]["moderate"],
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_record["sha256"],
                "run_complete_file": marker_record["path"],
                "run_complete_sha256": marker_record["sha256"],
                "frames": payload["frame_count"],
                "input_audit_frames": payload["actual_consumed_frame_count"],
                "heavy_artifacts_retained": payload.get("heavy_artifacts_retained"),
            }
        )
    return rows


def stage_paths(run_root: Path, arm: str, phase: str) -> tuple[Path, Path, Path, Path]:
    stem = f"pointrcnn_{arm}_{phase}_convergence"
    report_root = run_root / "reports"
    return (
        report_root / f"{stem}.json",
        report_root / f"{stem}.csv",
        report_root / f"{stem}.png",
        report_root / f"{stem}.pdf",
    )


def plot_stage(rows: list[dict[str, object]], summary: dict[str, object], png: Path, pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    values = [float(row["car_3d_ap_r40_moderate"]) for row in rows]
    fig, axis = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    axis.plot(
        epochs,
        values,
        color="#1570A6",
        marker="o",
        linewidth=2.2,
        label="current full-val AP",
    )
    unadapted_ap = float(summary["unadapted_same_input_ap"])
    axis.axhline(
        unadapted_ap,
        color="#7A5195",
        linestyle="--",
        label=f"unadapted, same PU-GCN input: {unadapted_ap:.2f}",
    )
    old_epoch3_ap = float(summary["old_epoch3_ap"])
    old_epoch3_label = (
        "previous 3-epoch adaptation"
        if summary["phase"] == "rcnn"
        else "previous 3-epoch final (cross-stage context)"
    )
    axis.axhline(
        old_epoch3_ap,
        color="#E69F00",
        linestyle="--",
        label=f"{old_epoch3_label}: {old_epoch3_ap:.2f}",
    )
    baseline_ap = float(summary["baseline_ap"])
    baseline_name = (
        "original-N official baseline"
        if summary["arm"].startswith("line_a")
        else "existing sparse-input adapted baseline"
    )
    axis.axhline(
        baseline_ap,
        color="#C43C39",
        linestyle=":",
        linewidth=2.0,
        label=f"{baseline_name}: {baseline_ap:.2f}",
    )
    best_epoch = int(summary["global_best_epoch"])
    best_ap = float(summary["global_best_ap"])
    axis.scatter([best_epoch], [best_ap], color="#009E73", s=90, zorder=5, label="global best")
    axis.annotate(
        f"best e{best_epoch}: {best_ap:.2f}",
        (best_epoch, best_ap),
        xytext=(8, 8),
        textcoords="offset points",
    )
    first_stop_epoch = summary.get("first_stop_epoch")
    if first_stop_epoch is not None:
        axis.axvline(
            int(first_stop_epoch),
            color="#666666",
            linestyle="-.",
            linewidth=1.2,
            label=f"first patience trigger: e{first_stop_epoch}",
        )
    if bool(summary.get("terminal_plateau_reached")):
        terminal_start = max(
            epochs[0], epochs[-1] - int(summary["patience_epochs"]) + 1
        )
        axis.axvspan(
            terminal_start - 0.45,
            epochs[-1] + 0.45,
            color="#009E73",
            alpha=0.08,
            label="terminal patience window",
        )
    title_arm = "Line A: original N + selected generated 3N" if summary["arm"].startswith("line_a") else "Line B: sparse M + selected generated 3M"
    title_phase = "RPN adaptation with official RCNN fixed" if summary["phase"] == "rpn" else "offline RCNN adaptation with selected RPN fixed"
    axis.set_title(f"PointRCNN — {title_arm}\n{title_phase}")
    axis.set_xlabel(f"{summary['phase'].upper()} adaptation epoch")
    axis.set_ylabel("Car 3D Moderate AP_R40 (%)")
    axis.set_xticks(epochs)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)


def summarize_stage(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    rows = load_stage_rows(run_root, args.arm, args.phase, args.expected_epochs)
    summary = early_stop(rows, args.min_delta, args.patience)
    refs = REFERENCES[args.arm]
    old_epoch3_ap = load_reference(refs["old_epoch3"])
    unadapted_same_input_ap = load_reference(refs["unadapted_same_input"])
    baseline_ap = load_reference(refs["baseline"])
    summary.update(
        {
            "status": "CONVERGED" if summary["terminal_plateau_reached"] else "NOT_YET_CONVERGED",
            "detector": "PointRCNN",
            "arm": args.arm,
            "phase": args.phase,
            "expected_epochs": args.expected_epochs,
            "validation_frames_per_epoch": 3769,
            "old_epoch3_ap": old_epoch3_ap,
            "change_vs_old_epoch3_ap": summary["global_best_ap"] - old_epoch3_ap,
            "unadapted_same_input_ap": unadapted_same_input_ap,
            "gain_vs_unadapted_same_input_ap": (
                summary["global_best_ap"] - unadapted_same_input_ap
            ),
            "baseline_label": refs["baseline_label"],
            "baseline_ap": baseline_ap,
            "gap_to_baseline_ap": summary["global_best_ap"] - baseline_ap,
            "reference_files": {
                key: str(value) for key, value in refs.items() if isinstance(value, Path)
            },
        }
    )
    json_path, csv_path, png_path, pdf_path = stage_paths(run_root, args.arm, args.phase)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    trace_by_epoch = {
        int(item["epoch"]): item for item in summary["convergence_trace"]
    }
    csv_rows = []
    for row in rows:
        trace_item = trace_by_epoch[int(row["epoch"])]
        csv_rows.append(
            {
                **row,
                "threshold_to_reset_ap": trace_item["threshold_to_reset_ap"],
                "significant_improvement": trace_item["significant_improvement"],
                "significant_best_ap_after_epoch": trace_item[
                    "significant_best_ap_after_epoch"
                ],
                "significant_best_epoch_after_epoch": trace_item[
                    "significant_best_epoch_after_epoch"
                ],
                "consecutive_non_improving_epochs": trace_item[
                    "consecutive_non_improving_epochs"
                ],
                "patience_reached": trace_item["patience_reached"],
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    plot_stage(rows, summary, png_path, pdf_path)
    print(json.dumps(summary, indent=2))
    return 0


def plot_aggregate(run_root: Path, summaries: dict[str, dict[str, dict[str, object]]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0), constrained_layout=True)
    for row_index, arm in enumerate(ARMS):
        for col_index, phase in enumerate(PHASES):
            axis = axes[row_index][col_index]
            _, csv_path, _, _ = stage_paths(run_root, arm, phase)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            item = summaries[arm][phase]
            epochs = [int(row["epoch"]) for row in rows]
            values = [float(row["car_3d_ap_r40_moderate"]) for row in rows]
            axis.plot(
                epochs,
                values,
                color="#1570A6",
                marker="o",
                linewidth=2.0,
                label="current full-val AP",
            )
            unadapted_ap = float(item["unadapted_same_input_ap"])
            axis.axhline(
                unadapted_ap,
                color="#7A5195",
                linestyle="--",
                label=f"unadapted, same input: {unadapted_ap:.2f}",
            )
            old_epoch3_ap = float(item["old_epoch3_ap"])
            old_epoch3_context = "" if phase == "rcnn" else " (final context)"
            axis.axhline(
                old_epoch3_ap,
                color="#E69F00",
                linestyle="--",
                label=f"previous 3-epoch{old_epoch3_context}: {old_epoch3_ap:.2f}",
            )
            baseline_ap = float(item["baseline_ap"])
            baseline_name = (
                "original-N baseline"
                if arm.startswith("line_a")
                else "sparse adapted baseline"
            )
            axis.axhline(
                baseline_ap,
                color="#C43C39",
                linestyle=":",
                linewidth=2.0,
                label=f"{baseline_name}: {baseline_ap:.2f}",
            )
            best_epoch = int(item["global_best_epoch"])
            best_ap = float(item["global_best_ap"])
            axis.scatter([best_epoch], [best_ap], color="#009E73", s=75, zorder=5)
            axis.annotate(f"e{best_epoch}: {best_ap:.2f}", (best_epoch, best_ap), xytext=(7, 7), textcoords="offset points", fontsize=9)
            arm_title = "Line A" if arm.startswith("line_a") else "Line B"
            phase_title = "RPN (official RCNN fixed)" if phase == "rpn" else "RCNN (selected RPN fixed)"
            axis.set_title(f"{arm_title} — {phase_title}")
            axis.set_xlabel("Adaptation epoch")
            axis.set_ylabel("Car 3D Moderate AP_R40 (%)")
            axis.set_xticks(epochs)
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8, loc="best")
    fig.suptitle("PointRCNN stage-wise convergence on fixed PU-GCN outputs — KITTI val (3,769 frames)", fontsize=14)
    report_root = run_root / "reports"
    fig.savefig(report_root / "pointrcnn_epoch_validation_curves.png", dpi=220)
    fig.savefig(report_root / "pointrcnn_epoch_validation_curves.pdf")
    plt.close(fig)


def aggregate(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    report_root = run_root / "reports"
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    arms_report: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        summaries[arm] = {}
        for phase in PHASES:
            json_path, _, _, _ = stage_paths(run_root, arm, phase)
            item = json.loads(json_path.read_text(encoding="utf-8"))
            if item.get("status") != "CONVERGED":
                raise RuntimeError(f"cannot aggregate unconverged stage: {json_path}")
            if int(item.get("expected_epochs", 0)) != args.expected_epochs:
                raise RuntimeError(f"stage epoch count mismatch: {json_path}")
            summaries[arm][phase] = item
        final = summaries[arm]["rcnn"]
        best_ap = float(final["global_best_ap"])
        arms_report[arm] = {
            "rpn": summaries[arm]["rpn"],
            "rcnn": summaries[arm]["rcnn"],
            "selected_rpn_epoch": int(summaries[arm]["rpn"]["global_best_epoch"]),
            "selected_rcnn_epoch": int(final["global_best_epoch"]),
            "final_best_ap": best_ap,
            "old_epoch3_ap": float(final["old_epoch3_ap"]),
            "gain_vs_old_epoch3_ap": best_ap - float(final["old_epoch3_ap"]),
            "unadapted_same_input_ap": float(final["unadapted_same_input_ap"]),
            "gain_vs_unadapted_same_input_ap": best_ap - float(final["unadapted_same_input_ap"]),
            "baseline_label": final["baseline_label"],
            "baseline_ap": float(final["baseline_ap"]),
            "gap_to_existing_baseline_ap": best_ap - float(final["baseline_ap"]),
        }

    report = {
        "status": "CONVERGED",
        "detector": "PointRCNN",
        "pu_gcn_training": "frozen released PU1K model-100; not retrained here",
        "protocol": "stage-wise RPN then offline RCNN adaptation; selected upstream stage fixed downstream",
        "previous_three_epoch_interpretation": (
            "The historical 3-epoch run was a fixed compute-budget adaptation, not a "
            "validated convergence point. It is retained as a comparison only. Because "
            "Adam OneCycle depends on the planned total number of epochs, this convergence "
            f"study is a clean {args.expected_epochs}-epoch schedule from the same initialization rather than a "
            "scheduler-inconsistent continuation of the old epoch-3 checkpoint."
        ),
        "comparability_to_previous_three_epoch": (
            "Same fixed PU-GCN inputs, PointRCNN initialization, split, seed, batch size, "
            "workers, and disabled GT-database augmentation; planned epoch count and "
            "checkpoint save interval differ."
        ),
        "validation_frames_per_epoch": 3769,
        "primary_metric": "Car 3D Moderate AP_R40",
        "arms": arms_report,
        "evidence": {
            "fixed_point_rcnn_initialization": file_record(PROJECT_ROOT / "tools/PointRCNN.pth"),
            "fixed_pu_gcn_checkpoint": [file_record(path) for path in PU_GCN_CHECKPOINT_FILES],
            "schedule_extension_protocol": (
                extension_protocol_record(
                    run_root / "schedule_extension_protocol.json", args.expected_epochs
                )
                if (run_root / "schedule_extension_protocol.json").is_file()
                else None
            ),
            "fixed_input_protocols": {
                arm: {
                    "train": input_protocol_record(paths["train"], 3712),
                    "validation": input_protocol_record(paths["validation"], 3769),
                    "strict_x4_validation_audit": strict_x4_audit_record(paths["strict_x4_audit"]),
                }
                for arm, paths in INPUT_PROTOCOLS.items()
            },
            "stage_training_protocols": {
                arm: {
                    "rpn": training_protocol_record(
                        run_root / "pointrcnn" / arm / "rpn/adaptation_protocol.json",
                        "rpn",
                        args.expected_epochs,
                    ),
                    "rcnn": training_protocol_record(
                        run_root
                        / "pointrcnn"
                        / arm
                        / f"rcnn_from_rpn_epoch_{arms_report[arm]['selected_rpn_epoch']}"
                        / "adaptation_protocol.json",
                        "rcnn_offline",
                        args.expected_epochs,
                    ),
                }
                for arm in ARMS
            },
            "source_code": [file_record(path) for path in CODE_FILES],
        },
    }
    json_path = report_root / "pointrcnn_convergence_summary.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_aggregate(run_root, summaries)

    lines = [
        "# PU-GCN PointRCNN adaptation convergence",
        "",
        "Overall status: **CONVERGED**",
        "",
        "PU-GCN remains fixed at the released PU1K model-100 checkpoint. RPN and offline RCNN adaptation are assessed stage-wise.",
        "",
        f"The previous 3-epoch run was a fixed compute-budget experiment, not convergence evidence. It remains a historical comparison. Since Adam OneCycle depends on the planned total epochs, the present run uses one clean {args.expected_epochs}-epoch schedule from the same initialization instead of appending to the old epoch-3 checkpoint.",
        "",
        "Each AP value uses all 3,769 KITTI validation frames. Stopping criterion: Car 3D Moderate AP_R40, min_delta=0.2 AP, patience=3 epochs.",
        "",
        "| Arm | Selected RPN epoch | Selected RCNN epoch | Best AP | Previous e3 | Change vs e3 | Same-input unadapted | Gain vs unadapted | Baseline AP | Gap to baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in arms_report.items():
        lines.append(
            f"| {arm} | {item['selected_rpn_epoch']} | {item['selected_rcnn_epoch']} | "
            f"{item['final_best_ap']:.4f} | {item['old_epoch3_ap']:.4f} | "
            f"{item['gain_vs_old_epoch3_ap']:+.4f} | "
            f"{item['unadapted_same_input_ap']:.4f} | "
            f"{item['gain_vs_unadapted_same_input_ap']:+.4f} | "
            f"{item['baseline_ap']:.4f} | "
            f"{item['gap_to_existing_baseline_ap']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "The companion JSON records SHA-256 hashes for the fixed PointRCNN initialization, all three PU-GCN model-100 checkpoint files, the exact train/validation generation protocols, strict 4x validation audits, stage training protocols, and execution source files.",
        ]
    )
    (report_root / "pointrcnn_convergence_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "stage":
        return summarize_stage(args)
    return aggregate(args)


if __name__ == "__main__":
    raise SystemExit(main())
