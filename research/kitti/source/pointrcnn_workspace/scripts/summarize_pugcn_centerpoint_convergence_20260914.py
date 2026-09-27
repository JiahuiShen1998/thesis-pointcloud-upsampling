#!/usr/bin/env python3
"""Summarize per-epoch full-validation evidence for PU-GCN detector adaptation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ARMS = ("line_a_pugcn_observed_first", "line_b_pugcn_observed_first")
REPO = Path(__file__).resolve().parents[1]
OLD_EVAL = REPO / "results/pugcn_detector_adaptation_full_val_20260908/evaluations/centerpoint"
REFERENCES = {
    "line_a_pugcn_observed_first": {
        "old_epoch3": OLD_EVAL / "fullval3769_20260908_line_a_pugcn_observed_first_adapted/result_summary.json",
        "unadapted_same_input": OLD_EVAL / "fullval3769_20260908_line_a_pugcn_observed_first_official/result_summary.json",
        "baseline": OLD_EVAL / "fullval3769_20260908_line_a_baseline_official/result_summary.json",
        "baseline_label": "official detector on original-N Line A input",
    },
    "line_b_pugcn_observed_first": {
        "old_epoch3": OLD_EVAL / "fullval3769_20260908_line_b_pugcn_observed_first_adapted/result_summary.json",
        "unadapted_same_input": OLD_EVAL / "fullval3769_20260908_line_b_pugcn_observed_first_official/result_summary.json",
        "baseline": OLD_EVAL / "fullval3769_20260908_line_b_baseline_adapted/result_summary.json",
        "baseline_label": "existing 3-epoch adapted sparse Line B baseline",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    parser.add_argument("--min-delta", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def early_stop(rows: list[dict[str, object]], min_delta: float, patience: int) -> dict[str, object]:
    best_value = float("-inf")
    best_epoch = -1
    bad_epochs = 0
    stop_epoch = None
    best_epoch_at_first_stop = None
    best_ap_at_first_stop = None
    for row in rows:
        value = float(row["car_3d_ap_r40_moderate"])
        epoch = int(row["epoch"])
        if value > best_value + min_delta:
            best_value = value
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if stop_epoch is None and bad_epochs >= patience:
                stop_epoch = epoch
                best_epoch_at_first_stop = best_epoch
                best_ap_at_first_stop = best_value
    global_best = max(rows, key=lambda row: float(row["car_3d_ap_r40_moderate"]))
    return {
        "criterion": "Car 3D Moderate AP_R40; improvement must exceed min_delta",
        "min_delta_ap_points": min_delta,
        "patience_epochs": patience,
        "early_stop_triggered": stop_epoch is not None,
        "terminal_plateau_reached": bad_epochs >= patience,
        "terminal_non_improving_epochs": bad_epochs,
        "first_stop_epoch": stop_epoch,
        "best_epoch_at_first_stop": best_epoch_at_first_stop,
        "best_ap_at_first_stop": best_ap_at_first_stop,
        "global_best_epoch": int(global_best["epoch"]),
        "global_best_ap": float(global_best["car_3d_ap_r40_moderate"]),
    }


def load_car_moderate(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS" or int(payload.get("frames", 0)) != 3769:
        raise RuntimeError(f"invalid reference result: {path}")
    return float(payload["metrics_percent"]["Car"]["3d_ap_r40"]["moderate"])


def plot_curves(
    rows_by_arm: dict[str, list[dict[str, object]]],
    summaries: dict[str, dict[str, object]],
    report_root: Path,
) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        (report_root / "centerpoint_epoch_validation_curve.plot_pending.txt").write_text(
            "Run this summarizer with /home/ra87racy/miniconda3/envs/pointrcnn_old/bin/python "
            "to render the PNG/PDF AP curves.\n",
            encoding="utf-8",
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "line_a_pugcn_observed_first": "Line A: original N + selected generated 3N",
        "line_b_pugcn_observed_first": "Line B: sparse M + selected generated 3M",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), constrained_layout=True)
    for axis, arm in zip(axes, ARMS):
        rows = rows_by_arm[arm]
        epochs = [int(row["epoch"]) for row in rows]
        values = [float(row["car_3d_ap_r40_moderate"]) for row in rows]
        item = summaries[arm]
        axis.plot(epochs, values, color="#1570A6", marker="o", linewidth=2.2, label="12-epoch schedule")
        axis.axhline(float(item["old_epoch3_ap"]), color="#E69F00", linestyle="--", linewidth=1.7, label="previous 3-epoch")
        axis.axhline(float(item["baseline_ap"]), color="#C43C39", linestyle=":", linewidth=2.0, label="existing baseline")
        best_epoch = int(item["global_best_epoch"])
        best_ap = float(item["global_best_ap"])
        axis.scatter([best_epoch], [best_ap], s=85, color="#009E73", zorder=5, label="best epoch")
        axis.annotate(
            f"best e{best_epoch}: {best_ap:.2f}",
            (best_epoch, best_ap),
            xytext=(7, 8),
            textcoords="offset points",
            fontsize=9,
        )
        axis.set_title(titles[arm])
        axis.set_xlabel("Detector adaptation epoch")
        axis.set_ylabel("Car 3D Moderate AP_R40 (%)")
        axis.set_xticks(epochs)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, loc="best")
    fig.suptitle("CenterPoint adaptation on fixed PU-GCN outputs — full KITTI validation (3,769 frames)", fontsize=13)
    fig.savefig(report_root / "centerpoint_epoch_validation_curve.png", dpi=220)
    fig.savefig(report_root / "centerpoint_epoch_validation_curve.pdf")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    eval_root = run_root / "evaluations/centerpoint"
    report_root = run_root / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    summaries: dict[str, dict[str, object]] = {}
    rows_by_arm: dict[str, list[dict[str, object]]] = {}

    for arm in ARMS:
        arm_rows: list[dict[str, object]] = []
        for epoch in range(1, args.expected_epochs + 1):
            path = eval_root / f"{arm}_epoch_{epoch}" / "result_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "PASS" or int(payload.get("frames", 0)) != 3769:
                raise RuntimeError(f"incomplete evaluation: {path}")
            metrics = payload["metrics_percent"]
            row = {
                "arm": arm,
                "epoch": epoch,
                "car_3d_ap_r40_easy": metrics["Car"]["3d_ap_r40"]["easy"],
                "car_3d_ap_r40_moderate": metrics["Car"]["3d_ap_r40"]["moderate"],
                "car_3d_ap_r40_hard": metrics["Car"]["3d_ap_r40"]["hard"],
                "car_bev_ap_r40_moderate": metrics["Car"]["bev_ap_r40"]["moderate"],
                "checkpoint": payload["checkpoint"],
                "frames": payload["frames"],
            }
            arm_rows.append(row)
            all_rows.append(row)
        rows_by_arm[arm] = arm_rows
        summary = early_stop(arm_rows, args.min_delta, args.patience)
        refs = REFERENCES[arm]
        old_epoch3_ap = load_car_moderate(refs["old_epoch3"])
        unadapted_same_input_ap = load_car_moderate(refs["unadapted_same_input"])
        baseline_ap = load_car_moderate(refs["baseline"])
        best_ap = float(summary["global_best_ap"])
        summary.update(
            {
                "old_epoch3_ap": old_epoch3_ap,
                "gain_vs_old_epoch3_ap": best_ap - old_epoch3_ap,
                "unadapted_same_input_ap": unadapted_same_input_ap,
                "gain_vs_unadapted_same_input_ap": best_ap - unadapted_same_input_ap,
                "baseline_label": refs["baseline_label"],
                "baseline_ap": baseline_ap,
                "gap_to_existing_baseline_ap": best_ap - baseline_ap,
                "reference_files": {key: str(value) for key, value in refs.items() if isinstance(value, Path)},
            }
        )
        summaries[arm] = summary

    csv_path = report_root / "centerpoint_epoch_validation_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    status = "CONVERGED" if all(
        bool(item["terminal_plateau_reached"]) for item in summaries.values()
    ) else "NOT_YET_CONVERGED"
    report = {
        "status": status,
        "detector": "CenterPoint",
        "pu_gcn_training": "frozen released PU1K model-100; not retrained here",
        "adaptation_schedule": f"clean {args.expected_epochs}-epoch schedule from official detector checkpoint",
        "validation_frames_per_epoch": 3769,
        "primary_metric": "Car 3D Moderate AP_R40",
        "arms": summaries,
    }
    json_path = report_root / "centerpoint_convergence_summary.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_curves(rows_by_arm, summaries, report_root)

    md = [
        "# PU-GCN CenterPoint adaptation convergence",
        "",
        f"Overall status: **{status}**",
        "",
        "PU-GCN remains fixed at the released PU1K model-100 checkpoint. The epochs below are detector-adaptation epochs.",
        "",
        f"Stopping evidence uses full 3,769-frame KITTI validation Car 3D Moderate AP_R40, min_delta={args.min_delta:.3f} AP, patience={args.patience} epochs.",
        "",
        "| Arm | Terminal plateau | Best epoch | Best AP | Previous e3 | Change vs e3 | Baseline AP | Gap to baseline |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, item in summaries.items():
        md.append(
            f"| {arm} | {item['terminal_plateau_reached']} | {item['global_best_epoch']} | "
            f"{item['global_best_ap']:.4f} | {item['old_epoch3_ap']:.4f} | "
            f"{item['gain_vs_old_epoch3_ap']:+.4f} | {item['baseline_ap']:.4f} | "
            f"{item['gap_to_existing_baseline_ap']:+.4f} |"
        )
    (report_root / "centerpoint_convergence_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
