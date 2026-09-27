#!/usr/bin/env python3
"""Extend only Line B RCNN using separately recorded, complete OneCycle schedules.

Never append epochs to an exhausted OneCycle schedule. Each larger schedule
starts from the same selected RPN + official RCNN initialization; a restart
within an unchanged schedule may resume its own checkpoint. Existing runs are
read-only. All checkpoints in each schedule receive full KITTI validation.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import summarize_pugcn_pointrcnn_convergence_20260914 as audit

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/pugcn_detector_convergence_e18_20260915"
DEST = ROOT / "results/pugcn_line_b_rcnn_to_convergence_20260916"
SCRATCH = Path("/tmp/pugcn_line_b_rcnn_to_convergence_20260916")
PY = Path("/home/ra87racy/miniconda3/envs/pointrcnn_old/bin/python")
ARM = "line_b_pugcn_observed_first"
TRAIN_INPUT = ROOT / "results/pugcn_full_retrain_20260824/centerpoint_inputs" / ARM
VAL_INPUT = ROOT / "results/pugcn_detector_adaptation_full_val_20260908/inputs" / ARM
TRAIN_SPLIT = ROOT / "data/KITTI/ImageSets/train.txt"
VAL_SPLIT = ROOT / "data/KITTI/ImageSets/val.txt"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def preflight():
    summaries = {}
    for arm in audit.ARMS:
        for phase in audit.PHASES:
            path = BASE / "reports" / f"pointrcnn_{arm}_{phase}_convergence.json"
            summary = read(path)
            rows = audit.load_stage_rows(BASE, arm, phase, 18)
            recalculated = audit.early_stop(rows, 0.2, 3)
            for key in ("global_best_epoch", "global_best_ap", "terminal_plateau_reached"):
                require(summary[key] == recalculated[key], f"Summary mismatch: {path}: {key}")
            needs_extension = arm == ARM and phase == "rcnn"
            require(summary["status"] == ("NOT_YET_CONVERGED" if needs_extension else "CONVERGED"),
                    f"Unexpected predecessor status: {path}")
            summaries[(arm, phase)] = summary
    require(summaries[(ARM, "rpn")]["global_best_epoch"] == 14, "RPN selection changed")
    for path, count in ((TRAIN_SPLIT, 3712), (VAL_SPLIT, 3769)):
        ids = path.read_text().split()
        require(len(ids) == len(set(ids)) == count, f"Invalid split: {path}")
    return summaries


def run(script, arguments, log):
    command = [str(PY), str(ROOT / "scripts" / script), *map(str, arguments)]
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", script, "log=", log, flush=True)
    with log.open("a") as stream:
        stream.write("\nCOMMAND " + json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def evaluation_valid(marker, checkpoint):
    if not marker.is_file():
        return False
    x = read(marker)
    require(x.get("status") == "PASS", f"Invalid evaluation status: {marker}")
    require(all(x.get(k) == 3769 for k in
                ("frame_count", "prediction_count", "actual_consumed_frame_count")),
            f"Incomplete full validation: {marker}")
    require(x.get("checkpoint") == str(checkpoint), f"Wrong checkpoint: {marker}")
    require(x.get("ap_protocol") == "KITTI_AP_R40_41_precision_samples_exclude_recall_zero",
            f"Wrong AP protocol: {marker}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    previous = preflight()
    rpn = BASE / "pointrcnn" / ARM / "rpn/ckpt/checkpoint_epoch_14.pth"
    init = BASE / "pointrcnn" / ARM / "selected_rpn_epoch_14_official_rcnn.pth"
    records = {"rpn": audit.file_record(rpn), "initialization": audit.file_record(init),
               "predecessor_rcnn_summary": audit.file_record(
                   BASE / "reports" / f"pointrcnn_{ARM}_rcnn_convergence.json")}
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_PASS", "next_schedule_epochs": 24,
                          "fixed_inputs": str(TRAIN_INPUT), "evidence": records}, indent=2))
        return

    DEST.mkdir(parents=True, exist_ok=True)
    lock = (DEST / "driver.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = {
        "scope": "Line B offline RCNN only; Line A and Line B RPN remain completed",
        "criterion": {"metric": "Car 3D Moderate AP_R40", "min_delta": 0.2, "patience": 3},
        "schedule_policy": "24,30,36,... complete OneCycle schedules from the same initialization",
        "reason": "18-epoch predecessor ended with only 2 non-improving epochs after e16 improvement",
        "resume_policy": "Only resume within an unchanged total-epoch schedule",
        "evidence": records, "training_split": str(TRAIN_SPLIT),
        "validation_split": str(VAL_SPLIT), "train_input": str(TRAIN_INPUT),
        "val_input": str(VAL_INPUT), "driver": audit.file_record(Path(__file__)),
    }
    protocol_path = DEST / "extension_protocol.json"
    if protocol_path.exists():
        require(read(protocol_path) == protocol, "Extension protocol changed; refusing to mix experiments")
    else:
        write(protocol_path, protocol)

    export = SCRATCH / "rpn_export_epoch_14"
    export_marker = export / "export_complete.json"
    write(DEST / "status.json", {"status": "RUNNING", "stage": "RPN_FEATURE_EXPORT", "pid": os.getpid()})
    if not export_marker.exists():
        run("run_pointrcnn_rpn_feature_export.py", [
            "--lidar-dir", TRAIN_INPUT, "--split-file", TRAIN_SPLIT,
            "--rpn-ckpt", rpn, "--output-dir", export, "--epoch-label", 14, "--workers", 0,
        ], DEST / "rpn_export_epoch_14.log")
    x = read(export_marker)
    require(x.get("status") == "PASS" and x.get("frames") == x.get("rois") == 3712
            and x.get("rpn_ckpt") == str(rpn), "Export audit failed")
    write(DEST / "export_complete_evidence.json", x)
    feature_dir = Path(x["feature_dir"])
    roi_dir = Path(x["roi_dir"])
    require(len(list(feature_dir.glob("*_xyz.npy"))) == 3712
            and len(list(roi_dir.glob("*.txt"))) == 3712, "Export files missing")

    epochs = 24
    while True:
        schedule = DEST / f"schedule_{epochs}"
        train_dir = schedule / "pointrcnn" / ARM / "rcnn_from_rpn_epoch_14"
        ckpt_dir = train_dir / "ckpt"
        write(DEST / "status.json", {"status": "RUNNING", "stage": "RCNN_TRAIN",
                                    "schedule_epochs": epochs, "run_root": str(schedule), "pid": os.getpid()})
        if not (ckpt_dir / f"checkpoint_epoch_{epochs}.pth").exists():
            saved = sorted(ckpt_dir.glob("checkpoint_epoch_*.pth"),
                           key=lambda p: int(p.stem.rsplit("_", 1)[-1]))
            arguments = ["--mode", "rcnn_offline", "--lidar-dir", TRAIN_INPUT,
                         "--split-file", TRAIN_SPLIT, "--output-dir", train_dir,
                         "--init-ckpt", init, "--epochs", epochs, "--ckpt-save-interval", 1,
                         "--workers", 0, "--batch-size", 1,
                         "--rcnn-training-roi-dir", roi_dir,
                         "--rcnn-training-feature-dir", feature_dir]
            if saved:
                require(read(train_dir / "adaptation_protocol.json")["epochs"] == epochs,
                        "Cannot resume a different OneCycle schedule")
                arguments += ["--resume-ckpt", saved[-1]]
            run("run_pointrcnn_full_train_stage.py", arguments, train_dir / "runner_stdout.log")
        for epoch in range(1, epochs + 1):
            checkpoint = ckpt_dir / f"checkpoint_epoch_{epoch}.pth"
            require(checkpoint.is_file() and checkpoint.stat().st_size > 0,
                    f"Missing trained checkpoint: {checkpoint}")
            combined = schedule / "pointrcnn" / ARM / "combined_rcnn_curve_rpn_epoch_14" / checkpoint.name
            if not combined.exists():
                run("combine_pointrcnn_rpn_with_official_rcnn.py", [
                    "--official", ROOT / "tools/PointRCNN.pth", "--rpn", rpn,
                    "--rcnn", checkpoint, "--output", combined,
                ], schedule / "combine.log")
            out = schedule / "evaluations/pointrcnn" / f"{ARM}_rcnn_epoch_{epoch}"
            write(DEST / "status.json", {"status": "RUNNING", "stage": "RCNN_FULL_VALIDATION",
                                        "schedule_epochs": epochs, "epoch": epoch,
                                        "run_root": str(schedule), "pid": os.getpid()})
            if not evaluation_valid(out / "run_complete.json", combined):
                run("run_pointrcnn_checkpoint_split_eval.py", [
                    "--lidar-dir", VAL_INPUT, "--checkpoint", combined,
                    "--split-file", VAL_SPLIT, "--split-name", "val", "--output-dir", out,
                    "--batch-size", 1, "--workers", 0, "--seed", 20260908,
                    "--resume", "--cleanup-heavy-artifacts",
                ], out / "runner_stdout.log")
                require(evaluation_valid(out / "run_complete.json", combined), "Validation failed")
        run("summarize_pugcn_pointrcnn_convergence_20260914.py", [
            "stage", "--run-root", schedule, "--arm", ARM, "--phase", "rcnn",
            "--expected-epochs", epochs, "--min-delta", 0.2, "--patience", 3,
        ], schedule / "summary.log")
        summary_path = schedule / "reports" / f"pointrcnn_{ARM}_rcnn_convergence.json"
        summary = read(summary_path)
        print("SCHEDULE_RESULT", epochs, summary["status"], summary["global_best_ap"], flush=True)
        if summary["status"] == "CONVERGED":
            phases = {f"{arm}/{phase}": item for (arm, phase), item in previous.items()}
            phases[f"{ARM}/rcnn"] = summary
            require(all(item["status"] == "CONVERGED" for item in phases.values()), "Incomplete study")
            write(DEST / "final_comparison.json", {
                "status": "CONVERGED", "note": "Mixed schedule lengths; each phase uses its own complete curve",
                "phases": phases, "extension_protocol": protocol,
                "predecessor_line_b_rcnn": previous[(ARM, "rcnn")],
                "selected_line_b_rcnn_summary": audit.file_record(summary_path),
            })
            lines = ["# PointRCNN convergence results", "",
                     "Metric: Car 3D Moderate AP_R40, all 3,769 KITTI validation frames.", "",
                     "| Arm | Selected RPN epoch | RCNN schedule | Best RCNN epoch | AP | Gain vs unadapted | Change vs old 3-epoch | Gap to baseline |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for arm in audit.ARMS:
                item = phases[f"{arm}/rcnn"]
                values = [arm, phases[f"{arm}/rpn"]["global_best_epoch"], item["expected_epochs"],
                          item["global_best_epoch"], *[f"{item[k]:.4f}" for k in
                          ("global_best_ap", "gain_vs_unadapted_same_input_ap", "change_vs_old_epoch3_ap", "gap_to_baseline_ap")]]
                lines.append("| " + " | ".join(map(str, values)) + " |")
            lines += ["", "Line A baseline: official detector on original-N input. Line B baseline: existing 3-epoch adapted sparse input.",
                      "", "The earlier 18-epoch Line B RCNN best remains 56.88760049103507 (epoch 16), but that schedule did not meet terminal patience.",
                      "", "Longer RCNN schedules are separate experiments from the same initialization, not appended epochs. Full curves and source paths are in final_comparison.json."]
            (DEST / "FINAL_RESULTS.md").write_text("\n".join(lines) + "\n")
            write(DEST / "status.json", {"status": "CONVERGED", "schedule_epochs": epochs,
                                        "best_ap": summary["global_best_ap"], "report": str(DEST / "final_comparison.json")})
            return
        require(summary["status"] == "NOT_YET_CONVERGED", "Unexpected summary status")
        epochs += 6


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DRIVER_FAILED: {exc}", file=sys.stderr, flush=True)
        raise
