#!/usr/bin/env python3
"""Refresh on-disk progress for the running PU-GCN full-validation pipeline."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shutil
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/pugcn_detector_adaptation_full_val_20260908"


def snapshot() -> dict:
    output = {"updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "total_frames": 3769}
    generation = {}
    for line, run, variant, folder in (
        ("A", "val3769_linea", "pu_gcn_linea_surface_pr1_c32", "line_a_original_x4_up"),
        ("B", "val3769_lineb", "pu_gcn_lineb_c2048_r4", "line_b_downsampled_x4_up"),
    ):
        base = RESULT / "generation" / run / variant / folder
        checked = 0
        try:
            with (base / "run_manifest.csv").open() as handle:
                checked = sum(row.get("status") in ("PASS", "PASS_REUSED") for row in csv.DictReader(handle))
        except FileNotFoundError:
            pass
        generation[line] = {
            "patch_metadata_files": len(list((base / "manifests").glob("*_patch_metadata.json"))),
            "final_bin_files": len(list((base / "final_bin").glob("*.bin"))),
            "completed_frames_in_manifest": checked,
        }
    output["generation"] = generation
    evaluations = []
    for detector, pattern in (("pointrcnn", "*/run_complete.json"), ("centerpoint", "*/result_summary.json")):
        for path in (RESULT / "evaluations" / detector).glob(pattern):
            try:
                payload = json.loads(path.read_text())
                evaluations.append({"detector": detector, "arm": path.parent.name, "status": payload.get("status"), "frames": payload.get("frame_count", payload.get("frames"))})
            except (OSError, ValueError):
                pass
    output["completed_evaluations"] = evaluations
    output["free_gib"] = round(shutil.disk_usage(RESULT).free / 1024**3, 2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch-pid", type=int)
    args = parser.parse_args()
    while True:
        payload = snapshot()
        active = None
        if args.watch_pid:
            try:
                os.kill(args.watch_pid, 0)
                active = True
            except ProcessLookupError:
                active = False
            payload["pipeline_pid"] = args.watch_pid
            payload["pipeline_process_alive"] = active
        report = RESULT / "reports/current_progress.json"
        temp = report.with_suffix(".json.tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n")
        temp.replace(report)
        lines = ["# Full KITTI validation progress", "", f"Updated (UTC): {payload['updated_utc']}", "", "| Line | Patch metadata | Final point-cloud files | Completed in manifest | Total |", "|---|---:|---:|---:|---:|"]
        for line, counts in payload["generation"].items():
            lines.append(f"| {line} | {counts['patch_metadata_files']} | {counts['final_bin_files']} | {counts['completed_frames_in_manifest']} | 3769 |")
        complete = sum(row["status"] == "PASS" and row["frames"] == 3769 for row in payload["completed_evaluations"])
        lines.extend(["", f"Completed detector evaluations: {complete}/20.", "", f"Free disk: {payload['free_gib']} GiB.", "", "File counts describe progress only. Final matrix acceptance also verifies AP protocol, prediction coverage and paired detector inputs.", ""])
        if active is not None:
            lines.append(f"Pipeline process alive: {active}; PID: {args.watch_pid}.")
        (RESULT / "reports/current_progress.md").write_text("\n".join(lines))
        print(json.dumps(payload, separators=(",", ":")), flush=True)
        if not args.watch_pid or active is False:
            break
        time.sleep(30)


if __name__ == "__main__":
    main()
