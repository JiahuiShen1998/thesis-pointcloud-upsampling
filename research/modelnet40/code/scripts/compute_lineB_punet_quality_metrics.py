#!/usr/bin/env python3
"""Quality metrics for Line B PU-Net vs original; merge into completed-methods summary."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from compute_lineB_quality_metrics import load_manifest_off_map, run_comparison_group, summarize
from modelnet40_x4_protocol import ORIGINAL_ROOT, detect_original_point_count, lineB_paths, reports_dir

P2F_STATUS = (
    "P2F pending because original .off meshes exist but full mesh-distance computation is slow."
)
GROUP = "downsampled_x4_up_pu_net_vs_original"
METHOD = "PU-Net"
BASELINE_SUMMARY = reports_dir() / "modelnet40_lineB_quality_metrics_summary.csv"
EXISTING_COMPLETED = reports_dir() / "modelnet40_lineB_completed_methods_quality_metrics_summary.csv"


def load_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summaries(all_rows: list[dict]) -> list[dict]:
    summaries = []
    for split in ("train", "test", "all"):
        if split == "all":
            subset = [r for r in all_rows if r["comparison_group"] == GROUP]
            if not subset:
                continue
            s = summarize(all_rows, METHOD, GROUP, "train")
            s["split"] = "all"
            s["valid_samples"] = len(subset)
            s["p2f_status"] = P2F_STATUS
            for key, src in (("cd_mean", "cd_total"), ("hd_mean", "hd_total"), ("nuc_mean", "nuc_mean")):
                vals = [float(r[src]) for r in subset if r.get(src) not in ("", None)]
                if vals:
                    s[key] = float(np.mean(vals))
                    s[key.replace("_mean", "_std")] = float(np.std(vals))
            summaries.append(s)
        else:
            s = summarize(all_rows, METHOD, GROUP, split)
            s["p2f_status"] = P2F_STATUS
            if s["valid_samples"] > 0:
                summaries.append(s)
    return summaries


def merge_completed_summary(punet_summaries: list[dict]) -> list[dict]:
    baseline_rows = load_csv_rows(BASELINE_SUMMARY)
    ear_pdans_rows = load_csv_rows(EXISTING_COMPLETED)
    if not ear_pdans_rows:
        ear_pdans_rows = [r for r in baseline_rows if r.get("comparison_group") != "downsampled_x4_vs_original"]
    else:
        ear_pdans_rows = [r for r in ear_pdans_rows if r.get("comparison_group") != GROUP]

    baseline_part = [r for r in baseline_rows if r.get("comparison_group") == "downsampled_x4_vs_original"]
    if baseline_part:
        for r in baseline_part:
            if r.get("p2f_status") == "unavailable":
                r["p2f_status"] = P2F_STATUS

    merged = baseline_part + ear_pdans_rows + punet_summaries
    order = {
        "downsampled_x4_vs_original": 0,
        "downsampled_x4_up_ear_vs_original": 1,
        "downsampled_x4_up_pdans_vs_original": 2,
        "downsampled_x4_up_pu_net_vs_original": 3,
    }
    split_order = {"train": 0, "test": 1, "all": 2}
    merged.sort(key=lambda r: (order.get(r["comparison_group"], 99), split_order.get(r["split"], 99)))
    return merged


def main() -> int:
    n = detect_original_point_count()
    down_n = n // 4
    off_map = load_manifest_off_map()
    strict = lineB_paths("pu_net")["strict_N"]
    rep = reports_dir()

    rows = run_comparison_group(
        GROUP,
        strict,
        ORIGINAL_ROOT,
        off_map,
        source_point_count=down_n,
        target_point_count=n,
        max_samples=0,
        compute_p2f=False,
    )

    per_csv = rep / "modelnet40_lineB_punet_quality_metrics_per_sample.csv"
    sum_csv = rep / "modelnet40_lineB_punet_quality_metrics_summary.csv"
    merged_csv = rep / "modelnet40_lineB_completed_methods_quality_metrics_summary.csv"

    write_csv(per_csv, rows)
    summaries = build_summaries(rows)
    write_csv(sum_csv, summaries)

    merged = merge_completed_summary(summaries)
    write_csv(merged_csv, merged)

    print(f"PU-Net per-sample: {len(rows)} rows -> {per_csv}")
    print(f"PU-Net summary: {len(summaries)} rows -> {sum_csv}")
    print(f"Merged summary: {len(merged)} rows -> {merged_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
