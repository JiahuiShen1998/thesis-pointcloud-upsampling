#!/usr/bin/env python3
"""Line B PU-GCN quality metrics vs original (CD / HD / NUC)."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from compute_lineB_quality_metrics import (
    load_manifest_off_map,
    run_comparison_group,
    summarize,
)
from modelnet40_x4_protocol import (
    ORIGINAL_ROOT,
    detect_original_point_count,
    lineB_paths,
    reports_dir,
)

GROUP = "downsampled_x4_up_pu_gcn_vs_original"
P2F_STATUS = (
    "P2F pending because original .off meshes exist but full mesh-distance computation is slow."
)


def merge_into_completed_summary(new_rows: list[dict], new_summaries: list[dict]) -> None:
    rep = reports_dir()
    completed_sum = rep / "modelnet40_lineB_completed_methods_quality_metrics_summary.csv"
    completed_per = rep / "modelnet40_lineB_completed_methods_quality_metrics_per_sample.csv"

    existing_sum: list[dict] = []
    if completed_sum.is_file():
        with open(completed_sum, newline="", encoding="utf-8") as handle:
            existing_sum = [r for r in csv.DictReader(handle) if r.get("comparison_group") != GROUP]

    existing_per: list[dict] = []
    if completed_per.is_file():
        with open(completed_per, newline="", encoding="utf-8") as handle:
            existing_per = [r for r in csv.DictReader(handle) if r.get("comparison_group") != GROUP]

    all_sum = existing_sum + new_summaries
    all_per = existing_per + new_rows

    if all_sum:
        with open(completed_sum, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_sum[0].keys()))
            writer.writeheader()
            writer.writerows(all_sum)
    if all_per:
        with open(completed_per, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_per[0].keys()))
            writer.writeheader()
            writer.writerows(all_per)


def main() -> int:
    n = detect_original_point_count()
    down_n = n // 4
    strict = lineB_paths("pu_gcn")["strict_N"]
    if not strict.is_dir():
        print(f"Missing strict outputs: {strict}")
        return 1

    off_map = load_manifest_off_map()
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

    rep = reports_dir()
    per_csv = rep / "modelnet40_lineB_pugcn_quality_metrics_per_sample.csv"
    sum_csv = rep / "modelnet40_lineB_pugcn_quality_metrics_summary.csv"

    summaries = []
    for split in ("train", "test", "all"):
        if split == "all":
            subset = [r for r in rows if r["comparison_group"] == GROUP]
            if not subset:
                continue
            import numpy as np

            s = summarize(rows, "PU-GCN", GROUP, "train")
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
            s = summarize(rows, "PU-GCN", GROUP, split)
            if s:
                s["p2f_status"] = P2F_STATUS
                summaries.append(s)

    if rows:
        with open(per_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if summaries:
        with open(sum_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    merge_into_completed_summary(rows, summaries)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"Computed {len(rows)} per-sample metrics at {ts}")
    print(f"Wrote {per_csv}, {sum_csv}")
    print(f"Merged into modelnet40_lineB_completed_methods_quality_metrics_summary.csv")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
