#!/usr/bin/env python3
"""Quality metrics for Line B completed methods (EAR, PDANS) vs original."""

from __future__ import annotations

import csv
import shutil
from datetime import datetime, timezone
from pathlib import Path

from compute_lineB_quality_metrics import (
    load_manifest_off_map,
    run_comparison_group,
    summarize,
)
from modelnet40_x4_protocol import (
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    lineB_paths,
    reports_dir,
)

COMPLETED = {
    "ear": ("EAR", "downsampled_x4_up_ear_vs_original"),
    "pdans": ("PDANS", "downsampled_x4_up_pdans_vs_original"),
}


def main() -> int:
    n = detect_original_point_count()
    down_n = n // 4
    off_map = load_manifest_off_map()
    all_rows = []

    for method, (label, group) in COMPLETED.items():
        strict = lineB_paths(method)["strict_N"]
        if not strict.is_dir():
            continue
        rows = run_comparison_group(
            group,
            strict,
            ORIGINAL_ROOT,
            off_map,
            source_point_count=down_n,
            target_point_count=n,
            max_samples=0,
            compute_p2f=False,
        )
        all_rows.extend(rows)

    rep = reports_dir()
    per_csv = rep / "modelnet40_lineB_completed_methods_quality_metrics_per_sample.csv"
    sum_csv = rep / "modelnet40_lineB_completed_methods_quality_metrics_summary.csv"

    if all_rows:
        with open(per_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    summaries = []
    for method, (label, group) in COMPLETED.items():
        for split in ("train", "test", "all"):
            if split == "all":
                subset = [r for r in all_rows if r["comparison_group"] == group]
                if not subset:
                    continue
                s = summarize(all_rows, label, group, "train")
                s["split"] = "all"
                s["valid_samples"] = len(subset)
                s["p2f_status"] = "pending — full mesh-distance computation is slow; .off meshes exist in manifest"
                for key, src in (("cd_mean", "cd_total"), ("hd_mean", "hd_total"), ("nuc_mean", "nuc_mean")):
                    vals = [float(r[src]) for r in subset if r.get(src) not in ("", None)]
                    if vals:
                        import numpy as np
                        s[key] = float(np.mean(vals))
                        s[key.replace("_mean", "_std")] = float(np.std(vals))
                summaries.append(s)
            else:
                s = summarize(all_rows, label, group, split)
                s["p2f_status"] = "pending — full mesh-distance computation is slow; .off meshes exist in manifest"
                if s["valid_samples"] > 0:
                    summaries.append(s)

    if summaries:
        with open(sum_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    print(f"Computed {len(all_rows)} per-sample rows, {len(summaries)} summary rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
