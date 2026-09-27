#!/usr/bin/env python3
"""Full-generation audit for EAR ×4 outputs (post job completion)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"

VARIANTS = (
    {
        "line": "A",
        "label": "Original + EAR ×4",
        "output_root": PROJECT_ROOT / "datasets" / "modelnet40_original_up" / "ear_x4",
        "expected_input": 1024,
        "expected_output": 4096,
        "expected_train": 9843,
        "expected_test": 2468,
        "expected_ratio": 4.0,
    },
    {
        "line": "B",
        "label": "Downsampled50 + EAR ×4",
        "output_root": PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear_x4",
        "expected_input": 512,
        "expected_output": 2048,
        "expected_train": 9843,
        "expected_test": 2468,
        "expected_ratio": 4.0,
    },
)

LEGACY_EAR_X2 = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear"


def audit_variant(v: dict) -> dict:
    meta = v["output_root"] / "metadata" / "class_to_idx.json"
    src_meta = PROJECT_ROOT / "datasets" / "modelnet40_original" / "metadata" / "class_to_idx.json"
    class_to_idx = json.loads((meta if meta.is_file() else src_meta).read_text(encoding="utf-8"))

    counts = {"train": 0, "test": 0}
    bad_shape = 0
    nan_inf = False
    ratios: list[float] = []

    for split in ("train", "test"):
        split_dir = v["output_root"] / split
        if not split_dir.is_dir():
            continue
        for fpath in split_dir.rglob("*.npy"):
            counts[split] += 1
            arr = np.load(fpath)
            if arr.shape != (v["expected_output"], 3):
                bad_shape += 1
            if not np.isfinite(arr).all():
                nan_inf = True
            ratios.append(arr.shape[0] / v["expected_input"])

    legacy_count = sum(1 for _ in LEGACY_EAR_X2.rglob("*.npy")) if LEGACY_EAR_X2.is_dir() else 0
    count_ok = counts["train"] == v["expected_train"] and counts["test"] == v["expected_test"]
    shape_ok = bad_shape == 0 and counts["train"] + counts["test"] > 0
    ratio_ok = bool(ratios) and min(ratios) == max(ratios) == v["expected_ratio"]

    status = "PASS" if count_ok and shape_ok and not nan_inf and ratio_ok else "FAIL"
    issues = []
    if not count_ok:
        issues.append(f"counts {counts['train']}/{counts['test']} != {v['expected_train']}/{v['expected_test']}")
    if not shape_ok:
        issues.append(f"bad shapes: {bad_shape}")
    if nan_inf:
        issues.append("NaN/Inf")
    if not ratio_ok:
        issues.append("ratio mismatch")

    return {
        "variant": v["label"],
        "line": v["line"],
        "output_root": str(v["output_root"].relative_to(PROJECT_ROOT)),
        "train_count": counts["train"],
        "test_count": counts["test"],
        "expected_train": v["expected_train"],
        "expected_test": v["expected_test"],
        "expected_shape": f"{v['expected_output']}x3",
        "bad_shape_files": bad_shape,
        "nan_inf": nan_inf,
        "actual_ratio": f"{ratios[0]:.1f}" if ratios else "",
        "class_count": len(class_to_idx),
        "legacy_ear_x2_intact": legacy_count > 0,
        "status": status,
        "issues": "; ".join(issues),
    }


def write_md(rows: list[dict], path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    overall = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    lines = [
        "# EAR ×4 Full Generation Audit",
        "",
        f"- Generated at: {ts}",
        f"- Overall: **{overall}**",
        "",
        "| Variant | Train | Test | Shape | Ratio | Status |",
        "| --- | ---: | ---: | --- | ---: | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['train_count']}/{r['expected_train']} "
            f"| {r['test_count']}/{r['expected_test']} | {r['expected_shape']} "
            f"| {r['actual_ratio']} | **{r['status']}** |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = [audit_variant(v) for v in VARIANTS]
    csv_path = REPORTS / "ear_x4_full_generation_audit.csv"
    md_path = REPORTS / "ear_x4_full_generation_audit.md"
    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_md(rows, md_path)
    print(f"Wrote {csv_path}")
    return 0 if all(r["status"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
