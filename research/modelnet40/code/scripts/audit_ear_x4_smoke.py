#!/usr/bin/env python3
"""Post-smoke audit for EAR ×4 outputs."""

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
        "label": "Original + EAR ×4 smoke",
        "output_root": PROJECT_ROOT / "datasets" / "modelnet40_original_up" / "ear_x4_smoke",
        "input_root": PROJECT_ROOT / "datasets" / "modelnet40_original",
        "expected_input": 1024,
        "expected_output": 4096,
        "expected_ratio": 4.0,
    },
    {
        "line": "B",
        "label": "Downsampled50 + EAR ×4 smoke",
        "output_root": PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear_x4_smoke",
        "input_root": PROJECT_ROOT / "datasets" / "modelnet40_downsampled50",
        "expected_input": 512,
        "expected_output": 2048,
        "expected_ratio": 4.0,
    },
)

LEGACY_EAR_X2 = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear"
EXPECTED_TRAIN = 20
EXPECTED_TEST = 20


def load_class_to_idx(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_variant(v: dict) -> dict:
    meta = v["output_root"] / "metadata" / "class_to_idx.json"
    src_meta = PROJECT_ROOT / "datasets" / "modelnet40_original" / "metadata" / "class_to_idx.json"
    class_to_idx = load_class_to_idx(meta if meta.is_file() else src_meta)

    issues: list[str] = []
    shape_ok = True
    nan_inf = False
    label_mismatch = 0
    counts = {"train": 0, "test": 0}
    bad_shapes: list[str] = []

    for split in ("train", "test"):
        split_dir = v["output_root"] / split
        if not split_dir.is_dir():
            issues.append(f"missing split dir: {split_dir}")
            continue
        for fpath in sorted(split_dir.rglob("*.npy")):
            counts[split] += 1
            arr = np.load(fpath)
            if arr.shape != (v["expected_output"], 3):
                shape_ok = False
                bad_shapes.append(f"{fpath.name}:{arr.shape}")
            if not np.isfinite(arr).all():
                nan_inf = True
            class_name = fpath.parent.name
            if class_name not in class_to_idx:
                label_mismatch += 1

    legacy_count = sum(1 for _ in LEGACY_EAR_X2.rglob("*.npy")) if LEGACY_EAR_X2.is_dir() else 0
    legacy_intact = legacy_count > 0

    status = "PASS"
    if not shape_ok or nan_inf or label_mismatch:
        status = "FAIL"
    if counts["train"] == 0 or counts["test"] == 0:
        status = "FAIL"
        issues.append("empty train or test split")
    if not legacy_intact:
        issues.append("warning: legacy ear x2 dir appears empty or missing")

    return {
        "variant": v["label"],
        "line": v["line"],
        "output_root": str(v["output_root"].relative_to(PROJECT_ROOT)),
        "train_count": counts["train"],
        "test_count": counts["test"],
        "expected_train": EXPECTED_TRAIN,
        "expected_test": EXPECTED_TEST,
        "expected_shape": f"{v['expected_output']}x3",
        "shape_ok": shape_ok,
        "bad_shapes": "; ".join(bad_shapes[:5]),
        "nan_inf": nan_inf,
        "class_count": len(class_to_idx),
        "label_mismatch_files": label_mismatch,
        "expected_ratio": v["expected_ratio"],
        "legacy_ear_x2_intact": legacy_intact,
        "legacy_ear_x2_file_count": legacy_count,
        "status": status,
        "issues": "; ".join(issues),
    }


def write_md(rows: list[dict], path: Path) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    overall = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    lines = [
        "# EAR ×4 Smoke Audit",
        "",
        f"- Generated at: {ts}",
        f"- Overall: **{overall}**",
        "",
        "| Variant | Train | Test | Shape | NaN/Inf | Classes | Legacy ×2 intact | Status |",
        "| --- | ---: | ---: | --- | --- | ---: | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['train_count']} | {r['test_count']} | {r['expected_shape']} "
            f"| {r['nan_inf']} | {r['class_count']} | {r['legacy_ear_x2_intact']} | **{r['status']}** |"
        )
    lines.extend(["", "## Details", ""])
    for r in rows:
        lines.extend([
            f"### {r['variant']}",
            "",
            f"- Output: `{r['output_root']}`",
            f"- Shape OK: {r['shape_ok']}",
            f"- Issues: {r['issues'] or 'none'}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    rows = [audit_variant(v) for v in VARIANTS]
    csv_path = REPORTS / "ear_x4_smoke_audit.csv"
    md_path = REPORTS / "ear_x4_smoke_audit.md"
    REPORTS.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    write_md(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0 if all(r["status"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
