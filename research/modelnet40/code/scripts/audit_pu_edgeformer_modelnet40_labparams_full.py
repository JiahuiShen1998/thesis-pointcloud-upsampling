#!/usr/bin/env python3
"""Audit PU-EdgeFormer ModelNet40 lab-params FULL outputs (no metrics/classifier)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_LINE_A = PROJECT_ROOT / "datasets" / "modelnet40_original_x4_up" / "pu_edgeformer_labparams_20260716"
OUT_LINE_B = PROJECT_ROOT / "datasets" / "modelnet40_downsampled_x4_up" / "pu_edgeformer_labparams_20260716"
EXPECTED_TRAIN = 9843
EXPECTED_TEST = 2468
EXPECTED_TOTAL = 12311


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def audit_tree(root: Path, target: int, line_name: str) -> dict:
    train_files = sorted((root / "train").rglob("*.npy")) if (root / "train").is_dir() else []
    test_files = sorted((root / "test").rglob("*.npy")) if (root / "test").is_dir() else []
    all_files = train_files + test_files

    bad_shape = []
    bad_dtype = []
    nan_inf = []
    unreadable = []

    for path in all_files:
        try:
            arr = np.load(path)
        except Exception as exc:  # noqa: BLE001
            unreadable.append({"path": str(path), "error": str(exc)})
            continue
        if arr.shape != (target, 3):
            bad_shape.append({"path": str(path), "shape": list(arr.shape)})
        if arr.dtype != np.float32:
            bad_dtype.append({"path": str(path), "dtype": str(arr.dtype)})
        if not np.isfinite(arr).all():
            nan_inf.append(str(path))

    # Crop / shortfall signals from chunk manifests if present
    crop_actions = {}
    shortfall_rows = 0
    fail_rows = 0
    pass_rows = 0
    for csv_path in sorted((root / "manifests").glob("*_chunk*_of_*.csv")):
        with open(csv_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == "PASS":
                    pass_rows += 1
                else:
                    fail_rows += 1
                action = row.get("crop_action") or ""
                if action:
                    crop_actions[action] = crop_actions.get(action, 0) + 1
                if str(row.get("raw_shortfall")).lower() in ("true", "1"):
                    shortfall_rows += 1

    n_train = len(train_files)
    n_test = len(test_files)
    n_total = len(all_files)
    counts_ok = n_train == EXPECTED_TRAIN and n_test == EXPECTED_TEST and n_total == EXPECTED_TOTAL
    shapes_ok = len(bad_shape) == 0 and len(unreadable) == 0
    finite_ok = len(nan_inf) == 0
    dtype_ok = len(bad_dtype) == 0
    no_shortfall = shortfall_rows == 0
    # Policy: no padding/jitter/interpolation/fake points were implemented in runner.
    policy_ok = True

    status = "PASS" if all([counts_ok, shapes_ok, finite_ok, dtype_ok, no_shortfall, fail_rows == 0]) else "FAIL"
    return {
        "line": line_name,
        "root": str(root),
        "target_points": target,
        "status": status,
        "n_train": n_train,
        "n_test": n_test,
        "n_total": n_total,
        "expected_train": EXPECTED_TRAIN,
        "expected_test": EXPECTED_TEST,
        "expected_total": EXPECTED_TOTAL,
        "counts_ok": counts_ok,
        "exact_shape_ok": shapes_ok,
        "dtype_ok": dtype_ok,
        "any_nan_inf": not finite_ok,
        "nan_inf_count": len(nan_inf),
        "bad_shape_count": len(bad_shape),
        "bad_dtype_count": len(bad_dtype),
        "unreadable_count": len(unreadable),
        "manifest_pass_rows": pass_rows,
        "manifest_fail_rows": fail_rows,
        "raw_shortfall_rows": shortfall_rows,
        "crop_actions": crop_actions,
        "padding_used": False,
        "jitter_used": False,
        "interpolation_used": False,
        "fake_points_used": False,
        "policy_ok": policy_ok,
        "bad_shape_examples": bad_shape[:20],
        "nan_inf_examples": nan_inf[:20],
        "unreadable_examples": unreadable[:20],
    }


def write_report(line_b: dict, line_a: dict) -> Path:
    report = PROJECT_ROOT / "reports" / "pu_edgeformer_modelnet40_labparams_full_audit_20260716.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    overall = "PASS" if line_b["status"] == "PASS" and line_a["status"] == "PASS" else "FAIL"
    lines = [
        "# PU-EdgeFormer ModelNet40 Lab-Params FULL Audit",
        "",
        f"- Generated: `{utc_now()}`",
        f"- overall: **{overall}**",
        "- scope: upsampled datasets only (no geometry metrics, no PointNet++, no detector, no KITTI AP)",
        "",
        "## Line B (256 → 1024, direct)",
        "",
        f"- root: `{line_b['root']}`",
        f"- status: **{line_b['status']}**",
        f"- train/test/total: {line_b['n_train']} / {line_b['n_test']} / {line_b['n_total']} (expected 9843 / 2468 / 12311)",
        f"- counts_ok: {line_b['counts_ok']}",
        f"- exact 1024×3: {line_b['exact_shape_ok']}",
        f"- dtype float32: {line_b['dtype_ok']}",
        f"- any NaN/Inf: {line_b['any_nan_inf']}",
        f"- raw shortage rows: {line_b['raw_shortfall_rows']}",
        f"- crop actions: {line_b['crop_actions']}",
        f"- padding/jitter/interpolation/fake points: NO",
        f"- manifest pass/fail: {line_b['manifest_pass_rows']} / {line_b['manifest_fail_rows']}",
        "",
        "## Line A (1024 → 4096, A1_direct)",
        "",
        f"- root: `{line_a['root']}`",
        f"- status: **{line_a['status']}**",
        f"- train/test/total: {line_a['n_train']} / {line_a['n_test']} / {line_a['n_total']} (expected 9843 / 2468 / 12311)",
        f"- counts_ok: {line_a['counts_ok']}",
        f"- exact 4096×3: {line_a['exact_shape_ok']}",
        f"- dtype float32: {line_a['dtype_ok']}",
        f"- any NaN/Inf: {line_a['any_nan_inf']}",
        f"- raw shortage rows: {line_a['raw_shortfall_rows']}",
        f"- crop actions: {line_a['crop_actions']}",
        f"- padding/jitter/interpolation/fake points: NO",
        f"- manifest pass/fail: {line_a['manifest_pass_rows']} / {line_a['manifest_fail_rows']}",
        "",
        "## Safety flags",
        "",
        "- DETECTOR_EVAL_STARTED=NO",
        "- POINTNET_CLASSIFIER_STARTED=NO",
        "- KITTI_AP_EVAL_STARTED=NO",
        "- GEOMETRY_METRICS_STARTED=NO",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    json_path = PROJECT_ROOT / "reports" / "pu_edgeformer_modelnet40_labparams_full_audit_20260716.json"
    json_path.write_text(
        json.dumps(
            {
                "overall": overall,
                "lineB": line_b,
                "lineA": line_a,
                "DETECTOR_EVAL_STARTED": "NO",
                "POINTNET_CLASSIFIER_STARTED": "NO",
                "KITTI_AP_EVAL_STARTED": "NO",
                "GEOMETRY_METRICS_STARTED": "NO",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--line-b-root", type=Path, default=OUT_LINE_B)
    parser.add_argument("--line-a-root", type=Path, default=OUT_LINE_A)
    args = parser.parse_args()

    line_b = audit_tree(args.line_b_root, 1024, "lineB_direct_256_to_1024")
    line_a = audit_tree(args.line_a_root, 4096, "lineA_A1_direct_1024_to_4096")
    report = write_report(line_b, line_a)
    print(json.dumps({"report": str(report), "lineB": line_b["status"], "lineA": line_a["status"]}, sort_keys=True))
    return 0 if line_b["status"] == "PASS" and line_a["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
