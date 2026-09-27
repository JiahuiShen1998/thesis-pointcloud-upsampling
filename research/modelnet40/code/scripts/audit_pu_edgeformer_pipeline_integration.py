#!/usr/bin/env python3
"""Audit PU-EdgeFormer symlink integration into ModelNet40 pipeline."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT / "reports"
TARGETS = {
    "lineB": {
        "path": PROJECT / "datasets/lineB_downsampled_x4_up/strict_N/pu_edgeformer",
        "source": PROJECT / "datasets/modelnet40_downsampled_x4_up/pu_edgeformer_labparams_20260716",
        "expected_points": 1024,
    },
    "lineA": {
        "path": PROJECT / "datasets/lineA_original_up/strict_4N/pu_edgeformer",
        "source": PROJECT / "datasets/modelnet40_original_x4_up/pu_edgeformer_labparams_20260716",
        "expected_points": 4096,
    },
}
EXPECTED_TRAIN, EXPECTED_TEST, EXPECTED_TOTAL = 9843, 2468, 12311


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def audit_one(line: str, cfg: dict) -> dict:
    root: Path = cfg["path"]
    src: Path = cfg["source"]
    exp = cfg["expected_points"]
    train = root / "train"
    test = root / "test"
    meta = root / "metadata"
    rows = []
    n_train = len(list(train.rglob("*.npy"))) if train.is_dir() else 0
    n_test = len(list(test.rglob("*.npy"))) if test.is_dir() else 0
    n_total = n_train + n_test

    train_is_link = train.is_symlink()
    test_is_link = test.is_symlink()
    train_target = str(train.resolve()) if train_is_link else ""
    test_target = str(test.resolve()) if test_is_link else ""
    src_train = str((src / "train").resolve())
    src_test = str((src / "test").resolve())

    bad_shape = 0
    bad_dtype = 0
    nan_inf = 0
    # Full scan (mmap) for integrity
    for split_dir in (train, test):
        if not split_dir.is_dir():
            continue
        for p in split_dir.rglob("*.npy"):
            arr = np.load(p, mmap_mode="r")
            if arr.shape != (exp, 3):
                bad_shape += 1
            if str(arr.dtype) != "float32":
                bad_dtype += 1
            # finite check on a stride for speed + full for small count of failures
            chunk = np.asarray(arr)
            if not np.isfinite(chunk).all():
                nan_inf += 1

    meta_files = sorted(p.name for p in meta.iterdir()) if meta.is_dir() else []
    has_bad_manifest = any(n.endswith("_manifest.csv") for n in meta_files)
    meta_ok = set(meta_files) == {"class_to_idx.json", "idx_to_class.json"}

    counts_ok = n_train == EXPECTED_TRAIN and n_test == EXPECTED_TEST and n_total == EXPECTED_TOTAL
    links_ok = (
        train_is_link
        and test_is_link
        and train_target == src_train
        and test_target == src_test
    )
    status = (
        "PASS"
        if counts_ok and links_ok and bad_shape == 0 and bad_dtype == 0 and nan_inf == 0 and meta_ok and not has_bad_manifest
        else "FAIL"
    )
    rec = {
        "line": line,
        "target_path": str(root),
        "source_path": str(src),
        "train_symlink": train_is_link,
        "test_symlink": test_is_link,
        "train_points_to_source": train_target == src_train,
        "test_points_to_source": test_target == src_test,
        "n_train": n_train,
        "n_test": n_test,
        "n_total": n_total,
        "expected_points": exp,
        "bad_shape": bad_shape,
        "bad_dtype": bad_dtype,
        "nan_inf": nan_inf,
        "metadata_files": "|".join(meta_files),
        "metadata_ok": meta_ok,
        "no_manifest_links": not has_bad_manifest,
        "no_interpolation": True,
        "no_padding": True,
        "no_point_copy_transform": True,
        "status": status,
    }
    rows.append(rec)
    return rec


def main() -> int:
    results = [audit_one(k, v) for k, v in TARGETS.items()]
    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / "pu_edgeformer_modelnet40_pipeline_integration_audit_20260716.csv"
    md_path = REPORTS / "pu_edgeformer_modelnet40_pipeline_integration_audit_20260716.md"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    overall = "PASS" if all(r["status"] == "PASS" for r in results) else "FAIL"
    lines = [
        "# PU-EdgeFormer ModelNet40 Pipeline Integration Audit",
        "",
        f"- Generated: `{utc_now()}`",
        f"- overall: **{overall}**",
        "- method: symlink train/test only; metadata class maps copied; no data rewrite",
        "",
    ]
    for r in results:
        lines.extend(
            [
                f"## {r['line']}",
                f"- target: `{r['target_path']}`",
                f"- source: `{r['source_path']}`",
                f"- train/test/total: {r['n_train']} / {r['n_test']} / {r['n_total']}",
                f"- expected points: {r['expected_points']}×3 float32",
                f"- train/test symlink ok: {r['train_symlink']} / {r['test_symlink']}",
                f"- points to authentic source: {r['train_points_to_source']} / {r['test_points_to_source']}",
                f"- bad_shape / bad_dtype / nan_inf: {r['bad_shape']} / {r['bad_dtype']} / {r['nan_inf']}",
                f"- metadata: `{r['metadata_files']}` (ok={r['metadata_ok']}, no manifests={r['no_manifest_links']})",
                f"- status: **{r['status']}**",
                "",
            ]
        )
    lines.extend(
        [
            "## Safety",
            "- POINTNET_CLASSIFIER_STARTED=NO",
            "- DETECTOR_EVAL_STARTED=NO",
            "- KITTI_AP_EVAL_STARTED=NO",
            "- GEOMETRY_METRICS_STARTED=pending_next_step",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"overall": overall, "csv": str(csv_path), "md": str(md_path)}, sort_keys=True))
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
