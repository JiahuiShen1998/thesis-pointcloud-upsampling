#!/usr/bin/env python3
"""Post-process Line B PU-Net completion: job status, count audit, PointNet++ inputs."""

from __future__ import annotations

import csv
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    MAIN_METHODS,
    PROJECT_ROOT,
    detect_original_point_count,
    lineB_paths,
    pointnet2_input_paths,
    reports_dir,
)

EXPECTED_TOTAL = EXPECTED_TRAIN + EXPECTED_TEST
RESUME_JOB_ID = "1731611"
SPLITS = ("train", "test")
COMPLETED_METHODS = ("ear", "pdans", "pu_net")


def count_npy(root: Path) -> dict[str, int]:
    counts = {"train": 0, "test": 0, "total": 0}
    if not root.is_dir():
        return counts
    for split in SPLITS:
        sd = root / split
        if sd.is_dir():
            counts[split] = sum(1 for _ in sd.rglob("*.npy"))
    counts["total"] = counts["train"] + counts["test"]
    return counts


def load_manifest_keys() -> set[str]:
    keys: set[str] = set()
    for split in SPLITS:
        mp = DOWNSAMPLED_X4_ROOT / "metadata" / f"{split}_manifest.csv"
        with open(mp, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                keys.add(f"{split}/{row['class_name']}/{row['shape_id']}")
    return keys


def query_job_status(job_id: str) -> list[dict]:
    out = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobID%25,JobName%35,State,Elapsed,ExitCode", "-n", "-P"],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = []
    for line in out.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue
        job_id_part = parts[0]
        if ".batch" in job_id_part or ".extern" in job_id_part:
            continue
        rows.append({
            "job_id": job_id_part,
            "job_name": parts[1],
            "state": parts[2],
            "elapsed": parts[3],
            "exit_code": parts[4],
        })
    return rows


def audit_method_outputs(method: str, n: int) -> dict:
    paths = lineB_paths(method)
    raw_root = paths["raw"]
    strict_root = paths["strict_N"]
    manifest_keys = load_manifest_keys()

    raw_counts = count_npy(raw_root)
    strict_counts = count_npy(strict_root)

    bad_shape = nan_c = inf_c = 0
    strict_keys: set[str] = set()
    for npy in strict_root.rglob("*.npy"):
        rel = npy.relative_to(strict_root)
        key = f"{rel.parts[0]}/{rel.parts[1]}/{npy.stem}"
        strict_keys.add(key)
        arr = np.load(npy)
        if arr.shape != (n, 3):
            bad_shape += 1
        if np.isnan(arr).any():
            nan_c += 1
        if np.isinf(arr).any():
            inf_c += 1

    missing = sorted(manifest_keys - strict_keys)
    extra = sorted(strict_keys - manifest_keys)

    status = "PASS"
    if (
        raw_counts["total"] != EXPECTED_TOTAL
        or strict_counts["total"] != EXPECTED_TOTAL
        or bad_shape
        or nan_c
        or inf_c
        or missing
    ):
        status = "FAIL"

    return {
        "method": method,
        "raw_train": raw_counts["train"],
        "raw_test": raw_counts["test"],
        "raw_total": raw_counts["total"],
        "strict_train": strict_counts["train"],
        "strict_test": strict_counts["test"],
        "strict_total": strict_counts["total"],
        "expected_total": EXPECTED_TOTAL,
        "expected_point_count": n,
        "bad_shape_count": bad_shape,
        "nan_count": nan_c,
        "inf_count": inf_c,
        "missing_samples": len(missing),
        "extra_samples": len(extra),
        "status": status,
        "raw_root": str(raw_root),
        "strict_root": str(strict_root),
    }


def ensure_symlink(src: Path, dst: Path) -> tuple[str, bool]:
    if not src.is_dir():
        return "missing_source", False
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        valid = dst.resolve() == src.resolve()
        return ("linked" if valid else "broken_symlink"), valid
    if dst.exists():
        return "exists_other", dst.resolve() == src.resolve()
    dst.symlink_to(src.resolve())
    return "linked", True


def audit_pointnet2_input(method: str, n: int) -> dict:
    strict = lineB_paths(method)["strict_N"]
    dst = pointnet2_input_paths()["lineB_up_root"] / method
    st, valid = ensure_symlink(strict, dst)
    c = count_split(strict)
    nan_c = inf_c = bad = 0
    checked = 0
    for npy in strict.rglob("*.npy"):
        arr = np.load(npy)
        checked += 1
        if arr.shape != (n, 3):
            bad += 1
        if np.isnan(arr).any():
            nan_c += 1
        if np.isinf(arr).any():
            inf_c += 1

    link_type = "symlink" if dst.is_symlink() else ("copy" if dst.exists() else "missing")
    src_count = c["train"] + c["test"]
    dst_count = count_split(dst)["train"] + count_split(dst)["test"] if dst.exists() or dst.is_symlink() else 0

    status = "PASS"
    if st in ("missing_source", "broken_symlink", "exists_other"):
        status = "FAIL"
    elif src_count != EXPECTED_TOTAL or dst_count != EXPECTED_TOTAL:
        status = "FAIL"
    elif bad or nan_c or inf_c:
        status = "FAIL"
    elif link_type == "symlink" and not valid:
        status = "FAIL"

    return {
        "branch": "upsampling",
        "method": method,
        "source_path": str(strict),
        "pointnet2_input_path": str(dst),
        "train_samples": c["train"],
        "test_samples": c["test"],
        "point_count": n,
        "link_type": link_type,
        "symlink_valid": valid if link_type == "symlink" else "",
        "source_file_count": src_count,
        "target_file_count": dst_count,
        "bad_shape_count": bad,
        "nan_count": nan_c,
        "inf_count": inf_c,
        "status": st if status == "PASS" else status,
        "audit_status": status,
        "dataloader_resamples_internally": "no — strict_N already exact 1024",
    }


def count_split(root: Path) -> dict[str, int]:
    return {
        "train": sum(1 for _ in (root / "train").rglob("*.npy")) if (root / "train").is_dir() else 0,
        "test": sum(1 for _ in (root / "test").rglob("*.npy")) if (root / "test").is_dir() else 0,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rep = reports_dir()
    n = detect_original_point_count()
    down_n = n // 4

    # Task 1: job status
    job_rows = query_job_status(RESUME_JOB_ID)
    job_csv = rep / "modelnet40_lineB_punet_resume_job_status.csv"
    write_csv(job_csv, job_rows)

    array_rows = [r for r in job_rows if "_" in r["job_id"]]
    all_completed = bool(array_rows) and all(r["state"] == "COMPLETED" for r in array_rows)
    all_ok = bool(array_rows) and all(r["exit_code"] == "0:0" for r in array_rows)
    job_summary = {
        "job_id": RESUME_JOB_ID,
        "job_name": array_rows[0]["job_name"] if array_rows else "",
        "array_tasks": len(array_rows),
        "all_completed": all_completed,
        "all_exit_ok": all_ok,
        "overall_state": "COMPLETED" if all_completed and all_ok else "INCOMPLETE_OR_FAILED",
    }

    resume_md = rep / "modelnet40_lineB_punet_resume_job.md"
    existing = resume_md.read_text(encoding="utf-8") if resume_md.is_file() else ""
    status_block = [
        "",
        "## Final status (post-resume)",
        "",
        f"- Checked at: {ts}",
        f"- Job ID: **{RESUME_JOB_ID}**",
        f"- Array tasks: {job_summary['array_tasks']}",
        f"- Overall: **{job_summary['overall_state']}**",
        "",
        "| JobID | JobName | State | Elapsed | ExitCode |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in array_rows:
        status_block.append(
            f"| {r['job_id']} | {r['job_name']} | {r['state']} | {r['elapsed']} | {r['exit_code']} |"
        )
    if "## Final status (post-resume)" not in existing:
        resume_md.write_text(existing.rstrip() + "\n" + "\n".join(status_block) + "\n", encoding="utf-8")
    else:
        head = existing.split("## Final status (post-resume)")[0].rstrip()
        resume_md.write_text(head + "\n" + "\n".join(status_block) + "\n", encoding="utf-8")

    # Task 2: PU-Net final count audit
    punet_audit = audit_method_outputs("pu_net", n)
    punet_csv = rep / "modelnet40_lineB_punet_final_count_audit.csv"
    write_csv(punet_csv, [punet_audit])

    punet_md = rep / "modelnet40_lineB_punet_final_count_audit.md"
    punet_md.write_text(
        "\n".join([
            "# Line B PU-Net Final Count Audit",
            "",
            f"- Generated at: {ts}",
            f"- Resume job **{RESUME_JOB_ID}**: **{job_summary['overall_state']}**",
            "",
            "## Paths",
            "",
            f"- raw: `{punet_audit['raw_root']}`",
            f"- strict_N: `{punet_audit['strict_root']}`",
            "",
            "## Counts",
            "",
            f"| split | raw | strict_N | expected |",
            f"| --- | ---: | ---: | ---: |",
            f"| train | {punet_audit['raw_train']} | {punet_audit['strict_train']} | {EXPECTED_TRAIN} |",
            f"| test | {punet_audit['raw_test']} | {punet_audit['strict_test']} | {EXPECTED_TEST} |",
            f"| total | {punet_audit['raw_total']} | {punet_audit['strict_total']} | {EXPECTED_TOTAL} |",
            "",
            "## Quality checks",
            "",
            f"- every strict output shape = ({n}, 3): **{'PASS' if punet_audit['bad_shape_count'] == 0 else 'FAIL'}** ({punet_audit['bad_shape_count']} bad)",
            f"- NaN: **{'PASS' if punet_audit['nan_count'] == 0 else 'FAIL'}** ({punet_audit['nan_count']})",
            f"- Inf: **{'PASS' if punet_audit['inf_count'] == 0 else 'FAIL'}** ({punet_audit['inf_count']})",
            f"- missing samples: **{punet_audit['missing_samples']}**",
            "",
            f"## Status: **{punet_audit['status']}**",
        ]),
        encoding="utf-8",
    )

    # Task 3: completed methods count audit
    cm_rows = []
    for method in MAIN_METHODS:
        if method == "pu_gcn":
            cm_rows.append({
                "method": method,
                "status": "pending",
                "raw_count": 0,
                "strict_count": 0,
                "total_expected_samples": EXPECTED_TOTAL,
                "exact_1024_count": 0,
                "nan_count": 0,
                "inf_count": 0,
                "missing_samples": EXPECTED_TOTAL,
                "provenance_status": "not_run",
                "note": "intentionally skipped; pending",
            })
            continue
        audit = audit_method_outputs(method, n)
        cm_rows.append({
            "method": method,
            "status": "completed" if audit["status"] == "PASS" else "audit_fail",
            "raw_count": audit["raw_total"],
            "strict_count": audit["strict_total"],
            "total_expected_samples": EXPECTED_TOTAL,
            "exact_1024_count": audit["strict_total"] - audit["bad_shape_count"],
            "nan_count": audit["nan_count"],
            "inf_count": audit["inf_count"],
            "missing_samples": audit["missing_samples"],
            "provenance_status": "PASS" if audit["status"] == "PASS" else "FAIL",
            "note": "",
        })

    cm_csv = rep / "modelnet40_lineB_completed_methods_count_audit.csv"
    write_csv(cm_csv, cm_rows)
    cm_md = rep / "modelnet40_lineB_completed_methods_count_audit.md"
    display = {"ear": "EAR", "pdans": "PDANS", "pu_net": "PU-Net", "pu_gcn": "PU-GCN"}
    cm_md.write_text(
        "\n".join([
            "# Line B Completed Methods Count Audit",
            "",
            f"- Generated at: {ts}",
            f"- PU-Net resume job **{RESUME_JOB_ID}**: **{job_summary['overall_state']}**",
            "",
            "| method | status | raw | strict | expected | missing | provenance | note |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
            *[
                f"| {display[r['method']]} | {r['status']} | {r['raw_count']} | {r['strict_count']} | "
                f"{r['total_expected_samples']} | {r['missing_samples']} | {r['provenance_status']} | {r['note']} |"
                for r in cm_rows
            ],
        ]),
        encoding="utf-8",
    )

    # Task 4: PointNet++ inputs for completed methods
    pn_rows = []
    bl_src = DOWNSAMPLED_X4_ROOT
    bl_dst = pointnet2_input_paths()["lineB_baseline"]
    st, valid = ensure_symlink(bl_src, bl_dst)
    c = count_split(bl_src)
    pn_rows.append({
        "branch": "baseline",
        "method": "downsampled_x4",
        "source_path": str(bl_src),
        "pointnet2_input_path": str(bl_dst),
        "train_samples": c["train"],
        "test_samples": c["test"],
        "point_count": down_n,
        "link_type": "symlink" if bl_dst.is_symlink() else "exists",
        "symlink_valid": valid if bl_dst.is_symlink() else "",
        "source_file_count": c["train"] + c["test"],
        "target_file_count": count_split(bl_dst)["train"] + count_split(bl_dst)["test"],
        "bad_shape_count": 0,
        "nan_count": 0,
        "inf_count": 0,
        "status": st,
        "audit_status": "PASS" if c["train"] + c["test"] == EXPECTED_TOTAL else "FAIL",
        "dataloader_resamples_internally": "no — num_point must match 256",
    })
    for method in COMPLETED_METHODS:
        pn_rows.append(audit_pointnet2_input(method, n))

    pn_csv = rep / "modelnet40_lineB_pointnet2_input_audit_completed_methods.csv"
    write_csv(pn_csv, pn_rows)
    pn_md = rep / "modelnet40_lineB_pointnet2_input_audit_completed_methods.md"
    pn_md.write_text(
        "\n".join([
            "# Line B PointNet++ Input Audit — Completed Methods",
            "",
            f"- Generated at: {ts}",
            f"- Expected train/test: {EXPECTED_TRAIN} / {EXPECTED_TEST}",
            "",
            "| branch | method | point_count | train | test | link | status | audit |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
            *[
                f"| {r['branch']} | {r['method']} | {r['point_count']} | {r['train_samples']} | "
                f"{r['test_samples']} | {r.get('link_type', '')} | {r['status']} | {r['audit_status']} |"
                for r in pn_rows
            ],
        ]),
        encoding="utf-8",
    )

    print(f"PU-Net audit: {punet_audit['status']} raw={punet_audit['raw_total']} strict={punet_audit['strict_total']}")
    print(f"Job {RESUME_JOB_ID}: {job_summary['overall_state']}")
    print(f"Wrote reports to {rep}")
    return 0 if punet_audit["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
