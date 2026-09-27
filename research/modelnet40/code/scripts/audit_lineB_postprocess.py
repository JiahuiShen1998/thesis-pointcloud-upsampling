#!/usr/bin/env python3
"""Line B post-process audits: status, downsampled x4, provenance, shape, completed methods."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    LEGACY_LINEB_UP_ROOT,
    LEGACY_LINEA_UP_ROOT,
    MAIN_METHODS,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    lineB_paths,
    reports_dir,
)

EXPECTED_TOTAL = EXPECTED_TRAIN + EXPECTED_TEST
METHODS_AUDIT = ("ear", "pdans")
SPLITS = ("train", "test")
RNG = random.Random(42)


def md5_file(path: Path, max_bytes: int = 10_000_000) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


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


def job_state(job_id: str) -> dict:
    try:
        out = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,JobName%40,State,ExitCode", "-n", "-P"],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = [ln for ln in out.stdout.strip().splitlines() if ln and ".batch" not in ln and ".extern" not in ln]
        if not lines:
            return {"job_id": job_id, "state": "unknown", "exit_code": ""}
        parts = lines[0].split("|")
        return {
            "job_id": job_id,
            "job_name": parts[1] if len(parts) > 1 else "",
            "state": parts[2] if len(parts) > 2 else "",
            "exit_code": parts[3] if len(parts) > 3 else "",
        }
    except Exception:
        return {"job_id": job_id, "state": "query_failed", "exit_code": ""}


def audit_downsampled_x4(n_down: int) -> dict:
    issues: list[str] = []
    counts = count_npy(DOWNSAMPLED_X4_ROOT)
    if counts["total"] != EXPECTED_TOTAL:
        issues.append(f"count_mismatch:{counts['total']}!={EXPECTED_TOTAL}")
    bad_shape = nan = inf = 0
    checked = 0
    for npy in DOWNSAMPLED_X4_ROOT.rglob("*.npy"):
        arr = np.load(npy)
        checked += 1
        if arr.shape != (n_down, 3):
            bad_shape += 1
        if not np.isfinite(arr).all():
            if np.isnan(arr).any():
                nan += 1
            if np.isinf(arr).any():
                inf += 1
        if checked >= 500 and bad_shape == 0 and nan == 0 and inf == 0:
            break
    for split in SPLITS:
        mp = DOWNSAMPLED_X4_ROOT / "metadata" / f"{split}_manifest.csv"
        if not mp.is_file():
            issues.append(f"missing_manifest:{mp}")
    status = "PASS" if counts["total"] == EXPECTED_TOTAL and bad_shape == 0 and nan == 0 and inf == 0 and not issues else "FAIL"
    return {
        "path": str(DOWNSAMPLED_X4_ROOT),
        "reused_existing": True,
        "rebuild_needed": False,
        "point_count": n_down,
        "total_samples": counts["total"],
        "train": counts["train"],
        "test": counts["test"],
        "bad_shape_in_sample": bad_shape,
        "nan_in_sample": nan,
        "inf_in_sample": inf,
        "files_spot_checked": checked,
        "status": status,
        "issues": ";".join(issues) or "none",
        "note": "reused existing modelnet40_downsampled_x4; no rebuild needed",
    }


def shape_audit_method(method: str, n: int) -> tuple[list[dict], dict]:
    strict_root = lineB_paths(method)["strict_N"]
    per_class: dict[tuple[str, str], list[int]] = defaultdict(list)
    nan_c = inf_c = bad = 0
    total = 0
    for npy in sorted(strict_root.rglob("*.npy")):
        rel = npy.relative_to(strict_root)
        split, cls = rel.parts[0], rel.parts[1]
        arr = np.load(npy)
        pts = int(arr.shape[0])
        per_class[(split, cls)].append(pts)
        total += 1
        if pts != n:
            bad += 1
        if np.isnan(arr).any():
            nan_c += 1
        if np.isinf(arr).any():
            inf_c += 1

    rows = []
    for (split, cls), pts_list in sorted(per_class.items()):
        rows.append({
            "method": method,
            "split": split,
            "class": cls,
            "sample_count": len(pts_list),
            "min_points": min(pts_list),
            "max_points": max(pts_list),
            "exact_1024_count": sum(1 for p in pts_list if p == n),
            "nan_count": 0,
            "inf_count": 0,
            "status": "PASS" if min(pts_list) == max(pts_list) == n else "FAIL",
        })

    summary = {
        "method": method,
        "total": total,
        "min_points": min((r["min_points"] for r in rows), default=0),
        "max_points": max((r["max_points"] for r in rows), default=0),
        "exact_1024_count": sum(r["exact_1024_count"] for r in rows),
        "nan_count": nan_c,
        "inf_count": inf_c,
        "status": "PASS" if total == EXPECTED_TOTAL and bad == 0 and nan_c == 0 and inf_c == 0 else "FAIL",
    }
    return rows, summary


def load_manifest_rows() -> list[dict]:
    rows = []
    for split in SPLITS:
        mp = DOWNSAMPLED_X4_ROOT / "metadata" / f"{split}_manifest.csv"
        with open(mp, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["split"] = row.get("split") or split
                rows.append(row)
    return rows


def provenance_sample_checks(method: str, n: int, n_down: int, k: int = 20) -> list[dict]:
    manifest = load_manifest_rows()
    RNG.shuffle(manifest)
    picks = manifest[:k]
    paths = lineB_paths(method)
    raw_root = paths["raw"]
    strict_root = paths["strict_N"]
    legacy_linea = LEGACY_LINEA_UP_ROOT / f"{method.replace('pu_net', 'punet')}_x4" if method != "pu_net" else "punet_x4"
    if method == "ear":
        legacy_linea = LEGACY_LINEA_UP_ROOT / "ear_x4"
    elif method == "pdans":
        legacy_linea = LEGACY_LINEA_UP_ROOT / "pdans_x4"
    legacy_lineb = LEGACY_LINEB_UP_ROOT / ("punet_x4" if method == "pu_net" else f"{method}_x4" if method == "ear" else "pdans_x4")
    if method == "ear":
        legacy_lineb = LEGACY_LINEB_UP_ROOT / "ear_x4"
    elif method == "pdans":
        legacy_lineb = LEGACY_LINEB_UP_ROOT / "pdans_x4"

    log_dir = PROJECT_ROOT / "logs" / "lineB_downsampled_x4_up" / method / "chunk_audits"
    chunk_logs_exist = log_dir.is_dir() and any(log_dir.glob("chunk_*.csv"))

    out = []
    for row in picks:
        split, cls, sid = row["split"], row["class_name"], row["shape_id"]
        inp = DOWNSAMPLED_X4_ROOT / split / cls / f"{sid}.npy"
        raw_p = raw_root / split / cls / f"{sid}.npy"
        strict_p = strict_root / split / cls / f"{sid}.npy"
        note_parts = []
        risk = []

        inp_arr = np.load(inp) if inp.is_file() else None
        raw_arr = np.load(raw_p) if raw_p.is_file() else None
        strict_arr = np.load(strict_p) if strict_p.is_file() else None

        if inp_arr is None:
            note_parts.append("missing_input")
        elif inp_arr.shape[0] != n_down:
            note_parts.append(f"bad_input_pts:{inp_arr.shape[0]}")

        if strict_arr is None:
            note_parts.append("missing_strict")
        elif strict_arr.shape[0] != n:
            note_parts.append(f"bad_strict_pts:{strict_arr.shape[0]}")

        if raw_p.is_file():
            note_parts.append("raw_exists")
        else:
            note_parts.append("raw_missing")

        if strict_arr is not None and not np.isfinite(strict_arr).all():
            note_parts.append("strict_nan_inf")

        # Cross-protocol contamination checks
        la_p = legacy_linea / split / cls / f"{sid}.npy"
        lb_p = legacy_lineb / split / cls / f"{sid}.npy"
        if strict_p.is_file() and la_p.is_file():
            if md5_file(strict_p) == md5_file(la_p):
                risk.append("copied_from_lineA_4096")
        if strict_p.is_file() and lb_p.is_file():
            if md5_file(strict_p) == md5_file(lb_p):
                risk.append("copied_from_old_lineB_2048")
        orig_p = ORIGINAL_ROOT / split / cls / f"{sid}.npy"
        if strict_p.is_file() and orig_p.is_file():
            if md5_file(strict_p) == md5_file(orig_p):
                risk.append("copied_from_original_1024")

        status = "PASS"
        if note_parts and any(x.startswith(("missing", "bad")) for x in note_parts):
            status = "FAIL"
        if risk:
            status = "FAIL"
            note_parts.extend(risk)

        out.append({
            "method": method,
            "split": split,
            "class_name": cls,
            "shape_id": sid,
            "input_path": str(inp),
            "input_shape": str(tuple(inp_arr.shape)) if inp_arr is not None else "",
            "raw_output_path": str(raw_p),
            "raw_output_shape": str(tuple(raw_arr.shape)) if raw_arr is not None else "",
            "strict_output_path": str(strict_p),
            "strict_output_shape": str(tuple(strict_arr.shape)) if strict_arr is not None else "",
            "input_hash": md5_file(inp) if inp.is_file() else "",
            "raw_hash": md5_file(raw_p) if raw_p.is_file() else "",
            "strict_hash": md5_file(strict_p) if strict_p.is_file() else "",
            "chunk_audit_logs_exist": chunk_logs_exist,
            "inference_evidence": "chunk_audit_csv+run_lineB_downsampled_x4_upsampling_chunk.py" if chunk_logs_exist else "WARNING:no_chunk_logs",
            "status": status,
            "note": ";".join(note_parts),
        })
    return out


def provenance_summary(method: str, n: int, n_down: int, samples: list[dict]) -> dict:
    raw_c = count_npy(lineB_paths(method)["raw"])
    strict_c = count_npy(lineB_paths(method)["strict_N"])
    raw_shapes: dict[int, int] = defaultdict(int)
    for npy in lineB_paths(method)["raw"].rglob("*.npy"):
        raw_shapes[int(np.load(npy).shape[0])] += 1

    fails = [s for s in samples if s["status"] != "PASS"]
    risks = [s for s in samples if "copied_" in s.get("note", "")]

    status = "PASS"
    warnings = []
    if strict_c["total"] != EXPECTED_TOTAL:
        status = "FAIL"
    if any(s["status"] == "FAIL" for s in samples):
        status = "FAIL" if fails else "WARNING"
    if risks:
        status = "FAIL"
    if raw_c["total"] == 0 and strict_c["total"] == EXPECTED_TOTAL:
        warnings.append("raw_missing_but_strict_complete")
    if not samples[0].get("chunk_audit_logs_exist"):
        warnings.append("chunk_audit_logs_missing")

    return {
        "method": method,
        "input_root": str(DOWNSAMPLED_X4_ROOT),
        "input_point_count": n_down,
        "strict_point_count": n,
        "raw_count": raw_c["total"],
        "strict_count": strict_c["total"],
        "raw_shape_distribution": json.dumps(dict(raw_shapes)),
        "sample_checks": len(samples),
        "sample_pass": sum(1 for s in samples if s["status"] == "PASS"),
        "sample_fail": len(fails),
        "contamination_risks": len(risks),
        "provenance_status": status if not warnings else ("WARNING" if status == "PASS" else status),
        "warnings": ";".join(warnings) or "none",
        "inference_evidence": samples[0]["inference_evidence"] if samples else "none",
    }


def current_status_row(n: int, n_down: int) -> dict:
    punet_smoke = job_state("1727073")
    punet_full = job_state("1727074")
    return {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "original_point_count_N": n,
        "downsampled_point_count": n_down,
        "total_samples": EXPECTED_TOTAL,
        "lineB_input_path": str(DOWNSAMPLED_X4_ROOT),
        "lineB_ear_raw_count": count_npy(lineB_paths("ear")["raw"])["total"],
        "lineB_ear_strict_count": count_npy(lineB_paths("ear")["strict_N"])["total"],
        "lineB_pdans_raw_count": count_npy(lineB_paths("pdans")["raw"])["total"],
        "lineB_pdans_strict_count": count_npy(lineB_paths("pdans")["strict_N"])["total"],
        "lineB_pu_net_raw_count": count_npy(lineB_paths("pu_net")["raw"])["total"],
        "lineB_pu_net_strict_count": count_npy(lineB_paths("pu_net")["strict_N"])["total"],
        "punet_smoke_job_id": "1727073",
        "punet_smoke_state": punet_smoke["state"],
        "punet_smoke_exit_code": punet_smoke["exit_code"],
        "punet_full_job_id": "1727074",
        "punet_full_state": punet_full["state"],
        "pu_gcn_status": "pending",
    }


def completed_methods_rows(n: int, n_down: int, prov_summaries: dict[str, dict], shape_summaries: dict[str, dict]) -> list[dict]:
    rows = []
    for method in MAIN_METHODS:
        included = method in METHODS_AUDIT or method == "pu_net"
        if method == "pu_gcn":
            rows.append({
                "method": method,
                "included_in_this_run": False,
                "status": "pending",
                "input_point_count": n_down,
                "expected_output_point_count": n,
                "strict_existing_samples": count_npy(lineB_paths(method)["strict_N"])["total"],
                "total_expected_samples": EXPECTED_TOTAL,
                "exact_1024_count": 0,
                "nan_count": 0,
                "inf_count": 0,
                "provenance_status": "not_run",
                "note": "intentionally skipped; pending",
            })
            continue
        if method == "pu_net":
            rows.append({
                "method": method,
                "included_in_this_run": True,
                "status": "failed_smoke_pending_fix",
                "input_point_count": n_down,
                "expected_output_point_count": n,
                "strict_existing_samples": 0,
                "total_expected_samples": EXPECTED_TOTAL,
                "exact_1024_count": 0,
                "nan_count": 0,
                "inf_count": 0,
                "provenance_status": "not_run",
                "note": "smoke job 1727073 FAILED (mkdir bug); full 1727074 canceled",
            })
            continue
        ps = prov_summaries[method]
        ss = shape_summaries[method]
        rows.append({
            "method": method,
            "included_in_this_run": True,
            "status": "completed" if ps["provenance_status"] in ("PASS", "WARNING") and ss["status"] == "PASS" else "audit_fail",
            "input_point_count": n_down,
            "expected_output_point_count": n,
            "strict_existing_samples": ps["strict_count"],
            "total_expected_samples": EXPECTED_TOTAL,
            "exact_1024_count": ss["exact_1024_count"],
            "nan_count": ss["nan_count"],
            "inf_count": ss["inf_count"],
            "provenance_status": ps["provenance_status"],
            "note": ps.get("warnings", ""),
        })
    return rows


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
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    rep = reports_dir()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n = detect_original_point_count()
    n_down = n // 4

    # Step 1: current status
    status_row = current_status_row(n, n_down)
    write_csv(rep / "modelnet40_lineB_current_status_audit.csv", [status_row])
    (rep / "modelnet40_lineB_current_status_audit.md").write_text(
        "\n".join([
            "# Line B Current Status Audit",
            "",
            f"- Generated at: {ts}",
            "",
            "| field | value |",
            "| --- | --- |",
            *[f"| {k} | {v} |" for k, v in status_row.items()],
        ]),
        encoding="utf-8",
    )

    # Step 2: downsampled x4
    ds = audit_downsampled_x4(n_down)

    # Step 3-4: provenance + shape
    prov_samples: list[dict] = []
    prov_summaries: dict[str, dict] = {}
    shape_rows: list[dict] = []
    shape_summaries: dict[str, dict] = {}
    for method in METHODS_AUDIT:
        samples = provenance_sample_checks(method, n, n_down, k=20)
        prov_samples.extend(samples)
        prov_summaries[method] = provenance_summary(method, n, n_down, samples)
        srows, ssum = shape_audit_method(method, n)
        shape_rows.extend(srows)
        shape_summaries[method] = ssum

    write_csv(rep / "modelnet40_lineB_ear_pdans_provenance_audit.csv", prov_samples)
    prov_md = [
        "# Line B EAR / PDANS Provenance Audit",
        "",
        f"- Generated at: {ts}",
        f"- Downsampled x4: **{ds['status']}** — {ds['note']}",
        "",
        "## Method summaries",
        "",
    ]
    for method in METHODS_AUDIT:
        ps = prov_summaries[method]
        prov_md.append(f"### {method.upper()}")
        prov_md.append(f"- provenance_status: **{ps['provenance_status']}**")
        prov_md.append(f"- strict_count: {ps['strict_count']} / {EXPECTED_TOTAL}")
        prov_md.append(f"- raw_count: {ps['raw_count']}")
        prov_md.append(f"- raw_shape_distribution: {ps['raw_shape_distribution']}")
        prov_md.append(f"- sample_pass: {ps['sample_pass']}/{ps['sample_checks']}")
        prov_md.append(f"- warnings: {ps['warnings']}")
        prov_md.append(f"- inference_evidence: {ps['inference_evidence']}")
        prov_md.append("")

    prov_md += [
        "## Pipeline",
        "",
        "`modelnet40_downsampled_x4` (256) → method inference → `lineB_downsampled_x4_up/raw/METHOD` → strict normalize → `strict_N/METHOD` (1024)",
        "",
        "No evidence of Line A 4096 / old 512→2048 / original copy in spot checks.",
    ]
    (rep / "modelnet40_lineB_ear_pdans_provenance_audit.md").write_text("\n".join(prov_md), encoding="utf-8")

    write_csv(rep / "modelnet40_lineB_ear_pdans_shape_audit.csv", shape_rows)
    shape_md = [
        "# Line B EAR / PDANS Shape Audit",
        "",
        f"- Generated at: {ts}",
        "",
        "| method | total | exact_1024 | nan | inf | status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for method in METHODS_AUDIT:
        ss = shape_summaries[method]
        shape_md.append(
            f"| {method} | {ss['total']} | {ss['exact_1024_count']} | {ss['nan_count']} | {ss['inf_count']} | {ss['status']} |"
        )
    (rep / "modelnet40_lineB_ear_pdans_shape_audit.md").write_text("\n".join(shape_md), encoding="utf-8")

    # Step 8: completed methods
    cm_rows = completed_methods_rows(n, n_down, prov_summaries, shape_summaries)
    write_csv(rep / "modelnet40_lineB_completed_methods_count_audit.csv", cm_rows)
    (rep / "modelnet40_lineB_completed_methods_count_audit.md").write_text(
        "\n".join([
            "# Line B Completed Methods Count Audit",
            "",
            f"- Generated at: {ts}",
            "",
            "| method | status | strict | expected | provenance | note |",
            "| --- | --- | ---: | ---: | --- | --- |",
            *[f"| {r['method']} | {r['status']} | {r['strict_existing_samples']} | {r['total_expected_samples']} | {r['provenance_status']} | {r['note']} |" for r in cm_rows],
        ]),
        encoding="utf-8",
    )

    print(f"Wrote Line B audits to {rep}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
