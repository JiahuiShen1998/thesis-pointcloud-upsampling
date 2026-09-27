#!/usr/bin/env python3
"""Audit PointNet++ Line A smoke jobs and write CSV/MD reports."""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS = PROJECT_ROOT / "reports"

BRANCHES = [
    {
        "branch": "lineA_original_baseline_1024",
        "method": "baseline",
        "expected_point_count": 1024,
        "job_id": "1733475",
        "log_dir": PROJECT_ROOT / "logs/pointnet2_smoke/lineA_original_baseline_1024",
        "metrics_json": PROJECT_ROOT
        / "pointnet2_results/x4_two_line_smoke/lineA_original_baseline/metrics.json",
    },
    {
        "branch": "lineA_ear_4096",
        "method": "EAR",
        "expected_point_count": 4096,
        "job_id": "1733476",
        "log_dir": PROJECT_ROOT / "logs/pointnet2_smoke/lineA_ear_4096",
        "metrics_json": PROJECT_ROOT
        / "pointnet2_results/x4_two_line_smoke/lineA_original_up/ear/metrics.json",
    },
    {
        "branch": "lineA_pdans_4096",
        "method": "PDANS",
        "expected_point_count": 4096,
        "job_id": "1733477",
        "log_dir": PROJECT_ROOT / "logs/pointnet2_smoke/lineA_pdans_4096",
        "metrics_json": PROJECT_ROOT
        / "pointnet2_results/x4_two_line_smoke/lineA_original_up/pdans/metrics.json",
    },
    {
        "branch": "lineA_punet_4096",
        "method": "PU-Net",
        "expected_point_count": 4096,
        "job_id": "1733478",
        "log_dir": PROJECT_ROOT / "logs/pointnet2_smoke/lineA_punet_4096",
        "metrics_json": PROJECT_ROOT
        / "pointnet2_results/x4_two_line_smoke/lineA_original_up/pu_net/metrics.json",
    },
    {
        "branch": "lineA_pugcn_4096",
        "method": "PU-GCN",
        "expected_point_count": 4096,
        "job_id": "1733479",
        "log_dir": PROJECT_ROOT / "logs/pointnet2_smoke/lineA_pugcn_4096",
        "metrics_json": PROJECT_ROOT
        / "pointnet2_results/x4_two_line_smoke/lineA_original_up/pu_gcn/metrics.json",
    },
]

FIELDS = [
    "line",
    "branch",
    "method",
    "expected_point_count",
    "actual_batch_shape",
    "actual_point_count",
    "first_loss",
    "first_loss_finite",
    "forward_status",
    "backward_status",
    "eval_loop_status",
    "metrics_json_exists",
    "metrics_json_path",
    "job_id",
    "job_state",
    "exit_code",
    "log_path",
    "status",
    "note",
]


def sacct_state(job_id: str) -> tuple[str, str]:
    cmd = [
        "sacct",
        "-j",
        job_id,
        "--format=State,ExitCode",
        "-n",
        "-P",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN", ""
    for line in out.splitlines():
        if line.startswith(f"{job_id}|"):
            parts = line.split("|")
            if len(parts) >= 3:
                return parts[1], parts[2]
    first = out.splitlines()[0] if out else ""
    parts = first.split("|")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return "UNKNOWN", ""


def find_log(log_dir: Path, job_id: str) -> Path | None:
    for pattern in (f"slurm_{job_id}.out", f"slurm_{job_id}.err"):
        p = log_dir / pattern
        if p.exists():
            return p
    outs = sorted(log_dir.glob("slurm_*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
    return outs[0] if outs else None


def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_log(text: str, err_text: str) -> dict:
    combined = text + "\n" + err_text
    batch_shape = ""
    point_count = ""
    first_loss = ""
    first_loss_finite = "false"

    m = re.search(
        r"First train batch tensor shape before device move: \(([^)]+)\)",
        text,
    )
    if m:
        batch_shape = f"({m.group(1)})"
        nums = [int(x.strip()) for x in m.group(1).split(",")]
        if len(nums) >= 3:
            point_count = str(nums[2])

    m = re.search(r"First train batch loss: ([0-9.eE+-]+)", text)
    if m:
        first_loss = m.group(1)
        try:
            val = float(first_loss)
            first_loss_finite = "true" if math.isfinite(val) else "false"
        except ValueError:
            first_loss_finite = "false"

    forward_status = "PASS" if "First train batch loss:" in text else "FAIL"
    backward_status = (
        "PASS"
        if re.search(r"Epoch \d+/\d+ train_acc=", text)
        or "Training finished" in text
        else "FAIL"
    )
    eval_status = (
        "PASS"
        if re.search(r"test_overall=", text) or "Training finished" in text
        else "FAIL"
    )

    error_patterns = [
        r"Traceback",
        r"FileNotFoundError",
        r"RuntimeError",
        r"CUDA out of memory",
        r"nan",
        r"inf",
        r"dataloader",
    ]
    errors = []
    for pat in error_patterns:
        if re.search(pat, combined, re.IGNORECASE):
            errors.append(pat)

    note_parts = []
    if "smoke_status=PASS" in text:
        note_parts.append("smoke_status=PASS in slurm out")
    if "Training finished" in text:
        note_parts.append("training loop completed")
    if errors:
        note_parts.append(f"errors detected: {', '.join(errors)}")

    return {
        "actual_batch_shape": batch_shape,
        "actual_point_count": point_count,
        "first_loss": first_loss,
        "first_loss_finite": first_loss_finite,
        "forward_status": forward_status,
        "backward_status": backward_status,
        "eval_loop_status": eval_status,
        "error_hits": errors,
        "note_parts": note_parts,
    }


def audit_branch(cfg: dict) -> dict:
    job_id = cfg["job_id"]
    job_state, exit_code = sacct_state(job_id)
    log_out = find_log(cfg["log_dir"], job_id)
    log_err = cfg["log_dir"] / f"slurm_{job_id}.err"
    if not log_err.exists() and log_out:
        alt = log_out.with_suffix(".err")
        log_err = alt if alt.exists() else log_err

    out_text = read_text(log_out)
    err_text = read_text(log_err if log_err.exists() else None)
    parsed = parse_log(out_text, err_text)

    metrics_path = cfg["metrics_json"]
    metrics_exists = metrics_path.exists()

    expected = cfg["expected_point_count"]
    actual_pts = parsed["actual_point_count"]
    point_match = actual_pts == str(expected)

    no_errors = not parsed["error_hits"] or (
        job_state == "COMPLETED"
        and exit_code == "0:0"
        and "Traceback" not in (out_text + err_text)
    )

    pass_cond = (
        job_state == "COMPLETED"
        and exit_code == "0:0"
        and point_match
        and parsed["first_loss_finite"] == "true"
        and parsed["forward_status"] == "PASS"
        and parsed["backward_status"] == "PASS"
        and parsed["eval_loop_status"] == "PASS"
        and metrics_exists
        and no_errors
    )

    status = "PASS" if pass_cond else "FAIL"
    note = "; ".join(parsed["note_parts"]) if parsed["note_parts"] else ""
    if status == "FAIL" and err_text.strip():
        err_line = ""
        for line in err_text.splitlines():
            if "Error" in line or "Traceback" in line or "Exception" in line:
                err_line = line.strip()
                break
        if not err_line and "Traceback" in err_text:
            err_line = "see slurm err for Traceback"
        if err_line:
            note = (note + "; " if note else "") + err_line

    return {
        "line": "A",
        "branch": cfg["branch"],
        "method": cfg["method"],
        "expected_point_count": expected,
        "actual_batch_shape": parsed["actual_batch_shape"],
        "actual_point_count": actual_pts,
        "first_loss": parsed["first_loss"],
        "first_loss_finite": parsed["first_loss_finite"],
        "forward_status": parsed["forward_status"],
        "backward_status": parsed["backward_status"],
        "eval_loop_status": parsed["eval_loop_status"],
        "metrics_json_exists": str(metrics_exists).lower(),
        "metrics_json_path": str(metrics_path) if metrics_exists else "",
        "job_id": job_id,
        "job_state": job_state,
        "exit_code": exit_code,
        "log_path": str(log_out) if log_out else "",
        "status": status,
        "note": note,
    }


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows: list[dict], path: Path) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pass_count = sum(1 for r in rows if r["status"] == "PASS")
    all_pass = pass_count == len(rows)
    lines = [
        "# ModelNet40 PointNet++ Line A Smoke Audit",
        "",
        f"- Generated at: {now}",
        f"- Submit time: 2026-07-07 20:10:18 UTC",
        "- Submit command: `sbatch.tinygpu`",
        f"- Job IDs: {', '.join(r['job_id'] for r in rows)}",
        f"- Overall result: **{pass_count} / {len(rows)} PASS**"
        + (" — full training **ready**" if all_pass else " — full training **blocked**"),
        "",
        "## Summary",
        "",
        "| branch | method | expected pts | actual pts | job id | state | exit | status |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for r in rows:
        actual = r["actual_point_count"] or "—"
        lines.append(
            f"| {r['branch']} | {r['method']} | {r['expected_point_count']} | {actual} "
            f"| {r['job_id']} | {r['job_state']} | {r['exit_code']} | **{r['status']}** |"
        )

    for r in rows:
        lines.extend(
            [
                "",
                f"## {r['status']} branch detail: {r['branch']}",
                "",
                f"- Log: `{r['log_path']}`",
                f"- Expected point count: **{r['expected_point_count']}**",
                f"- Actual batch shape: `{r['actual_batch_shape'] or '—'}`",
                f"- First loss: **{r['first_loss'] or '—'}** (finite={r['first_loss_finite']})",
                f"- Forward / backward / eval: {r['forward_status']} / {r['backward_status']} / {r['eval_loop_status']}",
                f"- metrics.json exists: {r['metrics_json_exists']}",
            ]
        )
        if r["metrics_json_path"]:
            lines.append(f"- metrics.json: `{r['metrics_json_path']}`")
        if r["note"]:
            lines.append(f"- Note: {r['note']}")

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"- **5 / 5 Line A smoke PASS:** {'Yes' if all_pass else 'No'}",
            f"- **Line A full training ready:** {'Yes' if all_pass else 'No'}",
        ]
    )
    if not all_pass:
        lines.append("- See: `reports/modelnet40_pointnet2_lineA_smoke_failure_diagnosis.md`")
    else:
        lines.append(
            "- See: `reports/modelnet40_pointnet2_lineA_full_training_ready_commands.md`"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [audit_branch(cfg) for cfg in BRANCHES]
    csv_path = REPORTS / "modelnet40_pointnet2_lineA_smoke_audit.csv"
    md_path = REPORTS / "modelnet40_pointnet2_lineA_smoke_audit.md"
    write_csv(rows, csv_path)
    write_md(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    pass_count = sum(1 for r in rows if r["status"] == "PASS")
    print(f"PASS: {pass_count}/{len(rows)}")


if __name__ == "__main__":
    main()
