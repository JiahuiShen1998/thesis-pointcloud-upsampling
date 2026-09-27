#!/usr/bin/env python3
"""Generate consolidated ×4 parallel generation monitoring report."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from upsampling_x4_common import LEGACY_EAR_X2, PROJECT_ROOT, count_npy_files, output_root

REPORT_PATH = PROJECT_ROOT / "reports" / "modelnet40_x4_parallel_generation_status.md"

METHODS = {
    "ear": {
        "label": "EAR",
        "smoke_jobs": {"A": "1716174", "B": "1716175"},
        "full_jobs": {"A": "1716174", "B": "1716175"},
        "subdir": "ear_x4",
        "smoke_subdir": "ear_x4_smoke",
    },
    "pdans": {"label": "PDANS", "subdir": "pdans_x4", "smoke_subdir": "pdans_x4_smoke"},
    "punet": {"label": "PU-Net", "subdir": "punet_x4", "smoke_subdir": "punet_x4_smoke"},
    "pugcn": {"label": "PU-GCN", "subdir": "pugcn_x4", "smoke_subdir": "pugcn_x4_smoke"},
}

LINE_SPECS = {
    "A": {
        "input_pts": 1024,
        "output_pts": 4096,
        "full_train": 9843,
        "full_test": 2468,
        "smoke_train": 20,
        "smoke_test": 20,
        "up_root": "modelnet40_original_up",
    },
    "B": {
        "input_pts": 512,
        "output_pts": 2048,
        "full_train": 9843,
        "full_test": 2468,
        "smoke_train": 20,
        "smoke_test": 20,
        "up_root": "modelnet40_downsampled50_up",
    },
}


def squeue_user() -> str:
    try:
        proc = subprocess.run(
            ["squeue", "-u", subprocess.check_output(["whoami"], text=True).strip()],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout.strip() or "(empty)"
    except Exception as exc:
        return f"(squeue failed: {exc})"


def load_job_ids(method: str) -> dict[str, str]:
    env_path = PROJECT_ROOT / "reports" / f"{method}_x4_generation" / "job_ids.env"
    data: dict[str, str] = {}
    if not env_path.is_file():
        return data
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip()] = value.strip()
    return data


def job_state(job_id: str) -> str:
  if not job_id or job_id == "—":
    return "—"
  try:
    proc = subprocess.run(
      ["sacct", "-j", job_id, "--format=JobID,State", "-n", "-P"],
      capture_output=True,
      text=True,
      check=False,
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln and not ln.endswith("_batch")]
    if not lines:
      return "unknown"
    return lines[0].split("|")[-1].strip()
  except Exception:
    return "unknown"


def read_audit_status(method: str, kind: str) -> str:
    if kind == "smoke":
        paths = list((PROJECT_ROOT / "reports").glob(f"{method}_x4_smoke_audit*.md"))
    else:
        paths = [PROJECT_ROOT / "reports" / f"{method}_x4_full_generation_audit.md"]
    paths = [p for p in paths if p.is_file()]
    if not paths:
        return "not_run"
    text = paths[0].read_text(encoding="utf-8")
    if "**PASS**" in text and "**FAIL**" not in text.split("Overall")[1][:80]:
        return "PASS"
    if "PASS" in text:
        return "PARTIAL"
    return "FAIL"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit-log", type=Path, default=None)
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# ModelNet40 ×4 Parallel Generation Status",
        "",
        f"- Updated: {ts}",
        f"- Project: `{PROJECT_ROOT.name}`",
        "",
        "## Slurm queue (current user)",
        "",
        "```",
        squeue_user(),
        "```",
        "",
        "## EAR ×4 (pre-existing jobs — not cancelled)",
        "",
        "| Line | Job ID | Output dir | Full count (train/test) | Smoke | Full audit | PointNet++ ready |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for line_key, spec in LINE_SPECS.items():
        out_dir = PROJECT_ROOT / "datasets" / spec["up_root"] / "ear_x4"
        smoke_dir = PROJECT_ROOT / "datasets" / spec["up_root"] / "ear_x4_smoke"
        counts = count_npy_files(out_dir)
        smoke_counts = count_npy_files(smoke_dir)
        smoke_status = "PASS" if smoke_counts["train"] == 20 and smoke_counts["test"] == 20 else "pending"
        full_audit = read_audit_status("ear", "full") if counts["train"] + counts["test"] else "not_run"
        pn_ready = "No — await full audit PASS" if full_audit != "PASS" else "Yes (user confirmation still required)"
        lines.append(
            f"| {line_key} | 1716174/1716175 | `{out_dir.relative_to(PROJECT_ROOT)}` | "
            f"{counts['train']}/{counts['test']} (exp {spec['full_train']}/{spec['full_test']}) | "
            f"{smoke_status} ({smoke_counts['train']}/{smoke_counts['test']}) | {full_audit} | {pn_ready} |"
        )

    for method_key in ("pdans", "punet", "pugcn"):
        meta = METHODS[method_key]
        jobs = load_job_ids(method_key)
        lines.extend(["", f"## {meta['label']} ×4", ""])
        if jobs:
            lines.extend(
                [
                    f"- Smoke A: **{jobs.get('SMOKE_A', '—')}** ({job_state(jobs.get('SMOKE_A', ''))})",
                    f"- Full A array: **{jobs.get('FULL_A', '—')}** ({job_state(jobs.get('FULL_A', ''))})",
                    f"- Smoke B: **{jobs.get('SMOKE_B', '—')}** ({job_state(jobs.get('SMOKE_B', ''))})",
                    f"- Full B array: **{jobs.get('FULL_B', '—')}** ({job_state(jobs.get('FULL_B', ''))})",
                    "",
                ]
            )
        lines.append(
            "| Line | Smoke dir | Full dir | Smoke audit | Full count | Full audit | PointNet++ ready |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for line_key, spec in LINE_SPECS.items():
            smoke_dir = PROJECT_ROOT / "datasets" / spec["up_root"] / meta["smoke_subdir"]
            full_dir = PROJECT_ROOT / "datasets" / spec["up_root"] / meta["subdir"]
            smoke_counts = count_npy_files(smoke_dir)
            full_counts = count_npy_files(full_dir)
            smoke_audit = read_audit_status(method_key, "smoke")
            if smoke_counts["train"] == 0 and smoke_counts["test"] == 0:
                smoke_audit = "pending"
            full_audit = read_audit_status(method_key, "full") if full_counts["train"] + full_counts["test"] else "not_run"
            pn_ready = "No"
            if full_audit == "PASS":
                pn_ready = "Eligible after user confirmation"
            lines.append(
                f"| {line_key} | `{smoke_dir.relative_to(PROJECT_ROOT)}` | "
                f"`{full_dir.relative_to(PROJECT_ROOT)}` | {smoke_audit} "
                f"({smoke_counts['train']}/{smoke_counts['test']}) | "
                f"{full_counts['train']}/{full_counts['test']} | {full_audit} | {pn_ready} |"
            )

    lines.extend(
        [
            "",
            "## Protocol targets",
            "",
            "| Line | Input | Output | Ratio |",
            "| --- | ---: | ---: | ---: |",
            "| A | 1024 | 4096 | 4.0 |",
            "| B | 512 | 2048 | 4.0 |",
            "",
            "## Notes",
            "",
            "- Legacy EAR ×2 data under `datasets/modelnet40_downsampled50_up/ear` is preserved.",
            f"- Legacy ×2 file count: {sum(1 for _ in LEGACY_EAR_X2.rglob('*.npy')) if LEGACY_EAR_X2.is_dir() else 0}",
            "- **No PointNet++ training** submitted from this pipeline.",
            "- Full audit command per method: `python scripts/audit_upsampling_x4_full.py --method <method>`",
            "",
        ]
    )

    if args.submit_log and args.submit_log.is_file():
        submit_path = args.submit_log.resolve()
        try:
            rel = submit_path.relative_to(PROJECT_ROOT.resolve())
        except ValueError:
            rel = submit_path
        lines.extend(["## Latest submit log", "", f"See `{rel}`", ""])

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
