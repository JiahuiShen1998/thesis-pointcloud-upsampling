#!/usr/bin/env python3
"""Audit PU-GCN readiness for ModelNet40 x4 two-line protocol."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pugcn_paths import (  # noqa: E402
    DEFAULT_PUGCN_CKPT,
    DEFAULT_PUGCN_PYTHON,
    DEFAULT_PUGCN_REPO,
    LOCAL_FALLBACK_ROOT,
    WOODY_PROJECT_ROOT,
    line_paths,
    resolve_project_root,
)


def check_tf_gpu(python_bin: Path) -> tuple[bool, str]:
    if not python_bin.exists():
        return False, f"python missing: {python_bin}"
    cmd = [
        str(python_bin),
        "-c",
        "import tensorflow as tf; print('gpu=' + str(tf.test.is_gpu_available()))",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        ok = proc.returncode == 0 and "gpu=True" in proc.stdout
        note = (proc.stdout + proc.stderr).strip().replace("\n", " ")[:500]
        return ok, note
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def count_npy(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("*.npy"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--repo-dir", default=str(DEFAULT_PUGCN_REPO))
    parser.add_argument("--checkpoint", default=str(DEFAULT_PUGCN_CKPT))
    parser.add_argument("--python-bin", default=str(DEFAULT_PUGCN_PYTHON))
    args = parser.parse_args()

    project_root = resolve_project_root(Path(args.project_root) if args.project_root else None)
    woody_root = WOODY_PROJECT_ROOT
    repo = Path(args.repo_dir)
    ckpt = Path(args.checkpoint)
    py_bin = Path(args.python_bin)

    ckpt_ok = (ckpt / "checkpoint").exists()
    repo_ok = (repo / "Upsampling" / "model.py").exists()
    gpu_ok, gpu_note = check_tf_gpu(py_bin)

    line_a = line_paths("lineA", project_root)
    line_b = line_paths("lineB", project_root)

    existing_a_raw = count_npy(line_a["raw_output_root"])
    existing_a_strict = count_npy(line_a["strict_output_root"])
    existing_b_raw = count_npy(line_b["raw_output_root"])
    existing_b_strict = count_npy(line_b["strict_output_root"])

    input_a_ok = line_a["input_root"].is_dir() and count_npy(line_a["input_root"]) > 0
    input_b_ok = line_b["input_root"].is_dir() and count_npy(line_b["input_root"]) > 0

    wrapper_ok = (project_root / "scripts" / "run_pugcn_modelnet40_x4.py").exists()

    if not repo_ok or not ckpt_ok:
        status = "pending_checkpoint" if not ckpt_ok else "not_found"
    elif not py_bin.exists():
        status = "pending_environment"
    elif not gpu_ok:
        status = "pending_environment"
    elif not wrapper_ok:
        status = "needs_wrapper"
    elif existing_a_strict >= 12311 and existing_b_strict >= 12311:
        status = "completed_existing_valid"
    elif not woody_root.exists() and project_root == LOCAL_FALLBACK_ROOT:
        status = "ready" if (input_a_ok or input_b_ok) else "blocked"
    elif woody_root.exists():
        status = "ready" if (input_a_ok and input_b_ok) else "blocked"
    else:
        status = "ready" if (input_a_ok or input_b_ok) else "blocked"

    notes = []
    if not woody_root.exists():
        notes.append(
            f"PROJECT_ROOT woody path not mounted on this host ({os.uname().nodename}); "
            f"using fallback {project_root}"
        )
    if not input_a_ok:
        notes.append(f"Line A input missing or empty: {line_a['input_root']}")
    if not input_b_ok:
        notes.append(f"Line B input missing or empty: {line_b['input_root']}")
    if existing_a_strict or existing_b_strict:
        notes.append(
            f"existing outputs lineA strict={existing_a_strict} raw={existing_a_raw}; "
            f"lineB strict={existing_b_strict} raw={existing_b_raw}"
        )
    notes.append(f"gpu_check: {gpu_note}")

    row = {
        "method": "PU-GCN",
        "code_path": str(repo.resolve()) if repo_ok else str(repo),
        "checkpoint_path": str(ckpt.resolve()) if ckpt_ok else str(ckpt),
        "environment": str(py_bin),
        "supports_x4": "true" if repo_ok and ckpt_ok else "false",
        "supports_lineA_1024_to_4096": "true" if repo_ok and ckpt_ok else "false",
        "supports_lineB_256_to_1024": "true" if repo_ok and ckpt_ok else "false",
        "input_format": "train/test/class/sample.npy float32 (N,3)",
        "output_format": "raw .npy + strict .npy (4096 or 1024,3)",
        "existing_outputs": (
            f"lineA raw={existing_a_raw} strict={existing_a_strict}; "
            f"lineB raw={existing_b_raw} strict={existing_b_strict}"
        ),
        "status": status,
        "note": " | ".join(notes),
    }

    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "modelnet40_pugcn_capability_audit.csv"
    md_path = reports / "modelnet40_pugcn_capability_audit.md"
    fields = list(row.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    md_lines = [
        "# ModelNet40 PU-GCN Capability Audit",
        "",
        f"- Host: `{os.uname().nodename}`",
        f"- Resolved project root: `{project_root}`",
        f"- Woody PROJECT_ROOT exists: `{woody_root.exists()}`",
        f"- Status: **{status}**",
        "",
        "## Summary",
        "",
        f"- code_path: `{row['code_path']}`",
        f"- checkpoint_path: `{row['checkpoint_path']}`",
        f"- environment: `{row['environment']}`",
        f"- supports_x4: {row['supports_x4']}",
        f"- supports_lineA_1024_to_4096: {row['supports_lineA_1024_to_4096']}",
        f"- supports_lineB_256_to_1024: {row['supports_lineB_256_to_1024']}",
        f"- existing_outputs: {row['existing_outputs']}",
        "",
        "## Note",
        "",
        row["note"],
        "",
        "## PU-EdgeFormer / TULIP (not in this run)",
        "",
        "- PU-EdgeFormer: `pending_checkpoint` (checkpoint not transferred yet)",
        "- TULIP: `pending / not included in this run`",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"status={status}")
    return 0 if status in {"ready", "completed_existing_valid"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
