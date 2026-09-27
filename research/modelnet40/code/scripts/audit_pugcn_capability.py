#!/usr/bin/env python3
"""PU-GCN capability audit for ModelNet40 ×4 two-line protocol."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    lineA_paths,
    lineB_paths,
    reports_dir,
)
from pugcn_modelnet40_utils import (
    DEFAULT_PUGCN_ROOT,
    DEFAULT_RESTORE,
    DEFAULT_TF_PYTHON,
)

PUGCN_ROOT = DEFAULT_PUGCN_ROOT
CHECKPOINT = DEFAULT_RESTORE
TF_PYTHON = DEFAULT_TF_PYTHON
ENV_NAME = "tf15_upsampling"


def _count_npy(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("*.npy"))


def _sample_shape(root: Path) -> tuple | None:
    sample = next(root.rglob("*.npy"), None) if root.is_dir() else None
    if sample is None:
        return None
    return tuple(np.load(sample).shape)


def _check_tf_import() -> tuple[str, str]:
    if not TF_PYTHON.is_file():
        return "fail", f"python missing: {TF_PYTHON}"
    env = os.environ.copy()
    try:
        proc = subprocess.run(
            [str(TF_PYTHON), "-c", "import tensorflow as tf; print(tf.__version__)"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        if proc.returncode != 0:
            return "fail", (proc.stderr or proc.stdout)[-500:]
        return "ok", proc.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return "fail", str(exc)


def _check_gpu() -> tuple[str, str]:
    if shutil.which("nvidia-smi"):
        proc = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            return "available", proc.stdout.strip().splitlines()[0]
        return "no_gpu", proc.stderr or "nvidia-smi empty"
    return "not_on_login_node", "nvidia-smi not found (expected on GPU node)"


def main() -> int:
    n = detect_original_point_count()
    down_n = n // 4
    four_n = n * 4

    lineA_raw = lineA_paths("pu_gcn")["raw"]
    lineA_strict = lineA_paths("pu_gcn")["strict_4N"]
    lineB_raw = lineB_paths("pu_gcn")["raw"]
    lineB_strict = lineB_paths("pu_gcn")["strict_N"]

    legacy_a = PROJECT_ROOT / "datasets" / "modelnet40_original_up" / "pugcn_x4"
    legacy_b = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "pugcn_x4"

    ckpt_ok = (CHECKPOINT / "checkpoint").is_file()
    code_ok = (PUGCN_ROOT / "main.py").is_file()
    so_ok = (PUGCN_ROOT / "tf_ops" / "grouping" / "tf_grouping_so.so").is_file()
    tf_import, tf_note = _check_tf_import()
    gpu_status, gpu_note = _check_gpu()

    existing = {
        "lineA_raw": _count_npy(lineA_raw),
        "lineA_strict": _count_npy(lineA_strict),
        "lineB_raw": _count_npy(lineB_raw),
        "lineB_strict": _count_npy(lineB_strict),
        "legacy_lineA_pugcn_x4": _count_npy(legacy_a),
        "legacy_lineB_pugcn_x4": _count_npy(legacy_b),
    }

    notes: list[str] = []
    status = "ready"

    if not code_ok:
        status = "not_found"
        notes.append("PU-GCN main.py missing")
    if not ckpt_ok:
        status = "pending_checkpoint"
        notes.append(f"checkpoint missing under {CHECKPOINT}")
    if not TF_PYTHON.is_file():
        status = "pending_environment"
        notes.append(f"conda env python missing: {TF_PYTHON}")
    elif tf_import != "ok":
        status = "pending_environment"
        notes.append(f"tensorflow import failed: {tf_note[:200]}")

    if existing["lineA_strict"] == 12311:
        sa = _sample_shape(lineA_strict)
        if sa == (four_n, 3):
            status = "completed_existing_valid"
            notes.append("Line A strict already complete at protocol path")
    if existing["lineB_strict"] == 12311:
        sb = _sample_shape(lineB_strict)
        if sb == (n, 3):
            if status == "ready":
                status = "completed_existing_valid"
            notes.append("Line B strict already complete at protocol path")

    if legacy_a.exists() or legacy_b.exists():
        notes.append(
            f"legacy outputs exist (lineA={existing['legacy_lineA_pugcn_x4']}, "
            f"lineB={existing['legacy_lineB_pugcn_x4']}) — NOT valid for two-line protocol"
        )

    if gpu_status == "not_on_login_node":
        notes.append("GPU/CUDA runtime validation requires GPU node (activate_upsampling_env.sh pugcn)")
    elif gpu_status == "no_gpu":
        status = "blocked" if status == "ready" else status
        notes.append(f"GPU issue: {gpu_note}")

    if status == "ready" and not so_ok:
        notes.append("tf_grouping_so.so missing — will recompile on GPU node before smoke")

    row = {
        "method": "PU-GCN",
        "code_path": str(PUGCN_ROOT),
        "checkpoint_path": str(CHECKPOINT),
        "environment": f"conda:{ENV_NAME} ({TF_PYTHON})",
        "supports_x4": "yes",
        "supports_lineA_1024_to_4096": f"yes (up_ratio=4, num_point=1024 -> ~4096)",
        "supports_lineB_256_to_1024": f"yes (up_ratio=4, num_point=256 -> ~1024)",
        "input_format": ".npy (N,3) float32 xyz; converted to .xyz for PU-GCN test phase",
        "output_format": ".xyz from PU-GCN result dir -> .npy raw + strict normalized",
        "existing_outputs": (
            f"lineA_raw={existing['lineA_raw']}, lineA_strict={existing['lineA_strict']}, "
            f"lineB_raw={existing['lineB_raw']}, lineB_strict={existing['lineB_strict']}"
        ),
        "status": status,
        "note": "; ".join(notes) if notes else "code+checkpoint+env present; proceed to GPU smoke",
    }

    rep = reports_dir()
    rep.mkdir(parents=True, exist_ok=True)
    csv_path = rep / "modelnet40_pugcn_capability_audit.csv"
    md_path = rep / "modelnet40_pugcn_capability_audit.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    md_lines = [
        "# PU-GCN Capability Audit",
        "",
        f"- Generated at: {ts}",
        f"- PROJECT_ROOT: `{PROJECT_ROOT}`",
        "",
        "## Summary",
        "",
        f"- **status:** `{status}`",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for k, v in row.items():
        md_lines.append(f"| {k} | {v} |")

    md_lines += [
        "",
        "## Environment checks",
        "",
        f"- TensorFlow import: **{tf_import}** ({tf_note[:120]})",
        f"- GPU: **{gpu_status}** ({gpu_note[:120]})",
        f"- Custom op .so: **{'present' if so_ok else 'missing'}**",
        "",
        "## Input datasets",
        "",
        f"- Line A input: `{ORIGINAL_ROOT}` ({n} pts)",
        f"- Line B input: `{DOWNSAMPLED_X4_ROOT}` ({down_n} pts)",
        "",
        "## Target output paths (two-line protocol)",
        "",
        f"- Line A raw: `{lineA_raw}`",
        f"- Line A strict: `{lineA_strict}` (expected {four_n} pts)",
        f"- Line B raw: `{lineB_raw}`",
        f"- Line B strict: `{lineB_strict}` (expected {n} pts)",
        "",
        "## Not in scope this run",
        "",
        "- PU-EdgeFormer: **pending_checkpoint** (checkpoint not transferred)",
        "- TULIP: **pending / not included in this run**",
    ]

    if status in ("blocked", "pending_checkpoint", "pending_environment", "not_found"):
        md_lines += ["", "## Blocked", "", "Do not submit full generation until resolved."]
    elif status == "completed_existing_valid":
        md_lines += ["", "## Note", "", "Existing valid outputs found; skip regeneration if audit PASS."]
    else:
        md_lines += ["", "## Next step", "", "Run GPU smoke test before full generation."]

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"status={status}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0 if status in ("ready", "completed_existing_valid") else 1


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
