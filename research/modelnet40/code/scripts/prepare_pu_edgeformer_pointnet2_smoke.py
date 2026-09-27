#!/usr/bin/env python3
"""Prepare PointNet++ inputs/configs/smoke jobs for PU-EdgeFormer only.

Smoke only — does NOT write or submit full-training jobs.
Does NOT start detector / KITTI AP evaluation.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    PROJECT_ROOT,
    detect_original_point_count,
    lineA_paths,
    lineB_paths,
    pointnet2_input_paths,
    reports_dir,
)

EXPECTED_TOTAL = EXPECTED_TRAIN + EXPECTED_TEST
CONFIGS_DIR = PROJECT_ROOT / "configs" / "pointnet2_x4_two_line"
SMOKE_JOBS_DIR = PROJECT_ROOT / "jobs" / "pointnet2_smoke"
SMOKE_RESULTS_ROOT = PROJECT_ROOT / "pointnet2_results" / "x4_two_line_smoke"
LOGS_SMOKE = PROJECT_ROOT / "logs" / "pointnet2_smoke"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


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


def count_npy(root: Path) -> int:
    if not root.is_dir() and not root.is_symlink():
        return 0
    return sum(1 for _ in root.rglob("*.npy"))


def count_split(root: Path, split: str) -> int:
    split_dir = root / split
    if not split_dir.is_dir() and not split_dir.is_symlink():
        return 0
    return sum(1 for _ in split_dir.rglob("*.npy"))


def metadata_status(root: Path) -> dict:
    meta = root / "metadata"
    class_map = meta / "class_to_idx.json"
    idx_map = meta / "idx_to_class.json"
    train_manifest = meta / "train_manifest.csv"
    test_manifest = meta / "test_manifest.csv"
    return {
        "metadata_dir_exists": meta.is_dir(),
        "has_class_to_idx": class_map.is_file(),
        "has_idx_to_class": idx_map.is_file(),
        "has_train_manifest": train_manifest.exists(),
        "has_test_manifest": test_manifest.exists(),
        "class_count": (
            len(json.loads(class_map.read_text(encoding="utf-8")))
            if class_map.is_file()
            else 0
        ),
    }


def sample_shape(root: Path, split: str) -> str:
    split_dir = root / split
    for npy in sorted(split_dir.rglob("*.npy")):
        arr = np.load(npy)
        return str(tuple(arr.shape))
    return ""


def audit_branch(
    line: str,
    method: str,
    source: Path,
    input_path: Path,
    expected_pts: int,
) -> dict:
    link_status, symlink_valid = ensure_symlink(source, input_path)
    meta = metadata_status(input_path)
    train_n = count_split(input_path, "train")
    test_n = count_split(input_path, "test")
    total = count_npy(input_path)
    shape = sample_shape(input_path, "train")
    status = "PASS"
    reasons: list[str] = []
    if link_status != "linked" and not (input_path.is_symlink() and symlink_valid):
        status = "FAIL"
        reasons.append(f"symlink:{link_status}")
    if train_n != EXPECTED_TRAIN or test_n != EXPECTED_TEST:
        status = "FAIL"
        reasons.append(f"counts:{train_n}/{test_n}")
    if not meta["has_class_to_idx"] or not meta["has_idx_to_class"]:
        status = "FAIL"
        reasons.append("missing_class_maps")
    if meta["has_train_manifest"] or meta["has_test_manifest"]:
        status = "FAIL"
        reasons.append("wrong_manifest_present")
    if shape != f"({expected_pts}, 3)":
        status = "FAIL"
        reasons.append(f"shape:{shape}")
    return {
        "line": line,
        "method": method,
        "source_path": str(source),
        "pointnet2_input_path": str(input_path),
        "link_type": "symlink" if input_path.is_symlink() else "other",
        "symlink_valid": symlink_valid,
        "link_status": link_status,
        "expected_point_count": expected_pts,
        "sample_train_shape": shape,
        "train_count": train_n,
        "test_count": test_n,
        "total_count": total,
        **meta,
        "status": status,
        "fail_reasons": ";".join(reasons),
    }


def write_lineb_config(input_rel: str, out: Path) -> None:
    text = f"""# PointNet++ config: Line B PU-EdgeFormer (smoke/config prep only)
experiment_name: lineB_downsampled_x4_up_pu_edgeformer
line: B
branch: lineB_downsampled_x4_up_pu_edgeformer
method: PU-EdgeFormer
data_root: {input_rel}
expected_point_count: 1024
num_point: 1024
allow_resample: false
model: pointnet2_cls_ssg
num_category: 40
# Classification comparison (do not mix with geometry deltas):
#   primary: vs Downsampled x4 baseline 256
#   secondary gap: vs Original baseline 1024
comparison_baseline_primary: Downsampled x4 baseline 256
comparison_baseline_secondary: Original baseline 1024
training:
  epoch: 200
  batch_size: 24
  learning_rate: 0.001
  decay_rate: 0.0001
  optimizer: Adam
  seed: 42
  num_workers: 4
output_dir: pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/pu_edgeformer
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def write_linea_config(input_rel: str, out: Path) -> None:
    text = f"""# PointNet++ config: Line A PU-EdgeFormer (smoke/config prep only)
experiment_name: lineA_original_up_pu_edgeformer
line: A
branch: lineA_original_up_pu_edgeformer
method: PU-EdgeFormer
data_root: {input_rel}
expected_point_count: 4096
num_point: 4096
allow_resample: false
model: pointnet2_cls_ssg
num_category: 40
# Classification comparison:
#   baseline: Original baseline 1024
comparison_baseline_primary: Original baseline 1024
training:
  epoch: 200
  batch_size: 24
  learning_rate: 0.001
  decay_rate: 0.0001
  optimizer: Adam
  seed: 42
  num_workers: 4
output_dir: pointnet2_results/x4_two_line_final/lineA_original_up/pu_edgeformer
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")


def smoke_sbatch(
    *,
    job_name: str,
    job_slug: str,
    branch_name: str,
    input_path: Path,
    output_dir: Path,
    num_point: int,
    variant: str,
) -> str:
    log_dir = LOGS_SMOKE / job_slug
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output={log_dir}/slurm_%j.out
#SBATCH --error={log_dir}/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="{PROJECT_ROOT}"
BRANCH_NAME="{branch_name}"
INPUT_PATH="{input_path}"
OUTPUT_DIR="{output_dir}"
LOG_FILE="${{OUTPUT_DIR}}/train.log"

mkdir -p "${{OUTPUT_DIR}}" "{log_dir}"

echo "=== PointNet++ smoke job (PU-EdgeFormer) ==="
echo "branch=${{BRANCH_NAME}}"
echo "input_path=${{INPUT_PATH}}"
echo "expected_point_count={num_point}"
echo "output_dir=${{OUTPUT_DIR}}"
echo "POINTNET_FULL_TRAINING_STARTED=NO"
echo "DETECTOR_EVAL_STARTED=NO"
echo "KITTI_AP_EVAL_STARTED=NO"
echo "POINTNET_SMOKE_STARTED=YES"
echo "host=$(hostname)"
echo "start=$(date)"

module purge 2>/dev/null || true
module load python/pytorch2.6py3.12
pip install --user -q tqdm

cd "${{PROJECT_ROOT}}"
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python scripts/train_pointnet2.py \\
  --variant "{variant}" \\
  --data-root "${{INPUT_PATH}}" \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --log-file "${{LOG_FILE}}" \\
  --reports-dir "${{OUTPUT_DIR}}" \\
  --model pointnet2_cls_ssg \\
  --num-category 40 \\
  --num-point {num_point} \\
  --no-allow-resample \\
  --batch-size 8 \\
  --epoch 1 \\
  --learning-rate 0.001 \\
  --decay-rate 1e-4 \\
  --optimizer Adam \\
  --seed 42 \\
  --num-workers 2 \\
  --max-train-batches 2 \\
  --max-eval-batches 2

echo "smoke_status=PASS"
echo "end=$(date)"
"""


def main() -> int:
    n = detect_original_point_count()
    four_n = n * 4
    p2 = pointnet2_input_paths()

    lineb_src = lineB_paths("pu_edgeformer")["strict_N"]
    linea_src = lineA_paths("pu_edgeformer")["strict_4N"]
    lineb_in = p2["lineB_up_root"] / "pu_edgeformer"
    linea_in = p2["lineA_up_root"] / "pu_edgeformer"

    rows = [
        audit_branch("B", "pu_edgeformer", lineb_src, lineb_in, n),
        audit_branch("A", "pu_edgeformer", linea_src, linea_in, four_n),
    ]

    cfg_b = CONFIGS_DIR / "lineB_downsampled_x4_up_pu_edgeformer_1024.yaml"
    cfg_a = CONFIGS_DIR / "lineA_original_up_pu_edgeformer_4096.yaml"
    write_lineb_config("pointnet2_inputs/lineB_downsampled_x4_up/pu_edgeformer", cfg_b)
    write_linea_config("pointnet2_inputs/lineA_original_up/pu_edgeformer", cfg_a)

    smoke_b = SMOKE_JOBS_DIR / "smoke_lineB_pu_edgeformer_1024.sbatch"
    smoke_a = SMOKE_JOBS_DIR / "smoke_lineA_pu_edgeformer_4096.sbatch"
    smoke_b.write_text(
        smoke_sbatch(
            job_name="pn2smk_lineB_edgef_1024",
            job_slug="lineB_pu_edgeformer_1024",
            branch_name="lineB_downsampled_x4_up_pu_edgeformer",
            input_path=lineb_in.resolve(),
            output_dir=SMOKE_RESULTS_ROOT / "lineB_downsampled_x4_up" / "pu_edgeformer",
            num_point=n,
            variant="lineB_downsampled_x4_up_pu_edgeformer_smoke",
        ),
        encoding="utf-8",
    )
    smoke_a.write_text(
        smoke_sbatch(
            job_name="pn2smk_lineA_edgef_4096",
            job_slug="lineA_pu_edgeformer_4096",
            branch_name="lineA_original_up_pu_edgeformer",
            input_path=linea_in.resolve(),
            output_dir=SMOKE_RESULTS_ROOT / "lineA_original_up" / "pu_edgeformer",
            num_point=four_n,
            variant="lineA_original_up_pu_edgeformer_smoke",
        ),
        encoding="utf-8",
    )
    smoke_b.chmod(0o755)
    smoke_a.chmod(0o755)

    # Input prep audit (not the final smoke runtime audit)
    rep = reports_dir()
    prep_csv = rep / "modelnet40_pointnet2_pu_edgeformer_input_prep_20260716.csv"
    prep_md = rep / "modelnet40_pointnet2_pu_edgeformer_input_prep_20260716.md"
    fields = list(rows[0].keys())
    with prep_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PointNet++ PU-EdgeFormer Input/Config Prep",
        "",
        f"- Generated: `{utc_now()}`",
        "- Scope: input symlink + config + smoke sbatch only",
        "- POINTNET_FULL_TRAINING_STARTED=NO",
        "- DETECTOR_EVAL_STARTED=NO",
        "- KITTI_AP_EVAL_STARTED=NO",
        "- POINTNET_SMOKE_STARTED=pending_submit",
        "",
        f"- Line B config: `{cfg_b.relative_to(PROJECT_ROOT)}`",
        f"- Line A config: `{cfg_a.relative_to(PROJECT_ROOT)}`",
        f"- Line B smoke job: `{smoke_b.relative_to(PROJECT_ROOT)}`",
        f"- Line A smoke job: `{smoke_a.relative_to(PROJECT_ROOT)}`",
        "",
        "| line | input | train | test | pts | manifests | status |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        manif = (
            "NONE (OK)"
            if not r["has_train_manifest"] and not r["has_test_manifest"]
            else "PRESENT (BAD)"
        )
        lines.append(
            f"| {r['line']} | `{Path(r['pointnet2_input_path']).relative_to(PROJECT_ROOT)}` | "
            f"{r['train_count']} | {r['test_count']} | {r['expected_point_count']} | "
            f"{manif} | {r['status']} |"
        )
    prep_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("configs:", cfg_b, cfg_a)
    print("smoke_jobs:", smoke_b, smoke_a)
    for r in rows:
        print(
            f"line{r['line']}: status={r['status']} "
            f"train/test={r['train_count']}/{r['test_count']} "
            f"shape={r['sample_train_shape']} "
            f"manifests={r['has_train_manifest']}/{r['has_test_manifest']}"
        )
        if r["status"] != "PASS":
            print("  fail:", r["fail_reasons"])

    return 0 if all(r["status"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
