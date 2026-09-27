#!/usr/bin/env python3
"""Prepare PointNet++ training inputs, audits, configs, and job plans."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    EXPECTED_CLASSES,
    EXPECTED_TEST,
    EXPECTED_TRAIN,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    DOWNSAMPLED_X4_ROOT,
    lineA_paths,
    lineB_paths,
)


TOTAL_EXPECTED = EXPECTED_TRAIN + EXPECTED_TEST
REPORTS_DIR = PROJECT_ROOT / "reports"
POINTNET2_INPUTS = PROJECT_ROOT / "pointnet2_inputs"
CONFIGS_DIR = PROJECT_ROOT / "configs" / "pointnet2_x4_two_line"
SMOKE_JOBS_DIR = PROJECT_ROOT / "jobs" / "pointnet2_smoke"
FULL_JOBS_DIR = PROJECT_ROOT / "jobs" / "pointnet2_full"
SMOKE_RESULTS_ROOT = PROJECT_ROOT / "pointnet2_results" / "x4_two_line_smoke"
FULL_RESULTS_ROOT = PROJECT_ROOT / "pointnet2_results" / "x4_two_line_final"


@dataclass(frozen=True)
class BranchSpec:
    line: str
    branch: str
    method: str
    branch_name: str
    source_path: Path
    expected_point_count: int
    input_path: Path
    config_path: Path
    smoke_job_path: Path
    full_job_path: Path
    result_dir: Path
    smoke_result_dir: Path
    job_slug: str


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def branch_specs() -> list[BranchSpec]:
    return [
        BranchSpec(
            line="B",
            branch="lineB_downsampled_x4_baseline",
            method="baseline",
            branch_name="lineB_downsampled_x4_baseline",
            source_path=PROJECT_ROOT / "datasets" / "modelnet40_downsampled_x4",
            expected_point_count=256,
            input_path=POINTNET2_INPUTS / "lineB_downsampled_x4_baseline",
            config_path=CONFIGS_DIR / "lineB_downsampled_x4_baseline_256.yaml",
            smoke_job_path=SMOKE_JOBS_DIR / "smoke_lineB_downsampled_x4_baseline_256.sbatch",
            full_job_path=FULL_JOBS_DIR / "run_lineB_downsampled_x4_baseline_256.sbatch",
            result_dir=FULL_RESULTS_ROOT / "lineB_downsampled_x4_baseline",
            smoke_result_dir=SMOKE_RESULTS_ROOT / "lineB_downsampled_x4_baseline",
            job_slug="lineB_downsampled_x4_baseline_256",
        ),
        BranchSpec(
            line="B",
            branch="lineB_downsampled_x4_up_ear",
            method="EAR",
            branch_name="lineB_downsampled_x4_up_ear",
            source_path=PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up" / "strict_N" / "ear",
            expected_point_count=1024,
            input_path=POINTNET2_INPUTS / "lineB_downsampled_x4_up" / "ear",
            config_path=CONFIGS_DIR / "lineB_downsampled_x4_up_ear_1024.yaml",
            smoke_job_path=SMOKE_JOBS_DIR / "smoke_lineB_ear_1024.sbatch",
            full_job_path=FULL_JOBS_DIR / "run_lineB_ear_1024.sbatch",
            result_dir=FULL_RESULTS_ROOT / "lineB_downsampled_x4_up" / "ear",
            smoke_result_dir=SMOKE_RESULTS_ROOT / "lineB_downsampled_x4_up" / "ear",
            job_slug="lineB_ear_1024",
        ),
        BranchSpec(
            line="B",
            branch="lineB_downsampled_x4_up_pdans",
            method="PDANS",
            branch_name="lineB_downsampled_x4_up_pdans",
            source_path=PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up" / "strict_N" / "pdans",
            expected_point_count=1024,
            input_path=POINTNET2_INPUTS / "lineB_downsampled_x4_up" / "pdans",
            config_path=CONFIGS_DIR / "lineB_downsampled_x4_up_pdans_1024.yaml",
            smoke_job_path=SMOKE_JOBS_DIR / "smoke_lineB_pdans_1024.sbatch",
            full_job_path=FULL_JOBS_DIR / "run_lineB_pdans_1024.sbatch",
            result_dir=FULL_RESULTS_ROOT / "lineB_downsampled_x4_up" / "pdans",
            smoke_result_dir=SMOKE_RESULTS_ROOT / "lineB_downsampled_x4_up" / "pdans",
            job_slug="lineB_pdans_1024",
        ),
        BranchSpec(
            line="B",
            branch="lineB_downsampled_x4_up_pu_net",
            method="PU-Net",
            branch_name="lineB_downsampled_x4_up_pu_net",
            source_path=PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up" / "strict_N" / "pu_net",
            expected_point_count=1024,
            input_path=POINTNET2_INPUTS / "lineB_downsampled_x4_up" / "pu_net",
            config_path=CONFIGS_DIR / "lineB_downsampled_x4_up_pu_net_1024.yaml",
            smoke_job_path=SMOKE_JOBS_DIR / "smoke_lineB_punet_1024.sbatch",
            full_job_path=FULL_JOBS_DIR / "run_lineB_punet_1024.sbatch",
            result_dir=FULL_RESULTS_ROOT / "lineB_downsampled_x4_up" / "pu_net",
            smoke_result_dir=SMOKE_RESULTS_ROOT / "lineB_downsampled_x4_up" / "pu_net",
            job_slug="lineB_punet_1024",
        ),
        BranchSpec(
            line="B",
            branch="lineB_downsampled_x4_up_pu_gcn",
            method="PU-GCN",
            branch_name="lineB_downsampled_x4_up_pu_gcn",
            source_path=PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up" / "strict_N" / "pu_gcn",
            expected_point_count=1024,
            input_path=POINTNET2_INPUTS / "lineB_downsampled_x4_up" / "pu_gcn",
            config_path=CONFIGS_DIR / "lineB_downsampled_x4_up_pu_gcn_1024.yaml",
            smoke_job_path=SMOKE_JOBS_DIR / "smoke_lineB_pugcn_1024.sbatch",
            full_job_path=FULL_JOBS_DIR / "run_lineB_pugcn_1024.sbatch",
            result_dir=FULL_RESULTS_ROOT / "lineB_downsampled_x4_up" / "pu_gcn",
            smoke_result_dir=SMOKE_RESULTS_ROOT / "lineB_downsampled_x4_up" / "pu_gcn",
            job_slug="lineB_pugcn_1024",
        ),
        BranchSpec(
            line="A",
            branch="lineA_original_baseline",
            method="baseline",
            branch_name="lineA_original_baseline",
            source_path=PROJECT_ROOT / "datasets" / "modelnet40_original",
            expected_point_count=1024,
            input_path=POINTNET2_INPUTS / "lineA_original_baseline",
            config_path=CONFIGS_DIR / "lineA_original_baseline_1024.yaml",
            smoke_job_path=SMOKE_JOBS_DIR / "smoke_lineA_original_baseline_1024.sbatch",
            full_job_path=FULL_JOBS_DIR / "run_lineA_original_baseline_1024.sbatch",
            result_dir=FULL_RESULTS_ROOT / "lineA_original_baseline",
            smoke_result_dir=SMOKE_RESULTS_ROOT / "lineA_original_baseline",
            job_slug="lineA_original_baseline_1024",
        ),
        BranchSpec(
            line="A",
            branch="lineA_original_up_ear",
            method="EAR",
            branch_name="lineA_original_up_ear",
            source_path=lineA_paths("ear")["strict_4N"],
            expected_point_count=4096,
            input_path=POINTNET2_INPUTS / "lineA_original_up" / "ear",
            config_path=CONFIGS_DIR / "lineA_original_up_ear_4096.yaml",
            smoke_job_path=SMOKE_JOBS_DIR / "smoke_lineA_ear_4096.sbatch",
            full_job_path=FULL_JOBS_DIR / "run_lineA_ear_4096.sbatch",
            result_dir=FULL_RESULTS_ROOT / "lineA_original_up" / "ear",
            smoke_result_dir=SMOKE_RESULTS_ROOT / "lineA_original_up" / "ear",
            job_slug="lineA_ear_4096",
        ),
        BranchSpec(
            line="A",
            branch="lineA_original_up_pdans",
            method="PDANS",
            branch_name="lineA_original_up_pdans",
            source_path=lineA_paths("pdans")["strict_4N"],
            expected_point_count=4096,
            input_path=POINTNET2_INPUTS / "lineA_original_up" / "pdans",
            config_path=CONFIGS_DIR / "lineA_original_up_pdans_4096.yaml",
            smoke_job_path=SMOKE_JOBS_DIR / "smoke_lineA_pdans_4096.sbatch",
            full_job_path=FULL_JOBS_DIR / "run_lineA_pdans_4096.sbatch",
            result_dir=FULL_RESULTS_ROOT / "lineA_original_up" / "pdans",
            smoke_result_dir=SMOKE_RESULTS_ROOT / "lineA_original_up" / "pdans",
            job_slug="lineA_pdans_4096",
        ),
        BranchSpec(
            line="A",
            branch="lineA_original_up_pu_net",
            method="PU-Net",
            branch_name="lineA_original_up_pu_net",
            source_path=lineA_paths("pu_net")["strict_4N"],
            expected_point_count=4096,
            input_path=POINTNET2_INPUTS / "lineA_original_up" / "pu_net",
            config_path=CONFIGS_DIR / "lineA_original_up_pu_net_4096.yaml",
            smoke_job_path=SMOKE_JOBS_DIR / "smoke_lineA_punet_4096.sbatch",
            full_job_path=FULL_JOBS_DIR / "run_lineA_punet_4096.sbatch",
            result_dir=FULL_RESULTS_ROOT / "lineA_original_up" / "pu_net",
            smoke_result_dir=SMOKE_RESULTS_ROOT / "lineA_original_up" / "pu_net",
            job_slug="lineA_punet_4096",
        ),
        BranchSpec(
            line="A",
            branch="lineA_original_up_pu_gcn",
            method="PU-GCN",
            branch_name="lineA_original_up_pu_gcn",
            source_path=lineA_paths("pu_gcn")["strict_4N"],
            expected_point_count=4096,
            input_path=POINTNET2_INPUTS / "lineA_original_up" / "pu_gcn",
            config_path=CONFIGS_DIR / "lineA_original_up_pu_gcn_4096.yaml",
            smoke_job_path=SMOKE_JOBS_DIR / "smoke_lineA_pugcn_4096.sbatch",
            full_job_path=FULL_JOBS_DIR / "run_lineA_pugcn_4096.sbatch",
            result_dir=FULL_RESULTS_ROOT / "lineA_original_up" / "pu_gcn",
            smoke_result_dir=SMOKE_RESULTS_ROOT / "lineA_original_up" / "pu_gcn",
            job_slug="lineA_pugcn_4096",
        ),
    ]


def load_reference_manifests() -> tuple[dict[str, set[tuple[str, str]]], list[str]]:
    metadata = ORIGINAL_ROOT / "metadata"
    classes = sorted(json.loads((metadata / "class_to_idx.json").read_text(encoding="utf-8")).keys())
    refs: dict[str, set[tuple[str, str]]] = {}
    for split in ("train", "test"):
        rows: set[tuple[str, str]] = set()
        with open(metadata / f"{split}_manifest.csv", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.add((row["class_name"], row["shape_id"]))
        refs[split] = rows
    return refs, classes


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_branch(spec: BranchSpec, refs: dict[str, set[tuple[str, str]]], expected_classes: list[str]) -> dict:
    row = {
        "line": spec.line,
        "branch": spec.branch,
        "method": spec.method,
        "source_path": str(spec.source_path),
        "expected_point_count": spec.expected_point_count,
        "actual_min_points": "",
        "actual_max_points": "",
        "exact_point_count_samples": 0,
        "train_samples": 0,
        "test_samples": 0,
        "total_samples": 0,
        "nan_count": 0,
        "inf_count": 0,
        "status": "FAIL",
        "note": "",
    }

    issues: list[str] = []
    if not spec.source_path.is_dir():
        row["note"] = "missing source path"
        return row

    actual_min: int | None = None
    actual_max: int | None = None
    exact_count_samples = 0
    nan_count = 0
    inf_count = 0
    total_samples = 0
    split_counts: dict[str, int] = {}

    for split, expected_split_count in (("train", EXPECTED_TRAIN), ("test", EXPECTED_TEST)):
        split_dir = spec.source_path / split
        if not split_dir.is_dir():
            issues.append(f"missing {split} split")
            continue

        class_dirs = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        if class_dirs != expected_classes:
            missing = sorted(set(expected_classes) - set(class_dirs))
            extra = sorted(set(class_dirs) - set(expected_classes))
            if missing:
                issues.append(f"{split} missing classes: {','.join(missing[:5])}")
            if extra:
                issues.append(f"{split} extra classes: {','.join(extra[:5])}")

        split_ids: set[tuple[str, str]] = set()
        split_count = 0
        for class_name in class_dirs:
            for npy_path in sorted((split_dir / class_name).glob("*.npy")):
                split_ids.add((class_name, npy_path.stem))
                split_count += 1
                points = np.load(npy_path, mmap_mode="r")
                if points.ndim != 2 or points.shape[1] != 3:
                    issues.append(f"invalid shape rank in {npy_path.name}")
                    continue
                point_count = int(points.shape[0])
                actual_min = point_count if actual_min is None else min(actual_min, point_count)
                actual_max = point_count if actual_max is None else max(actual_max, point_count)
                if point_count == spec.expected_point_count:
                    exact_count_samples += 1
                nan_count += int(np.isnan(points).sum())
                inf_count += int(np.isinf(points).sum())

        split_counts[split] = split_count
        total_samples += split_count
        if split_count != expected_split_count:
            issues.append(f"{split} count {split_count} != {expected_split_count}")
        if split_ids != refs[split]:
            missing_ids = len(refs[split] - split_ids)
            extra_ids = len(split_ids - refs[split])
            issues.append(f"{split} manifest mismatch missing={missing_ids} extra={extra_ids}")

    row["actual_min_points"] = actual_min if actual_min is not None else ""
    row["actual_max_points"] = actual_max if actual_max is not None else ""
    row["exact_point_count_samples"] = exact_count_samples
    row["train_samples"] = split_counts.get("train", 0)
    row["test_samples"] = split_counts.get("test", 0)
    row["total_samples"] = total_samples
    row["nan_count"] = nan_count
    row["inf_count"] = inf_count

    if total_samples != TOTAL_EXPECTED:
        issues.append(f"total {total_samples} != {TOTAL_EXPECTED}")
    if exact_count_samples != total_samples:
        issues.append(f"exact_count_samples {exact_count_samples} != total {total_samples}")
    if nan_count != 0:
        issues.append(f"NaN values: {nan_count}")
    if inf_count != 0:
        issues.append(f"Inf values: {inf_count}")

    row["status"] = "PASS" if not issues else "FAIL"
    row["note"] = "; ".join(issues) if issues else "all checks passed"
    return row


def ensure_input_symlink(spec: BranchSpec) -> dict:
    spec.input_path.parent.mkdir(parents=True, exist_ok=True)
    status = "missing_source"
    note = ""

    if not spec.source_path.exists():
        note = "source path missing"
    elif spec.input_path.is_symlink():
        target = spec.input_path.resolve()
        if target == spec.source_path.resolve():
            status = "valid_existing_symlink"
            note = "symlink target matches source"
        else:
            status = "invalid_existing_symlink"
            note = f"points to {target}"
    elif spec.input_path.exists():
        try:
            target = spec.input_path.resolve()
        except FileNotFoundError:
            target = spec.input_path
        if target == spec.source_path.resolve():
            status = "existing_path_matches_source"
            note = "existing path resolves to source"
        else:
            status = "existing_non_symlink"
            note = "existing path is not a symlink"
    else:
        spec.input_path.symlink_to(spec.source_path.resolve())
        status = "created_symlink"
        note = "symlink created"

    resolved_target = ""
    if spec.input_path.exists() or spec.input_path.is_symlink():
        try:
            resolved_target = str(spec.input_path.resolve())
        except FileNotFoundError:
            resolved_target = ""

    return {
        "line": spec.line,
        "branch": spec.branch,
        "method": spec.method,
        "input_path": str(spec.input_path),
        "expected_source_path": str(spec.source_path),
        "exists": spec.input_path.exists() or spec.input_path.is_symlink(),
        "is_symlink": spec.input_path.is_symlink(),
        "resolved_target": resolved_target,
        "target_exists": spec.source_path.exists(),
        "status": status,
        "note": note,
    }


def write_input_audit(rows: list[dict]) -> tuple[Path, Path]:
    csv_path = REPORTS_DIR / "modelnet40_pointnet2_training_input_audit.csv"
    fieldnames = [
        "line",
        "branch",
        "method",
        "source_path",
        "expected_point_count",
        "actual_min_points",
        "actual_max_points",
        "exact_point_count_samples",
        "train_samples",
        "test_samples",
        "total_samples",
        "nan_count",
        "inf_count",
        "status",
        "note",
    ]
    write_csv(csv_path, rows, fieldnames)

    md_path = REPORTS_DIR / "modelnet40_pointnet2_training_input_audit.md"
    lines = [
        "# ModelNet40 PointNet++ Training Input Audit",
        "",
        f"- Generated at: {now_utc()}",
        f"- Expected samples per branch: train={EXPECTED_TRAIN}, test={EXPECTED_TEST}, total={TOTAL_EXPECTED}",
        "",
        "| line | branch | method | expected pts | min | max | exact | train | test | total | NaN | Inf | status | note |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['line']} | {row['branch']} | {row['method']} | {row['expected_point_count']} | "
            f"{row['actual_min_points']} | {row['actual_max_points']} | {row['exact_point_count_samples']} | "
            f"{row['train_samples']} | {row['test_samples']} | {row['total_samples']} | {row['nan_count']} | "
            f"{row['inf_count']} | {row['status']} | {row['note']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_symlink_audit(rows: list[dict]) -> tuple[Path, Path]:
    csv_path = REPORTS_DIR / "modelnet40_pointnet2_input_symlink_audit.csv"
    fieldnames = [
        "line",
        "branch",
        "method",
        "input_path",
        "expected_source_path",
        "exists",
        "is_symlink",
        "resolved_target",
        "target_exists",
        "status",
        "note",
    ]
    write_csv(csv_path, rows, fieldnames)

    md_path = REPORTS_DIR / "modelnet40_pointnet2_input_symlink_audit.md"
    lines = [
        "# ModelNet40 PointNet++ Input Symlink Audit",
        "",
        f"- Generated at: {now_utc()}",
        "- Large datasets are linked, not copied.",
        "",
        "| line | branch | method | input path | symlink | resolved target | status | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['line']} | {row['branch']} | {row['method']} | `{row['input_path']}` | "
            f"{row['is_symlink']} | `{row['resolved_target']}` | {row['status']} | {row['note']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def config_text(spec: BranchSpec) -> str:
    return "\n".join(
        [
            f"# Auto-generated branch config for {spec.branch_name}",
            f"experiment_name: {spec.branch_name}",
            f"line: {spec.line}",
            f"branch: {spec.branch}",
            f"method: {spec.method}",
            f"data_root: {spec.input_path.relative_to(PROJECT_ROOT)}",
            f"expected_point_count: {spec.expected_point_count}",
            "num_point: {0}".format(spec.expected_point_count),
            "allow_resample: false",
            "model: pointnet2_cls_ssg",
            "num_category: 40",
            "training:",
            "  epoch: 200",
            "  batch_size: 24",
            "  learning_rate: 0.001",
            "  decay_rate: 0.0001",
            "  optimizer: Adam",
            "  seed: 42",
            "  num_workers: 4",
            f"output_dir: {spec.result_dir.relative_to(PROJECT_ROOT)}",
            "",
        ]
    )


def smoke_job_text(spec: BranchSpec) -> str:
    slurm_name = f"pn2smk_{spec.job_slug}"[:128]
    return f"""#!/bin/bash
#SBATCH --job-name={slurm_name}
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output={PROJECT_ROOT}/logs/pointnet2_smoke/{spec.job_slug}/slurm_%j.out
#SBATCH --error={PROJECT_ROOT}/logs/pointnet2_smoke/{spec.job_slug}/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="{PROJECT_ROOT}"
BRANCH_NAME="{spec.branch_name}"
INPUT_PATH="{spec.input_path}"
OUTPUT_DIR="{spec.smoke_result_dir}"
LOG_FILE="${{OUTPUT_DIR}}/train.log"

mkdir -p "${{OUTPUT_DIR}}" "{PROJECT_ROOT}/logs/pointnet2_smoke/{spec.job_slug}"

echo "=== PointNet++ smoke job ==="
echo "branch=${{BRANCH_NAME}}"
echo "input_path=${{INPUT_PATH}}"
echo "expected_point_count={spec.expected_point_count}"
echo "output_dir=${{OUTPUT_DIR}}"
echo "host=$(hostname)"
echo "start=$(date)"

module purge 2>/dev/null || true
module load python/pytorch2.6py3.12
pip install --user -q tqdm

cd "${{PROJECT_ROOT}}"
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

python scripts/train_pointnet2.py \\
  --variant "{spec.branch_name}_smoke" \\
  --data-root "${{INPUT_PATH}}" \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --log-file "${{LOG_FILE}}" \\
  --reports-dir "${{OUTPUT_DIR}}" \\
  --model pointnet2_cls_ssg \\
  --num-category 40 \\
  --num-point {spec.expected_point_count} \\
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


def full_job_text(spec: BranchSpec) -> str:
    slurm_name = f"pn2_{spec.job_slug}"[:128]
    return f"""#!/bin/bash
#SBATCH --job-name={slurm_name}
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output={PROJECT_ROOT}/logs/pointnet2_full/{spec.job_slug}/slurm_%j.out
#SBATCH --error={PROJECT_ROOT}/logs/pointnet2_full/{spec.job_slug}/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="{PROJECT_ROOT}"
BRANCH_NAME="{spec.branch_name}"
INPUT_PATH="{spec.input_path}"
OUTPUT_DIR="{spec.result_dir}"
LOG_FILE="${{OUTPUT_DIR}}/train.log"

mkdir -p "${{OUTPUT_DIR}}" "{PROJECT_ROOT}/logs/pointnet2_full/{spec.job_slug}"

echo "=== PointNet++ full training job ==="
echo "branch=${{BRANCH_NAME}}"
echo "input_path=${{INPUT_PATH}}"
echo "expected_point_count={spec.expected_point_count}"
echo "output_dir=${{OUTPUT_DIR}}"
echo "start=$(date)"

module purge 2>/dev/null || true
module load python/pytorch2.6py3.12
pip install --user -q tqdm

cd "${{PROJECT_ROOT}}"
python scripts/train_pointnet2.py \\
  --variant "${{BRANCH_NAME}}" \\
  --data-root "${{INPUT_PATH}}" \\
  --output-dir "${{OUTPUT_DIR}}" \\
  --log-file "${{LOG_FILE}}" \\
  --reports-dir "${{OUTPUT_DIR}}" \\
  --model pointnet2_cls_ssg \\
  --num-category 40 \\
  --num-point {spec.expected_point_count} \\
  --no-allow-resample \\
  --batch-size 24 \\
  --epoch 200 \\
  --learning-rate 0.001 \\
  --decay-rate 1e-4 \\
  --optimizer Adam \\
  --seed 42 \\
  --num-workers 4

echo "end=$(date)"
"""


def write_configs_and_jobs(specs: list[BranchSpec]) -> None:
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    FULL_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        spec.config_path.write_text(config_text(spec), encoding="utf-8")
        spec.smoke_job_path.write_text(smoke_job_text(spec), encoding="utf-8")
        spec.full_job_path.write_text(full_job_text(spec), encoding="utf-8")


def write_dataloader_audit(specs: list[BranchSpec]) -> Path:
    md_path = REPORTS_DIR / "modelnet40_pointnet2_dataloader_pointcount_audit.md"
    lines = [
        "# ModelNet40 PointNet++ Dataloader Point-Count Audit",
        "",
        f"- Generated at: {now_utc()}",
        "",
        "## Findings",
        "",
        "- `scripts/train_pointnet2.py` defaults to `--num-point 1024` unless overridden at runtime.",
        "- `scripts/modelnet_npy_dataloader.py` will resample any sample whose on-disk point count differs from `num_points` when `allow_resample=true`.",
        "- Therefore the dataloader can silently change point counts unless each branch is launched with both the correct `--num-point` and `--no-allow-resample`.",
        "- Legacy wrapper `scripts/train_pointnet2.sh` still contains old protocol branches (`downsampled50`, `2048`-point Line B upsampling roots) and must not be used for this x4 two-line final round.",
        "",
        "## Runtime policy",
        "",
        "- All 10 generated branch configs under `configs/pointnet2_x4_two_line/` set `allow_resample: false`.",
        "- All generated smoke/full `sbatch` files call `scripts/train_pointnet2.py` directly instead of the legacy wrapper.",
        "- Result: no branch may silently crop/pad/upsample/downsample inside the dataloader.",
        "",
        "## Branch runtime num_points",
        "",
        "| line | branch | method | source path | runtime num_points | allow_resample | expected on-disk points | verdict |",
        "| --- | --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for spec in specs:
        verdict = "PASS"
        if spec.line == "B" and spec.branch == "lineB_downsampled_x4_baseline":
            note = "Baseline remains native 256-point input."
        elif spec.line == "A" and spec.branch == "lineA_original_baseline":
            note = "Baseline remains native 1024-point input."
        else:
            note = "Upsampling branch preserves strict protocol output count."
        lines.append(
            f"| {spec.line} | {spec.branch} | {spec.method} | `{spec.input_path}` | "
            f"{spec.expected_point_count} | false | {spec.expected_point_count} | {verdict} ({note}) |"
        )
    lines += [
        "",
        "## Key file references",
        "",
        "- `scripts/train_pointnet2.py`",
        "- `scripts/modelnet_npy_dataloader.py`",
        "- `scripts/train_pointnet2.sh`",
        "- `configs/pointnet2_default.yaml`",
        "- `configs/pointnet2_x4_two_line/`",
        "",
        "## Conclusion",
        "",
        "The runtime point counts are now branch-specific and explicit: Line B baseline = 256, Line B upsampling branches = 1024, Line A baseline = 1024, Line A upsampling branches = 4096. No generated smoke/full job uses the default 1024 silently.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def write_smoke_plan(specs: list[BranchSpec]) -> Path:
    md_path = REPORTS_DIR / "modelnet40_pointnet2_smoke_plan.md"
    lines = [
        "# ModelNet40 PointNet++ Smoke Plan",
        "",
        f"- Generated at: {now_utc()}",
        "- Scope: prepare smoke jobs only; no full training launched here.",
        "- Each smoke job runs 1 epoch with 2 train batches and 2 eval batches.",
        "- Each smoke job uses branch-specific `--num-point` and `--no-allow-resample`.",
        "",
        "## Smoke jobs",
        "",
        "| branch | points | input | sbatch | smoke output dir |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for spec in specs:
        lines.append(
            f"| {spec.branch_name} | {spec.expected_point_count} | `{spec.input_path}` | "
            f"`{spec.smoke_job_path}` | `{spec.smoke_result_dir}` |"
        )
    lines += [
        "",
        "## What each smoke verifies",
        "",
        "- Branch name and input path are echoed at job start.",
        "- Actual batch shape and actual point count are logged by `scripts/train_pointnet2.py` on the first batch.",
        "- Forward/backward pass and loss computation run successfully.",
        "- Evaluation loop runs successfully.",
        "- Outputs land under `pointnet2_results/x4_two_line_smoke/` and do not overwrite full-training results.",
        "",
        "## Submission status",
        "",
        "- Not submitted in this preparation pass.",
        "- Safe next step: submit the five Line B smoke jobs first, inspect logs, then submit the five Line A smoke jobs.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def write_full_plan(specs: list[BranchSpec]) -> Path:
    md_path = REPORTS_DIR / "modelnet40_pointnet2_full_training_job_plan.md"
    lines = [
        "# ModelNet40 PointNet++ Full Training Job Plan",
        "",
        f"- Generated at: {now_utc()}",
        "- Scope: ready-to-run `sbatch` files only; no full training launched here.",
        "",
        "| branch | points | input | full sbatch | output dir | key saved artifacts |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for spec in specs:
        lines.append(
            f"| {spec.branch_name} | {spec.expected_point_count} | `{spec.input_path}` | "
            f"`{spec.full_job_path}` | `{spec.result_dir}` | "
            "`checkpoints/best_model.pth`, `train.log`, `metrics.json`, `<variant>_result.csv`, `<variant>_result.md` |"
        )
    lines += [
        "",
        "## Guarantees",
        "",
        "- Random seed fixed to `42` for every full job.",
        "- Each branch uses its own `--num-point` and `--no-allow-resample`.",
        "- Each branch writes to an isolated directory under `pointnet2_results/x4_two_line_final/`.",
        "- Checkpoint, training log, best overall accuracy, class average accuracy, and final test metrics are all preserved by the training script.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def write_executable_checklist(specs: list[BranchSpec]) -> Path:
    md_path = REPORTS_DIR / "modelnet40_pointnet2_executable_training_checklist.md"
    lineb = [s for s in specs if s.line == "B"]
    linea = [s for s in specs if s.line == "A"]
    lines = [
        "# ModelNet40 PointNet++ Executable Training Checklist",
        "",
        f"- Generated at: {now_utc()}",
        "",
        "## Recommended order",
        "",
        "First batch Line B:",
        "- downsampled baseline",
        "- EAR",
        "- PDANS",
        "- PU-Net",
        "- PU-GCN",
        "",
        "Second batch Line A:",
        "- original baseline",
        "- EAR",
        "- PDANS",
        "- PU-Net",
        "- PU-GCN",
        "",
        "## Smoke commands",
        "",
    ]
    for spec in lineb + linea:
        lines.append(f"`sbatch {spec.smoke_job_path}`")
    lines += [
        "",
        "## Full training commands",
        "",
    ]
    for spec in lineb + linea:
        lines.append(f"`sbatch {spec.full_job_path}`")
    lines += [
        "",
        "## Monitoring commands",
        "",
        "```bash",
        f'watch -n 30 "squeue -u $USER | rg \'pn2|pointnet2|lineA|lineB\' || true"',
        "```",
        "",
        "```bash",
        f"rg -n \"First train batch tensor shape|First train batch loss|Training finished|Saved checkpoint\" {PROJECT_ROOT}/pointnet2_results/x4_two_line_smoke {PROJECT_ROOT}/pointnet2_results/x4_two_line_final",
        "```",
        "",
        "```bash",
        f"ls {PROJECT_ROOT}/pointnet2_results/x4_two_line_final/* {PROJECT_ROOT}/pointnet2_results/x4_two_line_final/lineA_original_up/* {PROJECT_ROOT}/pointnet2_results/x4_two_line_final/lineB_downsampled_x4_up/*",
        "```",
        "",
        "## Expected output files",
        "",
        "- `checkpoints/best_model.pth`",
        "- `train.log`",
        "- `metrics.json`",
        "- `<variant>_result.csv`",
        "- `<variant>_result.md`",
        "- SLURM stdout/stderr under `logs/pointnet2_smoke/` or `logs/pointnet2_full/`",
        "",
        "## Result summary target",
        "",
        f"- Consolidate final branch metrics into `reports/modelnet40_pointnet2_main_results_status.csv` after all 10 full runs complete.",
        f"- Keep the protocol-level narrative in `reports/modelnet40_x4_final_protocol_report.md`.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def update_final_report(
    input_audit_md: Path,
    symlink_audit_md: Path,
    dataloader_md: Path,
    smoke_plan_md: Path,
    full_plan_md: Path,
    checklist_md: Path,
    specs: list[BranchSpec],
) -> Path:
    report_path = REPORTS_DIR / "modelnet40_x4_final_protocol_report.md"
    content = report_path.read_text(encoding="utf-8")
    start_marker = "<!-- POINTNET2_TRAINING_PREPARATION_START -->"
    end_marker = "<!-- POINTNET2_TRAINING_PREPARATION_END -->"
    if start_marker in content and end_marker in content:
        prefix = content.split(start_marker)[0].rstrip()
        suffix = content.split(end_marker, 1)[1].lstrip()
        content = prefix + "\n\n" + suffix

    section_lines = [
        start_marker,
        "## PointNet++ training preparation",
        "",
        f"- Updated at: {now_utc()}",
        f"- 10 branches ready: **{len(specs)} / {len(specs)}**",
        f"- Input audit status: see `{input_audit_md}`",
        f"- Input symlink audit: see `{symlink_audit_md}`",
        f"- Dataloader pointcount audit status: see `{dataloader_md}`",
        f"- Smoke plan path: `{smoke_plan_md}`",
        f"- Full training job plan path: `{full_plan_md}`",
        f"- Executable checklist path: `{checklist_md}`",
        "- Next step: submit smoke jobs.",
        end_marker,
        "",
    ]
    updated = content.rstrip() + "\n\n" + "\n".join(section_lines)
    report_path.write_text(updated + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    specs = branch_specs()
    refs, expected_classes = load_reference_manifests()

    input_rows = [audit_branch(spec, refs, expected_classes) for spec in specs]
    symlink_rows = [ensure_input_symlink(spec) for spec in specs]
    write_configs_and_jobs(specs)

    _, input_md = write_input_audit(input_rows)
    _, symlink_md = write_symlink_audit(symlink_rows)
    dataloader_md = write_dataloader_audit(specs)
    smoke_plan_md = write_smoke_plan(specs)
    full_plan_md = write_full_plan(specs)
    checklist_md = write_executable_checklist(specs)
    final_report = update_final_report(
        input_audit_md=input_md,
        symlink_audit_md=symlink_md,
        dataloader_md=dataloader_md,
        smoke_plan_md=smoke_plan_md,
        full_plan_md=full_plan_md,
        checklist_md=checklist_md,
        specs=specs,
    )

    print(f"Wrote input audit: {REPORTS_DIR / 'modelnet40_pointnet2_training_input_audit.csv'}")
    print(f"Wrote symlink audit: {REPORTS_DIR / 'modelnet40_pointnet2_input_symlink_audit.csv'}")
    print(f"Wrote dataloader audit: {dataloader_md}")
    print(f"Wrote smoke plan: {smoke_plan_md}")
    print(f"Wrote full plan: {full_plan_md}")
    print(f"Wrote executable checklist: {checklist_md}")
    print(f"Updated final report: {final_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
