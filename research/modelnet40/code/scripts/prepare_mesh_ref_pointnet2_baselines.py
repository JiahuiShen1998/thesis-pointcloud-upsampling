#!/usr/bin/env python3
"""Prepare PointNet++ training for mesh-sampled equal-N baselines (256 / 4096).

Adds label maps + labeled manifests, symlinks pointnet2_inputs, writes configs/jobs.
"""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_META = PROJECT_ROOT / "datasets" / "modelnet40_original" / "metadata"
REPORTS = PROJECT_ROOT / "reports"
POINTNET2_INPUTS = PROJECT_ROOT / "pointnet2_inputs"
CONFIGS = PROJECT_ROOT / "configs" / "pointnet2_x4_two_line"
JOBS = PROJECT_ROOT / "jobs" / "pointnet2_full"
RESULTS = PROJECT_ROOT / "pointnet2_results" / "x4_two_line_final"

BRANCHES = (
    {
        "name": "mesh_ref_baseline_256",
        "num_points": 256,
        "source": PROJECT_ROOT / "datasets" / "modelnet40_mesh_ref_256",
        "input": POINTNET2_INPUTS / "mesh_ref_baseline_256",
        "result": RESULTS / "mesh_ref_baseline_256",
        "slug": "mesh_ref_baseline_256",
    },
    {
        "name": "mesh_ref_baseline_4096",
        "num_points": 4096,
        "source": PROJECT_ROOT / "datasets" / "modelnet40_mesh_ref_4096",
        "input": POINTNET2_INPUTS / "mesh_ref_baseline_4096",
        "result": RESULTS / "mesh_ref_baseline_4096",
        "slug": "mesh_ref_baseline_4096",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def enrich_metadata(source: Path) -> None:
    meta = source / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    for name in ("class_to_idx.json", "idx_to_class.json"):
        src = ORIGINAL_META / name
        dst = meta / name
        shutil.copy2(src, dst)

    class_to_idx = json.loads((meta / "class_to_idx.json").read_text(encoding="utf-8"))
    for split in ("train", "test"):
        in_path = meta / f"{split}_manifest.csv"
        rows = []
        with open(in_path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                class_name = row["class_name"]
                row["label"] = str(class_to_idx[class_name])
                # Prefer relative layout under data_root for robustness
                row["output_npy"] = str(source / split / class_name / f"{row['shape_id']}.npy")
                rows.append(row)
        fields = [
            "shape_id",
            "class_name",
            "label",
            "split",
            "source_off",
            "output_npy",
            "num_points",
            "seed",
            "status",
        ]
        with open(in_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def ensure_symlink(source: Path, input_path: Path) -> None:
    input_path.parent.mkdir(parents=True, exist_ok=True)
    if input_path.is_symlink() or input_path.exists():
        if input_path.resolve() == source.resolve():
            return
        if input_path.is_symlink() or input_path.is_file():
            input_path.unlink()
        else:
            raise RuntimeError(f"Refusing to replace non-symlink path: {input_path}")
    input_path.symlink_to(source.resolve())


def write_config(branch: dict) -> Path:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    path = CONFIGS / f"{branch['name']}_{branch['num_points']}.yaml"
    text = f"""# Auto-generated mesh-ref PointNet++ baseline
experiment_name: {branch['name']}
line: mesh_ref
branch: {branch['name']}
method: mesh_ref_baseline
data_root: {branch['input'].relative_to(PROJECT_ROOT)}
expected_point_count: {branch['num_points']}
num_point: {branch['num_points']}
allow_resample: false
model: pointnet2_cls_ssg
num_category: 40
training:
  epoch: 200
  batch_size: 24
  learning_rate: 0.001
  decay_rate: 0.0001
  optimizer: Adam
  seed: 42
  num_workers: 4
output_dir: {branch['result'].relative_to(PROJECT_ROOT)}
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_sbatch(branch: dict, partition: str = "v100") -> Path:
    JOBS.mkdir(parents=True, exist_ok=True)
    slug = branch["slug"]
    path = JOBS / f"run_{slug}.sbatch"
    n = branch["num_points"]
    text = f"""#!/bin/bash
#SBATCH --job-name=pn2_{slug}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output={PROJECT_ROOT}/logs/pointnet2_full/{slug}/slurm_%j.out
#SBATCH --error={PROJECT_ROOT}/logs/pointnet2_full/{slug}/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="{PROJECT_ROOT}"
BRANCH_NAME="{branch['name']}"
INPUT_PATH="{branch['input']}"
OUTPUT_DIR="{branch['result']}"
LOG_FILE="${{OUTPUT_DIR}}/train.log"

mkdir -p "${{OUTPUT_DIR}}" "{PROJECT_ROOT}/logs/pointnet2_full/{slug}"

echo "=== PointNet++ mesh-ref baseline ==="
echo "branch=${{BRANCH_NAME}}"
echo "input_path=${{INPUT_PATH}}"
echo "expected_point_count={n}"
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
  --num-point {n} \\
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
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def main() -> None:
    lines = [
        "# Mesh-ref PointNet++ baseline prep",
        "",
        f"- Generated: {utc_now()}",
        "",
        "| branch | points | source | input symlink | config | sbatch |",
        "|---|---:|---|---|---|---|",
    ]
    for branch in BRANCHES:
        if not branch["source"].exists():
            raise SystemExit(f"Missing source: {branch['source']}")
        enrich_metadata(branch["source"])
        ensure_symlink(branch["source"], branch["input"])
        cfg = write_config(branch)
        job = write_sbatch(branch, partition="v100")
        lines.append(
            f"| {branch['name']} | {branch['num_points']} | `{branch['source']}` | "
            f"`{branch['input']}` | `{cfg}` | `{job}` |"
        )
        print(f"prepared {branch['name']}: input={branch['input']} job={job}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "modelnet40_pointnet2_mesh_ref_baseline_prep.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
