#!/usr/bin/env python3
"""Merge EAR Line B chunk audits and write full dataset manifests + report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
DEFAULT_CHUNK_AUDIT_DIR = PROJECT_ROOT / "logs" / "ear_full_lineB" / "chunk_audits"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50_up" / "ear"
DEFAULT_AUDIT = PROJECT_ROOT / "reports" / "step7b_ear_full_lineB_audit.csv"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "step7b_ear_full_lineB_report.md"
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50"

EXPECTED_TRAIN = 9843
EXPECTED_TEST = 2468
EXPECTED_TOTAL = EXPECTED_TRAIN + EXPECTED_TEST


def load_chunk_audits(chunk_audit_dir: Path) -> list[dict]:
    rows: list[dict] = []
    audit_files = sorted(chunk_audit_dir.glob("chunk_*_audit.csv"))
    if not audit_files:
        raise FileNotFoundError(f"No chunk audit CSV files in {chunk_audit_dir}")
    for path in audit_files:
        with open(path, newline="", encoding="utf-8") as handle:
            rows.extend(list(csv.DictReader(handle)))
    return rows


def write_split_manifests(output_root: Path, audit_rows: list[dict]) -> None:
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    ok_rows = [r for r in audit_rows if r["status"] in ("success", "skipped_existing")]
    for split in ("train", "test"):
        manifest_path = metadata_dir / f"{split}_manifest.csv"
        split_rows = [r for r in ok_rows if r["split"] == split]
        with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["split", "class_name", "label", "shape_id", "output_npy"],
            )
            writer.writeheader()
            for row in sorted(split_rows, key=lambda r: (r["class_name"], r["shape_id"])):
                writer.writerow(
                    {
                        "split": row["split"],
                        "class_name": row["class_name"],
                        "label": row["label"],
                        "shape_id": row["shape_id"],
                        "output_npy": row["final_output_path"],
                    }
                )


def count_output_files(output_root: Path) -> dict:
    counts = {}
    for split in ("train", "test"):
        split_dir = output_root / split
        counts[split] = len(list(split_dir.rglob("*.npy"))) if split_dir.is_dir() else 0
    counts["total"] = counts["train"] + counts["test"]
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge EAR Line B chunk audits.")
    parser.add_argument("--chunk-audit-dir", type=Path, default=DEFAULT_CHUNK_AUDIT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--array-job-id", type=str, default="")
    args = parser.parse_args()

    audit_rows = load_chunk_audits(args.chunk_audit_dir)
    fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    args.audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.audit_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    write_split_manifests(args.output_root, audit_rows)

    status_counts = Counter(r["status"] for r in audit_rows)
    file_counts = count_output_files(args.output_root)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    complete = (
        file_counts["train"] == EXPECTED_TRAIN
        and file_counts["test"] == EXPECTED_TEST
        and status_counts.get("failed", 0) == 0
        and len(audit_rows) == EXPECTED_TOTAL
    )

    lines = [
        "# Step 7b — EAR Full Line B Upsampling Report",
        "",
        f"- Generated at: {now}",
        f"- Status: **{'COMPLETE' if complete else 'IN PROGRESS OR FAILED'}**",
        "",
        "## Scope",
        "",
        "- Line: **Downsampled50 → EAR → 1024** (Line B only)",
        "- Line A (Original + EAR) **skipped** — identity no-op at 1024→1024",
        "",
        f"- Input root: `{args.input_root}`",
        f"- Output root: `{args.output_root}`",
        f"- EAR script: `/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_experiments/external_lab_migrated/EAR/ear_upsampling.py`",
        "",
        "## Job Info",
        "",
        f"- Slurm array job ID: `{args.array_job_id or 'N/A'}`",
        f"- Chunk audit dir: `{args.chunk_audit_dir}`",
        "",
        "## Progress",
        "",
        f"- Audit rows: {len(audit_rows)} (expected {EXPECTED_TOTAL})",
        f"- Status counts: {dict(status_counts)}",
        f"- Output files train/test/total: {file_counts['train']}/{file_counts['test']}/{file_counts['total']}",
        f"- Expected train/test/total: {EXPECTED_TRAIN}/{EXPECTED_TEST}/{EXPECTED_TOTAL}",
        "",
        "## Artifacts",
        "",
        f"- Full audit CSV: `{args.audit_path}`",
        f"- Output manifests: `{args.output_root / 'metadata'}`",
        "",
        "## Next Step",
        "",
        f"- Ready for Step 8 (Downsampled50+EAR → PointNet++): **{'Yes' if complete else 'No — wait for jobs / fix failures'}**",
    ]
    args.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Merged {len(audit_rows)} audit rows -> {args.audit_path}")
    print(f"Output files: {file_counts}")
    print(f"Report: {args.report_path}")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
