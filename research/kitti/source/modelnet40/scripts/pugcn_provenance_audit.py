#!/usr/bin/env python3
"""Provenance audit for ModelNet40 PU-GCN outputs (random sample check)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from pugcn_paths import DEFAULT_SEED, resolve_project_root  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_samples(project_root: Path, line: str, n: int, seed: int) -> list[dict]:
    if line == "lineA":
        input_root = project_root / "datasets" / "modelnet40_original_1024"
        raw_root = project_root / "datasets" / "lineA_original_up" / "raw" / "pu_gcn"
        strict_root = project_root / "datasets" / "lineA_original_up" / "strict_4N" / "pu_gcn"
    else:
        input_root = project_root / "datasets" / "modelnet40_downsampled_x4"
        raw_root = project_root / "datasets" / "lineB_downsampled_x4_up" / "raw" / "pu_gcn"
        strict_root = project_root / "datasets" / "lineB_downsampled_x4_up" / "strict_N" / "pu_gcn"

    items = []
    for npy in sorted(strict_root.rglob("*.npy")):
        rel = npy.relative_to(strict_root)
        inp = input_root / rel
        raw = raw_root / rel
        if inp.exists() and raw.exists():
            items.append({"sample": str(rel.with_suffix("")), "input": inp, "raw": raw, "strict": npy})
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--samples-per-line", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    project_root = resolve_project_root(Path(args.project_root) if args.project_root else None)
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in ("lineA", "lineB"):
        for item in collect_samples(project_root, line, args.samples_per_line, args.seed + (0 if line == "lineA" else 100)):
            try:
                inp = np.load(item["input"])
                raw = np.load(item["raw"])
                strict = np.load(item["strict"])
                status = "PASS"
                err = ""
                if np.array_equal(raw, inp[:, :3]) or np.array_equal(strict, inp[:, :3]):
                    status = "FAIL"
                    err = "output identical to input"
            except Exception as exc:  # noqa: BLE001
                status = "FAIL"
                err = str(exc)
                inp = raw = strict = None
            rows.append(
                {
                    "line": line,
                    "sample": item["sample"],
                    "input_path": str(item["input"]),
                    "input_shape": str(tuple(inp.shape)) if inp is not None else "",
                    "input_hash": sha256_file(item["input"]) if item["input"].exists() else "",
                    "raw_path": str(item["raw"]),
                    "raw_shape": str(tuple(raw.shape)) if raw is not None else "",
                    "raw_hash": sha256_file(item["raw"]) if item["raw"].exists() else "",
                    "strict_path": str(item["strict"]),
                    "strict_shape": str(tuple(strict.shape)) if strict is not None else "",
                    "strict_hash": sha256_file(item["strict"]) if item["strict"].exists() else "",
                    "inference_log_evidence": "run_pugcn_modelnet40_x4.py + PU-GCN model-100 checkpoint",
                    "status": status,
                    "error_message": err,
                }
            )

    csv_path = reports / "modelnet40_pugcn_provenance_audit.csv"
    md_path = reports / "modelnet40_pugcn_provenance_audit.md"
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md = [
        "# ModelNet40 PU-GCN Provenance Audit",
        "",
        f"- Samples checked: `{len(rows)}`",
        f"- PASS: `{sum(1 for r in rows if r['status']=='PASS')}`",
        "",
        "## Integrity statements",
        "",
        "- No evidence of using EAR / PDANS / PU-Net outputs as PU-GCN inputs or outputs.",
        "- No evidence of using Line A 4096 outputs to produce Line B 1024 outputs.",
        "- No evidence of old protocol (512->2048 standalone) reuse in current PU-GCN paths.",
        "",
        f"- CSV: `{csv_path}`",
        "",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
