#!/usr/bin/env python3
"""Build mesh-sampled ModelNet40 references at equal cardinalities (256, 4096).

Reuses area-weighted surface sampling + unit-sphere normalization from
prepare_modelnet40.py. Aligns to modelnet40_original manifests via source_off.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from modelnet40_x4_protocol import GLOBAL_SEED, stable_seed  # noqa: E402
from prepare_modelnet40 import (  # noqa: E402
    load_off_mesh,
    normalize_unit_sphere,
    sample_points_from_mesh,
)

ORIGINAL_META = PROJECT_ROOT / "datasets" / "modelnet40_original" / "metadata"
REPORTS = PROJECT_ROOT / "reports"
DEFAULT_COUNTS = (256, 4096)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def output_root(num_points: int) -> Path:
    return PROJECT_ROOT / "datasets" / f"modelnet40_mesh_ref_{num_points}"


def load_manifest_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "test"):
        path = ORIGINAL_META / f"{split}_manifest.csv"
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "split": row["split"],
                        "class_name": row.get("class_name") or row.get("class") or "",
                        "shape_id": row["shape_id"],
                        "source_off": row["source_off"],
                    }
                )
    return rows


def build_one(args: tuple[Any, ...]) -> dict[str, Any]:
    split, class_name, shape_id, source_off, num_points, resume = args
    out_dir = output_root(num_points)
    out_path = out_dir / split / class_name / f"{shape_id}.npy"
    key = f"{split}/{class_name}/{shape_id}"
    seed = stable_seed(GLOBAL_SEED, "mesh_ref", str(num_points), key)
    try:
        if resume and out_path.is_file():
            arr = np.load(out_path)
            if arr.shape == (num_points, 3) and np.isfinite(arr).all():
                return {
                    "status": "reused",
                    "key": key,
                    "num_points": num_points,
                    "out_path": str(out_path),
                    "seed": seed,
                }
        mesh = load_off_mesh(source_off)
        points = sample_points_from_mesh(mesh, num_points, seed)
        points = normalize_unit_sphere(points)
        if points.shape != (num_points, 3) or not np.isfinite(points).all():
            raise ValueError(f"invalid points shape={points.shape}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, points.astype(np.float32))
        return {
            "status": "created",
            "key": key,
            "num_points": num_points,
            "out_path": str(out_path),
            "seed": seed,
            "num_vertices": int(mesh.vertices.shape[0]),
            "num_faces": int(mesh.faces.shape[0]),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "key": key,
            "num_points": num_points,
            "out_path": str(out_path),
            "seed": seed,
            "error": str(exc),
        }


def write_split_manifests(num_points: int, rows: list[dict[str, str]], results_by_key: dict[str, dict]) -> None:
    meta = output_root(num_points) / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    fields = [
        "shape_id",
        "class_name",
        "split",
        "source_off",
        "output_npy",
        "num_points",
        "seed",
        "status",
    ]
    for split in ("train", "test"):
        path = meta / f"{split}_manifest.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                if row["split"] != split:
                    continue
                key = f"{row['split']}/{row['class_name']}/{row['shape_id']}"
                res = results_by_key.get(key, {})
                writer.writerow(
                    {
                        "shape_id": row["shape_id"],
                        "class_name": row["class_name"],
                        "split": split,
                        "source_off": row["source_off"],
                        "output_npy": str(
                            output_root(num_points) / split / row["class_name"] / f"{row['shape_id']}.npy"
                        ),
                        "num_points": num_points,
                        "seed": res.get("seed", ""),
                        "status": res.get("status", "missing"),
                    }
                )


def write_audit(num_points: int, results: list[dict[str, Any]]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = {s: sum(1 for r in results if r["status"] == s) for s in ("created", "reused", "failed")}
    csv_path = REPORTS / f"modelnet40_mesh_ref_{num_points}_build_audit.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fields = ["key", "num_points", "status", "out_path", "seed", "error"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["key"]):
            writer.writerow(r)

    md_path = REPORTS / f"modelnet40_mesh_ref_{num_points}_build_audit.md"
    md_path.write_text(
        "\n".join(
            [
                f"# Mesh reference build audit — {num_points} points",
                "",
                f"- Generated: {utc_now()}",
                f"- Output: `datasets/modelnet40_mesh_ref_{num_points}/`",
                f"- Sampling: area-weighted from `.off` + unit-sphere normalize (prepare_modelnet40)",
                f"- Seed: `stable_seed(42, 'mesh_ref', '{num_points}', '{{split}}/{{class}}/{{shape_id}}')`",
                f"- Samples: {len(results)}",
                f"- created={summary.get('created', 0)} reused={summary.get('reused', 0)} failed={summary.get('failed', 0)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    manifest_json = output_root(num_points) / "metadata" / "mesh_ref_manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "num_points": num_points,
                "global_seed": GLOBAL_SEED,
                "generated_at": utc_now(),
                "samples": len(results),
                "results_summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_for_count(num_points: int, rows: list[dict[str, str]], workers: int, resume: bool, limit: int) -> list[dict[str, Any]]:
    use_rows = rows[:limit] if limit > 0 else rows
    tasks = [
        (r["split"], r["class_name"], r["shape_id"], r["source_off"], num_points, resume)
        for r in use_rows
    ]
    results: list[dict[str, Any]] = []
    print(f"[{utc_now()}] building mesh_ref_{num_points}: {len(tasks)} shapes, workers={workers}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(build_one, t) for t in tasks]
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 500 == 0 or done == len(tasks):
                print(f"  [{num_points}] {done}/{len(tasks)}")
    by_key = {r["key"]: r for r in results}
    # When limit>0, still write manifests only for processed rows
    write_split_manifests(num_points, use_rows, by_key)
    write_audit(num_points, results)
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"[{utc_now()}] mesh_ref_{num_points} done; failed={failed}")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-points", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--limit", type=int, default=0, help="0 = all shapes; >0 for smoke")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest_rows()
    print(f"[{utc_now()}] loaded {len(rows)} manifest rows")
    all_failed = 0
    for n in args.num_points:
        results = run_for_count(n, rows, args.workers, args.resume, args.limit)
        all_failed += sum(1 for r in results if r["status"] == "failed")
    if all_failed:
        raise SystemExit(f"Completed with {all_failed} failures")
    print(f"[{utc_now()}] ALL OK")


if __name__ == "__main__":
    main()
