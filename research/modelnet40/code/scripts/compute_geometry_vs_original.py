#!/usr/bin/env python3
"""Recompute ModelNet40 geometry metrics vs corresponding Original point clouds.

Protocol:
  Each upsampled / downsampled point cloud is evaluated directly against the
  corresponding Original 1024-point cloud. No mesh or dense surface reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_original"
DOWNSAMPLED_ROOT = PROJECT_ROOT / "datasets" / "modelnet40_downsampled_x4"
LINEA_STRICT = PROJECT_ROOT / "datasets" / "lineA_original_up" / "strict_4N"
LINEB_STRICT = PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up" / "strict_N"
REPORTS = PROJECT_ROOT / "reports"

GLOBAL_SEED = 42
NUC_QUERY_CENTERS = 128
NUC_RADII = (0.02, 0.05, 0.10)

METHODS = ("ear", "pdans", "pu_net", "pu_gcn", "pu_edgeformer")
METHOD_DISPLAY = {
    "original": "Original",
    "downsampled_x4": "Downsampled ×4",
    "ear": "EAR",
    "pdans": "PDANS",
    "pu_net": "PU-Net",
    "pu_gcn": "PU-GCN",
    "pu_edgeformer": "PU-EdgeFormer",
}


@dataclass(frozen=True)
class EvalGroup:
    line: str
    method_key: str
    display: str
    point_count: int
    root: Path | None  # None => identity (Original vs Original)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def nn_distances(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    tree = cKDTree(target)
    dists, _ = tree.query(source, k=1, workers=-1)
    return np.asarray(dists, dtype=np.float64)


def chamfer_hd(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    fwd = nn_distances(a, b)
    bwd = nn_distances(b, a)
    return {
        "cd_forward": float(np.mean(fwd)),
        "cd_backward": float(np.mean(bwd)),
        "cd": float(np.mean(fwd) + np.mean(bwd)),
        "hd_forward": float(np.max(fwd)),
        "hd_backward": float(np.max(bwd)),
        "hd": float(max(np.max(fwd), np.max(bwd))),
    }


def compute_nuc(points: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    k = min(NUC_QUERY_CENTERS, n)
    centers = points[rng.choice(n, size=k, replace=False)]
    nuc_vals: list[float] = []
    out: dict[str, float] = {}
    tree = cKDTree(points)
    for ri, radius in enumerate(NUC_RADII, start=1):
        counts = []
        for c in centers:
            # subtract self
            cnt = int(tree.query_ball_point(c, r=radius, return_length=True)) - 1
            counts.append(max(cnt, 0))
        arr = np.asarray(counts, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = std / mean if mean > 1e-8 else 0.0
        out[f"nuc_r{ri}"] = cv
        nuc_vals.append(cv)
    out["nuc"] = float(np.mean(nuc_vals))
    return out


def load_manifest(split: str) -> list[tuple[str, str, str]]:
    path = ORIGINAL_ROOT / "metadata" / f"{split}_manifest.csv"
    rows: list[tuple[str, str, str]] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            class_name = row.get("class_name") or row.get("class")
            rows.append((split, class_name, row["shape_id"]))
    return rows


def sample_path(root: Path, split: str, class_name: str, shape_id: str) -> Path:
    return root / split / class_name / f"{shape_id}.npy"


def eval_one(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        line,
        method_key,
        display,
        expected_n,
        root_str,
        split,
        class_name,
        shape_id,
        compute_nuc_flag,
    ) = args
    original = np.load(sample_path(ORIGINAL_ROOT, split, class_name, shape_id)).astype(np.float64)
    if original.shape != (1024, 3):
        return {
            "ok": False,
            "reason": f"bad original shape {original.shape}",
            "line": line,
            "method": method_key,
            "split": split,
            "class": class_name,
            "shape_id": shape_id,
        }

    if root_str is None:
        evaluated = original
    else:
        path = sample_path(Path(root_str), split, class_name, shape_id)
        if not path.exists():
            return {
                "ok": False,
                "reason": f"missing file {path}",
                "line": line,
                "method": method_key,
                "split": split,
                "class": class_name,
                "shape_id": shape_id,
            }
        evaluated = np.load(path).astype(np.float64)

    if evaluated.ndim != 2 or evaluated.shape[1] != 3:
        return {
            "ok": False,
            "reason": f"bad eval shape {evaluated.shape}",
            "line": line,
            "method": method_key,
            "split": split,
            "class": class_name,
            "shape_id": shape_id,
        }
    if evaluated.shape[0] != expected_n:
        return {
            "ok": False,
            "reason": f"point count {evaluated.shape[0]} != expected {expected_n}",
            "line": line,
            "method": method_key,
            "split": split,
            "class": class_name,
            "shape_id": shape_id,
        }

    if method_key == "original":
        metrics = {
            "cd_forward": 0.0,
            "cd_backward": 0.0,
            "cd": 0.0,
            "hd_forward": 0.0,
            "hd_backward": 0.0,
            "hd": 0.0,
        }
    else:
        metrics = chamfer_hd(evaluated, original)

    seed = GLOBAL_SEED + abs(hash(f"{split}:{class_name}:{shape_id}:{method_key}")) % 1_000_000
    if compute_nuc_flag:
        nuc = compute_nuc(evaluated, seed)
    else:
        nuc = {"nuc": float("nan"), "nuc_r1": float("nan"), "nuc_r2": float("nan"), "nuc_r3": float("nan")}

    return {
        "ok": True,
        "line": line,
        "method": method_key,
        "display": display,
        "point_count": int(evaluated.shape[0]),
        "reference": "Original 1024",
        "split": split,
        "class": class_name,
        "shape_id": shape_id,
        **metrics,
        **nuc,
    }


def define_groups() -> list[EvalGroup]:
    return [
        EvalGroup("A", "original", "Original", 1024, None),
        EvalGroup("A", "ear", "EAR", 4096, LINEA_STRICT / "ear"),
        EvalGroup("A", "pdans", "PDANS", 4096, LINEA_STRICT / "pdans"),
        EvalGroup("A", "pu_net", "PU-Net", 4096, LINEA_STRICT / "pu_net"),
        EvalGroup("A", "pu_gcn", "PU-GCN", 4096, LINEA_STRICT / "pu_gcn"),
        EvalGroup("A", "pu_edgeformer", "PU-EdgeFormer", 4096, LINEA_STRICT / "pu_edgeformer"),
        EvalGroup("B", "original", "Original", 1024, None),
        EvalGroup("B", "downsampled_x4", "Downsampled ×4", 256, DOWNSAMPLED_ROOT),
        EvalGroup("B", "ear", "EAR", 1024, LINEB_STRICT / "ear"),
        EvalGroup("B", "pdans", "PDANS", 1024, LINEB_STRICT / "pdans"),
        EvalGroup("B", "pu_net", "PU-Net", 1024, LINEB_STRICT / "pu_net"),
        EvalGroup("B", "pu_gcn", "PU-GCN", 1024, LINEB_STRICT / "pu_gcn"),
        EvalGroup("B", "pu_edgeformer", "PU-EdgeFormer", 1024, LINEB_STRICT / "pu_edgeformer"),
    ]


def summarize(rows: list[dict[str, Any]], original_nuc_by_key: dict[str, float]) -> list[dict[str, Any]]:
    from collections import defaultdict

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("ok"):
            buckets[(r["line"], r["method"])].append(r)

    out: list[dict[str, Any]] = []
    for group in define_groups():
        key = (group.line, group.method_key)
        items = buckets.get(key, [])
        if not items:
            out.append(
                {
                    "line": group.line,
                    "method": group.display,
                    "method_key": group.method_key,
                    "points": group.point_count,
                    "reference": "Original 1024",
                    "cd_vs_original": "Pending recomputation",
                    "hd_vs_original": "Pending recomputation",
                    "nuc": "Pending recomputation",
                    "delta_nuc_vs_original": "Pending recomputation",
                    "valid_samples": 0,
                    "missing_or_invalid": "all",
                }
            )
            continue

        cds = np.array([x["cd"] for x in items], dtype=np.float64)
        hds = np.array([x["hd"] for x in items], dtype=np.float64)
        nucs = np.array([x["nuc"] for x in items], dtype=np.float64)
        # ΔNUC vs paired Original NUC for same sample
        deltas = []
        for x in items:
            ok_key = f"{x['split']}/{x['class']}/{x['shape_id']}"
            base = original_nuc_by_key.get(ok_key)
            if base is not None and np.isfinite(x["nuc"]):
                deltas.append(x["nuc"] - base)
        delta_arr = np.asarray(deltas, dtype=np.float64) if deltas else np.array([0.0])

        out.append(
            {
                "line": group.line,
                "method": group.display,
                "method_key": group.method_key,
                "points": group.point_count,
                "reference": "Original 1024",
                "cd_vs_original": float(np.mean(cds)),
                "cd_std": float(np.std(cds)),
                "hd_vs_original": float(np.mean(hds)),
                "hd_std": float(np.std(hds)),
                "nuc": float(np.mean(nucs)),
                "nuc_std": float(np.std(nucs)),
                "delta_nuc_vs_original": 0.0 if group.method_key == "original" else float(np.mean(delta_arr)),
                "valid_samples": len(items),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="test", help="Comma-separated: test,train")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--no-nuc", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit samples per split (debug)")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    samples: list[tuple[str, str, str]] = []
    for split in splits:
        rows = load_manifest(split)
        if args.limit > 0:
            rows = rows[: args.limit]
        samples.extend(rows)

    groups = define_groups()
    jobs: list[tuple[Any, ...]] = []
    for g in groups:
        root_str = None if g.root is None else str(g.root)
        for split, class_name, shape_id in samples:
            jobs.append(
                (
                    g.line,
                    g.method_key,
                    g.display,
                    g.point_count,
                    root_str,
                    split,
                    class_name,
                    shape_id,
                    not args.no_nuc,
                )
            )

    print(f"[{utc_now()}] jobs={len(jobs)} samples={len(samples)} workers={args.workers}")
    t0 = time.time()
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(eval_one, job) for job in jobs]
        done = 0
        for fut in as_completed(futs):
            row = fut.result()
            done += 1
            if row.get("ok"):
                results.append(row)
            else:
                failures.append(row)
            if done % 2000 == 0 or done == len(jobs):
                print(f"  progress {done}/{len(jobs)} elapsed={time.time()-t0:.1f}s")

    # Original NUC map from Line A original rows (same cloud)
    original_nuc: dict[str, float] = {}
    for r in results:
        if r["method"] == "original" and r["line"] == "A" and np.isfinite(r["nuc"]):
            original_nuc[f"{r['split']}/{r['class']}/{r['shape_id']}"] = float(r["nuc"])

    summary = summarize(results, original_nuc)
    line_a = [r for r in summary if r["line"] == "A"]
    line_b = [r for r in summary if r["line"] == "B"]

    fields = [
        "line",
        "method",
        "method_key",
        "points",
        "reference",
        "cd_vs_original",
        "cd_std",
        "hd_vs_original",
        "hd_std",
        "nuc",
        "nuc_std",
        "delta_nuc_vs_original",
        "valid_samples",
    ]
    out_a = REPORTS / "modelnet40_geometry_vs_original_lineA.csv"
    out_b = REPORTS / "modelnet40_geometry_vs_original_lineB.csv"
    out_all = REPORTS / "modelnet40_geometry_vs_original_summary.csv"
    per_sample = REPORTS / "modelnet40_geometry_vs_original_per_sample.csv"
    audit = REPORTS / "modelnet40_geometry_vs_original_audit.md"

    write_csv(out_a, line_a, fields)
    write_csv(out_b, line_b, fields)
    write_csv(out_all, summary, fields)

    sample_fields = [
        "line",
        "method",
        "display",
        "point_count",
        "reference",
        "split",
        "class",
        "shape_id",
        "cd",
        "hd",
        "cd_forward",
        "cd_backward",
        "hd_forward",
        "hd_backward",
        "nuc",
        "nuc_r1",
        "nuc_r2",
        "nuc_r3",
    ]
    write_csv(per_sample, results, sample_fields)

    # Failure CSV
    if failures:
        write_csv(
            REPORTS / "modelnet40_geometry_vs_original_failures.csv",
            failures,
            ["line", "method", "split", "class", "shape_id", "reason"],
        )

    elapsed = time.time() - t0
    audit_lines = [
        "# Geometry vs Original — Audit",
        "",
        f"- Generated: {utc_now()}",
        f"- Script: `scripts/compute_geometry_vs_original.py`",
        f"- Reference: corresponding Original 1024-point cloud only (no mesh / dense surface / P2F)",
        f"- Splits: {', '.join(splits)}",
        f"- Samples per split used: {len(samples) // max(len(splits), 1)} (total sample IDs: {len(samples)})",
        f"- Averaging: arithmetic mean over valid samples",
        f"- CD definition: mean NN L2(a→b) + mean NN L2(b→a)",
        f"- HD definition: max( max NN L2(a→b), max NN L2(b→a) )",
        f"- NUC: CV of neighbor counts at radii {NUC_RADII}, {NUC_QUERY_CENTERS} query centers, seed base {GLOBAL_SEED}",
        f"- Workers: {args.workers}",
        f"- Elapsed: {elapsed:.1f}s",
        f"- Valid result rows: {len(results)}",
        f"- Failures: {len(failures)}",
        "",
        "## Input directories",
        "",
        f"- Original: `{ORIGINAL_ROOT}`",
        f"- Downsampled ×4: `{DOWNSAMPLED_ROOT}`",
        f"- Line A strict 4N: `{LINEA_STRICT}/{{method}}`",
        f"- Line B strict N: `{LINEB_STRICT}/{{method}}`",
        "",
        "## Summary (CD / HD / NUC)",
        "",
        "| Line | Method | Points | CD vs Original | HD vs Original | NUC | ΔNUC | N |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        def fmt(v: Any) -> str:
            if isinstance(v, float):
                return f"{v:.6f}"
            return str(v)

        audit_lines.append(
            f"| {r['line']} | {r['method']} | {r['points']} | {fmt(r['cd_vs_original'])} | "
            f"{fmt(r['hd_vs_original'])} | {fmt(r['nuc'])} | {fmt(r['delta_nuc_vs_original'])} | "
            f"{r['valid_samples']} |"
        )

    audit_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Original vs Original has CD=0 and HD=0 by definition.",
            "- Line A compares unequal cardinalities (4096 vs 1024); metrics measure consistency with the discrete Original set, not continuous-surface accuracy.",
            "- P2F is intentionally not computed.",
            "- Old mesh-reference CD/HD values must not be reused.",
            "",
            "## Outputs",
            "",
            f"- `{out_a}`",
            f"- `{out_b}`",
            f"- `{out_all}`",
            f"- `{per_sample}`",
            f"- `{audit}`",
        ]
    )
    audit.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    # JSON mirror for presentation builder
    (REPORTS / "modelnet40_geometry_vs_original_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"[{utc_now()}] done. Wrote {out_a.name}, {out_b.name}, {audit.name}")
    for r in summary:
        print(
            f"  Line {r['line']:1} {r['method']:16} CD={r['cd_vs_original']} HD={r['hd_vs_original']} "
            f"NUC={r['nuc']} n={r['valid_samples']}"
        )


if __name__ == "__main__":
    main()
