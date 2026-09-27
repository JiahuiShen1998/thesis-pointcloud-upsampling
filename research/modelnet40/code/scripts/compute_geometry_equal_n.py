#!/usr/bin/env python3
"""Equal-cardinality geometry metrics vs mesh-sampled / Original references.

Protocol:
  Line A upsamplers (4096) vs modelnet40_mesh_ref_4096
  Line B Downsampled ×4 (256) vs modelnet40_mesh_ref_256
  Line B upsamplers (1024) vs modelnet40_original (1024)
  Mesh-ref self rows: CD=HD=0 by definition
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
MESH_REF_256 = PROJECT_ROOT / "datasets" / "modelnet40_mesh_ref_256"
MESH_REF_4096 = PROJECT_ROOT / "datasets" / "modelnet40_mesh_ref_4096"
LINEA_STRICT = PROJECT_ROOT / "datasets" / "lineA_original_up" / "strict_4N"
LINEB_STRICT = PROJECT_ROOT / "datasets" / "lineB_downsampled_x4_up" / "strict_N"
REPORTS = PROJECT_ROOT / "reports"

GLOBAL_SEED = 42
NUC_QUERY_CENTERS = 128
NUC_RADII = (0.02, 0.05, 0.10)
METHODS = ("ear", "pdans", "pu_net", "pu_gcn", "pu_edgeformer")
METHOD_DISPLAY = {
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
    eval_root: Path | None  # None => use reference as evaluated (self)
    ref_root: Path
    ref_label: str


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


def define_groups() -> list[EvalGroup]:
    groups = [
        EvalGroup("A", "mesh_ref_4096", "Mesh-ref 4096", 4096, None, MESH_REF_4096, "Mesh-ref 4096"),
    ]
    for m in METHODS:
        groups.append(
            EvalGroup("A", m, METHOD_DISPLAY[m], 4096, LINEA_STRICT / m, MESH_REF_4096, "Mesh-ref 4096")
        )
    groups.append(EvalGroup("B", "mesh_ref_256", "Mesh-ref 256", 256, None, MESH_REF_256, "Mesh-ref 256"))
    groups.append(
        EvalGroup(
            "B",
            "downsampled_x4",
            "Downsampled ×4",
            256,
            DOWNSAMPLED_ROOT,
            MESH_REF_256,
            "Mesh-ref 256",
        )
    )
    groups.append(EvalGroup("B", "original", "Original", 1024, None, ORIGINAL_ROOT, "Original 1024"))
    for m in METHODS:
        groups.append(
            EvalGroup("B", m, METHOD_DISPLAY[m], 1024, LINEB_STRICT / m, ORIGINAL_ROOT, "Original 1024")
        )
    return groups


def eval_one(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        line,
        method_key,
        display,
        expected_n,
        eval_root_str,
        ref_root_str,
        ref_label,
        split,
        class_name,
        shape_id,
        compute_nuc_flag,
    ) = args
    ref_root = Path(ref_root_str)
    ref_path = sample_path(ref_root, split, class_name, shape_id)
    if not ref_path.exists():
        return {
            "ok": False,
            "reason": f"missing reference {ref_path}",
            "line": line,
            "method": method_key,
            "split": split,
            "class": class_name,
            "shape_id": shape_id,
        }
    reference = np.load(ref_path).astype(np.float64)
    if reference.shape != (expected_n, 3):
        return {
            "ok": False,
            "reason": f"bad reference shape {reference.shape}",
            "line": line,
            "method": method_key,
            "split": split,
            "class": class_name,
            "shape_id": shape_id,
        }

    if eval_root_str is None:
        evaluated = reference
        is_self = True
    else:
        is_self = False
        path = sample_path(Path(eval_root_str), split, class_name, shape_id)
        if not path.exists():
            return {
                "ok": False,
                "reason": f"missing eval {path}",
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

    if is_self:
        metrics = {
            "cd_forward": 0.0,
            "cd_backward": 0.0,
            "cd": 0.0,
            "hd_forward": 0.0,
            "hd_backward": 0.0,
            "hd": 0.0,
        }
    else:
        metrics = chamfer_hd(evaluated, reference)

    seed = GLOBAL_SEED + abs(hash(f"{split}:{class_name}:{shape_id}:{method_key}:eq")) % 1_000_000
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
        "reference": ref_label,
        "split": split,
        "class": class_name,
        "shape_id": shape_id,
        **metrics,
        **nuc,
    }


def summarize(rows: list[dict[str, Any]], ref_nuc: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    from collections import defaultdict

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("ok"):
            buckets[(r["line"], r["method"])].append(r)

    out: list[dict[str, Any]] = []
    for group in define_groups():
        items = buckets.get((group.line, group.method_key), [])
        if not items:
            out.append(
                {
                    "line": group.line,
                    "method": group.display,
                    "method_key": group.method_key,
                    "points": group.point_count,
                    "reference": group.ref_label,
                    "cd_vs_ref": "Pending",
                    "hd_vs_ref": "Pending",
                    "nuc": "Pending",
                    "delta_nuc_vs_ref": "Pending",
                    "valid_samples": 0,
                }
            )
            continue
        cds = np.array([x["cd"] for x in items], dtype=np.float64)
        hds = np.array([x["hd"] for x in items], dtype=np.float64)
        nucs = np.array([x["nuc"] for x in items], dtype=np.float64)
        deltas = []
        for x in items:
            base = ref_nuc.get((group.ref_label, f"{x['split']}/{x['class']}/{x['shape_id']}"))
            if base is not None and np.isfinite(x["nuc"]):
                deltas.append(x["nuc"] - base)
        delta_arr = np.asarray(deltas, dtype=np.float64) if deltas else np.array([0.0])
        self_keys = {"mesh_ref_256", "mesh_ref_4096", "original"}
        out.append(
            {
                "line": group.line,
                "method": group.display,
                "method_key": group.method_key,
                "points": group.point_count,
                "reference": group.ref_label,
                "cd_vs_ref": float(np.mean(cds)),
                "cd_std": float(np.std(cds)),
                "hd_vs_ref": float(np.mean(hds)),
                "hd_std": float(np.std(hds)),
                "nuc": float(np.mean(nucs)),
                "nuc_std": float(np.std(nucs)),
                "delta_nuc_vs_ref": 0.0
                if group.method_key in self_keys and group.eval_root is None
                else float(np.mean(delta_arr)),
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


def write_audit(summary: list[dict[str, Any]], n_samples: int, elapsed: float, failures: int) -> None:
    lines = [
        "# Geometry Equal-N — Audit",
        "",
        f"- Generated: {utc_now()}",
        "- Script: `scripts/compute_geometry_equal_n.py`",
        "- Protocol: equal-cardinality CD/HD only",
        "  - Line A: upsamplers 4096 vs `datasets/modelnet40_mesh_ref_4096`",
        "  - Line B Downsampled: 256 vs `datasets/modelnet40_mesh_ref_256`",
        "  - Line B upsamplers: 1024 vs `datasets/modelnet40_original`",
        f"- Split: test (N={n_samples})",
        "- CD: mean NN L2(a→b) + mean NN L2(b→a)",
        "- HD: max(max NN a→b, max NN b→a)",
        f"- Workers elapsed: {elapsed:.1f}s",
        f"- Failures: {failures}",
        "",
        "## Summary",
        "",
        "| Line | Method | Points | Reference | CD | HD | NUC | ΔNUC | N |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        def fmt(v: Any) -> str:
            return v if isinstance(v, str) else f"{float(v):.6f}"

        lines.append(
            f"| {r['line']} | {r['method']} | {r['points']} | {r['reference']} | "
            f"{fmt(r['cd_vs_ref'])} | {fmt(r['hd_vs_ref'])} | {fmt(r['nuc'])} | "
            f"{fmt(r['delta_nuc_vs_ref'])} | {r['valid_samples']} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Unequal-cardinality CD/HD (4096 vs 1024, 256 vs 1024) are intentionally not reported here.",
        "- Mesh-ref rows have CD=0 / HD=0 by self-comparison.",
        "",
    ]
    path = REPORTS / "modelnet40_geometry_equal_n_audit.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", nargs="+", default=["test"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--no-nuc", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    for root in (MESH_REF_256, MESH_REF_4096):
        if not root.exists():
            raise SystemExit(f"Missing mesh reference root: {root}")

    samples: list[tuple[str, str, str]] = []
    for split in args.splits:
        samples.extend(load_manifest(split))
    if args.limit > 0:
        samples = samples[: args.limit]

    groups = define_groups()
    tasks = []
    for g in groups:
        for split, class_name, shape_id in samples:
            tasks.append(
                (
                    g.line,
                    g.method_key,
                    g.display,
                    g.point_count,
                    None if g.eval_root is None else str(g.eval_root),
                    str(g.ref_root),
                    g.ref_label,
                    split,
                    class_name,
                    shape_id,
                    not args.no_nuc,
                )
            )

    print(f"[{utc_now()}] equal-N geometry: {len(tasks)} tasks, workers={args.workers}")
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(eval_one, t) for t in tasks]
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            rows.append(r)
            if not r.get("ok"):
                failures += 1
            done += 1
            if done % 2000 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} failures={failures}")
    elapsed = time.time() - t0

    # Reference NUC keyed by (ref_label, sample_key)
    ref_nuc: dict[tuple[str, str], float] = {}
    for r in rows:
        if not r.get("ok"):
            continue
        if r["method"] in ("mesh_ref_256", "mesh_ref_4096", "original") and np.isfinite(r["nuc"]):
            key = f"{r['split']}/{r['class']}/{r['shape_id']}"
            ref_nuc[(r["reference"], key)] = float(r["nuc"])

    summary = summarize(rows, ref_nuc)

    per_fields = [
        "ok",
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
        "reason",
    ]
    write_csv(REPORTS / "modelnet40_geometry_equal_n_per_sample.csv", rows, per_fields)

    sum_fields = [
        "line",
        "method",
        "method_key",
        "points",
        "reference",
        "cd_vs_ref",
        "cd_std",
        "hd_vs_ref",
        "hd_std",
        "nuc",
        "nuc_std",
        "delta_nuc_vs_ref",
        "valid_samples",
    ]
    write_csv(REPORTS / "modelnet40_geometry_equal_n_summary.csv", summary, sum_fields)

    for line_key, name in (("A", "lineA"), ("B", "lineB")):
        subset = [r for r in summary if r["line"] == line_key]
        write_csv(REPORTS / f"modelnet40_geometry_equal_n_{name}.csv", subset, sum_fields)

    write_audit(summary, len(samples), elapsed, failures)
    meta = {
        "generated_at": utc_now(),
        "splits": args.splits,
        "n_samples": len(samples),
        "n_tasks": len(tasks),
        "failures": failures,
        "elapsed_s": elapsed,
    }
    (REPORTS / "modelnet40_geometry_equal_n_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[{utc_now()}] DONE failures={failures} elapsed={elapsed:.1f}s")
    if failures:
        # Print a few failure reasons
        bad = [r for r in rows if not r.get("ok")][:10]
        for b in bad:
            print("  FAIL", b.get("method"), b.get("shape_id"), b.get("reason"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
