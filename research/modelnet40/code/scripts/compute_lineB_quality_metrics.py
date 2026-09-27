#!/usr/bin/env python3
"""Compute Line B intermediate quality metrics: CD, HD, P2F, NUC."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

from modelnet40_x4_protocol import (
    DOWNSAMPLED_X4_ROOT,
    GLOBAL_SEED,
    MAIN_METHODS,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    lineB_paths,
    protocol_counts,
    reports_dir,
    stable_seed,
)

SPLITS = ("train", "test")
NUC_QUERY_CENTERS = 128
NUC_RADII = (0.02, 0.05, 0.10)


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Symmetric Chamfer using brute-force (N<=4096)."""
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    fwd = float(np.mean(np.min(d2, axis=1)))
    bwd = float(np.mean(np.min(d2, axis=0)))
    return fwd, bwd, fwd + bwd


def hausdorff_distance(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    diff = a[:, None, :] - b[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    fwd = float(np.max(np.min(d, axis=1)))
    bwd = float(np.max(np.min(d, axis=0)))
    return fwd, bwd, max(fwd, bwd)


def parse_off(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse OFF mesh -> vertices (V,3), faces (F,3) indices."""
    lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if not lines[0].strip().upper().startswith("OFF"):
        raise ValueError(f"Not OFF: {path}")
    counts = lines[1].split()
    nv, nf = int(counts[0]), int(counts[1])
    verts = []
    idx = 2
    for _ in range(nv):
        verts.append([float(x) for x in lines[idx].split()[:3]])
        idx += 1
    faces = []
    for _ in range(nf):
        parts = lines[idx].split()
        idx += 1
        if len(parts) < 4:
            continue
        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def point_triangle_distance(p: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    """Minimum distance from point p to triangle v0,v1,v2."""
    # Project onto plane then clamp barycentrics (Ericson)
    e0, e1 = v1 - v0, v2 - v0
    a = np.dot(e0, e0)
    b = np.dot(e0, e1)
    c = np.dot(e1, e1)
    vp = p - v0
    d = np.dot(vp, e0)
    e = np.dot(vp, e1)
    det = a * c - b * b
    if abs(det) < 1e-12:
        # degenerate: min dist to edges/verts
        candidates = [v0, v1, v2]
        return float(min(np.linalg.norm(p - q) for q in candidates))
    u = (c * d - b * e) / det
    v = (a * e - b * d) / det
    if u >= 0 and v >= 0 and u + v <= 1:
        closest = v0 + u * e0 + v * e1
        return float(np.linalg.norm(p - closest))
    # edges
    def seg_dist(p, a, b):
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
        return float(np.linalg.norm(p - (a + t * ab)))
    return min(seg_dist(p, v0, v1), seg_dist(p, v1, v2), seg_dist(p, v2, v0))


def point_to_mesh_distances(points: np.ndarray, verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    dists = np.empty(points.shape[0], dtype=np.float64)
    for i, p in enumerate(points):
        best = np.inf
        for f in faces:
            d = point_triangle_distance(p, verts[f[0]], verts[f[1]], verts[f[2]])
            if d < best:
                best = d
        dists[i] = best
    return dists


def compute_nuc(points: np.ndarray, seed: int) -> dict[str, float]:
    """NUC: coefficient of variation of neighbor counts at fixed radii."""
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    k = min(NUC_QUERY_CENTERS, n)
    center_idx = rng.choice(n, size=k, replace=False)
    centers = points[center_idx]
    nuc_vals = []
    out = {}
    for ri, radius in enumerate(NUC_RADII, start=1):
        counts = []
        r2 = radius * radius
        for c in centers:
            diff = points - c
            cnt = int(np.sum(np.sum(diff * diff, axis=1) < r2) - 1)  # exclude self
            counts.append(max(cnt, 0))
        arr = np.asarray(counts, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = std / mean if mean > 1e-8 else 0.0
        out[f"nuc_r{ri}"] = cv
        nuc_vals.append(cv)
    out["nuc_mean"] = float(np.mean(nuc_vals))
    return out


@lru_cache(maxsize=4096)
def load_mesh(off_path: str) -> tuple[np.ndarray, np.ndarray]:
    return parse_off(Path(off_path))


def load_manifest_off_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split in SPLITS:
        manifest = ORIGINAL_ROOT / "metadata" / f"{split}_manifest.csv"
        if not manifest.is_file():
            continue
        with open(manifest, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = f"{row['split']}/{row['class_name']}/{row['shape_id']}"
                mapping[key] = row.get("source_off", "")
    return mapping


def eval_sample(
    pred: np.ndarray,
    original: np.ndarray,
    off_path: str | None,
    seed: int,
    compute_p2f: bool,
) -> dict:
    cd_f, cd_b, cd_t = chamfer_distance(pred, original)
    hd_f, hd_b, hd_t = hausdorff_distance(pred, original)
    nuc = compute_nuc(pred, seed)
    row = {
        "cd_forward": cd_f,
        "cd_backward": cd_b,
        "cd_total": cd_t,
        "hd_forward": hd_f,
        "hd_backward": hd_b,
        "hd_total": hd_t,
        "p2f_mean": "",
        "p2f_median": "",
        "p2f_95": "",
        "p2f_max": "",
        "p2f_status": "skipped",
        **nuc,
    }
    if compute_p2f and off_path and Path(off_path).is_file():
        try:
            verts, faces = load_mesh(off_path)
            # normalize mesh same as point cloud (unit sphere) for fair comparison
            centroid = np.mean(verts, axis=0)
            verts = verts - centroid
            max_r = np.max(np.linalg.norm(verts, axis=1))
            if max_r > 0:
                verts = verts / max_r
            pred_n = pred.copy()
            c2 = np.mean(pred_n, axis=0)
            pred_n = pred_n - c2
            mr = np.max(np.linalg.norm(pred_n, axis=1))
            if mr > 0:
                pred_n = pred_n / mr
            p2f = point_to_mesh_distances(pred_n.astype(np.float64), verts, faces)
            row["p2f_mean"] = float(np.mean(p2f))
            row["p2f_median"] = float(np.median(p2f))
            row["p2f_95"] = float(np.percentile(p2f, 95))
            row["p2f_max"] = float(np.max(p2f))
            row["p2f_status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            row["p2f_status"] = f"error:{exc}"
    elif compute_p2f:
        row["p2f_status"] = "mesh_missing"
    return row


def run_comparison_group(
    name: str,
    pred_root: Path,
    original_root: Path,
    off_map: dict[str, str],
    source_point_count: int,
    target_point_count: int,
    max_samples: int = 0,
    compute_p2f: bool = True,
) -> list[dict]:
    rows: list[dict] = []
    count = 0
    for split in SPLITS:
        pred_split = pred_root / split
        if not pred_split.is_dir():
            continue
        for npy_path in sorted(pred_split.rglob("*.npy")):
            if max_samples and count >= max_samples:
                break
            rel = npy_path.relative_to(pred_root)
            orig_path = original_root / rel
            if not orig_path.is_file():
                continue
            pred = np.load(npy_path).astype(np.float32)
            original = np.load(orig_path).astype(np.float32)
            key = f"{split}/{rel.parent.name}/{npy_path.stem}"
            off_path = off_map.get(key, "")
            seed = stable_seed(GLOBAL_SEED, "nuc", name, key)
            metrics = eval_sample(pred, original, off_path, seed, compute_p2f=compute_p2f)
            rows.append({
                "comparison_group": name,
                "split": split,
                "class_name": rel.parts[-2],
                "shape_id": npy_path.stem,
                "source_point_count": source_point_count,
                "target_point_count": target_point_count,
                "pred_points": int(pred.shape[0]),
                "original_points": int(original.shape[0]),
                **metrics,
            })
            count += 1
    return rows


def summarize(rows: list[dict], method: str, group: str, split: str) -> dict:
    subset = [r for r in rows if r["comparison_group"] == group and (split == "all" or r["split"] == split)]
    if not subset:
        return {
            "comparison_group": group,
            "method": method,
            "split": split,
            "source_point_count": "",
            "target_point_count": "",
            "valid_samples": 0,
            "p2f_status": "no_data",
        }

    def stat(key: str) -> tuple[float, float]:
        vals = [float(r[key]) for r in subset if r.get(key) not in ("", None)]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    cd_m, cd_s = stat("cd_total")
    hd_m, hd_s = stat("hd_total")
    p2f_m, p2f_s = stat("p2f_mean")
    nuc_m, nuc_s = stat("nuc_mean")
    p2f_ok = sum(1 for r in subset if r.get("p2f_status") == "ok")
    p2f_status = "available" if p2f_ok == len(subset) else ("partial" if p2f_ok else "unavailable")

    return {
        "comparison_group": group,
        "method": method,
        "split": split,
        "source_point_count": subset[0]["source_point_count"],
        "target_point_count": subset[0]["target_point_count"],
        "cd_mean": cd_m,
        "cd_std": cd_s,
        "hd_mean": hd_m,
        "hd_std": hd_s,
        "p2f_mean": p2f_m,
        "p2f_std": p2f_s,
        "nuc_mean": nuc_m,
        "nuc_std": nuc_s,
        "valid_samples": len(subset),
        "p2f_status": p2f_status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=0, help="0=all")
    parser.add_argument("--smoke", action="store_true", help="Run on 50 samples per group")
    parser.add_argument("--skip-p2f", action="store_true", help="Skip P2F (much faster for full dataset)")
    args = parser.parse_args()

    n = detect_original_point_count()
    down_n = n // 4
    max_samples = 50 if args.smoke else args.max_samples
    compute_p2f = not args.skip_p2f
    off_map = load_manifest_off_map()

    per_sample_dir = PROJECT_ROOT / "quality_metrics" / "lineB_intermediate_comparison" / "per_sample"
    summary_dir = PROJECT_ROOT / "quality_metrics" / "lineB_intermediate_comparison" / "summary"
    per_sample_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    # Downsampled x4 baseline vs original
    if DOWNSAMPLED_X4_ROOT.is_dir():
        rows = run_comparison_group(
            "downsampled_x4_vs_original",
            DOWNSAMPLED_X4_ROOT,
            ORIGINAL_ROOT,
            off_map,
            source_point_count=down_n,
            target_point_count=n,
            max_samples=max_samples,
            compute_p2f=compute_p2f,
        )
        all_rows.extend(rows)

    # Line B upsampled methods
    method_groups = {
        "ear": "downsampled_x4_up_ear_vs_original",
        "pu_net": "downsampled_x4_up_pu_net_vs_original",
        "pu_gcn": "downsampled_x4_up_pu_gcn_vs_original",
        "pdans": "downsampled_x4_up_pdans_vs_original",
    }
    for method, group in method_groups.items():
        strict_root = lineB_paths(method)["strict_N"]
        if strict_root.is_dir() and any(strict_root.rglob("*.npy")):
            rows = run_comparison_group(
                group,
                strict_root,
                ORIGINAL_ROOT,
                off_map,
                source_point_count=down_n,
                target_point_count=n,
                max_samples=max_samples,
                compute_p2f=compute_p2f,
            )
            all_rows.extend(rows)

    per_sample_csv = reports_dir() / "modelnet40_lineB_quality_metrics_per_sample.csv"
    summary_csv = reports_dir() / "modelnet40_lineB_quality_metrics_summary.csv"

    if all_rows:
        with open(per_sample_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)

    summaries = []
    method_names = {
        "downsampled_x4_vs_original": "downsampled_x4_baseline",
        "downsampled_x4_up_ear_vs_original": "EAR",
        "downsampled_x4_up_pu_net_vs_original": "PU-Net",
        "downsampled_x4_up_pu_gcn_vs_original": "PU-GCN",
        "downsampled_x4_up_pdans_vs_original": "PDANS",
    }
    groups = ["downsampled_x4_vs_original"] + list(method_groups.values())
    for group in groups:
        for split in ("train", "test"):
            s = summarize(all_rows, method_names.get(group, group), group, split)
            if s["valid_samples"] > 0:
                summaries.append(s)
        subset = [r for r in all_rows if r["comparison_group"] == group]
        if subset:
            s_all = summarize(all_rows, method_names.get(group, group), group, "train")
            s_all["split"] = "all"
            s_all["valid_samples"] = len(subset)
            for key, src in (("cd_mean", "cd_total"), ("hd_mean", "hd_total"), ("p2f_mean", "p2f_mean"), ("nuc_mean", "nuc_mean")):
                vals = [float(r[src]) for r in subset if r.get(src) not in ("", None)]
                if vals:
                    s_all[key] = float(np.mean(vals))
                    s_all[key.replace("_mean", "_std")] = float(np.std(vals))
            summaries.append(s_all)

    # Clean summary: only groups with data
    summaries = [s for s in summaries if s.get("valid_samples", 0) > 0]

    if summaries:
        with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

    nuc_def = (
        "NUC (Neighborhood Uniformity Coefficient): For each sample, select "
        f"{NUC_QUERY_CENTERS} query centers (fixed seed). At radii {NUC_RADII}, "
        "count neighbors within each radius; compute coefficient of variation (std/mean) "
        "across centers. Lower NUC = more uniform local density. nuc_mean = mean of CV at 3 radii."
    )

    md = reports_dir() / "modelnet40_lineB_quality_metrics_report.md"
    md.write_text(
        "\n".join([
            "# Line B Quality Metrics Report",
            "",
            f"- Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Metric definitions",
            "",
            "- **CD**: Chamfer distance (forward + backward mean squared)",
            "- **HD**: Hausdorff distance (max min-distance)",
            "- **P2F**: Point-to-surface distance to original CAD .off mesh (unit-sphere normalized)",
            f"- **NUC**: {nuc_def}",
            "",
            f"- Per-sample CSV: `{per_sample_csv}`",
            f"- Summary CSV: `{summary_csv}`",
            f"- Samples computed: {len(all_rows)}",
        ]),
        encoding="utf-8",
    )

    print(f"Computed {len(all_rows)} per-sample metrics, {len(summaries)} summary rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
