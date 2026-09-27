#!/usr/bin/env python3
"""ModelNet40 x4 two-line protocol: geometric quality metrics vs dense mesh-sampled GT."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from modelnet40_x4_protocol import (  # noqa: E402
    DOWNSAMPLED_X4_ROOT,
    GLOBAL_SEED,
    MAIN_METHODS,
    ORIGINAL_ROOT,
    PROJECT_ROOT,
    detect_original_point_count,
    lineA_paths,
    lineB_paths,
    reports_dir,
    stable_seed,
)
from prepare_modelnet40 import (  # noqa: E402
    load_off_mesh,
    normalize_unit_sphere,
    sample_points_from_mesh,
)

SPLITS = ("train", "test")
GT_POINT_COUNT = 10_000
GT_ROOT = PROJECT_ROOT / "quality_metrics" / f"gt_reference_points_{GT_POINT_COUNT}"
NUC_QUERY_CENTERS = 128
NUC_RADII = (0.02, 0.05, 0.10)
P2F_APPROX_SURFACE_POINTS = 50_000

METHOD_DISPLAY = {
    "ear": "EAR",
    "pdans": "PDANS",
    "pu_net": "PU-Net",
    "pu_gcn": "PU-GCN",
}


@dataclass(frozen=True)
class SourceGroup:
    group: str
    line: str
    method: str
    source_root: Path
    expected_point_count: int
    tier: str  # main | supplementary


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def try_import_ckdtree():
    try:
        from scipy.spatial import cKDTree

        return cKDTree
    except Exception:
        return None


def nn_distances(source: np.ndarray, target: np.ndarray, cKDTree_cls) -> np.ndarray:
    if cKDTree_cls is not None:
        tree = cKDTree_cls(target)
        dists, _ = tree.query(source, k=1, workers=-1)
        return np.asarray(dists, dtype=np.float64)
    diff = source[:, None, :] - target[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).min(axis=1)


def chamfer_distance_l2(a: np.ndarray, b: np.ndarray, cKDTree_cls) -> tuple[float, float, float]:
    fwd = float(np.mean(nn_distances(a, b, cKDTree_cls)))
    bwd = float(np.mean(nn_distances(b, a, cKDTree_cls)))
    return fwd, bwd, fwd + bwd


def hausdorff_distance_l2(a: np.ndarray, b: np.ndarray, cKDTree_cls) -> tuple[float, float, float]:
    fwd = float(np.max(nn_distances(a, b, cKDTree_cls)))
    bwd = float(np.max(nn_distances(b, a, cKDTree_cls)))
    return fwd, bwd, max(fwd, bwd)


def compute_nuc(points: np.ndarray, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    k = min(NUC_QUERY_CENTERS, n)
    center_idx = rng.choice(n, size=k, replace=False)
    centers = points[center_idx]
    nuc_vals: list[float] = []
    out: dict[str, float] = {}
    for ri, radius in enumerate(NUC_RADII, start=1):
        counts: list[int] = []
        r2 = radius * radius
        for c in centers:
            diff = points - c
            cnt = int(np.sum(np.sum(diff * diff, axis=1) < r2) - 1)
            counts.append(max(cnt, 0))
        arr = np.asarray(counts, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = std / mean if mean > 1e-8 else 0.0
        out[f"nuc_r{ri}"] = cv
        nuc_vals.append(cv)
    out["nuc_mean"] = float(np.mean(nuc_vals))
    return out


def load_manifest_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in SPLITS:
        manifest = ORIGINAL_ROOT / "metadata" / f"{split}_manifest.csv"
        with open(manifest, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)
    return rows


def rel_key(split: str, class_name: str, shape_id: str) -> str:
    return f"{split}/{class_name}/{shape_id}"


def gt_output_path(split: str, class_name: str, shape_id: str) -> Path:
    return GT_ROOT / split / class_name / f"{shape_id}.npy"


def define_source_groups(n: int) -> list[SourceGroup]:
    down_n = n // 4
    four_n = n * 4
    groups = [
        SourceGroup(
            "original_baseline_vs_gt",
            "baseline",
            "original",
            ORIGINAL_ROOT,
            n,
            "main",
        ),
        SourceGroup(
            "downsampled_x4_baseline_vs_gt",
            "lineB_baseline",
            "downsampled_x4",
            DOWNSAMPLED_X4_ROOT,
            down_n,
            "supplementary",
        ),
    ]
    for method in MAIN_METHODS:
        groups.append(
            SourceGroup(
                f"downsampled_x4_up_{method}_vs_gt",
                "lineB",
                method,
                lineB_paths(method)["strict_N"],
                n,
                "main",
            )
        )
        groups.append(
            SourceGroup(
                f"original_up_{method}_vs_gt",
                "lineA",
                method,
                lineA_paths(method)["strict_4N"],
                four_n,
                "supplementary",
            )
        )
    return groups


def audit_point_cloud_root(root: Path, expected_points: int) -> dict[str, Any]:
    expected_samples = EXPECTED_TRAIN + EXPECTED_TEST
    min_pts = 10**9
    max_pts = 0
    nan_count = 0
    inf_count = 0
    bad_shape = 0
    if not root.is_dir():
        return {
            "expected_samples": expected_samples,
            "actual_samples": 0,
            "expected_point_count": expected_points,
            "min_points": "",
            "max_points": "",
            "nan_count": "",
            "inf_count": "",
            "status": "missing",
            "note": "directory not found",
        }
    paths = sorted(root.rglob("*.npy"))
    actual = len(paths)
    if actual == 0:
        return {
            "expected_samples": expected_samples,
            "actual_samples": 0,
            "expected_point_count": expected_points,
            "min_points": "",
            "max_points": "",
            "nan_count": "",
            "inf_count": "",
            "status": "empty",
            "note": "no npy files",
        }
    stride = max(1, actual // 200)
    for npy in paths[::stride]:
        arr = np.load(npy, mmap_mode="r")
        if arr.ndim != 2 or arr.shape[1] != 3:
            bad_shape += 1
            continue
        n = int(arr.shape[0])
        min_pts = min(min_pts, n)
        max_pts = max(max_pts, n)
        if n <= 8192:
            chunk = np.array(arr)
            nan_count += int(np.isnan(chunk).sum())
            inf_count += int(np.isinf(chunk).sum())
    if min_pts == 10**9:
        min_pts = max_pts = 0
    if actual != expected_samples:
        status = "partial"
        note = f"expected {expected_samples}, got {actual}"
    elif bad_shape:
        status = "bad_shape"
        note = f"{bad_shape} invalid arrays"
    elif nan_count or inf_count:
        status = "non_finite"
        note = "contains NaN/Inf"
    elif min_pts != expected_points or max_pts != expected_points:
        status = "point_count_mismatch"
        note = f"points in [{min_pts}, {max_pts}], expected {expected_points}"
    else:
        status = "ok"
        note = ""
    return {
        "expected_samples": expected_samples,
        "actual_samples": actual,
        "expected_point_count": expected_points,
        "min_points": min_pts if actual else "",
        "max_points": max_pts if actual else "",
        "nan_count": nan_count,
        "inf_count": inf_count,
        "status": status,
        "note": note,
    }


EXPECTED_TRAIN = 9843
EXPECTED_TEST = 2468


def run_source_audit(groups: list[SourceGroup]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for g in groups:
        audit = audit_point_cloud_root(g.source_root, g.expected_point_count)
        rows.append(
            {
                "group": g.group,
                "source_path": str(g.source_root),
                **audit,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_source_audit_reports(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = reports_dir() / "modelnet40_quality_metrics_source_audit.csv"
    md_path = reports_dir() / "modelnet40_quality_metrics_source_audit.md"
    write_csv(csv_path, rows)
    lines = [
        "# ModelNet40 Quality Metrics Source Audit",
        "",
        f"- Generated at: {utc_now()}",
        "",
        "| group | actual_samples | expected_point_count | status | note |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['actual_samples']} | {row['expected_point_count']} | "
            f"{row['status']} | {row.get('note', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def gt_build_task(row: dict[str, str]) -> dict[str, Any]:
    split = row["split"]
    class_name = row["class_name"]
    shape_id = row["shape_id"]
    off_path = row["source_off"]
    out_path = gt_output_path(split, class_name, shape_id)
    key = rel_key(split, class_name, shape_id)
    seed = stable_seed(GLOBAL_SEED, "gt_reference", str(GT_POINT_COUNT), key)
    try:
        if out_path.is_file():
            arr = np.load(out_path)
            if arr.shape == (GT_POINT_COUNT, 3) and np.isfinite(arr).all():
                return {"status": "reused", "key": key, "out_path": str(out_path)}
        mesh = load_off_mesh(off_path)
        points = sample_points_from_mesh(mesh, GT_POINT_COUNT, seed)
        points = normalize_unit_sphere(points)
        if points.shape != (GT_POINT_COUNT, 3) or not np.isfinite(points).all():
            raise ValueError("invalid GT points after sampling")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, points.astype(np.float32))
        return {"status": "created", "key": key, "out_path": str(out_path)}
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "key": key, "out_path": str(out_path), "error": str(exc)}


def build_gt_reference(workers: int = 8) -> list[dict[str, Any]]:
    manifest_rows = load_manifest_rows()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(gt_build_task, row) for row in manifest_rows]
        for fut in as_completed(futures):
            results.append(fut.result())
    manifest_path = GT_ROOT / "metadata" / "gt_reference_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "gt_point_count": GT_POINT_COUNT,
                "global_seed": GLOBAL_SEED,
                "generated_at": utc_now(),
                "samples": len(manifest_rows),
                "results_summary": {
                    s: sum(1 for r in results if r["status"] == s)
                    for s in ("created", "reused", "failed")
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return results


def write_gt_audit_reports(results: list[dict[str, Any]]) -> tuple[Path, Path]:
    csv_path = reports_dir() / "modelnet40_gt_reference_points_audit.csv"
    md_path = reports_dir() / "modelnet40_gt_reference_points_audit.md"
    rows = []
    for r in results:
        out = Path(r["out_path"])
        exists = out.is_file()
        pts = shape = nan_count = inf_count = ""
        status = r["status"]
        if exists:
            arr = np.load(out)
            shape = str(tuple(arr.shape))
            pts = int(arr.shape[0]) if arr.ndim == 2 else ""
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            if status != "failed" and (arr.shape != (GT_POINT_COUNT, 3) or nan_count or inf_count):
                status = "invalid"
        rows.append(
            {
                "key": r["key"],
                "out_path": str(out),
                "status": status,
                "shape": shape,
                "point_count": pts,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "error": r.get("error", ""),
            }
        )
    write_csv(csv_path, rows)
    summary = {s: sum(1 for r in rows if r["status"] == s) for s in ("created", "reused", "ok", "invalid", "failed")}
    md = [
        "# ModelNet40 GT Reference Points Audit",
        "",
        f"- Generated at: {utc_now()}",
        f"- GT root: `{GT_ROOT}`",
        f"- GT point count: **{GT_POINT_COUNT}**",
        f"- Sampling: triangle-area weighted surface sampling from original `.off` mesh, unit-sphere normalized",
        f"- Seed: global={GLOBAL_SEED}, per-sample via `stable_seed`",
        "",
        "## Summary",
        "",
    ]
    for k, v in summary.items():
        if v:
            md.append(f"- {k}: {v}")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return csv_path, md_path


class P2FComputer:
    def __init__(self, mode: str = "auto"):
        self.mode = mode
        self.resolved_mode = ""
        self._surface_cache: dict[str, np.ndarray] = {}
        self._exact_scene_cache: dict[str, Any] = {}

    def _load_normalized_mesh(self, off_path: str) -> tuple[np.ndarray, np.ndarray]:
        mesh = load_off_mesh(off_path)
        verts = normalize_unit_sphere(mesh.vertices.copy())
        return verts, mesh.faces

    def _exact_distances(self, off_path: str, points: np.ndarray) -> np.ndarray:
        import open3d as o3d

        if off_path not in self._exact_scene_cache:
            verts, faces = self._load_normalized_mesh(off_path)
            tmesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(verts),
                o3d.utility.Vector3iVector(faces),
            )
            tmesh.compute_triangle_normals()
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(tmesh))
            self._exact_scene_cache[off_path] = scene
        scene = self._exact_scene_cache[off_path]
        t = o3d.core.Tensor(points.astype(np.float32))
        return scene.compute_distance(t).numpy()

    def _approx_distances(self, off_path: str, points: np.ndarray, cKDTree_cls) -> np.ndarray:
        if off_path not in self._surface_cache:
            mesh = load_off_mesh(off_path)
            seed = stable_seed(GLOBAL_SEED, "p2f_surface", str(P2F_APPROX_SURFACE_POINTS), off_path)
            surf = sample_points_from_mesh(mesh, P2F_APPROX_SURFACE_POINTS, seed)
            surf = normalize_unit_sphere(surf)
            self._surface_cache[off_path] = surf.astype(np.float64)
        return nn_distances(points.astype(np.float64), self._surface_cache[off_path], cKDTree_cls)

    def compute(self, off_path: str | None, points: np.ndarray, cKDTree_cls) -> dict[str, Any]:
        empty = {
            "p2f_mean": "",
            "p2f_median": "",
            "p2f_95": "",
            "p2f_max": "",
            "p2f_status": "mesh_missing",
        }
        if not off_path or not Path(off_path).is_file():
            return empty
        try:
            if self.mode == "skip":
                return {**empty, "p2f_status": "pending_slow_mesh_distance"}
            if self.mode == "approx":
                dists = self._approx_distances(off_path, points, cKDTree_cls)
                status = "approximate"
            elif self.mode == "exact":
                dists = self._exact_distances(off_path, points)
                status = "exact"
            else:
                try:
                    dists = self._exact_distances(off_path, points)
                    self.resolved_mode = "exact"
                    status = "exact"
                except Exception:
                    dists = self._approx_distances(off_path, points, cKDTree_cls)
                    self.resolved_mode = "approximate"
                    status = "approximate"
            return {
                "p2f_mean": float(np.mean(dists)),
                "p2f_median": float(np.median(dists)),
                "p2f_95": float(np.percentile(dists, 95)),
                "p2f_max": float(np.max(dists)),
                "p2f_status": status,
            }
        except Exception as exc:  # noqa: BLE001
            return {**empty, "p2f_status": f"error:{exc}"}


def eval_one_sample(
    source: np.ndarray,
    gt: np.ndarray,
    off_path: str | None,
    nuc_seed: int,
    p2f: P2FComputer,
    cKDTree_cls,
) -> dict[str, Any]:
    cd_f, cd_b, cd_t = chamfer_distance_l2(source, gt, cKDTree_cls)
    hd_f, hd_b, hd_t = hausdorff_distance_l2(source, gt, cKDTree_cls)
    nuc = compute_nuc(source, nuc_seed)
    p2f_row = p2f.compute(off_path, source, cKDTree_cls)
    return {
        "cd_forward": cd_f,
        "cd_backward": cd_b,
        "cd_total": cd_t,
        "hd_forward": hd_f,
        "hd_backward": hd_b,
        "hd_total": hd_t,
        **nuc,
        **p2f_row,
    }


def iter_samples(source_root: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for split in SPLITS:
        split_dir = source_root / split
        if not split_dir.is_dir():
            continue
        for npy in sorted(split_dir.rglob("*.npy")):
            rel = npy.relative_to(source_root)
            key = f"{rel.parts[0]}/{rel.parts[1]}/{npy.stem}"
            out.append((key, npy))
    return out


def _eval_sample_task(task: dict[str, Any]) -> dict[str, Any] | None:
    key = task["key"]
    parts = key.split("/")
    split, class_name, shape_id = parts[0], parts[1], parts[2]
    gt_path = gt_output_path(split, class_name, shape_id)
    if not gt_path.is_file():
        return None
    source = np.load(task["src_path"]).astype(np.float32)
    gt = np.load(gt_path).astype(np.float32)
    nuc_seed = stable_seed(GLOBAL_SEED, "nuc", task["group"], key)
    cKDTree_cls = try_import_ckdtree()
    p2f = P2FComputer(mode=task["p2f_mode"])
    metrics = eval_one_sample(source, gt, task.get("off_path"), nuc_seed, p2f, cKDTree_cls)
    return {
        "group": task["group"],
        "line": task["line"],
        "method": task["method"],
        "split": split,
        "class_name": class_name,
        "shape_id": shape_id,
        "source_point_count": int(source.shape[0]),
        "gt_point_count": int(gt.shape[0]),
        **metrics,
    }


def compute_group_metrics(
    group: SourceGroup,
    off_map: dict[str, str],
    max_samples: int,
    p2f_mode: str,
    workers: int = 1,
) -> list[dict[str, Any]]:
    samples = iter_samples(group.source_root)
    if max_samples:
        samples = samples[:max_samples]
    tasks = [
        {
            "key": key,
            "src_path": str(src_path),
            "off_path": off_map.get(key),
            "group": group.group,
            "line": group.line,
            "method": group.method,
            "p2f_mode": p2f_mode,
        }
        for key, src_path in samples
    ]
    rows: list[dict[str, Any]] = []
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(_eval_sample_task, tasks, chunksize=8):
                if row is not None:
                    rows.append(row)
    else:
        p2f = P2FComputer(mode=p2f_mode)
        cKDTree_cls = try_import_ckdtree()
        for task in tasks:
            key = task["key"]
            parts = key.split("/")
            split, class_name, shape_id = parts[0], parts[1], parts[2]
            gt_path = gt_output_path(split, class_name, shape_id)
            if not gt_path.is_file():
                continue
            source = np.load(task["src_path"]).astype(np.float32)
            gt = np.load(gt_path).astype(np.float32)
            nuc_seed = stable_seed(GLOBAL_SEED, "nuc", group.group, key)
            metrics = eval_one_sample(source, gt, task.get("off_path"), nuc_seed, p2f, cKDTree_cls)
            rows.append(
                {
                    "group": group.group,
                    "line": group.line,
                    "method": group.method,
                    "split": split,
                    "class_name": class_name,
                    "shape_id": shape_id,
                    "source_point_count": int(source.shape[0]),
                    "gt_point_count": int(gt.shape[0]),
                    **metrics,
                }
            )
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    groups = sorted({r["group"] for r in rows})
    for group in groups:
        subset = [r for r in rows if r["group"] == group]
        if not subset:
            continue
        first = subset[0]

        def mean_std(key: str) -> tuple[float, float]:
            vals = [float(r[key]) for r in subset if r.get(key) not in ("", None)]
            if not vals:
                return float("nan"), float("nan")
            return float(np.mean(vals)), float(np.std(vals))

        cd_m, cd_s = mean_std("cd_total")
        cd_f_m, _ = mean_std("cd_forward")
        cd_b_m, _ = mean_std("cd_backward")
        hd_m, hd_s = mean_std("hd_total")
        hd_f_m, _ = mean_std("hd_forward")
        hd_b_m, _ = mean_std("hd_backward")
        nuc_m, nuc_s = mean_std("nuc_mean")
        nuc_r1, _ = mean_std("nuc_r1")
        nuc_r2, _ = mean_std("nuc_r2")
        nuc_r3, _ = mean_std("nuc_r3")
        p2f_m, p2f_s = mean_std("p2f_mean")
        p2f_med, _ = mean_std("p2f_median")
        p2f_95, _ = mean_std("p2f_95")
        p2f_max, _ = mean_std("p2f_max")
        statuses = [r.get("p2f_status", "") for r in subset]
        if all(s in ("exact", "approximate") for s in statuses):
            p2f_status = statuses[0]
        elif any(s in ("exact", "approximate") for s in statuses):
            p2f_status = "partial"
        elif any(s == "pending_slow_mesh_distance" for s in statuses):
            p2f_status = "pending_slow_mesh_distance"
        else:
            p2f_status = statuses[0] if statuses else "pending_slow_mesh_distance"

        summaries.append(
            {
                "group": group,
                "line": first["line"],
                "method": first["method"],
                "source_point_count": first["source_point_count"],
                "gt_point_count": first["gt_point_count"],
                "cd_mean": cd_m,
                "cd_std": cd_s,
                "cd_forward_mean": cd_f_m,
                "cd_backward_mean": cd_b_m,
                "hd_mean": hd_m,
                "hd_std": hd_s,
                "hd_forward_mean": hd_f_m,
                "hd_backward_mean": hd_b_m,
                "nuc_mean": nuc_m,
                "nuc_std": nuc_s,
                "nuc_r1_mean": nuc_r1,
                "nuc_r2_mean": nuc_r2,
                "nuc_r3_mean": nuc_r3,
                "p2f_mean": p2f_m,
                "p2f_std": p2f_s,
                "p2f_median_mean": p2f_med,
                "p2f_95_mean": p2f_95,
                "p2f_max_mean": p2f_max,
                "p2f_status": p2f_status,
                "valid_samples": len(subset),
            }
        )
    return summaries


def row_label(group: str, method: str) -> str:
    if group == "original_baseline_vs_gt":
        return "Original baseline"
    if group == "downsampled_x4_baseline_vs_gt":
        return "Downsampled x4 baseline"
    if group.startswith("downsampled_x4_up_"):
        return f"Downsampled x4 + {METHOD_DISPLAY.get(method, method)}"
    if group.startswith("original_up_"):
        return f"Original + {METHOD_DISPLAY.get(method, method)}"
    return group


def build_original_vs_downsampled_up(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    main_groups = [
        "original_baseline_vs_gt",
        "downsampled_x4_up_ear_vs_gt",
        "downsampled_x4_up_pdans_vs_gt",
        "downsampled_x4_up_pu_net_vs_gt",
        "downsampled_x4_up_pu_gcn_vs_gt",
    ]
    by_group = {r["group"]: r for r in summary_rows}
    orig = by_group.get("original_baseline_vs_gt")
    if not orig:
        return []
    out: list[dict[str, Any]] = []
    for group in main_groups:
        row = by_group.get(group)
        if not row:
            continue
        def delta(key: str) -> float | str:
            if row is orig:
                return 0.0
            a, b = row.get(key), orig.get(key)
            if a in ("", None) or b in ("", None):
                return ""
            return float(a) - float(b)
        out.append(
            {
                "method": row_label(group, str(row["method"])),
                "source_point_count": row["source_point_count"],
                "gt_reference": f"dense_mesh_surface_{GT_POINT_COUNT}",
                "cd_total_mean": row["cd_mean"],
                "delta_cd_vs_original": delta("cd_mean"),
                "hd_total_mean": row["hd_mean"],
                "delta_hd_vs_original": delta("hd_mean"),
                "nuc_mean": row["nuc_mean"],
                "delta_nuc_vs_original": delta("nuc_mean"),
                "p2f_mean": row["p2f_mean"],
                "delta_p2f_vs_original": delta("p2f_mean"),
                "p2f_status": row["p2f_status"],
                "valid_samples": row["valid_samples"],
            }
        )
    return out


def build_extended_comparison(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = [
        "original_baseline_vs_gt",
        "downsampled_x4_baseline_vs_gt",
        "downsampled_x4_up_ear_vs_gt",
        "downsampled_x4_up_pdans_vs_gt",
        "downsampled_x4_up_pu_net_vs_gt",
        "downsampled_x4_up_pu_gcn_vs_gt",
        "original_up_ear_vs_gt",
        "original_up_pdans_vs_gt",
        "original_up_pu_net_vs_gt",
        "original_up_pu_gcn_vs_gt",
    ]
    by_group = {r["group"]: r for r in summary_rows}
    out: list[dict[str, Any]] = []
    for group in order:
        row = by_group.get(group)
        if not row:
            continue
        out.append(
            {
                "method": row_label(group, str(row["method"])),
                "group": group,
                "line": row["line"],
                "source_point_count": row["source_point_count"],
                "gt_reference": f"dense_mesh_surface_{GT_POINT_COUNT}",
                "cd_total_mean": row["cd_mean"],
                "hd_total_mean": row["hd_mean"],
                "nuc_mean": row["nuc_mean"],
                "p2f_mean": row["p2f_mean"],
                "p2f_status": row["p2f_status"],
                "valid_samples": row["valid_samples"],
            }
        )
    return out


def write_comparison_md(path: Path, title: str, rows: list[dict[str, Any]], extra_note: str = "") -> None:
    lines = [f"# {title}", "", f"- Generated at: {utc_now()}", ""]
    if extra_note:
        lines.extend([extra_note, ""])
    if not rows:
        lines.append("_No data._")
    else:
        headers = list(rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics_definition_report(p2f_status: str) -> Path:
    path = reports_dir() / "modelnet40_quality_metrics_definition.md"
    text = f"""# ModelNet40 Geometric Quality Metrics Definition

- Generated at: {utc_now()}

## GT reference

All CD/HD comparisons use a **dense ground-truth point cloud** sampled from the original ModelNet40 `.off` mesh surface:

- Path: `quality_metrics/gt_reference_points_{GT_POINT_COUNT}/{{split}}/{{class}}/{{shape_id}}.npy`
- Point count: **{GT_POINT_COUNT}** per shape
- Sampling: triangle-area weighted surface sampling, unit-sphere normalization (same convention as `modelnet40_original`)
- Seed: global={GLOBAL_SEED}, per-sample deterministic via `stable_seed`

**Why not use original 1024 as GT?** If target = original 1024, then Original baseline CD/HD vs itself would be **0**, which is not informative. Dense mesh-sampled GT measures deviation from the underlying continuous surface.

## CD (Chamfer Distance) — lower is better

- `cd_forward` = mean nearest-neighbor **L2** distance from source → GT
- `cd_backward` = mean nearest-neighbor **L2** distance from GT → source
- `cd_total` = `cd_forward + cd_backward`
- Uses Euclidean (L2) distance, **not** squared L2.

## HD (Hausdorff Distance) — lower is better

- `hd_forward` = max nearest-neighbor L2 distance from source → GT
- `hd_backward` = max nearest-neighbor L2 distance from GT → source
- `hd_total` = max(`hd_forward`, `hd_backward`)

## P2F (Point-to-Surface Distance) — lower is better

Measures distance from each source point to the original `.off` mesh surface.

- `p2f_mean`, `p2f_median`, `p2f_95`, `p2f_max`
- Current status: **{p2f_status}**
- Preferred: Open3D `RaycastingScene` exact unsigned distance
- Fallback: approximate P2F using {P2F_APPROX_SURFACE_POINTS} mesh surface sample points + nearest-neighbor distance

## NUC (Neighborhood Uniformity Coefficient) — lower is more uniform

- Query centers: **{NUC_QUERY_CENTERS}** (fixed seed per sample)
- Radii: **{NUC_RADII}**
- At each center, count neighbors within radius; compute coefficient of variation `std/mean` across centers
- `nuc_r1`, `nuc_r2`, `nuc_r3` at the three radii; `nuc_mean` = mean of the three

## Main vs supplementary groups

**Main groups** (primary paper comparison: Original baseline vs Downsampled×4 + Upsampling):

1. `original_baseline_vs_gt` — original 1024 pts
2. `downsampled_x4_up_{{ear,pdans,pu_net,pu_gcn}}_vs_gt` — Line B upsampled 1024 pts

**Supplementary groups**:

- `downsampled_x4_baseline_vs_gt` — downsampled 256 pts (degradation from downsampling)
- `original_up_{{method}}_vs_gt` — Line A upsampled 4096 pts
"""
    path.write_text(text, encoding="utf-8")
    return path


def _fnum(val: Any, default: float = float("nan")) -> float:
    if val in ("", None):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def update_final_report(
    main_summary: list[dict[str, Any]],
    supp_summary: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    p2f_status: str,
) -> Path:
    path = reports_dir() / "modelnet40_x4_final_protocol_report.md"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if "## ModelNet40 geometric quality metrics" in existing:
        existing = existing.split("## ModelNet40 geometric quality metrics")[0].rstrip()

    lines = [
        "",
        "## ModelNet40 geometric quality metrics",
        "",
        f"- Updated at: {utc_now()}",
        "",
        "### Primary comparison",
        "",
        "**Original baseline (1024 pts)** vs **Downsampled ×4 + Upsampling (1024 pts)**",
        "All metrics computed against dense GT ({GT_POINT_COUNT} surface points from `.off` mesh).".format(
            GT_POINT_COUNT=GT_POINT_COUNT
        ),
        "",
        "### Main groups",
        "",
        "| Method | CD total | HD total | NUC | P2F |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['method']} | {_fnum(row['cd_total_mean']):.6f} | {_fnum(row['hd_total_mean']):.6f} | "
            f"{_fnum(row['nuc_mean']):.6f} | {row.get('p2f_mean', '')} |"
        )

    lines.extend(
        [
            "",
            "### Supplementary groups",
            "",
            "- Downsampled ×4 baseline (256 pts)",
            "- Original + Upsampling methods (4096 pts): EAR, PDANS, PU-Net, PU-GCN",
            "",
            "### Metric definitions",
            "",
            "- **CD/HD**: source vs dense mesh-sampled GT (L2, lower better)",
            f"- **P2F**: source vs original `.off` mesh surface ({p2f_status})",
            "- **NUC**: local density uniformity (lower better)",
            "",
            "### GT reference",
            "",
            f"`quality_metrics/gt_reference_points_{GT_POINT_COUNT}/` — triangle-area weighted surface sampling, seed={GLOBAL_SEED}",
            "",
            "### Output files",
            "",
            "- `reports/modelnet40_main_quality_metrics_summary.csv`",
            "- `reports/modelnet40_original_vs_downsampled_up_metrics_summary.csv`",
            "- `reports/modelnet40_quality_metrics_extended_comparison_summary.csv`",
            "- `reports/modelnet40_quality_metrics_definition.md`",
            "",
        ]
    )

    if supp_summary:
        lines.append("### Supplementary summary snapshot")
        lines.append("")
        lines.append("| group | CD | HD | NUC |")
        lines.append("| --- | ---: | ---: | ---: |")
        for row in supp_summary:
            lines.append(
                f"| {row['group']} | {_fnum(row['cd_mean']):.6f} | {_fnum(row['hd_mean']):.6f} | {_fnum(row['nuc_mean']):.6f} |"
            )
        lines.append("")

    path.write_text(existing + "\n".join(lines), encoding="utf-8")
    return path


def p2f_smoke_test(n_samples: int = 20) -> tuple[str, float]:
    manifest = load_manifest_rows()[:n_samples]
    p2f = P2FComputer(mode="exact")
    cKDTree_cls = try_import_ckdtree()
    t0 = time.time()
    ok = 0
    for row in manifest:
        key = rel_key(row["split"], row["class_name"], row["shape_id"])
        src = ORIGINAL_ROOT / row["split"] / row["class_name"] / f"{row['shape_id']}.npy"
        pts = np.load(src).astype(np.float32)
        out = p2f.compute(row["source_off"], pts, cKDTree_cls)
        if out["p2f_status"] == "exact":
            ok += 1
    elapsed = time.time() - t0
    per_sample = elapsed / max(len(manifest), 1)
    return ("exact" if ok == len(manifest) else "approximate"), per_sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=("all", "audit", "gt", "metrics", "reports"), default="all")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--smoke", action="store_true", help="20 samples per group")
    parser.add_argument("--p2f", choices=("auto", "exact", "approx", "skip"), default="auto")
    parser.add_argument("--gt-workers", type=int, default=8)
    parser.add_argument("--metrics-workers", type=int, default=8)
    parser.add_argument("--skip-gt", action="store_true", help="Skip GT generation if exists")
    args = parser.parse_args()

    n = detect_original_point_count()
    groups = define_source_groups(n)
    max_samples = 20 if args.smoke else args.max_samples

    audit_csv = audit_md = None
    gt_csv = gt_md = None

    if args.step in ("all", "audit"):
        print("Step 1: source audit...")
        audit_rows = run_source_audit(groups)
        audit_csv, audit_md = write_source_audit_reports(audit_rows)
        print(f"  wrote {audit_csv}")

    if args.step in ("all", "gt") and not args.skip_gt:
        print("Step 2: GT reference points...")
        gt_results = build_gt_reference(workers=args.gt_workers)
        gt_csv, gt_md = write_gt_audit_reports(gt_results)
        failed = sum(1 for r in gt_results if r["status"] == "failed")
        print(f"  GT done, failed={failed}, audit={gt_csv}")
        if failed:
            print("WARNING: some GT samples failed")

    off_map = {
        rel_key(r["split"], r["class_name"], r["shape_id"]): r["source_off"]
        for r in load_manifest_rows()
    }

    main_rows: list[dict[str, Any]] = []
    supp_rows: list[dict[str, Any]] = []
    p2f_status = "pending_slow_mesh_distance"

    if args.step in ("all", "metrics"):
        if args.p2f == "auto" and not args.smoke:
            mode, per_sample = p2f_smoke_test(20)
            est_hours = per_sample * 12311 * 10 / 3600
            print(f"P2F smoke: mode={mode}, ~{per_sample:.3f}s/sample, est full ~{est_hours:.1f}h")
            if est_hours > 6 and mode == "exact":
                print("  Using exact P2F (within reasonable budget)")
        p2f_mode = args.p2f if args.p2f != "auto" else "auto"
        if args.p2f == "auto" and not args.smoke:
            mode, per_sample = p2f_smoke_test(20)
            est_hours = per_sample * 12311 * 10 / 3600
            print(f"P2F smoke: mode={mode}, ~{per_sample:.3f}s/sample, est full ~{est_hours:.1f}h")
            p2f_mode = mode

        metrics_workers = 1 if args.smoke else args.metrics_workers
        for group in groups:
            print(f"Computing {group.group} ({group.tier})...")
            rows = compute_group_metrics(
                group, off_map, max_samples, p2f_mode, workers=metrics_workers
            )
            if group.tier == "main":
                main_rows.extend(rows)
            else:
                supp_rows.extend(rows)
            print(f"  {len(rows)} samples")

        if args.p2f == "skip":
            p2f_status = "pending_slow_mesh_distance"
        elif main_rows:
            statuses = {r.get("p2f_status") for r in main_rows + supp_rows}
            if statuses == {"exact"} or statuses <= {"exact"}:
                p2f_status = "exact"
            elif "exact" in statuses:
                p2f_status = "exact"
            elif "approximate" in statuses:
                p2f_status = "approximate"
            elif "pending_slow_mesh_distance" in statuses:
                p2f_status = "pending_slow_mesh_distance"

        main_ps = reports_dir() / "modelnet40_main_quality_metrics_per_sample.csv"
        main_sum = reports_dir() / "modelnet40_main_quality_metrics_summary.csv"
        write_csv(main_ps, main_rows)
        write_csv(main_sum, summarize_rows(main_rows))
        print(f"  main per-sample: {main_ps}")

        if supp_rows:
            supp_ps = reports_dir() / "modelnet40_supplementary_quality_metrics_per_sample.csv"
            supp_sum = reports_dir() / "modelnet40_supplementary_quality_metrics_summary.csv"
            write_csv(supp_ps, supp_rows)
            write_csv(supp_sum, summarize_rows(supp_rows))
            print(f"  supplementary summary: {supp_sum}")

    if args.step in ("all", "reports", "metrics"):
        main_summary = []
        supp_summary = []
        main_sum_path = reports_dir() / "modelnet40_main_quality_metrics_summary.csv"
        supp_sum_path = reports_dir() / "modelnet40_supplementary_quality_metrics_summary.csv"
        if main_sum_path.is_file():
            with open(main_sum_path, newline="", encoding="utf-8") as handle:
                main_summary = list(csv.DictReader(handle))
        if supp_sum_path.is_file():
            with open(supp_sum_path, newline="", encoding="utf-8") as handle:
                supp_summary = list(csv.DictReader(handle))

        all_summary = main_summary + supp_summary
        comp_rows = build_original_vs_downsampled_up(all_summary)
        ext_rows = build_extended_comparison(all_summary)

        comp_csv = reports_dir() / "modelnet40_original_vs_downsampled_up_metrics_summary.csv"
        comp_md = reports_dir() / "modelnet40_original_vs_downsampled_up_metrics_summary.md"
        write_csv(comp_csv, comp_rows)
        write_comparison_md(
            comp_md,
            "Original vs Downsampled x4 + Upsampling Quality Metrics",
            comp_rows,
            "Delta columns = method − Original baseline. Positive delta = worse than original.",
        )

        ext_csv = reports_dir() / "modelnet40_quality_metrics_extended_comparison_summary.csv"
        ext_md = reports_dir() / "modelnet40_quality_metrics_extended_comparison_summary.md"
        write_csv(ext_csv, ext_rows)
        write_comparison_md(ext_md, "Extended Quality Metrics Comparison", ext_rows)

        def_path = write_metrics_definition_report(p2f_status)
        final_path = update_final_report(main_summary, supp_summary, comp_rows, p2f_status)
        print(f"  comparison: {comp_csv}")
        print(f"  definition: {def_path}")
        print(f"  final report: {final_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
