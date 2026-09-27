#!/usr/bin/env python3
"""Export thesis-meeting KITTI full-frame + object-crop visualizations.

Consumes existing point cloud outputs only. Does not run upsampling, detection,
or KITTI preprocessing.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KITTI_ROOT = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
IMAGESETS_ROOT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "interactive_object_crop_visualization_improved_v6"
PREFERRED_FRAMES = ["000001", "003219", "006833", "007458"]
NEW_THRESHOLDS = [0.05, 0.08, 0.10]
DEFAULT_NEW_THRESHOLD = 0.10
SUMMARY_FIELDS = [
    "method", "line", "frame_id", "object_id", "class_name", "distance_x",
    "bbox_h", "bbox_w", "bbox_l", "bbox_x", "bbox_y", "bbox_z", "bbox_rotation_y",
    "margin", "base_full_points", "upsampled_full_points", "base_crop_points",
    "upsampled_crop_points", "new_points_count", "new_points_ratio",
    "new_points_inside_bbox_ratio", "outside_bbox_ratio", "artifact_score",
    "structure_score_base", "change_visibility_score", "diversity_group",
    "new_point_threshold", "anchor_group", "anchor_line", "anchor_rank",
    "is_shared_anchor", "methods_available_for_same_anchor",
    "base_pointcloud_path", "upsampled_pointcloud_path",
    "full_frame_html_path", "object_crop_html_path", "full_frame_png_path",
    "object_crop_png_path", "metadata_json_path", "selected_rank", "is_recommended",
    "recommended_reason", "rejected_reason",
]


@dataclass(frozen=True)
class KittiObject:
    object_id: int
    class_name: str
    truncated: float
    occluded: int
    alpha: float
    bbox2d: Tuple[float, float, float, float]
    h: float
    w: float
    l: float
    x: float
    y: float
    z: float
    ry: float


@dataclass(frozen=True)
class Comparison:
    method: str
    line: str
    input_root: Path
    upsampled_root: Optional[Path]
    input_label: str
    upsampled_label: str
    input_color: str
    upsampled_color: str
    new_color: str
    smoke_only: bool = False
    missing_reason: str = ""


@dataclass
class Candidate:
    comp: Comparison
    frame_id: str
    obj: KittiObject
    input_path: Path
    upsampled_path: Path
    base_full_count: int
    up_full_count: int
    crop_min: np.ndarray
    crop_max: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    bbox_corners: np.ndarray
    base_crop: np.ndarray
    up_crop: np.ndarray
    new_points: np.ndarray
    new_threshold: float
    distance_x: float
    new_ratio: float
    inside_ratio: float
    outside_ratio: float
    artifact_score: float
    structure_score_base: float
    change_visibility_score: float
    diversity_group: str
    visibility_score: float
    recommended_reason: str


@dataclass(frozen=True)
class Anchor:
    line_kind: str
    rank: int
    frame_id: str
    obj: KittiObject
    score: float
    base_crop_points: int
    distance_x: float
    methods_available: Tuple[str, ...]
    note: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--kitti-root", type=Path, default=KITTI_ROOT)
    p.add_argument("--val-split", type=Path, default=IMAGESETS_ROOT / "val.txt")
    p.add_argument("--max-candidate-frames", type=int, default=260)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--min-base-points", type=int, default=80)
    p.add_argument("--min-upsampled-points", type=int, default=80)
    p.add_argument("--min-new-points", type=int, default=60)
    p.add_argument("--min-new-ratio", type=float, default=0.05)
    p.add_argument("--max-distance-x", type=float, default=40.0)
    p.add_argument("--per-line-candidates", type=int, default=6)
    p.add_argument("--meeting-limit", type=int, default=10)
    p.add_argument("--full-frame-display-points", type=int, default=70000)
    p.add_argument("--object-display-points", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260612)
    p.add_argument(
        "--add-pdans-only",
        action="store_true",
        help="Preserve existing v6 method dashboards and add/update only PDANS plus root reports.",
    )
    return p.parse_args()


def rel(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def read_frames(path: Path, limit: int) -> List[str]:
    frames = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    frames = [f"{int(x):06d}" if x.isdigit() else x for x in frames]
    out, seen = [], set()
    for f in PREFERRED_FRAMES + frames:
        if f not in seen:
            out.append(f); seen.add(f)
    return out[: max(limit, len(PREFERRED_FRAMES))]


def point_file(root: Optional[Path], frame_id: str) -> Optional[Path]:
    if root is None:
        return None
    for ext in (".bin", ".npy", ".ply", ".xyz"):
        p = root / f"{frame_id}{ext}"
        if p.exists():
            return p
    for p in sorted(root.glob(f"**/{frame_id}*")):
        if p.suffix.lower() in {".bin", ".npy", ".ply", ".xyz"}:
            return p
    return None


def load_ply_points(path: Path) -> np.ndarray:
    dtype_map = {
        "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
        "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
        "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
        "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    }
    with path.open("rb") as f:
        header: List[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"bad ply header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header.append(text)
            if text == "end_header":
                break
        fmt = next((x.split()[1] for x in header if x.startswith("format ")), "")
        vertex_count = 0
        props: List[Tuple[str, str]] = []
        in_vertex = False
        for line in header:
            parts = line.split()
            if len(parts) >= 3 and parts[:2] == ["element", "vertex"]:
                vertex_count = int(parts[2]); in_vertex = True; continue
            if len(parts) >= 2 and parts[0] == "element" and parts[1] != "vertex":
                in_vertex = False
            if in_vertex and len(parts) == 3 and parts[0] == "property":
                props.append((parts[2], parts[1]))
        if not vertex_count or not props:
            raise ValueError(f"missing vertex data in ply: {path}")
        names = [p[0] for p in props]
        if fmt == "ascii":
            arr = np.loadtxt(f, dtype=np.float32, max_rows=vertex_count)
            arr = np.atleast_2d(arr)
            cols = {name: arr[:, i] for i, name in enumerate(names)}
        elif fmt == "binary_little_endian":
            dtype = np.dtype([(name, dtype_map[typ]) for name, typ in props])
            raw = np.fromfile(f, dtype=dtype, count=vertex_count)
            cols = {name: raw[name].astype(np.float32) for name in names}
        else:
            raise ValueError(f"unsupported ply format {fmt}: {path}")
    xyz = [cols.get(k) for k in ("x", "y", "z")]
    if any(v is None for v in xyz):
        raise ValueError(f"ply lacks x/y/z properties: {path}")
    intensity = cols.get("intensity", cols.get("i", np.zeros(vertex_count, dtype=np.float32)))
    return np.column_stack([xyz[0], xyz[1], xyz[2], intensity]).astype(np.float32)


def load_points(path: Path) -> np.ndarray:
    if path.suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4:
            raise ValueError(f"bad bin shape: {path}")
        arr = raw.reshape(-1, 4)
    elif path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".xyz":
        arr = np.loadtxt(path, dtype=np.float32)
    elif path.suffix == ".ply":
        arr = load_ply_points(path)
    else:
        raise ValueError(f"unsupported point format: {path}")
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"expected Nx3/Nx4: {path}")
    arr = arr.astype(np.float32, copy=False)
    if arr.shape[1] == 3:
        arr = np.column_stack([arr, np.zeros(len(arr), dtype=np.float32)])
    return arr[:, :4]


def parse_labels(path: Path) -> List[KittiObject]:
    out: List[KittiObject] = []
    if not path.exists():
        return out
    for i, line in enumerate(path.read_text().splitlines()):
        p = line.split()
        if len(p) < 15:
            continue
        out.append(KittiObject(i, p[0], float(p[1]), int(float(p[2])), float(p[3]), tuple(float(x) for x in p[4:8]), float(p[8]), float(p[9]), float(p[10]), float(p[11]), float(p[12]), float(p[13]), float(p[14])))
    return out


def read_calib(path: Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            data[k] = np.array([float(x) for x in v.split()], dtype=np.float64)
    r0 = np.eye(4); r0[:3, :3] = data["R0_rect"].reshape(3, 3)
    v2c = np.eye(4); v2c[:3, :4] = data["Tr_velo_to_cam"].reshape(3, 4)
    data["rect_to_velo"] = np.linalg.inv(r0 @ v2c)
    return data


def camera_corners(obj: KittiObject) -> np.ndarray:
    x = np.array([obj.l/2, obj.l/2, -obj.l/2, -obj.l/2, obj.l/2, obj.l/2, -obj.l/2, -obj.l/2])
    y = np.array([0, 0, 0, 0, -obj.h, -obj.h, -obj.h, -obj.h])
    z = np.array([obj.w/2, -obj.w/2, -obj.w/2, obj.w/2, obj.w/2, -obj.w/2, -obj.w/2, obj.w/2])
    rot = np.array([[math.cos(obj.ry), 0, math.sin(obj.ry)], [0, 1, 0], [-math.sin(obj.ry), 0, math.cos(obj.ry)]])
    return np.vstack([x, y, z]).T @ rot.T + np.array([obj.x, obj.y, obj.z])


def rect_to_velo(points: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    hom = np.hstack([points, np.ones((len(points), 1))])
    return (hom @ calib["rect_to_velo"].T)[:, :3].astype(np.float32)


def crop_bounds(obj: KittiObject, calib: Dict[str, np.ndarray], margin: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    corners = rect_to_velo(camera_corners(obj), calib)
    bmin, bmax = corners.min(axis=0), corners.max(axis=0)
    return bmin - margin, bmax + margin, bmin, bmax, corners


def crop(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    m = np.all((points[:, :3] >= lo) & (points[:, :3] <= hi), axis=1)
    return points[m]


def inside_mask(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    return np.all((points[:, :3] >= lo) & (points[:, :3] <= hi), axis=1)


def build_comps(kitti_root: Path) -> List[Comparison]:
    def r(name: str) -> Path: return kitti_root / name
    original = r("velodyne_original_val")
    down = r("velodyne_downsampled_50_val")
    spupmd = PROJECT_ROOT / "results" / "pdans_spupmd_lab_feasibility_audit" / "spupmd_smoke_test" / "converted_kitti_bin_zero_intensity"
    pdans = PROJECT_ROOT / "results" / "pdans_main_kitti_pipeline"
    return [
        Comparison("EAR", "line_a_original_vs_ear", original, r("velodyne_ear_val"), "original", "original + EAR", "#003399", "#ff0000", "#ff0000"),
        Comparison("EAR", "line_b_downsampled_vs_ear", down, r("velodyne_downsampled_50_ear_val"), "downsampled", "downsampled + EAR", "#003399", "#00e84d", "#00ff40"),
        Comparison("PU-Net", "line_a_original_vs_punet", original, r("velodyne_punet_x2_fullframe"), "original", "original + PU-Net", "#003399", "#ff0000", "#ff0000"),
        Comparison("PU-Net", "line_b_downsampled_vs_punet", down, r("velodyne_downsampled_50_punet_x2_fullframe"), "downsampled", "downsampled + PU-Net", "#003399", "#00e84d", "#00ff40"),
        Comparison("PU-GCN", "line_a_original_vs_pugcn", original, r("pugcn_cap_100k"), "original", "original + PU-GCN", "#003399", "#ff0000", "#ff0000"),
        Comparison("PU-GCN", "line_b_downsampled_vs_pugcn", down, r("pugcn_cap_100k_downsampled50_filled"), "downsampled", "downsampled + PU-GCN", "#003399", "#00e84d", "#00ff40"),
        Comparison("TULIP", "line_a_original_vs_tulip", original, r("tulip_original_up_bin"), "original", "original + TULIP", "#003399", "#ff0000", "#ff0000"),
        Comparison("TULIP", "line_b_downsampled_vs_tulip", down, r("tulip_downsampled_up_bin"), "downsampled", "downsampled + TULIP", "#003399", "#00e84d", "#00ff40"),
        Comparison("SPU-PMD", "line_a_original_vs_spupmd_smoke4", original, spupmd, "original", "original + SPU-PMD smoke-only", "#003399", "#ff0000", "#ff0000", True),
        Comparison("SPU-PMD", "line_b_downsampled_vs_spupmd", down, None, "downsampled", "downsampled + SPU-PMD", "#003399", "#00e84d", "#00ff40", True, "SPU-PMD line B converted full-frame output not found"),
        Comparison("PDANS", "line_a_original_vs_pdans", original, pdans / "pdans_original_up_bin", "original", "original + PDANS", "#003399", "#ff0000", "#ff0000"),
        Comparison("PDANS", "line_b_downsampled_vs_pdans", down, pdans / "pdans_downsampled50_up_bin", "downsampled", "downsampled + PDANS", "#003399", "#00e84d", "#00ff40"),
    ]


def new_points(base: np.ndarray, up: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(base) == 0:
        return up, np.full(len(up), np.inf, dtype=np.float32)
    dist, _ = cKDTree(base[:, :3]).query(up[:, :3], k=1, workers=-1)
    return up[dist > threshold], dist.astype(np.float32)


def rng(points: np.ndarray, axis: int) -> float:
    return 0.0 if len(points) == 0 else float(points[:, axis].max() - points[:, axis].min())


def scores(base: np.ndarray, new: np.ndarray, nn: np.ndarray, crop_lo: np.ndarray, crop_hi: np.ndarray, bbox_lo: np.ndarray, bbox_hi: np.ndarray, dist_x: float) -> Tuple[float, float, float, float, float, str]:
    dims = np.maximum(crop_hi - crop_lo, 1e-6)
    base_spread = float(np.mean([rng(base, i) / dims[i] for i in range(3)])) if len(base) else 0.0
    base_score = 60 * min(len(base) / 300, 1) + 40 * min(base_spread, 1)
    inside = inside_mask(new, bbox_lo, bbox_hi)
    inside_ratio = float(inside.mean()) if len(new) else 0.0
    outside_ratio = 1.0 - inside_ratio if len(new) else 1.0
    ranges = np.array([rng(new, i) / dims[i] for i in range(3)], dtype=np.float32) if len(new) else np.zeros(3)
    coverage = float(np.clip(np.mean(ranges), 0, 1))
    nn_new = nn[nn > DEFAULT_NEW_THRESHOLD]
    nn_mean = float(nn_new.mean()) if len(nn_new) else 0.0
    change_score = 45 * min(len(new) / 800, 1) + 35 * coverage + 20 * inside_ratio + 10 * min(nn_mean / 0.3, 1)
    artifact = 100 * outside_ratio + max(0.0, float(ranges.max()) - 0.95) * 50
    group = ("near" if dist_x < 20 else "mid" if dist_x < 30 else "far") + "_" + ("inside" if inside_ratio > 0.65 else "mixed" if inside_ratio > 0.35 else "outside") + "_" + ("spread" if coverage > 0.35 else "local")
    return base_score, change_score, inside_ratio, outside_ratio, artifact, group


def reject(args: argparse.Namespace, base_n: int, up_n: int, new_n: int, new_ratio: float, inside: float, artifact: float, base_score: float, change: float, dist: float) -> str:
    if base_n < args.min_base_points: return f"base crop too sparse ({base_n})"
    if up_n < args.min_upsmpled_points if False else False: return ""
    if up_n < args.min_upsampled_points: return f"upsampled crop too sparse ({up_n})"
    if dist > args.max_distance_x: return f"object too far ({dist:.1f}m)"
    if base_score < 35: return f"object not recognizable in base view (structure_score={base_score:.1f})"
    if new_n < args.min_new_points: return f"too few visible new points ({new_n})"
    if new_ratio < args.min_new_ratio: return f"new point ratio too low ({new_ratio:.3f})"
    if inside < 0.25: return f"too many new points outside bbox (inside_ratio={inside:.2f})"
    if artifact > 95: return f"artifact/outside-bbox score too high ({artifact:.1f})"
    if change < 35: return f"change not clearly visible (change_score={change:.1f})"
    return ""


def sample(points: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if len(points) <= max_n: return points
    g = np.random.default_rng(seed)
    idx = g.choice(len(points), size=max_n, replace=False); idx.sort()
    return points[idx]


def candidate_for(comp: Comparison, frame: str, obj: KittiObject, ip: Path, up: Path, base_full: np.ndarray, up_full: np.ndarray, calib: Dict[str, np.ndarray], args: argparse.Namespace) -> Tuple[Optional[Candidate], str]:
    crop_lo, crop_hi, bbox_lo, bbox_hi, corners = crop_bounds(obj, calib, args.margin)
    bc, uc = crop(base_full, crop_lo, crop_hi), crop(up_full, crop_lo, crop_hi)
    npnts, nn = new_points(bc, uc, DEFAULT_NEW_THRESHOLD)
    ratio = len(npnts) / max(len(uc), 1)
    base_s, change_s, inside, outside, artifact, group = scores(bc, npnts, nn, crop_lo, crop_hi, bbox_lo, bbox_hi, abs(obj.z))
    why = reject(args, len(bc), len(uc), len(npnts), ratio, inside, artifact, base_s, change_s, abs(obj.z))
    if why: return None, why
    visibility = 2*base_s + 3*change_s + min(len(npnts), 1600)*0.8 + 90*min(ratio, 0.7) + 80*inside - 8*abs(obj.z) - 0.5*artifact
    reason = f"base structure visible; {len(npnts)} new points ({ratio:.1%}); inside bbox {inside:.1%}; artifact score {artifact:.1f}; full-frame location clear; object-level new points are interpretable"
    return Candidate(comp, frame, obj, ip, up, len(base_full), len(up_full), crop_lo, crop_hi, bbox_lo, bbox_hi, corners, bc, uc, npnts, DEFAULT_NEW_THRESHOLD, abs(obj.z), ratio, inside, outside, artifact, base_s, change_s, group, visibility, reason), ""


def scan(comp: Comparison, frames: Sequence[str], args: argparse.Namespace) -> Tuple[List[Candidate], Dict[str, int], List[Dict[str, object]]]:
    stats = {"missing": 0, "read_error": 0, "rejected": 0, "accepted": 0}
    rejected_rows: List[Dict[str, object]] = []
    cands: List[Candidate] = []
    if comp.upsampled_root is None:
        stats["missing"] = len(frames)
        return [], stats, [blank_row(comp, comp.missing_reason or "upsampled root missing")]
    for frame in frames:
        ip, up = point_file(comp.input_root, frame), point_file(comp.upsampled_root, frame)
        if not ip or not up:
            stats["missing"] += 1; continue
        try:
            bf, uf = load_points(ip), load_points(up)
            calib = read_calib(args.kitti_root / "calib" / f"{frame}.txt")
            objs = [o for o in parse_labels(args.kitti_root / "label_2" / f"{frame}.txt") if o.class_name == "Car"]
        except Exception as e:
            stats["read_error"] += 1
            if len(rejected_rows) < 20: rejected_rows.append(blank_row(comp, f"read error {frame}: {e}"))
            continue
        for obj in objs:
            cand, why = candidate_for(comp, frame, obj, ip, up, bf, uf, calib, args)
            if cand:
                cands.append(cand); stats["accepted"] += 1
            else:
                stats["rejected"] += 1
                if len(rejected_rows) < 20: rejected_rows.append(blank_row(comp, why, frame, obj))
    cands.sort(key=lambda c: c.visibility_score, reverse=True)
    return cands, stats, rejected_rows


def blank_row(comp: Comparison, reason: str, frame: str = "", obj: Optional[KittiObject] = None) -> Dict[str, object]:
    row = {k: "" for k in SUMMARY_FIELDS}
    row.update({"method": comp.method, "line": comp.line, "frame_id": frame, "object_id": "" if obj is None else obj.object_id, "class_name": "" if obj is None else obj.class_name, "rejected_reason": reason})
    return row


def point_payload(points: np.ndarray) -> Dict[str, List[float]]:
    return {"x": np.round(points[:,0], 3).tolist(), "y": np.round(points[:,1], 3).tolist(), "z": np.round(points[:,2], 3).tolist(), "i": np.round(points[:,3], 3).tolist()}


def bbox_payload(corners: np.ndarray) -> Dict[str, List[Optional[float]]]:
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    x=[]; y=[]; z=[]
    for a,b in edges:
        x += [float(corners[a,0]), float(corners[b,0]), None]
        y += [float(corners[a,1]), float(corners[b,1]), None]
        z += [float(corners[a,2]), float(corners[b,2]), None]
    return {"x": x, "y": y, "z": z}


def write_interactive(path: Path, title: str, payload: Dict[str, object], full_frame: bool) -> None:
    default_mode = str(payload.get("default_mode", "bbox" if full_frame else "new"))
    sizes = {"base": 0.75 if full_frame else 1.15, "up": 0.55 if full_frame else 0.9, "new": 1.0 if full_frame else 1.2}
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{margin:0;background:#e8eef4;font-family:Arial,sans-serif;color:#1f2933}}.bar{{padding:9px 12px;background:#d9e3ec;border-bottom:1px solid #b4c1ce}}button{{margin-right:6px;padding:6px 9px;border:1px solid #8da1b5;background:#f8fafc;border-radius:4px;cursor:pointer}}button.active{{background:#1f2933;color:white}}#plot{{width:100vw;height:80vh}}.meta{{padding:10px 14px;font-size:13px;background:#f8fafc;border-top:1px solid #b4c1ce;line-height:1.45}}code{{white-space:pre-wrap}}</style></head>
<body><div class="bar"><button id="btn-base" onclick="render('base')">Base only</button><button id="btn-up" onclick="render('up')">Upsampled only</button><button id="btn-overlay" onclick="render('overlay')">Overlay</button><button id="btn-new" onclick="render('new')">New points</button><button id="btn-bbox" onclick="render('bbox')">BBox highlight</button></div><div id="plot"></div><div class="meta"><strong>{html.escape(title)}</strong><br>{html.escape(str(payload['reason']))}<br>Base: <code>{html.escape(str(payload['base_path']))}</code><br>Upsampled: <code>{html.escape(str(payload['up_path']))}</code></div>
<script>
const P={json.dumps(payload, separators=(",", ":"))};
const S={json.dumps(sizes)};
function scat(p,n,c,s,o){{return {{type:'scatter3d',mode:'markers',name:n,x:p.x,y:p.y,z:p.z,customdata:p.i,marker:{{size:s,color:c,opacity:o}},hovertemplate:'x=%{{x:.2f}}<br>y=%{{y:.2f}}<br>z=%{{z:.2f}}<br>i=%{{customdata:.2f}}<extra>%{{fullData.name}}</extra>'}}}}
function bbox(){{return {{type:'scatter3d',mode:'lines',name:'selected GT 3D bbox',x:P.bbox.x,y:P.bbox.y,z:P.bbox.z,line:{{color:'#ffd400',width:7}},hoverinfo:'skip'}}}}
function layout(mode){{return {{title:{{text:P.title+' | '+mode,x:0.02}},paper_bgcolor:'#e8eef4',plot_bgcolor:'#e8eef4',margin:{{l:0,r:0,b:0,t:74}},legend:{{itemsizing:'constant'}},scene:{{aspectmode:'data',xaxis:{{title:'x',backgroundcolor:'#eef3f7',gridcolor:'#cbd5df'}},yaxis:{{title:'y',backgroundcolor:'#eef3f7',gridcolor:'#cbd5df'}},zaxis:{{title:'z',backgroundcolor:'#eef3f7',gridcolor:'#cbd5df'}},camera:{{eye:{{x:1.45,y:-1.65,z:1.05}}}}}}}}}}
function traces(m){{if(m==='base')return[scat(P.base,P.base_label,P.base_color,S.base,0.9),bbox()];if(m==='up')return[scat(P.up,P.up_label,P.up_color,S.up,0.9),bbox()];if(m==='overlay')return[scat(P.base,P.base_label,P.base_color,S.base,0.82),scat(P.up,P.up_label,P.up_color,S.up,0.62),bbox()];if(m==='new')return[scat(P.base,'base reference','#315a9f',S.base,0.42),scat(P.new,'new points',P.new_color,S.new,0.92),bbox()];return[scat(P.base,P.base_label,P.base_color,S.base,0.75),scat(P.new,'new points near selected object',P.new_color,S.new,0.92),bbox()]}}
function render(m){{document.querySelectorAll('button').forEach(b=>b.classList.remove('active'));document.getElementById('btn-'+m).classList.add('active');Plotly.react('plot',traces(m),layout(m),{{responsive:true,displaylogo:false}})}}render('{default_mode}');
</script></body></html>"""
    path.write_text(body, encoding="utf-8")


def hex_rgb(color: str) -> Tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def draw_png(path: Path, title: str, base: np.ndarray, new: np.ndarray, bbox: np.ndarray, full: bool, new_color: str) -> None:
    w, h = 1200, 850
    img = Image.new("RGB", (w, h), (236, 242, 247)); d = ImageDraw.Draw(img)
    pts = base[:, [0,1]] if full else base[:, [0,1]]
    new2 = new[:, [0,1]] if len(new) else np.zeros((0,2))
    box = bbox[:, [0,1]]
    allp = np.vstack([pts, new2, box]) if len(new2) else np.vstack([pts, box])
    if len(allp) == 0:
        d.text((20,20), title + " (no points)", fill=(0,0,0)); img.save(path); return
    lo, hi = allp.min(axis=0), allp.max(axis=0); pad = np.maximum((hi-lo)*0.08, 1.0); lo -= pad; hi += pad
    def xy(p):
        x = int((p[0]-lo[0])/(hi[0]-lo[0]+1e-6)*(w-80)+40)
        y = int(h-50-(p[1]-lo[1])/(hi[1]-lo[1]+1e-6)*(h-110))
        return x,y
    b = sample(base, 90000 if full else 30000, 7)
    for p in b[:, :2]:
        x,y = xy(p); d.point((x,y), fill=(30,95,210))
    nc = hex_rgb(new_color)
    for p in new[:, :2]:
        x,y = xy(p); d.point((x,y), fill=nc)
    edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a,bb in edges:
        d.line([xy(bbox[a,[0,1]]), xy(bbox[bb,[0,1]])], fill=(230,180,0), width=3)
    d.text((20,15), title, fill=(20,30,40))
    d.text((20,h-28), "BEV x-y: dark blue=base, bright red/green=new points, yellow=selected GT bbox", fill=(20,30,40))
    img.save(path)


def write_sample(c: Candidate, out_dir: Path, rank: int, args: argparse.Namespace, collection: str = "recommended_samples", is_recommended: str = "yes") -> Dict[str, object]:
    folder = out_dir / collection / f"{slug(c.comp.method)}_{slug(c.comp.line)}_frame_{c.frame_id}_obj_{c.obj.object_id}_{c.obj.class_name}"
    folder.mkdir(parents=True, exist_ok=True)
    bf, uf = load_points(c.input_path), load_points(c.upsampled_path)
    full_base = sample(bf, args.full_frame_display_points, args.seed + rank)
    full_up = sample(uf, args.full_frame_display_points, args.seed + rank + 1)
    # full-frame new points are object-crop new points only, intentionally highlighting selected object.
    common = {
        "title": f"{c.comp.method} {c.line if False else c.comp.line} frame {c.frame_id} obj {c.obj.object_id} {c.obj.class_name}",
        "base_label": c.comp.input_label, "up_label": c.comp.upsampled_label,
        "base_color": c.comp.input_color, "up_color": c.comp.upsampled_color, "new_color": c.comp.new_color,
        "bbox": bbox_payload(c.bbox_corners), "reason": c.recommended_reason,
        "base_path": str(c.input_path), "up_path": str(c.upsampled_path),
    }
    full_payload = dict(common, base=point_payload(full_base), up=point_payload(full_up), new=point_payload(c.new_points))
    obj_payload = dict(common, base=point_payload(sample(c.base_crop, args.object_display_points, args.seed+rank+2)), up=point_payload(sample(c.up_crop, args.object_display_points, args.seed+rank+3)), new=point_payload(sample(c.new_points, args.object_display_points, args.seed+rank+4)))
    full_html, obj_html = folder/"full_frame.html", folder/"object_crop.html"
    full_png, obj_png = folder/"full_frame.png", folder/"object_crop.png"
    write_interactive(full_html, f"Full-frame {common['title']} selected object bbox", full_payload, True)
    write_interactive(obj_html, f"Object crop {common['title']} new-points view", obj_payload, False)
    draw_png(full_png, f"Full frame {c.comp.method} {c.comp.line} frame {c.frame_id} obj {c.obj.object_id}", full_base, c.new_points, c.bbox_corners, True, c.comp.new_color)
    draw_png(obj_png, f"Object crop {c.comp.method} {c.comp.line} frame {c.frame_id} obj {c.obj.object_id}", c.base_crop, c.new_points, c.bbox_corners, False, c.comp.new_color)
    meta = {
        "frame_id": c.frame_id, "object_id": c.obj.object_id, "class_name": c.obj.class_name,
        "method": c.comp.method, "line": c.comp.line, "distance_x": c.distance_x,
        "bbox": {"h": c.obj.h, "w": c.obj.w, "l": c.obj.l, "x": c.obj.x, "y": c.obj.y, "z": c.obj.z, "rotation_y": c.obj.ry},
        "base_point_cloud_path": str(c.input_path), "upsampled_point_cloud_path": str(c.upsampled_path),
        "base_crop_points": len(c.base_crop), "upsampled_crop_points": len(c.up_crop),
        "new_points_count": len(c.new_points), "new_points_ratio": c.new_ratio,
        "new_point_threshold": c.new_threshold, "selected_reason": c.recommended_reason,
    }
    (folder/"metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    dash = folder/"combined_dashboard.html"
    dash.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>Combined {html.escape(common['title'])}</title><style>body{{font-family:Arial;margin:20px;background:#f8fafc}}iframe{{width:100%;height:760px;border:1px solid #b4c1ce;margin-bottom:20px}}</style></head><body><h1>{html.escape(common['title'])}</h1><p>{html.escape(c.recommended_reason)}</p><h2>Full frame</h2><iframe src="full_frame.html"></iframe><h2>Object crop</h2><iframe src="object_crop.html"></iframe></body></html>""", encoding="utf-8")
    return row_for(c, rank, full_html, obj_html, full_png, obj_png, folder/"metadata.json", is_recommended, args.margin)


def row_for(c: Candidate, rank: int, full_html: Path, obj_html: Path, full_png: Path, obj_png: Path, meta: Path, rec: str, margin: float) -> Dict[str, object]:
    return {
        "method": c.comp.method, "line": c.comp.line, "frame_id": c.frame_id, "object_id": c.obj.object_id, "class_name": c.obj.class_name,
        "distance_x": f"{c.distance_x:.3f}", "bbox_h": c.obj.h, "bbox_w": c.obj.w, "bbox_l": c.obj.l, "bbox_x": c.obj.x, "bbox_y": c.obj.y, "bbox_z": c.obj.z, "bbox_rotation_y": c.obj.ry,
        "margin": margin, "base_full_points": c.base_full_count, "upsampled_full_points": c.up_full_count, "base_crop_points": len(c.base_crop), "upsampled_crop_points": len(c.up_crop),
        "new_points_count": len(c.new_points), "new_points_ratio": f"{c.new_ratio:.6f}", "new_points_inside_bbox_ratio": f"{c.inside_ratio:.6f}", "outside_bbox_ratio": f"{c.outside_ratio:.6f}",
        "artifact_score": f"{c.artifact_score:.3f}", "structure_score_base": f"{c.structure_score_base:.3f}", "change_visibility_score": f"{c.change_visibility_score:.3f}",
        "diversity_group": c.diversity_group, "new_point_threshold": c.new_threshold, "base_pointcloud_path": str(c.input_path), "upsampled_pointcloud_path": str(c.upsampled_path),
        "full_frame_html_path": str(full_html), "object_crop_html_path": str(obj_html), "full_frame_png_path": str(full_png), "object_crop_png_path": str(obj_png), "metadata_json_path": str(meta),
        "selected_rank": rank, "is_recommended": rec, "recommended_reason": c.recommended_reason, "rejected_reason": "",
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,"") for k in SUMMARY_FIELDS})


def clean(root: Path) -> None:
    if root.exists():
        for pat in ("**/*.html","**/*.csv","**/*.md","**/*.json","**/*.png"):
            for p in root.glob(pat):
                if p.is_file(): p.unlink()
    root.mkdir(parents=True, exist_ok=True)


def cand_key(c: Candidate) -> Tuple[str, str, str, int]:
    return (c.comp.method, c.comp.line, c.frame_id, c.obj.object_id)


def select_meeting(cands: List[Candidate], limit: int) -> List[Candidate]:
    by_method: Dict[str, List[Candidate]] = {}
    for c in cands: by_method.setdefault(c.comp.method, []).append(c)
    for v in by_method.values(): v.sort(key=lambda c:c.visibility_score, reverse=True)
    chosen: List[Candidate] = []
    groups: Dict[str,int] = {}
    used = set()
    for method in ["EAR","PU-Net","PU-GCN","TULIP","SPU-PMD"]:
        n=0
        for c in by_method.get(method, []):
            if n>=2: break
            if groups.get(c.diversity_group,0)>=3: continue
            chosen.append(c); used.add(cand_key(c)); groups[c.diversity_group]=groups.get(c.diversity_group,0)+1; n+=1
            if len(chosen)>=limit: return chosen
    by_line: Dict[Tuple[str, str], List[Candidate]] = {}
    for c in cands:
        by_line.setdefault((c.comp.method, c.comp.line), []).append(c)
    for vals in by_line.values():
        vals.sort(key=lambda c:c.visibility_score, reverse=True)
    for vals in by_line.values():
        for c in vals:
            if cand_key(c) not in used:
                chosen.append(c); used.add(cand_key(c))
                break
        if len(chosen)>=limit: return chosen
    rest=[c for vals in by_method.values() for c in vals if cand_key(c) not in used]
    rest.sort(key=lambda c:c.visibility_score, reverse=True)
    for c in rest:
        if len(chosen)>=limit: break
        if sum(1 for x in chosen if x.comp.method==c.comp.method)>=4: continue
        chosen.append(c)
    return chosen[:limit]


def line_kind(line: str) -> str:
    return "line_a" if line.startswith("line_a_") else "line_b"


def line_title(kind: str) -> str:
    return "Line A: original vs original + upsampling" if kind == "line_a" else "Line B: downsampled vs downsampled + upsampling"


def anchor_group(anchor: Anchor) -> str:
    return f"{anchor.line_kind}_anchor_{anchor.rank}"


def select_anchors(kind: str, comps: Sequence[Comparison], frames: Sequence[str], args: argparse.Namespace) -> List[Anchor]:
    line_comps = [c for c in comps if line_kind(c.line) == kind]
    if not line_comps:
        return []
    base_root = line_comps[0].input_root
    pool: List[Anchor] = []
    for frame in frames:
        ip = point_file(base_root, frame)
        if not ip:
            continue
        try:
            base_full = load_points(ip)
            calib = read_calib(args.kitti_root / "calib" / f"{frame}.txt")
            objs = [o for o in parse_labels(args.kitti_root / "label_2" / f"{frame}.txt") if o.class_name == "Car"]
        except Exception:
            continue
        for obj in objs:
            crop_lo, crop_hi, _, _, _ = crop_bounds(obj, calib, args.margin)
            bc = crop(base_full, crop_lo, crop_hi)
            dist = abs(obj.z)
            if len(bc) < max(args.min_base_points, 100) or dist > args.max_distance_x:
                continue
            dims = np.maximum(crop_hi - crop_lo, 1e-6)
            spread = float(np.mean([rng(bc, i) / dims[i] for i in range(3)]))
            available = tuple(c.method for c in line_comps if c.upsampled_root is not None and point_file(c.upsampled_root, frame))
            if not available:
                continue
            preferred_bonus = 80 if frame in PREFERRED_FRAMES else 0
            method_bonus = 35 * len(set(available))
            score = min(len(bc), 700) + 220 * min(spread, 1.0) + method_bonus + preferred_bonus - 6 * dist
            note = f"base_crop={len(bc)}, spread={spread:.2f}, dist={dist:.1f}m, available={','.join(available)}"
            pool.append(Anchor(kind, 0, frame, obj, score, len(bc), dist, available, note))
    pool.sort(key=lambda a: a.score, reverse=True)
    chosen: List[Anchor] = []
    used_frames = set()
    distance_bins: Dict[str, int] = {}
    for a in pool:
        b = "near" if a.distance_x < 20 else "mid" if a.distance_x < 30 else "far"
        if a.frame_id in used_frames:
            continue
        if distance_bins.get(b, 0) >= 3 and len(chosen) < 4:
            continue
        chosen.append(Anchor(a.line_kind, len(chosen) + 1, a.frame_id, a.obj, a.score, a.base_crop_points, a.distance_x, a.methods_available, a.note))
        used_frames.add(a.frame_id)
        distance_bins[b] = distance_bins.get(b, 0) + 1
        if len(chosen) >= 5:
            break
    if len(chosen) < 5:
        for a in pool:
            key = (a.frame_id, a.obj.object_id)
            if any((x.frame_id, x.obj.object_id) == key for x in chosen):
                continue
            chosen.append(Anchor(a.line_kind, len(chosen) + 1, a.frame_id, a.obj, a.score, a.base_crop_points, a.distance_x, a.methods_available, a.note))
            if len(chosen) >= 5:
                break
    return chosen


def candidate_for_anchor(comp: Comparison, anchor: Anchor, args: argparse.Namespace) -> Tuple[Optional[Candidate], str]:
    ip, up = point_file(comp.input_root, anchor.frame_id), point_file(comp.upsampled_root, anchor.frame_id)
    if comp.upsampled_root is None:
        return None, comp.missing_reason or "upsampled root missing"
    if not ip or not up:
        return None, "missing point cloud for shared anchor frame"
    try:
        bf, uf = load_points(ip), load_points(up)
        calib = read_calib(args.kitti_root / "calib" / f"{anchor.frame_id}.txt")
    except Exception as e:
        return None, f"read error: {e}"
    crop_lo, crop_hi, bbox_lo, bbox_hi, corners = crop_bounds(anchor.obj, calib, args.margin)
    bc, uc = crop(bf, crop_lo, crop_hi), crop(uf, crop_lo, crop_hi)
    if len(bc) < args.min_base_points:
        return None, f"shared anchor rejected for this method: base crop too sparse ({len(bc)})"
    if len(uc) < 50:
        return None, f"shared anchor rejected for this method: upsampled crop too sparse ({len(uc)})"
    npnts, nn = new_points(bc, uc, DEFAULT_NEW_THRESHOLD)
    ratio = len(npnts) / max(len(uc), 1)
    base_s, change_s, inside, outside, artifact, group = scores(bc, npnts, nn, crop_lo, crop_hi, bbox_lo, bbox_hi, abs(anchor.obj.z))
    visibility = 2 * base_s + 2 * change_s + min(len(npnts), 1600) * 0.5 + 50 * min(ratio, 0.7) + 30 * inside - 5 * abs(anchor.obj.z)
    if len(npnts) < args.min_new_points or ratio < args.min_new_ratio:
        reason = f"shared anchor included for cross-method comparison; weak visible new-point change ({len(npnts)} new, ratio {ratio:.1%}); {anchor.note}"
    else:
        reason = f"shared anchor with visible change; {len(npnts)} new points ({ratio:.1%}); inside bbox {inside:.1%}; {anchor.note}"
    if comp.smoke_only:
        reason += "; SPU-PMD smoke-only output"
    return Candidate(comp, anchor.frame_id, anchor.obj, ip, up, len(bf), len(uf), crop_lo, crop_hi, bbox_lo, bbox_hi, corners, bc, uc, npnts, DEFAULT_NEW_THRESHOLD, abs(anchor.obj.z), ratio, inside, outside, artifact, base_s, change_s, group, visibility, reason), ""


def write_anchor_sample(c: Candidate, root: Path, anchor: Anchor, args: argparse.Namespace, methods_available: Sequence[str]) -> Dict[str, object]:
    folder = root / c.comp.method / c.comp.line / f"frame_{c.frame_id}_obj_{c.obj.object_id}_{c.obj.class_name}"
    folder.mkdir(parents=True, exist_ok=True)
    bf, uf = load_points(c.input_path), load_points(c.upsampled_path)
    full_base = sample(bf, args.full_frame_display_points, args.seed + anchor.rank)
    full_up = sample(uf, args.full_frame_display_points, args.seed + anchor.rank + 1)
    title = (
        f"{c.comp.method} | {line_title(anchor.line_kind)} | frame {c.frame_id} obj {c.obj.object_id} {c.obj.class_name} | "
        f"distance_x {c.distance_x:.1f}m | crop {len(c.base_crop)} -> {len(c.up_crop)} | "
        f"new {len(c.new_points)} | margin {args.margin:.1f}m | score {c.visibility_score:.1f} | {anchor_group(anchor)}"
    )
    common = {
        "title": title,
        "base_label": c.comp.input_label, "up_label": c.comp.upsampled_label,
        "base_color": c.comp.input_color, "up_color": c.comp.upsampled_color, "new_color": c.comp.new_color,
        "bbox": bbox_payload(c.bbox_corners), "reason": c.recommended_reason,
        "base_path": str(c.input_path), "up_path": str(c.upsampled_path),
    }
    full_payload = dict(common, default_mode="bbox", base=point_payload(full_base), up=point_payload(full_up), new=point_payload(c.new_points))
    obj_default = "new" if len(c.new_points) >= 20 else "overlay"
    obj_payload = dict(common, default_mode=obj_default, base=point_payload(sample(c.base_crop, args.object_display_points, args.seed+anchor.rank+2)), up=point_payload(sample(c.up_crop, args.object_display_points, args.seed+anchor.rank+3)), new=point_payload(sample(c.new_points, args.object_display_points, args.seed+anchor.rank+4)))
    full_html, obj_html = folder/"full_frame.html", folder/"object_crop.html"
    full_png, obj_png = folder/"full_frame.png", folder/"object_crop.png"
    write_interactive(full_html, "Full-frame selected object | " + title, full_payload, True)
    write_interactive(obj_html, "Object crop | " + title, obj_payload, False)
    draw_png(full_png, "Full frame | " + title, full_base, c.new_points, c.bbox_corners, True, c.comp.new_color)
    draw_png(obj_png, "Object crop | " + title, c.base_crop, c.new_points, c.bbox_corners, False, c.comp.new_color)
    meta = {
        "anchor_group": anchor_group(anchor), "anchor_line": anchor.line_kind, "anchor_rank": anchor.rank,
        "methods_available_for_same_anchor": list(methods_available), "anchor_note": anchor.note,
        "method": c.comp.method, "line": c.comp.line, "frame_id": c.frame_id, "object_id": c.obj.object_id,
        "class_name": c.obj.class_name, "distance_x": c.distance_x,
        "base_point_cloud_path": str(c.input_path), "upsampled_point_cloud_path": str(c.upsampled_path),
        "base_crop_points": len(c.base_crop), "upsampled_crop_points": len(c.up_crop),
        "new_points_count": len(c.new_points), "new_points_ratio": c.new_ratio,
        "new_point_threshold": c.new_threshold, "selected_reason": c.recommended_reason,
    }
    meta_path = folder/"metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (folder/"combined_dashboard.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>body{{font-family:Arial;margin:20px;background:#f8fafc}}iframe{{width:100%;height:760px;border:1px solid #b4c1ce;margin-bottom:20px}}</style></head><body><h1>{html.escape(title)}</h1><p>{html.escape(c.recommended_reason)}</p><h2>Full frame</h2><iframe src="full_frame.html"></iframe><h2>Object crop</h2><iframe src="object_crop.html"></iframe></body></html>""", encoding="utf-8")
    row = row_for(c, anchor.rank, full_html, obj_html, full_png, obj_png, meta_path, "yes", args.margin)
    row.update({
        "anchor_group": anchor_group(anchor), "anchor_line": line_title(anchor.line_kind), "anchor_rank": anchor.rank,
        "is_shared_anchor": "yes", "methods_available_for_same_anchor": ",".join(methods_available),
    })
    return row


def anchor_blank_row(comp: Comparison, anchor: Anchor, reason: str, methods_available: Sequence[str]) -> Dict[str, object]:
    row = blank_row(comp, reason, anchor.frame_id, anchor.obj)
    row.update({
        "distance_x": f"{anchor.distance_x:.3f}", "anchor_group": anchor_group(anchor), "anchor_line": line_title(anchor.line_kind),
        "anchor_rank": anchor.rank, "is_shared_anchor": "yes", "methods_available_for_same_anchor": ",".join(methods_available),
    })
    return row


def load_existing_rows(summary_path: Path) -> List[Dict[str, object]]:
    if not summary_path.exists():
        return []
    with summary_path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_existing_anchors(root: Path, args: argparse.Namespace) -> Dict[str, List[Anchor]]:
    anchors: Dict[str, Dict[int, Anchor]] = {"line_a": {}, "line_b": {}}
    for meta_path in sorted(root.glob("*/*/frame_*_obj_*_*/metadata.json")):
        if "PDANS" in meta_path.parts:
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        kind = str(data.get("anchor_line", ""))
        if kind not in anchors:
            continue
        rank = int(data.get("anchor_rank", 0) or 0)
        if rank <= 0 or rank in anchors[kind]:
            continue
        frame_id = str(data["frame_id"])
        object_id = int(data["object_id"])
        objs = parse_labels(args.kitti_root / "label_2" / f"{frame_id}.txt")
        obj = next((o for o in objs if o.object_id == object_id), None)
        if obj is None:
            continue
        methods = tuple(str(x) for x in data.get("methods_available_for_same_anchor", []) if str(x))
        anchors[kind][rank] = Anchor(
            kind,
            rank,
            frame_id,
            obj,
            0.0,
            int(float(data.get("base_crop_points", 0) or 0)),
            float(data.get("distance_x", abs(obj.z)) or abs(obj.z)),
            methods,
            str(data.get("anchor_note", "")),
        )
    return {kind: [vals[k] for k in sorted(vals)] for kind, vals in anchors.items()}


def pdans_audit_row(comp: Comparison, anchor: Anchor, cand: Optional[Candidate], reason: str) -> Dict[str, object]:
    baseline_path = point_file(comp.input_root, anchor.frame_id)
    pdans_path = point_file(comp.upsampled_root, anchor.frame_id) if comp.upsampled_root is not None else None
    row = {
        "method": comp.method,
        "line": comp.line,
        "frame_id": anchor.frame_id,
        "object_id": anchor.obj.object_id,
        "class_name": anchor.obj.class_name,
        "baseline_path": "" if baseline_path is None else str(baseline_path),
        "pdans_path": "" if pdans_path is None else str(pdans_path),
        "baseline_point_count": "",
        "pdans_point_count": "",
        "crop_baseline_point_count": "",
        "crop_pdans_point_count": "",
        "status": "generated" if cand else "skipped",
        "reason": reason,
    }
    if cand:
        row.update({
            "baseline_path": str(cand.input_path),
            "pdans_path": str(cand.upsampled_path),
            "baseline_point_count": cand.base_full_count,
            "pdans_point_count": cand.up_full_count,
            "crop_baseline_point_count": len(cand.base_crop),
            "crop_pdans_point_count": len(cand.up_crop),
            "reason": cand.recommended_reason,
        })
    elif baseline_path is None or pdans_path is None:
        row["status"] = "missing input"
    return row


def write_pdans_audit(root: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "method", "line", "frame_id", "object_id", "class_name", "baseline_path",
        "pdans_path", "baseline_point_count", "pdans_point_count",
        "crop_baseline_point_count", "crop_pdans_point_count", "status", "reason",
    ]
    csv_path = root / "pdans_addition_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})
    md = [
        "# PDANS Addition Audit",
        "",
        "| Method | Line | Frame | Object | Class | Baseline points | PDANS points | Crop baseline | Crop PDANS | Status | Reason |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        md.append(
            f"| {row.get('method')} | {row.get('line')} | {row.get('frame_id')} | {row.get('object_id')} | "
            f"{row.get('class_name')} | {row.get('baseline_point_count')} | {row.get('pdans_point_count')} | "
            f"{row.get('crop_baseline_point_count')} | {row.get('crop_pdans_point_count')} | "
            f"{row.get('status')} | {html.escape(str(row.get('reason', '')))} |"
        )
    (root / "pdans_addition_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def row_link(root: Path, row: Dict[str, object], label: str) -> str:
    p = row.get(label, "")
    if not p:
        return ""
    try:
        href = Path(str(p)).relative_to(root)
    except ValueError:
        href = Path(str(p))
    return html.escape(str(href))


def write_reports(root: Path, comps: Sequence[Comparison], frames: Sequence[str], anchors: Dict[str, List[Anchor]], rows: List[Dict[str, object]], rejected: List[Dict[str, object]]) -> None:
    counts: Dict[Tuple[str, str], int] = {}
    for r in rows:
        if r.get("full_frame_html_path"):
            counts[(str(r["method"]), str(r["line"]))] = counts.get((str(r["method"]), str(r["line"])), 0) + 1

    avail=["# V6 Input Availability Report\n", "| Method | Line | Input root | Upsampled root | Common checked frames | Generated | Note |", "|---|---|---|---|---:|---:|---|"]
    for comp in comps:
        common=sum(1 for f in frames if point_file(comp.input_root,f) and point_file(comp.upsampled_root,f))
        note = comp.missing_reason or ("smoke-only" if comp.smoke_only else "available")
        avail.append(f"| {comp.method} | {comp.line} | `{rel(comp.input_root)}` | `{rel(comp.upsampled_root)}` | {common} | {counts.get((comp.method, comp.line), 0)} | {note} |")
    (root/"input_availability_report.md").write_text("\n".join(avail)+"\n", encoding="utf-8")

    rej=["# V6 Rejected Samples Report\n", "V6 rejects only anchors that are missing, unreadable, or still too sparse for a meaningful display. Weak visible changes can still be included to preserve cross-method comparison on the same anchor.\n"]
    for r in rejected:
        rej.append(f"- {r.get('method')} {r.get('line')} {r.get('anchor_group')} frame {r.get('frame_id')} obj {r.get('object_id')}: {r.get('rejected_reason')}")
    (root/"rejected_samples_report.md").write_text("\n".join(rej)+"\n", encoding="utf-8")

    ear=["# EAR Debug Report\n", "EAR is included on shared anchors when the object crop is nonblank. Many EAR rows may have weak new-point counts because the output is close to the base cloud under the 0.10 m KDTree new-point threshold.\n"]
    for r in rows:
        if r.get("method") == "EAR" and r.get("full_frame_html_path"):
            ear.append(f"- {r.get('line')} {r.get('anchor_group')} frame {r.get('frame_id')} obj {r.get('object_id')}: crop {r.get('base_crop_points')} -> {r.get('upsampled_crop_points')}, new {r.get('new_points_count')}, note: {r.get('recommended_reason')}")
    (root/"ear_debug_report.md").write_text("\n".join(ear)+"\n", encoding="utf-8")

    sections = []
    for kind in ("line_a", "line_b"):
        sections.append(f"<h2>{html.escape(line_title(kind))}</h2>")
        for anchor in anchors.get(kind, []):
            sections.append(f"<h3>{anchor_group(anchor)}: frame {anchor.frame_id} obj {anchor.obj.object_id} {anchor.obj.class_name}, dist {anchor.distance_x:.1f}m</h3><p>{html.escape(anchor.note)}</p><table><tr><th>Method</th><th>Crop</th><th>New</th><th>Full frame</th><th>Object crop</th><th>Preview</th><th>Note</th></tr>")
            for comp in [c for c in comps if line_kind(c.line) == kind]:
                match = next((r for r in rows if r.get("method") == comp.method and r.get("line") == comp.line and r.get("anchor_group") == anchor_group(anchor) and r.get("full_frame_html_path")), None)
                if match:
                    sections.append(
                        f"<tr><td>{html.escape(comp.method)}</td><td>{match.get('base_crop_points')} -> {match.get('upsampled_crop_points')}</td><td>{match.get('new_points_count')}</td>"
                        f"<td><a href=\"{row_link(root, match, 'full_frame_html_path')}\">full_frame.html</a></td>"
                        f"<td><a href=\"{row_link(root, match, 'object_crop_html_path')}\">object_crop.html</a></td>"
                        f"<td><a href=\"{row_link(root, match, 'full_frame_png_path')}\">full PNG</a> | <a href=\"{row_link(root, match, 'object_crop_png_path')}\">crop PNG</a></td>"
                        f"<td>{html.escape(str(match.get('recommended_reason','')))}</td></tr>"
                    )
                else:
                    miss = next((r for r in rejected if r.get("method") == comp.method and r.get("line") == comp.line and r.get("anchor_group") == anchor_group(anchor)), None)
                    sections.append(f"<tr><td>{html.escape(comp.method)}</td><td colspan=\"5\">not generated</td><td>{html.escape(str(miss.get('rejected_reason') if miss else comp.missing_reason or 'missing'))}</td></tr>")
            sections.append("</table>")
    style = "body{font-family:Arial;line-height:1.45;margin:24px;background:#f8fafc;color:#172033}table{border-collapse:collapse;width:100%;margin-bottom:28px;background:white}td,th{border:1px solid #cbd5df;padding:7px;vertical-align:top}th{background:#e8eef4}a{color:#0645ad}"
    (root/"recommended_for_meeting.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>V6 Recommended For Meeting</title><style>{style}</style></head><body><h1>V6 Recommended For Meeting</h1><p>Organized by line, fixed anchor object, then method for direct horizontal comparison.</p>{''.join(sections)}<p><a href='index.html'>Back to index</a></p></body></html>", encoding="utf-8")

    cards = []
    for comp in comps:
        n = counts.get((comp.method, comp.line), 0)
        cards.append(f"<li><b>{html.escape(comp.method)}</b> {html.escape(comp.line)}: {n} HTML sample folders</li>")
    (root/"index.html").write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>V6 KITTI Visualization</title><style>{style}</style></head><body><h1>V6 KITTI Full-frame + Object Crop Visualization</h1><p><a href="recommended_for_meeting.html">recommended_for_meeting.html</a> | <a href="summary_all.csv">summary_all.csv</a> | <a href="input_availability_report.md">input_availability_report.md</a> | <a href="rejected_samples_report.md">rejected_samples_report.md</a> | <a href="ear_debug_report.md">ear_debug_report.md</a> | <a href="pdans_addition_audit.csv">pdans_addition_audit.csv</a> | <a href="pdans_addition_audit.md">pdans_addition_audit.md</a></p><ul>{''.join(cards)}</ul><p>Folders are grouped as <code>METHOD/line_x/frame_ID_obj_ID_Class/</code>.</p></body></html>""", encoding="utf-8")

    anchor_text = []
    for kind, vals in anchors.items():
        anchor_text.append(f"\n## {line_title(kind)}")
        for a in vals:
            anchor_text.append(f"- {anchor_group(a)}: frame {a.frame_id}, obj {a.obj.object_id}, {a.obj.class_name}, distance {a.distance_x:.1f}m, score {a.score:.1f}; {a.note}")
    (root/"README.md").write_text(f"""# V6 Interactive Object-Level KITTI Visualization\n\nV6 keeps the v4/v5 full-frame view plus selected-object crop view. It changes the organization to method/line folders and uses fixed anchor objects so EAR, PU-Net, PU-GCN, TULIP, PDANS, and any available smoke-only SPU-PMD output can be compared on the same vehicles.\n\nOpen locally:\n\n```bash\ncd {root}\npython -m http.server 8000\n```\n\nThen open `http://localhost:8000/recommended_for_meeting.html`.\n\n## Visual Design\n\n- Base/original/downsampled points: dark blue.\n- Line A new/up points: bright red.\n- Line B new/up points: bright green.\n- Selected GT 3D bbox: yellow wireframe.\n- Marker sizes are intentionally small: full frame base/up/new about 0.75/0.55/1.0; object crop base/up/new about 1.15/0.9/1.2.\n\n## Anchor Selection Rule\n\nFor each line, V6 selects five Car anchors from the baseline point cloud using tight GT bbox crop margin `{args.margin if 'args' in globals() else 0.5}` m. Candidate anchors require at least 100 base crop points and distance under 40 m. Ranking score is:\n\n`min(base_crop_points, 700) + 220 * crop_spread + 35 * available_method_count + preferred_frame_bonus - 6 * distance_x`\n\nWeak visible new-point changes are still generated when the shared anchor is nonblank, because the goal is direct method comparison on identical objects.\n{''.join(anchor_text)}\n\n## Missing/Partial Methods\n\n- PDANS: Line A uses `results/pdans_main_kitti_pipeline/pdans_original_up_bin`; Line B uses `results/pdans_main_kitti_pipeline/pdans_downsampled50_up_bin`.\n- SPU-PMD: Line A uses smoke-only converted output if anchor frames are available; Line B is missing converted output.\n- Methods with fewer than five samples usually lack files for selected anchors or fail the nonblank crop filter.\n""", encoding="utf-8")


def add_pdans_only(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    frames = read_frames(args.val_split, args.max_candidate_frames)
    comps = build_comps(args.kitti_root)
    pdans_comps = [c for c in comps if c.method == "PDANS"]
    anchors = load_existing_anchors(args.output_root, args)
    if not anchors.get("line_a") or not anchors.get("line_b"):
        raise RuntimeError("Could not load existing v6 shared anchors from metadata.json files")

    rows = [r for r in load_existing_rows(args.output_root / "summary_all.csv") if r.get("method") != "PDANS"]
    pdans_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []

    for comp in pdans_comps:
        (args.output_root / comp.method / comp.line).mkdir(parents=True, exist_ok=True)

    for kind, anchor_list in anchors.items():
        line_comps = [c for c in comps if line_kind(c.line) == kind]
        pdans_line_comps = [c for c in pdans_comps if line_kind(c.line) == kind]
        for anchor in anchor_list:
            methods_available = sorted({
                c.method
                for c in line_comps
                if c.upsampled_root is not None and point_file(c.upsampled_root, anchor.frame_id)
            })
            for comp in pdans_line_comps:
                cand, why = candidate_for_anchor(comp, anchor, args)
                if cand:
                    row = write_anchor_sample(cand, args.output_root, anchor, args, methods_available)
                    pdans_rows.append(row)
                    audit_rows.append(pdans_audit_row(comp, anchor, cand, "generated"))
                else:
                    row = anchor_blank_row(comp, anchor, why, methods_available)
                    pdans_rows.append(row)
                    audit_rows.append(pdans_audit_row(comp, anchor, None, why))
            print(f"PDANS {anchor_group(anchor)} {anchor.frame_id} obj {anchor.obj.object_id}: {methods_available}")

    rows.extend(pdans_rows)
    write_csv(args.output_root / "summary_all.csv", rows)
    rejected = [r for r in rows if not r.get("full_frame_html_path")]
    write_reports(args.output_root, comps, frames, anchors, rows, rejected)
    write_pdans_audit(args.output_root, audit_rows)
    print(f"Wrote PDANS addition: {args.output_root}")
    print(f"Generated PDANS sample folders: {sum(1 for r in pdans_rows if r.get('full_frame_html_path'))}")
    print(f"Skipped/missing PDANS anchors: {sum(1 for r in pdans_rows if not r.get('full_frame_html_path'))}")


def main() -> None:
    args = parse_args()
    if args.add_pdans_only:
        add_pdans_only(args)
        return
    clean(args.output_root)
    frames = read_frames(args.val_split, args.max_candidate_frames)
    comps = build_comps(args.kitti_root)
    for comp in comps:
        (args.output_root / comp.method / comp.line).mkdir(parents=True, exist_ok=True)

    anchors = {
        "line_a": select_anchors("line_a", comps, frames, args),
        "line_b": select_anchors("line_b", comps, frames, args),
    }
    rows: List[Dict[str, object]] = []
    rejected: List[Dict[str, object]] = []
    for kind, anchor_list in anchors.items():
        line_comps = [c for c in comps if line_kind(c.line) == kind]
        for anchor in anchor_list:
            methods_available = sorted({c.method for c in line_comps if c.upsampled_root is not None and point_file(c.upsampled_root, anchor.frame_id)})
            for comp in line_comps:
                cand, why = candidate_for_anchor(comp, anchor, args)
                if cand:
                    rows.append(write_anchor_sample(cand, args.output_root, anchor, args, methods_available))
                else:
                    rejected.append(anchor_blank_row(comp, anchor, why, methods_available))
            print(f"{anchor_group(anchor)} {anchor.frame_id} obj {anchor.obj.object_id}: methods available {methods_available}")

    write_csv(args.output_root / "summary_all.csv", rows + rejected)
    write_reports(args.output_root, comps, frames, anchors, rows, rejected)
    print(f"Wrote v6 results: {args.output_root}")
    print(f"Generated sample folders: {len(rows)}")


if __name__=="__main__":
    main()
