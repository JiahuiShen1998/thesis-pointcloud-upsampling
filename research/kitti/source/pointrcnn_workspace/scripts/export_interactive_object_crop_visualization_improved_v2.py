#!/usr/bin/env python3
"""Export v2 interactive KITTI object crop visualizations with new-point views.

This script only reads existing point cloud outputs. It does not run
upsampling, PointRCNN, or KITTI preprocessing.
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
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KITTI_ROOT = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
IMAGESETS_ROOT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "interactive_object_crop_visualization_improved_v2"
PREFERRED_FRAMES = ["000001", "003219", "006833", "007458"]
NEW_POINT_THRESHOLD_M = 0.05
SUMMARY_FIELDS = [
    "method",
    "line",
    "frame_id",
    "object_id",
    "class_name",
    "distance_x",
    "base_crop_points",
    "upsampled_crop_points",
    "new_points_count",
    "new_points_ratio",
    "new_points_x_range",
    "new_points_y_range",
    "new_points_z_range",
    "new_points_bbox_coverage",
    "new_points_nn_distance_to_base_mean",
    "structure_score_base",
    "structure_score_change",
    "visibility_score",
    "diversity_group",
    "selected_rank",
    "margin",
    "html_path",
    "input_pointcloud_path",
    "upsampled_pointcloud_path",
    "recommended_reason",
    "rejected_reason",
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
    crop_min: np.ndarray
    crop_max: np.ndarray
    bbox_corners_lidar: np.ndarray
    base_crop: np.ndarray
    up_crop: np.ndarray
    new_points: np.ndarray
    new_nn_dist: np.ndarray
    distance_x: float
    new_ratio: float
    x_range: float
    y_range: float
    z_range: float
    bbox_coverage: float
    nn_mean: float
    structure_score_base: float
    structure_score_change: float
    visibility_score: float
    diversity_group: str
    recommended_reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kitti-root", type=Path, default=KITTI_ROOT)
    parser.add_argument("--val-split", type=Path, default=IMAGESETS_ROOT / "val.txt")
    parser.add_argument("--max-candidate-frames", type=int, default=260)
    parser.add_argument("--samples-per-line", type=int, default=6)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--min-base-points", type=int, default=80)
    parser.add_argument("--min-upsampled-points", type=int, default=80)
    parser.add_argument("--min-new-points", type=int, default=80)
    parser.add_argument("--min-new-ratio", type=float, default=0.08)
    parser.add_argument("--max-new-ratio", type=float, default=0.97)
    parser.add_argument("--max-distance-x", type=float, default=40.0)
    parser.add_argument("--max-display-points", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260612)
    return parser.parse_args()


def rel(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def read_val_frames(path: Path, limit: int) -> List[str]:
    frames = [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    frames = [f"{int(x):06d}" if x.isdigit() else x for x in frames]
    out: List[str] = []
    seen = set()
    for frame in PREFERRED_FRAMES + frames:
        if frame not in seen:
            out.append(frame)
            seen.add(frame)
    return out[: max(limit, len(PREFERRED_FRAMES))]


def point_file(root: Optional[Path], frame_id: str) -> Optional[Path]:
    if root is None:
        return None
    for suffix in (".bin", ".npy", ".ply", ".xyz"):
        path = root / f"{frame_id}{suffix}"
        if path.exists():
            return path
    for path in sorted(root.glob(f"**/{frame_id}*")):
        if path.suffix.lower() in {".bin", ".npy", ".ply", ".xyz"}:
            return path
    return None


def load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"Invalid KITTI bin shape: {path}")
        arr = raw.reshape(-1, 4)
    elif suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".xyz":
        arr = np.loadtxt(path, dtype=np.float32)
    elif suffix == ".ply":
        arr = load_ascii_ply(path)
    else:
        raise ValueError(f"Unsupported point cloud format: {path}")
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected Nx3/Nx4 point cloud: {path}")
    arr = arr.astype(np.float32, copy=False)
    if arr.shape[1] == 3:
        arr = np.column_stack([arr, np.zeros(len(arr), dtype=np.float32)])
    return arr[:, :4]


def load_ascii_ply(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Bad PLY header: {path}")
            text = line.decode("ascii", errors="replace").strip()
            header.append(text)
            if text == "end_header":
                break
        fmt = next((x for x in header if x.startswith("format ")), "")
        if not fmt.startswith("format ascii"):
            raise ValueError(f"Only ASCII PLY is supported in v2 reader: {path}")
        n_line = next((x for x in header if x.startswith("element vertex ")), "")
        n_vertices = int(n_line.split()[-1]) if n_line else 0
        props = [x.split()[-1] for x in header if x.startswith("property ")]
        rows = [[float(v) for v in f.readline().decode("ascii", errors="replace").split()[: len(props)]] for _ in range(n_vertices)]
    arr = np.asarray(rows, dtype=np.float32)
    col = {name: i for i, name in enumerate(props)}
    intensity = col.get("intensity", col.get("reflectance"))
    vals = [arr[:, col["x"]], arr[:, col["y"]], arr[:, col["z"]]]
    vals.append(arr[:, intensity] if intensity is not None else np.zeros(len(arr), dtype=np.float32))
    return np.column_stack(vals).astype(np.float32)


def parse_label_file(path: Path) -> List[KittiObject]:
    objects: List[KittiObject] = []
    if not path.exists():
        return objects
    for object_id, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if len(parts) < 15:
            continue
        objects.append(
            KittiObject(
                object_id=object_id,
                class_name=parts[0],
                truncated=float(parts[1]),
                occluded=int(float(parts[2])),
                alpha=float(parts[3]),
                bbox2d=tuple(float(x) for x in parts[4:8]),  # type: ignore[arg-type]
                h=float(parts[8]),
                w=float(parts[9]),
                l=float(parts[10]),
                x=float(parts[11]),
                y=float(parts[12]),
                z=float(parts[13]),
                ry=float(parts[14]),
            )
        )
    return objects


def read_calib(path: Path) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        data[key] = np.asarray([float(x) for x in val.split()], dtype=np.float64)
    r0 = np.eye(4, dtype=np.float64)
    r0[:3, :3] = data["R0_rect"].reshape(3, 3)
    v2c = np.eye(4, dtype=np.float64)
    v2c[:3, :4] = data["Tr_velo_to_cam"].reshape(3, 4)
    data["rect_to_velo"] = np.linalg.inv(r0 @ v2c)
    return data


def camera_box_corners(obj: KittiObject) -> np.ndarray:
    x = np.array([obj.l / 2, obj.l / 2, -obj.l / 2, -obj.l / 2, obj.l / 2, obj.l / 2, -obj.l / 2, -obj.l / 2])
    y = np.array([0, 0, 0, 0, -obj.h, -obj.h, -obj.h, -obj.h])
    z = np.array([obj.w / 2, -obj.w / 2, -obj.w / 2, obj.w / 2, obj.w / 2, -obj.w / 2, -obj.w / 2, obj.w / 2])
    rot = np.array([[math.cos(obj.ry), 0, math.sin(obj.ry)], [0, 1, 0], [-math.sin(obj.ry), 0, math.cos(obj.ry)]], dtype=np.float64)
    corners = np.vstack([x, y, z]).T @ rot.T
    corners += np.array([obj.x, obj.y, obj.z], dtype=np.float64)
    return corners


def rect_to_velo(points_rect: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    hom = np.hstack([points_rect, np.ones((len(points_rect), 1), dtype=np.float64)])
    return (hom @ calib["rect_to_velo"].T)[:, :3]


def crop_info(obj: KittiObject, calib: Dict[str, np.ndarray], margin: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners = rect_to_velo(camera_box_corners(obj), calib).astype(np.float32)
    return corners.min(axis=0) - margin, corners.max(axis=0) + margin, corners


def crop_points(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    mask = np.all((points[:, :3] >= lo) & (points[:, :3] <= hi), axis=1)
    return points[mask]


def build_comparisons(kitti_root: Path) -> List[Comparison]:
    def root(name: str) -> Path:
        return kitti_root / name

    original = root("velodyne_original_val")
    downsampled = root("velodyne_downsampled_50_val")
    spupmd = PROJECT_ROOT / "results" / "pdans_spupmd_lab_feasibility_audit" / "spupmd_smoke_test" / "converted_kitti_bin_zero_intensity"
    return [
        Comparison("EAR", "line_a_original_vs_ear", original, root("velodyne_ear_val"), "original", "original + EAR", "#0066ff", "#ff7a00", "#ff1f1f"),
        Comparison("EAR", "line_b_downsampled_vs_ear", downsampled, root("velodyne_downsampled_50_ear_val"), "downsampled", "downsampled + EAR", "#0066ff", "#00a651", "#00ff66"),
        Comparison("PU-Net", "line_a_original_vs_punet", original, root("velodyne_punet_x2_fullframe"), "original", "original + PU-Net", "#0066ff", "#ff3b30", "#ff1f1f"),
        Comparison("PU-Net", "line_b_downsampled_vs_punet", downsampled, root("velodyne_downsampled_50_punet_x2_fullframe"), "downsampled", "downsampled + PU-Net", "#0066ff", "#00a651", "#00ff66"),
        Comparison("PU-GCN", "line_a_original_vs_pugcn", original, root("pugcn_cap_100k"), "original", "original + PU-GCN", "#0066ff", "#ff7a00", "#ff1f1f"),
        Comparison("PU-GCN", "line_b_downsampled_vs_pugcn", downsampled, root("pugcn_cap_100k_downsampled50_filled"), "downsampled", "downsampled + PU-GCN", "#0066ff", "#00a651", "#00ff66"),
        Comparison("TULIP", "line_a_original_vs_tulip", original, root("tulip_original_up_bin"), "original", "original + TULIP", "#0066ff", "#ff7a00", "#ff1f1f"),
        Comparison("TULIP", "line_b_downsampled_vs_tulip", downsampled, root("tulip_downsampled_up_bin"), "downsampled", "downsampled + TULIP", "#0066ff", "#00a651", "#00ff66"),
        Comparison("SPU-PMD", "line_a_original_vs_spupmd_smoke4", original, spupmd, "original", "original + SPU-PMD smoke-only", "#0066ff", "#ff3b30", "#ff1f1f", True),
        Comparison("SPU-PMD", "line_b_downsampled_vs_spupmd", downsampled, None, "downsampled", "downsampled + SPU-PMD", "#0066ff", "#00a651", "#00ff66", True, "SPU-PMD line B converted full-frame output not found"),
        Comparison("PDANS", "line_a_original_vs_pdans", original, None, "original", "original + PDANS", "#0066ff", "#ff3b30", "#ff1f1f", False, "PDANS converted full-frame output not found"),
        Comparison("PDANS", "line_b_downsampled_vs_pdans", downsampled, None, "downsampled", "downsampled + PDANS", "#0066ff", "#00a651", "#00ff66", False, "PDANS converted full-frame output not found"),
    ]


def new_point_mask(base_crop: np.ndarray, up_crop: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(base_crop) == 0 or len(up_crop) == 0:
        return np.ones(len(up_crop), dtype=bool), np.full(len(up_crop), np.inf, dtype=np.float32)
    tree = cKDTree(base_crop[:, :3])
    dist, _ = tree.query(up_crop[:, :3], k=1, workers=-1)
    return dist > threshold, dist.astype(np.float32)


def range_or_zero(points: np.ndarray, axis: int) -> float:
    if len(points) == 0:
        return 0.0
    return float(points[:, axis].max() - points[:, axis].min())


def coverage_score(new_points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    if len(new_points) == 0:
        return 0.0
    dims = np.maximum(hi - lo, 1e-6)
    ranges = np.array([range_or_zero(new_points, i) for i in range(3)], dtype=np.float32)
    return float(np.clip(np.mean(ranges / dims), 0.0, 1.0))


def diversity_group(obj: KittiObject, distance_x: float, new_points: np.ndarray, bbox_corners: np.ndarray) -> str:
    dist = "near" if distance_x < 20 else "mid" if distance_x < 30 else "far"
    if len(new_points) == 0:
        zone = "none"
    else:
        z_mid = float((bbox_corners[:, 2].min() + bbox_corners[:, 2].max()) / 2)
        y_mid = float((bbox_corners[:, 1].min() + bbox_corners[:, 1].max()) / 2)
        front_back = "front" if float(np.median(new_points[:, 0])) >= float(np.median(bbox_corners[:, 0])) else "rear"
        vertical = "upper" if float(np.median(new_points[:, 2])) >= z_mid else "lower"
        lateral = "side" if abs(float(np.median(new_points[:, 1])) - y_mid) > max(obj.w * 0.18, 0.25) else "center"
        zone = f"{front_back}_{vertical}_{lateral}"
    return f"{dist}_{zone}"


def structure_base_score(base_crop: np.ndarray, obj: KittiObject, lo: np.ndarray, hi: np.ndarray) -> float:
    if len(base_crop) == 0:
        return 0.0
    dims = np.maximum(hi - lo, 1e-6)
    ranges = np.array([range_or_zero(base_crop, i) for i in range(3)], dtype=np.float32)
    spread = float(np.clip(np.mean(ranges / dims), 0.0, 1.0))
    density = min(len(base_crop) / 300.0, 1.0)
    return 60.0 * density + 40.0 * spread


def structure_change_score(new_points: np.ndarray, nn_dist: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Tuple[float, float, float, float, float]:
    if len(new_points) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    xr, yr, zr = (range_or_zero(new_points, i) for i in range(3))
    coverage = coverage_score(new_points, lo, hi)
    nn_mean = float(np.mean(nn_dist[nn_dist > NEW_POINT_THRESHOLD_M])) if np.any(nn_dist > NEW_POINT_THRESHOLD_M) else 0.0
    score = min(len(new_points) / 500.0, 1.0) * 45.0 + coverage * 45.0 + min(nn_mean / 0.20, 1.0) * 10.0
    return score, xr, yr, zr, coverage


def visibility_score(base_score: float, change_score: float, new_count: int, new_ratio: float, distance_x: float, group_seen: int) -> float:
    ratio_score = 120.0 * min(new_ratio, 0.65)
    count_score = min(new_count, 1500) * 0.9
    distance_penalty = 7.0 * distance_x
    diversity_bonus = max(0.0, 80.0 - 25.0 * group_seen)
    return base_score * 2.0 + change_score * 3.0 + count_score + ratio_score + diversity_bonus - distance_penalty


def reject_reason(args: argparse.Namespace, base_n: int, up_n: int, new_n: int, new_ratio: float, distance_x: float, change_score: float) -> str:
    if base_n < args.min_base_points:
        return f"base crop too sparse ({base_n} < {args.min_base_points})"
    if up_n < args.min_upsampled_points:
        return f"upsampled crop too sparse ({up_n} < {args.min_upsampled_points})"
    if distance_x > args.max_distance_x:
        return f"vehicle too far ({distance_x:.1f}m > {args.max_distance_x:.1f}m)"
    if new_n < args.min_new_points:
        return f"too few new points ({new_n} < {args.min_new_points})"
    if new_ratio < args.min_new_ratio:
        return f"new point ratio too low ({new_ratio:.3f} < {args.min_new_ratio:.3f})"
    if new_ratio > args.max_new_ratio:
        return f"new point ratio too high/noisy ({new_ratio:.3f} > {args.max_new_ratio:.3f})"
    if change_score < 25:
        return f"new points spatially too concentrated (change_score={change_score:.1f})"
    return ""


def make_candidate(comp: Comparison, frame_id: str, obj: KittiObject, input_path: Path, up_path: Path, base_points: np.ndarray, up_points: np.ndarray, calib: Dict[str, np.ndarray], args: argparse.Namespace, group_seen: Dict[str, int]) -> Tuple[Optional[Candidate], str]:
    distance_x = float(abs(obj.z))
    lo, hi, corners = crop_info(obj, calib, args.margin)
    base_crop = crop_points(base_points, lo, hi)
    up_crop = crop_points(up_points, lo, hi)
    new_mask, nn_dist = new_point_mask(base_crop, up_crop, NEW_POINT_THRESHOLD_M)
    new_points = up_crop[new_mask]
    new_nn = nn_dist[new_mask]
    new_ratio = float(len(new_points) / max(len(up_crop), 1))
    base_score = structure_base_score(base_crop, obj, lo, hi)
    change_score, xr, yr, zr, cov = structure_change_score(new_points, nn_dist, lo, hi)
    reason = reject_reason(args, len(base_crop), len(up_crop), len(new_points), new_ratio, distance_x, change_score)
    if reason:
        return None, reason
    group = diversity_group(obj, distance_x, new_points, corners)
    score = visibility_score(base_score, change_score, len(new_points), new_ratio, distance_x, group_seen.get(group, 0))
    nn_mean = float(np.mean(new_nn)) if len(new_nn) else 0.0
    rec_reason = (
        f"visible new points: {len(new_points)} ({new_ratio:.1%}); "
        f"base structure score {base_score:.1f}; change spread score {change_score:.1f}; "
        f"group {group}; NN threshold {NEW_POINT_THRESHOLD_M:.2f}m"
    )
    return Candidate(comp, frame_id, obj, input_path, up_path, lo, hi, corners, base_crop, up_crop, new_points, new_nn, distance_x, new_ratio, xr, yr, zr, cov, nn_mean, base_score, change_score, score, group, rec_reason), ""


def rank_candidates(comp: Comparison, frames: Sequence[str], args: argparse.Namespace) -> Tuple[List[Candidate], Dict[str, int], List[Dict[str, object]]]:
    stats = {"missing": 0, "read_error": 0, "rejected": 0, "accepted": 0}
    rejected_examples: List[Dict[str, object]] = []
    candidates: List[Candidate] = []
    if comp.upsampled_root is None:
        stats["missing"] = len(frames)
        return [], stats, [missing_row(comp, comp.missing_reason or "upsampled root missing")]
    group_seen: Dict[str, int] = {}
    for frame_id in frames:
        input_path = point_file(comp.input_root, frame_id)
        up_path = point_file(comp.upsampled_root, frame_id)
        if input_path is None or up_path is None:
            stats["missing"] += 1
            continue
        try:
            base_points = load_points(input_path)
            up_points = load_points(up_path)
            calib = read_calib(args.kitti_root / "calib" / f"{frame_id}.txt")
            objects = [o for o in parse_label_file(args.kitti_root / "label_2" / f"{frame_id}.txt") if o.class_name == "Car"]
        except Exception:
            stats["read_error"] += 1
            continue
        for obj in objects:
            cand, reason = make_candidate(comp, frame_id, obj, input_path, up_path, base_points, up_points, calib, args, group_seen)
            if cand is None:
                stats["rejected"] += 1
                if len(rejected_examples) < 25:
                    rejected_examples.append(rejected_row(comp, frame_id, obj.object_id, reason))
                continue
            candidates.append(cand)
            group_seen[cand.diversity_group] = group_seen.get(cand.diversity_group, 0) + 1
            stats["accepted"] += 1
    candidates.sort(key=lambda c: c.visibility_score, reverse=True)
    return candidates, stats, rejected_examples


def missing_row(comp: Comparison, reason: str) -> Dict[str, object]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update({"method": comp.method, "line": comp.line, "rejected_reason": reason})
    return row


def rejected_row(comp: Comparison, frame_id: str, object_id: int, reason: str) -> Dict[str, object]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update({"method": comp.method, "line": comp.line, "frame_id": frame_id, "object_id": object_id, "rejected_reason": reason})
    return row


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    idx.sort()
    return points[idx]


def points_json(points: np.ndarray) -> Dict[str, List[float]]:
    return {
        "x": np.round(points[:, 0], 4).tolist(),
        "y": np.round(points[:, 1], 4).tolist(),
        "z": np.round(points[:, 2], 4).tolist(),
        "i": np.round(points[:, 3], 4).tolist(),
    }


def bbox_json(corners: np.ndarray) -> Dict[str, List[Optional[float]]]:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    for a, b in edges:
        xs.extend([float(corners[a, 0]), float(corners[b, 0]), None])
        ys.extend([float(corners[a, 1]), float(corners[b, 1]), None])
        zs.extend([float(corners[a, 2]), float(corners[b, 2]), None])
    return {"x": xs, "y": ys, "z": zs}


def write_html(path: Path, cand: Candidate, rank: int, args: argparse.Namespace) -> Dict[str, object]:
    base = sample_points(cand.base_crop, args.max_display_points, args.seed + rank)
    up = sample_points(cand.up_crop, args.max_display_points, args.seed + rank + 11)
    new = sample_points(cand.new_points, args.max_display_points, args.seed + rank + 23)
    title = (
        f"{cand.comp.method} | {cand.comp.line} | frame {cand.frame_id} obj {cand.obj.object_id} {cand.obj.class_name} | "
        f"base {len(cand.base_crop)} up {len(cand.up_crop)} new {len(cand.new_points)} ({cand.new_ratio:.1%}) | "
        f"dist {cand.distance_x:.1f}m | visibility {cand.visibility_score:.1f}"
    )
    payload = {
        "title": title,
        "base": points_json(base),
        "up": points_json(up),
        "new": points_json(new),
        "bbox": bbox_json(cand.bbox_corners_lidar),
        "labels": {"base": cand.comp.input_label, "up": cand.comp.upsampled_label},
        "colors": {"base": cand.comp.input_color, "up": cand.comp.upsampled_color, "new": cand.comp.new_color},
        "meta": {
            "method": cand.comp.method,
            "line": cand.comp.line,
            "frame_id": cand.frame_id,
            "object_id": cand.obj.object_id,
            "class_name": cand.obj.class_name,
            "distance_x": cand.distance_x,
            "base_crop_points": len(cand.base_crop),
            "upsampled_crop_points": len(cand.up_crop),
            "new_points_count": len(cand.new_points),
            "new_points_ratio": cand.new_ratio,
            "visibility_score": cand.visibility_score,
            "diversity_group": cand.diversity_group,
            "recommended_reason": cand.recommended_reason,
            "input_pointcloud_path": str(cand.input_path),
            "upsampled_pointcloud_path": str(cand.upsampled_path),
        },
    }
    js = json.dumps(payload, separators=(",", ":"))
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin:0; background:#e7edf3; color:#1f2933; font-family:Arial,sans-serif; }}
    .bar {{ display:flex; gap:8px; align-items:center; padding:10px 14px; background:#dfe7ef; border-bottom:1px solid #b9c5d1; flex-wrap:wrap; }}
    button {{ border:1px solid #8ca0b3; background:#f8fafc; padding:7px 10px; cursor:pointer; border-radius:4px; font-weight:600; }}
    button.active {{ background:#1f2933; color:white; }}
    #plot {{ width:100vw; height:78vh; }}
    .meta {{ padding:10px 14px; font-size:13px; line-height:1.45; background:#f8fafc; border-top:1px solid #b9c5d1; }}
    code {{ white-space:pre-wrap; }}
  </style>
</head>
<body>
  <div class="bar">
    <button id="btn-base" onclick="renderMode('base')">Base only</button>
    <button id="btn-up" onclick="renderMode('up')">Upsampled only</button>
    <button id="btn-overlay" onclick="renderMode('overlay')">Overlay</button>
    <button id="btn-new" onclick="renderMode('new')">New points only</button>
    <button id="btn-side" onclick="renderMode('side')">Side-by-side</button>
  </div>
  <div id="plot"></div>
  <div class="meta">
    <strong>{html.escape(title)}</strong><br>
    {html.escape(cand.recommended_reason)}<br>
    Input: <code>{html.escape(str(cand.input_path))}</code><br>
    Upsampled: <code>{html.escape(str(cand.upsampled_path))}</code>
  </div>
  <script>
    const P = {js};
    function scatter(points, name, color, size, opacity, scene='scene') {{
      return {{type:'scatter3d', mode:'markers', name, x:points.x, y:points.y, z:points.z, customdata:points.i,
        marker:{{size, color, opacity}}, scene,
        hovertemplate:'x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<br>intensity=%{{customdata:.3f}}<extra>%{{fullData.name}}</extra>'}};
    }}
    function bbox(scene='scene') {{
      return {{type:'scatter3d', mode:'lines', name:'GT 3D bbox', x:P.bbox.x, y:P.bbox.y, z:P.bbox.z,
        line:{{color:'#ffd400', width:8}}, hoverinfo:'skip', scene}};
    }}
    function sceneLayout(domain) {{
      return {{domain, aspectmode:'data',
        xaxis:{{title:'x', backgroundcolor:'#eef3f7', gridcolor:'#cfd8e3', zerolinecolor:'#b7c4d0'}},
        yaxis:{{title:'y', backgroundcolor:'#eef3f7', gridcolor:'#cfd8e3', zerolinecolor:'#b7c4d0'}},
        zaxis:{{title:'z', backgroundcolor:'#eef3f7', gridcolor:'#cfd8e3', zerolinecolor:'#b7c4d0'}},
        camera:{{eye:{{x:1.45,y:-1.65,z:1.1}}, center:{{x:0,y:0,z:0}}}}}};
    }}
    function layout(mode) {{
      const base = {{title:{{text:P.title + ' | ' + mode, x:0.02}}, paper_bgcolor:'#e7edf3', plot_bgcolor:'#e7edf3',
        margin:{{l:0,r:0,b:0,t:78}}, legend:{{itemsizing:'constant'}}}};
      if (mode === 'side') {{
        base.scene = sceneLayout({{x:[0,0.49], y:[0,1]}});
        base.scene2 = sceneLayout({{x:[0.51,1], y:[0,1]}});
        base.annotations = [
          {{text:'Base only', x:0.23, y:1.02, xref:'paper', yref:'paper', showarrow:false, font:{{size:16}}}},
          {{text:'New points + upsampled context', x:0.75, y:1.02, xref:'paper', yref:'paper', showarrow:false, font:{{size:16}}}}
        ];
      }} else {{
        base.scene = sceneLayout({{x:[0,1], y:[0,1]}});
      }}
      return base;
    }}
    function traces(mode) {{
      if (mode === 'base') return [scatter(P.base, P.labels.base, P.colors.base, 5, 0.95), bbox()];
      if (mode === 'up') return [scatter(P.up, P.labels.up, P.colors.up, 4, 0.90), bbox()];
      if (mode === 'overlay') return [scatter(P.base, P.labels.base, P.colors.base, 5, 0.95), scatter(P.up, P.labels.up, P.colors.up, 2.5, 0.35), bbox()];
      if (mode === 'new') return [scatter(P.base, P.labels.base + ' reference', '#9aa0a6', 3, 0.20), scatter(P.new, 'new points only', P.colors.new, 5, 0.92), bbox()];
      return [scatter(P.base, P.labels.base, P.colors.base, 5, 0.95, 'scene'), bbox('scene'),
              scatter(P.up, P.labels.up + ' context', P.colors.up, 2.5, 0.18, 'scene2'), scatter(P.new, 'new points', P.colors.new, 5, 0.92, 'scene2'), bbox('scene2')];
    }}
    function renderMode(mode) {{
      document.querySelectorAll('button').forEach(b => b.classList.remove('active'));
      const id = mode === 'up' ? 'btn-up' : mode === 'new' ? 'btn-new' : mode === 'side' ? 'btn-side' : mode === 'overlay' ? 'btn-overlay' : 'btn-base';
      document.getElementById(id).classList.add('active');
      Plotly.react('plot', traces(mode), layout(mode), {{responsive:true, displaylogo:false}});
    }}
    window.visualizationMeta = P.meta;
    renderMode('new');
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return summary_row(cand, rank, path, "", args.margin)


def summary_row(cand: Candidate, rank: int, path: Path, rejected: str, margin: float) -> Dict[str, object]:
    return {
        "method": cand.comp.method,
        "line": cand.comp.line,
        "frame_id": cand.frame_id,
        "object_id": cand.obj.object_id,
        "class_name": cand.obj.class_name,
        "distance_x": f"{cand.distance_x:.3f}",
        "base_crop_points": len(cand.base_crop),
        "upsampled_crop_points": len(cand.up_crop),
        "new_points_count": len(cand.new_points),
        "new_points_ratio": f"{cand.new_ratio:.6f}",
        "new_points_x_range": f"{cand.x_range:.4f}",
        "new_points_y_range": f"{cand.y_range:.4f}",
        "new_points_z_range": f"{cand.z_range:.4f}",
        "new_points_bbox_coverage": f"{cand.bbox_coverage:.6f}",
        "new_points_nn_distance_to_base_mean": f"{cand.nn_mean:.6f}",
        "structure_score_base": f"{cand.structure_score_base:.3f}",
        "structure_score_change": f"{cand.structure_score_change:.3f}",
        "visibility_score": f"{cand.visibility_score:.3f}",
        "diversity_group": cand.diversity_group,
        "selected_rank": rank,
        "margin": margin,
        "html_path": str(path),
        "input_pointcloud_path": str(cand.input_path),
        "upsampled_pointcloud_path": str(cand.upsampled_path),
        "recommended_reason": cand.recommended_reason,
        "rejected_reason": rejected,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def clean_output(root: Path) -> None:
    if not root.exists():
        return
    for pattern in ("**/*.html", "**/*.csv", "**/*.md", "**/*.json"):
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()


def select_diverse(candidates: Sequence[Candidate], limit: int) -> List[Candidate]:
    selected: List[Candidate] = []
    group_counts: Dict[str, int] = {}
    frame_counts: Dict[str, int] = {}
    for cand in candidates:
        if len(selected) >= limit:
            break
        if group_counts.get(cand.diversity_group, 0) >= 2:
            continue
        if frame_counts.get(cand.frame_id, 0) >= 2:
            continue
        selected.append(cand)
        group_counts[cand.diversity_group] = group_counts.get(cand.diversity_group, 0) + 1
        frame_counts[cand.frame_id] = frame_counts.get(cand.frame_id, 0) + 1
    return selected


def write_line_index(line_dir: Path, comp: Comparison, rows: Sequence[Dict[str, object]], stats: Dict[str, int]) -> None:
    items = []
    for row in rows:
        if row.get("html_path"):
            p = Path(str(row["html_path"]))
            items.append(f"<li><a href=\"{html.escape(p.name)}\">{html.escape(p.name)}</a> new {row['new_points_count']} ({row['new_points_ratio']}), visibility {row['visibility_score']}</li>")
    body = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(comp.method)} {html.escape(comp.line)} v2</title>
<style>body{{font-family:Arial,sans-serif;line-height:1.5;margin:24px;max-width:1100px;background:#f8fafc;color:#1f2933}}</style></head>
<body><h1>{html.escape(comp.method)} {html.escape(comp.line)} v2</h1>
<p>Each HTML opens in new-points mode first and includes buttons for base-only, upsampled-only, overlay, new-points-only, and side-by-side.</p>
<p>Stats: {html.escape(str(stats))}</p><ul>{''.join(items)}</ul><p><a href="summary.csv">summary.csv</a> | <a href="README.md">README.md</a></p></body></html>"""
    (line_dir / "index.html").write_text(body, encoding="utf-8")


def write_line_readme(line_dir: Path, comp: Comparison, args: argparse.Namespace, stats: Dict[str, int]) -> None:
    text = f"""# {comp.method} {comp.line} v2

This folder contains v2 object-level crop visualizations with explicit new-point extraction.

- New-point rule: each upsampled point is compared to the nearest base point with `scipy.spatial.cKDTree`; distance > {NEW_POINT_THRESHOLD_M:.2f} m means the point is treated as generated/new.
- Filtering: base >= {args.min_base_points}, upsampled >= {args.min_upsampled_points}, new points >= {args.min_new_points}, new ratio in [{args.min_new_ratio}, {args.max_new_ratio}], distance_x <= {args.max_distance_x} m, and spatial change score >= 25.
- Visibility score: `2*structure_score_base + 3*structure_score_change + 0.9*min(new_points_count,1500) + 120*min(new_points_ratio,0.65) + diversity_bonus - 7*distance_x`.
- Diversity: candidates are grouped by distance and new-point spatial zone; selected samples limit repeated groups and frames.
- HTML modes: base-only, upsampled-only, overlay, new-points-only, side-by-side.
- Overlay uses larger opaque base points and smaller transparent upsampled points, so base is not drowned out.
- Smoke-only: `{comp.smoke_only}`.
- Stats: {stats}
"""
    (line_dir / "README.md").write_text(text, encoding="utf-8")


def availability_report(root: Path, comps: Sequence[Comparison], frames: Sequence[str]) -> Path:
    path = root / "input_availability_report.md"
    lines = [
        "# V2 Input Availability Report",
        "",
        f"Project root: `{PROJECT_ROOT}`",
        f"Candidate frames scanned: {len(frames)}",
        "",
        "| Method | Line | Input root | Upsampled root | Common candidate frames | Note |",
        "|---|---|---|---|---:|---|",
    ]
    for comp in comps:
        common = sum(1 for f in frames if point_file(comp.input_root, f) is not None and point_file(comp.upsampled_root, f) is not None)
        note = comp.missing_reason or ("smoke-only" if comp.smoke_only else "available")
        lines.append(f"| {comp.method} | {comp.line} | `{rel(comp.input_root)}` | `{rel(comp.upsampled_root)}` | {common} | {note} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def choose_recommendations(rows: Sequence[Dict[str, object]], limit: int = 10) -> List[Dict[str, object]]:
    valid = [r for r in rows if r.get("html_path")]
    by_line: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in valid:
        by_line.setdefault((str(row["method"]), str(row["line"])), []).append(row)
    for vals in by_line.values():
        vals.sort(key=lambda r: float(r["visibility_score"]), reverse=True)
    recs: List[Dict[str, object]] = []
    for method in ["EAR", "PU-Net", "PU-GCN", "TULIP", "SPU-PMD"]:
        for key in sorted([k for k in by_line if k[0] == method]):
            if by_line[key]:
                recs.append(by_line[key].pop(0))
                if len(recs) >= limit:
                    return recs
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for vals in by_line.values():
        for row in vals:
            by_method.setdefault(str(row["method"]), []).append(row)
    for vals in by_method.values():
        vals.sort(key=lambda r: float(r["visibility_score"]), reverse=True)
    remaining = []
    counts = {str(r["method"]): 1 for r in recs}
    for method, vals in by_method.items():
        remaining.extend(vals[: max(0, 3 - counts.get(method, 0))])
    remaining.sort(key=lambda r: float(r["visibility_score"]), reverse=True)
    recs.extend(remaining[: max(0, limit - len(recs))])
    if len(recs) < limit:
        used = {str(r["html_path"]) for r in recs}
        fallback = [r for r in valid if str(r["html_path"]) not in used]
        fallback.sort(key=lambda r: float(r["visibility_score"]), reverse=True)
        recs.extend(fallback[: max(0, limit - len(recs))])
    return recs[:limit]


def write_global_pages(root: Path, rows: List[Dict[str, object]], recs: List[Dict[str, object]], report: Path, skipped: int) -> None:
    rec_paths = {str(r["html_path"]) for r in recs}
    for row in rows:
        if str(row.get("html_path", "")) in rec_paths:
            row["recommended_reason"] = "MEETING TOP: " + str(row.get("recommended_reason", ""))
    counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        if row.get("html_path"):
            counts[(str(row["method"]), str(row["line"]))] = counts.get((str(row["method"]), str(row["line"])), 0) + 1
    links = []
    for (method, line), count in sorted(counts.items()):
        line_dir = Path(next(str(r["html_path"]) for r in rows if r.get("html_path") and r["method"] == method and r["line"] == line)).parent
        links.append(f"<li><a href=\"{html.escape(str(line_dir.relative_to(root) / 'index.html'))}\">{html.escape(method)} / {html.escape(line)}</a> ({count})</li>")
    (root / "index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>V2 KITTI Object Crop Visualization</title>
<style>body{{font-family:Arial,sans-serif;line-height:1.5;margin:24px;max-width:1100px;background:#f8fafc;color:#1f2933}}</style></head>
<body><h1>V2 KITTI Object Crop Visualization</h1>
<p><a href="recommended_for_meeting.html">recommended_for_meeting.html</a> | <a href="summary_all.csv">summary_all.csv</a> | <a href="{html.escape(report.name)}">input_availability_report.md</a></p>
<ul>{''.join(links)}</ul></body></html>""", encoding="utf-8")
    rec_items = []
    for i, row in enumerate(recs, 1):
        p = Path(str(row["html_path"]))
        rec_items.append(f"<li><a href=\"{html.escape(str(p.relative_to(root)))}\">#{i} {html.escape(row['method'])} {html.escape(row['line'])} frame {html.escape(row['frame_id'])} obj {html.escape(str(row['object_id']))}</a> new {row['new_points_count']} ({row['new_points_ratio']}), visibility {row['visibility_score']}, group {html.escape(row['diversity_group'])}</li>")
    (root / "recommended_for_meeting.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>V2 Recommended For Meeting</title>
<style>body{{font-family:Arial,sans-serif;line-height:1.5;margin:24px;max-width:1100px;background:#f8fafc;color:#1f2933}}</style></head>
<body><h1>V2 Recommended For Meeting</h1><p>Open these first; each starts in new-points-only mode.</p><ol>{''.join(rec_items)}</ol><p><a href="index.html">Back</a></p></body></html>""", encoding="utf-8")
    (root / "README.md").write_text(f"""# V2 Interactive Object Crop Visualization

Open locally:

```bash
cd {root}
python -m http.server 8000
```

Then open `http://localhost:8000/recommended_for_meeting.html`.

V2 fixes the overlay problem by using a single HTML with five modes: base-only, upsampled-only, overlay, new-points-only, and side-by-side. The default mode is new-points-only, where base points are faint grey context and generated/new points are highlighted.

New points are computed by nearest-neighbor distance from upsampled crop points to base crop points. If nearest distance is greater than {NEW_POINT_THRESHOLD_M:.2f} m, the point is counted as new.

Visibility scoring combines base structure, new point count, new point ratio, spatial spread/coverage, nearest-neighbor distance, distance penalty, and diversity bonus. This favors samples where the added points form visible spatial structure instead of merely increasing total point count.

- Skipped/rejected candidates: {skipped}
- PDANS: converted full-frame output still missing.
- SPU-PMD: only smoke Line A exists; no candidate passed strict v2 quality filters.
""", encoding="utf-8")


def main() -> None:
    args = parse_args()
    frames = read_val_frames(args.val_split, args.max_candidate_frames)
    comps = build_comparisons(args.kitti_root)
    clean_output(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = availability_report(args.output_root, comps, frames)
    all_rows: List[Dict[str, object]] = []
    skipped_total = 0
    for comp in comps:
        line_dir = args.output_root / comp.method / comp.line
        candidates, stats, rejected = rank_candidates(comp, frames, args)
        skipped_total += stats.get("missing", 0) + stats.get("read_error", 0) + stats.get("rejected", 0)
        selected = select_diverse(candidates, args.samples_per_line)
        rows: List[Dict[str, object]] = []
        if not selected:
            rows.extend(rejected[:5] if rejected else [missing_row(comp, comp.missing_reason or "no v2 candidate passed visible-change filters")])
        for rank, cand in enumerate(selected, 1):
            name = f"frame_{cand.frame_id}_obj_{cand.obj.object_id}_{cand.obj.class_name}_{slug(comp.line)}_v2.html"
            rows.append(write_html(line_dir / name, cand, rank, args))
        write_csv(line_dir / "summary.csv", rows)
        write_line_index(line_dir, comp, rows, stats)
        write_line_readme(line_dir, comp, args, stats)
        all_rows.extend(rows)
        print(f"{comp.method} {comp.line}: generated {len(selected)} HTML, accepted {len(candidates)}, stats {stats}")
    recs = choose_recommendations(all_rows, 10)
    write_global_pages(args.output_root, all_rows, recs, report, skipped_total)
    write_csv(args.output_root / "summary_all.csv", all_rows)
    print(f"Wrote v2 index: {args.output_root / 'index.html'}")
    print(f"Wrote v2 recommended page: {args.output_root / 'recommended_for_meeting.html'}")
    print(f"Skipped/rejected candidates: {skipped_total}")


if __name__ == "__main__":
    main()
