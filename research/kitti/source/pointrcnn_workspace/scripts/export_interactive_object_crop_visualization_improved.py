#!/usr/bin/env python3
"""Export improved KITTI object-level interactive Plotly crop visualizations.

This script only consumes existing point cloud outputs. It does not run
upsampling, detection, or KITTI data generation.
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KITTI_ROOT = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
IMAGESETS_ROOT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "interactive_object_crop_visualization_improved"
PREFERRED_FRAMES = ["000001", "003219", "006833", "007458"]
SUMMARY_FIELDS = [
    "method",
    "line",
    "frame_id",
    "object_id",
    "class_name",
    "distance_x",
    "input_crop_points",
    "upsampled_crop_points",
    "point_count_change",
    "score",
    "selected_rank",
    "margin",
    "html_path",
    "input_pointcloud_path",
    "upsampled_pointcloud_path",
    "skipped_reason",
    "is_recommended",
    "visualization_quality_note",
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
    input_crop_points: int
    upsampled_crop_points: int
    distance_x: float
    score: float
    quality_note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kitti-root", type=Path, default=KITTI_ROOT)
    parser.add_argument("--val-split", type=Path, default=IMAGESETS_ROOT / "val.txt")
    parser.add_argument("--max-candidate-frames", type=int, default=260)
    parser.add_argument("--samples-per-line", type=int, default=8)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--min-crop-points", type=int, default=50)
    parser.add_argument("--preferred-min-points", type=int, default=80)
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
    frames = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frames = [f"{int(f):06d}" if f.isdigit() else f for f in frames]
    ordered: List[str] = []
    seen = set()
    for frame in PREFERRED_FRAMES + frames:
        if frame not in seen:
            ordered.append(frame)
            seen.add(frame)
    return ordered[: max(limit, len(PREFERRED_FRAMES))]


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
        arr = load_ply(path)
    else:
        raise ValueError(f"Unsupported point cloud file: {path}")
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected Nx3/Nx4 point cloud: {path}")
    arr = arr.astype(np.float32, copy=False)
    if arr.shape[1] == 3:
        arr = np.column_stack([arr, np.zeros(len(arr), dtype=np.float32)])
    return arr[:, :4]


def load_ply(path: Path) -> np.ndarray:
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
        n_line = next((x for x in header if x.startswith("element vertex ")), "")
        n_vertices = int(n_line.split()[-1]) if n_line else 0
        props = [x.split()[-1] for x in header if x.startswith("property ")]
        if fmt.startswith("format ascii"):
            rows = []
            for _ in range(n_vertices):
                rows.append([float(v) for v in f.readline().decode("ascii", errors="replace").split()[: len(props)]])
            arr = np.asarray(rows, dtype=np.float32)
        else:
            raise ValueError(f"Only ASCII PLY is supported by this lightweight reader: {path}")
    col = {name: i for i, name in enumerate(props)}
    if not all(k in col for k in ("x", "y", "z")):
        raise ValueError(f"PLY lacks x/y/z: {path}")
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
    rot = np.array(
        [[math.cos(obj.ry), 0, math.sin(obj.ry)], [0, 1, 0], [-math.sin(obj.ry), 0, math.cos(obj.ry)]],
        dtype=np.float64,
    )
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
    spupmd_smoke = PROJECT_ROOT / "results" / "pdans_spupmd_lab_feasibility_audit" / "spupmd_smoke_test" / "converted_kitti_bin_zero_intensity"
    return [
        Comparison("EAR", "line_a_original_vs_ear", original, root("velodyne_ear_val"), "original", "original + EAR", "#0066ff", "#ff3b30"),
        Comparison("EAR", "line_b_downsampled_vs_ear", downsampled, root("velodyne_downsampled_50_ear_val"), "downsampled", "downsampled + EAR", "#0066ff", "#00a651"),
        Comparison("PU-Net", "line_a_original_vs_punet", original, root("velodyne_punet_x2_fullframe"), "original", "original + PU-Net", "#0066ff", "#ff3b30"),
        Comparison("PU-Net", "line_b_downsampled_vs_punet", downsampled, root("velodyne_downsampled_50_punet_x2_fullframe"), "downsampled", "downsampled + PU-Net", "#0066ff", "#00a651"),
        Comparison("PU-GCN", "line_a_original_vs_pugcn", original, root("pugcn_cap_100k"), "original", "original + PU-GCN", "#0066ff", "#ff7a00"),
        Comparison("PU-GCN", "line_b_downsampled_vs_pugcn", downsampled, root("pugcn_cap_100k_downsampled50_filled"), "downsampled", "downsampled + PU-GCN", "#0066ff", "#00a651"),
        Comparison("TULIP", "line_a_original_vs_tulip", original, root("tulip_original_up_bin"), "original", "original + TULIP", "#0066ff", "#ff7a00"),
        Comparison("TULIP", "line_b_downsampled_vs_tulip", downsampled, root("tulip_downsampled_up_bin"), "downsampled", "downsampled + TULIP", "#0066ff", "#00a651"),
        Comparison("SPU-PMD", "line_a_original_vs_spupmd_smoke4", original, spupmd_smoke, "original", "original + SPU-PMD smoke-only", "#0066ff", "#ff3b30", True),
        Comparison("SPU-PMD", "line_b_downsampled_vs_spupmd", downsampled, None, "downsampled", "downsampled + SPU-PMD", "#0066ff", "#00a651", True, "SPU-PMD line B converted full-frame output not found"),
        Comparison("PDANS", "line_a_original_vs_pdans", original, None, "original", "original + PDANS", "#0066ff", "#ff3b30", False, "PDANS converted full-frame output not found"),
        Comparison("PDANS", "line_b_downsampled_vs_pdans", downsampled, None, "downsampled", "downsampled + PDANS", "#0066ff", "#00a651", False, "PDANS converted full-frame output not found"),
    ]


def candidate_score(input_n: int, up_n: int, distance_x: float) -> float:
    return min(input_n, 500) + min(up_n, 1000) + 2.0 * abs(up_n - input_n) - 5.0 * distance_x


def quality_note(input_n: int, up_n: int, distance_x: float, score: float, smoke: bool) -> str:
    notes = []
    if input_n >= 100 and up_n >= 100:
        notes.append(">=100 crop points in both clouds")
    elif input_n >= 80 and up_n >= 80:
        notes.append(">=80 crop points in both clouds")
    if distance_x < 30:
        notes.append("near vehicle")
    elif distance_x < 40:
        notes.append("middle-distance vehicle")
    if abs(up_n - input_n) >= 100:
        notes.append("clear point-count change")
    if smoke:
        notes.append("smoke-only output")
    notes.append(f"score={score:.1f}")
    return "; ".join(notes)


def rank_candidates(comp: Comparison, frames: Sequence[str], args: argparse.Namespace) -> Tuple[List[Candidate], Dict[str, int], List[Dict[str, object]]]:
    stats = {"missing": 0, "read_error": 0, "no_label": 0, "low_quality": 0, "accepted": 0}
    skipped_rows: List[Dict[str, object]] = []
    accepted: List[Candidate] = []
    if comp.upsampled_root is None:
        skipped_rows.append(missing_summary_row(comp, comp.missing_reason or "upsampled root missing"))
        stats["missing"] += len(frames)
        return accepted, stats, skipped_rows

    label_root = args.kitti_root / "label_2"
    calib_root = args.kitti_root / "calib"
    for frame_id in frames:
        input_path = point_file(comp.input_root, frame_id)
        up_path = point_file(comp.upsampled_root, frame_id)
        if input_path is None or up_path is None:
            stats["missing"] += 1
            continue
        label_path = label_root / f"{frame_id}.txt"
        calib_path = calib_root / f"{frame_id}.txt"
        if not label_path.exists() or not calib_path.exists():
            stats["no_label"] += 1
            continue
        try:
            input_points = load_points(input_path)
            up_points = load_points(up_path)
            calib = read_calib(calib_path)
        except Exception:
            stats["read_error"] += 1
            continue
        for obj in parse_label_file(label_path):
            if obj.class_name != "Car":
                continue
            distance_x = float(abs(obj.z))
            lo, hi, corners = crop_info(obj, calib, args.margin)
            input_n = int(len(crop_points(input_points, lo, hi)))
            up_n = int(len(crop_points(up_points, lo, hi)))
            change = abs(up_n - input_n)
            if input_n < args.min_crop_points or up_n < args.min_crop_points:
                stats["low_quality"] += 1
                continue
            if min(input_n, up_n) < args.preferred_min_points and change < 80:
                stats["low_quality"] += 1
                continue
            if distance_x > args.max_distance_x:
                stats["low_quality"] += 1
                continue
            score = candidate_score(input_n, up_n, distance_x)
            if score < 120:
                stats["low_quality"] += 1
                continue
            accepted.append(
                Candidate(
                    comp=comp,
                    frame_id=frame_id,
                    obj=obj,
                    input_path=input_path,
                    upsampled_path=up_path,
                    crop_min=lo,
                    crop_max=hi,
                    bbox_corners_lidar=corners,
                    input_crop_points=input_n,
                    upsampled_crop_points=up_n,
                    distance_x=distance_x,
                    score=score,
                    quality_note=quality_note(input_n, up_n, distance_x, score, comp.smoke_only),
                )
            )
            stats["accepted"] += 1
    accepted.sort(key=lambda c: c.score, reverse=True)
    return accepted, stats, skipped_rows


def missing_summary_row(comp: Comparison, reason: str) -> Dict[str, object]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update({"method": comp.method, "line": comp.line, "skipped_reason": reason, "visualization_quality_note": reason})
    return row


def sample_for_display(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    idx.sort()
    return points[idx]


def point_trace(points: np.ndarray, name: str, color: str) -> Dict[str, object]:
    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": name,
        "x": np.round(points[:, 0], 4).tolist(),
        "y": np.round(points[:, 1], 4).tolist(),
        "z": np.round(points[:, 2], 4).tolist(),
        "customdata": np.round(points[:, 3], 4).tolist(),
        "marker": {"size": 4.0, "color": color, "opacity": 0.88},
        "hovertemplate": "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>intensity=%{customdata:.3f}<extra>%{fullData.name}</extra>",
    }


def bbox_trace(corners: np.ndarray) -> Dict[str, object]:
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []
    zs: List[Optional[float]] = []
    for a, b in edges:
        xs.extend([float(corners[a, 0]), float(corners[b, 0]), None])
        ys.extend([float(corners[a, 1]), float(corners[b, 1]), None])
        zs.extend([float(corners[a, 2]), float(corners[b, 2]), None])
    return {
        "type": "scatter3d",
        "mode": "lines",
        "name": "GT 3D bbox",
        "x": xs,
        "y": ys,
        "z": zs,
        "line": {"color": "#ffd400", "width": 8},
        "hoverinfo": "skip",
    }


def write_html(path: Path, cand: Candidate, rank: int, args: argparse.Namespace) -> Dict[str, object]:
    input_points = load_points(cand.input_path)
    up_points = load_points(cand.upsampled_path)
    input_crop = crop_points(input_points, cand.crop_min, cand.crop_max)
    up_crop = crop_points(up_points, cand.crop_min, cand.crop_max)
    a = sample_for_display(input_crop, args.max_display_points, args.seed + rank)
    b = sample_for_display(up_crop, args.max_display_points, args.seed + rank + 97)
    change = len(up_crop) - len(input_crop)
    title = (
        f"{cand.comp.method} | {cand.comp.line} | frame {cand.frame_id} obj {cand.obj.object_id} {cand.obj.class_name} | "
        f"distance_x={cand.distance_x:.1f}m | crop {len(input_crop)} -> {len(up_crop)} | "
        f"margin={args.margin:.1f}m | score={cand.score:.1f}"
    )
    data = [
        point_trace(a, f"{cand.comp.input_label} | n={len(input_crop)} | {rel(cand.input_path)}", cand.comp.input_color),
        point_trace(b, f"{cand.comp.upsampled_label} | n={len(up_crop)} | {rel(cand.upsampled_path)}", cand.comp.upsampled_color),
        bbox_trace(cand.bbox_corners_lidar),
    ]
    layout = {
        "title": {"text": title, "x": 0.02},
        "scene": {
            "xaxis": {"title": "x", "gridcolor": "#d8dde6", "backgroundcolor": "#f4f6f8"},
            "yaxis": {"title": "y", "gridcolor": "#d8dde6", "backgroundcolor": "#f4f6f8"},
            "zaxis": {"title": "z", "gridcolor": "#d8dde6", "backgroundcolor": "#f4f6f8"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.45, "y": -1.65, "z": 1.1}, "center": {"x": 0, "y": 0, "z": 0}},
        },
        "paper_bgcolor": "#eef2f5",
        "plot_bgcolor": "#eef2f5",
        "legend": {"itemsizing": "constant"},
        "margin": {"l": 0, "r": 0, "b": 0, "t": 96},
    }
    meta = {
        "method": cand.comp.method,
        "line": cand.comp.line,
        "frame_id": cand.frame_id,
        "object_id": cand.obj.object_id,
        "class_name": cand.obj.class_name,
        "distance_x": cand.distance_x,
        "input_crop_points": len(input_crop),
        "upsampled_crop_points": len(up_crop),
        "point_count_change": change,
        "score": cand.score,
        "margin": args.margin,
        "input_pointcloud_path": str(cand.input_path),
        "upsampled_pointcloud_path": str(cand.upsampled_path),
        "quality_note": cand.quality_note,
    }
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #1f2933; background: #eef2f5; }}
    #plot {{ width: 100vw; height: 84vh; }}
    .meta {{ padding: 12px 16px; font-size: 13px; line-height: 1.45; background: #f8fafc; border-top: 1px solid #c8d1dc; }}
    .meta code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <div class="meta">
    <strong>{html.escape(cand.comp.method)} / {html.escape(cand.comp.line)}</strong><br>
    Frame {html.escape(cand.frame_id)}, object {cand.obj.object_id} {html.escape(cand.obj.class_name)}, distance_x {cand.distance_x:.2f} m,
    crop points {len(input_crop)} -> {len(up_crop)}, score {cand.score:.1f}, margin {args.margin:.1f} m.<br>
    Input: <code>{html.escape(str(cand.input_path))}</code><br>
    Upsampled: <code>{html.escape(str(cand.upsampled_path))}</code><br>
    Note: {html.escape(cand.quality_note)}
  </div>
  <script>
    const data = {json.dumps(data, separators=(",", ":"))};
    const layout = {json.dumps(layout, separators=(",", ":"))};
    Plotly.newPlot("plot", data, layout, {{responsive: true, displaylogo: false}});
    window.visualizationMeta = {json.dumps(meta, separators=(",", ":"))};
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return {
        "method": cand.comp.method,
        "line": cand.comp.line,
        "frame_id": cand.frame_id,
        "object_id": cand.obj.object_id,
        "class_name": cand.obj.class_name,
        "distance_x": f"{cand.distance_x:.3f}",
        "input_crop_points": len(input_crop),
        "upsampled_crop_points": len(up_crop),
        "point_count_change": change,
        "score": f"{cand.score:.3f}",
        "selected_rank": rank,
        "margin": args.margin,
        "html_path": str(path),
        "input_pointcloud_path": str(cand.input_path),
        "upsampled_pointcloud_path": str(cand.upsampled_path),
        "skipped_reason": "",
        "is_recommended": "",
        "visualization_quality_note": cand.quality_note,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def write_line_index(line_dir: Path, comp: Comparison, rows: Sequence[Dict[str, object]]) -> None:
    items = []
    for row in rows:
        if not row.get("html_path"):
            continue
        path = Path(str(row["html_path"]))
        items.append(
            f"<li><a href=\"{html.escape(path.name)}\">{html.escape(path.name)}</a> "
            f"score {html.escape(str(row['score']))}, crop {html.escape(str(row['input_crop_points']))} -> {html.escape(str(row['upsampled_crop_points']))}, "
            f"distance_x {html.escape(str(row['distance_x']))} m</li>"
        )
    text = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(comp.method)} {html.escape(comp.line)}</title>
<style>body{{font-family:Arial,sans-serif;line-height:1.5;margin:24px;max-width:1100px;background:#f8fafc;color:#1f2933}}</style></head>
<body><h1>{html.escape(comp.method)} {html.escape(comp.line)}</h1>
<p>Improved tight object crops sorted by quality score.</p><ul>{''.join(items)}</ul>
<p><a href="summary.csv">summary.csv</a> | <a href="README.md">README.md</a></p></body></html>
"""
    (line_dir / "index.html").write_text(text, encoding="utf-8")


def write_line_readme(line_dir: Path, comp: Comparison, args: argparse.Namespace, stats: Dict[str, int], rows: Sequence[Dict[str, object]]) -> None:
    generated = sum(1 for r in rows if r.get("html_path"))
    text = f"""# {comp.method} {comp.line}

Improved object-level KITTI crop visualizations.

- Comparison: `{comp.input_label}` vs `{comp.upsampled_label}`
- Input root: `{comp.input_root}`
- Upsampled root: `{comp.upsampled_root}`
- Smoke-only: `{comp.smoke_only}`
- Crop: GT Car 3D bbox transformed to LiDAR coordinates, tight axis-aligned crop with margin {args.margin:.2f} m.
- Filtering: require both crop point counts >= {args.min_crop_points}, prefer >= {args.preferred_min_points}, require distance_x <= {args.max_distance_x:.1f} m, and reject weak low-score samples.
- Score: `min(input_crop_points, 500) + min(upsampled_crop_points, 1000) + 2 * abs(upsampled_crop_points - input_crop_points) - 5 * distance_x`.
- Colors: input `{comp.input_color}`, upsampled `{comp.upsampled_color}`, GT 3D bbox yellow wireframe.
- Generated HTML: {generated}
- Scan stats: {stats}
"""
    (line_dir / "README.md").write_text(text, encoding="utf-8")


def availability_report(output_root: Path, comps: Sequence[Comparison], frames: Sequence[str]) -> Path:
    path = output_root / "input_availability_report.md"
    lines = [
        "# Improved Input Availability Report",
        "",
        f"Project root: `{PROJECT_ROOT}`",
        f"Candidate frames scanned per full line: {len(frames)}",
        "",
        "| Method | Line | Input root | Upsampled root | Common candidate frames | Note |",
        "|---|---|---|---|---:|---|",
    ]
    for comp in comps:
        common = 0
        for frame in frames:
            if point_file(comp.input_root, frame) is not None and point_file(comp.upsampled_root, frame) is not None:
                common += 1
        note = comp.missing_reason or ("smoke-only" if comp.smoke_only else "available")
        lines.append(f"| {comp.method} | {comp.line} | `{rel(comp.input_root)}` | `{rel(comp.upsampled_root)}` | {common} | {note} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def choose_recommendations(rows: Sequence[Dict[str, object]], limit: int = 10) -> List[Dict[str, object]]:
    valid = [r for r in rows if r.get("html_path")]
    by_method: Dict[str, List[Dict[str, object]]] = {}
    for row in valid:
        by_method.setdefault(str(row["method"]), []).append(row)
    for vals in by_method.values():
        vals.sort(key=lambda r: float(r["score"]), reverse=True)
    recs: List[Dict[str, object]] = []
    for method in ["EAR", "PU-Net", "PU-GCN", "TULIP", "SPU-PMD"]:
        if by_method.get(method):
            recs.append(by_method[method].pop(0))
    remaining = []
    method_counts = {str(r["method"]): 1 for r in recs}
    for method, vals in by_method.items():
        remaining.extend(vals[: max(0, 3 - method_counts.get(method, 0))])
    remaining.sort(key=lambda r: float(r["score"]), reverse=True)
    recs.extend(remaining[: max(0, limit - len(recs))])
    return recs[:limit]


def clean_improved_output(output_root: Path) -> None:
    if not output_root.exists():
        return
    for pattern in ("**/frame_*.html", "**/index.html", "**/recommended_for_meeting.html", "**/summary*.csv", "**/README.md", "**/input_availability_report.md"):
        for path in output_root.glob(pattern):
            if path.is_file():
                path.unlink()


def write_global_pages(output_root: Path, rows: List[Dict[str, object]], recs: List[Dict[str, object]], report_path: Path, skipped_total: int, zip_note: str = "") -> None:
    for rec in recs:
        rec["is_recommended"] = "yes"
    sections = []
    counts: Dict[Tuple[str, str], int] = {}
    for row in rows:
        if row.get("html_path"):
            counts[(str(row["method"]), str(row["line"]))] = counts.get((str(row["method"]), str(row["line"])), 0) + 1
    for (method, line), count in sorted(counts.items()):
        line_dir = Path(next(str(r["html_path"]) for r in rows if r.get("html_path") and r["method"] == method and r["line"] == line)).parent
        sections.append(f"<li><a href=\"{html.escape(str(line_dir.relative_to(output_root) / 'index.html'))}\">{html.escape(method)} / {html.escape(line)}</a> ({count})</li>")
    index = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Improved KITTI Object Crop Visualization</title>
<style>body{{font-family:Arial,sans-serif;line-height:1.5;margin:24px;max-width:1100px;background:#f8fafc;color:#1f2933}}</style></head>
<body><h1>Improved KITTI Object Crop Visualization</h1>
<p><a href="recommended_for_meeting.html">recommended_for_meeting.html</a> | <a href="summary_all.csv">summary_all.csv</a> | <a href="{html.escape(report_path.name)}">input_availability_report.md</a></p>
<ul>{''.join(sections)}</ul></body></html>
"""
    (output_root / "index.html").write_text(index, encoding="utf-8")
    rec_items = []
    for idx, row in enumerate(recs, 1):
        path = Path(str(row["html_path"]))
        rec_items.append(
            f"<li><a href=\"{html.escape(str(path.relative_to(output_root)))}\">#{idx} {html.escape(row['method'])} {html.escape(row['line'])} frame {html.escape(row['frame_id'])} obj {html.escape(str(row['object_id']))}</a> "
            f"score {html.escape(str(row['score']))}, crop {html.escape(str(row['input_crop_points']))}->{html.escape(str(row['upsampled_crop_points']))}, distance_x {html.escape(str(row['distance_x']))}m</li>"
        )
    rec_html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Recommended For Meeting</title>
<style>body{{font-family:Arial,sans-serif;line-height:1.5;margin:24px;max-width:1100px;background:#f8fafc;color:#1f2933}}</style></head>
<body><h1>Recommended For Meeting</h1><ol>{''.join(rec_items)}</ol><p><a href="index.html">Back to index</a></p></body></html>
"""
    (output_root / "recommended_for_meeting.html").write_text(rec_html, encoding="utf-8")
    readme = f"""# Improved Interactive Object Crop Visualization

Open locally:

```bash
cd {output_root}
python -m http.server 8000
```

Then open `http://localhost:8000/index.html`.

Screening rule:

`score = min(input_crop_points, 500) + min(upsampled_crop_points, 1000) + 2 * abs(upsampled_crop_points - input_crop_points) - 5 * distance_x`

The exporter uses tight GT Car bbox crops with margin 0.5 m, rejects sparse/far/low-score objects, adds a yellow GT 3D bbox wireframe, and uses stronger point colors and marker sizes for presentation.

- Low-quality or missing candidates skipped: {skipped_total}
- PDANS: missing converted full-frame output; no HTML generated.
- SPU-PMD: smoke-only Line A output was found; Line B is missing.
{zip_note}
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    frames = read_val_frames(args.val_split, args.max_candidate_frames)
    comps = build_comparisons(args.kitti_root)
    clean_improved_output(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = availability_report(args.output_root, comps, frames)
    all_rows: List[Dict[str, object]] = []
    skipped_total = 0
    stats_by_line: Dict[Tuple[str, str], Dict[str, int]] = {}

    for comp in comps:
        line_dir = args.output_root / comp.method / comp.line
        candidates, stats, missing_rows = rank_candidates(comp, frames, args)
        stats_by_line[(comp.method, comp.line)] = stats
        skipped_total += stats.get("missing", 0) + stats.get("read_error", 0) + stats.get("no_label", 0) + stats.get("low_quality", 0)
        rows: List[Dict[str, object]] = list(missing_rows)
        selected = candidates[: args.samples_per_line]
        for rank, cand in enumerate(selected, 1):
            name = f"frame_{cand.frame_id}_obj_{cand.obj.object_id}_{cand.obj.class_name}_{slug(comp.line.replace('line_a_', '').replace('line_b_', ''))}_tight.html"
            rows.append(write_html(line_dir / name, cand, rank, args))
        if not selected and not rows:
            reason = comp.missing_reason or ("no smoke-only candidate passed improved quality filters" if comp.smoke_only else "no candidate passed improved quality filters")
            rows.append(missing_summary_row(comp, reason))
        write_csv(line_dir / "summary.csv", rows)
        write_line_index(line_dir, comp, rows)
        write_line_readme(line_dir, comp, args, stats, rows)
        all_rows.extend(rows)
        print(f"{comp.method} {comp.line}: generated {len(selected)} HTML, accepted {len(candidates)}, stats {stats}")

    recs = choose_recommendations(all_rows, 10)
    write_global_pages(args.output_root, all_rows, recs, report_path, skipped_total)
    write_csv(args.output_root / "summary_all.csv", all_rows)
    print(f"Wrote improved index: {args.output_root / 'index.html'}")
    print(f"Wrote recommended page: {args.output_root / 'recommended_for_meeting.html'}")
    print(f"Skipped low-quality/missing candidates: {skipped_total}")


if __name__ == "__main__":
    main()
