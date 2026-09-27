#!/usr/bin/env python3
"""Export interactive Plotly HTML object-crop point cloud comparisons for KITTI.

The HTML files use Plotly from a CDN, so the script does not require the
Python plotly package to be installed.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KITTI_TRAIN_ROOT = PROJECT_ROOT / "data" / "KITTI" / "object" / "training"
IMAGESETS_ROOT = PROJECT_ROOT / "data" / "KITTI" / "ImageSets"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "interactive_object_crop_visualization"
DEFAULT_PREFERRED_FRAMES = ["000001", "003219", "006833", "007458"]
SUMMARY_FIELDS = [
    "method",
    "line",
    "frame_id",
    "object_id",
    "class_name",
    "crop_center_x",
    "crop_center_y",
    "crop_center_z",
    "bbox_h",
    "bbox_w",
    "bbox_l",
    "bbox_x",
    "bbox_y",
    "bbox_z",
    "bbox_rotation_y",
    "margin",
    "num_points_input_before_crop",
    "num_points_upsampled_before_crop",
    "num_points_input_after_crop",
    "num_points_upsampled_after_crop",
    "input_pointcloud_path",
    "upsampled_pointcloud_path",
    "html_path",
    "notes",
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
    comparison_name: str
    input_root: Path
    upsampled_root: Optional[Path]
    input_label: str
    upsampled_label: str
    input_color: str
    upsampled_color: str


@dataclass
class Sample:
    frame_id: str
    obj: KittiObject
    crop_min: np.ndarray
    crop_max: np.ndarray
    center: np.ndarray
    input_after: int
    upsampled_after: int
    ratio_delta: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--kitti-root", type=Path, default=KITTI_TRAIN_ROOT)
    parser.add_argument("--val-split", type=Path, default=IMAGESETS_ROOT / "val.txt")
    parser.add_argument("--preferred-frames", nargs="*", default=DEFAULT_PREFERRED_FRAMES)
    parser.add_argument("--samples-per-line", type=int, default=8)
    parser.add_argument("--min-samples-per-line", type=int, default=5)
    parser.add_argument("--max-display-points", type=int, default=20000)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--min-crop-points", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--methods", nargs="*", default=None, help="Optional method filter, e.g. EAR TULIP")
    return parser.parse_args()


def rel(path: Optional[Path]) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def point_file(root: Optional[Path], frame_id: str) -> Optional[Path]:
    if root is None:
        return None
    for suffix in (".bin", ".npy", ".ply", ".xyz"):
        path = root / f"{frame_id}{suffix}"
        if path.exists():
            return path
    matches = sorted(root.glob(f"**/{frame_id}*"))
    for path in matches:
        if path.suffix.lower() in {".bin", ".npy", ".ply", ".xyz"}:
            return path
    return None


def load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4 != 0:
            raise ValueError(f"Invalid KITTI bin size: {path}")
        arr = raw.reshape(-1, 4)
    elif suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".ply":
        arr = load_ascii_or_binary_ply(path)
    elif suffix == ".xyz":
        arr = np.loadtxt(path, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported point cloud format: {path}")
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected Nx3 or Nx4 point cloud: {path}")
    arr = arr.astype(np.float32, copy=False)
    if arr.shape[1] == 3:
        arr = np.column_stack([arr, np.zeros(len(arr), dtype=np.float32)])
    return arr[:, :4]


def load_ascii_or_binary_ply(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"PLY header did not end: {path}")
            header_lines.append(line.decode("ascii", errors="replace").strip())
            if header_lines[-1] == "end_header":
                break
        fmt = next((line for line in header_lines if line.startswith("format ")), "")
        vertex_line = next((line for line in header_lines if line.startswith("element vertex ")), "")
        n_vertices = int(vertex_line.split()[-1]) if vertex_line else 0
        props = [line.split()[-1] for line in header_lines if line.startswith("property ")]
        if fmt.startswith("format ascii"):
            data = []
            for _ in range(n_vertices):
                vals = f.readline().decode("ascii", errors="replace").split()
                data.append([float(v) for v in vals[: len(props)]])
            arr = np.asarray(data, dtype=np.float32)
        elif fmt.startswith("format binary_little_endian"):
            dtype_fields = []
            ply_to_np = {
                "float": "<f4",
                "float32": "<f4",
                "double": "<f8",
                "uchar": "u1",
                "uint8": "u1",
                "int": "<i4",
                "int32": "<i4",
                "uint": "<u4",
                "uint32": "<u4",
            }
            for line in header_lines:
                if line.startswith("property "):
                    _, typ, name = line.split()[:3]
                    dtype_fields.append((name, ply_to_np.get(typ, "<f4")))
            structured = np.fromfile(f, dtype=np.dtype(dtype_fields), count=n_vertices)
            arr = np.column_stack([structured[name] for name, _ in dtype_fields]).astype(np.float32)
        else:
            raise ValueError(f"Unsupported PLY format in {path}: {fmt}")
    names = {name: i for i, name in enumerate(props)}
    xyz = [names.get(k) for k in ("x", "y", "z")]
    if any(i is None for i in xyz):
        raise ValueError(f"PLY lacks x/y/z properties: {path}")
    intensity_idx = names.get("intensity", names.get("reflectance"))
    cols = [arr[:, xyz[0]], arr[:, xyz[1]], arr[:, xyz[2]]]
    cols.append(arr[:, intensity_idx] if intensity_idx is not None else np.zeros(len(arr), dtype=np.float32))
    return np.column_stack(cols).astype(np.float32)


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
        arr = np.asarray([float(x) for x in val.split()], dtype=np.float64)
        data[key] = arr
    r0 = np.eye(4, dtype=np.float64)
    r0[:3, :3] = data["R0_rect"].reshape(3, 3)
    v2c = np.eye(4, dtype=np.float64)
    v2c[:3, :4] = data["Tr_velo_to_cam"].reshape(3, 4)
    data["R0_rect_4x4"] = r0
    data["V2C_4x4"] = v2c
    data["rect_to_velo_4x4"] = np.linalg.inv(r0 @ v2c)
    return data


def camera_box_corners(obj: KittiObject) -> np.ndarray:
    x_corners = np.array([obj.l / 2, obj.l / 2, -obj.l / 2, -obj.l / 2, obj.l / 2, obj.l / 2, -obj.l / 2, -obj.l / 2])
    y_corners = np.array([0, 0, 0, 0, -obj.h, -obj.h, -obj.h, -obj.h])
    z_corners = np.array([obj.w / 2, -obj.w / 2, -obj.w / 2, obj.w / 2, obj.w / 2, -obj.w / 2, -obj.w / 2, obj.w / 2])
    rot = np.array(
        [
            [math.cos(obj.ry), 0, math.sin(obj.ry)],
            [0, 1, 0],
            [-math.sin(obj.ry), 0, math.cos(obj.ry)],
        ],
        dtype=np.float64,
    )
    corners = np.vstack([x_corners, y_corners, z_corners]).T @ rot.T
    corners += np.array([obj.x, obj.y, obj.z], dtype=np.float64)
    return corners


def rect_to_velo(points_rect: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    ones = np.ones((len(points_rect), 1), dtype=np.float64)
    hom = np.hstack([points_rect, ones])
    return (hom @ calib["rect_to_velo_4x4"].T)[:, :3]


def crop_bounds(obj: KittiObject, calib: Dict[str, np.ndarray], margin: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    corners_velo = rect_to_velo(camera_box_corners(obj), calib)
    lo = corners_velo.min(axis=0) - margin
    hi = corners_velo.max(axis=0) + margin
    center = (lo + hi) / 2.0
    return lo.astype(np.float32), hi.astype(np.float32), center.astype(np.float32)


def crop_points(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    mask = np.all((points[:, :3] >= lo) & (points[:, :3] <= hi), axis=1)
    return points[mask]


def read_val_frames(path: Path) -> List[str]:
    frames = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [f"{int(f):06d}" if f.isdigit() else f for f in frames]


def build_comparisons(kitti_root: Path, method_filter: Optional[Sequence[str]]) -> List[Comparison]:
    def root(name: str) -> Path:
        return kitti_root / name

    original = root("velodyne_original_val")
    downsampled = root("velodyne_downsampled_50_val")
    spupmd_smoke = PROJECT_ROOT / "results" / "pdans_spupmd_lab_feasibility_audit" / "spupmd_smoke_test" / "converted_kitti_bin_zero_intensity"
    specs = [
        ("EAR", "line_a_original_vs_ear", original, root("velodyne_ear_val"), "original", "original + EAR", "#9aa0a6", "#f04b35"),
        ("EAR", "line_b_downsampled_vs_ear", downsampled, root("velodyne_downsampled_50_ear_val"), "downsampled", "downsampled + EAR", "#2f80ed", "#2fb344"),
        ("PU-Net", "line_a_original_vs_punet", original, root("velodyne_punet_x2_fullframe"), "original", "original + PU-Net", "#9aa0a6", "#f04b35"),
        ("PU-Net", "line_b_downsampled_vs_punet", downsampled, root("velodyne_downsampled_50_punet_x2_fullframe"), "downsampled", "downsampled + PU-Net", "#2f80ed", "#2fb344"),
        ("PU-GCN", "line_a_original_vs_pugcn", original, root("pugcn_cap_100k"), "original", "original + PU-GCN", "#9aa0a6", "#ff7f0e"),
        ("PU-GCN", "line_b_downsampled_vs_pugcn", downsampled, root("pugcn_cap_100k_downsampled50_filled"), "downsampled", "downsampled + PU-GCN", "#2f80ed", "#2fb344"),
        ("TULIP", "line_a_original_vs_tulip", original, root("tulip_original_up_bin"), "original", "original + TULIP", "#9aa0a6", "#ff7f0e"),
        ("TULIP", "line_b_downsampled_vs_tulip", downsampled, root("tulip_downsampled_up_bin"), "downsampled", "downsampled + TULIP", "#2f80ed", "#2fb344"),
        ("SPU-PMD", "line_a_original_vs_spupmd_smoke4", original, spupmd_smoke, "original", "original + SPU-PMD smoke", "#9aa0a6", "#f04b35"),
    ]
    filters = {m.lower() for m in method_filter} if method_filter else None
    comps = []
    for method, line, inp, up, inp_label, up_label, inp_color, up_color in specs:
        if filters and method.lower() not in filters and method.lower().replace("-", "") not in filters:
            continue
        comps.append(Comparison(method, line, line, inp, up, inp_label, up_label, inp_color, up_color))
    return comps


def sample_for_display(points: np.ndarray, max_points: int, seed: int) -> Tuple[np.ndarray, str]:
    if len(points) <= max_points:
        return points, ""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), size=max_points, replace=False)
    idx.sort()
    return points[idx], f"display sampled to {max_points} points"


def points_trace(points: np.ndarray, name: str, color: str) -> Dict[str, object]:
    return {
        "type": "scatter3d",
        "mode": "markers",
        "name": name,
        "x": np.round(points[:, 0], 4).tolist(),
        "y": np.round(points[:, 1], 4).tolist(),
        "z": np.round(points[:, 2], 4).tolist(),
        "customdata": np.round(points[:, 3], 4).tolist(),
        "marker": {"size": 2.2, "color": color, "opacity": 0.72},
        "hovertemplate": "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>intensity=%{customdata:.3f}<extra>%{fullData.name}</extra>",
    }


def write_plotly_html(
    path: Path,
    comp: Comparison,
    frame_id: str,
    obj: KittiObject,
    input_crop: np.ndarray,
    up_crop: np.ndarray,
    input_path: Path,
    up_path: Path,
    center: np.ndarray,
    crop_min: np.ndarray,
    crop_max: np.ndarray,
    max_display_points: int,
    seed: int,
    notes: str,
) -> str:
    input_display, note_a = sample_for_display(input_crop, max_display_points, seed)
    up_display, note_b = sample_for_display(up_crop, max_display_points, seed + 1)
    title = (
        f"{frame_id} object {obj.object_id} {obj.class_name} | "
        f"{comp.input_label} vs {comp.upsampled_label} | "
        f"crop points {len(input_crop)} -> {len(up_crop)}"
    )
    data = [
        points_trace(input_display, f"{comp.input_label} | n={len(input_crop)} | {rel(input_path)}", comp.input_color),
        points_trace(up_display, f"{comp.upsampled_label} | n={len(up_crop)} | {rel(up_path)}", comp.upsampled_color),
    ]
    layout = {
        "title": {"text": title, "x": 0.02},
        "scene": {
            "xaxis": {"title": "x"},
            "yaxis": {"title": "y"},
            "zaxis": {"title": "z"},
            "aspectmode": "data",
            "camera": {"eye": {"x": 1.6, "y": -1.8, "z": 1.15}, "center": {"x": 0, "y": 0, "z": 0}},
        },
        "legend": {"itemsizing": "constant"},
        "margin": {"l": 0, "r": 0, "b": 0, "t": 88},
    }
    details = {
        "method": comp.method,
        "line": comp.line,
        "frame_id": frame_id,
        "object_id": obj.object_id,
        "class_name": obj.class_name,
        "bbox_camera_h_w_l": [obj.h, obj.w, obj.l],
        "bbox_camera_location_xyz": [obj.x, obj.y, obj.z],
        "bbox_rotation_y": obj.ry,
        "crop_center_lidar_xyz": np.round(center, 4).tolist(),
        "crop_min_lidar_xyz": np.round(crop_min, 4).tolist(),
        "crop_max_lidar_xyz": np.round(crop_max, 4).tolist(),
        "input_pointcloud_path": str(input_path),
        "upsampled_pointcloud_path": str(up_path),
        "notes": "; ".join(x for x in [notes, note_a, note_b] if x),
    }
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #202124; }}
    #plot {{ width: 100vw; height: 82vh; }}
    .meta {{ padding: 12px 16px; font-size: 13px; line-height: 1.45; background: #f8f9fa; border-top: 1px solid #dadce0; }}
    .meta code {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <div id="plot"></div>
  <div class="meta">
    <strong>{html.escape(comp.method)} / {html.escape(comp.line)}</strong><br>
    Frame {html.escape(frame_id)}, object {obj.object_id} {html.escape(obj.class_name)}.
    Input crop count: {len(input_crop)}. Upsampled crop count: {len(up_crop)}.<br>
    Input: <code>{html.escape(str(input_path))}</code><br>
    Upsampled: <code>{html.escape(str(up_path))}</code>
  </div>
  <script>
    const data = {json.dumps(data, separators=(",", ":"))};
    const layout = {json.dumps(layout, separators=(",", ":"))};
    Plotly.newPlot("plot", data, layout, {{responsive: true, displaylogo: false}});
    window.visualizationMeta = {json.dumps(details, separators=(",", ":"))};
  </script>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return "; ".join(x for x in [notes, note_a, note_b] if x)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def choose_samples(
    comp: Comparison,
    frames: Sequence[str],
    kitti_root: Path,
    margin: float,
    target: int,
    min_crop_points: int,
) -> Tuple[List[Sample], List[Dict[str, str]]]:
    samples: List[Sample] = []
    missing_rows: List[Dict[str, str]] = []
    diagnostic_frames = set(DEFAULT_PREFERRED_FRAMES)
    label_root = kitti_root / "label_2"
    calib_root = kitti_root / "calib"
    for frame_id in frames:
        if len(samples) >= target:
            break
        input_path = point_file(comp.input_root, frame_id)
        up_path = point_file(comp.upsampled_root, frame_id)
        if input_path is None or up_path is None:
            if frame_id in diagnostic_frames:
                missing_rows.append(make_missing_row(comp, frame_id, input_path, up_path, "missing input or upsampled file"))
            continue
        label_path = label_root / f"{frame_id}.txt"
        calib_path = calib_root / f"{frame_id}.txt"
        if not label_path.exists() or not calib_path.exists():
            if frame_id in diagnostic_frames:
                missing_rows.append(make_missing_row(comp, frame_id, input_path, up_path, "missing GT label or calibration"))
            continue
        try:
            input_points = load_points(input_path)
            up_points = load_points(up_path)
            calib = read_calib(calib_path)
        except Exception as exc:
            if frame_id in diagnostic_frames:
                missing_rows.append(make_missing_row(comp, frame_id, input_path, up_path, f"read error: {exc}"))
            continue
        objects = [o for o in parse_label_file(label_path) if o.class_name == "Car"]
        best: Optional[Sample] = None
        for obj in objects:
            lo, hi, center = crop_bounds(obj, calib, margin)
            input_after = len(crop_points(input_points, lo, hi))
            up_after = len(crop_points(up_points, lo, hi))
            if input_after < min_crop_points or up_after < min_crop_points:
                continue
            delta = up_after - input_after
            candidate = Sample(frame_id, obj, lo, hi, center, input_after, up_after, delta)
            if best is None or candidate.ratio_delta > best.ratio_delta:
                best = candidate
        if best is None:
            if frame_id in diagnostic_frames:
                missing_rows.append(make_missing_row(comp, frame_id, input_path, up_path, "no Car crop with enough points"))
            continue
        samples.append(best)
    samples.sort(key=lambda s: (0 if s.frame_id in DEFAULT_PREFERRED_FRAMES else 1, -s.ratio_delta, s.frame_id))
    return samples[:target], missing_rows


def make_missing_row(comp: Comparison, frame_id: str, input_path: Optional[Path], up_path: Optional[Path], note: str) -> Dict[str, str]:
    row = {field: "" for field in SUMMARY_FIELDS}
    row.update(
        {
            "method": comp.method,
            "line": comp.line,
            "frame_id": frame_id,
            "input_pointcloud_path": str(input_path or ""),
            "upsampled_pointcloud_path": str(up_path or ""),
            "notes": note,
        }
    )
    return row


def make_row(
    comp: Comparison,
    sample: Sample,
    input_points: np.ndarray,
    up_points: np.ndarray,
    input_crop: np.ndarray,
    up_crop: np.ndarray,
    input_path: Path,
    up_path: Path,
    html_path: Path,
    margin: float,
    notes: str,
) -> Dict[str, object]:
    obj = sample.obj
    return {
        "method": comp.method,
        "line": comp.line,
        "frame_id": sample.frame_id,
        "object_id": obj.object_id,
        "class_name": obj.class_name,
        "crop_center_x": f"{sample.center[0]:.4f}",
        "crop_center_y": f"{sample.center[1]:.4f}",
        "crop_center_z": f"{sample.center[2]:.4f}",
        "bbox_h": obj.h,
        "bbox_w": obj.w,
        "bbox_l": obj.l,
        "bbox_x": obj.x,
        "bbox_y": obj.y,
        "bbox_z": obj.z,
        "bbox_rotation_y": obj.ry,
        "margin": margin,
        "num_points_input_before_crop": len(input_points),
        "num_points_upsampled_before_crop": len(up_points),
        "num_points_input_after_crop": len(input_crop),
        "num_points_upsampled_after_crop": len(up_crop),
        "input_pointcloud_path": str(input_path),
        "upsampled_pointcloud_path": str(up_path),
        "html_path": str(html_path),
        "notes": notes,
    }


def ordered_candidate_frames(preferred: Sequence[str], val_frames: Sequence[str]) -> List[str]:
    preferred_norm = [f"{int(f):06d}" if f.isdigit() else f for f in preferred]
    seen = set()
    ordered = []
    for frame in list(preferred_norm) + list(val_frames):
        if frame not in seen:
            ordered.append(frame)
            seen.add(frame)
    return ordered


def availability_for_comparison(comp: Comparison, frames: Sequence[str]) -> Dict[str, object]:
    input_files = []
    up_files = []
    common = []
    for frame in frames:
        inp = point_file(comp.input_root, frame)
        up = point_file(comp.upsampled_root, frame)
        if inp:
            input_files.append(frame)
        if up:
            up_files.append(frame)
        if inp and up:
            common.append(frame)
    return {
        "method": comp.method,
        "line": comp.line,
        "input_root": comp.input_root,
        "upsampled_root": comp.upsampled_root,
        "input_available": len(input_files),
        "upsampled_available": len(up_files),
        "common_available": len(common),
        "preferred_common": [f for f in DEFAULT_PREFERRED_FRAMES if f in common],
    }


def write_availability_report(output_root: Path, comps: Sequence[Comparison], frames: Sequence[str]) -> Path:
    report_path = output_root / "input_availability_report.md"
    lines = [
        "# Input Availability Report",
        "",
        f"Project root: `{PROJECT_ROOT}`",
        f"KITTI training root: `{KITTI_TRAIN_ROOT}`",
        f"Val frame count scanned: {len(frames)}",
        "",
        "## Canonical Comparisons",
        "",
        "| Method | Line | Input root | Upsampled root | Input frames | Upsampled frames | Common frames | Preferred common |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for comp in comps:
        info = availability_for_comparison(comp, frames)
        lines.append(
            "| {method} | {line} | `{inp}` | `{up}` | {inp_n} | {up_n} | {common_n} | {pref} |".format(
                method=comp.method,
                line=comp.line,
                inp=rel(info["input_root"]),
                up=rel(info["upsampled_root"]),
                inp_n=info["input_available"],
                up_n=info["upsampled_available"],
                common_n=info["common_available"],
                pref=", ".join(info["preferred_common"]) or "-",
            )
        )
    lines.extend(
        [
            "",
            "## Methods Not Found As Full Point Cloud Roots",
            "",
            "- PDANS: no KITTI full-frame point cloud output directory was found during canonical data-root scan.",
            "- SPU-PMD: only a 4-frame smoke converted KITTI bin output was found; it is exported as `line_a_original_vs_spupmd_smoke4` when files are readable.",
            "- PU-EdgeFormer: external code exists, but no KITTI full-frame output directory was found in `data/KITTI/object/training`.",
            "",
            "The exporter skips missing method/line files without interrupting. `summary.csv` records generated visualizations plus diagnostic skips for the preferred frames.",
        ]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def write_line_index(path: Path, title: str, rows: Sequence[Dict[str, object]]) -> None:
    links = []
    for row in rows:
        html_path = Path(str(row.get("html_path", "")))
        if html_path.suffix != ".html" or not html_path.exists():
            continue
        links.append(
            f"<li><a href=\"{html.escape(html_path.name)}\">{html.escape(html_path.name)}</a> "
            f"frame {html.escape(str(row['frame_id']))}, obj {html.escape(str(row['object_id']))}, "
            f"{html.escape(str(row['num_points_input_after_crop']))} -> {html.escape(str(row['num_points_upsampled_after_crop']))} crop points</li>"
        )
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:Arial,sans-serif;line-height:1.5;margin:24px;max-width:1100px}} code{{background:#f1f3f4;padding:1px 4px}}</style></head>
<body><h1>{html.escape(title)}</h1><p>Open any file below to rotate, zoom, and pan the local object crop.</p><ul>
{chr(10).join(links)}
</ul><p><a href="summary.csv">summary.csv</a> | <a href="README.md">README.md</a></p></body></html>
"""
    path.write_text(body, encoding="utf-8")


def write_line_readme(path: Path, comp: Comparison, rows: Sequence[Dict[str, object]], margin: float, max_display_points: int) -> None:
    html_count = sum(1 for row in rows if str(row.get("html_path", "")).endswith(".html"))
    text = f"""# {comp.method} {comp.line}

Generated interactive Plotly object-crop comparisons for KITTI validation frames.

- Comparison: `{comp.input_label}` vs `{comp.upsampled_label}`
- Input root: `{comp.input_root}`
- Upsampled root: `{comp.upsampled_root}`
- Crop logic: KITTI GT `Car` 3D boxes are read from `label_2`, transformed from camera/rect coordinates into LiDAR coordinates using `calib`, then cropped with an axis-aligned box plus {margin:.2f} m margin.
- Colors: `{comp.input_label}` uses `{comp.input_color}`; `{comp.upsampled_label}` uses `{comp.upsampled_color}`.
- Display cap: each trace is randomly sampled to at most {max_display_points} points for HTML size; `summary.csv` keeps the original crop counts.
- HTML files generated: {html_count}

Use `index.html` in this folder to open individual visualizations.
"""
    path.write_text(text, encoding="utf-8")


def write_global_index(output_root: Path, all_rows: Sequence[Dict[str, object]], report_path: Path) -> None:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for row in all_rows:
        if str(row.get("html_path", "")).endswith(".html"):
            grouped.setdefault((str(row["method"]), str(row["line"])), []).append(row)
    sections = []
    for (method, line), rows in sorted(grouped.items()):
        first = Path(str(rows[0]["html_path"]))
        line_dir = first.parent
        sections.append(f"<h2>{html.escape(method)} / {html.escape(line)}</h2>")
        sections.append(f"<p><a href=\"{html.escape(str(line_dir.relative_to(output_root) / 'index.html'))}\">folder index</a> ({len(rows)} HTML files)</p>")
    body = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>KITTI Interactive Object Crop Visualization</title>
<style>body{{font-family:Arial,sans-serif;line-height:1.5;margin:24px;max-width:1100px}} code{{background:#f1f3f4;padding:1px 4px}}</style></head>
<body><h1>KITTI Interactive Object Crop Visualization</h1>
<p>Serve this directory with <code>python -m http.server 8000</code>, then open this index in the browser.</p>
<p><a href="{html.escape(report_path.name)}">input_availability_report.md</a> | <a href="summary_all.csv">summary_all.csv</a> | <a href="README.md">README.md</a></p>
{chr(10).join(sections)}
</body></html>
"""
    (output_root / "index.html").write_text(body, encoding="utf-8")


def write_global_readme(output_root: Path, all_rows: Sequence[Dict[str, object]], skipped: Sequence[Dict[str, object]], report_path: Path) -> None:
    counts: Dict[Tuple[str, str], int] = {}
    for row in all_rows:
        if str(row.get("html_path", "")).endswith(".html"):
            counts[(str(row["method"]), str(row["line"]))] = counts.get((str(row["method"]), str(row["line"])), 0) + 1
    lines = [
        "# Interactive Object Crop Visualization",
        "",
        "Open locally:",
        "",
        "```bash",
        f"cd {output_root}",
        "python -m http.server 8000",
        "```",
        "",
        "Then browse to `http://localhost:8000/index.html`.",
        "",
        f"Availability report: `{report_path}`",
        "",
        "## Generated HTML Counts",
        "",
    ]
    for (method, line), count in sorted(counts.items()):
        lines.append(f"- {method} / {line}: {count}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Crops use GT Car boxes when labels and calibration are available.",
            "- Axis-aligned LiDAR crop bounds are expanded around each GT 3D box.",
            "- Plotly is loaded from CDN in each HTML file.",
            f"- Skipped rows recorded: {len(skipped)}",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    val_frames = read_val_frames(args.val_split)
    frames = ordered_candidate_frames(args.preferred_frames, val_frames)
    comps = build_comparisons(args.kitti_root, args.methods)
    report_path = write_availability_report(output_root, comps, frames)

    all_rows: List[Dict[str, object]] = []
    skipped_rows: List[Dict[str, object]] = []
    for comp in comps:
        line_dir = output_root / comp.method / comp.line
        selected, missing = choose_samples(comp, frames, args.kitti_root, args.margin, args.samples_per_line, args.min_crop_points)
        rows: List[Dict[str, object]] = list(missing)
        skipped_rows.extend(missing)
        for sample in selected:
            input_path = point_file(comp.input_root, sample.frame_id)
            up_path = point_file(comp.upsampled_root, sample.frame_id)
            if input_path is None or up_path is None:
                continue
            input_points = load_points(input_path)
            up_points = load_points(up_path)
            input_crop = crop_points(input_points, sample.crop_min, sample.crop_max)
            up_crop = crop_points(up_points, sample.crop_min, sample.crop_max)
            file_name = f"frame_{sample.frame_id}_obj_{sample.obj.object_id}_{sample.obj.class_name}_{comp.comparison_name.replace('line_a_', '').replace('line_b_', '')}.html"
            html_path = line_dir / file_name
            notes = write_plotly_html(
                html_path,
                comp,
                sample.frame_id,
                sample.obj,
                input_crop,
                up_crop,
                input_path,
                up_path,
                sample.center,
                sample.crop_min,
                sample.crop_max,
                args.max_display_points,
                args.seed + int(sample.frame_id),
                "GT box crop in LiDAR AABB with margin",
            )
            rows.append(
                make_row(
                    comp,
                    sample,
                    input_points,
                    up_points,
                    input_crop,
                    up_crop,
                    input_path,
                    up_path,
                    html_path,
                    args.margin,
                    notes,
                )
            )
        write_csv(line_dir / "summary.csv", rows)
        write_line_index(line_dir / "index.html", f"{comp.method} {comp.line}", rows)
        write_line_readme(line_dir / "README.md", comp, rows, args.margin, args.max_display_points)
        all_rows.extend(rows)
        html_count = sum(1 for r in rows if str(r.get("html_path", "")).endswith(".html"))
        print(f"{comp.method} {comp.line}: wrote {html_count} HTML files to {line_dir}")

    write_csv(output_root / "summary_all.csv", all_rows)
    write_global_index(output_root, all_rows, report_path)
    write_global_readme(output_root, all_rows, skipped_rows, report_path)
    print(f"Wrote availability report: {report_path}")
    print(f"Wrote global index: {output_root / 'index.html'}")


if __name__ == "__main__":
    main()
