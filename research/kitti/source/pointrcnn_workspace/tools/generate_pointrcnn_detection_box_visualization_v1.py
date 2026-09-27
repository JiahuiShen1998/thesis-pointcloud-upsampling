#!/usr/bin/env python3
"""Generate interactive PointRCNN detection-box visualizations.

The script is intentionally dependency-free. It writes Plotly HTML that loads
Plotly from the CDN, parses KITTI-style detection files, transforms KITTI camera
boxes to LiDAR coordinates with calibration files, and matches detections by a
same-class center-distance fallback.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import shutil
import struct
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KITTI = ROOT / "data/KITTI/object/training"
OUT = ROOT / "results/pointrcnn_detection_box_visualization_v1"
FRAMES = ["000093", "000242", "003219", "006833", "007458"]
METHODS = ["EAR", "PU-Net", "PU-GCN", "PDANS", "TULIP"]
MARGIN_M = 1.5
MATCH_CENTER_THRESHOLD_M = 2.0
FOV_NEAR_M = 2.0
FOV_FAR_M = 80.0
FOV_NOTE = (
    "The full point cloud is the complete Velodyne frame, while KITTI labels and "
    "PointRCNN detections are camera-FOV based. Therefore, detections are expected "
    "to appear mainly in the front camera-visible sector, not across the full "
    "360-degree LiDAR scene."
)
GT_PRED_CENTER_THRESHOLD_M = 2.0
SUSPICIOUS_PATH_TOKENS = (
    "smoke",
    "dryrun",
    "debug",
    "tmp",
    "temp",
    "test_original",
    "nms_workaround",
    "feasibility",
    "retry_low_num_points",
)
SUBSET_PATH_TOKENS = ("20frame", "50frame", "smoke5", "selected")
AP_LIKE_PATH_TOKENS = (
    "full_validation",
    "main_kitti_pipeline",
    "rpn4096_no_distance_propose_comparison",
    "clean_original_baseline",
    "kitti_cpp_eval",
    "ap_eval",
    "50frame_default_validation",
)


def p(*parts: str) -> Path:
    return ROOT.joinpath(*parts)


SPECS = {
    "EAR": {
        "line_a_original_vs_ear": {
            "base_cloud": KITTI / "velodyne_original_val",
            "up_cloud": KITTI / "velodyne_ear_val",
            "base_det": p("results/pointrcnn_clean_original_baseline_20260508_203809/evaluation/raw_output/eval/epoch_no_number/val/final_result/data"),
            "up_det": p("results/pointrcnn_50frame_default_validation_cap100k/inference/ear_default/eval/epoch_no_number/val_pugcn_cap100k_50/test_mode/final_result/data"),
        },
        "line_b_downsampled_vs_ear": {
            "base_cloud": KITTI / "velodyne_downsampled_50_val",
            "up_cloud": KITTI / "velodyne_downsampled_50_ear_val",
            "base_det": p("results/rpn4096_no_distance_propose_comparison/01_downsampled50/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
            "up_det": p("results/rpn4096_no_distance_propose_comparison/02_downsampled50_EAR/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
        },
    },
    "PU-Net": {
        "line_a_original_vs_punet": {
            "base_cloud": KITTI / "velodyne_original_val",
            "up_cloud": KITTI / "velodyne_punet_x2_fullframe",
            "base_det": p("results/pointrcnn_clean_original_baseline_20260508_203809/evaluation/raw_output/eval/epoch_no_number/val/final_result/data"),
            "up_det": p("results/pointrcnn_50frame_default_validation_cap100k/inference/punet_default/eval/epoch_no_number/val_pugcn_cap100k_50/test_mode/final_result/data"),
        },
        "line_b_downsampled_vs_punet": {
            "base_cloud": KITTI / "velodyne_downsampled_50_val",
            "up_cloud": KITTI / "velodyne_downsampled_50_punet_x2_fullframe",
            "base_det": p("results/rpn4096_no_distance_propose_comparison/01_downsampled50/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
            "up_det": p("results/rpn4096_no_distance_propose_comparison/03_downsampled50_PUNet/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
        },
    },
    "PU-GCN": {
        "line_a_original_vs_pugcn": {
            "base_cloud": KITTI / "velodyne_original_val",
            "up_cloud": KITTI / "pugcn_cap_100k",
            "base_det": p("results/pointrcnn_clean_original_baseline_20260508_203809/evaluation/raw_output/eval/epoch_no_number/val/final_result/data"),
            "up_det": p("results/pointrcnn_full_validation_pugcn_cap100k/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
        },
        "line_b_downsampled_vs_pugcn": {
            "base_cloud": KITTI / "velodyne_downsampled_50_val",
            "up_cloud": KITTI / "pugcn_cap_100k_downsampled50_filled",
            "base_det": p("results/rpn4096_no_distance_propose_comparison/01_downsampled50/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
            "up_det": p("results/pugcn_line_b_downsampled50_filled_pointRCNN_validation/inference_test_mode_rpn26000_safe_sampling/eval/epoch_no_number/val/test_mode/final_result/data"),
        },
    },
    "PDANS": {
        "line_a_original_vs_pdans": {
            "base_cloud": KITTI / "velodyne_original_val",
            "up_cloud": p("results/pdans_main_kitti_pipeline/pdans_original_up_bin"),
            "base_det": p("results/pointrcnn_clean_original_baseline_20260508_203809/evaluation/raw_output/eval/epoch_no_number/val/final_result/data"),
            "up_det": p("results/pdans_main_kitti_pipeline/pointrcnn_pdans_original_eval/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
        },
        "line_b_downsampled_vs_pdans": {
            "base_cloud": KITTI / "velodyne_downsampled_50_val",
            "up_cloud": p("results/pdans_main_kitti_pipeline/pdans_downsampled50_up_bin"),
            "base_det": p("results/rpn4096_no_distance_propose_comparison/01_downsampled50/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
            "up_det": p("results/pdans_main_kitti_pipeline/pointrcnn_pdans_downsampled50_eval/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
        },
    },
    "TULIP": {
        "line_a_original_vs_tulip": {
            "base_cloud": KITTI / "velodyne_original_val",
            "up_cloud": KITTI / "tulip_original_up_bin",
            "base_det": p("results/pointrcnn_clean_original_baseline_20260508_203809/evaluation/raw_output/eval/epoch_no_number/val/final_result/data"),
            "up_det": p("results/tulip_original_full_validation_pointrcnn/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
        },
        "line_b_downsampled_vs_tulip": {
            "base_cloud": KITTI / "velodyne_downsampled_50_val",
            "up_cloud": KITTI / "tulip_downsampled_up_bin",
            "base_det": p("results/rpn4096_no_distance_propose_comparison/01_downsampled50/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
            "up_det": p("results/tulip_line_b_downsampled_up_rpn4096_no_distance_propose_full_validation/inference/eval/epoch_no_number/val/test_mode/final_result/data"),
        },
    },
}


def read_points(path: Path) -> list[tuple[float, float, float, float]]:
    data = path.read_bytes()
    pts = []
    for x, y, z, r in struct.iter_unpack("ffff", data[: len(data) // 16 * 16]):
        pts.append((x, y, z, r))
    return pts


def raw_detection_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def read_png_size(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    data = path.read_bytes()[:24]
    if len(data) < 24 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return width, height


def split_xyz(points):
    return [[round(q[i], 4) for q in points] for i in range(3)]


def parse_kitti_boxes(path: Path, calib: dict[str, list[float]], source: str) -> list[dict]:
    boxes = []
    if not path.exists():
        return boxes
    for idx, line in enumerate(path.read_text().splitlines()):
        parts = line.split()
        if len(parts) < 15:
            continue
        if parts[0] == "DontCare":
            continue
        try:
            truncation = float(parts[1])
            occlusion = int(float(parts[2]))
            bbox = [float(x) for x in parts[4:8]]
            h, w, l = map(float, parts[8:11])
            x, y, z = map(float, parts[11:14])
            ry = float(parts[14])
            score = float(parts[15]) if len(parts) > 15 else None
        except ValueError:
            continue
        if h <= 0 or w <= 0 or l <= 0:
            continue
        box = {
            "id": idx,
            "class": parts[0],
            "truncation": truncation,
            "occlusion": occlusion,
            "bbox": bbox,
            "bbox_height": bbox[3] - bbox[1],
            "h": h,
            "w": w,
            "l": l,
            "center_camera": [x, y, z],
            "ry": ry,
            "score": score,
            "source": source,
            "corners_lidar": camera_box_to_lidar_corners(h, w, l, x, y, z, ry, calib),
        }
        boxes.append(box)
    return boxes


def parse_calib(path: Path) -> dict[str, list[float]]:
    out = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, rest = line.split(":", 1)
        vals = [float(x) for x in rest.split()]
        out[k] = vals
    return out


def matmul(a, b):
    rows, cols, inner = len(a), len(b[0]), len(b)
    return [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]


def matvec(a, v):
    return [sum(row[i] * v[i] for i in range(len(v))) for row in a]


def inv4(m):
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(4)] for i, row in enumerate(m)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("singular calibration matrix")
        a[col], a[pivot] = a[pivot], a[col]
        div = a[col][col]
        a[col] = [v / div for v in a[col]]
        for r in range(4):
            if r == col:
                continue
            fac = a[r][col]
            a[r] = [a[r][c] - fac * a[col][c] for c in range(8)]
    return [row[4:] for row in a]


def cam_to_lidar_matrix(calib):
    tr = calib.get("Tr_velo_to_cam") or calib.get("Tr_velo_cam")
    r0 = calib.get("R0_rect") or calib.get("R_rect")
    if not tr or not r0:
        raise ValueError("calibration missing Tr_velo_to_cam or R0_rect")
    tr4 = [tr[0:4], tr[4:8], tr[8:12], [0.0, 0.0, 0.0, 1.0]]
    r4 = [
        [r0[0], r0[1], r0[2], 0.0],
        [r0[3], r0[4], r0[5], 0.0],
        [r0[6], r0[7], r0[8], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return inv4(matmul(r4, tr4))


def velo_to_rect_matrix(calib):
    tr = calib.get("Tr_velo_to_cam") or calib.get("Tr_velo_cam")
    r0 = calib.get("R0_rect") or calib.get("R_rect")
    if not tr or not r0:
        raise ValueError("calibration missing Tr_velo_to_cam or R0_rect")
    tr4 = [tr[0:4], tr[4:8], tr[8:12], [0.0, 0.0, 0.0, 1.0]]
    r4 = [
        [r0[0], r0[1], r0[2], 0.0],
        [r0[3], r0[4], r0[5], 0.0],
        [r0[6], r0[7], r0[8], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return matmul(r4, tr4)


def p2_matrix(calib):
    p2 = calib.get("P2")
    if not p2:
        raise ValueError("calibration missing P2")
    return [p2[0:4], p2[4:8], p2[8:12]]


def camera_box_to_lidar_corners(h, w, l, x, y, z, ry, calib):
    c, s = math.cos(ry), math.sin(ry)
    xs = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    ys = [0, 0, 0, 0, -h, -h, -h, -h]
    zs = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    inv = cam_to_lidar_matrix(calib)
    corners = []
    for xc, yc, zc in zip(xs, ys, zs):
        xr = c * xc + s * zc + x
        yr = yc + y
        zr = -s * xc + c * zc + z
        lv = matvec(inv, [xr, yr, zr, 1.0])
        corners.append([round(lv[0], 4), round(lv[1], 4), round(lv[2], 4)])
    return corners


def project_lidar_to_image(point, velo_to_rect, p2):
    rect = matvec(velo_to_rect, [point[0], point[1], point[2], 1.0])
    if rect[2] <= 0:
        return None
    uvw = matvec(p2, [rect[0], rect[1], rect[2], 1.0])
    if abs(uvw[2]) < 1e-9:
        return None
    return uvw[0] / uvw[2], uvw[1] / uvw[2], rect[2]


def fov_points(points, calib, image_size):
    velo_to_rect = velo_to_rect_matrix(calib)
    p2 = p2_matrix(calib)
    width, height = image_size
    inside = []
    for point in points:
        projected = project_lidar_to_image(point, velo_to_rect, p2)
        if not projected:
            continue
        u, v, _ = projected
        if 0 <= u < width and 0 <= v < height:
            inside.append(point)
    return inside


def fov_frustum_traces(calib, image_size):
    width, height = image_size
    p2 = calib.get("P2")
    if not p2:
        return []
    fx, fy, cx, cy = p2[0], p2[5], p2[2], p2[6]
    inv = cam_to_lidar_matrix(calib)

    def cam_point(u, v, depth):
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        return [x, y, depth, 1.0]

    pixel_corners = [(0, 0), (width, 0), (width, height), (0, height)]
    near = [matvec(inv, cam_point(u, v, FOV_NEAR_M))[:3] for u, v in pixel_corners]
    far = [matvec(inv, cam_point(u, v, FOV_FAR_M))[:3] for u, v in pixel_corners]
    vertices = near + far
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [round(vertices[a][0], 4), round(vertices[b][0], 4), None]
        ys += [round(vertices[a][1], 4), round(vertices[b][1], 4), None]
        zs += [round(vertices[a][2], 4), round(vertices[b][2], 4), None]
    mesh = {
        "type": "mesh3d",
        "x": [round(v[0], 4) for v in vertices],
        "y": [round(v[1], 4) for v in vertices],
        "z": [round(v[2], 4) for v in vertices],
        "i": [0, 0, 0, 1, 2, 3, 4, 4, 4, 5],
        "j": [1, 2, 4, 5, 6, 7, 5, 6, 0, 1],
        "k": [2, 3, 5, 6, 7, 4, 6, 7, 1, 2],
        "name": "camera FOV wedge",
        "color": "#111827",
        "opacity": 0.09,
        "hoverinfo": "name",
    }
    lines = {
        "type": "scatter3d",
        "mode": "lines",
        "x": xs,
        "y": ys,
        "z": zs,
        "name": "camera FOV boundary",
        "line": {"color": "#111827", "width": 4},
        "hoverinfo": "name",
    }
    return [mesh, lines]


def center_dist(a, b):
    return math.sqrt(sum((a["center_camera"][i] - b["center_camera"][i]) ** 2 for i in range(3)))


def match_boxes(base, up):
    candidates = []
    for i, a in enumerate(base):
        for j, b in enumerate(up):
            if a["class"] == b["class"]:
                d = center_dist(a, b)
                if d <= MATCH_CENTER_THRESHOLD_M:
                    candidates.append((d, i, j))
    candidates.sort()
    used_a, used_b, matches = set(), set(), []
    for d, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append({"base_idx": i, "up_idx": j, "center_shift": d})
    missed = [i for i in range(len(base)) if i not in used_a]
    new = [j for j in range(len(up)) if j not in used_b]
    return matches, missed, new


def maybe_gt(frame, calib):
    label = KITTI / "label_2" / f"{frame}.txt"
    return parse_kitti_boxes(label, calib, "gt") if label.exists() else []


def box_key(prefix, idx):
    return f"{prefix}{idx:03d}"


def box_trace(box, color, name, width=6, dashed=False):
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        ca = box["corners_lidar"][a]
        cb = box["corners_lidar"][b]
        if dashed:
            segments = 7
            for k in range(segments):
                if k % 2:
                    continue
                t0, t1 = k / segments, (k + 1) / segments
                p0 = [ca[i] + (cb[i] - ca[i]) * t0 for i in range(3)]
                p1 = [ca[i] + (cb[i] - ca[i]) * t1 for i in range(3)]
                xs += [round(p0[0], 4), round(p1[0], 4), None]
                ys += [round(p0[1], 4), round(p1[1], 4), None]
                zs += [round(p0[2], 4), round(p1[2], 4), None]
        else:
            xs += [ca[0], cb[0], None]
            ys += [ca[1], cb[1], None]
            zs += [ca[2], cb[2], None]
    score = "" if box.get("score") is None else f" score={box['score']:.3f}"
    return {
        "type": "scatter3d",
        "mode": "lines",
        "x": xs,
        "y": ys,
        "z": zs,
        "name": f"{name} {box['id']} {box['class']}{score}",
        "line": {"color": color, "width": width},
        "hoverinfo": "name",
    }


def point_trace(points, color, name, size=1.4, opacity=0.55):
    x, y, z = split_xyz(points)
    return {
        "type": "scatter3d",
        "mode": "markers",
        "x": x,
        "y": y,
        "z": z,
        "name": name,
        "marker": {"size": size, "color": color, "opacity": opacity},
    }


def point_bounds(points):
    if not points:
        return None
    mins = [min(pt[i] for pt in points) for i in range(3)]
    maxs = [max(pt[i] for pt in points) for i in range(3)]
    return pad_bounds(mins, maxs)


def combined_bounds(*bounds):
    valid = [b for b in bounds if b]
    if not valid:
        return None
    mins = [min(b[0][i] for b in valid) for i in range(3)]
    maxs = [max(b[1][i] for b in valid) for i in range(3)]
    return pad_bounds(mins, maxs)


def pad_bounds(mins, maxs):
    out_min, out_max = [], []
    for mn, mx in zip(mins, maxs):
        pad = max((mx - mn) * 0.04, 0.5)
        out_min.append(round(mn - pad, 3))
        out_max.append(round(mx + pad, 3))
    return out_min, out_max


def relayout_for_bounds(bounds):
    if not bounds:
        return {"scene.xaxis.autorange": True, "scene.yaxis.autorange": True, "scene.zaxis.autorange": True}
    mins, maxs = bounds
    return {
        "scene.xaxis.range": [mins[0], maxs[0]],
        "scene.yaxis.range": [mins[1], maxs[1]],
        "scene.zaxis.range": [mins[2], maxs[2]],
    }


def camera_default():
    return {"eye": {"x": -1.55, "y": -1.85, "z": 0.9}, "up": {"x": 0, "y": 0, "z": 1}}


def write_plot(path: Path, title: str, traces, height=760, bounds=None, baseline_bounds=None, upsampled_bounds=None):
    payload = json.dumps(traces, separators=(",", ":"))
    layout = {
        "title": title,
        "height": height,
        "paper_bgcolor": "#f8fafc",
        "plot_bgcolor": "#f8fafc",
        "legend": {"orientation": "h"},
        "scene": {
            "xaxis": {"title": "LiDAR x"},
            "yaxis": {"title": "LiDAR y"},
            "zaxis": {"title": "LiDAR z"},
            "aspectmode": "data",
            "camera": camera_default(),
        },
        "margin": {"l": 0, "r": 0, "t": 48, "b": 0},
        "updatemenus": [{
            "type": "buttons",
            "direction": "right",
            "x": 0.01,
            "y": 1.08,
            "buttons": [
                {"label": "reset full scene", "method": "relayout", "args": [{**relayout_for_bounds(bounds), "scene.camera": camera_default()}]},
                {"label": "top view", "method": "relayout", "args": [{"scene.camera": {"eye": {"x": 0, "y": 0, "z": 2.6}, "up": {"x": 0, "y": 1, "z": 0}}}]},
                {"label": "front view", "method": "relayout", "args": [{"scene.camera": {"eye": {"x": 0, "y": -2.6, "z": 0.25}, "up": {"x": 0, "y": 0, "z": 1}}}]},
                {"label": "fit baseline", "method": "relayout", "args": [relayout_for_bounds(baseline_bounds or bounds)]},
                {"label": "fit upsampled", "method": "relayout", "args": [relayout_for_bounds(upsampled_bounds or bounds)]},
            ],
        }],
    }
    if bounds:
        b = relayout_for_bounds(bounds)
        layout["scene"]["xaxis"]["range"] = b["scene.xaxis.range"]
        layout["scene"]["yaxis"]["range"] = b["scene.yaxis.range"]
        layout["scene"]["zaxis"]["range"] = b["scene.zaxis.range"]
    path.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title><script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\"></script>"
        "<style>body{margin:0;background:#f8fafc;font-family:Arial;color:#172033}#plot{width:100%;height:100vh}</style>"
        "</head><body><div id=\"plot\"></div><script>"
        f"Plotly.newPlot('plot',{payload},{json.dumps(layout)},{{responsive:true,displaylogo:false,displayModeBar:true,scrollZoom:true,doubleClick:'reset'}});"
        "</script></body></html>"
    )


def crop_points(points, corners, margin):
    mins = [min(c[i] for c in corners) - margin for i in range(3)]
    maxs = [max(c[i] for c in corners) + margin for i in range(3)]
    return [pt for pt in points if mins[0] <= pt[0] <= maxs[0] and mins[1] <= pt[1] <= maxs[1] and mins[2] <= pt[2] <= maxs[2]]


def union_corners(record):
    corners = []
    for k in ("base_box", "up_box", "gt_box"):
        if record.get(k):
            corners.extend(record[k]["corners_lidar"])
    return corners


def nearest_gt(record, gt_boxes):
    ref = record.get("up_box") or record.get("base_box")
    if not ref or not gt_boxes:
        return None
    same = [g for g in gt_boxes if g["class"] == ref["class"]]
    if not same:
        return None
    g = min(same, key=lambda q: center_dist(ref, q))
    return g if center_dist(ref, g) <= 3.0 else None


def build_records(base_boxes, up_boxes, matches, missed, new, gt_boxes):
    records = []
    for m in matches:
        rec = {
            "status": "matched",
            "base_box": base_boxes[m["base_idx"]],
            "up_box": up_boxes[m["up_idx"]],
            "center_shift": m["center_shift"],
        }
        rec["gt_box"] = nearest_gt(rec, gt_boxes)
        records.append(rec)
    for i in missed:
        rec = {"status": "missed_after_upsampling", "base_box": base_boxes[i], "up_box": None, "center_shift": None}
        rec["gt_box"] = nearest_gt(rec, gt_boxes)
        records.append(rec)
    for j in new:
        rec = {"status": "new_after_upsampling", "base_box": None, "up_box": up_boxes[j], "center_shift": None}
        rec["gt_box"] = nearest_gt(rec, gt_boxes)
        records.append(rec)
    return records


def detection_tables(base_boxes, up_boxes, matches):
    matched_base = {m["base_idx"]: i for i, m in enumerate(matches)}
    matched_up = {m["up_idx"]: i for i, m in enumerate(matches)}
    base_rows = []
    up_rows = []
    pair_rows = []
    for i, box in enumerate(base_boxes):
        base_rows.append({
            "key": box_key("B", i),
            "class": box["class"],
            "score": box.get("score"),
            "status": f"M{matched_base[i]:03d}" if i in matched_base else "baseline-only",
        })
    for i, box in enumerate(up_boxes):
        up_rows.append({
            "key": box_key("U", i),
            "class": box["class"],
            "score": box.get("score"),
            "status": f"M{matched_up[i]:03d}" if i in matched_up else "upsampled-only",
        })
    for i, m in enumerate(matches):
        b = base_boxes[m["base_idx"]]
        u = up_boxes[m["up_idx"]]
        pair_rows.append({
            "key": box_key("M", i),
            "baseline": box_key("B", m["base_idx"]),
            "upsampled": box_key("U", m["up_idx"]),
            "class": b["class"],
            "base_score": b.get("score"),
            "up_score": u.get("score"),
            "score_change": None if b.get("score") is None or u.get("score") is None else u["score"] - b["score"],
            "center_shift": m["center_shift"],
        })
    return base_rows, up_rows, pair_rows


def full_frame_traces(base_points, up_points, base_boxes, up_boxes, gt_boxes, matches, mode, fov_traces=None):
    matched_base = {m["base_idx"] for m in matches}
    matched_up = {m["up_idx"] for m in matches}
    traces = []
    if fov_traces:
        traces.extend(fov_traces)
    if mode in ("baseline", "overlay"):
        traces.append(point_trace(base_points, "#c4c9d4", f"baseline/original points ({len(base_points)})", 1.15, 0.42))
        for i, box in enumerate(base_boxes):
            label = box_key("B", i)
            traces.append(box_trace(box, "#f28e2b", f"{label} baseline detection", 6, dashed=i not in matched_base))
    if mode in ("upsampled", "overlay"):
        traces.append(point_trace(up_points, "#1397d5", f"upsampled points ({len(up_points)})", 1.1, 0.55))
        for i, box in enumerate(up_boxes):
            label = box_key("U", i)
            traces.append(box_trace(box, "#d62728", f"{label} upsampled detection", 6, dashed=i not in matched_up))
    traces.extend(box_trace(b, "#5b2a86", "ground truth", 4) for b in gt_boxes)
    return traces


def crop_traces(bcrop, ucrop, rec, mode):
    traces = []
    if mode in ("baseline", "overlay"):
        traces.append(point_trace(bcrop, "#c4c9d4", f"baseline crop points ({len(bcrop)})", 2.2, 0.52))
        if rec.get("base_box"):
            traces.append(box_trace(rec["base_box"], "#f28e2b", "baseline detection", 6, dashed=rec["status"] != "matched"))
    if mode in ("upsampled", "overlay"):
        traces.append(point_trace(ucrop, "#2ca02c", f"upsampled crop points ({len(ucrop)})", 2.2, 0.68))
        if rec.get("up_box"):
            traces.append(box_trace(rec["up_box"], "#d62728", "upsampled detection", 6, dashed=rec["status"] != "matched"))
    if rec.get("gt_box"):
        traces.append(box_trace(rec["gt_box"], "#5b2a86", "ground truth", 4))
    return traces


def evaluation_traces(base_points, up_points, gt_boxes, base_boxes, up_boxes, base_audit, up_audit, mode="overlay", fov_traces=None):
    traces = []
    if fov_traces:
        traces.extend(fov_traces)
    if mode in ("baseline", "overlay"):
        traces.append(point_trace(base_points, "#c4c9d4", f"baseline points ({len(base_points)})", 1.2, 0.42))
    if mode in ("upsampled", "overlay"):
        traces.append(point_trace(up_points, "#1397d5", f"upsampled points ({len(up_points)})", 1.15, 0.56))
    gt = car_gt_boxes(gt_boxes)
    base_preds = car_pred_boxes(base_boxes)
    up_preds = car_pred_boxes(up_boxes)

    base_tp = set(base_audit["tp_pred_indices"])
    base_fp = set(base_audit["fp_pred_indices"])
    base_fn = set(base_audit["fn_gt_indices"])
    up_tp = set(up_audit["tp_pred_indices"])
    up_fp = set(up_audit["fp_pred_indices"])
    up_fn = set(up_audit["fn_gt_indices"])
    any_fn = base_fn | up_fn

    for i, box in enumerate(gt):
        if i in any_fn:
            traces.append(box_trace(box, "#5b2a86", f"FN / missed GT G{i:03d}", 8, dashed=True))
        else:
            traces.append(box_trace(box, "#111111", f"GT G{i:03d}", 5))
    if mode in ("baseline", "overlay"):
        for i, box in enumerate(base_preds):
            if i in base_tp:
                traces.append(box_trace(box, "#f28e2b", f"baseline TP B{i:03d}", 7))
            elif i in base_fp:
                traces.append(box_trace(box, "#d62728", f"baseline FP B{i:03d}", 6, dashed=True))
    if mode in ("upsampled", "overlay"):
        for i, box in enumerate(up_preds):
            if i in up_tp:
                traces.append(box_trace(box, "#2ca02c", f"upsampled TP U{i:03d}", 7))
            elif i in up_fp:
                traces.append(box_trace(box, "#d62728", f"upsampled FP U{i:03d}", 6, dashed=True))
    return traces


def eval_case_corners(case, gt_boxes, pred_boxes):
    corners = []
    gt = car_gt_boxes(gt_boxes)
    preds = car_pred_boxes(pred_boxes)
    if case.get("gt_idx") is not None and case["gt_idx"] < len(gt):
        corners.extend(gt[case["gt_idx"]]["corners_lidar"])
    if case.get("pred_idx") is not None and case["pred_idx"] < len(preds):
        corners.extend(preds[case["pred_idx"]]["corners_lidar"])
    return corners


def write_eval_object_dashboard(frame_dir, page_name, title, base_points, up_points, gt_boxes, pred_boxes, case, source):
    corners = eval_case_corners(case, gt_boxes, pred_boxes)
    if not corners:
        return None
    bcrop = crop_points(base_points, corners, MARGIN_M)
    ucrop = crop_points(up_points, corners, MARGIN_M)
    gt = car_gt_boxes(gt_boxes)
    preds = car_pred_boxes(pred_boxes)
    traces = [
        point_trace(bcrop, "#c4c9d4", f"baseline crop points ({len(bcrop)})", 2.2, 0.52),
        point_trace(ucrop, "#2ca02c", f"upsampled crop points ({len(ucrop)})", 2.2, 0.68),
    ]
    if case.get("gt_idx") is not None and case["gt_idx"] < len(gt):
        gt_label = "FN / missed GT" if case["status"] == "FN" else "matched GT"
        traces.append(box_trace(gt[case["gt_idx"]], "#5b2a86", gt_label, 8 if case["status"] == "FN" else 5, dashed=case["status"] == "FN"))
    if case.get("pred_idx") is not None and case["pred_idx"] < len(preds):
        color = "#2ca02c" if case["status"] == "TP" else "#d62728"
        traces.append(box_trace(preds[case["pred_idx"]], color, f"{source} {case['status']} prediction", 7, dashed=case["status"] == "FP"))
    bounds = combined_bounds(point_bounds(bcrop), point_bounds(ucrop))
    plot_file = page_name.replace(".html", "_plot.html")
    write_plot(frame_dir / plot_file, title, traces, 720, bounds=bounds)
    details = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in case.items())
    frame_dir.joinpath(page_name).write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:Arial;margin:20px;background:#f8fafc;color:#172033;line-height:1.45}iframe{width:100%;height:720px;border:1px solid #b4c1ce;background:white}table{border-collapse:collapse;background:white;margin:12px 0 18px}td,th{border:1px solid #cbd5df;padding:7px;text-align:left}th{background:#e8eef4}</style>"
        "</head><body>"
        f"<h1>{html.escape(title)}</h1><p><a href=\"combined_dashboard.html\">combined dashboard</a></p><table>{details}</table>"
        f"<iframe src=\"{plot_file}\"></iframe></body></html>"
    )
    return page_name


def explanation(base_n, up_n, matches, avg_score_change, avg_shift, base_crop, up_crop):
    bits = []
    if up_n < base_n:
        bits.append(f"Upsampling reduced the detection count ({base_n} to {up_n}), so missed boxes are a likely AP-degradation contributor.")
    elif up_n > base_n:
        bits.append(f"Upsampling increased the detection count ({base_n} to {up_n}); new boxes may include recovered objects or false positives.")
    else:
        bits.append(f"Detection count stayed unchanged at {base_n}, so score and localization changes are the main things to inspect.")
    if matches:
        bits.append(f"Matched boxes shifted by {avg_shift:.2f} m on average.")
        if avg_score_change < -0.01:
            bits.append(f"Confidence decreased on matched boxes by {avg_score_change:.3f} on average.")
        elif avg_score_change > 0.01:
            bits.append(f"Confidence increased on matched boxes by {avg_score_change:.3f} on average.")
        else:
            bits.append("Matched-box confidence was approximately stable.")
    else:
        bits.append("No same-class boxes matched within the center-distance threshold.")
    if up_crop > base_crop * 1.35:
        bits.append("Local crops often contain many generated points; inspect boundaries for noisy fill-in or object-edge distortion.")
    elif up_crop < base_crop * 0.8:
        bits.append("Local crops are still sparse after upsampling, which can leave PointRCNN with weak object support.")
    else:
        bits.append("Local crop density is similar; AP changes may come from subtle boundary distortion or background clutter near boxes.")
    return " ".join(bits)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def csv_value(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return v


def gt_difficulty_counts(gt_boxes):
    car = [g for g in gt_boxes if g["class"] == "Car"]

    def is_easy(g):
        return g["bbox_height"] >= 40 and g["occlusion"] <= 0 and g["truncation"] <= 0.15

    def is_moderate(g):
        return g["bbox_height"] >= 25 and g["occlusion"] <= 1 and g["truncation"] <= 0.30

    def is_hard(g):
        return g["bbox_height"] >= 25 and g["occlusion"] <= 2 and g["truncation"] <= 0.50

    return {
        "gt_car_count": len(car),
        "gt_car_easy_count": sum(1 for g in car if is_easy(g)),
        "gt_car_moderate_count": sum(1 for g in car if is_moderate(g)),
        "gt_car_hard_count": sum(1 for g in car if is_hard(g)),
    }


def gt_prediction_audit(gt_boxes, pred_boxes, class_name="Car", threshold=GT_PRED_CENTER_THRESHOLD_M):
    gt = [g for g in gt_boxes if g["class"] == class_name]
    preds = [p for p in pred_boxes if p["class"] == class_name]
    candidates = []
    for gi, g in enumerate(gt):
        for pi, pred in enumerate(preds):
            d = center_dist(g, pred)
            if d <= threshold:
                candidates.append((d, gi, pi, bev_aabb_iou(g, pred)))
    candidates.sort()
    used_g, used_p, matches = set(), set(), []
    for d, gi, pi, iou in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append({"gt_idx": gi, "pred_idx": pi, "center_distance": d, "bev_iou": iou, "score": preds[pi].get("score")})
    tp = len(used_g)
    fn = len(gt) - tp
    fp = len(preds) - len(used_p)
    tp_scores = [m["score"] for m in matches if m.get("score") is not None]
    fp_scores = [preds[i].get("score") for i in range(len(preds)) if i not in used_p and preds[i].get("score") is not None]
    ious = [m["bev_iou"] for m in matches]
    return {
        "gt_pred_match_policy": f"same-class camera-center distance <= {threshold:.1f} m fallback; BEV IoU is axis-aligned approximate, not official KITTI AP",
        "gt_pred_gt_count": len(gt),
        "gt_pred_prediction_count": len(preds),
        "gt_pred_matched_prediction_count": tp,
        "gt_pred_missed_gt_count": fn,
        "gt_pred_false_positive_count": fp,
        "gt_pred_approx_recall": (tp / len(gt)) if gt else None,
        "gt_pred_approx_precision": (tp / len(preds)) if preds else None,
        "gt_pred_mean_iou": (sum(ious) / len(ious)) if ious else None,
        "gt_pred_mean_tp_score": (sum(tp_scores) / len(tp_scores)) if tp_scores else None,
        "gt_pred_mean_fp_score": (sum(fp_scores) / len(fp_scores)) if fp_scores else None,
        "matches": matches,
        "tp_pred_indices": sorted(used_p),
        "fp_pred_indices": [i for i in range(len(preds)) if i not in used_p],
        "fn_gt_indices": [i for i in range(len(gt)) if i not in used_g],
    }


def bev_aabb_iou(a, b):
    ax = [p[0] for p in a["corners_lidar"]]
    ay = [p[1] for p in a["corners_lidar"]]
    bx = [p[0] for p in b["corners_lidar"]]
    by = [p[1] for p in b["corners_lidar"]]
    aminx, amaxx, aminy, amaxy = min(ax), max(ax), min(ay), max(ay)
    bminx, bmaxx, bminy, bmaxy = min(bx), max(bx), min(by), max(by)
    ix = max(0.0, min(amaxx, bmaxx) - max(aminx, bminx))
    iy = max(0.0, min(amaxy, bmaxy) - max(aminy, bminy))
    inter = ix * iy
    area_a = max(0.0, amaxx - aminx) * max(0.0, amaxy - aminy)
    area_b = max(0.0, bmaxx - bminx) * max(0.0, bmaxy - bminy)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def car_gt_boxes(gt_boxes):
    return [g for g in gt_boxes if g["class"] == "Car"]


def car_pred_boxes(pred_boxes):
    return [p for p in pred_boxes if p["class"] == "Car"]


def eval_interpretation(base_audit, up_audit, base_pred_count, up_pred_count):
    bits = []
    if up_audit["gt_pred_missed_gt_count"] > base_audit["gt_pred_missed_gt_count"]:
        bits.append("Upsampling increases missed GT objects, suggesting recall degradation.")
    if up_audit["gt_pred_false_positive_count"] > base_audit["gt_pred_false_positive_count"]:
        bits.append("Upsampling introduces more false positives, suggesting precision degradation.")
    b_iou = base_audit.get("gt_pred_mean_iou")
    u_iou = up_audit.get("gt_pred_mean_iou")
    if b_iou is not None and u_iou is not None and u_iou + 0.02 < b_iou:
        bits.append("Upsampling keeps some detections but worsens localization.")
    if up_pred_count < base_pred_count:
        bits.append("The lower prediction count is meaningful only relative to GT count and TP/FN matching.")
    if not bits:
        bits.append("Frame-level TP/FP/FN metrics do not show a single dominant degradation mode; inspect object crops for localization and density changes.")
    return " ".join(bits)


def eval_cases_from_audit(audit, source):
    cases = []
    for match_idx, m in enumerate(audit["matches"]):
        cases.append({
            "source": source,
            "status": "TP",
            "case_id": f"{source}_TP_{match_idx:03d}",
            "gt_idx": m["gt_idx"],
            "pred_idx": m["pred_idx"],
            "bev_iou": m["bev_iou"],
            "center_distance": m["center_distance"],
            "score": m.get("score"),
        })
    for i, pred_idx in enumerate(audit["fp_pred_indices"]):
        cases.append({
            "source": source,
            "status": "FP",
            "case_id": f"{source}_FP_{i:03d}",
            "gt_idx": None,
            "pred_idx": pred_idx,
            "bev_iou": None,
            "center_distance": None,
            "score": None,
        })
    for i, gt_idx in enumerate(audit["fn_gt_indices"]):
        cases.append({
            "source": source,
            "status": "FN",
            "case_id": f"{source}_FN_{i:03d}",
            "gt_idx": gt_idx,
            "pred_idx": None,
            "bev_iou": None,
            "center_distance": None,
            "score": None,
        })
    return cases


def build_candidate_cache(frames):
    wanted = {f"{frame}.txt": frame for frame in frames}
    cache = {frame: [] for frame in frames}
    out_rel = rel(OUT)
    try:
        proc = subprocess.run(
            ["rg", "--files", "results"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        for line in proc.stdout.splitlines():
            name = Path(line).name
            if name not in wanted:
                continue
            if line.startswith(out_rel + "/"):
                continue
            path = ROOT / line
            if path.is_file():
                cache[wanted[name]].append(path)
        for frame in cache:
            cache[frame] = sorted(cache[frame])
        return cache
    except (OSError, subprocess.CalledProcessError):
        pass

    results_root = ROOT / "results"
    if not results_root.exists():
        return cache
    skip_names = {
        OUT.name,
        "interactive_object_crop_visualization",
        "interactive_object_crop_visualization_improved",
        "interactive_object_crop_visualization_improved_v2",
        "interactive_object_crop_visualization_improved_v4",
        "interactive_object_crop_visualization_improved_v5",
        "interactive_object_crop_visualization_improved_v6",
    }
    for dirpath, dirnames, filenames in os.walk(results_root):
        dirnames[:] = [d for d in dirnames if d not in skip_names and not d.endswith("_visualization")]
        for name in filenames:
            if name in wanted:
                cache[wanted[name]].append(Path(dirpath) / name)
    for frame in cache:
        cache[frame] = sorted(cache[frame])
    return cache


def classify_detection_path(path, method, line, role, exists=True):
    rel_path = rel(path)
    low = rel_path.lower()
    warnings = []
    if not exists:
        return "ERROR", ["detection txt missing"], False, False, False
    suspicious = any(tok in low for tok in SUSPICIOUS_PATH_TOKENS)
    subset = any(tok in low for tok in SUBSET_PATH_TOKENS)
    ap_like = any(tok in low for tok in AP_LIKE_PATH_TOKENS) and "final_result/data" in low
    if suspicious:
        warnings.append("path contains smoke/test/debug/temp-style token")
    if subset:
        warnings.append("path appears to be from a limited-frame subset")
    if not ap_like:
        warnings.append("path cannot be confidently linked to an AP/evaluation output folder")
    method_low = method.lower().replace("-", "")
    line_low = line.lower().replace("-", "")
    normalized = low.replace("-", "").replace("_", "")
    method_match = True
    if role == "baseline":
        if "line_a" in line_low:
            method_match = "original" in normalized or "cleanoriginalbaseline" in normalized
        elif "line_b" in line_low:
            method_match = "downsampled" in normalized or "01downsampled50" in normalized
    else:
        if method_low == "pugcn":
            method_match = "pugcn" in normalized
        elif method_low == "punet":
            method_match = "punet" in normalized
        else:
            method_match = method_low in normalized
    if not method_match:
        warnings.append("path does not clearly match method/line role")
    if suspicious or not exists:
        verdict = "ERROR" if not exists else "WARNING"
    elif warnings:
        verdict = "WARNING"
    else:
        verdict = "OK"
    return verdict, warnings, suspicious, subset, ap_like


def combine_verdicts(*verdicts):
    if any(v == "ERROR" for v in verdicts):
        return "ERROR"
    if any(v == "WARNING" for v in verdicts):
        return "WARNING"
    if all(v == "SKIPPED" for v in verdicts):
        return "SKIPPED"
    return "OK"


def selected_path_reason(method, line, role):
    if role == "baseline":
        if "line_a" in line:
            return "explicit generator mapping to original PointRCNN baseline final_result/data"
        return "explicit generator mapping to downsampled50 PointRCNN baseline final_result/data"
    return f"explicit generator mapping to {method} {line} upsampled PointRCNN final_result/data"


def generate():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    summary_rows = []
    availability = []
    ap_audit_rows = []
    candidate_rows = []
    gt_matching_rows = []
    index_sections = []
    candidate_cache = build_candidate_cache(FRAMES)

    for method in METHODS:
        method_links = []
        for line, spec in SPECS[method].items():
            line_links = []
            for frame in FRAMES:
                paths = {
                    "baseline_point_cloud": spec["base_cloud"] / f"{frame}.bin",
                    "upsampled_point_cloud": spec["up_cloud"] / f"{frame}.bin",
                    "baseline_detection": spec["base_det"] / f"{frame}.txt",
                    "upsampled_detection": spec["up_det"] / f"{frame}.txt",
                    "calibration": KITTI / "calib" / f"{frame}.txt",
                    "image_2": KITTI / "image_2" / f"{frame}.png",
                }
                candidates = candidate_cache[frame]
                selected_base_rel = rel(paths["baseline_detection"])
                selected_up_rel = rel(paths["upsampled_detection"])
                for cand in candidates:
                    cand_rel = rel(cand)
                    candidate_rows.append({
                        "method": method,
                        "line": line,
                        "frame_id": frame,
                        "candidate_detection_txt_path": cand_rel,
                        "raw_line_count": raw_detection_line_count(cand),
                        "selected_as_baseline": cand_rel == selected_base_rel,
                        "selected_as_upsampled": cand_rel == selected_up_rel,
                        "selection_reason": (
                            selected_path_reason(method, line, "baseline") if cand_rel == selected_base_rel
                            else selected_path_reason(method, line, "upsampled") if cand_rel == selected_up_rel
                            else "not selected by explicit generator mapping"
                        ),
                    })
                missing = [name for name, path in paths.items() if not path.exists()]
                availability.append({
                    "method": method,
                    "line": line,
                    "frame": frame,
                    "status": "skipped" if missing else "visualized",
                    "missing": missing,
                    "paths": {k: rel(v) for k, v in paths.items()},
                })
                if missing:
                    base_verdict, base_warn, _, _, _ = classify_detection_path(paths["baseline_detection"], method, line, "baseline", paths["baseline_detection"].exists())
                    up_verdict, up_warn, _, _, _ = classify_detection_path(paths["upsampled_detection"], method, line, "upsampled", paths["upsampled_detection"].exists())
                    verdict = "ERROR" if "baseline_detection" in missing or "upsampled_detection" in missing else "SKIPPED"
                    warnings = [f"missing inputs: {', '.join(missing)}"] + base_warn + up_warn
                    ap_audit_rows.append({
                        "method": method,
                        "line": line,
                        "frame_id": frame,
                        "verdict": verdict,
                        "warnings": "; ".join(dict.fromkeys(warnings)),
                        "candidate_detection_file_count": len(candidates),
                        "baseline_detection_txt_path": selected_base_rel,
                        "upsampled_detection_txt_path": selected_up_rel,
                        "baseline_path_verdict": base_verdict,
                        "upsampled_path_verdict": up_verdict,
                        "baseline_detection_raw_line_count": raw_detection_line_count(paths["baseline_detection"]),
                        "upsampled_detection_raw_line_count": raw_detection_line_count(paths["upsampled_detection"]),
                        "baseline_boxes_rendered_in_html": "",
                        "upsampled_boxes_rendered_in_html": "",
                        "metadata_baseline_detection_count": "",
                        "metadata_upsampled_detection_count": "",
                        "gt_car_count": "",
                        "gt_car_easy_count": "",
                        "gt_car_moderate_count": "",
                        "gt_car_hard_count": "",
                        "baseline_gt_approx_recall": "",
                        "baseline_gt_approx_precision": "",
                        "upsampled_gt_approx_recall": "",
                        "upsampled_gt_approx_precision": "",
                    })
                    continue

                frame_dir = OUT / method / line / f"frame_{frame}"
                frame_dir.mkdir(parents=True, exist_ok=True)
                calib = parse_calib(paths["calibration"])
                image_size = read_png_size(paths["image_2"]) or (1242, 375)
                base_points = read_points(paths["baseline_point_cloud"])
                up_points = read_points(paths["upsampled_point_cloud"])
                base_boxes = parse_kitti_boxes(paths["baseline_detection"], calib, "baseline")
                up_boxes = parse_kitti_boxes(paths["upsampled_detection"], calib, "upsampled")
                base_raw_lines = raw_detection_line_count(paths["baseline_detection"])
                up_raw_lines = raw_detection_line_count(paths["upsampled_detection"])
                gt_boxes = maybe_gt(frame, calib)
                gt_counts = gt_difficulty_counts(gt_boxes)
                matches, missed, new = match_boxes(base_boxes, up_boxes)
                records = build_records(base_boxes, up_boxes, matches, missed, new, gt_boxes)
                baseline_gt_audit = gt_prediction_audit(gt_boxes, base_boxes)
                upsampled_gt_audit = gt_prediction_audit(gt_boxes, up_boxes)
                base_path_verdict, base_path_warnings, base_suspicious, base_subset, base_ap_like = classify_detection_path(paths["baseline_detection"], method, line, "baseline")
                up_path_verdict, up_path_warnings, up_suspicious, up_subset, up_ap_like = classify_detection_path(paths["upsampled_detection"], method, line, "upsampled")
                base_bounds = point_bounds(base_points)
                up_bounds = point_bounds(up_points)
                all_bounds = combined_bounds(base_bounds, up_bounds)
                base_fov_points = fov_points(base_points, calib, image_size)
                up_fov_points = fov_points(up_points, calib, image_size)
                base_fov_bounds = point_bounds(base_fov_points)
                up_fov_bounds = point_bounds(up_fov_points)
                all_fov_bounds = combined_bounds(base_fov_bounds, up_fov_bounds)
                frustum_traces = fov_frustum_traces(calib, image_size)
                write_plot(
                    frame_dir / "baseline_full_frame.html",
                    f"{method} {line} frame {frame}: baseline detections only",
                    full_frame_traces(base_points, up_points, base_boxes, up_boxes, gt_boxes, matches, "baseline", frustum_traces),
                    bounds=base_bounds,
                    baseline_bounds=base_bounds,
                    upsampled_bounds=up_bounds,
                )
                write_plot(
                    frame_dir / "upsampled_full_frame.html",
                    f"{method} {line} frame {frame}: upsampled detections only",
                    full_frame_traces(base_points, up_points, base_boxes, up_boxes, gt_boxes, matches, "upsampled", frustum_traces),
                    bounds=up_bounds,
                    baseline_bounds=base_bounds,
                    upsampled_bounds=up_bounds,
                )
                write_plot(
                    frame_dir / "full_frame.html",
                    f"{method} {line} frame {frame}: overlay comparison",
                    full_frame_traces(base_points, up_points, base_boxes, up_boxes, gt_boxes, matches, "overlay", frustum_traces),
                    bounds=all_bounds,
                    baseline_bounds=base_bounds,
                    upsampled_bounds=up_bounds,
                )
                write_plot(
                    frame_dir / "baseline_fov_frame.html",
                    f"{method} {line} frame {frame}: baseline camera-FOV view",
                    full_frame_traces(base_fov_points, up_fov_points, base_boxes, up_boxes, gt_boxes, matches, "baseline", frustum_traces),
                    bounds=base_fov_bounds or base_bounds,
                    baseline_bounds=base_fov_bounds,
                    upsampled_bounds=up_fov_bounds,
                )
                write_plot(
                    frame_dir / "upsampled_fov_frame.html",
                    f"{method} {line} frame {frame}: upsampled camera-FOV view",
                    full_frame_traces(base_fov_points, up_fov_points, base_boxes, up_boxes, gt_boxes, matches, "upsampled", frustum_traces),
                    bounds=up_fov_bounds or up_bounds,
                    baseline_bounds=base_fov_bounds,
                    upsampled_bounds=up_fov_bounds,
                )
                write_plot(
                    frame_dir / "overlay_fov_frame.html",
                    f"{method} {line} frame {frame}: overlay camera-FOV view",
                    evaluation_traces(base_fov_points, up_fov_points, gt_boxes, base_boxes, up_boxes, baseline_gt_audit, upsampled_gt_audit, "overlay", frustum_traces),
                    bounds=all_fov_bounds or all_bounds,
                    baseline_bounds=base_fov_bounds,
                    upsampled_bounds=up_fov_bounds,
                )
                write_plot(
                    frame_dir / "baseline_fov_frame.html",
                    f"{method} {line} frame {frame}: baseline camera-FOV evaluation view",
                    evaluation_traces(base_fov_points, up_fov_points, gt_boxes, base_boxes, up_boxes, baseline_gt_audit, upsampled_gt_audit, "baseline", frustum_traces),
                    bounds=base_fov_bounds or base_bounds,
                    baseline_bounds=base_fov_bounds,
                    upsampled_bounds=up_fov_bounds,
                )
                write_plot(
                    frame_dir / "upsampled_fov_frame.html",
                    f"{method} {line} frame {frame}: upsampled camera-FOV evaluation view",
                    evaluation_traces(base_fov_points, up_fov_points, gt_boxes, base_boxes, up_boxes, baseline_gt_audit, upsampled_gt_audit, "upsampled", frustum_traces),
                    bounds=up_fov_bounds or up_bounds,
                    baseline_bounds=base_fov_bounds,
                    upsampled_bounds=up_fov_bounds,
                )

                score_changes = []
                shifts = []
                total_base_crop = 0
                total_up_crop = 0
                object_links = []
                eval_object_links = []
                for idx, case in enumerate(eval_cases_from_audit(baseline_gt_audit, "baseline")):
                    fname = f"eval_object_baseline_{idx:03d}_{case['status'].lower()}.html"
                    title = f"{method} {line} frame {frame} baseline {case['status']} {idx:03d}"
                    page = write_eval_object_dashboard(frame_dir, fname, title, base_points, up_points, gt_boxes, base_boxes, case, "baseline")
                    if page:
                        eval_object_links.append({**case, "file": page})
                for idx, case in enumerate(eval_cases_from_audit(upsampled_gt_audit, "upsampled")):
                    fname = f"eval_object_upsampled_{idx:03d}_{case['status'].lower()}.html"
                    title = f"{method} {line} frame {frame} upsampled {case['status']} {idx:03d}"
                    page = write_eval_object_dashboard(frame_dir, fname, title, base_points, up_points, gt_boxes, up_boxes, case, "upsampled")
                    if page:
                        eval_object_links.append({**case, "file": page})
                for idx, rec in enumerate(records):
                    corners = union_corners(rec)
                    if not corners:
                        continue
                    bcrop = crop_points(base_points, corners, MARGIN_M)
                    ucrop = crop_points(up_points, corners, MARGIN_M)
                    total_base_crop += len(bcrop)
                    total_up_crop += len(ucrop)
                    if rec.get("base_box") and rec.get("up_box"):
                        a, b = rec["base_box"], rec["up_box"]
                        if a.get("score") is not None and b.get("score") is not None:
                            score_changes.append(b["score"] - a["score"])
                        shifts.append(rec["center_shift"])
                    crop_name = f"object_box_{idx:03d}.html"
                    title = f"{method} {line} frame {frame} {rec['status']} box {idx:03d}"
                    crop_base_bounds = point_bounds(bcrop)
                    crop_up_bounds = point_bounds(ucrop)
                    crop_all_bounds = combined_bounds(crop_base_bounds, crop_up_bounds)
                    base_crop_file = f"object_box_{idx:03d}_baseline_crop.html"
                    up_crop_file = f"object_box_{idx:03d}_upsampled_crop.html"
                    overlay_crop_file = f"object_box_{idx:03d}_overlay_crop.html"
                    write_plot(
                        frame_dir / base_crop_file,
                        f"{title}: baseline crop",
                        crop_traces(bcrop, ucrop, rec, "baseline"),
                        720,
                        bounds=crop_base_bounds or crop_all_bounds,
                        baseline_bounds=crop_base_bounds,
                        upsampled_bounds=crop_up_bounds,
                    )
                    write_plot(
                        frame_dir / up_crop_file,
                        f"{title}: upsampled crop",
                        crop_traces(bcrop, ucrop, rec, "upsampled"),
                        720,
                        bounds=crop_up_bounds or crop_all_bounds,
                        baseline_bounds=crop_base_bounds,
                        upsampled_bounds=crop_up_bounds,
                    )
                    write_plot(
                        frame_dir / overlay_crop_file,
                        f"{title}: overlay crop",
                        crop_traces(bcrop, ucrop, rec, "overlay"),
                        720,
                        bounds=crop_all_bounds,
                        baseline_bounds=crop_base_bounds,
                        upsampled_bounds=crop_up_bounds,
                    )
                    write_object_dashboard(
                        frame_dir,
                        crop_name,
                        title,
                        rec,
                        base_crop_file,
                        up_crop_file,
                        overlay_crop_file,
                        len(bcrop),
                        len(ucrop),
                    )
                    ref_box = rec.get("up_box") or rec.get("base_box")
                    object_links.append({
                        "idx": idx,
                        "file": crop_name,
                        "status": rec["status"],
                        "class": ref_box["class"] if ref_box else "",
                        "base_score": rec["base_box"].get("score") if rec.get("base_box") else None,
                        "up_score": rec["up_box"].get("score") if rec.get("up_box") else None,
                        "center_shift": rec.get("center_shift"),
                        "base_crop_points": len(bcrop),
                        "upsampled_crop_points": len(ucrop),
                    })

                avg_score = sum(score_changes) / len(score_changes) if score_changes else None
                avg_shift = sum(shifts) / len(shifts) if shifts else None
                notes = explanation(len(base_boxes), len(up_boxes), matches, avg_score or 0.0, avg_shift or 0.0, total_base_crop, total_up_crop)
                eval_notes = eval_interpretation(baseline_gt_audit, upsampled_gt_audit, len(base_boxes), len(up_boxes))
                audit_warnings = []
                audit_warnings.extend(base_path_warnings)
                audit_warnings.extend(up_path_warnings)
                if paths["baseline_detection"] == paths["upsampled_detection"]:
                    audit_warnings.append("baseline and upsampled detection paths are identical")
                if base_raw_lines != len(base_boxes):
                    audit_warnings.append("baseline raw txt line count differs from rendered parsed box count")
                if up_raw_lines != len(up_boxes):
                    audit_warnings.append("upsampled raw txt line count differs from rendered parsed box count")
                if not (KITTI / "label_2" / f"{frame}.txt").exists():
                    audit_warnings.append("GT label_2 file missing")
                if not frustum_traces:
                    audit_warnings.append("camera FOV frustum could not be constructed from calibration")
                ap_verdict = combine_verdicts(base_path_verdict, up_path_verdict)
                if audit_warnings and ap_verdict == "OK":
                    ap_verdict = "WARNING"

                metadata = {
                    "method": method,
                    "line": line,
                    "frame_id": frame,
                    "paths": {k: rel(v) for k, v in paths.items()},
                    "gt_label_path": rel(KITTI / "label_2" / f"{frame}.txt") if (KITTI / "label_2" / f"{frame}.txt").exists() else None,
                    "baseline_point_count": len(base_points),
                    "upsampled_point_count": len(up_points),
                    "baseline_fov_point_count": len(base_fov_points),
                    "upsampled_fov_point_count": len(up_fov_points),
                    "baseline_detection_count": len(base_boxes),
                    "upsampled_detection_count": len(up_boxes),
                    "gt_box_count": len(gt_boxes),
                    **gt_counts,
                    "matched_detection_count": len(matches),
                    "missed_detection_count": len(missed),
                    "new_detection_count": len(new),
                    "average_score_change": avg_score,
                    "average_center_shift": avg_shift,
                    "match_policy": f"same class and camera-center distance <= {MATCH_CENTER_THRESHOLD_M} m; greedy nearest-first fallback, not exact 3D IoU",
                    "full_frame_render_policy": "Every valid KITTI-format line from each final PointRCNN txt is rendered. No score threshold, top-k filter, or matched-only filter is applied.",
                    "camera_fov_note": FOV_NOTE,
                    "camera_fov_image_size": {"width": image_size[0], "height": image_size[1]},
                    "detection_count_audit": {
                        "baseline_detection_txt_path": rel(paths["baseline_detection"]),
                        "upsampled_detection_txt_path": rel(paths["upsampled_detection"]),
                        "baseline_detection_raw_line_count": base_raw_lines,
                        "upsampled_detection_raw_line_count": up_raw_lines,
                        "baseline_boxes_rendered_in_html": len(base_boxes),
                        "upsampled_boxes_rendered_in_html": len(up_boxes),
                        "candidate_detection_file_count_for_frame": len(candidates),
                        "baseline_selection_reason": selected_path_reason(method, line, "baseline"),
                        "upsampled_selection_reason": selected_path_reason(method, line, "upsampled"),
                        "baseline_path_verdict": base_path_verdict,
                        "upsampled_path_verdict": up_path_verdict,
                    },
                    "ap_consistency_verdict": ap_verdict,
                    "ap_consistency_warnings": list(dict.fromkeys(audit_warnings)),
                    "baseline_gt_prediction_audit": baseline_gt_audit,
                    "upsampled_gt_prediction_audit": upsampled_gt_audit,
                    "evaluation_interpretation": eval_notes,
                    "crop_margin_m": MARGIN_M,
                    "baseline_detections": detection_tables(base_boxes, up_boxes, matches)[0],
                    "upsampled_detections": detection_tables(base_boxes, up_boxes, matches)[1],
                    "matched_pairs": detection_tables(base_boxes, up_boxes, matches)[2],
                    "object_boxes": object_links,
                    "evaluation_object_pages": eval_object_links,
                    "notes": notes,
                }
                (frame_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
                base_detection_rows, up_detection_rows, matched_pair_rows = detection_tables(base_boxes, up_boxes, matches)
                write_dashboard(frame_dir, method, line, frame, metadata, object_links, base_detection_rows, up_detection_rows, matched_pair_rows)
                line_links.append(f"frame_{frame}/combined_dashboard.html")
                summary_rows.append({
                    "method": method,
                    "line": line,
                    "frame_id": frame,
                    "baseline_point_count": len(base_points),
                    "upsampled_point_count": len(up_points),
                    "baseline_fov_point_count": len(base_fov_points),
                    "upsampled_fov_point_count": len(up_fov_points),
                    "baseline_detection_count": len(base_boxes),
                    "upsampled_detection_count": len(up_boxes),
                    "gt_box_count": len(gt_boxes),
                    **gt_counts,
                    "matched_detection_count": len(matches),
                    "missed_detection_count": len(missed),
                    "new_detection_count": len(new),
                    "baseline_detection_txt_path": rel(paths["baseline_detection"]),
                    "upsampled_detection_txt_path": rel(paths["upsampled_detection"]),
                    "baseline_detection_raw_line_count": base_raw_lines,
                    "upsampled_detection_raw_line_count": up_raw_lines,
                    "baseline_boxes_rendered_in_html": len(base_boxes),
                    "upsampled_boxes_rendered_in_html": len(up_boxes),
                    "ap_consistency_verdict": ap_verdict,
                    "ap_consistency_warnings": "; ".join(dict.fromkeys(audit_warnings)),
                    "baseline_gt_matched_prediction_count": baseline_gt_audit["gt_pred_matched_prediction_count"],
                    "baseline_gt_missed_gt_count": baseline_gt_audit["gt_pred_missed_gt_count"],
                    "baseline_gt_false_positive_count": baseline_gt_audit["gt_pred_false_positive_count"],
                    "baseline_gt_approx_recall": baseline_gt_audit["gt_pred_approx_recall"],
                    "baseline_gt_approx_precision": baseline_gt_audit["gt_pred_approx_precision"],
                    "baseline_gt_mean_iou": baseline_gt_audit["gt_pred_mean_iou"],
                    "baseline_gt_mean_tp_score": baseline_gt_audit["gt_pred_mean_tp_score"],
                    "baseline_gt_mean_fp_score": baseline_gt_audit["gt_pred_mean_fp_score"],
                    "upsampled_gt_matched_prediction_count": upsampled_gt_audit["gt_pred_matched_prediction_count"],
                    "upsampled_gt_missed_gt_count": upsampled_gt_audit["gt_pred_missed_gt_count"],
                    "upsampled_gt_false_positive_count": upsampled_gt_audit["gt_pred_false_positive_count"],
                    "upsampled_gt_approx_recall": upsampled_gt_audit["gt_pred_approx_recall"],
                    "upsampled_gt_approx_precision": upsampled_gt_audit["gt_pred_approx_precision"],
                    "upsampled_gt_mean_iou": upsampled_gt_audit["gt_pred_mean_iou"],
                    "upsampled_gt_mean_tp_score": upsampled_gt_audit["gt_pred_mean_tp_score"],
                    "upsampled_gt_mean_fp_score": upsampled_gt_audit["gt_pred_mean_fp_score"],
                    "evaluation_interpretation": eval_notes,
                    "average_score_change": avg_score,
                    "average_center_shift": avg_shift,
                    "notes": notes,
                })
                gt_matching_rows.append({
                    "method": method,
                    "line": line,
                    "frame_id": frame,
                    "gt_car_count": gt_counts["gt_car_count"],
                    "baseline_prediction_count": len(car_pred_boxes(base_boxes)),
                    "upsampled_prediction_count": len(car_pred_boxes(up_boxes)),
                    "baseline_tp": baseline_gt_audit["gt_pred_matched_prediction_count"],
                    "baseline_fp": baseline_gt_audit["gt_pred_false_positive_count"],
                    "baseline_fn": baseline_gt_audit["gt_pred_missed_gt_count"],
                    "baseline_precision": baseline_gt_audit["gt_pred_approx_precision"],
                    "baseline_recall": baseline_gt_audit["gt_pred_approx_recall"],
                    "baseline_mean_iou": baseline_gt_audit["gt_pred_mean_iou"],
                    "baseline_mean_tp_score": baseline_gt_audit["gt_pred_mean_tp_score"],
                    "baseline_mean_fp_score": baseline_gt_audit["gt_pred_mean_fp_score"],
                    "upsampled_tp": upsampled_gt_audit["gt_pred_matched_prediction_count"],
                    "upsampled_fp": upsampled_gt_audit["gt_pred_false_positive_count"],
                    "upsampled_fn": upsampled_gt_audit["gt_pred_missed_gt_count"],
                    "upsampled_precision": upsampled_gt_audit["gt_pred_approx_precision"],
                    "upsampled_recall": upsampled_gt_audit["gt_pred_approx_recall"],
                    "upsampled_mean_iou": upsampled_gt_audit["gt_pred_mean_iou"],
                    "upsampled_mean_tp_score": upsampled_gt_audit["gt_pred_mean_tp_score"],
                    "upsampled_mean_fp_score": upsampled_gt_audit["gt_pred_mean_fp_score"],
                    "evaluation_interpretation": eval_notes,
                    "dashboard": f"{method}/{line}/frame_{frame}/combined_dashboard.html",
                })
                ap_audit_rows.append({
                    "method": method,
                    "line": line,
                    "frame_id": frame,
                    "verdict": ap_verdict,
                    "warnings": "; ".join(dict.fromkeys(audit_warnings)),
                    "candidate_detection_file_count": len(candidates),
                    "baseline_detection_txt_path": selected_base_rel,
                    "upsampled_detection_txt_path": selected_up_rel,
                    "baseline_path_verdict": base_path_verdict,
                    "upsampled_path_verdict": up_path_verdict,
                    "baseline_detection_raw_line_count": base_raw_lines,
                    "upsampled_detection_raw_line_count": up_raw_lines,
                    "baseline_boxes_rendered_in_html": len(base_boxes),
                    "upsampled_boxes_rendered_in_html": len(up_boxes),
                    "metadata_baseline_detection_count": len(base_boxes),
                    "metadata_upsampled_detection_count": len(up_boxes),
                    **gt_counts,
                    "baseline_gt_approx_recall": baseline_gt_audit["gt_pred_approx_recall"],
                    "baseline_gt_approx_precision": baseline_gt_audit["gt_pred_approx_precision"],
                    "upsampled_gt_approx_recall": upsampled_gt_audit["gt_pred_approx_recall"],
                    "upsampled_gt_approx_precision": upsampled_gt_audit["gt_pred_approx_precision"],
                })
            method_links.append((line, line_links))
        index_sections.append((method, method_links))

    write_summary(summary_rows)
    write_availability(availability)
    write_candidate_detection_files(candidate_rows)
    write_ap_consistency_audit(ap_audit_rows)
    write_gt_prediction_matching_audit(gt_matching_rows)
    write_meeting_ready_examples(gt_matching_rows)
    write_index(index_sections, summary_rows)
    write_readme()


def fmt_score(value):
    return "" if value is None else f"{value:.3f}"


def write_object_dashboard(frame_dir, page_name, title, rec, base_crop_file, up_crop_file, overlay_crop_file, base_crop_points, up_crop_points):
    base = rec.get("base_box")
    up = rec.get("up_box")
    rows = [
        ("status", rec["status"]),
        ("baseline score", fmt_score(base.get("score")) if base else ""),
        ("upsampled score", fmt_score(up.get("score")) if up else ""),
        ("center shift", "" if rec.get("center_shift") is None else f"{rec['center_shift']:.3f} m"),
        ("baseline crop points", base_crop_points),
        ("upsampled crop points", up_crop_points),
    ]
    table = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    frame_dir.joinpath(page_name).write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:Arial;margin:20px;background:#f8fafc;color:#172033;line-height:1.45}"
        "iframe{width:100%;height:720px;border:1px solid #b4c1ce;background:white;margin:12px 0 22px}"
        "table{border-collapse:collapse;background:white;margin:12px 0 18px}td,th{border:1px solid #cbd5df;padding:7px;text-align:left}th{background:#e8eef4}a{color:#0645ad}</style>"
        "</head><body>"
        f"<h1>{html.escape(title)}</h1>"
        "<p><a href=\"combined_dashboard.html\">combined dashboard</a> | "
        f"<a href=\"{base_crop_file}\">baseline crop</a> | <a href=\"{up_crop_file}\">upsampled crop</a> | <a href=\"{overlay_crop_file}\">overlay crop</a></p>"
        f"<table>{table}</table>"
        "<h2>Baseline crop view</h2>"
        f"<iframe src=\"{base_crop_file}\"></iframe>"
        "<h2>Upsampled crop view</h2>"
        f"<iframe src=\"{up_crop_file}\"></iframe>"
        "<h2>Overlay crop view</h2>"
        f"<iframe src=\"{overlay_crop_file}\"></iframe>"
        "</body></html>"
    )


def detection_table_html(rows, kind):
    if not rows:
        return "<p>No detections.</p>"
    if kind == "matched":
        head = "<tr><th>pair</th><th>baseline</th><th>upsampled</th><th>class</th><th>baseline score</th><th>upsampled score</th><th>score change</th><th>center shift m</th></tr>"
        body = []
        for row in rows:
            body.append(
                "<tr>"
                f"<td>{row['key']}</td><td>{row['baseline']}</td><td>{row['upsampled']}</td><td>{html.escape(row['class'])}</td>"
                f"<td>{fmt_score(row['base_score'])}</td><td>{fmt_score(row['up_score'])}</td><td>{fmt_score(row['score_change'])}</td>"
                f"<td>{fmt_score(row['center_shift'])}</td>"
                "</tr>"
            )
        return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"
    head = "<tr><th>id</th><th>class</th><th>score</th><th>status</th></tr>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['key']}</td><td>{html.escape(row['class'])}</td><td>{fmt_score(row['score'])}</td><td>{html.escape(row['status'])}</td>"
            "</tr>"
        )
    return f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"


def write_dashboard(frame_dir, method, line, frame, metadata, object_links, base_detection_rows, up_detection_rows, matched_pair_rows):
    rows = []
    for obj in object_links:
        rows.append(
            "<tr>"
            f"<td><a href=\"{obj['file']}\">object_box_{obj['idx']:03d}</a></td>"
            f"<td>{html.escape(obj['status'])}</td><td>{html.escape(obj['class'])}</td>"
            f"<td>{fmt_score(obj['base_score'])}</td>"
            f"<td>{fmt_score(obj['up_score'])}</td>"
            f"<td>{fmt_score(obj['center_shift'])}</td>"
            f"<td>{obj['base_crop_points']}</td><td>{obj['upsampled_crop_points']}</td>"
            "</tr>"
        )
    stats = [
        ("full LiDAR points", metadata["baseline_point_count"]),
        ("camera-FOV points", metadata["baseline_fov_point_count"]),
        ("GT Car", metadata["gt_car_count"]),
        ("Easy / Mod / Hard", f"{metadata['gt_car_easy_count']} / {metadata['gt_car_moderate_count']} / {metadata['gt_car_hard_count']}"),
        ("baseline detections", metadata["baseline_detection_count"]),
        ("upsampled detections", metadata["upsampled_detection_count"]),
        ("baseline TP/FP/FN", f"{metadata['baseline_gt_prediction_audit']['gt_pred_matched_prediction_count']} / {metadata['baseline_gt_prediction_audit']['gt_pred_false_positive_count']} / {metadata['baseline_gt_prediction_audit']['gt_pred_missed_gt_count']}"),
        ("upsampled TP/FP/FN", f"{metadata['upsampled_gt_prediction_audit']['gt_pred_matched_prediction_count']} / {metadata['upsampled_gt_prediction_audit']['gt_pred_false_positive_count']} / {metadata['upsampled_gt_prediction_audit']['gt_pred_missed_gt_count']}"),
        ("baseline P/R", f"{fmt_score(metadata['baseline_gt_prediction_audit']['gt_pred_approx_precision'])} / {fmt_score(metadata['baseline_gt_prediction_audit']['gt_pred_approx_recall'])}"),
        ("upsampled P/R", f"{fmt_score(metadata['upsampled_gt_prediction_audit']['gt_pred_approx_precision'])} / {fmt_score(metadata['upsampled_gt_prediction_audit']['gt_pred_approx_recall'])}"),
        ("baseline mean IoU", fmt_score(metadata["baseline_gt_prediction_audit"]["gt_pred_mean_iou"])),
        ("upsampled mean IoU", fmt_score(metadata["upsampled_gt_prediction_audit"]["gt_pred_mean_iou"])),
        ("baseline TP/FP score", f"{fmt_score(metadata['baseline_gt_prediction_audit']['gt_pred_mean_tp_score'])} / {fmt_score(metadata['baseline_gt_prediction_audit']['gt_pred_mean_fp_score'])}"),
        ("upsampled TP/FP score", f"{fmt_score(metadata['upsampled_gt_prediction_audit']['gt_pred_mean_tp_score'])} / {fmt_score(metadata['upsampled_gt_prediction_audit']['gt_pred_mean_fp_score'])}"),
        ("avg score change", "" if metadata["average_score_change"] is None else f"{metadata['average_score_change']:.3f}"),
        ("avg center shift", "" if metadata["average_center_shift"] is None else f"{metadata['average_center_shift']:.3f} m"),
    ]
    stat_html = "".join(f"<div class=\"stat\"><b>{v}</b><span>{html.escape(k)}</span></div>" for k, v in stats)
    info_rows = [
        ("method", method),
        ("line", line),
        ("frame id", frame),
        ("baseline point cloud", metadata["paths"]["baseline_point_cloud"]),
        ("upsampled point cloud", metadata["paths"]["upsampled_point_cloud"]),
        ("baseline detection txt", metadata["detection_count_audit"]["baseline_detection_txt_path"]),
        ("upsampled detection txt", metadata["detection_count_audit"]["upsampled_detection_txt_path"]),
        ("GT label path", metadata.get("gt_label_path") or ""),
        ("calibration path", metadata["paths"]["calibration"]),
        ("AP consistency verdict", metadata["ap_consistency_verdict"]),
    ]
    info_html = "".join(f"<tr><th>{html.escape(k)}</th><td><code>{html.escape(str(v))}</code></td></tr>" for k, v in info_rows)
    warning_items = metadata.get("ap_consistency_warnings") or []
    warning_html = "<p>No AP/source consistency warnings for the selected paths.</p>"
    if warning_items:
        warning_html = "<ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in warning_items) + "</ul>"
    baseline_gt = metadata["baseline_gt_prediction_audit"]
    up_gt = metadata["upsampled_gt_prediction_audit"]
    gt_audit_html = (
        "<table><thead><tr><th>source</th><th>GT count</th><th>prediction count</th><th>TP</th><th>FP</th><th>FN</th><th>precision</th><th>recall</th><th>mean IoU</th><th>mean TP score</th><th>mean FP score</th></tr></thead><tbody>"
        f"<tr><td>baseline</td><td>{baseline_gt['gt_pred_gt_count']}</td><td>{baseline_gt['gt_pred_prediction_count']}</td><td>{baseline_gt['gt_pred_matched_prediction_count']}</td><td>{baseline_gt['gt_pred_false_positive_count']}</td><td>{baseline_gt['gt_pred_missed_gt_count']}</td><td>{fmt_score(baseline_gt['gt_pred_approx_precision'])}</td><td>{fmt_score(baseline_gt['gt_pred_approx_recall'])}</td><td>{fmt_score(baseline_gt['gt_pred_mean_iou'])}</td><td>{fmt_score(baseline_gt['gt_pred_mean_tp_score'])}</td><td>{fmt_score(baseline_gt['gt_pred_mean_fp_score'])}</td></tr>"
        f"<tr><td>upsampled</td><td>{up_gt['gt_pred_gt_count']}</td><td>{up_gt['gt_pred_prediction_count']}</td><td>{up_gt['gt_pred_matched_prediction_count']}</td><td>{up_gt['gt_pred_false_positive_count']}</td><td>{up_gt['gt_pred_missed_gt_count']}</td><td>{fmt_score(up_gt['gt_pred_approx_precision'])}</td><td>{fmt_score(up_gt['gt_pred_approx_recall'])}</td><td>{fmt_score(up_gt['gt_pred_mean_iou'])}</td><td>{fmt_score(up_gt['gt_pred_mean_tp_score'])}</td><td>{fmt_score(up_gt['gt_pred_mean_fp_score'])}</td></tr>"
        "</tbody></table>"
    )
    eval_links = metadata.get("evaluation_object_pages", [])
    eval_links_html = "<p>No TP/FP/FN object pages generated.</p>"
    if eval_links:
        eval_rows = "".join(
            f"<tr><td><a href=\"{html.escape(e['file'])}\">{html.escape(e['case_id'])}</a></td><td>{html.escape(e['source'])}</td><td>{html.escape(e['status'])}</td><td>{html.escape(str(e.get('gt_idx')))}</td><td>{html.escape(str(e.get('pred_idx')))}</td><td>{fmt_score(e.get('bev_iou'))}</td><td>{fmt_score(e.get('score'))}</td></tr>"
            for e in eval_links
        )
        eval_links_html = "<table><thead><tr><th>page</th><th>source</th><th>status</th><th>GT idx</th><th>pred idx</th><th>IoU</th><th>score</th></tr></thead><tbody>" + eval_rows + "</tbody></table>"
    frame_dir.joinpath("combined_dashboard.html").write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{html.escape(method)} {html.escape(line)} frame {frame}</title>"
        "<style>body{font-family:Arial;margin:20px;background:#f8fafc;color:#172033;line-height:1.45}"
        "iframe{width:100%;height:760px;border:1px solid #b4c1ce;background:white;margin:12px 0 22px}"
        ".stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}.stat{background:white;border:1px solid #cbd5df;padding:10px}.stat b{display:block;font-size:22px}.stat span{color:#526070}"
        "table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #cbd5df;padding:7px;text-align:left}th{background:#e8eef4}a{color:#0645ad}.note{background:#fff;border-left:4px solid #d62728;padding:12px;margin:12px 0}</style>"
        "</head><body>"
        f"<h1>{html.escape(method)} | {html.escape(line)} | frame {frame}</h1>"
        "<p><a href=\"../../../index.html\">index</a> | <a href=\"metadata.json\">metadata.json</a> | <a href=\"full_frame.html\">full_frame.html</a></p>"
        f"<h2>Basic info</h2><table><tbody>{info_html}</tbody></table>"
        f"<div class=\"stats\">{stat_html}</div>"
        f"<h2>Warnings</h2>{warning_html}"
        f"<div class=\"note\">{html.escape(metadata['evaluation_interpretation'])}</div>"
        f"<div class=\"note\">{html.escape(metadata['camera_fov_note'])}</div>"
        "<div class=\"note\">Full-frame rendering uses every valid KITTI-format line from each final PointRCNN txt. No score threshold, top-k filter, or matched-only filter is applied.</div>"
        f"<div class=\"note\">{html.escape(metadata['notes'])}</div>"
        "<h2>View links</h2><p>"
        "<a href=\"baseline_full_frame.html\">baseline full</a> | "
        "<a href=\"upsampled_full_frame.html\">upsampled full</a> | "
        "<a href=\"full_frame.html\">overlay full</a> | "
        "<a href=\"baseline_fov_frame.html\">baseline FOV</a> | "
        "<a href=\"upsampled_fov_frame.html\">upsampled FOV</a> | "
        "<a href=\"overlay_fov_frame.html\">overlay FOV</a></p>"
        "<h2>Main evaluation view: overlay camera-FOV frame</h2><iframe src=\"overlay_fov_frame.html\"></iframe>"
        "<h2>Baseline camera-FOV frame</h2><iframe src=\"baseline_fov_frame.html\"></iframe>"
        "<h2>Upsampled camera-FOV frame</h2><iframe src=\"upsampled_fov_frame.html\"></iframe>"
        "<h2>Full LiDAR context views</h2>"
        "<div class=\"note\">This view shows the full Velodyne frame for spatial context. KITTI labels and PointRCNN evaluation are camera-FOV based, so detections are expected mainly inside the front camera-visible sector, not across the full 360-degree LiDAR scene. Do not interpret the absence of boxes outside the camera FOV as detector failure; these regions are not the main KITTI object-detection evaluation area.</div>"
        "<h3>Overlay full LiDAR context</h3><iframe src=\"full_frame.html\"></iframe>"
        "<h3>Baseline-only full LiDAR context</h3><iframe src=\"baseline_full_frame.html\"></iframe>"
        "<h3>Upsampled-only full LiDAR context</h3><iframe src=\"upsampled_full_frame.html\"></iframe>"
        "<h2>Detection count audit</h2>"
        f"<table><tbody><tr><th>baseline txt</th><td><code>{html.escape(metadata['detection_count_audit']['baseline_detection_txt_path'])}</code></td></tr>"
        f"<tr><th>upsampled txt</th><td><code>{html.escape(metadata['detection_count_audit']['upsampled_detection_txt_path'])}</code></td></tr>"
        f"<tr><th>baseline raw txt lines</th><td>{metadata['detection_count_audit']['baseline_detection_raw_line_count']}</td></tr>"
        f"<tr><th>upsampled raw txt lines</th><td>{metadata['detection_count_audit']['upsampled_detection_raw_line_count']}</td></tr>"
        f"<tr><th>baseline boxes rendered</th><td>{metadata['detection_count_audit']['baseline_boxes_rendered_in_html']}</td></tr>"
        f"<tr><th>upsampled boxes rendered</th><td>{metadata['detection_count_audit']['upsampled_boxes_rendered_in_html']}</td></tr></tbody></table>"
        "<h2>GT-prediction matching audit</h2>"
        "<p>This is a center-distance visualization helper, not official KITTI AP.</p>"
        + gt_audit_html
        + "<h2>TP / FP / FN object pages</h2>"
        + eval_links_html
        + "<h2>Baseline detections</h2>"
        + detection_table_html(base_detection_rows, "single")
        + "<h2>Upsampled detections</h2>"
        + detection_table_html(up_detection_rows, "single")
        + "<h2>Matched pairs</h2>"
        + detection_table_html(matched_pair_rows, "matched")
        + "<h2>Detected object boxes</h2><table><thead><tr><th>page</th><th>status</th><th>class</th><th>baseline score</th><th>upsampled score</th><th>center shift m</th><th>baseline crop pts</th><th>upsampled crop pts</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></body></html>"
    )


def write_summary(rows):
    fields = [
        "method",
        "line",
        "frame_id",
        "baseline_point_count",
        "upsampled_point_count",
        "baseline_fov_point_count",
        "upsampled_fov_point_count",
        "baseline_detection_count",
        "upsampled_detection_count",
        "gt_box_count",
        "gt_car_count",
        "gt_car_easy_count",
        "gt_car_moderate_count",
        "gt_car_hard_count",
        "matched_detection_count",
        "missed_detection_count",
        "new_detection_count",
        "baseline_detection_txt_path",
        "upsampled_detection_txt_path",
        "baseline_detection_raw_line_count",
        "upsampled_detection_raw_line_count",
        "baseline_boxes_rendered_in_html",
        "upsampled_boxes_rendered_in_html",
        "ap_consistency_verdict",
        "ap_consistency_warnings",
        "baseline_gt_matched_prediction_count",
        "baseline_gt_missed_gt_count",
        "baseline_gt_false_positive_count",
        "baseline_gt_approx_recall",
        "baseline_gt_approx_precision",
        "baseline_gt_mean_iou",
        "baseline_gt_mean_tp_score",
        "baseline_gt_mean_fp_score",
        "upsampled_gt_matched_prediction_count",
        "upsampled_gt_missed_gt_count",
        "upsampled_gt_false_positive_count",
        "upsampled_gt_approx_recall",
        "upsampled_gt_approx_precision",
        "upsampled_gt_mean_iou",
        "upsampled_gt_mean_tp_score",
        "upsampled_gt_mean_fp_score",
        "evaluation_interpretation",
        "average_score_change",
        "average_center_shift",
        "notes",
    ]
    with (OUT / "summary_all.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k)) for k in fields})


def write_availability(rows):
    lines = ["# Input Availability Report", "", "Only combinations with all required inputs were visualized.", ""]
    for row in rows:
        lines.append(f"## {row['method']} / {row['line']} / frame_{row['frame']}")
        lines.append(f"- status: {row['status']}")
        if row["missing"]:
            lines.append(f"- missing: {', '.join(row['missing'])}")
        for k, v in row["paths"].items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")
    (OUT / "input_availability_report.md").write_text("\n".join(lines))


def write_candidate_detection_files(rows):
    fields = [
        "method",
        "line",
        "frame_id",
        "candidate_detection_txt_path",
        "raw_line_count",
        "selected_as_baseline",
        "selected_as_upsampled",
        "selection_reason",
    ]
    with (OUT / "candidate_detection_files.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k)) for k in fields})


def write_ap_consistency_audit(rows):
    fields = [
        "method",
        "line",
        "frame_id",
        "verdict",
        "warnings",
        "candidate_detection_file_count",
        "baseline_detection_txt_path",
        "upsampled_detection_txt_path",
        "baseline_path_verdict",
        "upsampled_path_verdict",
        "baseline_detection_raw_line_count",
        "upsampled_detection_raw_line_count",
        "baseline_boxes_rendered_in_html",
        "upsampled_boxes_rendered_in_html",
        "metadata_baseline_detection_count",
        "metadata_upsampled_detection_count",
        "gt_car_count",
        "gt_car_easy_count",
        "gt_car_moderate_count",
        "gt_car_hard_count",
        "baseline_gt_approx_recall",
        "baseline_gt_approx_precision",
        "upsampled_gt_approx_recall",
        "upsampled_gt_approx_precision",
    ]
    with (OUT / "ap_consistency_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k)) for k in fields})

    grouped = {}
    for row in rows:
        grouped.setdefault((row["method"], row["line"]), []).append(row)
    lines = [
        "# AP Consistency Audit",
        "",
        "This report checks whether the visualization renders every valid line from the selected PointRCNN final-result txt files and whether selected paths look AP/evaluation-like.",
        "",
        "Important: path provenance is a heuristic. `OK` means the selected paths look like evaluation/final-result outputs and counts match. `WARNING` means counts match but AP provenance is unclear or the path appears subset-like. `ERROR` means a selected detection file is missing or likely wrong. `SKIPPED` means non-detection prerequisites were unavailable.",
        "",
    ]
    verdict_order = {"ERROR": 0, "WARNING": 1, "SKIPPED": 2, "OK": 3}
    for (method, line), items in sorted(grouped.items()):
        worst = min((item["verdict"] for item in items), key=lambda v: verdict_order.get(v, 9))
        counts = {v: sum(1 for item in items if item["verdict"] == v) for v in ("OK", "WARNING", "ERROR", "SKIPPED")}
        lines.append(f"## {method} / {line}")
        lines.append(f"- conclusion: {worst}")
        lines.append(f"- counts: OK={counts['OK']}, WARNING={counts['WARNING']}, ERROR={counts['ERROR']}, SKIPPED={counts['SKIPPED']}")
        for item in items:
            warn = item.get("warnings") or "none"
            lines.append(
                f"- frame_{item['frame_id']}: {item['verdict']} | candidates={item['candidate_detection_file_count']} | "
                f"baseline lines/rendered={item['baseline_detection_raw_line_count']}/{item['baseline_boxes_rendered_in_html']} | "
                f"upsampled lines/rendered={item['upsampled_detection_raw_line_count']}/{item['upsampled_boxes_rendered_in_html']} | warnings: {warn}"
            )
        lines.append("")
    (OUT / "ap_consistency_audit.md").write_text("\n".join(lines))


def write_gt_prediction_matching_audit(rows):
    fields = [
        "method",
        "line",
        "frame_id",
        "gt_car_count",
        "baseline_prediction_count",
        "upsampled_prediction_count",
        "baseline_tp",
        "baseline_fp",
        "baseline_fn",
        "baseline_precision",
        "baseline_recall",
        "baseline_mean_iou",
        "baseline_mean_tp_score",
        "baseline_mean_fp_score",
        "upsampled_tp",
        "upsampled_fp",
        "upsampled_fn",
        "upsampled_precision",
        "upsampled_recall",
        "upsampled_mean_iou",
        "upsampled_mean_tp_score",
        "upsampled_mean_fp_score",
        "evaluation_interpretation",
        "dashboard",
    ]
    with (OUT / "gt_prediction_matching_audit.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k)) for k in fields})
    lines = [
        "# GT Prediction Matching Audit",
        "",
        "This is a visualization helper, not official KITTI AP. Predictions and Car GT are matched by same-class camera-center distance fallback; mean IoU is approximate axis-aligned BEV IoU.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- [{row['method']} / {row['line']} / frame_{row['frame_id']}]({row['dashboard']}): "
            f"GT={row['gt_car_count']}, baseline TP/FP/FN={row['baseline_tp']}/{row['baseline_fp']}/{row['baseline_fn']}, "
            f"upsampled TP/FP/FN={row['upsampled_tp']}/{row['upsampled_fp']}/{row['upsampled_fn']}, "
            f"baseline P/R={fmt_score(row['baseline_precision'])}/{fmt_score(row['baseline_recall'])}, "
            f"upsampled P/R={fmt_score(row['upsampled_precision'])}/{fmt_score(row['upsampled_recall'])}. "
            f"{row['evaluation_interpretation']}"
        )
    (OUT / "gt_prediction_matching_audit.md").write_text("\n".join(lines))


def pick_example(rows, predicate, sort_key=None):
    matches = [r for r in rows if predicate(r)]
    if not matches:
        return None
    if sort_key:
        matches.sort(key=sort_key, reverse=True)
    return matches[0]


def write_meeting_ready_examples(rows):
    categories = [
        ("Upsampling increases FN", lambda r: r["upsampled_fn"] > r["baseline_fn"], lambda r: r["upsampled_fn"] - r["baseline_fn"]),
        ("Upsampling increases FP", lambda r: r["upsampled_fp"] > r["baseline_fp"], lambda r: r["upsampled_fp"] - r["baseline_fp"]),
        ("Upsampling lowers IoU", lambda r: r["baseline_mean_iou"] is not None and r["upsampled_mean_iou"] is not None and r["upsampled_mean_iou"] < r["baseline_mean_iou"], lambda r: (r["baseline_mean_iou"] or 0) - (r["upsampled_mean_iou"] or 0)),
        ("Upsampling reduces confidence", lambda r: r["baseline_mean_tp_score"] is not None and r["upsampled_mean_tp_score"] is not None and r["upsampled_mean_tp_score"] < r["baseline_mean_tp_score"], lambda r: (r["baseline_mean_tp_score"] or 0) - (r["upsampled_mean_tp_score"] or 0)),
        ("Baseline detects but upsampled misses", lambda r: r["baseline_tp"] > r["upsampled_tp"], lambda r: r["baseline_tp"] - r["upsampled_tp"]),
        ("Upsampled prediction-only false positives", lambda r: r["upsampled_fp"] > 0, lambda r: r["upsampled_fp"]),
    ]
    lines = [
        "# Meeting Ready Examples",
        "",
        "These examples are selected from the approximate GT/prediction matching audit to explain why upsampling may not improve PointRCNN AP. Links open the evaluation/FOV dashboard first.",
        "",
    ]
    for title, predicate, sorter in categories:
        row = pick_example(rows, predicate, sorter)
        lines.append(f"## {title}")
        if not row:
            lines.append("- No matching representative case found in the generated frame set.")
        else:
            lines.append(
                f"- [{row['method']} / {row['line']} / frame_{row['frame_id']}]({row['dashboard']})"
            )
            lines.append(
                f"- GT Car={row['gt_car_count']}; baseline TP/FP/FN={row['baseline_tp']}/{row['baseline_fp']}/{row['baseline_fn']}; "
                f"upsampled TP/FP/FN={row['upsampled_tp']}/{row['upsampled_fp']}/{row['upsampled_fn']}; "
                f"mean IoU {fmt_score(row['baseline_mean_iou'])} -> {fmt_score(row['upsampled_mean_iou'])}; "
                f"mean TP score {fmt_score(row['baseline_mean_tp_score'])} -> {fmt_score(row['upsampled_mean_tp_score'])}."
            )
            lines.append(f"- Interpretation: {row['evaluation_interpretation']}")
        lines.append("")
    (OUT / "meeting_ready_examples.md").write_text("\n".join(lines))


def write_index(sections, rows):
    body = [
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>PointRCNN Detection Box Visualization V1</title>",
        "<style>body{font-family:Arial;line-height:1.45;margin:24px;background:#f8fafc;color:#172033}table{border-collapse:collapse;width:100%;margin-bottom:28px;background:white}td,th{border:1px solid #cbd5df;padding:7px;vertical-align:top}th{background:#e8eef4}a{color:#0645ad}code{background:#eef2f7;padding:1px 4px}</style>",
        "</head><body><h1>PointRCNN Detection Box Visualization V1</h1>",
        '<p><a href="meeting_ready_examples.md">meeting_ready_examples.md</a> | <a href="gt_prediction_matching_audit.md">gt_prediction_matching_audit.md</a> | <a href="gt_prediction_matching_audit.csv">gt_prediction_matching_audit.csv</a> | <a href="summary_all.csv">summary_all.csv</a> | <a href="ap_consistency_audit.csv">ap_consistency_audit.csv</a> | <a href="ap_consistency_audit.md">ap_consistency_audit.md</a> | <a href="candidate_detection_files.csv">candidate_detection_files.csv</a> | <a href="README.md">README.md</a> | <a href="input_availability_report.md">input_availability_report.md</a></p>',
        f"<p>{len(rows)} frame visualizations generated. Folders are grouped as <code>METHOD/line_x/frame_ID/</code>.</p>",
    ]
    for method, links in sections:
        body.append(f"<h2>{html.escape(method)}</h2><ul>")
        for line, frame_links in links:
            body.append(f"<li><b>{html.escape(line)}</b>: {len(frame_links)} visualized frames<ul>")
            for link in frame_links:
                body.append(f"<li><a href=\"{method}/{line}/{link}\">{html.escape(link.split('/')[0])}</a></li>")
            body.append("</ul></li>")
        body.append("</ul>")
    body.append("</body></html>")
    (OUT / "index.html").write_text("".join(body))


def write_readme():
    (OUT / "README.md").write_text(
        "# PointRCNN Detection Box Visualization V1\n\n"
        "This package visualizes selected KITTI frames from the earlier V6 object-crop anchors with PointRCNN detector outputs overlaid on baseline and upsampled point clouds.\n\n"
        "## Parsing\n\n"
        "- Point clouds are read from KITTI `.bin` files as float32 `(x, y, z, intensity)` records.\n"
        "- Detection and ground-truth labels are parsed as KITTI object text rows: class, 2D bbox, dimensions `(h, w, l)`, camera-location `(x, y, z)`, yaw `ry`, and optional score.\n"
        "- 3D box corners are created in KITTI camera coordinates, then transformed into LiDAR coordinates with `R0_rect` and `Tr_velo_to_cam` from each calibration file.\n"
        "- Full-frame pages embed every point read from the selected baseline and upsampled `.bin` inputs. No object-crop subset and no score threshold is used for full-frame detection rendering.\n"
        "- Detection source paths are selected by an explicit generator mapping. They are not silently auto-detected from arbitrary candidate folders.\n"
        "- Camera-FOV pages project LiDAR points through `R0_rect`, `Tr_velo_to_cam`, and `P2`, then keep points with positive camera depth and image coordinates inside the corresponding `image_2` PNG dimensions.\n"
        "- Crop pages use local crops around each matched, missed, or new detection record with a fixed margin.\n\n"
        "## Camera FOV\n\n"
        f"{FOV_NOTE}\n\n"
        "Full-frame pages include an approximate front-camera frustum boundary in LiDAR coordinates. The FOV-only pages (`baseline_fov_frame.html`, `upsampled_fov_frame.html`, `overlay_fov_frame.html`) show projected camera-visible LiDAR points with all detector boxes still rendered.\n\n"
        "## Matching\n\n"
        f"Baseline and upsampled predictions are matched by same class and camera-center distance <= {MATCH_CENTER_THRESHOLD_M:.1f} m, greedily nearest first. This is a center-distance fallback, not exact 3D IoU. It is used because the requested robust exact 3D IoU implementation is more complex than needed for these diagnostic overlays.\n\n"
        "## Detection Audit\n\n"
        "For each frame, `metadata.json`, `summary_all.csv`, and `ap_consistency_audit.csv` include the baseline and upsampled detection txt paths, non-empty raw txt line counts, and rendered box counts. Full-frame rendering applies no score threshold, no top-k filter, and no matched-only filter; every valid line in each selected final PointRCNN txt is rendered.\n\n"
        "`candidate_detection_files.csv` enumerates all same-frame `.txt` files found under `results/` and marks which files were selected by the explicit mapping. `ap_consistency_audit.md` summarizes whether those selected paths look AP/evaluation-like, subset-like, suspicious, missing, or unclear.\n\n"
        "## GT And Difficulty Counts\n\n"
        "KITTI `label_2` files are parsed for GT overlays. `DontCare` rows and invalid non-positive boxes are skipped. Car GT counts are reported for all/easy/moderate/hard using KITTI-style bbox-height, occlusion, and truncation criteria. A simple same-class center-distance GT/prediction audit is included as a visualization helper only; it is not official KITTI AP.\n\n"
        "## Colors\n\n"
        "- baseline/original points: light gray\n"
        "- upsampled points: blue in full frame, green in crops\n"
        "- baseline detection boxes: orange; baseline-only boxes are drawn with segmented dashed orange lines\n"
        "- upsampled detection boxes: red; upsampled-only boxes are drawn with segmented dashed red lines\n"
        "- KITTI ground-truth boxes, when labels are present: purple\n\n"
        "## Outputs\n\n"
        "Each visualized frame has `combined_dashboard.html`, full-scene pages, camera-FOV pages, one `object_box_XXX.html` dashboard per matched/missed/new detection record, per-object baseline/up/overlay crop pages, and `metadata.json`. The root `summary_all.csv` aggregates frame-level detection counts, FOV counts, GT counts, audit fields, and shifts.\n"
    )


if __name__ == "__main__":
    generate()
