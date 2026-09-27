#!/usr/bin/env python3
"""Open a selected dual-detector case in a native Open3D window.

Examples
--------
Show the full E1 overlay (observed gray, generated blue):

    python scripts/view_dual_detector_evidence_open3d.py \
      --frame 005625 --line B --method pdans --detector centerpoint \
      --state overlay

Show the actual detector-effective upsampled input and crop around GT 0:

    python scripts/view_dual_detector_evidence_open3d.py \
      --frame 005625 --line B --method pdans --detector centerpoint \
      --state upsampled --effective --gt-index 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analyze_dual_detector_frame_transitions as audit  # noqa: E402
import generate_dual_detector_three_frame_evidence as evidence  # noqa: E402
import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402


DEFAULT_ROOT = audit.DEFAULT_OUTPUT
BOX_EDGES = np.asarray(evidence.BOX_EDGES, dtype=np.int32)
BOX_COLORS = {
    key: tuple(int(color[idx : idx + 2], 16) / 255.0 for idx in (1, 3, 5))
    for key, color in evidence.COLORS.items()
}


def point_cloud(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    cloud.paint_uniform_color(color)
    return cloud


def line_box(box: dict, color: tuple[float, float, float]) -> o3d.geometry.LineSet:
    geometry = o3d.geometry.LineSet()
    geometry.points = o3d.utility.Vector3dVector(
        np.asarray(box["corners_lidar"], dtype=np.float64)
    )
    geometry.lines = o3d.utility.Vector2iVector(BOX_EDGES)
    geometry.colors = o3d.utility.Vector3dVector(
        np.repeat(np.asarray(color, dtype=np.float64)[None, :], len(BOX_EDGES), axis=0)
    )
    return geometry


def exact_row_mask(reference: np.ndarray, points: np.ndarray) -> np.ndarray:
    return evidence.exact_row_mask(reference, points)


def crop(points: np.ndarray, box: dict, margin: float) -> np.ndarray:
    return evidence.crop_points(points, box, margin)


def load_manifest(args) -> tuple[Path, dict]:
    method = args.method.lower().replace("-", "_")
    path = (
        args.root
        / "frames"
        / args.frame
        / f"line_{args.line.lower()}"
        / method
        / args.detector
        / "case_manifest.json"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def state_points(manifest: dict, state: str, effective: bool) -> list[tuple[np.ndarray, tuple]]:
    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    observed = prep.read_bin(paths["observed"])
    e1 = prep.read_bin(paths["upsampled_e1"])
    if not effective:
        if state == "baseline":
            return [(observed, BOX_COLORS["baseline"])]
        if state == "upsampled":
            return [(e1, BOX_COLORS["upsampled"])]
        mask = exact_row_mask(observed, e1)
        return [
            (e1[mask], BOX_COLORS["observed"]),
            (e1[~mask], BOX_COLORS["generated"]),
        ]
    detector = manifest["protocol"]["detector"]
    if detector == "pointrcnn":
        if state == "baseline":
            return [(prep.read_bin(paths["baseline_e2"]), BOX_COLORS["baseline"])]
        if state == "upsampled":
            return [(prep.read_bin(paths["upsampled_e2"]), BOX_COLORS["upsampled"])]
        baseline = prep.read_bin(paths["baseline_e2"])
        upsampled = prep.read_bin(paths["upsampled_e2"])
        return [
            (baseline, BOX_COLORS["baseline"]),
            (upsampled, BOX_COLORS["upsampled"]),
        ]
    if state == "baseline":
        retained, _ = evidence.centerpoint_effective_points(observed, manifest["frame_id"])
        return [(retained, BOX_COLORS["baseline"])]
    if state == "upsampled":
        retained, _ = evidence.centerpoint_effective_points(e1, manifest["frame_id"])
        return [(retained, BOX_COLORS["upsampled"])]
    baseline, _ = evidence.centerpoint_effective_points(observed, manifest["frame_id"])
    upsampled, _ = evidence.centerpoint_effective_points(e1, manifest["frame_id"])
    return [
        (baseline, BOX_COLORS["baseline"]),
        (upsampled, BOX_COLORS["upsampled"]),
    ]


def box_geometries(manifest: dict, state: str) -> list[o3d.geometry.LineSet]:
    geometries = []
    transition = {
        int(key): value for key, value in manifest["transition_by_gt"].items()
    }
    audit_key = "baseline_audit" if state == "baseline" else "upsampled_audit"
    audit_result = manifest[audit_key]
    prediction_key = (
        "baseline_prediction_status"
        if state == "baseline"
        else "upsampled_prediction_status"
    )
    eval_gt_by_id = {
        int(box["id"]): idx for idx, box in enumerate(audit_result["gt"])
    }
    for gt_box in manifest["gt_all"]:
        eval_idx = eval_gt_by_id.get(int(gt_box["id"]))
        status = transition.get(eval_idx, "ignored_gt")
        geometries.append(line_box(gt_box, BOX_COLORS[status]))
    for box, status in zip(audit_result["pred"], manifest[prediction_key]):
        geometries.append(line_box(box, BOX_COLORS[status]))
    return geometries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--line", choices=("A", "B"), required=True)
    parser.add_argument(
        "--method",
        choices=("pdans", "pu_gcn", "pu_edgeformer", "pu_net"),
        required=True,
    )
    parser.add_argument(
        "--detector", choices=("pointrcnn", "centerpoint"), required=True
    )
    parser.add_argument(
        "--state", choices=("baseline", "upsampled", "overlay"), default="overlay"
    )
    parser.add_argument(
        "--effective",
        action="store_true",
        help="Show detector-effective input instead of full observed/E1 clouds.",
    )
    parser.add_argument(
        "--gt-index",
        type=int,
        help="Crop to one detector-evaluated GT index.",
    )
    parser.add_argument("--margin", type=float, default=2.0)
    parser.add_argument("--point-size", type=float, default=1.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate the case without opening an Open3D window.",
    )
    args = parser.parse_args()
    manifest_path, manifest = load_manifest(args)
    state_for_boxes = "baseline" if args.state == "baseline" else "upsampled"
    point_layers = state_points(manifest, args.state, args.effective)
    focus_box = None
    if args.gt_index is not None:
        eval_gt = manifest["baseline_audit"]["gt"]
        if args.gt_index < 0 or args.gt_index >= len(eval_gt):
            raise IndexError(
                f"GT index {args.gt_index} outside 0..{len(eval_gt) - 1}"
            )
        focus_box = eval_gt[args.gt_index]
        point_layers = [
            (crop(points, focus_box, args.margin), color)
            for points, color in point_layers
        ]
    geometries: list[o3d.geometry.Geometry] = [
        point_cloud(points, color) for points, color in point_layers if len(points)
    ]
    geometries.extend(box_geometries(manifest, state_for_boxes))
    if focus_box is not None:
        geometries.append(line_box(focus_box, BOX_COLORS["gt"]))
    title = (
        f"{manifest['frame_id']} Line {manifest['protocol']['line']} "
        f"{manifest['protocol']['method']} {args.detector} {args.state}"
    )
    print(f"manifest={manifest_path}")
    print("colors: gray=observed/baseline, blue=upsampled/generated, dashed semantics are in PNG")
    print(
        "boxes: green=maintained, orange=localization degraded, purple=confidence degraded, "
        "red=lost/false positive, cyan=recovered, yellow/gray=GT context"
    )
    print(
        "point_layers="
        + ",".join(str(len(points)) for points, _ in point_layers)
        + f"; geometries={len(geometries)}"
    )
    if args.dry_run:
        return 0
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name=title, width=1500, height=900)
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    options = visualizer.get_render_option()
    options.point_size = args.point_size
    options.background_color = np.asarray([0.04, 0.045, 0.055])
    options.line_width = 2.0
    visualizer.run()
    visualizer.destroy_window()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
