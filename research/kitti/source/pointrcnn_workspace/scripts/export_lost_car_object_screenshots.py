#!/usr/bin/env python3
"""Export categorized screenshots of Cars lost after x4 upsampling.

The exporter is read-only with respect to the frozen detector experiments.  It
uses the full-validation transition audit and the already-generated per-case
manifests.  For every method, experiment line, and detector it selects two Car
GT objects that are strict true positives before upsampling and strict false
negatives afterwards, then renders original-reference, before, and after crops.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import analyze_dual_detector_frame_transitions as audit  # noqa: E402
import generate_dual_detector_three_frame_evidence as evidence  # noqa: E402
import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402


SOURCE_ROOT = REPO / "results/dual_detector_three_frame_root_cause_20260730"
DEFAULT_OUTPUT = REPO / "results/lost_car_detected_to_missed_screenshots_20260818"
DEFAULT_FRAME = "005625"
LINES = ("A", "B")
METHODS = audit.METHODS
DETECTORS = ("pointrcnn", "centerpoint")
DETECTOR_LABEL = {"pointrcnn": "PointRCNN", "centerpoint": "CenterPoint"}
CAR_IOU_THRESHOLD = 0.70
BOX_EDGES = evidence.BOX_EDGES

COLORS = {
    "original": "#334155",
    "before": "#4B5563",
    "after": "#2563EB",
    "gt": "#E6A700",
    "detected": "#008A5A",
    "unmatched": "#E36A00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--objects-per-case", type=int, default=2)
    return parser.parse_args()


def read_bin(path: Path) -> np.ndarray:
    points = prep.read_bin(path)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"bad point cloud: {path}")
    return points


def manifest_path(root: Path, frame: str, line: str, method: str, detector: str) -> Path:
    return (
        root
        / "frames"
        / frame
        / f"line_{line.lower()}"
        / method
        / detector
        / "case_manifest.json"
    )


def load_manifest(root: Path, frame: str, line: str, method: str, detector: str) -> dict:
    path = manifest_path(root, frame, line, method, detector)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def box_center(box: dict) -> np.ndarray:
    return np.asarray(box["corners_lidar"], dtype=np.float64).mean(axis=0)


def object_basis(box: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return center, long/short/up row basis, and box dimensions."""
    corners = np.asarray(box["corners_lidar"], dtype=np.float64)
    center = corners.mean(axis=0)
    horizontal = [corners[1] - corners[0], corners[3] - corners[0]]
    horizontal.sort(key=lambda value: float(np.linalg.norm(value)), reverse=True)
    vertical = corners[4] - corners[0]
    raw = np.asarray([horizontal[0], horizontal[1], vertical], dtype=np.float64)
    dimensions = np.linalg.norm(raw, axis=1)
    basis = raw / np.maximum(dimensions[:, None], 1e-9)
    # Make the up axis visually consistent even if a source uses reversed corners.
    if basis[2, 2] < 0:
        basis[2] *= -1
    return center, basis, dimensions


def to_local_xyz(values: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    xyz = np.asarray(values, dtype=np.float64)
    return (xyz[..., :3] - center) @ basis.T


def local_box(box: dict, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return to_local_xyz(np.asarray(box["corners_lidar"], dtype=np.float64), center, basis)


def crop_local(
    points: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    dimensions: np.ndarray,
    margin: tuple[float, float, float] = (1.2, 1.2, 0.75),
) -> tuple[np.ndarray, np.ndarray]:
    local = to_local_xyz(points, center, basis)
    half = dimensions / 2.0 + np.asarray(margin, dtype=np.float64)
    mask = np.all(np.abs(local) <= half, axis=1)
    return points[mask], local[mask]


def inside_count(
    points: np.ndarray,
    center: np.ndarray,
    basis: np.ndarray,
    dimensions: np.ndarray,
) -> int:
    local = to_local_xyz(points, center, basis)
    half = dimensions / 2.0 + np.asarray([0.08, 0.08, 0.08])
    return int(np.all(np.abs(local) <= half, axis=1).sum())


def detector_effective_points(manifest: dict) -> tuple[np.ndarray, np.ndarray]:
    paths = {key: Path(value) for key, value in manifest["paths"].items()}
    if manifest["protocol"]["detector"] == "pointrcnn":
        return read_bin(paths["baseline_e2"]), read_bin(paths["upsampled_e2"])
    observed = read_bin(paths["observed"])
    upsampled = read_bin(paths["upsampled_e1"])
    baseline_effective, _ = evidence.centerpoint_effective_points(
        observed, manifest["frame_id"]
    )
    upsampled_effective, _ = evidence.centerpoint_effective_points(
        upsampled, manifest["frame_id"]
    )
    return baseline_effective, upsampled_effective


def candidate_rows(manifest: dict, original: np.ndarray) -> list[dict]:
    transition = {int(key): value for key, value in manifest["transition_by_gt"].items()}
    baseline_matches = {
        int(key): value for key, value in manifest["baseline_audit"]["match_by_gt"].items()
    }
    upsampled_matches = {
        int(key): value for key, value in manifest["upsampled_audit"]["match_by_gt"].items()
    }
    nearest = {
        int(key): value for key, value in manifest["upsampled_audit"]["nearest_by_gt"].items()
    }
    candidates = []
    for gt_index, gt_box in enumerate(manifest["baseline_audit"]["gt"]):
        if gt_box["class"] != "Car":
            continue
        if transition.get(gt_index) != "lost_after_upsampling":
            continue
        if gt_index not in baseline_matches or gt_index in upsampled_matches:
            continue
        center, basis, dimensions = object_basis(gt_box)
        original_inside = inside_count(original, center, basis, dimensions)
        original_crop, _ = crop_local(original, center, basis, dimensions)
        baseline_match = baseline_matches[gt_index]
        after_best = nearest.get(gt_index, {})
        after_best_iou = float(after_best.get("iou3d", 0.0) or 0.0)
        distance = float(center[0])
        truncation = float(gt_box.get("truncation", 0.0) or 0.0)
        occlusion = int(gt_box.get("occlusion", 0) or 0)
        bbox_height = float(gt_box.get("bbox_height", 0.0) or 0.0)
        # Favor visible, well-supported Cars and clear post-upsampling misses.
        quality = (
            24.0 * math.log1p(original_inside)
            + 0.24 * min(bbox_height, 180.0)
            + 28.0 * float(baseline_match["iou3d"])
            - 22.0 * after_best_iou
            - 55.0 * truncation
            - 7.0 * occlusion
            - 1.5 * max(distance - 38.0, 0.0)
        )
        candidates.append(
            {
                "gt_index": gt_index,
                "gt_box": gt_box,
                "center": center,
                "basis": basis,
                "dimensions": dimensions,
                "baseline_match": baseline_match,
                "after_best": after_best,
                "after_best_iou": after_best_iou,
                "original_inside_points": original_inside,
                "original_crop_points": int(len(original_crop)),
                "distance_m": distance,
                "truncation": truncation,
                "occlusion": occlusion,
                "bbox_height": bbox_height,
                "quality_score": quality,
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row["quality_score"]),
            -int(row["original_inside_points"]),
            int(row["gt_index"]),
        )
    )
    return candidates


def draw_box_3d_local(
    ax,
    box: dict,
    center: np.ndarray,
    basis: np.ndarray,
    color: str,
    linewidth: float,
    linestyle: str,
) -> None:
    corners = local_box(box, center, basis)
    for start, end in BOX_EDGES:
        ax.plot(
            corners[[start, end], 0],
            corners[[start, end], 1],
            corners[[start, end], 2],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=5,
        )


def draw_box_bev_local(
    ax,
    box: dict,
    center: np.ndarray,
    basis: np.ndarray,
    color: str,
    linewidth: float,
    linestyle: str,
) -> None:
    corners = local_box(box, center, basis)
    ring = np.vstack([corners[:4, :2], corners[0, :2]])
    ax.plot(
        ring[:, 0],
        ring[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=5,
    )


def nearby_after_box(manifest: dict, candidate: dict) -> dict | None:
    nearest = candidate["after_best"]
    if not nearest:
        return None
    pred_index = int(nearest["pred_idx"])
    predictions = manifest["upsampled_audit"]["pred"]
    if pred_index < 0 or pred_index >= len(predictions):
        return None
    box = predictions[pred_index]
    center_distance = float(np.linalg.norm(box_center(box) - candidate["center"]))
    if center_distance > 3.5 or candidate["after_best_iou"] <= 0.01:
        return None
    return box


def panel_points(
    ax3d,
    axbev,
    local: np.ndarray,
    color: str,
    title: str,
    dimensions: np.ndarray,
) -> None:
    if len(local):
        z = local[:, 2]
        sizes = np.full(len(local), 5.0 if len(local) < 1200 else 3.2)
        ax3d.scatter(
            local[:, 0],
            local[:, 1],
            z,
            c=color,
            s=sizes,
            alpha=0.78,
            depthshade=False,
            linewidths=0,
            rasterized=True,
        )
        axbev.scatter(
            local[:, 0],
            local[:, 1],
            c=color,
            s=sizes,
            alpha=0.76,
            linewidths=0,
            rasterized=True,
        )
    x_half, y_half, z_half = dimensions / 2.0
    ax3d.set_xlim(-x_half - 1.2, x_half + 1.2)
    ax3d.set_ylim(-y_half - 1.2, y_half + 1.2)
    ax3d.set_zlim(-z_half - 0.75, z_half + 0.75)
    ax3d.view_init(elev=21, azim=-58)
    ax3d.set_box_aspect((dimensions[0] + 2.4, dimensions[1] + 2.4, dimensions[2] + 1.5))
    ax3d.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax3d.set_xlabel("car length axis (m)", fontsize=7)
    ax3d.set_ylabel("car width axis (m)", fontsize=7)
    ax3d.set_zlabel("height (m)", fontsize=7)
    ax3d.tick_params(labelsize=6, pad=0)
    ax3d.grid(True, linewidth=0.25, alpha=0.25)
    axbev.set_xlim(-x_half - 1.2, x_half + 1.2)
    axbev.set_ylim(-y_half - 1.2, y_half + 1.2)
    axbev.set_aspect("equal", adjustable="box")
    axbev.set_xlabel("car length axis (m)", fontsize=7)
    axbev.set_ylabel("car width axis (m)", fontsize=7)
    axbev.tick_params(labelsize=6)
    axbev.grid(True, linewidth=0.3, alpha=0.25)


def render_object(
    output: Path,
    manifest: dict,
    candidate: dict,
    original: np.ndarray,
    baseline: np.ndarray,
    upsampled: np.ndarray,
    line: str,
    method: str,
    detector: str,
    slot: int,
) -> dict:
    box = candidate["gt_box"]
    center = candidate["center"]
    basis = candidate["basis"]
    dimensions = candidate["dimensions"]
    original_crop, original_local = crop_local(original, center, basis, dimensions)
    baseline_crop, baseline_local = crop_local(baseline, center, basis, dimensions)
    upsampled_crop, upsampled_local = crop_local(upsampled, center, basis, dimensions)

    baseline_pred_index = int(candidate["baseline_match"]["pred_idx"])
    baseline_pred = manifest["baseline_audit"]["pred"][baseline_pred_index]
    after_box = nearby_after_box(manifest, candidate)
    baseline_iou = float(candidate["baseline_match"]["iou3d"])
    baseline_score = float(candidate["baseline_match"]["score"] or 0.0)
    after_best_iou = float(candidate["after_best_iou"])

    fig = plt.figure(figsize=(17.2, 9.6), facecolor="#F8FAFC")
    axes3d = [fig.add_subplot(2, 3, index + 1, projection="3d") for index in range(3)]
    axesbev = [fig.add_subplot(2, 3, index + 4) for index in range(3)]
    panel_points(
        axes3d[0],
        axesbev[0],
        original_local,
        COLORS["original"],
        f"Original baseline reference\ncar shape: {candidate['original_inside_points']:,} points inside GT",
        dimensions,
    )
    panel_points(
        axes3d[1],
        axesbev[1],
        baseline_local,
        COLORS["before"],
        f"Before upsampling — DETECTED\n3D IoU={baseline_iou:.3f}, score={baseline_score:.3f}",
        dimensions,
    )
    panel_points(
        axes3d[2],
        axesbev[2],
        upsampled_local,
        COLORS["after"],
        f"After upsampling — MISSED\nbest final-box 3D IoU={after_best_iou:.3f} < {CAR_IOU_THRESHOLD:.2f}",
        dimensions,
    )

    for ax in axes3d:
        draw_box_3d_local(ax, box, center, basis, COLORS["gt"], 2.0, "--")
    for ax in axesbev:
        draw_box_bev_local(ax, box, center, basis, COLORS["gt"], 2.0, "--")
    draw_box_3d_local(axes3d[1], baseline_pred, center, basis, COLORS["detected"], 2.5, "-")
    draw_box_bev_local(axesbev[1], baseline_pred, center, basis, COLORS["detected"], 2.5, "-")
    if after_box is not None:
        draw_box_3d_local(axes3d[2], after_box, center, basis, COLORS["unmatched"], 2.0, ":")
        draw_box_bev_local(axesbev[2], after_box, center, basis, COLORS["unmatched"], 2.0, ":")
        axesbev[2].text(
            0.02,
            0.98,
            "orange dotted = nearest unmatched final box",
            transform=axesbev[2].transAxes,
            va="top",
            ha="left",
            fontsize=7,
            color=COLORS["unmatched"],
        )
    else:
        axesbev[2].text(
            0.5,
            0.5,
            "NO MATCHED\nDETECTION BOX",
            transform=axesbev[2].transAxes,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color="#C1121F",
            bbox={"facecolor": "white", "edgecolor": "#C1121F", "alpha": 0.86},
        )

    legend = [
        Line2D([0], [0], color=COLORS["gt"], linestyle="--", linewidth=2.2, label="KITTI Car GT box"),
        Line2D([0], [0], color=COLORS["detected"], linestyle="-", linewidth=2.6, label="valid baseline final detection"),
        Line2D([0], [0], color=COLORS["unmatched"], linestyle=":", linewidth=2.2, label="nearest unmatched after-box (when nearby)"),
    ]
    method_label = audit.METHOD_LABEL[method]
    detector_label = DETECTOR_LABEL[detector]
    fig.suptitle(
        f"Frame {manifest['frame_id']} | Line {line} | {method_label} | {detector_label} | object {slot}/2\n"
        f"Car GT index {candidate['gt_index']} at {candidate['distance_m']:.1f} m: detected before x4 upsampling, missed after x4 upsampling",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.007))
    fig.text(
        0.01,
        0.012,
        "Top: object-aligned 3D. Bottom: object-aligned BEV. A miss means no same-class final box reaches KITTI Car 3D IoU 0.70.",
        fontsize=8,
        color="#475569",
    )
    # Fixed margins keep long panel titles and complete 3D boxes inside the
    # canvas.  Matplotlib's tight bbox can otherwise crop the left 3D title for
    # long, narrow Cars because the projected axes temporarily extend left.
    fig.subplots_adjust(
        left=0.045,
        right=0.985,
        top=0.86,
        bottom=0.085,
        wspace=0.18,
        hspace=0.20,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=155, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {
        "line": line,
        "method": method_label,
        "detector": detector_label,
        "frame_id": manifest["frame_id"],
        "slot": slot,
        "gt_index": candidate["gt_index"],
        "kitti_label_id": box.get("id", ""),
        "class": "Car",
        "transition": "detected_before_to_missed_after",
        "distance_m": round(candidate["distance_m"], 4),
        "truncation": candidate["truncation"],
        "occlusion": candidate["occlusion"],
        "bbox_height_px": round(candidate["bbox_height"], 3),
        "original_inside_gt_points": candidate["original_inside_points"],
        "original_crop_points": len(original_crop),
        "baseline_effective_crop_points": len(baseline_crop),
        "upsampled_effective_crop_points": len(upsampled_crop),
        "baseline_prediction_index": baseline_pred_index,
        "baseline_score": baseline_score,
        "baseline_3d_iou": baseline_iou,
        "after_best_final_box_3d_iou": after_best_iou,
        "after_nearby_unmatched_box_drawn": after_box is not None,
        "selection_quality_score": round(candidate["quality_score"], 4),
        "screenshot": str(output),
    }


def make_pair_sheet(paths: list[Path], output: Path, title: str) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    target_width = 1900
    resized = []
    for image in images:
        height = round(image.height * target_width / image.width)
        resized.append(image.resize((target_width, height), Image.Resampling.LANCZOS))
    header_height = 72
    canvas = Image.new(
        "RGB",
        (target_width, header_height + sum(image.height for image in resized)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((24, 18), title, fill="#0F172A", font=font)
    y = header_height
    for image in resized:
        canvas.paste(image, (0, y))
        y += image.height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)
    for image in images:
        image.close()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output: Path, frame: str, rows: list[dict], pair_rows: list[dict]) -> None:
    text = f"""#  upsampling After missed detection  Car  Screenshot classification package

 This directory uses uniform  exact-x4  frozen results of the formal experiments. The fixed frames are  `{frame}`,  Because frame also covers  4  A methodology.  × 2  An experimental line  × 2  individual  detector,  And each combination has at least two strict ones.  Car `detected -> missed`  Objective.

##  Contents

- `line_A/<method>/<detector>/`: original baseline  with  original+x4.
- `line_B/<method>/<detector>/`: downsampled baseline  with  downsampled+x4;  The left column of each chart is still used  original scan  Show the car.
-  Each combination consists of two  object  Figure and one.  `00_two_objects.png`  Summary chart.
- `selection_index.csv`: 32  Targeting, point count, IoU,  Path.
- `pair_sheet_index.csv`: 16  Two. -object  Summarizes the path of the map.

##  Legends and decisions

-  Yellow dot line: KITTI Car GT 3D  box.
-  Green line: upsampling Formerly  GT  Matches the final test box.
-  Orange Point Line: If upsampling is followed by a space close to but does not reach the final frame of threshold.
- Car  threshold is  oriented 3D IoU >= 0.70; `missed`  Indicates that no after upsampling has reached the threshold final frame.
-  Top As  object-aligned 3D,  Downline As  object-aligned BEV;  All boxes are shown in full.

 Co-Export  {len(rows)}  individual  object  Screenshots and  {len(pair_rows)}  A summary of the combination.

##  Data boundary

 Only read complete screenshots  validation  Logo result, do not rerun upsampling or  detector. PointRCNN  Actual Figure  E2 16,384  Point input; CenterPoint  Replay the figure  FOV/range,  Every  voxel  Front  5  Point and  40k voxel  Post-ceiling effective point. PU-Net  The figure reflects the current frozen output, but the output is missing  normalization/inverse-transform wrapper  The issue of known validity should be retained in the paper.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.objects_per_case != 2:
        raise ValueError("this categorized package is defined for exactly two objects per case")
    output = args.output.resolve()
    rows: list[dict] = []
    pair_rows: list[dict] = []
    for line in LINES:
        for method in METHODS:
            for detector in DETECTORS:
                manifest = load_manifest(args.source_root, args.frame, line, method, detector)
                paths = {key: Path(value) for key, value in manifest["paths"].items()}
                original = read_bin(paths["original_reference"])
                baseline, upsampled = detector_effective_points(manifest)
                candidates = candidate_rows(manifest, original)
                if len(candidates) < args.objects_per_case:
                    raise RuntimeError(
                        f"{line}/{method}/{detector}/{args.frame}: only {len(candidates)} lost Cars"
                    )
                case_dir = output / f"line_{line}" / audit.METHOD_LABEL[method] / DETECTOR_LABEL[detector]
                object_paths = []
                case_rows = []
                for slot, candidate in enumerate(candidates[: args.objects_per_case], start=1):
                    object_path = case_dir / f"{slot:02d}_gt_{candidate['gt_index']:03d}_Car.png"
                    row = render_object(
                        object_path,
                        manifest,
                        candidate,
                        original,
                        baseline,
                        upsampled,
                        line,
                        method,
                        detector,
                        slot,
                    )
                    row["screenshot"] = str(object_path.relative_to(output))
                    rows.append(row)
                    case_rows.append(row)
                    object_paths.append(object_path)
                pair_path = case_dir / "00_two_objects.png"
                make_pair_sheet(
                    object_paths,
                    pair_path,
                    f"Frame {args.frame} | Line {line} | {audit.METHOD_LABEL[method]} | {DETECTOR_LABEL[detector]} | two lost Cars",
                )
                pair_row = {
                    "line": line,
                    "method": audit.METHOD_LABEL[method],
                    "detector": DETECTOR_LABEL[detector],
                    "frame_id": args.frame,
                    "object_1_gt_index": case_rows[0]["gt_index"],
                    "object_2_gt_index": case_rows[1]["gt_index"],
                    "pair_screenshot": str(pair_path.relative_to(output)),
                }
                pair_rows.append(pair_row)
                (case_dir / "metadata.json").write_text(
                    json.dumps(
                        {
                            "case": pair_row,
                            "objects": case_rows,
                            "source_manifest": str(
                                manifest_path(args.source_root, args.frame, line, method, detector)
                            ),
                            "matching_threshold": {"class": "Car", "oriented_3d_iou": 0.70},
                        },
                        indent=2,
                        ensure_ascii=False,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                print(
                    f"rendered Line {line} {audit.METHOD_LABEL[method]} {DETECTOR_LABEL[detector]}: "
                    + ", ".join(f"GT {row['gt_index']}" for row in case_rows),
                    flush=True,
                )
    write_csv(output / "selection_index.csv", rows)
    write_csv(output / "pair_sheet_index.csv", pair_rows)
    write_readme(output, args.frame, rows, pair_rows)
    validation = {
        "status": "PASS",
        "frame_id": args.frame,
        "methods": len(METHODS),
        "lines": len(LINES),
        "detectors": len(DETECTORS),
        "cases": len(pair_rows),
        "objects": len(rows),
        "all_before_iou_ge_0p70": all(float(row["baseline_3d_iou"]) >= 0.70 for row in rows),
        "all_after_best_iou_lt_0p70": all(float(row["after_best_final_box_3d_iou"]) < 0.70 for row in rows),
        "all_original_inside_gt_points_positive": all(int(row["original_inside_gt_points"]) > 0 for row in rows),
    }
    (output / "validation_report.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
