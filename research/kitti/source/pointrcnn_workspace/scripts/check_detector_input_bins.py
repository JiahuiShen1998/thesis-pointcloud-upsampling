#!/usr/bin/env python3
"""Sanity-check KITTI-style detector input .bin files across methods."""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "pugcn_detector_precheck"
DEFAULT_FRAME_IDS = ("000001", "000002", "000004", "000005", "000006")
DEFAULT_KITTI_RANGE = {
    "x_min": 0.0,
    "x_max": 80.0,
    "y_min": -40.0,
    "y_max": 40.0,
    "z_min": -3.5,
    "z_max": 2.0,
}
DEFAULT_INTENSITY_RANGE = {
    "min": 0.0,
    "max": 1.5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check KITTI-style detector input bins.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--frame-ids", nargs="*", default=list(DEFAULT_FRAME_IDS))
    parser.add_argument(
        "--folder",
        action="append",
        nargs=2,
        metavar=("METHOD", "PATH"),
        help="Folder to inspect, repeated as METHOD PATH.",
    )
    return parser.parse_args()


def default_folders() -> List[Tuple[str, Path]]:
    return [
        ("Original KITTI", PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_original_val"),
        ("EAR", PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_ear_val_smoke5"),
        ("PU-Net", PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_punet_x2_fullframe_smoke5"),
        ("PU-GCN", PROJECT_ROOT / "data" / "KITTI" / "object" / "training" / "velodyne_pugcn_filter_range_dedup_voxel_003_smoke5"),
    ]


def format_value(value: object) -> str:
    if isinstance(value, float):
        return ("%.6f" % value).rstrip("0").rstrip(".")
    return str(value)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_points(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError("File does not reshape to N x 4")
    return raw.reshape(-1, 4)


def xyz_bounds(points: np.ndarray) -> Dict[str, object]:
    if len(points) == 0:
        return {
            "x_min": "N/A",
            "x_max": "N/A",
            "y_min": "N/A",
            "y_max": "N/A",
            "z_min": "N/A",
            "z_max": "N/A",
        }
    xyz = points[:, :3]
    return {
        "x_min": float(np.min(xyz[:, 0])),
        "x_max": float(np.max(xyz[:, 0])),
        "y_min": float(np.min(xyz[:, 1])),
        "y_max": float(np.max(xyz[:, 1])),
        "z_min": float(np.min(xyz[:, 2])),
        "z_max": float(np.max(xyz[:, 2])),
    }


def intensity_bounds(points: np.ndarray) -> Dict[str, object]:
    if len(points) == 0:
        return {"intensity_min": "N/A", "intensity_max": "N/A", "intensity_mean": "N/A"}
    intensity = points[:, 3]
    return {
        "intensity_min": float(np.min(intensity)),
        "intensity_max": float(np.max(intensity)),
        "intensity_mean": float(np.mean(intensity)),
    }


def invalid_counts(points: np.ndarray) -> Dict[str, int]:
    if len(points) == 0:
        return {"nan_points": 0, "inf_points": 0, "invalid_points": 0}
    nan_points = int(np.isnan(points).any(axis=1).sum())
    inf_points = int(np.isinf(points).any(axis=1).sum())
    invalid_points = int((~np.isfinite(points).all(axis=1)).sum())
    return {
        "nan_points": nan_points,
        "inf_points": inf_points,
        "invalid_points": invalid_points,
    }


def out_of_range_count(points: np.ndarray) -> int:
    if len(points) == 0:
        return 0
    xyz = points[:, :3]
    mask = (
        np.isfinite(xyz).all(axis=1)
        & (xyz[:, 0] >= DEFAULT_KITTI_RANGE["x_min"])
        & (xyz[:, 0] <= DEFAULT_KITTI_RANGE["x_max"])
        & (xyz[:, 1] >= DEFAULT_KITTI_RANGE["y_min"])
        & (xyz[:, 1] <= DEFAULT_KITTI_RANGE["y_max"])
        & (xyz[:, 2] >= DEFAULT_KITTI_RANGE["z_min"])
        & (xyz[:, 2] <= DEFAULT_KITTI_RANGE["z_max"])
    )
    return int((~mask).sum())


def point_rcnn_ok(points: np.ndarray, file_size_bytes: int, invalid_points: int, intensity_min: object, intensity_max: object) -> Tuple[bool, str]:
    reasons: List[str] = []
    if file_size_bytes % 16 != 0:
        reasons.append("file_size_not_multiple_of_16")
    if points.ndim != 2 or points.shape[1] != 4:
        reasons.append("shape_not_nx4")
    if invalid_points != 0:
        reasons.append("invalid_points")
    if isinstance(intensity_min, float) and intensity_min < DEFAULT_INTENSITY_RANGE["min"]:
        reasons.append("intensity_below_0")
    if isinstance(intensity_max, float) and intensity_max > DEFAULT_INTENSITY_RANGE["max"]:
        reasons.append("intensity_above_1.5")
    return (len(reasons) == 0, ",".join(reasons) if reasons else "ok")


def analyze_file(method_name: str, folder: Path, frame_id: str) -> Dict[str, object]:
    path = folder / ("%s.bin" % frame_id)
    row: Dict[str, object] = {
        "method_name": method_name,
        "frame_id": frame_id,
        "file_path": str(path.resolve()),
        "file_exists": path.exists(),
    }
    if not path.exists():
        row.update(
            {
                "point_count": "MISSING",
                "dtype": "MISSING",
                "shape": "MISSING",
                "file_size_bytes": 0,
                "x_min": "N/A",
                "x_max": "N/A",
                "y_min": "N/A",
                "y_max": "N/A",
                "z_min": "N/A",
                "z_max": "N/A",
                "intensity_min": "N/A",
                "intensity_max": "N/A",
                "intensity_mean": "N/A",
                "nan_points": "N/A",
                "inf_points": "N/A",
                "invalid_points": "N/A",
                "out_of_range_points": "N/A",
                "matches_pointrcnn_expected_format": "no",
                "format_reason": "missing_file",
            }
        )
        return row

    file_size = path.stat().st_size
    try:
        points = load_points(path)
        bounds = xyz_bounds(points)
        intensities = intensity_bounds(points)
        invalid = invalid_counts(points)
        out_of_range = out_of_range_count(points)
        ok, reason = point_rcnn_ok(points, file_size, invalid["invalid_points"], intensities["intensity_min"], intensities["intensity_max"])
        row.update(
            {
                "point_count": int(points.shape[0]),
                "dtype": str(points.dtype),
                "shape": "%d x %d" % (points.shape[0], points.shape[1]),
                "file_size_bytes": int(file_size),
                "out_of_range_points": out_of_range,
                "matches_pointrcnn_expected_format": "yes" if ok else "no",
                "format_reason": reason,
            }
        )
        row.update(bounds)
        row.update(intensities)
        row.update(invalid)
    except Exception as exc:
        row.update(
            {
                "point_count": "ERROR",
                "dtype": "ERROR",
                "shape": "ERROR",
                "file_size_bytes": int(file_size),
                "x_min": "N/A",
                "x_max": "N/A",
                "y_min": "N/A",
                "y_max": "N/A",
                "z_min": "N/A",
                "z_max": "N/A",
                "intensity_min": "N/A",
                "intensity_max": "N/A",
                "intensity_mean": "N/A",
                "nan_points": "N/A",
                "inf_points": "N/A",
                "invalid_points": "N/A",
                "out_of_range_points": "N/A",
                "matches_pointrcnn_expected_format": "no",
                "format_reason": str(exc),
            }
        )
    return row


def summarize_method(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    valid_rows = [row for row in rows if row.get("file_exists") and row.get("matches_pointrcnn_expected_format") in ("yes", "no")]
    present_rows = [row for row in rows if row.get("file_exists")]
    mismatches = [row["frame_id"] for row in rows if row.get("matches_pointrcnn_expected_format") == "no"]
    summary: Dict[str, object] = {
        "frame_count": len(rows),
        "present_files": len(present_rows),
        "valid_format_files": sum(1 for row in rows if row.get("matches_pointrcnn_expected_format") == "yes"),
        "mismatch_frames": ", ".join(mismatches) if mismatches else "none",
    }
    if valid_rows:
        point_counts = [float(row["point_count"]) for row in valid_rows if isinstance(row["point_count"], (int, float))]
        out_of_range = [float(row["out_of_range_points"]) for row in valid_rows if isinstance(row["out_of_range_points"], (int, float))]
        summary["avg_point_count"] = float(sum(point_counts) / len(point_counts))
        summary["total_out_of_range_points"] = int(sum(out_of_range))
    else:
        summary["avg_point_count"] = "N/A"
        summary["total_out_of_range_points"] = "N/A"
    return summary


def write_markdown(path: Path, grouped_rows: Sequence[Tuple[str, List[Dict[str, object]]]]) -> None:
    lines = [
        "# Detector Input Bin Precheck",
        "",
        "This report checks whether the compared `.bin` files are valid KITTI-style point clouds for PointRCNN-style loading.",
        "",
    ]
    for method_name, rows in grouped_rows:
        summary = summarize_method(rows)
        lines.extend(
            [
                "## %s" % method_name,
                "",
                "- frame_count: %s" % format_value(summary["frame_count"]),
                "- present_files: %s" % format_value(summary["present_files"]),
                "- valid_format_files: %s" % format_value(summary["valid_format_files"]),
                "- avg_point_count: %s" % format_value(summary["avg_point_count"]),
                "- total_out_of_range_points: %s" % format_value(summary["total_out_of_range_points"]),
                "- mismatch_frames: %s" % summary["mismatch_frames"],
                "",
                "| Frame | Points | Dtype | Shape | Intensity min/max | Invalid | Out-of-range | Format ok | Note |",
                "|---|---:|---|---|---|---:|---:|---|---|",
            ]
        )
        for row in rows:
            lines.append(
                "| %s | %s | %s | %s | %s / %s | %s | %s | %s | %s |"
                % (
                    row["frame_id"],
                    format_value(row["point_count"]),
                    row["dtype"],
                    row["shape"],
                    format_value(row["intensity_min"]),
                    format_value(row["intensity_max"]),
                    format_value(row["invalid_points"]),
                    format_value(row["out_of_range_points"]),
                    row["matches_pointrcnn_expected_format"],
                    row["format_reason"],
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    folder_specs = [(name, Path(path)) for name, path in args.folder] if args.folder else default_folders()
    rows: List[Dict[str, object]] = []
    grouped: List[Tuple[str, List[Dict[str, object]]]] = []
    for method_name, folder in folder_specs:
        method_rows = [analyze_file(method_name, folder, frame_id) for frame_id in args.frame_ids]
        grouped.append((method_name, method_rows))
        rows.extend(method_rows)
    csv_path = args.output_root / "pugcn_detector_input_precheck.csv"
    md_path = args.output_root / "pugcn_detector_input_precheck.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, grouped)
    print("Wrote detector input precheck CSV:", csv_path)
    print("Wrote detector input precheck Markdown:", md_path)


if __name__ == "__main__":
    main()
