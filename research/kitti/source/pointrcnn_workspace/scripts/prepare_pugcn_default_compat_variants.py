#!/usr/bin/env python3
"""Prepare PU-GCN smoke5 detector-input variants for default PointRCNN sampling."""

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_IDS = ("000001", "000002", "000004", "000005", "000006")
RPN_NUM_POINTS = 16384
FAR_DEPTH_THRESHOLD = 40.0
RECT_SCOPE = np.array([[-40.0, 40.0], [-1.0, 3.0], [0.0, 70.4]], dtype=np.float32)
METRIC_RANGE = {
    "x_min": 0.0,
    "x_max": 80.0,
    "y_min": -40.0,
    "y_max": 40.0,
    "z_min": -3.5,
    "z_max": 2.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-folder", type=Path, default=PROJECT_ROOT / "data/KITTI/object/training/velodyne_pugcn_filter_range_dedup_voxel_003_smoke5")
    parser.add_argument("--original-folder", type=Path, default=PROJECT_ROOT / "data/KITTI/object/training/velodyne")
    parser.add_argument("--output-data-root", type=Path, default=PROJECT_ROOT / "data/KITTI/object/training")
    parser.add_argument("--output-results-root", type=Path, default=PROJECT_ROOT / "results/pointrcnn_smoke5_pugcn_default_compatibility")
    parser.add_argument("--frame-ids", nargs="*", default=list(FRAME_IDS))
    parser.add_argument("--seed", type=int, default=20260508)
    return parser.parse_args()


def load_points(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"{path} does not reshape to N x 4")
    return raw.reshape(-1, 4)


def write_points(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points.astype(np.float32, copy=False).tofile(path)


def read_calib(calib_path: Path) -> Dict[str, np.ndarray]:
    lines = calib_path.read_text().splitlines()
    data: Dict[str, np.ndarray] = {}
    for line in lines:
        if not line.strip() or ":" not in line:
            continue
        key, values = line.split(":", 1)
        vals = np.array([float(v) for v in values.split()], dtype=np.float32)
        data[key] = vals
    return {
        "P2": data["P2"].reshape(3, 4),
        "R0_rect": data["R0_rect"].reshape(3, 3),
        "Tr_velo_to_cam": data["Tr_velo_to_cam"].reshape(3, 4),
    }


def cart_to_hom(points: np.ndarray) -> np.ndarray:
    return np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))


def lidar_to_rect(xyz: np.ndarray, calib: Dict[str, np.ndarray]) -> np.ndarray:
    xyz_hom = cart_to_hom(xyz)
    return xyz_hom @ (calib["Tr_velo_to_cam"].T @ calib["R0_rect"].T)


def rect_to_img(rect: np.ndarray, calib: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    rect_hom = cart_to_hom(rect)
    pts_2d_hom = rect_hom @ calib["P2"].T
    img = (pts_2d_hom[:, 0:2].T / rect_hom[:, 2]).T
    depth = pts_2d_hom[:, 2] - calib["P2"].T[3, 2]
    return img, depth


def image_shape(frame_id: str) -> Tuple[int, int]:
    path = PROJECT_ROOT / "data/KITTI/object/training/image_2" / f"{frame_id}.png"
    with Image.open(path) as img:
        width, height = img.size
    return height, width


def detector_masks(points: np.ndarray, frame_id: str) -> Dict[str, np.ndarray]:
    calib = read_calib(PROJECT_ROOT / "data/KITTI/object/training/calib" / f"{frame_id}.txt")
    rect = lidar_to_rect(points[:, :3], calib)
    img, depth = rect_to_img(rect, calib)
    h, w = image_shape(frame_id)
    valid = (
        (img[:, 0] >= 0)
        & (img[:, 0] < w)
        & (img[:, 1] >= 0)
        & (img[:, 1] < h)
        & (depth >= 0)
        & (rect[:, 0] >= RECT_SCOPE[0, 0])
        & (rect[:, 0] <= RECT_SCOPE[0, 1])
        & (rect[:, 1] >= RECT_SCOPE[1, 0])
        & (rect[:, 1] <= RECT_SCOPE[1, 1])
        & (rect[:, 2] >= RECT_SCOPE[2, 0])
        & (rect[:, 2] <= RECT_SCOPE[2, 1])
    )
    far = valid & (rect[:, 2] >= FAR_DEPTH_THRESHOLD)
    near = valid & (rect[:, 2] < FAR_DEPTH_THRESHOLD)
    return {"rect": rect, "valid": valid, "near": near, "far": far}


def metric_out_of_range(points: np.ndarray) -> int:
    xyz = points[:, :3]
    in_range = (
        (xyz[:, 0] >= METRIC_RANGE["x_min"])
        & (xyz[:, 0] <= METRIC_RANGE["x_max"])
        & (xyz[:, 1] >= METRIC_RANGE["y_min"])
        & (xyz[:, 1] <= METRIC_RANGE["y_max"])
        & (xyz[:, 2] >= METRIC_RANGE["z_min"])
        & (xyz[:, 2] <= METRIC_RANGE["z_max"])
        & np.isfinite(xyz).all(axis=1)
    )
    return int((~in_range).sum())


def near_duplicate_ratio(points: np.ndarray, voxel: float = 0.03) -> float:
    if len(points) == 0:
        return 0.0
    keys = np.floor(points[:, :3] / voxel).astype(np.int64)
    unique = np.unique(keys, axis=0).shape[0]
    return float((len(points) - unique) / len(points))


def exact_overlap_mask(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = {tuple(row) for row in reference.astype(np.float32, copy=False)}
    return np.array([tuple(row) in ref for row in points.astype(np.float32, copy=False)], dtype=bool)


def choose_random(rng: np.random.Generator, idxs: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or len(idxs) == 0:
        return np.empty((0,), dtype=np.int64)
    n = min(n, len(idxs))
    return rng.choice(idxs, size=n, replace=False)


def random_cap(points: np.ndarray, original: np.ndarray, quota: int, rng: np.random.Generator) -> np.ndarray:
    original_mask = exact_overlap_mask(points, original)
    original_idxs = np.where(original_mask)[0]
    added_idxs = np.where(~original_mask)[0]
    if quota >= len(points):
        return points.copy()
    if len(original_idxs) >= quota:
        keep = choose_random(rng, original_idxs, quota)
    else:
        fill = choose_random(rng, added_idxs, quota - len(original_idxs))
        keep = np.concatenate((original_idxs, fill))
    rng.shuffle(keep)
    return points[keep].copy()


def balanced_cap(points: np.ndarray, frame_id: str, original: np.ndarray, quota: int, rng: np.random.Generator) -> np.ndarray:
    if quota >= len(points):
        return points.copy()
    masks = detector_masks(points, frame_id)
    rect = masks["rect"]
    original_mask = exact_overlap_mask(points, original)
    valid = masks["valid"]
    far = masks["far"]
    near = masks["near"]
    invalid = ~valid

    groups: List[np.ndarray] = []
    for orig_flag in (True, False):
        base = original_mask if orig_flag else ~original_mask
        for depth_mask in (near, far, invalid):
            idxs = np.where(base & depth_mask)[0]
            if len(idxs) == 0:
                continue
            if depth_mask is invalid:
                bins = np.zeros(len(idxs), dtype=np.int64)
            else:
                bins = np.clip((rect[idxs, 2] // 10).astype(np.int64), 0, 7)
            for b in np.unique(bins):
                groups.append(idxs[bins == b])

    selected: List[int] = []
    remaining_groups = [g.copy() for g in groups if len(g) > 0]
    while remaining_groups and len(selected) < quota:
        next_groups = []
        for group in remaining_groups:
            if len(selected) >= quota:
                break
            pick_pos = int(rng.integers(0, len(group)))
            selected.append(int(group[pick_pos]))
            group = np.delete(group, pick_pos)
            if len(group) > 0:
                next_groups.append(group)
        remaining_groups = next_groups

    if len(selected) < quota:
        rest = np.setdiff1d(np.arange(len(points)), np.array(selected, dtype=np.int64), assume_unique=False)
        fill = choose_random(rng, rest, quota - len(selected))
        selected.extend([int(x) for x in fill])
    keep = np.array(selected, dtype=np.int64)
    rng.shuffle(keep)
    return points[keep].copy()


def summarize_variant(method: str, frame_id: str, points: np.ndarray, original: np.ndarray) -> Dict[str, object]:
    masks = detector_masks(points, frame_id)
    valid_count = int(masks["valid"].sum())
    near_count = int(masks["near"].sum())
    far_count = int(masks["far"].sum())
    invalid_count = int((~np.isfinite(points).all(axis=1)).sum())
    original_overlap = int(exact_overlap_mask(points, original).sum())
    return {
        "variant": method,
        "frame_id": frame_id,
        "point_count": int(len(points)),
        "original_exact_overlap_points": original_overlap,
        "generated_or_nonoverlap_points": int(len(points) - original_overlap),
        "detector_valid_points": valid_count,
        "detector_near_points_depth_lt_40": near_count,
        "detector_far_points_depth_ge_40": far_count,
        "default_sampler_far_exceeds_16384": far_count > RPN_NUM_POINTS,
        "default_sampler_near_request": RPN_NUM_POINTS - far_count if valid_count > RPN_NUM_POINTS else "N/A",
        "nan_points": int(np.isnan(points).any(axis=1).sum()),
        "inf_points": int(np.isinf(points).any(axis=1).sum()),
        "invalid_points": invalid_count,
        "metric_out_of_range_points": metric_out_of_range(points),
        "intensity_min": float(points[:, 3].min()) if len(points) else "N/A",
        "intensity_max": float(points[:, 3].max()) if len(points) else "N/A",
        "near_duplicate_ratio_voxel_003": near_duplicate_ratio(points, 0.03),
        "format_ok": invalid_count == 0 and points.ndim == 2 and points.shape[1] == 4,
    }


def write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()
    variants = {
        "pugcn_cap_original_count": ("original_count", None, random_cap),
        "pugcn_cap_120k": ("fixed", 120000, random_cap),
        "pugcn_cap_100k": ("fixed", 100000, random_cap),
        "pugcn_cap_80k": ("fixed", 80000, random_cap),
        "pugcn_range_balanced_120k": ("fixed", 120000, balanced_cap),
    }
    args.output_results_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    diagnosis_rows: List[Dict[str, object]] = []

    for frame_id in args.frame_ids:
        source = load_points(args.source_folder / f"{frame_id}.bin")
        original = load_points(args.original_folder / f"{frame_id}.bin")
        diagnosis_rows.append(summarize_variant("source_filter_range_dedup_voxel_003", frame_id, source, original))
        for variant, (quota_type, fixed_quota, sampler) in variants.items():
            quota = len(original) if quota_type == "original_count" else int(fixed_quota)
            rng = np.random.default_rng(args.seed + int(frame_id) + quota)
            capped = sampler(source, frame_id, original, quota, rng) if sampler is balanced_cap else sampler(source, original, quota, rng)
            out_folder = args.output_data_root / variant
            write_points(out_folder / f"{frame_id}.bin", capped)
            row = summarize_variant(variant, frame_id, capped, original)
            row["quota"] = quota
            row["sampling_seed"] = args.seed + int(frame_id) + quota
            row["output_path"] = str((out_folder / f"{frame_id}.bin").resolve())
            rows.append(row)

    write_csv(args.output_results_root / "pugcn_default_failure_diagnosis.csv", diagnosis_rows)
    write_csv(args.output_results_root / "pugcn_variant_precheck.csv", rows)
    manifest = args.output_results_root / "variant_generation_manifest.md"
    manifest.write_text(
        "# PU-GCN Default-Compatible Variant Generation\n\n"
        f"Source folder: `{args.source_folder}`\n\n"
        f"Original reference folder: `{args.original_folder}`\n\n"
        f"Seed base: `{args.seed}`\n\n"
        "Variants:\n\n"
        "- `pugcn_cap_original_count`: deterministic random cap to the same frame's original KITTI point count, preserving exact original-overlap rows first when present.\n"
        "- `pugcn_cap_120k`: deterministic random cap to at most 120000 points, preserving exact original-overlap rows first when present.\n"
        "- `pugcn_cap_100k`: deterministic random cap to at most 100000 points, preserving exact original-overlap rows first when present.\n"
        "- `pugcn_cap_80k`: deterministic random cap to at most 80000 points, preserving exact original-overlap rows first when present.\n"
        "- `pugcn_range_balanced_120k`: deterministic range/depth-balanced cap to 120000 points using original/non-overlap and near/far/depth-bin groups.\n\n"
        "No PointRCNN sampler code was modified. No replacement sampling was used.\n"
    )
    print(args.output_results_root / "pugcn_default_failure_diagnosis.csv")
    print(args.output_results_root / "pugcn_variant_precheck.csv")


if __name__ == "__main__":
    main()
