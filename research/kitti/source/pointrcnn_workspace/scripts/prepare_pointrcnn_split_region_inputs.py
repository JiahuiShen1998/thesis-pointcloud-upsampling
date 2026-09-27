#!/usr/bin/env python3
"""Prepare label-free, lossless spatial regions for PointRCNN evaluation.

The same per-frame region boundaries are derived jointly from the baseline and
all compared methods.  A region is recursively split until every compared
cloud contains at most ``--max-points`` PointRCNN-valid points in that region.
The split uses point coordinates only; labels, boxes, detector outputs and AP
values are never read.

Every valid point belongs to exactly one disjoint core region.  Optionally, an
inference region can include a fixed spatial halo around its core.  Core bounds
remain disjoint and own predictions, while the overlapping halo supplies the
context needed to avoid cutting an object at a core boundary.  A density-valley
split policy can choose label-free boundaries from the observed LiDAR cloud.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402


CAUSAL = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
SURFACE = CAUSAL / "surface_candidate256_v1"
DEFAULT_WORKSPACE = REPO / "results/pointrcnn_split_region_surface256_v1_20260804"
DEFAULT_SPLIT = REPO / "data/KITTI/ImageSets/patch_causal_pilot256.txt"
DEFAULT_SOURCES = {
    "original_baseline": REPO / "data/KITTI/object/training/velodyne_original_val",
    "surface_c32_pu_gcn": SURFACE
    / "pu_gcn_surface_cover_exact_pr1_c32/line_a_original_x4_up/final_bin",
    "surface_c32_pu_edgeformer": SURFACE
    / "pu_edgeformer_surface_cover_exact_pr1_c32/line_a_original_x4_up/final_bin",
    "surface_c32_pdans": SURFACE
    / "pdans_surface_cover_exact_pr1_c32/line_a_original_x4_up/final_bin",
    "surface_c32_pu_net_fixed": SURFACE
    / "pu_net_fixed_surface_cover_exact_pr1_c32/line_a_original_x4_up/final_bin",
}
# PointRCNN's range check includes x == 40.0.  Use a one-micrometre-open
# sentinel so the disjoint half-open region convention still covers it.
X_BOUNDS = (-40.0, 40.000001)
Z_BOUNDS = (0.0, 70.40001)


@dataclass(frozen=True)
class Region:
    x_min: float
    x_max: float
    z_min: float
    z_max: float
    depth: int


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name or "/" in name:
        raise argparse.ArgumentTypeError(f"invalid source name: {name!r}")
    return name, Path(raw_path).expanduser().resolve()


def read_frames(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def in_region(xz: np.ndarray, region: Region, margin: float = 0.0) -> np.ndarray:
    return (
        (xz[:, 0] >= max(X_BOUNDS[0], region.x_min - margin))
        & (xz[:, 0] < min(X_BOUNDS[1], region.x_max + margin))
        & (xz[:, 1] >= max(Z_BOUNDS[0], region.z_min - margin))
        & (xz[:, 1] < min(Z_BOUNDS[1], region.z_max + margin))
    )


def counts_for(
    region: Region, coordinates: dict[str, np.ndarray], margin: float = 0.0
) -> dict[str, int]:
    return {
        name: int(in_region(xz, region, margin).sum()) for name, xz in coordinates.items()
    }


def candidate_split(values: np.ndarray, lower: float, upper: float) -> float | None:
    if values.size < 2:
        return None
    split = float(np.median(values))
    tolerance = max(1e-5, (upper - lower) * 1e-7)
    if not lower + tolerance < split < upper - tolerance:
        unique = np.unique(values)
        if unique.size < 2:
            return None
        middle = unique.size // 2
        split = float((unique[middle - 1] + unique[middle]) * 0.5)
    if not lower < split < upper:
        return None
    return split


def candidate_density_valley_split(
    values: np.ndarray,
    lower: float,
    upper: float,
    guard_m: float,
) -> float | None:
    """Choose a balanced cut with few observed points near the boundary."""
    values = values[np.isfinite(values)]
    if values.size < 2:
        return None
    quantiles = np.linspace(0.20, 0.80, 49)
    candidates = np.unique(np.quantile(values, quantiles))
    tolerance = max(1e-5, (upper - lower) * 1e-7)
    candidates = candidates[
        (candidates > lower + tolerance) & (candidates < upper - tolerance)
    ]
    if not candidates.size:
        return candidate_split(values, lower, upper)

    best: tuple[float, float] | None = None
    count = float(values.size)
    for candidate in candidates:
        left_fraction = float(np.count_nonzero(values < candidate)) / count
        if not 0.15 <= left_fraction <= 0.85:
            continue
        boundary_fraction = float(
            np.count_nonzero(np.abs(values - candidate) <= guard_m)
        ) / count
        # Boundary density dominates; the balance term prevents pathological
        # tiny cores when several valleys have similarly low occupancy.
        score = boundary_fraction + 0.10 * abs(left_fraction - 0.5)
        item = (score, float(candidate))
        if best is None or item < best:
            best = item
    return best[1] if best is not None else candidate_split(values, lower, upper)


def split_region(
    region: Region,
    coordinates: dict[str, np.ndarray],
    max_points: int,
    max_depth: int,
    halo_m: float = 0.0,
    split_policy: str = "median",
    guide_name: str = "original_baseline",
    boundary_guard_m: float = 1.0,
) -> list[Region]:
    counts = counts_for(region, coordinates, halo_m)
    densest_name, densest_count = max(counts.items(), key=lambda item: item[1])
    if densest_count <= max_points:
        return [region]
    if region.depth >= max_depth:
        raise RuntimeError(
            f"max recursion depth reached with {densest_count} points in {region}"
        )

    dense_xz = coordinates[densest_name]
    dense_xz = dense_xz[in_region(dense_xz, region)]
    guide_xz = coordinates.get(guide_name, dense_xz)
    guide_xz = guide_xz[in_region(guide_xz, region)]
    if guide_xz.shape[0] < 2:
        guide_xz = dense_xz
    spans = np.ptp(dense_xz, axis=0) if dense_xz.shape[0] else np.zeros(2)
    normalized = np.asarray(
        [spans[0] / (X_BOUNDS[1] - X_BOUNDS[0]), spans[1] / (Z_BOUNDS[1] - Z_BOUNDS[0])]
    )
    axis_order = list(np.argsort(-normalized))

    children: tuple[Region, Region] | None = None
    for axis in axis_order:
        if split_policy == "density_valley":
            split = candidate_density_valley_split(
                guide_xz[:, axis],
                region.x_min if axis == 0 else region.z_min,
                region.x_max if axis == 0 else region.z_max,
                boundary_guard_m,
            )
        else:
            split = candidate_split(
                dense_xz[:, axis],
                region.x_min if axis == 0 else region.z_min,
                region.x_max if axis == 0 else region.z_max,
            )
        if axis == 0:
            if split is None:
                continue
            children = (
                Region(region.x_min, split, region.z_min, region.z_max, region.depth + 1),
                Region(split, region.x_max, region.z_min, region.z_max, region.depth + 1),
            )
        else:
            if split is None:
                continue
            children = (
                Region(region.x_min, region.x_max, region.z_min, split, region.depth + 1),
                Region(region.x_min, region.x_max, split, region.z_max, region.depth + 1),
            )
        child_core_counts = [
            counts_for(child, coordinates)[densest_name] for child in children
        ]
        child_max_counts = [
            max(counts_for(child, coordinates, halo_m).values()) for child in children
        ]
        if min(child_core_counts) > 0 and max(child_max_counts) < densest_count:
            break
        children = None

    if children is None:
        raise RuntimeError(
            f"cannot split coincident points for {densest_name}: count={densest_count}, region={region}"
        )

    leaves: list[Region] = []
    for child in children:
        leaves.extend(
            split_region(
                child,
                coordinates,
                max_points,
                max_depth,
                halo_m,
                split_policy,
                guide_name,
                boundary_guard_m,
            )
        )
    return leaves


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_core_with_balanced_halo(
    xz: np.ndarray,
    region: Region,
    halo_m: float,
    max_points: int,
    voxel_m: float,
) -> tuple[np.ndarray, int, int]:
    """Keep every core point and deterministically fill capacity from the halo."""
    core_mask = in_region(xz, region)
    expanded_mask = in_region(xz, region, halo_m)
    core_indices = np.flatnonzero(core_mask)
    halo_indices = np.flatnonzero(expanded_mask & ~core_mask)
    if core_indices.size > max_points:
        raise RuntimeError(
            f"core capacity violation: core={core_indices.size} max={max_points} region={region}"
        )
    budget = max_points - core_indices.size
    if halo_indices.size <= budget:
        return np.concatenate((core_indices, halo_indices)), int(halo_indices.size), 0
    if budget <= 0:
        return core_indices, 0, int(halo_indices.size)

    halo_xz = xz[halo_indices]
    dx = np.maximum(
        np.maximum(region.x_min - halo_xz[:, 0], halo_xz[:, 0] - region.x_max), 0.0
    )
    dz = np.maximum(
        np.maximum(region.z_min - halo_xz[:, 1], halo_xz[:, 1] - region.z_max), 0.0
    )
    distance = np.hypot(dx, dz)
    order = np.argsort(distance, kind="stable")
    voxel = np.floor(halo_xz / voxel_m).astype(np.int64)
    chosen_local: list[int] = []
    chosen_set: set[int] = set()
    seen_voxels: set[tuple[int, int]] = set()
    for local in order:
        key = (int(voxel[local, 0]), int(voxel[local, 1]))
        if key in seen_voxels:
            continue
        seen_voxels.add(key)
        chosen_local.append(int(local))
        chosen_set.add(int(local))
        if len(chosen_local) == budget:
            break
    if len(chosen_local) < budget:
        for local in order:
            local = int(local)
            if local in chosen_set:
                continue
            chosen_local.append(local)
            if len(chosen_local) == budget:
                break
    chosen_halo = halo_indices[np.asarray(chosen_local, dtype=np.int64)]
    selected = np.concatenate((core_indices, chosen_halo))
    return selected, int(chosen_halo.size), int(halo_indices.size - chosen_halo.size)


def read_fixed_regions(path: Path) -> dict[str, list[Region]]:
    output: dict[str, list[Region]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output.setdefault(row["frame_id"], []).append(
                Region(
                    float(row["x_min_rect_m"]),
                    float(row["x_max_rect_m"]),
                    float(row["z_min_rect_m"]),
                    float(row["z_max_rect_m"]),
                    int(row["tree_depth"]),
                )
            )
    for regions in output.values():
        regions.sort(key=lambda item: (item.z_min, item.x_min, item.z_max, item.x_max))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--source", action="append", type=parse_source)
    parser.add_argument("--max-points", type=int, default=16384)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--halo-m", type=float, default=0.0)
    parser.add_argument(
        "--halo-cap-policy",
        choices=("strict", "balanced_sample"),
        default="strict",
        help="Strict keeps the entire halo; balanced_sample always keeps the core and samples halo context",
    )
    parser.add_argument("--halo-voxel-m", type=float, default=0.20)
    parser.add_argument(
        "--split-policy", choices=("median", "density_valley"), default="median"
    )
    parser.add_argument("--guide-source", default="original_baseline")
    parser.add_argument("--boundary-guard-m", type=float, default=1.0)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--fixed-regions",
        type=Path,
        help="Reuse an existing regions.csv instead of deriving new boundaries",
    )
    parser.add_argument(
        "--shared-observed-prefix",
        type=Path,
        help="Optimize sources that all begin with the same preserved observed cloud",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    split_file = args.split_file.resolve()
    frames = read_frames(split_file)
    sources = dict(args.source) if args.source else DEFAULT_SOURCES
    fixed_regions = read_fixed_regions(args.fixed_regions.resolve()) if args.fixed_regions else None
    shared_observed_root = args.shared_observed_prefix.resolve() if args.shared_observed_prefix else None
    if len(sources) < 2:
        raise ValueError("split-region comparison requires baseline plus at least one method")
    if args.halo_m < 0 or args.boundary_guard_m < 0 or args.halo_voxel_m <= 0:
        raise ValueError("halo/guard must be non-negative and halo voxel size must be positive")
    if args.guide_source not in sources:
        raise ValueError(f"guide source is not among compared sources: {args.guide_source}")
    for name, source in sources.items():
        missing = [frame for frame in frames if not (source / f"{frame}.bin").is_file()]
        if missing:
            raise FileNotFoundError(f"{name} is missing {len(missing)} frames: {missing[:10]}")

    workspace.mkdir(parents=True, exist_ok=True)
    region_rows: list[dict] = []
    source_rows: list[dict] = []
    split_members: dict[tuple[str, int], list[str]] = {}
    source_totals = {
        name: {"valid": 0, "staged": 0, "inference_staged": 0, "regions": 0}
        for name in sources
    }
    frame_region_counts: list[int] = []

    root = Region(X_BOUNDS[0], X_BOUNDS[1], Z_BOUNDS[0], Z_BOUNDS[1], 0)
    for position, frame in enumerate(frames, start=1):
        valid_points: dict[str, np.ndarray] = {}
        coordinates: dict[str, np.ndarray] = {}
        shared_points = shared_valid = shared_rect_valid = None
        if shared_observed_root is not None:
            shared_points = prep.read_bin(shared_observed_root / f"{frame}.bin")
            shared_mask, shared_rect = prep.fov_valid_mask(shared_points, frame)
            shared_valid = shared_points[shared_mask]
            shared_rect_valid = shared_rect[shared_mask]
        for name, source in sources.items():
            points = prep.read_bin(source / f"{frame}.bin")
            if shared_points is None:
                valid_mask, rect = prep.fov_valid_mask(points, frame)
                selected = points[valid_mask]
                selected_rect = rect[valid_mask]
            else:
                prefix_count = shared_points.shape[0]
                if points.shape[0] < prefix_count or not np.array_equal(
                    points[:prefix_count], shared_points
                ):
                    raise ValueError(f"{frame}/{name}: shared observed prefix is not exact")
                tail = points[prefix_count:]
                if tail.shape[0]:
                    tail_mask, tail_rect = prep.fov_valid_mask(tail, frame)
                    selected = np.concatenate((shared_valid, tail[tail_mask]), axis=0)
                    selected_rect = np.concatenate(
                        (shared_rect_valid, tail_rect[tail_mask]), axis=0
                    )
                else:
                    selected = shared_valid
                    selected_rect = shared_rect_valid
            valid_points[name] = selected
            coordinates[name] = selected_rect[:, (0, 2)].astype(np.float64, copy=False)
            source_totals[name]["valid"] += int(selected.shape[0])

        if fixed_regions is None:
            capacity_margin = args.halo_m if args.halo_cap_policy == "strict" else 0.0
            leaves = split_region(
                root,
                coordinates,
                args.max_points,
                args.max_depth,
                capacity_margin,
                args.split_policy,
                args.guide_source,
                args.boundary_guard_m,
            )
            leaves.sort(key=lambda item: (item.z_min, item.x_min, item.z_max, item.x_max))
        else:
            if frame not in fixed_regions:
                raise KeyError(f"fixed region manifest has no regions for {frame}")
            leaves = fixed_regions[frame]
        frame_region_counts.append(len(leaves))
        for name, xz in coordinates.items():
            covered = np.zeros(xz.shape[0], dtype=bool)
            for region in leaves:
                covered |= in_region(xz, region)
            if not covered.all():
                raise RuntimeError(
                    f"{frame}/{name}: {int((~covered).sum())} valid points are outside all leaves; "
                    f"first_uncovered={xz[~covered][:5].tolist()} root={root}"
                )

        for slot, region in enumerate(leaves):
            core_counts = counts_for(region, coordinates)
            available_counts = counts_for(region, coordinates, args.halo_m)
            if max(core_counts.values()) > args.max_points:
                raise RuntimeError(f"core capacity violation {frame}/slot_{slot:03d}: {core_counts}")
            if (
                args.halo_cap_policy == "strict"
                and max(available_counts.values()) > args.max_points
            ):
                raise RuntimeError(
                    f"strict halo capacity violation {frame}/slot_{slot:03d}: {available_counts}"
                )
            region_rows.append(
                {
                    "frame_id": frame,
                    "slot": slot,
                    "x_min_rect_m": region.x_min,
                    "x_max_rect_m": region.x_max,
                    "z_min_rect_m": region.z_min,
                    "z_max_rect_m": region.z_max,
                    "tree_depth": region.depth,
                    "halo_m": args.halo_m,
                    "max_source_core_points": max(core_counts.values()),
                    "max_source_available_points": max(available_counts.values()),
                    "max_source_points": min(max(available_counts.values()), args.max_points),
                    "max_source": max(available_counts, key=available_counts.get),
                }
            )
            for name in sources:
                core_mask = in_region(coordinates[name], region)
                core_count = int(core_mask.sum())
                if args.halo_cap_policy == "balanced_sample":
                    selected_indices, selected_halo, clipped_halo = select_core_with_balanced_halo(
                        coordinates[name],
                        region,
                        args.halo_m,
                        args.max_points,
                        args.halo_voxel_m,
                    )
                    staged = valid_points[name][selected_indices]
                else:
                    mask = in_region(coordinates[name], region, args.halo_m)
                    staged = valid_points[name][mask]
                    selected_halo = int(staged.shape[0]) - core_count
                    clipped_halo = 0
                count = int(staged.shape[0])
                source_totals[name]["staged"] += core_count
                source_totals[name]["inference_staged"] += count
                if count:
                    source_totals[name]["regions"] += 1
                    split_members.setdefault((name, slot), []).append(frame)
                    if not args.manifest_only:
                        output = workspace / "inputs" / name / f"slot_{slot:03d}" / f"{frame}.bin"
                        output.parent.mkdir(parents=True, exist_ok=True)
                        expected_bytes = count * 4 * np.dtype(np.float32).itemsize
                        if not output.is_file() or output.stat().st_size != expected_bytes:
                            staged.astype(np.float32, copy=False).tofile(output)
                source_rows.append(
                    {
                        "frame_id": frame,
                        "slot": slot,
                        "source": name,
                        "core_points": core_count,
                        "available_halo_points": available_counts[name] - core_count,
                        "selected_halo_points": selected_halo,
                        "clipped_halo_points": clipped_halo,
                        "points": count,
                        "nonempty": bool(count),
                        "within_native_16384": count <= args.max_points,
                    }
                )

        if position == 1 or position % 16 == 0 or position == len(frames):
            print(
                f"SPLIT_REGION_PREP {position}/{len(frames)} frame={frame} regions={len(leaves)}",
                flush=True,
            )

    for name, totals in source_totals.items():
        if totals["valid"] != totals["staged"]:
            raise RuntimeError(f"lossless coverage failed for {name}: {totals}")
    for (name, slot), members in split_members.items():
        split_path = workspace / "splits" / name / f"slot_{slot:03d}.txt"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        split_path.write_text("\n".join(members) + "\n")

    write_csv(workspace / "manifests/regions.csv", region_rows)
    write_csv(workspace / "manifests/source_region_counts.csv", source_rows)
    protocol = {
        "experiment": (
            "pointrcnn_object_preserving_core_halo"
            if args.halo_m > 0 or args.split_policy != "median"
            else "pointrcnn_split_region_surface256_v1"
        ),
        "split_file": str(split_file),
        "frames": len(frames),
        "sources": {name: str(path) for name, path in sources.items()},
        "region_coordinate_system": "rect_camera_x_z_after_PointRCNN_FOV_and_range_filter",
        "root_bounds": {"x": X_BOUNDS, "z": Z_BOUNDS},
        "max_points_per_nonempty_region": args.max_points,
        "split_policy": args.split_policy,
        "guide_source": args.guide_source,
        "boundary_guard_m": args.boundary_guard_m,
        "halo_m": args.halo_m,
        "halo_cap_policy": args.halo_cap_policy,
        "halo_voxel_m": args.halo_voxel_m,
        "shared_boundaries_across_sources": True,
        "disjoint_core_regions": True,
        "overlapping_inference_regions": args.halo_m > 0,
        "uses_labels_boxes_detector_outputs_or_ap": False,
        "point_loss_in_core_ownership_domain": 0,
        "halo_note": (
            "all core points retained; overlapping halo is deterministically spatially balanced "
            "when capacity is exceeded"
            if args.halo_cap_policy == "balanced_sample"
            else "entire halo retained"
        ),
        "frame_region_count": {
            "min": min(frame_region_counts),
            "median": float(np.median(frame_region_counts)),
            "max": max(frame_region_counts),
            "total_virtual_frames": int(sum(frame_region_counts)),
        },
        "source_totals": source_totals,
        "manifest_only": args.manifest_only,
        "fixed_region_manifest": str(args.fixed_regions.resolve()) if args.fixed_regions else None,
        "shared_observed_prefix": str(shared_observed_root) if shared_observed_root else None,
        "status": "PASS",
    }
    (workspace / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
