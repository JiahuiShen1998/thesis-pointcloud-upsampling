#!/usr/bin/env python3
"""Prepare auditable detector inputs for PointRCNN E1/E2 experiments.

E1 keeps every observed input point and fills the strict x4 frame with a
deterministic 3x subset of the already-audited predicted x4 cloud:

    Line A: N observed + 3N generated = 4N
    Line B: M observed + 3M generated = 4M

E2 is a detector-facing adapter derived from the exact same E1 cloud.  It
reproduces PointRCNN's FOV/range filtering, performs deterministic 0.10 m
voxel representative selection followed by proportional depth-stratified
sampling, and writes exactly 16,384 valid points.  The same E2 adapter is
also applied to the original and downsampled baselines.

No labels, boxes, AP values, or method-specific thresholds are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from lib.utils.calibration import Calibration  # noqa: E402


DEFAULT_SOURCE_ROOT = REPO / "results/kitti_unified_x4_current_methods_no_detector"
DEFAULT_WORKSPACE = REPO / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
TRAINING = REPO / "data/KITTI/object/training"
VAL_SPLIT = REPO / "data/KITTI/ImageSets/val.txt"
ORIGINAL = TRAINING / "velodyne_original_val"
DOWNSAMPLED = DEFAULT_SOURCE_ROOT / "downsampled_x4/velodyne_downsampled_x4_val"

METHODS = ("pu_net", "pu_gcn", "pu_edgeformer", "pdans")
LINES = {
    "line_a": {
        "input": ORIGINAL,
        "source_prefix": "line_a_original_x4_up",
        "variant_prefix": "original_x4",
    },
    "line_b": {
        "input": DOWNSAMPLED,
        "source_prefix": "line_b_downsampled_x4_up",
        "variant_prefix": "downsampled_x4",
    },
}

TARGET_POINTS = 16384
VOXEL_SIZE_M = 0.10
DEPTH_EDGES_M = (0.0, 20.0, 40.0, 60.0, 70.400001)
PC_SCOPE = np.asarray([[-40.0, 40.0], [-1.0, 3.0], [0.0, 70.4]], dtype=np.float32)
BASE_SEED = 20260718


def read_frames(path: Path = VAL_SPLIT) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"{path} is not an Nx4 float32 KITTI point cloud")
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN/Inf")
    return points


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(x) for x in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def variant_specs(source_root: Path) -> dict[str, dict[str, Path | str]]:
    specs: dict[str, dict[str, Path | str]] = {}
    for line, line_cfg in LINES.items():
        for method in METHODS:
            name = f"{line_cfg['variant_prefix']}_{method}"
            specs[name] = {
                "line": line,
                "method": method,
                "observed_input": Path(line_cfg["input"]),
                "predicted_x4": source_root / str(line_cfg["source_prefix"]) / method / "final_bin",
            }
    return specs


def exact_output_is_valid(path: Path, expected_points: int) -> bool:
    return path.is_file() and path.stat().st_size == expected_points * 4 * np.dtype(np.float32).itemsize


def prepare_e1_variant(
    name: str,
    spec: dict[str, Path | str],
    frames: Iterable[str],
    output_dir: Path,
    manifest_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    observed_dir = Path(spec["observed_input"])
    predicted_dir = Path(spec["predicted_x4"])

    for position, frame in enumerate(frames, start=1):
        input_path = observed_dir / f"{frame}.bin"
        predicted_path = predicted_dir / f"{frame}.bin"
        output_path = output_dir / f"{frame}.bin"
        observed = read_bin(input_path)
        input_count = int(observed.shape[0])
        target = 4 * input_count
        generated_count = 3 * input_count
        seed = stable_seed(BASE_SEED, "e1", name, frame)

        status = "REUSED"
        if not exact_output_is_valid(output_path, target):
            predicted = read_bin(predicted_path)
            if predicted.shape[0] != target:
                raise ValueError(
                    f"{name}/{frame}: audited predicted cloud has {predicted.shape[0]} points, expected {target}"
                )
            rng = np.random.default_rng(seed)
            generated_idx = rng.choice(predicted.shape[0], size=generated_count, replace=False)
            final = np.concatenate((observed, predicted[generated_idx]), axis=0).astype(np.float32, copy=False)
            final = final[rng.permutation(target)]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            final.tofile(output_path)
            status = "WRITTEN"

        rows.append(
            {
                "experiment": "E1_default_16384",
                "variant": name,
                "line": spec["line"],
                "method": spec["method"],
                "frame_id": frame,
                "observed_input_points": input_count,
                "generated_points": generated_count,
                "final_points": target,
                "observed_fraction": 0.25,
                "input_xyz_and_intensity_preserved": True,
                "candidate_policy": "deterministic_without_replacement_from_audited_all_predicted_exact_x4",
                "seed": seed,
                "status": status,
                "output_path": str(output_path),
            }
        )
        if position == 1 or position % 250 == 0:
            print(f"E1 {name}: {position} frames prepared", flush=True)

    write_csv(manifest_dir / f"{name}.csv", rows)
    return rows


def fov_valid_mask(points: np.ndarray, frame: str) -> tuple[np.ndarray, np.ndarray]:
    calib = Calibration(str(TRAINING / "calib" / f"{frame}.txt"))
    with Image.open(TRAINING / "image_2" / f"{frame}.png") as image:
        width, height = image.size
    rect = calib.lidar_to_rect(points[:, :3])
    image_xy, rect_depth = calib.rect_to_img(rect)
    valid = (
        (image_xy[:, 0] >= 0.0)
        & (image_xy[:, 0] < width)
        & (image_xy[:, 1] >= 0.0)
        & (image_xy[:, 1] < height)
        & (rect_depth >= 0.0)
        & (rect[:, 0] >= PC_SCOPE[0, 0])
        & (rect[:, 0] <= PC_SCOPE[0, 1])
        & (rect[:, 1] >= PC_SCOPE[1, 0])
        & (rect[:, 1] <= PC_SCOPE[1, 1])
        & (rect[:, 2] >= PC_SCOPE[2, 0])
        & (rect[:, 2] <= PC_SCOPE[2, 1])
    )
    return valid, rect


def largest_remainder_quotas(counts: np.ndarray, target: int) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if target < 0 or target > total:
        raise ValueError(f"quota target {target} is incompatible with {total} candidates")
    if target == total:
        return counts.copy()
    raw = counts.astype(np.float64) * (float(target) / float(total))
    quotas = np.floor(raw).astype(np.int64)
    remaining = target - int(quotas.sum())
    order = np.lexsort((np.arange(counts.size), -(raw - quotas)))
    for idx in order:
        if remaining == 0:
            break
        if quotas[idx] < counts[idx]:
            quotas[idx] += 1
            remaining -= 1
    if int(quotas.sum()) != target:
        raise RuntimeError("largest-remainder quota allocation failed")
    return quotas


def proportional_depth_sample(
    candidate_idx: np.ndarray,
    rect_depth: np.ndarray,
    target: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if target >= candidate_idx.size:
        return candidate_idx.copy()
    bin_id = np.searchsorted(np.asarray(DEPTH_EDGES_M[1:-1]), rect_depth[candidate_idx], side="right")
    groups = [candidate_idx[bin_id == idx] for idx in range(len(DEPTH_EDGES_M) - 1)]
    quotas = largest_remainder_quotas(np.asarray([group.size for group in groups]), target)
    chosen = []
    for group, quota in zip(groups, quotas):
        if quota:
            chosen.append(rng.choice(group, size=int(quota), replace=False))
    output = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    return output[rng.permutation(output.size)]


def canonical_choice(points: np.ndarray, frame: str, seed: int) -> tuple[np.ndarray, dict]:
    valid_mask, rect = fov_valid_mask(points, frame)
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        raise ValueError(f"{frame}: no PointRCNN-valid points")

    rng = np.random.default_rng(seed)
    shuffled = valid_idx[rng.permutation(valid_idx.size)]
    voxel_key = np.floor(rect[shuffled] / VOXEL_SIZE_M).astype(np.int64)
    _, first = np.unique(voxel_key, axis=0, return_index=True)
    unique_idx = shuffled[np.sort(first)]

    if unique_idx.size >= TARGET_POINTS:
        selected = proportional_depth_sample(unique_idx, rect[:, 2], TARGET_POINTS, rng)
        fill_policy = "none"
    else:
        selected = unique_idx.copy()
        selected_mask = np.zeros(points.shape[0], dtype=bool)
        selected_mask[selected] = True
        remaining_idx = valid_idx[~selected_mask[valid_idx]]
        needed = TARGET_POINTS - selected.size
        if remaining_idx.size >= needed:
            fill = proportional_depth_sample(remaining_idx, rect[:, 2], needed, rng)
            fill_policy = "unused_valid_without_replacement"
        else:
            first_fill = remaining_idx
            still_needed = needed - first_fill.size
            repeat_fill = rng.choice(valid_idx, size=still_needed, replace=True)
            fill = np.concatenate((first_fill, repeat_fill))
            fill_policy = "all_valid_then_repeat_with_replacement"
        selected = np.concatenate((selected, fill))
        selected = selected[rng.permutation(selected.size)]

    if selected.size != TARGET_POINTS:
        raise RuntimeError(f"{frame}: canonical sampler returned {selected.size}, expected {TARGET_POINTS}")
    stats = {
        "source_points": int(points.shape[0]),
        "fov_valid_points": int(valid_idx.size),
        "voxel_unique_points": int(unique_idx.size),
        "selected_points": int(selected.size),
        "fill_policy": fill_policy,
    }
    return selected, stats


def prepare_e2_variant(
    name: str,
    source_dir: Path,
    source_kind: str,
    frames: Iterable[str],
    output_dir: Path,
    manifest_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for position, frame in enumerate(frames, start=1):
        source_path = source_dir / f"{frame}.bin"
        output_path = output_dir / f"{frame}.bin"
        points = read_bin(source_path)
        seed = stable_seed(BASE_SEED, "e2", name, frame)
        selected, stats = canonical_choice(points, frame, seed)
        status = "REUSED"
        if not exact_output_is_valid(output_path, TARGET_POINTS):
            points[selected].astype(np.float32, copy=False).tofile(output_path)
            status = "WRITTEN"

        # Re-run the exact FOV test on the saved detector-facing file.  This
        # catches boundary or calibration mismatches before expensive AP runs.
        saved = read_bin(output_path)
        saved_valid, _ = fov_valid_mask(saved, frame)
        detector_valid = int(saved_valid.sum())
        if detector_valid != TARGET_POINTS:
            raise ValueError(
                f"{name}/{frame}: saved canonical input has {detector_valid}/{TARGET_POINTS} valid points"
            )

        rows.append(
            {
                "experiment": "E2_canonical_16384",
                "variant": name,
                "frame_id": frame,
                "source_kind": source_kind,
                **stats,
                "detector_valid_points_after_reload": detector_valid,
                "voxel_size_m": VOXEL_SIZE_M,
                "depth_edges_m": "|".join(str(x) for x in DEPTH_EDGES_M),
                "quota_policy": "proportional_largest_remainder",
                "seed": seed,
                "uses_labels_or_ap": False,
                "status": status,
                "output_path": str(output_path),
            }
        )
        if position == 1 or position % 250 == 0:
            print(f"E2 {name}: {position} frames prepared", flush=True)

    write_csv(manifest_dir / f"{name}.csv", rows)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_names(all_names: Iterable[str], requested: list[str] | None) -> list[str]:
    names = list(all_names)
    if not requested:
        return names
    unknown = sorted(set(requested) - set(names))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; choices={names}")
    return [name for name in names if name in requested]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--stage", choices=("e1", "e2", "all"), default="all")
    parser.add_argument("--variant", action="append", help="Repeat to prepare selected method variants")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test first N val frames")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    frames = read_frames()
    if args.limit:
        frames = frames[: args.limit]
    specs = variant_specs(args.source_root.resolve())
    method_names = select_names(specs, args.variant)
    started = time.time()

    protocol = {
        "protocol_version": "input_preserving_x4_e1_e2_v1",
        "created_by": str(Path(__file__).resolve()),
        "val_split": str(VAL_SPLIT),
        "frame_count": len(frames),
        "base_seed": BASE_SEED,
        "E1": {
            "line_a": "N observed + 3N generated = exact 4N",
            "line_b": "M observed + 3M generated = exact 4M",
            "candidate_source": "existing audited all-predicted exact-x4 detector-ready cloud",
            "candidate_selection": "deterministic random without replacement",
            "detector_sampler": "PointRCNN default sampler-safe 16384",
        },
        "E2": {
            "source": "same E1 cloud (baselines use their native external cloud)",
            "fov_and_range": "exact PointRCNN default KITTI filtering",
            "voxel_size_m": VOXEL_SIZE_M,
            "depth_edges_m": DEPTH_EDGES_M,
            "quota_policy": "candidate-proportional largest remainder",
            "target_points": TARGET_POINTS,
            "detector_sampler": "no near/far downsampling because saved valid count is exactly 16384",
        },
        "fairness": {
            "same_rule_all_methods": True,
            "same_rule_both_lines": True,
            "same_e2_rule_baselines": True,
            "uses_ground_truth": False,
            "uses_ap_for_selection": False,
            "method_specific_thresholds": False,
        },
    }
    write_json(workspace / "protocol.json", protocol)

    if args.stage in ("e1", "all"):
        for name in method_names:
            prepare_e1_variant(
                name,
                specs[name],
                frames,
                workspace / "inputs/e1_default_16384" / name,
                workspace / "manifests/e1_default_16384",
            )

    if args.stage in ("e2", "all"):
        e2_specs: dict[str, tuple[Path, str]] = {
            "original_baseline": (ORIGINAL, "native_original_N"),
            "downsampled_x4_baseline": (DOWNSAMPLED, "native_downsampled_M"),
        }
        for name in method_names:
            e2_specs[name] = (workspace / "inputs/e1_default_16384" / name, "E1_input_preserving_exact_x4")
        for name, (source_dir, source_kind) in e2_specs.items():
            prepare_e2_variant(
                name,
                source_dir,
                source_kind,
                frames,
                workspace / "inputs/e2_canonical_16384" / name,
                workspace / "manifests/e2_canonical_16384",
            )

    write_json(
        workspace / "preparation_status.json",
        {
            "status": "PASS",
            "stage": args.stage,
            "variants": method_names,
            "frame_count": len(frames),
            "runtime_seconds": time.time() - started,
        },
    )
    print(f"PREPARATION_PASS workspace={workspace} runtime={time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
