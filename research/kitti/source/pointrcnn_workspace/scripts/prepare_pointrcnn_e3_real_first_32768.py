#!/usr/bin/env python3
"""Prepare label-free real-first 32768 PointRCNN sensitivity inputs.

The audited upstream protocol remains exact x4: N observed + 3N generated.
This detector adapter first retains every FOV-valid observed measurement, then
fills the remaining detector capacity with the exact E1 generated candidate
set using the same 0.1 m voxel/depth policy as E2.  No labels, boxes, detector
outputs, or AP values are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import prepare_pointrcnn_e1_e2_inputs as prep  # noqa: E402


SOURCE = REPO / "results/kitti_unified_x4_current_methods_no_detector"
WORKSPACE = REPO / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
TARGET_POINTS = 32768
BASE_SEED = 20260720
METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
LINES = {
    "A": {
        "observed": prep.ORIGINAL,
        "baseline": "original_baseline",
        "source_prefix": "line_a_original_x4_up",
        "variant_prefix": "original_x4",
    },
    "B": {
        "observed": prep.DOWNSAMPLED,
        "baseline": "downsampled_x4_baseline",
        "source_prefix": "line_b_downsampled_x4_up",
        "variant_prefix": "downsampled_x4",
    },
}


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def choose_candidates(
    rect: np.ndarray,
    target: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Choose target rows from an already FOV-valid candidate array."""
    count = rect.shape[0]
    if count == 0:
        raise ValueError("no valid candidates")
    shuffled = rng.permutation(count)
    voxel_key = np.floor(rect[shuffled] / prep.VOXEL_SIZE_M).astype(np.int64)
    _, first = np.unique(voxel_key, axis=0, return_index=True)
    unique_idx = shuffled[np.sort(first)]

    if unique_idx.size >= target:
        selected = prep.proportional_depth_sample(unique_idx, rect[:, 2], target, rng)
        fill_policy = "none"
    else:
        selected = unique_idx.copy()
        selected_mask = np.zeros(count, dtype=bool)
        selected_mask[selected] = True
        remaining_idx = np.flatnonzero(~selected_mask)
        needed = target - selected.size
        if remaining_idx.size >= needed:
            fill = prep.proportional_depth_sample(remaining_idx, rect[:, 2], needed, rng)
            fill_policy = "unused_valid_without_replacement"
        else:
            first_fill = remaining_idx
            still_needed = needed - first_fill.size
            repeat_fill = rng.choice(np.arange(count), size=still_needed, replace=True)
            fill = np.concatenate((first_fill, repeat_fill))
            fill_policy = "all_valid_then_repeat_with_replacement"
        selected = np.concatenate((selected, fill))
        selected = selected[rng.permutation(selected.size)]

    if selected.size != target:
        raise RuntimeError(f"candidate selection produced {selected.size}, expected {target}")
    return selected, {
        "candidate_points": int(count),
        "candidate_voxel_unique_points": int(unique_idx.size),
        "fill_policy": fill_policy,
    }


def recreate_e1_generated(line: str, method: str, frame: str, observed_n: int) -> np.ndarray:
    cfg = LINES[line]
    predicted = prep.read_bin(
        SOURCE / cfg["source_prefix"] / method / "final_bin" / f"{frame}.bin"
    )
    expected = 4 * observed_n
    if predicted.shape[0] != expected:
        raise ValueError(
            f"{line}/{method}/{frame}: predicted {predicted.shape[0]} != exact x4 {expected}"
        )
    variant = f"{cfg['variant_prefix']}_{method}"
    rng = np.random.default_rng(
        prep.stable_seed(prep.BASE_SEED, "e1", variant, frame)
    )
    selected = rng.choice(expected, size=3 * observed_n, replace=False)
    return predicted[selected]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_baseline(line: str, frames: list[str], output_root: Path, manifest_root: Path) -> None:
    cfg = LINES[line]
    variant = cfg["baseline"]
    output_dir = output_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for position, frame in enumerate(frames, start=1):
        observed = prep.read_bin(Path(cfg["observed"]) / f"{frame}.bin")
        valid_mask, rect_all = prep.fov_valid_mask(observed, frame)
        observed_valid = observed[valid_mask]
        rect = rect_all[valid_mask]
        rng = np.random.default_rng(stable_seed(BASE_SEED, variant, frame))
        selected, stats = choose_candidates(rect, TARGET_POINTS, rng)
        final = observed_valid[selected]
        final.tofile(output_dir / f"{frame}.bin")
        rows.append(
            {
                "experiment": "E3_real_first_32768",
                "variant": variant,
                "frame_id": frame,
                "source_protocol": "baseline_observed_only",
                "target_points": TARGET_POINTS,
                "observed_points": int(observed.shape[0]),
                "observed_fov_points": int(observed_valid.shape[0]),
                "observed_selected_points": TARGET_POINTS,
                "generated_fov_points": 0,
                "generated_selected_points": 0,
                "observed_selected_fraction": 1.0,
                **stats,
                "uses_labels_or_ap": False,
                "status": "WRITTEN",
            }
        )
        if position == 1 or position % 250 == 0:
            print(f"E3 {variant}: {position}/{len(frames)}", flush=True)
    write_csv(manifest_root / f"{variant}.csv", rows)


def prepare_method(
    line: str,
    method: str,
    frames: list[str],
    output_root: Path,
    manifest_root: Path,
) -> None:
    cfg = LINES[line]
    variant = f"{cfg['variant_prefix']}_{method}"
    output_dir = output_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for position, frame in enumerate(frames, start=1):
        observed = prep.read_bin(Path(cfg["observed"]) / f"{frame}.bin")
        observed_mask, observed_rect_all = prep.fov_valid_mask(observed, frame)
        observed_valid = observed[observed_mask]
        observed_rect = observed_rect_all[observed_mask]
        rng = np.random.default_rng(stable_seed(BASE_SEED, variant, frame))

        if observed_valid.shape[0] >= TARGET_POINTS:
            observed_idx, observed_stats = choose_candidates(
                observed_rect, TARGET_POINTS, rng
            )
            final = observed_valid[observed_idx]
            generated_fov_points = 0
            generated_selected_points = 0
            generated_stats = {
                "candidate_points": 0,
                "candidate_voxel_unique_points": 0,
                "fill_policy": "not_used_observed_filled_target",
            }
        else:
            generated = recreate_e1_generated(line, method, frame, observed.shape[0])
            generated_mask, generated_rect_all = prep.fov_valid_mask(generated, frame)
            generated_valid = generated[generated_mask]
            generated_rect = generated_rect_all[generated_mask]
            needed = TARGET_POINTS - observed_valid.shape[0]
            generated_idx, generated_stats = choose_candidates(generated_rect, needed, rng)
            final = np.concatenate((observed_valid, generated_valid[generated_idx]), axis=0)
            final = final[rng.permutation(final.shape[0])]
            observed_stats = {
                "candidate_points": int(observed_valid.shape[0]),
                "candidate_voxel_unique_points": int(
                    np.unique(
                        np.floor(observed_rect / prep.VOXEL_SIZE_M).astype(np.int64), axis=0
                    ).shape[0]
                ),
                "fill_policy": "all_observed_retained",
            }
            generated_fov_points = int(generated_valid.shape[0])
            generated_selected_points = int(needed)

        if final.shape != (TARGET_POINTS, 4) or not np.isfinite(final).all():
            raise RuntimeError(f"{variant}/{frame}: invalid final shape/content {final.shape}")
        final.tofile(output_dir / f"{frame}.bin")
        rows.append(
            {
                "experiment": "E3_real_first_32768",
                "variant": variant,
                "frame_id": frame,
                "source_protocol": "E1_N_observed_plus_3N_generated_exact_x4",
                "target_points": TARGET_POINTS,
                "observed_points": int(observed.shape[0]),
                "observed_fov_points": int(observed_valid.shape[0]),
                "observed_selected_points": int(
                    TARGET_POINTS if observed_valid.shape[0] >= TARGET_POINTS else observed_valid.shape[0]
                ),
                "generated_fov_points": generated_fov_points,
                "generated_selected_points": generated_selected_points,
                "observed_selected_fraction": float(
                    min(observed_valid.shape[0], TARGET_POINTS) / TARGET_POINTS
                ),
                "observed_voxel_unique_points": observed_stats[
                    "candidate_voxel_unique_points"
                ],
                "observed_policy": observed_stats["fill_policy"],
                "generated_voxel_unique_points": generated_stats[
                    "candidate_voxel_unique_points"
                ],
                "generated_policy": generated_stats["fill_policy"],
                "uses_labels_or_ap": False,
                "status": "WRITTEN",
            }
        )
        if position == 1 or position % 250 == 0:
            print(f"E3 {variant}: {position}/{len(frames)}", flush=True)
    write_csv(manifest_root / f"{variant}.csv", rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--frames", type=int, default=0, help="0 means full val split")
    parser.add_argument("--variant", action="append", help="Repeat to prepare selected variants")
    args = parser.parse_args()

    frames = prep.read_frames()
    if args.frames:
        frames = frames[: args.frames]
    workspace = args.workspace.resolve()
    output_root = workspace / "inputs/e3_real_first_32768"
    manifest_root = workspace / "manifests/e3_real_first_32768"
    selected = set(args.variant or [])

    for line, cfg in LINES.items():
        if not selected or cfg["baseline"] in selected:
            prepare_baseline(line, frames, output_root, manifest_root)
        for method in METHODS:
            variant = f"{cfg['variant_prefix']}_{method}"
            if not selected or variant in selected:
                prepare_method(line, method, frames, output_root, manifest_root)
    print(
        f"E3_REAL_FIRST_32768_PREPARATION_PASS workspace={workspace} frames={len(frames)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
