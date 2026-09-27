#!/usr/bin/env python3
"""Prepare isolated 16,384-point observed/generated ratio recovery inputs.

The source E1 protocol is unchanged and read-only: every frame contains the
multiset ``N observed + 3N generated`` in deterministically shuffled order.
This adapter preserves provenance by replaying E1's original deterministic
3N candidate indices, applies the existing PointRCNN FOV/range transform, and
constructs detector inputs with an exact generated-point quota.

No labels, boxes, detector outputs, AP values, or method-specific tuning are
used.  Outputs and manifests are written under a separate recovery workspace.
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
from prepare_pointrcnn_e3_real_first_32768 import (  # noqa: E402
    choose_candidates,
    recreate_e1_generated,
)


SOURCE_WORKSPACE = REPO / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
DEFAULT_WORKSPACE = REPO / "results/kitti_x4_detector_recovery_ratio_v1_20260723"
TARGET_POINTS = 16384
BASE_SEED = 20260723
DEFAULT_RATIOS = (0.10, 0.15, 0.25, 0.35, 0.40, 0.50)
METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
LINES = {
    "A": {
        "observed": prep.ORIGINAL,
        "variant_prefix": "original_x4",
    },
    "B": {
        "observed": prep.DOWNSAMPLED,
        "variant_prefix": "downsampled_x4",
    },
}


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def read_frames(path: Path | None) -> list[str]:
    if path is None:
        return prep.read_frames()
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ratio_name(ratio: float) -> str:
    percent = int(round(100.0 * ratio))
    if abs(percent / 100.0 - ratio) > 1e-9:
        raise ValueError(f"ratio must be an integer percentage, got {ratio}")
    return f"g{percent:02d}_o{100 - percent:02d}_16384"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def category_selection(
    points: np.ndarray,
    rect: np.ndarray,
    target: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    if target == 0:
        return np.empty((0, 4), dtype=np.float32), {
            "candidate_points": int(points.shape[0]),
            "candidate_voxel_unique_points": 0,
            "fill_policy": "quota_zero",
        }
    if points.shape[0] == 0:
        raise ValueError(f"non-zero quota {target} has no FOV-valid candidates")
    indices, stats = choose_candidates(rect, target, rng)
    return points[indices], stats


def prepare_variant(
    line: str,
    method: str,
    frames: list[str],
    ratios: tuple[float, ...],
    workspace: Path,
) -> None:
    cfg = LINES[line]
    variant = f"{cfg['variant_prefix']}_{method}"
    e1_dir = SOURCE_WORKSPACE / "inputs/e1_default_16384" / variant
    output_dirs = {
        ratio: workspace / "inputs" / ratio_name(ratio) / variant for ratio in ratios
    }
    manifest_rows: dict[float, list[dict]] = {ratio: [] for ratio in ratios}
    for output_dir in output_dirs.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    for position, frame in enumerate(frames, start=1):
        observed = prep.read_bin(Path(cfg["observed"]) / f"{frame}.bin")
        e1_path = e1_dir / f"{frame}.bin"
        expected = 4 * observed.shape[0]
        if e1_path.stat().st_size != expected * 4 * np.dtype(np.float32).itemsize:
            raise ValueError(f"{variant}/{frame}: E1 file size is not exact 4N")
        # E1 shuffles observed and generated rows before writing. Recreate the
        # exact 3N generated multiset with the original deterministic seed
        # rather than guessing provenance from row proximity or row order.
        generated = recreate_e1_generated(line, method, frame, observed.shape[0])
        if generated.shape != (3 * observed.shape[0], 4):
            raise ValueError(f"{variant}/{frame}: recreated generated rows have shape {generated.shape}")
        observed_mask, observed_rect_all = prep.fov_valid_mask(observed, frame)
        generated_mask, generated_rect_all = prep.fov_valid_mask(generated, frame)
        observed_valid = observed[observed_mask]
        observed_rect = observed_rect_all[observed_mask]
        generated_valid = generated[generated_mask]
        generated_rect = generated_rect_all[generated_mask]

        for ratio in ratios:
            generated_target = int(round(TARGET_POINTS * ratio))
            observed_target = TARGET_POINTS - generated_target
            rng_observed = np.random.default_rng(
                stable_seed(BASE_SEED, ratio_name(ratio), variant, frame, "observed")
            )
            rng_generated = np.random.default_rng(
                stable_seed(BASE_SEED, ratio_name(ratio), variant, frame, "generated")
            )
            observed_selected, observed_stats = category_selection(
                observed_valid, observed_rect, observed_target, rng_observed
            )
            generated_selected, generated_stats = category_selection(
                generated_valid, generated_rect, generated_target, rng_generated
            )
            final = np.concatenate((observed_selected, generated_selected), axis=0)
            rng_final = np.random.default_rng(
                stable_seed(BASE_SEED, ratio_name(ratio), variant, frame, "shuffle")
            )
            final = final[rng_final.permutation(final.shape[0])].astype(np.float32, copy=False)
            if final.shape != (TARGET_POINTS, 4) or not np.isfinite(final).all():
                raise RuntimeError(f"{ratio_name(ratio)}/{variant}/{frame}: bad final {final.shape}")
            final.tofile(output_dirs[ratio] / f"{frame}.bin")
            manifest_rows[ratio].append(
                {
                    "experiment": "detector_recovery_ratio_v1",
                    "policy": ratio_name(ratio),
                    "line": line,
                    "method": method,
                    "variant": variant,
                    "frame_id": frame,
                    "target_points": TARGET_POINTS,
                    "observed_source_points": int(observed.shape[0]),
                    "generated_source_points": int(generated.shape[0]),
                    "observed_fov_points": int(observed_valid.shape[0]),
                    "generated_fov_points": int(generated_valid.shape[0]),
                    "observed_selected_points": int(observed_target),
                    "generated_selected_points": int(generated_target),
                    "observed_selected_fraction": observed_target / TARGET_POINTS,
                    "generated_selected_fraction": generated_target / TARGET_POINTS,
                    "observed_voxel_unique_points": observed_stats["candidate_voxel_unique_points"],
                    "generated_voxel_unique_points": generated_stats["candidate_voxel_unique_points"],
                    "observed_fill_policy": observed_stats["fill_policy"],
                    "generated_fill_policy": generated_stats["fill_policy"],
                    "uses_labels_boxes_detector_or_ap": False,
                    "source_e1": str(e1_path),
                    "generated_provenance": "recreated_exact_E1_3N_indices_with_original_seed",
                    "status": "WRITTEN",
                }
            )

        if position == 1 or position % 64 == 0 or position == len(frames):
            print(f"RECOVERY_INPUT {variant}: {position}/{len(frames)}", flush=True)

    for ratio in ratios:
        write_csv(
            workspace / "manifests" / ratio_name(ratio) / f"{variant}.csv",
            manifest_rows[ratio],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--ratio", type=float, action="append")
    parser.add_argument("--variant", action="append")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace == SOURCE_WORKSPACE.resolve():
        raise ValueError("recovery workspace must not be the original E1/E2 workspace")
    frames = read_frames(args.split_file)
    ratios = tuple(args.ratio or DEFAULT_RATIOS)
    if any(not 0.0 <= ratio <= 1.0 for ratio in ratios):
        raise ValueError(f"ratios must be in [0,1], got {ratios}")
    selected = set(args.variant or [])

    workspace.mkdir(parents=True, exist_ok=True)
    for line, cfg in LINES.items():
        for method in METHODS:
            variant = f"{cfg['variant_prefix']}_{method}"
            if selected and variant not in selected:
                continue
            prepare_variant(line, method, frames, ratios, workspace)

    print(
        f"RECOVERY_RATIO_INPUTS_PASS workspace={workspace} frames={len(frames)} "
        f"ratios={','.join(ratio_name(ratio) for ratio in ratios)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
