#!/usr/bin/env python3
"""Prepare nested fine-ratio PointRCNN inputs from strict-x4 candidates.

The upstream point clouds remain unchanged and exact x4.  This isolated
detector adapter starts from the already validated observed-only E2 baseline
input (exactly 16,384 rows), removes a nested deterministic subset of those
rows, and fills the same slots with generated candidates reconstructed from
the exact E1 3N pool.

Because every method on a comparison line uses the same baseline input and
the same removed observed rows, method and ratio differences are not
confounded by different observed-point samples.  Matched observed-fill
controls at every tested ratio use real observed measurements instead of
generated candidates.
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
DEFAULT_WORKSPACE = REPO / "results/kitti_x4_detector_recovery_fine_ratio_v2_20260725"
TARGET_POINTS = 16384
MAX_GENERATED_RATIO = 0.10
BASE_SEED = 20260725
RATIOS = (0.025, 0.05, 0.075, 0.10)
METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
POLICY_BY_RATIO = {
    0.025: "g025_o975_16384",
    0.05: "g050_o950_16384",
    0.075: "g075_o925_16384",
    0.10: "g100_o900_16384",
}
CONTROL_POLICY_BY_RATIO = {
    0.025: "c025_o975_16384",
    0.05: "c050_o950_16384",
    0.075: "c075_o925_16384",
    0.10: "c100_o900_16384",
}
LINES = {
    "A": {
        "observed": prep.ORIGINAL,
        "baseline_input": SOURCE_WORKSPACE / "inputs/e2_canonical_16384/original_baseline",
        "baseline_variant": "original_baseline",
        "variant_prefix": "original_x4",
        "control_variant": "original_observed_fill",
    },
    "B": {
        "observed": prep.DOWNSAMPLED,
        "baseline_input": SOURCE_WORKSPACE / "inputs/e2_canonical_16384/downsampled_x4_baseline",
        "baseline_variant": "downsampled_x4_baseline",
        "variant_prefix": "downsampled_x4",
        "control_variant": "downsampled_observed_fill",
    },
}


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def read_frames(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_ratio(ratio: float) -> float:
    for supported in RATIOS:
        if abs(ratio - supported) < 1e-9:
            return supported
    raise ValueError(f"unsupported ratio {ratio}; supported ratios are {RATIOS}")


def select_category(
    points: np.ndarray,
    rect: np.ndarray,
    target: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    if target <= 0:
        return np.empty((0, 4), dtype=np.float32), {
            "candidate_points": int(points.shape[0]),
            "candidate_voxel_unique_points": 0,
            "fill_policy": "quota_zero",
        }
    if points.shape[0] == 0:
        raise ValueError(f"non-zero quota {target} has no valid candidates")
    indices, stats = choose_candidates(rect, target, rng)
    return points[indices], stats


def baseline_and_observed(line: str, frame: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = LINES[line]
    baseline = prep.read_bin(Path(cfg["baseline_input"]) / f"{frame}.bin")
    if baseline.shape != (TARGET_POINTS, 4):
        raise ValueError(
            f"{line}/{frame}: baseline detector input has {baseline.shape}, "
            f"expected {(TARGET_POINTS, 4)}"
        )
    observed = prep.read_bin(Path(cfg["observed"]) / f"{frame}.bin")
    observed_mask, observed_rect_all = prep.fov_valid_mask(observed, frame)
    return baseline, observed[observed_mask], observed_rect_all[observed_mask]


def slot_order(line: str, frame: str) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(BASE_SEED, line, frame, "nested_observed_slots"))
    return rng.permutation(TARGET_POINTS)


def final_shuffle(points: np.ndarray, line: str, frame: str, policy: str) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(BASE_SEED, line, frame, policy, "final_shuffle"))
    return points[rng.permutation(points.shape[0])].astype(np.float32, copy=False)


def prepare_method_variant(
    line: str,
    method: str,
    frames: list[str],
    ratios: tuple[float, ...],
    workspace: Path,
) -> None:
    cfg = LINES[line]
    variant = f"{cfg['variant_prefix']}_{method}"
    outputs = {ratio: workspace / "inputs" / POLICY_BY_RATIO[ratio] / variant for ratio in ratios}
    manifests: dict[float, list[dict]] = {ratio: [] for ratio in ratios}
    for output in outputs.values():
        output.mkdir(parents=True, exist_ok=True)

    max_generated = int(round(TARGET_POINTS * MAX_GENERATED_RATIO))
    for position, frame in enumerate(frames, start=1):
        baseline, observed_valid, _ = baseline_and_observed(line, frame)
        observed_source = prep.read_bin(Path(cfg["observed"]) / f"{frame}.bin")
        generated = recreate_e1_generated(line, method, frame, observed_source.shape[0])
        generated_mask, generated_rect_all = prep.fov_valid_mask(generated, frame)
        generated_valid = generated[generated_mask]
        generated_rect = generated_rect_all[generated_mask]
        generated_rng = np.random.default_rng(
            stable_seed(BASE_SEED, line, variant, frame, "nested_generated_candidates")
        )
        generated_order, generated_stats = select_category(
            generated_valid,
            generated_rect,
            max_generated,
            generated_rng,
        )
        observed_order = slot_order(line, frame)

        for ratio in ratios:
            policy = POLICY_BY_RATIO[ratio]
            generated_target = int(round(TARGET_POINTS * ratio))
            observed_target = TARGET_POINTS - generated_target
            observed_core = baseline[observed_order[:observed_target]]
            generated_core = generated_order[:generated_target]
            final = final_shuffle(
                np.concatenate((observed_core, generated_core), axis=0),
                line,
                frame,
                policy,
            )
            if final.shape != (TARGET_POINTS, 4) or not np.isfinite(final).all():
                raise RuntimeError(f"{policy}/{variant}/{frame}: invalid final {final.shape}")
            final.tofile(outputs[ratio] / f"{frame}.bin")
            manifests[ratio].append(
                {
                    "experiment": "detector_recovery_fine_ratio_v2",
                    "policy": policy,
                    "line": line,
                    "method": method,
                    "variant": variant,
                    "frame_id": frame,
                    "target_points": TARGET_POINTS,
                    "strict_x4_upstream_unchanged": True,
                    "baseline_detector_input": str(
                        Path(cfg["baseline_input"]) / f"{frame}.bin"
                    ),
                    "baseline_observed_slots": TARGET_POINTS,
                    "observed_selected_points": observed_target,
                    "generated_selected_points": generated_target,
                    "observed_selected_fraction": observed_target / TARGET_POINTS,
                    "generated_selected_fraction": generated_target / TARGET_POINTS,
                    "nested_observed_slot_order_common_within_line": True,
                    "nested_generated_order_common_across_ratios": True,
                    "generated_fov_points": int(generated_valid.shape[0]),
                    "generated_voxel_unique_points_at_g10": generated_stats[
                        "candidate_voxel_unique_points"
                    ],
                    "generated_fill_policy_at_g10": generated_stats["fill_policy"],
                    "observed_source_fov_points": int(observed_valid.shape[0]),
                    "uses_labels_boxes_detector_or_ap": False,
                    "status": "WRITTEN",
                }
            )

        if position == 1 or position % 64 == 0 or position == len(frames):
            print(f"FINE_RATIO_INPUT {variant}: {position}/{len(frames)}", flush=True)

    for ratio in ratios:
        write_csv(
            workspace / "manifests" / POLICY_BY_RATIO[ratio] / f"{variant}.csv",
            manifests[ratio],
        )


def prepare_observed_fill_control(
    line: str,
    frames: list[str],
    workspace: Path,
    ratio: float,
) -> None:
    cfg = LINES[line]
    variant = str(cfg["control_variant"])
    policy = CONTROL_POLICY_BY_RATIO[ratio]
    output = workspace / "inputs" / policy / variant
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    fill_target = int(round(TARGET_POINTS * ratio))
    observed_target = TARGET_POINTS - fill_target

    for position, frame in enumerate(frames, start=1):
        baseline, observed_valid, observed_rect = baseline_and_observed(line, frame)
        observed_order = slot_order(line, frame)
        observed_core = baseline[observed_order[:observed_target]]
        fill_rng = np.random.default_rng(
            stable_seed(BASE_SEED, line, frame, "observed_fill_control")
        )
        observed_fill, fill_stats = select_category(
            observed_valid,
            observed_rect,
            fill_target,
            fill_rng,
        )
        final = final_shuffle(
            np.concatenate((observed_core, observed_fill), axis=0),
            line,
            frame,
            policy,
        )
        if final.shape != (TARGET_POINTS, 4) or not np.isfinite(final).all():
            raise RuntimeError(f"{policy}/{variant}/{frame}: invalid final {final.shape}")
        final.tofile(output / f"{frame}.bin")
        rows.append(
            {
                "experiment": "detector_recovery_fine_ratio_v2",
                "policy": policy,
                "line": line,
                "method": "observed_fill_control",
                "variant": variant,
                "frame_id": frame,
                "target_points": TARGET_POINTS,
                "baseline_core_points": observed_target,
                "observed_fill_points": fill_target,
                "generated_points": 0,
                "strict_x4_upstream_used": False,
                "purpose": "separate slot replacement from generated-geometry effect",
                "observed_fill_candidate_points": fill_stats["candidate_points"],
                "observed_fill_voxel_unique_points": fill_stats[
                    "candidate_voxel_unique_points"
                ],
                "observed_fill_policy": fill_stats["fill_policy"],
                "uses_labels_boxes_detector_or_ap": False,
                "status": "WRITTEN",
            }
        )
        if position == 1 or position % 64 == 0 or position == len(frames):
            print(f"OBSERVED_FILL_CONTROL {variant}: {position}/{len(frames)}", flush=True)

    write_csv(workspace / "manifests" / policy / f"{variant}.csv", rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--ratio", type=float, action="append")
    parser.add_argument("--control-ratio", type=float, action="append")
    parser.add_argument("--variant", action="append")
    parser.add_argument("--skip-controls", action="store_true")
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    if workspace in (SOURCE_WORKSPACE.resolve(),):
        raise ValueError("refusing to write fine-ratio inputs into the source workspace")
    frames = read_frames(args.split_file)
    ratios = tuple(validate_ratio(value) for value in (args.ratio or RATIOS))
    control_ratios = tuple(
        validate_ratio(value)
        for value in (args.control_ratio or tuple(CONTROL_POLICY_BY_RATIO))
    )
    unsupported_controls = set(control_ratios) - set(CONTROL_POLICY_BY_RATIO)
    if unsupported_controls:
        raise ValueError(
            f"observed-fill controls are not named for ratios {sorted(unsupported_controls)}"
        )
    selected = set(args.variant or [])
    known_variants = {
        f"{cfg['variant_prefix']}_{method}" for cfg in LINES.values() for method in METHODS
    }
    unknown = selected - known_variants
    if unknown:
        raise ValueError(f"unknown variants: {sorted(unknown)}")

    workspace.mkdir(parents=True, exist_ok=True)
    selected_lines: set[str] = set()
    for line, cfg in LINES.items():
        for method in METHODS:
            variant = f"{cfg['variant_prefix']}_{method}"
            if selected and variant not in selected:
                continue
            selected_lines.add(line)
            if not args.controls_only:
                prepare_method_variant(line, method, frames, ratios, workspace)
    if not args.skip_controls:
        for line in sorted(selected_lines or LINES):
            for ratio in control_ratios:
                prepare_observed_fill_control(line, frames, workspace, ratio)

    print(
        f"FINE_RATIO_INPUTS_PASS workspace={workspace} frames={len(frames)} "
        f"policies={','.join(POLICY_BY_RATIO[ratio] for ratio in ratios)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
