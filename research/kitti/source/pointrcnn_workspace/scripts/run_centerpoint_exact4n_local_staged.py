#!/usr/bin/env python3
"""Run the paired exact-4N order experiment from verified local staging.

Large detector input files are staged one method at a time on local ext4.
Both tracks for a variant are constructed from the same in-memory `intended`
point array, evaluated, and then removed only after both result.pkl files and
KITTI "Evaluation done" markers exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import run_centerpoint_exact4n_observed_first as base  # noqa: E402


LOCAL_ROOT = Path("/tmp/centerpoint_exact4n_order_safe_stage_20260729")
TRACKS = ("reconstructed_e1", "observed_first")
MIN_FREE_BYTES = 58 * 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[item.name for item in base.VARIANTS],
        default=None,
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-local", action="store_true")
    return parser.parse_args()


def inverse_permutation(
    variant: base.Variant, frame: str, observed_n: int
) -> tuple[np.ndarray, np.ndarray, int]:
    expected = 4 * observed_n
    seed = base.stable_seed(base.E1_BASE_SEED, "e1", variant.name, frame)
    rng = np.random.default_rng(seed)
    selected = rng.choice(expected, size=3 * observed_n, replace=False)
    permutation = rng.permutation(expected)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(expected, dtype=np.int64)
    return selected, permutation, seed


def intended_points(
    variant: base.Variant,
    frame: str,
    observed: np.ndarray,
    selected: np.ndarray,
    permutation: np.ndarray,
) -> tuple[np.ndarray, str, int]:
    expected = 4 * observed.shape[0]
    upstream = variant.predicted_source / f"{frame}.bin"
    try:
        predicted = base.read_bin_stable(upstream)
        if predicted.shape[0] != expected:
            raise ValueError(
                f"stable upstream has {predicted.shape[0]} rows, expected {expected}"
            )
        intended = np.concatenate((observed, predicted[selected]), axis=0)
        return intended, "stable_upstream_reconstruction", 0
    except (OSError, ValueError):
        legacy = (
            base.RESULT_ROOT
            / "inputs/reconstructed_e1"
            / variant.name
            / f"{frame}.bin"
        )
        control = base.read_bin_stable(legacy)
        if control.shape[0] != expected:
            raise ValueError(
                f"{variant.name}/{frame}: legacy control has "
                f"{control.shape[0]} rows, expected {expected}"
            )
        inverse = np.empty_like(permutation)
        inverse[permutation] = np.arange(expected, dtype=np.int64)
        intended = control[inverse].copy()
        mismatch = np.any(
            intended[: observed.shape[0]].view(np.uint32)
            != observed.view(np.uint32),
            axis=1,
        )
        repaired = int(mismatch.sum())
        intended[: observed.shape[0]] = observed
        return intended, "stable_legacy_control_fallback", repaired


def local_write(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.writing")
    np.ascontiguousarray(points, dtype=np.float32).tofile(temporary)
    os.replace(temporary, path)


def prepare_local_variant(
    variant: base.Variant, frame_ids: list[str]
) -> tuple[Path, Path, list[dict[str, object]]]:
    variant_root = LOCAL_ROOT / variant.name
    if variant_root.exists():
        shutil.rmtree(variant_root)
    control_dir = variant_root / "reconstructed_e1"
    observed_first_dir = variant_root / "observed_first"
    control_dir.mkdir(parents=True)
    observed_first_dir.mkdir(parents=True)
    rows: list[dict[str, object]] = []

    for position, frame in enumerate(frame_ids, start=1):
        observed, observed_rule = base.load_observed(variant, frame)
        selected, permutation, seed = inverse_permutation(
            variant, frame, observed.shape[0]
        )
        intended, source_mode, repaired = intended_points(
            variant, frame, observed, selected, permutation
        )
        if (
            intended.shape != (4 * observed.shape[0], 4)
            or not np.isfinite(intended).all()
            or not np.array_equal(intended[: observed.shape[0]], observed)
        ):
            raise RuntimeError(
                f"{variant.name}/{frame}: invalid locally staged intended cloud"
            )
        control = intended[permutation]
        local_write(control_dir / f"{frame}.bin", control)
        local_write(observed_first_dir / f"{frame}.bin", intended)
        rows.append(
            {
                "variant": variant.name,
                "line": variant.line,
                "method": variant.method,
                "frame_id": frame,
                "observed_rule": observed_rule,
                "observed_points": int(observed.shape[0]),
                "generated_points": int(3 * observed.shape[0]),
                "final_points_each_track": int(4 * observed.shape[0]),
                "seed": seed,
                "paired_tracks_same_float32_multiset": True,
                "source_mode": source_mode,
                "fallback_observed_rows_repaired": repaired,
                "uses_labels_boxes_detector_or_ap": False,
                "control_path": str(control_dir / f"{frame}.bin"),
                "observed_first_path": str(observed_first_dir / f"{frame}.bin"),
            }
        )
        if position == 1 or position % 64 == 0 or position == len(frame_ids):
            print(
                f"LOCAL_PREP {variant.name}: {position}/{len(frame_ids)} "
                f"source={source_mode}",
                flush=True,
            )

    manifest = (
        base.RESULT_ROOT
        / "manifests/local_staged_verified"
        / f"{variant.name}.csv"
    )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return control_dir, observed_first_dir, rows


def prepare_full_overlay(
    variant: base.Variant,
    track: str,
    source: Path,
    infos: list[dict],
) -> Path:
    return base.prepare_overlay(variant, track, "full", source, infos)


def cleanup_local_variant(variant: base.Variant) -> None:
    target = LOCAL_ROOT / variant.name
    if target.parent != LOCAL_ROOT or not target.name:
        raise RuntimeError(f"refusing unsafe local cleanup target {target}")
    shutil.rmtree(target)
    print(f"LOCAL_CLEANUP {variant.name}: {target}", flush=True)


def reusable_local_variant(
    variant: base.Variant, frame_ids: list[str]
) -> tuple[Path, Path] | None:
    variant_root = LOCAL_ROOT / variant.name
    control_dir = variant_root / "reconstructed_e1"
    observed_first_dir = variant_root / "observed_first"
    if not control_dir.is_dir() or not observed_first_dir.is_dir():
        return None
    if all(
        (control_dir / f"{frame}.bin").is_file()
        and (observed_first_dir / f"{frame}.bin").is_file()
        for frame in frame_ids
    ):
        return control_dir, observed_first_dir
    return None


def main() -> int:
    args = parse_args()
    selected_names = set(
        args.variants or [item.name for item in base.VARIANTS]
    )
    selected = [
        item for item in base.VARIANTS if item.name in selected_names
    ]
    frame_ids, infos = base.val_ids_and_infos()
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": "local_ext4_staged_exact4n_paired_order_only_v3",
        "frame_count": len(frame_ids),
        "local_root": str(LOCAL_ROOT),
        "tracks": TRACKS,
        "variant_order": [item.name for item in selected],
        "paired_construction": (
            "same in-memory intended=[N observed,3N selected generated]; "
            "control=intended[fixed permutation], observed_first=intended"
        ),
        "source_acceptance": "two consecutive SHA256-identical finite reads",
        "fallback": "stable legacy control inverse permutation + exact observed repair",
        "line_b_observed_rule": (
            "floor(N/4), seed=20260702+frame_id, sorted indices"
        ),
        "detector_config_and_checkpoint_frozen": True,
        "uses_labels_boxes_detector_or_ap_for_input": False,
    }
    (base.RESULT_ROOT / "local_staged_full_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )

    for variant in selected:
        track_roots = {
            track: base.RESULT_ROOT / "runs" / track / variant.name / "full"
            for track in TRACKS
        }
        if args.resume and all(
            base.evaluation_done(root) for root in track_roots.values()
        ):
            print(f"LOCAL_SKIP_EVALUATED {variant.name}", flush=True)
            continue
        reusable = reusable_local_variant(variant, frame_ids) if args.resume else None
        if reusable is not None:
            control_dir, observed_first_dir = reusable
            print(f"LOCAL_REUSE_COMPLETE {variant.name}", flush=True)
        else:
            free = shutil.disk_usage(LOCAL_ROOT.parent).free
            if free < MIN_FREE_BYTES:
                raise OSError(
                    f"local staging needs at least "
                    f"{MIN_FREE_BYTES / 1024**3:.1f} GiB; "
                    f"only {free / 1024**3:.1f} GiB is free"
                )
            control_dir, observed_first_dir, _ = prepare_local_variant(
                variant, frame_ids
            )
        sources = {
            "reconstructed_e1": control_dir,
            "observed_first": observed_first_dir,
        }
        for track in TRACKS:
            overlay = prepare_full_overlay(
                variant, track, sources[track], infos
            )
            base.run_eval(
                variant,
                track,
                "full",
                overlay,
                args.batch_size,
                args.workers,
                args.resume,
            )
        if not all(
            base.evaluation_done(root) for root in track_roots.values()
        ):
            raise RuntimeError(
                f"{variant.name}: refusing local cleanup before both evals complete"
            )
        if not args.keep_local:
            cleanup_local_variant(variant)

    base.write_summary(selected, list(TRACKS), "full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
