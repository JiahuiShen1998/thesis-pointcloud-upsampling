#!/usr/bin/env python3
"""Evaluate a paired, point-multiset-preserving CenterPoint order adapter.

The old E1 cache is bypassed because at least one PU-Net file was found to
return inconsistent bytes on repeated reads.  Both tracks are reconstructed
from the stable audited upstream exact-4N prediction with the original
20260718 deterministic 3N selection:

    reconstructed_e1: original deterministic final random permutation
    observed_first:   all N observed rows, then the identical selected 3N rows

The paired files therefore contain the exact same float32 row multiset.  Point
count, coordinates, intensities, detector config, checkpoint, and evaluation
split are unchanged.  Only row order differs, removing the accidental
dependence of spconv's first-seen point/voxel caps on E1's final permutation.
No labels, boxes, detector outputs, or AP values are used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
OPENPCDET = REPO / "external/OpenPCDet"
TOOLS = OPENPCDET / "tools"
PYTHON = Path("/home/ra87racy/miniconda3/envs/openpcdet_centerpoint/bin/python")
CONFIG = TOOLS / "cfgs/kitti_models/centerpoint.yaml"
CHECKPOINT = (
    REPO
    / "external/centerpoint_hpc_bundle_20260717"
    / "outputs/checkpoints/CenterPoint_KITTI_original_80epoch/checkpoint_epoch_80.pth"
)
KITTI_TRAINING = REPO / "data/KITTI/object/training"
VAL_SPLIT = OPENPCDET / "data/kitti/ImageSets/val.txt"
VAL_INFO = OPENPCDET / "data/kitti/kitti_infos_val.pkl"
PREDICTED_ROOT = REPO / "results/kitti_unified_x4_current_methods_no_detector"
RESULT_ROOT = REPO / "results/centerpoint_exact4n_reconstructed_order_safe_20260729"
EXPECTED_VAL_FRAMES = 3769
E1_BASE_SEED = 20260718
DOWNSAMPLED_BASE_SEED = 20260702


@dataclass(frozen=True)
class Variant:
    name: str
    line: str
    method: str
    observed_source: Path
    predicted_source: Path


VARIANTS = (
    *tuple(
        Variant(
            f"original_x4_{method}",
            "line_a",
            method,
            KITTI_TRAINING / "velodyne_original",
            PREDICTED_ROOT / "line_a_original_x4_up" / method / "final_bin",
        )
        for method in ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
    ),
    *tuple(
        Variant(
            f"downsampled_x4_{method}",
            "line_b",
            method,
            KITTI_TRAINING / "velodyne_original",
            PREDICTED_ROOT / "line_b_downsampled_x4_up" / method / "final_bin",
        )
        for method in ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", type=int, metavar="FRAMES")
    mode.add_argument("--full", action="store_true")
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=("reconstructed_e1", "observed_first"),
        default=("reconstructed_e1", "observed_first"),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[item.name for item in VARIANTS],
        default=None,
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_bin(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4:
        raise ValueError(f"{path} is not an Nx4 float32 point cloud")
    points = values.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN/Inf")
    return points


def read_bin_stable(path: Path, attempts: int = 4) -> np.ndarray:
    previous_digest: str | None = None
    previous_payload: bytes | None = None
    last_reason = "no read attempted"
    for _ in range(attempts):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) % 16:
            last_reason = f"byte size {len(payload)} is not divisible by 16"
        else:
            points = np.frombuffer(payload, dtype=np.float32).reshape(-1, 4)
            if not np.isfinite(points).all():
                last_reason = "contains NaN/Inf"
            elif digest == previous_digest and payload == previous_payload:
                return points.copy()
            else:
                last_reason = "two consecutive reads were not byte-identical"
        previous_digest = digest
        previous_payload = payload
    raise ValueError(f"{path} is not stable after {attempts} reads: {last_reason}")


def write_bin_verified(path: Path, points: np.ndarray, attempts: int = 3) -> None:
    payload = np.ascontiguousarray(points, dtype=np.float32).tobytes()
    expected = hashlib.sha256(payload).hexdigest()
    temporary = path.with_name(f".{path.name}.writing")
    for _ in range(attempts):
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        first = temporary.read_bytes()
        second = temporary.read_bytes()
        if (
            hashlib.sha256(first).hexdigest() == expected
            and hashlib.sha256(second).hexdigest() == expected
            and first == second
        ):
            os.replace(temporary, path)
            return
    raise OSError(f"failed to write and verify stable float32 bytes for {path}")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def load_observed(variant: Variant, frame: str) -> tuple[np.ndarray, str]:
    source_path = variant.observed_source / f"{frame}.bin"
    original = read_bin_stable(source_path)
    if variant.line == "line_a":
        return original, "native_original"
    target = original.shape[0] // 4
    rng = np.random.default_rng(DOWNSAMPLED_BASE_SEED + int(frame))
    indices = rng.choice(original.shape[0], size=target, replace=False)
    indices.sort()
    return (
        original[indices].astype(np.float32, copy=False),
        "reconstructed_floor_n_div_4_seed_20260702_plus_frame",
    )


def val_ids_and_infos() -> tuple[list[str], list[dict]]:
    ids = [line.strip() for line in VAL_SPLIT.read_text().splitlines() if line.strip()]
    with VAL_INFO.open("rb") as handle:
        infos = pickle.load(handle)
    if len(ids) != EXPECTED_VAL_FRAMES or len(infos) != EXPECTED_VAL_FRAMES:
        raise RuntimeError(f"invalid val inputs: {len(ids)} ids and {len(infos)} infos")
    info_ids = [str(item["point_cloud"]["lidar_idx"]) for item in infos]
    if info_ids != ids:
        raise RuntimeError("val.txt and kitti_infos_val.pkl orders differ")
    return ids, infos


def selected_positions(count: int | None) -> np.ndarray:
    if count is None:
        return np.arange(EXPECTED_VAL_FRAMES, dtype=np.int64)
    if not 8 <= count <= EXPECTED_VAL_FRAMES:
        raise ValueError("--pilot must be between 8 and 3769")
    return np.linspace(
        0, EXPECTED_VAL_FRAMES - 1, num=count, dtype=np.int64
    )


def prepare_variant(
    variant: Variant,
    frame_ids: list[str],
    track: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = RESULT_ROOT / "manifests" / track / f"{variant.name}.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle))
        is_current_observed_manifest = (
            track != "observed_first"
            or all(
                "paired_control_path" in row
                for row in existing_rows
            )
        )
        if (
            [row["frame_id"] for row in existing_rows] == frame_ids
            and is_current_observed_manifest
        ):
            print(
                f"PREP_SKIP_COMPLETE {track}/{variant.name}: {len(frame_ids)}",
                flush=True,
            )
            return
    rows: list[dict[str, object]] = []
    for position, frame in enumerate(frame_ids, start=1):
        observed_path = variant.observed_source / f"{frame}.bin"
        output_path = output_dir / f"{frame}.bin"
        observed, observed_rule = load_observed(variant, frame)
        expected_bytes = 4 * observed.shape[0] * 16
        expected_points = 4 * observed.shape[0]
        seed = stable_seed(E1_BASE_SEED, "e1", variant.name, frame)
        rng = np.random.default_rng(seed)
        generated_indices = rng.choice(
            expected_points,
            size=3 * observed.shape[0],
            replace=False,
        )
        if track == "reconstructed_e1":
            source_path = variant.predicted_source / f"{frame}.bin"
            predicted = read_bin(source_path)
            if predicted.shape[0] != expected_points:
                raise ValueError(
                    f"{variant.name}/{frame}: upstream prediction has "
                    f"{predicted.shape[0]} rows, expected {expected_points}"
                )
            intended = np.concatenate(
                (observed, predicted[generated_indices]), axis=0
            )
            final = intended[rng.permutation(intended.shape[0])]
        elif track == "observed_first":
            legacy_control_path = (
                RESULT_ROOT
                / "inputs/reconstructed_e1"
                / variant.name
                / f"{frame}.bin"
            )
            control_path = (
                RESULT_ROOT
                / "inputs/paired_control_verified"
                / variant.name
                / f"{frame}.bin"
            )
            control_path.parent.mkdir(parents=True, exist_ok=True)
            permutation = rng.permutation(expected_points)
            inverse = np.empty_like(permutation)
            inverse[permutation] = np.arange(expected_points, dtype=np.int64)
            try:
                source_path = variant.predicted_source / f"{frame}.bin"
                predicted = read_bin_stable(source_path)
                if predicted.shape[0] != expected_points:
                    raise ValueError(
                        f"stable upstream has {predicted.shape[0]} rows, "
                        f"expected {expected_points}"
                    )
                final = np.concatenate(
                    (observed, predicted[generated_indices]), axis=0
                )
                repaired_observed_rows = 0
                paired_source_mode = "stable_upstream_reconstruction"
            except (OSError, ValueError):
                source_path = legacy_control_path
                control = read_bin_stable(source_path)
                if control.shape[0] != expected_points:
                    raise ValueError(
                        f"{variant.name}/{frame}: stable paired control has "
                        f"{control.shape[0]} rows, expected {expected_points}"
                    )
                final = control[inverse].copy()
                mismatch = np.any(
                    final[: observed.shape[0]].view(np.uint32)
                    != observed.view(np.uint32),
                    axis=1,
                )
                repaired_observed_rows = int(mismatch.sum())
                final[: observed.shape[0]] = observed
                paired_source_mode = "stable_control_fallback_observed_repaired"
            paired_control = final[permutation]
            write_bin_verified(control_path, paired_control)
            if not np.array_equal(
                paired_control[inverse][: observed.shape[0]], observed
            ):
                raise RuntimeError(
                    f"{variant.name}/{frame}: paired control observed repair failed"
                )
        else:
            raise ValueError(f"unsupported preparation track {track}")
        status = "REUSED"
        if (
            track == "observed_first"
            or variant.line == "line_b"
            or not output_path.is_file()
            or output_path.stat().st_size != expected_bytes
        ):
            if track == "observed_first":
                write_bin_verified(output_path, final)
            else:
                final.tofile(output_path)
            status = "WRITTEN"
        rows.append(
            {
                "variant": variant.name,
                "line": variant.line,
                "method": variant.method,
                "frame_id": frame,
                "observed_points": int(observed.shape[0]),
                "observed_rule": observed_rule,
                "generated_points": int(3 * observed.shape[0]),
                "final_points": int(4 * observed.shape[0]),
                "track": track,
                "seed": seed,
                "generated_selection": "deterministic_without_replacement_from_stable_upstream_exact4n",
                "same_intended_float32_multiset_for_both_tracks": True,
                "observed_prefix_exact": track == "observed_first",
                "paired_control_observed_rows_repaired": (
                    repaired_observed_rows if track == "observed_first" else 0
                ),
                "paired_source_mode": (
                    paired_source_mode if track == "observed_first" else ""
                ),
                "paired_control_path": (
                    str(control_path) if track == "observed_first" else ""
                ),
                "uses_labels_boxes_detector_or_ap": False,
                "status": status,
                "source": str(source_path),
                "output": str(output_path),
            }
        )
        if position == 1 or position % 64 == 0 or position == len(frame_ids):
            print(
                f"PREP {variant.name}: {position}/{len(frame_ids)} {status}",
                flush=True,
            )
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"refusing to retarget {link}: {link.resolve()} != {target}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace non-symlink {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def prepare_overlay(
    variant: Variant,
    track: str,
    run_kind: str,
    source: Path,
    infos: list[dict],
) -> Path:
    overlay = RESULT_ROOT / "runs" / track / variant.name / run_kind / "data_overlay"
    training = overlay / "training"
    training.mkdir(parents=True, exist_ok=True)
    ensure_symlink(overlay / "ImageSets", OPENPCDET / "data/kitti/ImageSets")
    ensure_symlink(training / "calib", KITTI_TRAINING / "calib")
    ensure_symlink(training / "image_2", KITTI_TRAINING / "image_2")
    ensure_symlink(training / "label_2", KITTI_TRAINING / "label_2")
    ensure_symlink(training / "velodyne", source)
    info_path = overlay / "kitti_infos_val.pkl"
    if not info_path.exists():
        with info_path.open("wb") as handle:
            pickle.dump(infos, handle)
    return overlay


def evaluation_done(run_root: Path) -> bool:
    logs = sorted((run_root / "openpcdet_output").rglob("log_eval_*.txt"))
    result_pkls = sorted((run_root / "openpcdet_output").rglob("result.pkl"))
    return bool(result_pkls) and any(
        "Evaluation done." in item.read_text(errors="replace") for item in logs
    )


def run_eval(
    variant: Variant,
    track: str,
    run_kind: str,
    overlay: Path,
    batch_size: int,
    workers: int,
    resume: bool,
) -> None:
    run_root = RESULT_ROOT / "runs" / track / variant.name / run_kind
    output = run_root / "openpcdet_output"
    output.mkdir(parents=True, exist_ok=True)
    if resume and evaluation_done(run_root):
        print(f"SKIP {track}/{variant.name}/{run_kind}", flush=True)
        return
    tag = f"exact4n_{track}_{run_kind}_{variant.name}"
    output_link = OPENPCDET / "output/kitti_models/centerpoint" / tag
    output_link.parent.mkdir(parents=True, exist_ok=True)
    ensure_symlink(output_link, output)
    command = [
        str(PYTHON),
        "test.py",
        "--cfg_file",
        "cfgs/kitti_models/centerpoint.yaml",
        "--batch_size",
        str(batch_size),
        "--workers",
        str(workers),
        "--ckpt",
        str(CHECKPOINT),
        "--extra_tag",
        tag,
        "--eval_tag",
        "frozen_exact4n_observed_first",
        "--set",
        "DATA_CONFIG.DATA_PATH",
        str(overlay.resolve()),
    ]
    (run_root / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"START {track}/{variant.name}/{run_kind}", flush=True)
    start = time.monotonic()
    with (run_root / "runner_stdout.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=TOOLS,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    runtime = time.monotonic() - start
    completed = evaluation_done(run_root)
    effective_returncode = 0 if completed else result.returncode
    (run_root / "raw_returncode.txt").write_text(f"{result.returncode}\n")
    (run_root / "returncode.txt").write_text(f"{effective_returncode}\n")
    (run_root / "runtime_seconds.txt").write_text(f"{runtime:.3f}\n")
    print(
        f"END {track}/{variant.name}/{run_kind} raw_rc={result.returncode} "
        f"effective_rc={effective_returncode} "
        f"runtime={runtime:.1f}s",
        flush=True,
    )
    if not completed:
        tail = "\n".join(
            (run_root / "runner_stdout.log")
            .read_text(errors="replace")
            .splitlines()[-60:]
        )
        print(tail, file=sys.stderr)
        raise RuntimeError(f"CenterPoint failed for {track}/{variant.name}")


def parse_standard_r40(log_text: str, class_name: str) -> tuple[float, float, float]:
    import re

    iou = "0.70, 0.70, 0.70" if class_name == "Car" else "0.50, 0.50, 0.50"
    pattern = (
        rf"{class_name} AP_R40@{iou}:\s*"
        rf"bbox AP:[^\n]+\s*bev\s+AP:[^\n]+\s*"
        rf"3d\s+AP:([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"
    )
    matches = re.findall(pattern, log_text)
    if not matches:
        raise ValueError(f"missing standard AP_R40 for {class_name}")
    return tuple(float(value) for value in matches[-1])


def write_summary(selected: list[Variant], tracks: list[str], run_kind: str) -> None:
    rows: list[dict[str, object]] = []
    for track in tracks:
        for variant in selected:
            run_root = RESULT_ROOT / "runs" / track / variant.name / run_kind
            logs = sorted((run_root / "openpcdet_output").rglob("log_eval_*.txt"))
            row: dict[str, object] = {
                "track": track,
                "variant": variant.name,
                "line": variant.line,
                "method": variant.method,
                "status": "PASS" if evaluation_done(run_root) else "FAIL",
            }
            if logs and row["status"] == "PASS":
                text = logs[-1].read_text(errors="replace")
                for class_name in ("Car", "Pedestrian", "Cyclist"):
                    easy, moderate, hard = parse_standard_r40(text, class_name)
                    prefix = class_name.lower()
                    row[f"{prefix}_3d_ap_r40_easy"] = easy
                    row[f"{prefix}_3d_ap_r40_moderate"] = moderate
                    row[f"{prefix}_3d_ap_r40_hard"] = hard
            rows.append(row)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path = RESULT_ROOT / f"{run_kind}_ap_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"WROTE {path}", flush=True)


def main() -> int:
    args = parse_args()
    if args.prepare_only and args.eval_only:
        raise ValueError("--prepare-only and --eval-only are mutually exclusive")
    selected_names = set(args.variants or [item.name for item in VARIANTS])
    selected = [item for item in VARIANTS if item.name in selected_names]
    ids, all_infos = val_ids_and_infos()
    positions = selected_positions(None if args.full else args.pilot)
    frame_ids = [ids[int(index)] for index in positions]
    infos = [all_infos[int(index)] for index in positions]
    run_kind = "full" if args.full else f"pilot{args.pilot}"

    metadata = {
        "protocol": "reconstructed_exact4n_paired_order_only_v2",
        "run_kind": run_kind,
        "frames": len(frame_ids),
        "selection": "full_val" if args.full else "linspace_over_ordered_val",
        "frame_ids": frame_ids,
        "tracks": args.tracks,
        "variants": [item.name for item in selected],
        "checkpoint": str(CHECKPOINT),
        "openpcdet_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OPENPCDET, text=True
        ).strip(),
        "changes_point_count": False,
        "candidate_source": "stable audited upstream exact4n prediction",
        "candidate_selection_seed": E1_BASE_SEED,
        "line_b_observed_rule": (
            "reconstructed_from_native_original_with_floor_N_div_4, "
            "seed=20260702+frame_id, sorted_indices"
        ),
        "line_b_bypasses_corrupt_downsampled_cache": True,
        "paired_tracks_share_exact_float32_row_multiset": True,
        "reconstructed_e1_order": "original deterministic final permutation",
        "observed_first_order": "native observed then selected generated",
        "observed_first_source": (
            "exact inverse permutation of the paired reconstructed_e1 file"
        ),
        "bypasses_corrupt_size_only_e1_cache": True,
        "changes_detector_config_or_checkpoint": False,
        "uses_labels_boxes_detector_or_ap": False,
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / f"{run_kind}_protocol.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    if not args.eval_only:
        for track in args.tracks:
            for variant in selected:
                prepare_variant(
                    variant,
                    frame_ids,
                    track,
                    RESULT_ROOT / "inputs" / track / variant.name,
                )
    if args.prepare_only:
        return 0

    for track in args.tracks:
        for variant in selected:
            source = (
                RESULT_ROOT / "inputs/paired_control_verified" / variant.name
                if track == "reconstructed_e1"
                else RESULT_ROOT / "inputs" / track / variant.name
            )
            overlay = prepare_overlay(variant, track, run_kind, source, infos)
            run_eval(
                variant,
                track,
                run_kind,
                overlay,
                args.batch_size,
                args.workers,
                args.resume,
            )
    write_summary(selected, list(args.tracks), run_kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
