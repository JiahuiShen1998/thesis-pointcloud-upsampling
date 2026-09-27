#!/usr/bin/env python3
"""Audit and evaluate the frozen CenterPoint checkpoint on exact-4N E1 inputs.

The matrix follows the final PointRCNN root-cause report:
  * original and downsampled-x4 native baselines;
  * four exact-4N methods on Line A and Line B;
  * CenterPoint's native frozen voxelizer, with no E2/E3 point adapter.

Each variant receives an isolated read-only KITTI overlay.  The canonical
PointRCNN KITTI directory is never renamed or switched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
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
BUNDLE = REPO / "external/centerpoint_hpc_bundle_20260717"
CHECKPOINT = (
    BUNDLE
    / "outputs/checkpoints/CenterPoint_KITTI_original_80epoch/checkpoint_epoch_80.pth"
)
KITTI_TRAINING = REPO / "data/KITTI/object/training"
VAL_SPLIT = OPENPCDET / "data/kitti/ImageSets/val.txt"
VAL_INFO = OPENPCDET / "data/kitti/kitti_infos_val.pkl"
E1_ROOT = (
    REPO
    / "results/kitti_unified_x4_input_preserving_e1_e2_20260718"
    / "inputs/e1_default_16384"
)
RAW_DOWNSAMPLED = (
    REPO
    / "results/kitti_unified_x4_current_methods_no_detector"
    / "downsampled_x4/velodyne_downsampled_x4_val"
)
RESULT_ROOT = REPO / "results/centerpoint_exact4n_e1_20260729"
EXPECTED_VAL_FRAMES = 3769
SAMPLE_COUNT = 32


@dataclass(frozen=True)
class Variant:
    name: str
    line: str
    method: str
    source: Path
    source_kind: str


VARIANTS = (
    Variant(
        "original_baseline",
        "baseline",
        "none",
        KITTI_TRAINING / "velodyne_original",
        "native_original",
    ),
    Variant(
        "downsampled_x4_baseline",
        "baseline",
        "none",
        RAW_DOWNSAMPLED,
        "native_downsampled_x4",
    ),
    *tuple(
        Variant(
            f"original_x4_{method}",
            "line_a",
            method,
            E1_ROOT / f"original_x4_{method}",
            "exact4n_e1",
        )
        for method in ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
    ),
    *tuple(
        Variant(
            f"downsampled_x4_{method}",
            "line_b",
            method,
            E1_ROOT / f"downsampled_x4_{method}",
            "exact4n_e1",
        )
        for method in ("pdans", "pu_gcn", "pu_edgeformer", "pu_net")
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[item.name for item in VARIANTS],
        default=None,
    )
    parser.add_argument("--smoke-frames", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def val_ids() -> list[str]:
    ids = [line.strip() for line in VAL_SPLIT.read_text().splitlines() if line.strip()]
    if len(ids) != EXPECTED_VAL_FRAMES or len(set(ids)) != EXPECTED_VAL_FRAMES:
        raise RuntimeError(
            f"invalid val split: {len(ids)} rows, {len(set(ids))} unique"
        )
    return ids


def audit_variant(variant: Variant, expected_ids: list[str]) -> dict[str, object]:
    source = variant.source
    expected = set(expected_ids)
    files = sorted(source.glob("*.bin")) if source.is_dir() else []
    stems = {path.stem for path in files}
    missing = sorted(expected - stems)
    extra = sorted(stems - expected)
    malformed = [str(path) for path in files if path.stat().st_size % 16 != 0]
    sampled = [source / f"{frame}.bin" for frame in expected_ids[::118]][:SAMPLE_COUNT]
    nonfinite: list[str] = []
    sampled_points: list[int] = []
    for path in sampled:
        if not path.is_file():
            continue
        points = np.fromfile(path, dtype=np.float32)
        if points.size % 4:
            malformed.append(str(path))
            continue
        points = points.reshape(-1, 4)
        sampled_points.append(int(points.shape[0]))
        if not np.isfinite(points).all():
            nonfinite.append(str(path))
    status = (
        "PASS"
        if source.is_dir() and not missing and not malformed and not nonfinite
        else "FAIL"
    )
    return {
        "variant": variant.name,
        "line": variant.line,
        "method": variant.method,
        "source_kind": variant.source_kind,
        "source": str(source),
        "source_exists": source.is_dir(),
        "bin_count": len(files),
        "val_missing_count": len(missing),
        "val_extra_count": len(extra),
        "malformed_count": len(set(malformed)),
        "sample_nonfinite_count": len(nonfinite),
        "sample_min_points": min(sampled_points) if sampled_points else 0,
        "sample_max_points": max(sampled_points) if sampled_points else 0,
        "status": status,
        "missing_sample": "|".join(missing[:5]),
        "extra_sample": "|".join(extra[:5]),
    }


def write_audit(rows: list[dict[str, object]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_ROOT / "input_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": "CenterPoint_native_voxelizer_on_exact4n_E1",
        "expected_val_frames": EXPECTED_VAL_FRAMES,
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "openpcdet_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=OPENPCDET, text=True
        ).strip(),
        "rows": rows,
    }
    (RESULT_ROOT / "input_audit.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Input audit: {payload['status']} ({len(rows)} variants)", flush=True)
    if payload["status"] != "PASS":
        raise RuntimeError("CenterPoint input audit failed")


def ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"refusing to retarget {link}: {link.resolve()} != {target}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing non-symlink: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def load_val_infos() -> list[dict]:
    with VAL_INFO.open("rb") as handle:
        infos = pickle.load(handle)
    if len(infos) != EXPECTED_VAL_FRAMES:
        raise RuntimeError(f"val info has {len(infos)} entries")
    return infos


def prepare_overlay(
    variant: Variant,
    run_kind: str,
    limit: int | None,
    all_infos: list[dict],
) -> Path:
    overlay = RESULT_ROOT / variant.name / f"data_overlay_{run_kind}"
    training = overlay / "training"
    training.mkdir(parents=True, exist_ok=True)
    ensure_symlink(overlay / "ImageSets", OPENPCDET / "data/kitti/ImageSets")
    ensure_symlink(training / "calib", KITTI_TRAINING / "calib")
    ensure_symlink(training / "image_2", KITTI_TRAINING / "image_2")
    ensure_symlink(training / "label_2", KITTI_TRAINING / "label_2")
    ensure_symlink(training / "velodyne", variant.source)
    info_path = overlay / "kitti_infos_val.pkl"
    if limit is None:
        ensure_symlink(info_path, VAL_INFO)
    else:
        chosen = all_infos[:limit]
        with info_path.open("wb") as handle:
            pickle.dump(chosen, handle)
    return overlay


def output_paths(variant: Variant, run_kind: str) -> tuple[str, Path, Path]:
    tag = f"exact4n_e1_{run_kind}_{variant.name}"
    variant_root = RESULT_ROOT / variant.name / run_kind
    variant_root.mkdir(parents=True, exist_ok=True)
    opencdet_output = variant_root / "openpcdet_output"
    opencdet_output.mkdir(parents=True, exist_ok=True)
    output_link = (
        OPENPCDET / "output/kitti_models/centerpoint" / tag
    )
    output_link.parent.mkdir(parents=True, exist_ok=True)
    ensure_symlink(output_link, opencdet_output)
    return tag, variant_root, opencdet_output


def evaluation_done(variant_root: Path) -> bool:
    rc_file = variant_root / "returncode.txt"
    logs = sorted((variant_root / "openpcdet_output").rglob("log_eval_*.txt"))
    return (
        rc_file.is_file()
        and rc_file.read_text().strip() == "0"
        and any("Evaluation done." in path.read_text(errors="replace") for path in logs)
    )


def run_variant(
    variant: Variant,
    run_kind: str,
    overlay: Path,
    batch_size: int,
    workers: int,
    resume: bool,
) -> None:
    tag, variant_root, _ = output_paths(variant, run_kind)
    if resume and evaluation_done(variant_root):
        print(f"SKIP completed {run_kind}/{variant.name}", flush=True)
        return
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
        "frozen_exact4n_e1",
        "--set",
        "DATA_CONFIG.DATA_PATH",
        str(overlay.resolve()),
    ]
    (variant_root / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    print(f"START {run_kind}/{variant.name}", flush=True)
    start = time.monotonic()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    log_path = variant_root / "runner_stdout.log"
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=TOOLS,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    runtime = time.monotonic() - start
    (variant_root / "returncode.txt").write_text(f"{result.returncode}\n")
    (variant_root / "runtime_seconds.txt").write_text(f"{runtime:.3f}\n")
    print(
        f"END {run_kind}/{variant.name} rc={result.returncode} runtime={runtime:.1f}s",
        flush=True,
    )
    if result.returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
        print(tail, file=sys.stderr)
        raise RuntimeError(f"CenterPoint failed for {variant.name}")


def parse_standard_r40(log_text: str, class_name: str) -> tuple[float, float, float]:
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


def rebuild_summary(selected: list[Variant], run_kind: str) -> None:
    rows: list[dict[str, object]] = []
    for variant in selected:
        root = RESULT_ROOT / variant.name / run_kind
        logs = sorted((root / "openpcdet_output").rglob("log_eval_*.txt"))
        row: dict[str, object] = {
            "variant": variant.name,
            "line": variant.line,
            "method": variant.method,
            "status": "PASS" if evaluation_done(root) else "FAIL",
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
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    path = RESULT_ROOT / f"{run_kind}_ap_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}", flush=True)


def main() -> int:
    args = parse_args()
    selected_names = set(args.variants or [item.name for item in VARIANTS])
    selected = [item for item in VARIANTS if item.name in selected_names]
    required = (PYTHON, CONFIG, CHECKPOINT, VAL_SPLIT, VAL_INFO)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing CenterPoint prerequisites: {missing}")
    ids = val_ids()
    rows = [audit_variant(item, ids) for item in selected]
    write_audit(rows)
    if args.audit_only:
        return 0
    all_infos = load_val_infos()
    run_kind = f"smoke{args.smoke_frames}" if args.smoke else "full"
    limit = args.smoke_frames if args.smoke else None
    if limit is not None and not 1 <= limit <= EXPECTED_VAL_FRAMES:
        raise ValueError("--smoke-frames must be between 1 and 3769")
    for variant in selected:
        overlay = prepare_overlay(variant, run_kind, limit, all_infos)
        run_variant(
            variant,
            run_kind,
            overlay,
            args.batch_size,
            args.workers,
            args.resume,
        )
    rebuild_summary(selected, run_kind)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
