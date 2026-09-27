#!/usr/bin/env python3
"""Run one patch-causal pilot variant with the frozen KITTI CenterPoint.

For an upsampled variant, the detector input order is fixed to observed-first:
all N observed points followed by a deterministic 3N sample from the strict-4N
method output.  The sample seed intentionally uses the historical variant token
so old- and new-patch conditions select identical row positions.
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
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
OPENPCDET = REPO / "external/OpenPCDet"
TOOLS = OPENPCDET / "tools"
PYTHON = Path("/home/ra87racy/miniconda3/envs/openpcdet_centerpoint/bin/python")
CHECKPOINT = (
    REPO
    / "external/centerpoint_hpc_bundle_20260717"
    / "outputs/checkpoints/CenterPoint_KITTI_original_80epoch/checkpoint_epoch_80.pth"
)
KITTI_TRAINING = REPO / "data/KITTI/object/training"
ORIGINAL = KITTI_TRAINING / "velodyne_original_val"
DOWNSAMPLED = (
    REPO
    / "results/kitti_unified_x4_current_methods_no_detector"
    / "downsampled_x4/velodyne_downsampled_x4_val"
)
VAL_INFO = OPENPCDET / "data/kitti/kitti_infos_val.pkl"
DEFAULT_PROTOCOL = (
    REPO
    / "results/centerpoint_exact4n_reconstructed_order_safe_20260729"
    / "pilot256_protocol.json"
)
DEFAULT_RESULT_ROOT = (
    REPO
    / "results/kitti_patch_punet_causal_ablation_v1_20260731"
    / "detectors/centerpoint"
)
E1_BASE_SEED = 20260718


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--line", choices=("line_a", "line_b"), required=True)
    parser.add_argument(
        "--source-mode",
        choices=("baseline", "upsampled_observed_first", "direct"),
        required=True,
    )
    parser.add_argument("--predicted-dir", type=Path)
    parser.add_argument(
        "--reference-token",
        help="Historical token used in the paired sampling seed, e.g. original_x4_pu_gcn",
    )
    parser.add_argument("--protocol-json", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.eval_only:
        parser.error("--prepare-only and --eval-only are mutually exclusive")
    if args.source_mode == "upsampled_observed_first":
        if args.predicted_dir is None or not args.reference_token:
            parser.error("upsampled_observed_first requires --predicted-dir and --reference-token")
    if args.source_mode == "direct" and args.predicted_dir is None:
        parser.error("direct requires --predicted-dir")
    return args


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def read_bin(path: Path) -> np.ndarray:
    last_error: Exception | None = None
    for _ in range(20):
        try:
            expected_bytes = path.stat().st_size
            values = np.fromfile(path, dtype=np.float32)
            if values.nbytes != expected_bytes:
                raise OSError(
                    f"short read: expected {expected_bytes} bytes, got {values.nbytes}"
                )
            if values.size == 0 or values.size % 4:
                raise ValueError(f"{path} is not non-empty KITTI float32 XYZI")
            points = values.reshape(-1, 4)
            if not np.isfinite(points).all():
                raise ValueError(f"{path} contains NaN/Inf")
            return points
        except (OSError, ValueError) as error:
            last_error = error
            time.sleep(0.1)
    raise OSError(f"failed to read {path} after NFS retries: {last_error}")


def protocol_frames(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = [str(item) for item in payload["frame_ids"]]
    if not frames or len(frames) != len(set(frames)):
        raise ValueError(f"invalid frame_ids in {path}")
    return frames


def observed_source(line: str) -> Path:
    return ORIGINAL if line == "line_a" else DOWNSAMPLED


def prepare_observed_first(
    frames: list[str],
    line: str,
    predicted_dir: Path,
    reference_token: str,
    output_dir: Path,
) -> Path:
    observed_dir = observed_source(line)
    predicted_dir = predicted_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, frame in enumerate(frames, start=1):
        observed = read_bin(observed_dir / f"{frame}.bin")
        predicted_path = predicted_dir / f"{frame}.bin"
        predicted = read_bin(predicted_path)
        expected = 4 * observed.shape[0]
        if predicted.shape[0] != expected:
            raise ValueError(
                f"{frame}: predicted rows={predicted.shape[0]}, expected strict-4N={expected}"
            )
        seed = stable_seed(E1_BASE_SEED, "e1", reference_token, frame)
        rng = np.random.default_rng(seed)
        selected = rng.choice(expected, size=3 * observed.shape[0], replace=False)
        final = np.concatenate((observed, predicted[selected]), axis=0).astype(
            np.float32, copy=False
        )
        output = output_dir / f"{frame}.bin"
        final.tofile(output)
        rows.append(
            {
                "frame_id": frame,
                "line": line,
                "reference_token": reference_token,
                "seed": seed,
                "observed_points": int(observed.shape[0]),
                "candidate_points": int(predicted.shape[0]),
                "selected_generated_points": int(selected.size),
                "detector_input_points": int(final.shape[0]),
                "observed_prefix_exact": bool(
                    np.array_equal(final[: observed.shape[0]], observed)
                ),
                "predicted_source": str(predicted_path.resolve()),
                "output": str(output.resolve()),
            }
        )
        if index == 1 or index % 64 == 0 or index == len(frames):
            print(f"PREP {index}/{len(frames)} {frame}", flush=True)
    manifest = output_dir.parent / "observed_first_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_dir


def selected_infos(frames: list[str]) -> list[dict]:
    with VAL_INFO.open("rb") as handle:
        infos = pickle.load(handle)
    by_id = {
        str(item["point_cloud"]["lidar_idx"]): item
        for item in infos
    }
    missing = [frame for frame in frames if frame not in by_id]
    if missing:
        raise KeyError(f"validation info missing frames: {missing[:10]}")
    return [by_id[frame] for frame in frames]


def ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"refusing to retarget {link}: {link.resolve()} != {target}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace non-symlink {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def prepare_overlay(run_root: Path, source: Path, infos: list[dict]) -> Path:
    overlay = run_root / "data_overlay"
    training = overlay / "training"
    training.mkdir(parents=True, exist_ok=True)
    ensure_symlink(overlay / "ImageSets", OPENPCDET / "data/kitti/ImageSets")
    ensure_symlink(training / "calib", KITTI_TRAINING / "calib")
    ensure_symlink(training / "image_2", KITTI_TRAINING / "image_2")
    ensure_symlink(training / "label_2", KITTI_TRAINING / "label_2")
    ensure_symlink(training / "velodyne", source)
    info_path = overlay / "kitti_infos_val.pkl"
    with info_path.open("wb") as handle:
        pickle.dump(infos, handle)
    return overlay


def evaluation_done(run_root: Path) -> bool:
    logs = sorted((run_root / "openpcdet_output").rglob("log_eval_*.txt"))
    results = sorted((run_root / "openpcdet_output").rglob("result.pkl"))
    return bool(results) and any(
        "Evaluation done." in path.read_text(errors="replace") for path in logs
    )


def run_eval(
    name: str,
    run_root: Path,
    overlay: Path,
    batch_size: int,
    workers: int,
    resume: bool,
    checkpoint: Path,
) -> None:
    output = run_root / "openpcdet_output"
    output.mkdir(parents=True, exist_ok=True)
    if resume and evaluation_done(run_root):
        print(f"SKIP completed {name}", flush=True)
        return
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    tag = f"patch_causal_pilot256_{safe_name}"
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
        str(checkpoint),
        "--extra_tag",
        tag,
        "--eval_tag",
        "frozen_patch_causal_pilot256",
        "--set",
        "DATA_CONFIG.DATA_PATH",
        str(overlay.resolve()),
    ]
    (run_root / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    started = time.monotonic()
    print(f"START CenterPoint {name}", flush=True)
    with (run_root / "runner_stdout.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=TOOLS,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    completed = evaluation_done(run_root)
    (run_root / "raw_returncode.txt").write_text(f"{result.returncode}\n")
    (run_root / "runtime_seconds.txt").write_text(f"{elapsed:.3f}\n")
    if not completed:
        tail = "\n".join(
            (run_root / "runner_stdout.log")
            .read_text(errors="replace")
            .splitlines()[-80:]
        )
        print(tail, file=sys.stderr)
        raise RuntimeError(f"CenterPoint did not complete for {name}")
    print(f"END CenterPoint {name} runtime={elapsed:.1f}s", flush=True)


def parse_triplet(text: str) -> tuple[float, float, float]:
    values = tuple(float(item.strip()) for item in text.split(","))
    if len(values) != 3:
        raise ValueError(f"expected AP triplet, got {text!r}")
    return values


def parse_metrics(log_text: str) -> dict[str, dict[str, dict[str, float]]]:
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for class_name in ("Car", "Pedestrian", "Cyclist"):
        iou = "0.70, 0.70, 0.70" if class_name == "Car" else "0.50, 0.50, 0.50"
        pattern = (
            rf"{class_name} AP_R40@{iou}:\s*"
            rf"bbox AP:[^\n]+\s*"
            rf"bev\s+AP:([^\n]+)\s*"
            rf"3d\s+AP:([^\n]+)"
        )
        matches = re.findall(pattern, log_text)
        if not matches:
            raise ValueError(f"missing standard AP_R40 block for {class_name}")
        bev = parse_triplet(matches[-1][0])
        d3 = parse_triplet(matches[-1][1])
        metrics[class_name] = {
            "bev_ap_r40": dict(zip(("easy", "moderate", "hard"), bev)),
            "3d_ap_r40": dict(zip(("easy", "moderate", "hard"), d3)),
        }
    return metrics


def write_summary(
    args: argparse.Namespace,
    run_root: Path,
    frames: list[str],
    source: Path,
) -> None:
    logs = sorted((run_root / "openpcdet_output").rglob("log_eval_*.txt"))
    if not logs:
        raise FileNotFoundError("CenterPoint evaluation log not found")
    metrics = parse_metrics(logs[-1].read_text(errors="replace"))
    payload = {
        "status": "PASS",
        "name": args.name,
        "line": args.line,
        "source_mode": args.source_mode,
        "source": str(source.resolve()),
        "protocol_json": str(args.protocol_json.resolve()),
        "frames": len(frames),
        "reference_token": args.reference_token,
        "checkpoint": str(args.checkpoint),
        "detector_config": "cfgs/kitti_models/centerpoint.yaml",
        "metrics_percent": metrics,
    }
    (run_root / "result_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, object]] = []
    for class_name, class_metrics in metrics.items():
        for metric, values in class_metrics.items():
            rows.append({"class": class_name, "metric": metric, **values})
    with (run_root / "parsed_ap_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"PASS {args.name} Car moderate 3D="
        f"{metrics['Car']['3d_ap_r40']['moderate']:.4f}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    args.protocol_json = args.protocol_json.resolve()
    args.result_root = args.result_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    frames = protocol_frames(args.protocol_json)
    run_root = args.result_root / args.name
    run_root.mkdir(parents=True, exist_ok=True)
    if args.source_mode == "baseline":
        source = observed_source(args.line).resolve()
    elif args.source_mode == "direct":
        source = args.predicted_dir.resolve()
        missing = [frame for frame in frames if not (source / f"{frame}.bin").is_file()]
        if missing:
            raise FileNotFoundError(
                f"direct source missing {len(missing)} protocol frames: {missing[:10]}"
            )
    else:
        staged = run_root / "observed_first_input"
        if args.eval_only:
            source = staged
        else:
            source = prepare_observed_first(
                frames,
                args.line,
                args.predicted_dir,
                args.reference_token,
                staged,
            )
    if args.prepare_only:
        return 0
    overlay = prepare_overlay(run_root, source, selected_infos(frames))
    run_eval(
        args.name,
        run_root,
        overlay,
        args.batch_size,
        args.workers,
        args.resume,
        args.checkpoint,
    )
    write_summary(args, run_root, frames, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
