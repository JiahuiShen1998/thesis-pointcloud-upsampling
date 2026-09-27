#!/usr/bin/env python3
"""Evaluate one PointRCNN checkpoint on an explicit KITTI split and LiDAR tree.

This is the full-split counterpart of the historical pilot256 evaluator.  It
does not retarget ``data/KITTI/object/training/velodyne``: the dataset loader is
patched in-process so each arm reads only the explicitly supplied directory.
The KITTI rotated-IoU evaluator is replaced by the repository's audited CPU
implementation, which avoids a second CUDA dependency during AP calculation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import logging
import os
import random
import runpy
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
FAST_AP_PYTHON = Path("/home/ra87racy/miniconda3/envs/openpcdet_centerpoint/bin/python")
FAST_AP_SCRIPT = PROJECT_ROOT / "scripts/evaluate_kitti_r40_openpcdet_fast.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--cleanup-heavy-artifacts",
        action="store_true",
        help="retain AP/audit evidence but remove reproducible detection intermediates after PASS",
    )
    return parser.parse_args()


def read_frames(path: Path):
    frames = [item.strip() for item in path.read_text(encoding="utf-8").splitlines() if item.strip()]
    if not frames or len(frames) != len(set(frames)):
        raise RuntimeError("split must contain non-empty unique frame IDs: %s" % path)
    return frames


def load_cpu_evaluator():
    path = PROJECT_ROOT / "scripts/run_pointrcnn_full_validation_pugcn_cap100k_ap_eval.py"
    spec = importlib.util.spec_from_file_location("pointrcnn_cpu_ap_patch", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load CPU AP evaluator: %s" % path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_evaluator()

    # The repository evaluator assumes ``num >= num_part`` and otherwise
    # returns many zero-sized partitions.  That is true for the 3,769-frame
    # validation split, but not for deterministic smoke/subset audits.  Limit
    # the partition count to the number of examples so both routes execute the
    # same AP implementation without trying to concatenate empty arrays.
    import tools.kitti_object_eval_python.eval as kitti_eval_module

    original_get_split_parts = kitti_eval_module.get_split_parts

    def safe_get_split_parts(num, num_part):
        return original_get_split_parts(num, min(num_part, max(1, num)))

    kitti_eval_module.get_split_parts = safe_get_split_parts

    # This legacy evaluator reduces its 41-point precision grid to AP_R11 by
    # default. Match CenterPoint's AP_R40: exclude recall=0 and average 1..40.
    def get_map_r40(precision):
        if precision.shape[-1] != 41:
            raise ValueError("KITTI AP_R40 requires a 41-entry precision grid")
        return np.mean(precision[..., 1:], axis=-1) * 100.0

    kitti_eval_module.get_mAP = get_map_r40

    # eval_rcnn.py evaluates KITTI AP before returning from inference.  Its
    # legacy pure-Python rotated-IoU fallback takes roughly an hour on the full
    # split, and this wrapper historically evaluated the same files a second
    # time afterwards.  Defer that internal call; below we run OpenPCDet's
    # numba/CUDA implementation once.  It was cross-checked on all 3,769
    # baseline frames and matches every AP_R40 field to floating-point noise.
    import tools.kitti_object_eval_python.evaluate as kitti_evaluate_module

    def deferred_kitti_evaluate(*_args, **_kwargs):
        return "KITTI AP deferred to audited OpenPCDet AP_R40 backend\n", {}

    kitti_evaluate_module.evaluate = deferred_kitti_evaluate
    return module


def result_metrics(ap_dict):
    return {
        "bbox_ap_r40": {
            "easy": float(ap_dict["Car_image_easy"]),
            "moderate": float(ap_dict["Car_image_moderate"]),
            "hard": float(ap_dict["Car_image_hard"]),
        },
        "bev_ap_r40": {
            "easy": float(ap_dict["Car_bev_easy"]),
            "moderate": float(ap_dict["Car_bev_moderate"]),
            "hard": float(ap_dict["Car_bev_hard"]),
        },
        "3d_ap_r40": {
            "easy": float(ap_dict["Car_3d_easy"]),
            "moderate": float(ap_dict["Car_3d_moderate"]),
            "hard": float(ap_dict["Car_3d_hard"]),
        },
    }


def result_metrics_fast(ap_dict):
    return {
        "bbox_ap_r40": {
            "easy": float(ap_dict["Car_image/easy_R40"]),
            "moderate": float(ap_dict["Car_image/moderate_R40"]),
            "hard": float(ap_dict["Car_image/hard_R40"]),
        },
        "bev_ap_r40": {
            "easy": float(ap_dict["Car_bev/easy_R40"]),
            "moderate": float(ap_dict["Car_bev/moderate_R40"]),
            "hard": float(ap_dict["Car_bev/hard_R40"]),
        },
        "3d_ap_r40": {
            "easy": float(ap_dict["Car_3d/easy_R40"]),
            "moderate": float(ap_dict["Car_3d/moderate_R40"]),
            "hard": float(ap_dict["Car_3d/hard_R40"]),
        },
    }


def main() -> None:
    args = parse_args()
    if args.workers != 0:
        raise ValueError("per-frame input auditing requires --workers 0")
    lidar_dir = args.lidar_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    split_file = args.split_file.resolve()
    output_dir = args.output_dir.resolve()
    marker = output_dir / "run_complete.json"
    frames = read_frames(split_file)

    # Serialize every operation on one evaluation directory, including resume
    # checks and cleanup.  Cleanup used to happen before taking the lock, which
    # allowed a second invocation to remove an in-flight evaluator's files.
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(str(output_dir).encode("utf-8")).hexdigest()[:16]
    lock_file = (Path("/tmp") / ("pointrcnn_eval_" + lock_key + ".lock")).open("w")
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)

    expected_marker = {
        "split_file": str(split_file),
        "split_name": args.split_name,
        "frame_count": len(frames),
        "lidar_dir": str(lidar_dir),
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "point_sampling": "numpy_seed_base_plus_frame_id_v1",
        "ap_protocol": "KITTI_AP_R40_41_precision_samples_exclude_recall_zero",
        "cleanup_heavy_artifacts": args.cleanup_heavy_artifacts,
    }
    if args.resume and marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("status") == "PASS" and all(
            previous.get(key) == value for key, value in expected_marker.items()
        ):
            print("POINT_RCNN_EVAL_SKIP_PASS %s" % output_dir, flush=True)
            return

    def remove_known_output_child(path: Path) -> None:
        if path.parent != output_dir:
            raise RuntimeError("refusing to clean path outside evaluation output: %s" % path)
        if path.is_symlink() or path.is_file():
            path.unlink()
            return

        # Detach the tree from its well-known name first.  The legacy evaluator
        # can finish a final asynchronous file operation just after runpy
        # returns; deleting in place can then fail with ENOTEMPTY.  An atomic
        # rename plus bounded retry makes cleanup reliable without ever
        # touching a path outside this evaluation directory.
        for generation in range(5):
            if not path.is_dir():
                return
            tombstone = output_dir / (
                ".cleanup_%s_%s" % (path.name, uuid.uuid4().hex)
            )
            path.rename(tombstone)
            for attempt in range(20):
                try:
                    shutil.rmtree(str(tombstone))
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    if attempt == 19:
                        raise
                    time.sleep(0.1)
        if path.exists():
            raise RuntimeError("cleanup target kept being recreated: %s" % path)

    if args.cleanup_heavy_artifacts:
        # Full inference cannot resume mid-frame sequence.  Clear only the two
        # known reproducible subtrees when recovering an incomplete attempt.
        remove_known_output_child(output_dir / "eval")
        remove_known_output_child(output_dir / "selected_detection_files")

    missing = []
    invalid = []
    for frame in frames:
        path = lidar_dir / (frame + ".bin")
        if not path.is_file():
            missing.append(frame)
        else:
            size = path.stat().st_size
            if size <= 0 or size % 16:
                invalid.append((frame, size))
    if missing or invalid:
        raise RuntimeError(
            "LiDAR coverage failure: expected=%d missing=%s invalid=%s"
            % (len(frames), missing[:10], invalid[:10])
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    sys.path.insert(0, str(TOOLS_DIR))
    sys.path.insert(0, str(PROJECT_ROOT))
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from lib.datasets.kitti_dataset import KittiDataset
    from lib.datasets.kitti_rcnn_dataset import KittiRCNNDataset

    original_dataset_init = KittiDataset.__init__

    def dataset_init(self, root_dir, split="train"):
        original_dataset_init(self, root_dir=root_dir, split=split)
        if split == args.split_name:
            self.image_idx_list = list(frames)
            self.num_sample = len(frames)
            self.lidar_dir = str(lidar_dir)

    KittiDataset.__init__ = dataset_init
    original_get_rpn_sample = KittiRCNNDataset.get_rpn_sample
    frame_input_audit = {}

    def get_rpn_sample(self, index):
        frame = "%06d" % int(self.sample_id_list[index])
        frame_seed = (args.seed + int(frame)) & 0xFFFFFFFF
        np.random.seed(frame_seed)
        sample = original_get_rpn_sample(self, index)
        selected_points = np.ascontiguousarray(sample["pts_input"], dtype=np.float32)
        if frame in frame_input_audit:
            raise RuntimeError("frame evaluated more than once: " + frame)
        frame_input_audit[frame] = {
            "seed": frame_seed,
            "shape": list(selected_points.shape),
            "float32_sha256": hashlib.sha256(selected_points.tobytes()).hexdigest(),
        }
        return sample

    KittiRCNNDataset.get_rpn_sample = get_rpn_sample
    cpu_eval_module = load_cpu_evaluator()

    sys.argv = [
        "eval_rcnn.py",
        "--cfg_file", "cfgs/default.yaml",
        "--eval_mode", "rcnn",
        "--ckpt", str(checkpoint),
        "--batch_size", str(args.batch_size),
        "--workers", str(args.workers),
        "--output_dir", str(output_dir),
        "--save_result",
        "--set", "RPN.LOC_XZ_FINE", "False", "TEST.SPLIT", args.split_name,
    ]
    old_argv = sys.argv[:]
    del old_argv  # document that eval_rcnn intentionally owns argv from here
    os.chdir(str(TOOLS_DIR))
    runpy.run_path("eval_rcnn.py", run_name="__main__")

    if set(frame_input_audit) != set(frames):
        raise RuntimeError(
            "actual detector input coverage mismatch: expected=%d consumed=%d"
            % (len(frames), len(frame_input_audit))
        )
    (output_dir / "detector_frame_input_audit.json").write_text(
        json.dumps(frame_input_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result_dirs = sorted(output_dir.rglob("final_result/data"), key=lambda path: path.stat().st_mtime)
    if not result_dirs:
        raise RuntimeError("evaluation completed without final_result/data")
    detection_dir = result_dirs[-1]
    missing_detections = [frame for frame in frames if not (detection_dir / (frame + ".txt")).is_file()]
    if missing_detections:
        raise RuntimeError("missing %d detection files: %s" % (len(missing_detections), missing_detections[:10]))

    # eval_rcnn.py pads every ID in data/KITTI/ImageSets/<split-name>.txt with
    # an empty detection file, even when the in-process dataset was narrowed by
    # an explicit split file. Expose an exact-ID view to the offline evaluator
    # so subset smoke tests and full validation follow the same code path.
    selected_detection_dir = output_dir / "selected_detection_files"
    selected_detection_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        source = detection_dir / (frame + ".txt")
        target = selected_detection_dir / (frame + ".txt")
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError("selected detection symlink points elsewhere: %s" % target)
        elif target.exists():
            raise RuntimeError("refusing to replace selected detection path: %s" % target)
        else:
            target.symlink_to(source)

    label_dir = PROJECT_ROOT / "data/KITTI/object/training/label_2"
    if not FAST_AP_PYTHON.is_file() or not FAST_AP_SCRIPT.is_file():
        raise FileNotFoundError("fast KITTI AP backend is unavailable")
    fast_json = output_dir / "fast_kitti_ap.json"
    fast_text = output_dir / "fast_kitti_ap.txt"
    fast_stderr = output_dir / "fast_kitti_ap.stderr.txt"
    completed = subprocess.run(
        [
            str(FAST_AP_PYTHON),
            str(FAST_AP_SCRIPT),
            "--label-dir", str(label_dir),
            "--detection-dir", str(selected_detection_dir),
            "--split-file", str(split_file),
            "--output-json", str(fast_json),
            "--output-text", str(fast_text),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    fast_stderr.write_text(completed.stderr, encoding="utf-8")
    fast_payload = json.loads(fast_json.read_text(encoding="utf-8"))
    if fast_payload.get("status") != "PASS" or fast_payload.get("frame_count") != len(frames):
        raise RuntimeError("fast KITTI AP coverage/protocol failure: %s" % fast_payload)
    ap_text = fast_text.read_text(encoding="utf-8")
    ap_dict = fast_payload["metrics_percent"]
    (output_dir / "raw_kitti_ap.txt").write_text(
        "KITTI AP_R40 (mean of precision samples 1..40; recall=0 excluded)\n" + ap_text,
        encoding="utf-8",
    )
    payload = {
        "status": "PASS",
        **expected_marker,
        "prediction_count": len(frames),
        "actual_consumed_frame_count": len(frame_input_audit),
        "sampler_safe_fallback_count": KittiRCNNDataset._sampler_safe_fallback_count,
        "sampler_safe_fallback_frames": KittiRCNNDataset._sampler_safe_fallback_frames,
        "raw_detection_dir": str(detection_dir),
        "detection_dir": str(selected_detection_dir),
        "metrics_percent": {"Car": result_metrics_fast(ap_dict)},
        "ap_backend": fast_payload["protocol"],
        "ap_elapsed_seconds": fast_payload["elapsed_seconds"],
        "ap_evaluator_source": str(FAST_AP_SCRIPT.resolve()),
        "legacy_inprocess_ap_deferred_source": str(Path(cpu_eval_module.__file__).resolve()),
        "heavy_artifacts_retained": not args.cleanup_heavy_artifacts,
    }
    inference_log = detection_dir.parent.parent / "log_eval_one.txt"
    if inference_log.is_file():
        shutil.copy2(str(inference_log), str(output_dir / "inference_log.txt"))
    if args.cleanup_heavy_artifacts:
        # Close the legacy logger's still-registered file handler before its
        # directory is detached and removed.
        logging.shutdown()
        remove_known_output_child(output_dir / "eval")
        remove_known_output_child(output_dir / "selected_detection_files")
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(marker)
    print(
        "POINT_RCNN_EVAL_PASS frames=%d car_3d_moderate=%.4f"
        % (len(frames), payload["metrics_percent"]["Car"]["3d_ap_r40"]["moderate"]),
        flush=True,
    )


if __name__ == "__main__":
    main()
