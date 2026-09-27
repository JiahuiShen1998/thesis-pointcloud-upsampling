#!/usr/bin/env python3
"""Evaluate KITTI-format detections with OpenPCDet's optimized AP_R40 code.

This helper is intentionally detector-independent: it reads an explicit image
split, the matching KITTI labels and one directory of KITTI detection text
files.  It is useful for validating the slow legacy PointRCNN evaluator against
the maintained numba/CUDA implementation before adopting the faster backend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET_ROOT = PROJECT_ROOT / "external" / "OpenPCDet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--detection-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--class-name", default="Car")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-text", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_ids = [
        int(line.strip())
        for line in args.split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not frame_ids or len(frame_ids) != len(set(frame_ids)):
        raise RuntimeError("split must contain non-empty unique frame IDs")
    missing_labels = [idx for idx in frame_ids if not (args.label_dir / f"{idx:06d}.txt").is_file()]
    missing_detections = [idx for idx in frame_ids if not (args.detection_dir / f"{idx:06d}.txt").is_file()]
    if missing_labels or missing_detections:
        raise RuntimeError(
            f"coverage failure: missing_labels={missing_labels[:10]} "
            f"missing_detections={missing_detections[:10]}"
        )

    sys.path.insert(0, str(OPENPCDET_ROOT))
    from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common
    from pcdet.datasets.kitti.kitti_object_eval_python.eval import get_official_eval_result

    started = time.time()
    gt_annos = kitti_common.get_label_annos(str(args.label_dir), frame_ids)
    dt_annos = kitti_common.get_label_annos(str(args.detection_dir), frame_ids)
    result_text, result_dict = get_official_eval_result(gt_annos, dt_annos, [args.class_name])
    elapsed = time.time() - started
    payload = {
        "status": "PASS",
        "protocol": "OpenPCDet_KITTI_AP_R40",
        "frame_count": len(frame_ids),
        "class_name": args.class_name,
        "label_dir": str(args.label_dir.resolve()),
        "detection_dir": str(args.detection_dir.resolve()),
        "split_file": str(args.split_file.resolve()),
        "elapsed_seconds": elapsed,
        "metrics_percent": {key: float(value) for key, value in result_dict.items()},
    }
    if args.output_text:
        args.output_text.parent.mkdir(parents=True, exist_ok=True)
        args.output_text.write_text(result_text, encoding="utf-8")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output_json)
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
