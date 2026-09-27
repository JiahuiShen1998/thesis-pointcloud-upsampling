#!/usr/bin/env python3
"""Export frozen CenterPoint heatmap top-K boxes before NMS.

CenterPoint is single-stage, so these decoded heatmap candidates are the
closest valid analogue to a two-stage detector's internal proposals.  This
script uses no labels or GT boxes and preserves the model's native K=500 and
score threshold=0.1 while deliberately stopping before NMS.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
OPENPCDET = REPO / "external/OpenPCDet"
TOOLS = OPENPCDET / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(OPENPCDET))

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file  # noqa: E402
from pcdet.datasets import build_dataloader  # noqa: E402
from pcdet.models import build_network, load_data_to_gpu  # noqa: E402
from pcdet.models.model_utils import centernet_utils  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402


CONFIG = TOOLS / "cfgs/kitti_models/centerpoint.yaml"
CHECKPOINT = REPO / "external/centerpoint_hpc_bundle_20260717/outputs/checkpoints/CenterPoint_KITTI_original_80epoch/checkpoint_epoch_80.pth"
TRAINING = REPO / "data/KITTI/object/training"
OBSERVED = TRAINING / "velodyne_original_val"
VAL_INFO = OPENPCDET / "data/kitti/kitti_infos_val.pkl"
DEFAULT_PROTOCOL = REPO / "results/centerpoint_exact4n_reconstructed_order_safe_20260729/pilot256_protocol.json"
DEFAULT_OUTPUT = REPO / "results/detector_aware_pdans_v3_20260805/internal_proposals/centerpoint_prenms"


def protocol_frames(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    frames = [str(item) for item in payload["frame_ids"]]
    if len(frames) != len(set(frames)):
        raise ValueError("duplicate frame IDs in protocol")
    return frames


def ensure_symlink(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"unexpected symlink target for {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace non-symlink: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def prepare_overlay(output: Path, frames: list[str]) -> Path:
    overlay = output / "data_overlay"
    training = overlay / "training"
    training.mkdir(parents=True, exist_ok=True)
    ensure_symlink(overlay / "ImageSets", OPENPCDET / "data/kitti/ImageSets")
    ensure_symlink(training / "calib", TRAINING / "calib")
    ensure_symlink(training / "image_2", TRAINING / "image_2")
    ensure_symlink(training / "label_2", TRAINING / "label_2")
    ensure_symlink(training / "velodyne", OBSERVED)
    with VAL_INFO.open("rb") as handle:
        infos = pickle.load(handle)
    by_id = {str(item["point_cloud"]["lidar_idx"]): item for item in infos}
    missing = [frame for frame in frames if frame not in by_id]
    if missing:
        raise KeyError(f"missing validation infos: {missing[:10]}")
    with (overlay / "kitti_infos_val.pkl").open("wb") as handle:
        pickle.dump([by_id[frame] for frame in frames], handle)
    return overlay


def decode_before_nms(model, batch_size: int) -> list[dict[str, torch.Tensor]]:
    dense_head = model.dense_head
    post = dense_head.model_cfg.POST_PROCESSING
    limit = torch.as_tensor(post.POST_CENTER_LIMIT_RANGE, device="cuda", dtype=torch.float32)
    output = [{"boxes": [], "scores": [], "labels": []} for _ in range(batch_size)]
    for head_index, pred in enumerate(dense_head.forward_ret_dict["pred_dicts"]):
        decoded = centernet_utils.decode_bbox_from_heatmap(
            heatmap=pred["hm"].sigmoid(),
            rot_cos=pred["rot"][:, 0].unsqueeze(1),
            rot_sin=pred["rot"][:, 1].unsqueeze(1),
            center=pred["center"], center_z=pred["center_z"], dim=pred["dim"].exp(),
            vel=pred.get("vel"), iou=None,
            point_cloud_range=dense_head.point_cloud_range,
            voxel_size=dense_head.voxel_size,
            feature_map_stride=dense_head.feature_map_stride,
            K=int(post.MAX_OBJ_PER_SAMPLE), circle_nms=False,
            score_thresh=float(post.SCORE_THRESH), post_center_limit_range=limit,
        )
        mapping = dense_head.class_id_mapping_each_head[head_index]
        for batch_index, item in enumerate(decoded):
            output[batch_index]["boxes"].append(item["pred_boxes"])
            output[batch_index]["scores"].append(item["pred_scores"])
            output[batch_index]["labels"].append(mapping[item["pred_labels"].long()] + 1)
    merged = []
    for item in output:
        merged.append({
            "boxes": torch.cat(item["boxes"], dim=0),
            "scores": torch.cat(item["scores"], dim=0),
            "labels": torch.cat(item["labels"], dim=0),
        })
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=20, help="0 means all protocol frames")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    proposal_dir = output / "npz"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    frames_all = protocol_frames(args.protocol.resolve())
    frames = frames_all if args.limit == 0 else frames_all[: args.limit]
    if args.resume and all((proposal_dir / f"{frame}.npz").is_file() for frame in frames):
        print(f"CENTER_PRENMS_REUSE frames={len(frames)}", flush=True)
        return 0

    overlay = prepare_overlay(output, frames)
    original_cwd = Path.cwd()
    os.chdir(TOOLS)
    try:
        cfg_from_yaml_file(str(CONFIG), cfg)
        cfg_from_list(["DATA_CONFIG.DATA_PATH", str(overlay)], cfg)
        logger = common_utils.create_logger(output / "export.log", rank=0)
        dataset, loader, unused_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size, dist=False, workers=args.workers,
            logger=logger, training=False,
        )
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
        model.load_params_from_file(filename=str(CHECKPOINT), logger=logger, to_cpu=True)
        model.cuda().eval()
        total = class_counts = 0
        counts = []
        with torch.no_grad():
            for batch_index, batch in enumerate(loader, start=1):
                frame_ids = [str(item) for item in batch["frame_id"]]
                load_data_to_gpu(batch)
                model(batch)
                decoded = decode_before_nms(model, len(frame_ids))
                for frame, item in zip(frame_ids, decoded):
                    boxes = item["boxes"].detach().cpu().numpy().astype(np.float32)
                    scores = item["scores"].detach().cpu().numpy().astype(np.float32)
                    labels = item["labels"].detach().cpu().numpy().astype(np.int16)
                    np.savez_compressed(proposal_dir / f"{frame}.npz", boxes_lidar=boxes,
                                        scores=scores, labels=labels)
                    counts.append(int(len(scores)))
                    total += len(scores)
                    class_counts += len(np.unique(labels))
                if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
                    print(f"CENTER_PRENMS {min(batch_index * args.batch_size, len(frames))}/{len(frames)}", flush=True)
    finally:
        os.chdir(original_cwd)

    summary = {
        "status": "PASS",
        "stage": "CenterPoint decoded heatmap top-K after score/range mask and before NMS",
        "uses_gt": False,
        "frames": len(frames),
        "checkpoint": str(CHECKPOINT),
        "config": str(CONFIG),
        "K_per_head": 500,
        "native_score_threshold": 0.1,
        "total_candidates": total,
        "mean_candidates_per_frame": float(np.mean(counts)),
        "min_candidates_per_frame": int(min(counts)),
        "max_candidates_per_frame": int(max(counts)),
        "proposal_dir": str(proposal_dir),
    }
    (output / "export_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
