#!/usr/bin/env python3
"""Create the KITTI train-info cache required by OpenPCDet."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPCDET = PROJECT_ROOT / "external/OpenPCDet"
TOOLS = OPENPCDET / "tools"


def validate_infos(infos, frame_ids):
    info_ids = [str(info["point_cloud"]["lidar_idx"]) for info in infos]
    if info_ids != frame_ids:
        raise RuntimeError(
            f"train-info IDs do not match split: infos={len(info_ids)} split={len(frame_ids)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    split_file = args.split_file.resolve()
    output = args.output.resolve()
    frame_ids = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    if len(frame_ids) != len(set(frame_ids)):
        raise RuntimeError(f"duplicate frame IDs in {split_file}")

    if output.is_file():
        with output.open("rb") as handle:
            infos = pickle.load(handle)
        validate_infos(infos, frame_ids)
        print(f"CENTERPOINT_TRAIN_INFOS_PASS cached=True frames={len(infos)} path={output}")
        return

    sys.path.insert(0, str(TOOLS))
    sys.path.insert(0, str(OPENPCDET))
    from pcdet.config import cfg, cfg_from_yaml_file
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset

    os.chdir(TOOLS)
    cfg_from_yaml_file(str(TOOLS / "cfgs/kitti_models/centerpoint.yaml"), cfg)
    data_root = OPENPCDET / "data/kitti"
    dataset = KittiDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=data_root,
        logger=None,
    )
    dataset.set_split("train")
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        infos = dataset.get_infos(
            num_workers=args.workers,
            has_label=True,
            count_inside_pts=False,
            sample_id_list=frame_ids,
        )
    validate_infos(infos, frame_ids)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(infos, handle)
    temporary.replace(output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "frames": len(infos),
                "split_file": str(split_file),
                "count_inside_pts": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"CENTERPOINT_TRAIN_INFOS_PASS cached=False frames={len(infos)} path={output}")


if __name__ == "__main__":
    main()
