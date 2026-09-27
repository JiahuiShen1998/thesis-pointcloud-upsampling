#!/usr/bin/env python3
"""Save the exact local source/config snapshot used for full-val reproducibility."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pugcn_detector_adaptation_full_val_20260908/reports"
EXPERIMENT_FILES = [
    "scripts/prepare_kitti_downsampled_x4_val.py",
    "scripts/wrappers/kitti_patch_extractor.py",
    "scripts/wrappers/tf_pugcn_family_patch_infer_many.py",
    "scripts/wrappers/strict_x4_from_merged_raw.py",
    "scripts/wrappers/strict_x4_from_merged_raw_many.py",
    "scripts/run_patch_causal_upsampling.py",
    "scripts/run_kitti_unified_x4_current_methods_smoke.py",
    "scripts/prepare_centerpoint_observed_first_train.py",
    "scripts/run_pointrcnn_full_train_stage.py",
    "scripts/run_pointrcnn_full_adaptation_20260824.sh",
    "scripts/run_pointrcnn_rpn_feature_export.py",
    "scripts/combine_pointrcnn_rpn_with_official_rcnn.py",
    "scripts/run_centerpoint_full_train.py",
    "scripts/run_centerpoint_full_adaptation_20260824.sh",
    "scripts/run_pugcn_full_train_inputs_20260824.sh",
    "scripts/run_pointrcnn_checkpoint_split_eval.py",
    "scripts/run_pointrcnn_full_validation_pugcn_cap100k_ap_eval.py",
    "scripts/run_patch_causal_centerpoint_eval.py",
    "scripts/verify_patch_causal_strict_x4.py",
    "scripts/run_pugcn_detector_adaptation_full_val_20260908.sh",
    "scripts/summarize_pugcn_detector_adaptation_full_val_20260908.py",
    "scripts/analyze_pugcn_detector_adaptation_convergence.py",
    "scripts/package_pugcn_full_val_sources.py",
    "scripts/report_pugcn_full_val_progress.py",
    "lib/datasets/kitti_dataset.py",
    "lib/datasets/kitti_rcnn_dataset.py",
    "lib/config.py",
    "tools/cfgs/default.yaml",
    "tools/eval_rcnn.py",
    "tools/train_rcnn.py",
    "tools/kitti_object_eval_python/eval.py",
    "tools/kitti_object_eval_python/evaluate.py",
    "external/OpenPCDet/tools/cfgs/kitti_models/centerpoint.yaml",
    "external/OpenPCDet/tools/cfgs/dataset_configs/kitti_dataset.yaml",
    "external/OpenPCDet/pcdet/datasets/kitti/kitti_dataset.py",
    "external/OpenPCDet/pcdet/datasets/processor/data_processor.py",
    "external/OpenPCDet/pcdet/datasets/kitti/kitti_object_eval_python/eval.py",
    "external/PU-GCN/pretrained/pu1k-pugcn/args.txt",
    "external/PU-GCN/pretrained/pu1k-pugcn/checkpoint",
    "data/KITTI/ImageSets/train.txt",
    "data/KITTI/ImageSets/val.txt",
]


def main() -> None:
    paths = {ROOT / name for name in EXPERIMENT_FILES}
    pugcn_root = ROOT / "external/PU-GCN"
    for path in pugcn_root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() in {".py", ".cpp", ".cu", ".h", ".sh"} or path.name.lower().startswith(("readme", "license")):
            paths.add(path)
    payloads = {}
    for path in sorted(paths):
        payloads[str(path.relative_to(ROOT))] = path.read_bytes()
    manifest = {
        "description": "Byte-for-byte local workspace snapshot, not a claim of pristine upstream revision. Model weight binaries and data are excluded.",
        "files": [
            {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payloads.items()
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    (OUT / "source_manifest.json").write_bytes(manifest_bytes)
    archive = OUT / "pugcn_full_val_sources.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, data in {**payloads, "source_manifest.json": manifest_bytes}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            handle.addfile(info, io.BytesIO(data))
    lines = [
        "# PU-GCN full-validation source snapshot",
        "",
        "This is the local source used for this experiment. The archive includes PU-GCN Python/CUDA/C++ sources, wrappers, detector sampling/evaluation code, configurations and splits. Checkpoints and dataset binaries are excluded.",
        "",
    ]
    for name in EXPERIMENT_FILES:
        lines.extend(["## " + name, "", "````", payloads[name].decode("utf-8"), "````", ""])
    (OUT / "experiment_source_full.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"SOURCE_SNAPSHOT_PASS files={len(payloads)} archive={archive}")


if __name__ == "__main__":
    main()
