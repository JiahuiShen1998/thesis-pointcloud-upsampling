#!/usr/bin/env python3
"""Create Step 7 upsampling smoke-test sample manifest and input symlinks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
DEFAULT_ORIGINAL = PROJECT_ROOT / "datasets" / "modelnet40_original"
DEFAULT_DOWNSAMPLED = PROJECT_ROOT / "datasets" / "modelnet40_downsampled50"
DEFAULT_MANIFEST = PROJECT_ROOT / "reports" / "step7_smoke_sample_manifest.csv"
DEFAULT_SMOKE_ORIGINAL = PROJECT_ROOT / "datasets" / "smoke_inputs" / "original"
DEFAULT_SMOKE_DOWN = PROJECT_ROOT / "datasets" / "smoke_inputs" / "downsampled50"
SPLITS = ("train", "test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create Step 7 smoke sample manifest.")
    parser.add_argument("--original-root", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--downsampled-root", type=Path, default=DEFAULT_DOWNSAMPLED)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--smoke-original", type=Path, default=DEFAULT_SMOKE_ORIGINAL)
    parser.add_argument("--smoke-downsampled", type=Path, default=DEFAULT_SMOKE_DOWN)
    args = parser.parse_args()

    class_to_idx_path = args.original_root / "metadata" / "class_to_idx.json"
    if not class_to_idx_path.is_file():
        print(f"ERROR: missing {class_to_idx_path}", file=sys.stderr)
        return 1
    with open(class_to_idx_path, encoding="utf-8") as handle:
        class_to_idx = json.load(handle)

    rows: list[dict] = []
    for split in SPLITS:
        split_dir = args.original_root / split
        for class_name in sorted(class_to_idx):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                print(f"WARNING: missing class dir {class_dir}", file=sys.stderr)
                continue
            npy_files = sorted(class_dir.glob("*.npy"))
            if not npy_files:
                print(f"WARNING: no npy in {class_dir}", file=sys.stderr)
                continue
            source_original = npy_files[0]
            shape_id = source_original.stem
            source_down = args.downsampled_root / split / class_name / f"{shape_id}.npy"
            if not source_down.is_file():
                print(f"WARNING: missing downsampled {source_down}", file=sys.stderr)
                continue

            smoke_orig = args.smoke_original / split / class_name / f"{shape_id}.npy"
            smoke_down = args.smoke_downsampled / split / class_name / f"{shape_id}.npy"
            smoke_orig.parent.mkdir(parents=True, exist_ok=True)
            smoke_down.parent.mkdir(parents=True, exist_ok=True)
            if smoke_orig.exists() or smoke_orig.is_symlink():
                smoke_orig.unlink()
            if smoke_down.exists() or smoke_down.is_symlink():
                smoke_down.unlink()
            smoke_orig.symlink_to(source_original.resolve())
            smoke_down.symlink_to(source_down.resolve())

            rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "label": class_to_idx[class_name],
                    "shape_id": shape_id,
                    "source_original_path": str(source_original.resolve()),
                    "source_downsampled50_path": str(source_down.resolve()),
                    "smoke_original_path": str(smoke_orig),
                    "smoke_downsampled50_path": str(smoke_down),
                    "selected_for_smoke": 1,
                }
            )

    fieldnames = [
        "split",
        "class_name",
        "label",
        "shape_id",
        "source_original_path",
        "source_downsampled50_path",
        "smoke_original_path",
        "smoke_downsampled50_path",
        "selected_for_smoke",
    ]
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    train_n = sum(1 for r in rows if r["split"] == "train")
    test_n = sum(1 for r in rows if r["split"] == "test")
    print(f"Manifest: {args.manifest_path}")
    print(f"Samples: train={train_n} test={test_n} total={len(rows)}")
    print(f"Smoke original dir: {args.smoke_original}")
    print(f"Smoke downsampled dir: {args.smoke_downsampled}")
    return 0 if len(rows) == 80 else 2


if __name__ == "__main__":
    raise SystemExit(main())
