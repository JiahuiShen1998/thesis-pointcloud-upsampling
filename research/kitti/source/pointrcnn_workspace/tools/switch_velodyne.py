#!/usr/bin/env python3
"""Switch the active KITTI velodyne folder used by PointRCNN.

The script prefers a symlink at:
  data/KITTI/object/training/velodyne

It can bootstrap the current real directory into ``velodyne_original`` once,
then repoint ``velodyne`` at any of the prepared density variants.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


VALID_TARGETS = {
    "velodyne_original",
    "velodyne_upsampled",
    "velodyne_downsampled",
    "velodyne_down_up",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Switch the active KITTI velodyne folder")
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=Path("data/KITTI/object/training"),
        help="KITTI training directory containing velodyne folders.",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=sorted(VALID_TARGETS),
        help="Target velodyne subdirectory to activate.",
    )
    parser.add_argument(
        "--bootstrap-original",
        action="store_true",
        help=(
            "If velodyne_original does not exist, move the current real velodyne "
            "directory into velodyne_original and create the symlink in place."
        ),
    )
    parser.add_argument(
        "--record-file",
        type=Path,
        default=None,
        help="Optional file to record the final active path for reproducibility.",
    )
    return parser.parse_args()


def make_relative_symlink(link_path: Path, target_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        elif link_path.is_dir():
            raise RuntimeError(
                f"{link_path} is a real directory. Refusing to remove it automatically."
            )
    rel_target = os.path.relpath(target_path, link_path.parent)
    os.symlink(rel_target, link_path)


def bootstrap_original(training_dir: Path) -> Path:
    active = training_dir / "velodyne"
    original = training_dir / "velodyne_original"
    if original.exists():
        return original
    if not active.exists():
        raise FileNotFoundError(f"Cannot bootstrap original data because {active} does not exist.")
    if active.is_symlink():
        resolved = active.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Broken symlink: {active} -> {resolved}")
        shutil.copytree(resolved, original, copy_function=os.link)
        return original
    active.rename(original)
    make_relative_symlink(active, original)
    return original


def main() -> int:
    args = parse_args()
    training_dir = args.training_dir.resolve()
    target_dir = training_dir / args.target
    active = training_dir / "velodyne"

    if args.bootstrap_original:
        bootstrap_original(training_dir)

    if not target_dir.exists():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")

    if active.exists() and not active.is_symlink():
        raise RuntimeError(
            f"{active} is a real directory. Run with --bootstrap-original first so it can be replaced by a symlink."
        )

    make_relative_symlink(active, target_dir)

    resolved = active.resolve()
    message = f"active_velodyne={active} -> {active.readlink()} ({resolved})"
    print(message)
    if args.record_file is not None:
        args.record_file.parent.mkdir(parents=True, exist_ok=True)
        args.record_file.write_text(message + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
