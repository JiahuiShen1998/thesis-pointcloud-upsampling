#!/usr/bin/env python3
"""Quantify the Line B patch dilemma from extracted patch metadata.

At N/4 density a patch cannot be both local and built from 2048 unique source
points: the densest 2 m ball in a Line B frame holds about 1383 points.  This
script reads the patch metadata of each arm and reports the two quantities the
dilemma trades off -- unique support and patch extent -- so the empty cell of
the design is backed by measurement rather than by a failed run alone.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/lineb20_dilemma_v1"
)
ARM_LABEL = {
    "B1": "r=2 m, repeat fill",
    "B3": "r=4 m, repeat fill",
    "B2": "r=4 m, unique 2048",
}


def patch_stats(meta_path: Path) -> dict | None:
    payload = json.loads(meta_path.read_text())
    patches = payload.get("patches") or payload.get("patch_metadata") or []
    if not patches:
        return None
    unique, diameters, repeats = [], [], []
    for patch in patches:
        if "unique_original_index_count" in patch:
            unique.append(float(patch["unique_original_index_count"]))
        if "xy_bbox_diagonal_m" in patch:
            diameters.append(float(patch["xy_bbox_diagonal_m"]))
        if "support_repeat_count" in patch:
            repeats.append(float(patch["support_repeat_count"]))
    return {
        "patches": len(patches),
        "unique": unique,
        "diameters": diameters,
        "repeats": repeats,
        "keys": sorted(patches[0]) if patches else [],
    }


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--method", default="pu_gcn", help="patches are method-independent")
    args = parser.parse_args()

    if not args.root.exists():
        print(f"no such root: {args.root}")
        return 1

    print(f"{'arm':>4}  {'configuration':<20} {'patches':>8} {'unique p50':>11} "
          f"{'unique p10':>11} {'repeat %':>9} {'XY d p50':>9} {'XY d p90':>9}")
    for arm in ("B1", "B3", "B2"):
        variants = sorted(args.root.glob(f"{arm}_{args.method}_*"))
        if not variants:
            print(f"{arm:>4}  {ARM_LABEL[arm]:<20} {'NOT PRODUCED — extraction failed':>60}")
            continue
        manifest_dir = variants[0] / "line_b_downsampled_x4_up/manifests"
        metas = sorted(manifest_dir.glob("*_patch_metadata.json"))
        if not metas:
            print(f"{arm:>4}  {ARM_LABEL[arm]:<20} {'no metadata':>20}")
            continue
        unique, diameters, repeats, total = [], [], [], 0
        keys: list[str] = []
        for meta in metas:
            stats = patch_stats(meta)
            if not stats:
                continue
            total += stats["patches"]
            unique += stats["unique"]
            diameters += stats["diameters"]
            repeats += stats["repeats"]
            keys = keys or stats["keys"]
        repeat_pct = (
            100.0 * sum(repeats) / (2048 * total) if repeats and total else float("nan")
        )
        u50 = quantile(unique, 0.50)
        u10 = quantile(unique, 0.10)
        d50 = quantile(diameters, 0.50)
        d90 = quantile(diameters, 0.90)
        fmt = lambda v, w, p=1: f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"
        print(f"{arm:>4}  {ARM_LABEL[arm]:<20} {total:>8} {fmt(u50,11,0)} {fmt(u10,11,0)} "
              f"{repeat_pct:>8.1f}% {fmt(d50,9,2)} {fmt(d90,9,2)}")
        if not unique and keys:
            print(f"      available metadata keys: {keys}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
