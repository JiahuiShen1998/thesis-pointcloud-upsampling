#!/usr/bin/env python3
"""Assemble exact-4N frames by ranking merged-raw candidates instead of sampling them.

`strict_x4_from_merged_raw.py` draws the 3N generated points uniformly at random
from a candidate pool that is roughly 2.1-2.5x larger than needed, so more than
half of every method's output is decided by a coin flip.  This script keeps the
protocol identical -- N observed rows first, then exactly 3N generated rows --
but chooses the 3N by label-free geometric confidence, with a per-voxel quota so
that high-confidence flat regions cannot crowd out object surfaces.

Nothing here reads labels, boxes, detector outputs or AP.  The scoring function
is the one already validated for V1 (`prepare_detector_aware_pdans_inputs`).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v1_selector", REPO / "scripts/prepare_detector_aware_pdans_inputs.py"
)
v1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v1)

VOXEL_M = 0.10


def read_bin(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def voxel_quota_select(
    scores: np.ndarray, xyz: np.ndarray, target: int, quota: int
) -> np.ndarray:
    """Highest-scoring `target` candidates, at most `quota` per 0.10 m voxel.

    Ranking alone concentrates the budget on large planar surfaces, which score
    well everywhere; the quota spreads the same budget across occupied space.
    If the quota is too tight to reach `target`, the remainder is filled by
    global score order so the exact-4N count is always met.
    """
    order = np.argsort(-scores, kind="stable")
    keys = np.floor(xyz[order] / VOXEL_M).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)

    rank_in_voxel = np.zeros(inverse.size, dtype=np.int64)
    seen: dict[int, int] = {}
    for position, voxel in enumerate(inverse.tolist()):
        count = seen.get(voxel, 0)
        rank_in_voxel[position] = count
        seen[voxel] = count + 1

    accepted = order[rank_in_voxel < quota]
    if accepted.size >= target:
        return np.sort(accepted[:target])
    remaining = order[rank_in_voxel >= quota]
    return np.sort(np.concatenate((accepted, remaining[: target - accepted.size])))


def voxel_keys(xyz: np.ndarray, size: float) -> set[tuple[int, int, int]]:
    return set(map(tuple, np.floor(xyz / size).astype(np.int64).tolist()))


def geometry_report(observed: np.ndarray, generated: np.ndarray, size: float) -> dict:
    """Fraction of generated points that land on already-observed surface."""
    observed_voxels = voxel_keys(observed, size)
    generated_voxels = np.floor(generated / size).astype(np.int64)
    on_surface = np.fromiter(
        (tuple(k) in observed_voxels for k in generated_voxels.tolist()),
        dtype=bool,
        count=generated_voxels.shape[0],
    )
    unique_generated = voxel_keys(generated, size)
    return {
        "generated_voxel_precision": float(on_surface.mean()),
        "unsupported_voxel_fraction": float(
            len(unique_generated - observed_voxels) / max(len(unique_generated), 1)
        ),
        "occupied_voxels_generated": len(unique_generated),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-raw-dir", type=Path, required=True)
    parser.add_argument("--observed-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--voxel-quota", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--score", choices=["confidence", "utility"], default="confidence",
        help="confidence is pure surface quality; utility adds V1's sparse/far bias",
    )
    parser.add_argument("--seed", type=int, default=20260804)
    args = parser.parse_args()

    frames = [f.strip() for f in args.split_file.read_text().split() if f.strip()]
    if args.limit:
        frames = frames[: args.limit]
    bin_dir = args.out_dir / "final_bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for position, frame in enumerate(frames, start=1):
        observed = read_bin(args.observed_dir / f"{frame}.bin")
        candidates = np.load(args.merged_raw_dir / f"{frame}.npy").astype(np.float64)
        target = 3 * observed.shape[0]
        if candidates.shape[0] < target:
            raise RuntimeError(f"{frame}: pool {candidates.shape[0]} < target {target}")

        features = v1.geometry_features(
            observed[:, :3].astype(np.float64),
            candidates[:, :3],
            np.linalg.norm(candidates[:, :2], axis=1),
        )
        picked = voxel_quota_select(
            features[args.score], candidates[:, :3], target, args.voxel_quota
        )

        rng = np.random.default_rng(args.seed + int(frame))
        random_pick = rng.choice(candidates.shape[0], size=target, replace=False)

        quality = geometry_report(observed[:, :3], candidates[picked, :3], VOXEL_M)
        baseline = geometry_report(observed[:, :3], candidates[random_pick, :3], VOXEL_M)

        generated = np.zeros((target, 4), dtype=np.float32)
        generated[:, :3] = candidates[picked, :3]
        tree = v1.cKDTree(observed[:, :3])
        generated[:, 3] = observed[tree.query(candidates[picked, :3], workers=-1)[1], 3]
        final = np.concatenate((observed, generated), axis=0).astype(np.float32)
        assert final.shape[0] == 4 * observed.shape[0]
        final.tofile(bin_dir / f"{frame}.bin")

        row = {
            "frame_id": frame,
            "observed": observed.shape[0],
            "pool": candidates.shape[0],
            "target_3n": target,
            "oversupply": round(candidates.shape[0] / target, 3),
            **{f"quality_{k}": v for k, v in quality.items()},
            **{f"random_{k}": v for k, v in baseline.items()},
        }
        row["precision_gain"] = round(
            quality["generated_voxel_precision"] - baseline["generated_voxel_precision"], 4
        )
        rows.append(row)
        print(
            f"ASSEMBLE {frame} {position}/{len(frames)} "
            f"precision {baseline['generated_voxel_precision']:.4f} -> "
            f"{quality['generated_voxel_precision']:.4f} "
            f"(gain {row['precision_gain']:+.4f})",
            flush=True,
        )

    manifest = args.out_dir / "assemble_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "frames": len(rows),
        "score": args.score,
        "voxel_quota": args.voxel_quota,
        "median_oversupply": float(np.median([r["oversupply"] for r in rows])),
        "median_precision_random": float(
            np.median([r["random_generated_voxel_precision"] for r in rows])
        ),
        "median_precision_quality": float(
            np.median([r["quality_generated_voxel_precision"] for r in rows])
        ),
        "median_unsupported_random": float(
            np.median([r["random_unsupported_voxel_fraction"] for r in rows])
        ),
        "median_unsupported_quality": float(
            np.median([r["quality_unsupported_voxel_fraction"] for r in rows])
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
