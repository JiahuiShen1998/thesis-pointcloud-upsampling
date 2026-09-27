#!/usr/bin/env python
"""Consensus-ranked candidate selection for Line B exact-4N inputs.

The generation stage overproduces: 31 patches x 8192 points give ~254k candidates
where exact-4N needs only 3M ~= 90k.  `strict_x4_from_merged_raw.choose_strict`
resolves that 2.8x surplus with `rng.choice`, which keeps good and bad candidates at
the same rate.  This script spends the surplus instead.

Ranking rule (fixed before any evaluation, and derived only from pipeline constants):

1. **Cross-patch consensus.**  Every source point falls inside ~2 patches, so a real
   surface inside an overlap is reconstructed twice, independently.  A candidate whose
   nearest neighbour from *another* patch is close was corroborated; one that no other
   patch echoes was invented by a single patch.  Rank ascending by that distance.

2. **One point per 0.1 m voxel.**  The detector input stage already reduces to a voxel
   representative at 0.1 m, so a second point in an occupied voxel is discarded
   downstream -- spending budget on it buys nothing.  Admit in consensus order, skipping
   voxels already taken; if the budget outlives the unique voxels, keep filling in
   consensus order.

Both rules read only the method's own output and the detector's published constants.
No ground truth, no detector output, no AP, and in particular not the original cloud --
that is what Line B is trying to recover, so it may score this selection afterwards but
must never inform it.

Output per frame: `M observed + 3M selected`, shuffled, exactly 4M points, with each
generated point taking the intensity of its nearest observed point (the policy
`strict_x4_from_merged_raw` already uses).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/c2048_r4_pilot256_v1"
OBSERVED = REPO / "results/kitti_unified_x4_current_methods_no_detector/downsampled_x4/velodyne_downsampled_x4_val"
OUT_ROOT = GEN / "e1_consensus_selected"
SPLIT = REPO / "data/KITTI/ImageSets/patch_causal_pilot256.txt"
VOXEL = 0.1
SEED = 20260811
METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net_fixed")


def consensus_order(cand: np.ndarray, patch_id: np.ndarray) -> np.ndarray:
    """Indices of candidates, best-corroborated first."""
    tree = cKDTree(cand)
    k = min(16, len(cand))
    dists, idxs = tree.query(cand, k=k, workers=4)
    # scipy's threaded query has returned uninitialised indices here on some frames;
    # an out-of-range index means no neighbour was resolved in that slot, which is the
    # same thing as an infinite distance, so drop it rather than index with garbage.
    valid = (idxs >= 0) & (idxs < len(cand))
    other = np.zeros(idxs.shape, dtype=bool)
    safe = np.where(valid, idxs, 0)
    other[valid] = (patch_id[safe] != patch_id[:, None])[valid]
    # inf where every one of the k nearest neighbours came from the same patch
    best = np.where(other, dists, np.inf).min(axis=1)
    return np.argsort(best, kind="stable")


def select(cand: np.ndarray, patch_id: np.ndarray, budget: int) -> np.ndarray:
    order = consensus_order(cand, patch_id)
    keys = np.floor(cand[order] / VOXEL).astype(np.int64)
    # first occurrence of each voxel, restored to consensus order
    _, first = np.unique(keys, axis=0, return_index=True)
    deduped = order[np.sort(first)]
    if len(deduped) >= budget:
        return deduped[:budget]
    # budget outlives the unique voxels: keep filling in consensus order
    taken = np.zeros(len(cand), dtype=bool)
    taken[deduped] = True
    rest = order[~taken[order]]
    return np.concatenate([deduped, rest[: budget - len(deduped)]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    frames = SPLIT.read_text().split()
    if args.limit:
        frames = frames[: args.limit]
    summary: dict[str, dict] = {}

    for method in args.methods:
        base = GEN / f"{method}_c2048_r4/line_b_downsampled_x4_up"
        out_dir = OUT_ROOT / method
        out_dir.mkdir(parents=True, exist_ok=True)
        written = reused = 0
        voxel_short = 0

        for position, frame in enumerate(frames, start=1):
            observed = np.fromfile(OBSERVED / f"{frame}.bin", dtype=np.float32).reshape(-1, 4)
            m = len(observed)
            target = 4 * m
            out_path = out_dir / f"{frame}.bin"
            if out_path.exists() and out_path.stat().st_size == target * 16:
                reused += 1
                continue

            cand = np.load(base / "merged_raw" / f"{frame}.npy").astype(np.float64)[:, :3]
            prov = json.loads((base / "manifests" / f"{frame}_merged_raw_provenance.json").read_text())
            per_patch = int(prov["raw_patch_output_count_each"])
            if per_patch * int(prov["patch_count"]) != len(cand):
                raise ValueError(
                    f"{method}/{frame}: merged_raw has {len(cand)} points, not "
                    f"{per_patch}x{prov['patch_count']}; patch identity is not recoverable"
                )
            patch_id = np.arange(len(cand)) // per_patch

            need = 3 * m
            if len(cand) < need:
                raise ValueError(f"{method}/{frame}: only {len(cand)} candidates for {need} slots")
            pick = select(cand, patch_id, need)
            unique_voxels = len(np.unique(np.floor(cand[pick] / VOXEL).astype(np.int64), axis=0))
            if unique_voxels < need:
                voxel_short += 1

            gen_xyz = cand[pick]
            obs_tree = cKDTree(observed[:, :3].astype(np.float64))
            nearest = obs_tree.query(gen_xyz, k=1, workers=-1)[1]
            generated = np.hstack([gen_xyz, observed[nearest, 3:4]]).astype(np.float32)

            rng = np.random.default_rng(SEED)
            final = np.concatenate([observed, generated], axis=0).astype(np.float32)
            final = final[rng.permutation(target)]
            if final.shape != (target, 4) or not np.isfinite(final).all():
                raise ValueError(f"{method}/{frame}: bad final shape/values {final.shape}")
            # A bad neighbour index yields a huge but still finite coordinate, which
            # isfinite happily accepts; KITTI points are physically bounded, so check
            # the magnitude too.
            reach = float(np.abs(final[:, :3]).max())
            if reach > 200.0:
                raise ValueError(
                    f"{method}/{frame}: coordinate magnitude {reach:.3g} exceeds the KITTI range"
                )
            out_path.write_bytes(final.tobytes())
            written += 1

            if position % 32 == 0:
                print(f"SELECT {method} {position}/{len(frames)}", flush=True)

        summary[method] = {
            "frames_written": written,
            "frames_reused": reused,
            "frames_where_budget_exceeded_unique_voxels": voxel_short,
            "output_dir": str(out_dir),
        }
        print(f"SELECT_DONE {method} written={written} reused={reused} voxel_short={voxel_short}", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
