#!/usr/bin/env python
"""Does cross-patch consensus predict which generated points are real?

Line B covers every source point with ~2 overlapping patches (membership_count_median
is 2.0), so a surface inside an overlap is generated twice, independently.  Where two
patches agree, the point is likely on a real surface; where a point has no counterpart
from any other patch, it is likely invented.  That signal uses only the method's own
output geometry -- no ground truth, no detector output, no AP -- so it is legal as a
*selection* rule under the forward-only pipeline.

The original (pre-downsampling) cloud is the thing Line B is trying to recover, so it
appears here only to *score* a selection after the fact.  It must never enter the
ranking itself; doing so would be oracle leakage.

Reports, per method, the reference-voxel precision of:
  * the 3M points a uniform random draw keeps (what the pipeline does today), and
  * the 3M points the consensus ranking keeps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/c2048_r4_pilot256_v1"
OBSERVED = REPO / "results/kitti_unified_x4_current_methods_no_detector/downsampled_x4/velodyne_downsampled_x4_val"
REFERENCE = REPO / "data/KITTI/object/training/velodyne_original_val"
VOXEL = 0.1
METHODS = ("pdans", "pu_gcn", "pu_edgeformer", "pu_net_fixed")


def voxel_keys(xyz: np.ndarray) -> np.ndarray:
    return np.floor(xyz / VOXEL).astype(np.int64)


def precision_against_reference(sel: np.ndarray, ref_keys: set[tuple[int, int, int]]) -> float:
    if len(sel) == 0:
        return float("nan")
    keys = voxel_keys(sel)
    hit = sum(1 for k in map(tuple, keys) if k in ref_keys)
    return 100.0 * hit / len(keys)


def novel_stats(
    sel: np.ndarray,
    ref_keys: set[tuple[int, int, int]],
    obs_keys: set[tuple[int, int, int]],
    removed_keys: set[tuple[int, int, int]],
) -> tuple[float, float, int]:
    """Score only what the observed cloud does not already contain.

    A generated point landing in a voxel the observed cloud already occupies is
    trivially "correct" and carries no information -- the detector sees that voxel
    either way.  What Line B actually has to recover are the voxels the 4x
    downsampling removed, so precision and recall are measured against those.
    """
    if len(sel) == 0:
        return float("nan"), float("nan"), 0
    keys = list(map(tuple, voxel_keys(sel)))
    novel = [k for k in keys if k not in obs_keys]
    if not novel:
        return float("nan"), 0.0, 0
    hit = {k for k in novel if k in ref_keys}
    precision = 100.0 * sum(1 for k in novel if k in ref_keys) / len(novel)
    recall = 100.0 * len(hit & removed_keys) / max(len(removed_keys), 1)
    return precision, recall, len(novel)


def consensus_score(cand: np.ndarray, patch_id: np.ndarray) -> np.ndarray:
    """Distance to the nearest candidate produced by a *different* patch.

    Small distance means an independent patch put a point in the same place.
    Queried k-nearest and taking the first hit from another patch keeps this a single
    tree build; k=16 is enough because a point's own patch contributes at most a few
    of the very nearest neighbours in practice.
    """
    tree = cKDTree(cand)
    k = min(16, len(cand))
    dists, idxs = tree.query(cand, k=k, workers=-1)
    same = patch_id[idxs] == patch_id[:, None]
    masked = np.where(same, np.inf, dists)
    best = masked.min(axis=1)
    # Points whose k nearest neighbours are all from their own patch get the worst score.
    best[~np.isfinite(best)] = np.inf
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260811)
    args = ap.parse_args()

    split = (REPO / "data/KITTI/ImageSets/patch_causal_pilot256.txt").read_text().split()
    frames = split[: args.frames]
    out: dict[str, dict] = {}

    for method in METHODS:
        base = GEN / f"{method}_c2048_r4/line_b_downsampled_x4_up"
        rand_p, cons_p, all_p, gains = [], [], [], []

        for frame in frames:
            merged = np.load(base / "merged_raw" / f"{frame}.npy").astype(np.float64)
            if merged.ndim != 2 or merged.shape[1] < 3:
                raise ValueError(f"{method}/{frame}: unexpected merged_raw shape {merged.shape}")
            cand = merged[:, :3]
            prov = json.loads((base / "manifests" / f"{frame}_merged_raw_provenance.json").read_text())
            per_patch = int(prov["raw_patch_output_count_each"])
            n_patch = int(prov["patch_count"])
            if per_patch * n_patch != len(cand):
                raise ValueError(
                    f"{method}/{frame}: merged_raw has {len(cand)} points, "
                    f"expected {per_patch}*{n_patch}; patch identity is not recoverable by index"
                )
            patch_id = np.arange(len(cand)) // per_patch

            observed = np.fromfile(OBSERVED / f"{frame}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
            reference = np.fromfile(REFERENCE / f"{frame}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
            ref_keys = set(map(tuple, voxel_keys(reference.astype(np.float64))))
            need = 3 * len(observed)

            rng = np.random.default_rng(args.seed)
            pick_rand = cand[rng.choice(len(cand), size=min(need, len(cand)), replace=False)]

            score = consensus_score(cand, patch_id)
            order = np.argsort(score, kind="stable")
            pick_cons = cand[order[:need]]

            obs_keys = set(map(tuple, voxel_keys(observed.astype(np.float64))))
            removed_keys = ref_keys - obs_keys

            pr, rr, nr = novel_stats(pick_rand, ref_keys, obs_keys, removed_keys)
            pc, rc, nc = novel_stats(pick_cons, ref_keys, obs_keys, removed_keys)
            rand_p.append((pr, rr, nr))
            cons_p.append((pc, rc, nc))
            all_p.append(precision_against_reference(cand, ref_keys))
            gains.append(pc - pr)

        rp = np.array(rand_p, dtype=float)
        cp = np.array(cons_p, dtype=float)
        out[method] = {
            "pool_precision_all_voxels": float(np.mean(all_p)),
            "random_novel_precision": float(np.nanmean(rp[:, 0])),
            "random_removed_recall": float(np.nanmean(rp[:, 1])),
            "random_novel_voxels": float(np.mean(rp[:, 2])),
            "consensus_novel_precision": float(np.nanmean(cp[:, 0])),
            "consensus_removed_recall": float(np.nanmean(cp[:, 1])),
            "consensus_novel_voxels": float(np.mean(cp[:, 2])),
            "novel_precision_gain": float(np.nanmean(gains)),
        }
        r = out[method]
        print(
            f"{method:14s} 新体素精度 随机={r['random_novel_precision']:5.1f}% "
            f"共识={r['consensus_novel_precision']:5.1f}% ({r['novel_precision_gain']:+5.1f}) | "
            f"删除体素召回 随机={r['random_removed_recall']:5.1f}% 共识={r['consensus_removed_recall']:5.1f}% | "
            f"新体素数 随机={r['random_novel_voxels']:.0f} 共识={r['consensus_novel_voxels']:.0f}",
            flush=True,
        )

    (GEN / "consensus_probe.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
