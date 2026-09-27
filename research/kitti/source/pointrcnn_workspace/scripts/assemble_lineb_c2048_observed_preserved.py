#!/usr/bin/env python
"""Assemble protocol-compliant exact-4N clouds for Line B / c2048-r4.

The frozen exact-4N protocol says each frame must be `M observed + 3M generated`,
with every observed point preserved.  `final_bin`, which the split-region track
consumes directly, does not satisfy that: `choose_strict` draws all 4M points from
the method's merged_raw pool, so zero observed points survive (verified: all 30,067
observed points of frame 000001 are absent from the pu_gcn/pdans final_bin, on xyz
alone).  The split-region tables therefore measure replacement, not upsampling.

This script rebuilds the compliant cloud, reusing `prepare_pointrcnn_e1_e2_inputs`'s
own seeding and assembly so the result is constructed identically to the E1 files of
the main track: observed ⊕ 3M sampled from final_bin, shuffled, exactly 4M points.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731/c2048_r4_pilot256_v1"
OBSERVED = REPO / "results/kitti_unified_x4_current_methods_no_detector/downsampled_x4/velodyne_downsampled_x4_val"
OUT_ROOT = GEN / "e1_observed_preserved"
SPLIT = REPO / "data/KITTI/ImageSets/patch_causal_pilot256.txt"
METHODS = ("pu_gcn", "pu_net_fixed", "pu_edgeformer", "pdans")


def load_e1_module():
    path = REPO / "scripts/prepare_pointrcnn_e1_e2_inputs.py"
    spec = importlib.util.spec_from_file_location("e1prep", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO / "scripts"))
    spec.loader.exec_module(module)
    return module


def main() -> int:
    e1 = load_e1_module()
    frames = [line.strip() for line in SPLIT.read_text().splitlines() if line.strip()]
    summary = {}

    for method in METHODS:
        predicted_dir = GEN / f"{method}_c2048_r4/line_b_downsampled_x4_up/final_bin"
        out_dir = OUT_ROOT / method
        out_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        observed_kept = []

        for frame in frames:
            observed = e1.read_bin(OBSERVED / f"{frame}.bin")
            predicted = e1.read_bin(predicted_dir / f"{frame}.bin")
            m = int(observed.shape[0])
            target = 4 * m
            if predicted.shape[0] != target:
                raise ValueError(
                    f"{method}/{frame}: final_bin has {predicted.shape[0]} points, expected {target}"
                )
            seed = e1.stable_seed(e1.BASE_SEED, "e1", f"lineb_c2048_{method}", frame)
            rng = np.random.default_rng(seed)
            idx = rng.choice(predicted.shape[0], size=3 * m, replace=False)
            final = np.concatenate((observed, predicted[idx]), axis=0).astype(np.float32, copy=False)
            final = final[rng.permutation(target)]
            (out_dir / f"{frame}.bin").write_bytes(final.tobytes())
            written += 1

            if len(observed_kept) < 8:
                got = {tuple(r) for r in final.tolist()}
                want = {tuple(r) for r in observed.tolist()}
                observed_kept.append(len(want - got))

        summary[method] = {
            "frames": written,
            "output_dir": str(out_dir),
            "observed_points_missing_first8": observed_kept,
        }
        print(f"ASSEMBLED {method} frames={written} observed_missing={observed_kept}", flush=True)

    (OUT_ROOT / "assembly_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
