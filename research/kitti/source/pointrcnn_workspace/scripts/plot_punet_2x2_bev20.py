#!/usr/bin/env python3
"""Render the PU-Net 20-frame 2x2 BEV audit without PointRCNN dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results/kitti_patch_punet_causal_ablation_v1_20260731"
RUN = ROOT / "punet20_2x2"
LINE = "line_a_original_x4_up"
OBSERVED = REPO / "data/KITTI/object/training/velodyne_original_val"
SUMMARY = RUN / "reports/geometry_summary.json"
CONDITIONS = {
    "A legacy + old patch": RUN / "pu_net_A_legacy_oldpatch" / LINE / "final_bin",
    "B fixed + old patch": RUN / "pu_net_B_fixed_oldpatch" / LINE / "final_bin",
    "C legacy + local patch": RUN / "pu_net_C_legacy_cover_knn_v3" / LINE / "final_bin",
    "D fixed + local patch": RUN / "pu_net_D_fixed_cover_knn_v3" / LINE / "final_bin",
}


def read_bin(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0xFFFFFFFF


def main() -> int:
    payload = json.loads(SUMMARY.read_text())
    frame = str(payload["visual_frame"])
    observed = read_bin(OBSERVED / f"{frame}.bin")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for axis, (label, directory) in zip(axes.flat, CONDITIONS.items()):
        predicted = read_bin(directory / f"{frame}.bin")
        rng = np.random.default_rng(seed("punet2x2", label, frame))
        real = observed[
            rng.choice(len(observed), size=min(20000, len(observed)), replace=False)
        ]
        generated = predicted[
            rng.choice(len(predicted), size=min(20000, len(predicted)), replace=False)
        ]
        axis.scatter(real[:, 0], real[:, 1], s=0.3, c="0.75", label="real")
        axis.scatter(
            generated[:, 0], generated[:, 1], s=0.35, c="#d62728", alpha=0.55
        )
        axis.set_title(label)
        axis.set_xlim(-10, 85)
        axis.set_ylim(-45, 45)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
    axes[0, 0].legend(markerscale=8, loc="upper right")
    fig.supxlabel("LiDAR x (m)")
    fig.supylabel("LiDAR y (m)")
    fig.suptitle(f"PU-Net normalization × patch BEV audit, frame {frame}")
    fig.tight_layout()
    output = SUMMARY.parent / f"punet_2x2_bev_{frame}.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    payload["visualization"] = str(output)
    SUMMARY.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
