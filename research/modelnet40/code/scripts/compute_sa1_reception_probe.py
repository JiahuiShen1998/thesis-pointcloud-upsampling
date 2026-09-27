#!/usr/bin/env python3
"""What PointNet++ actually sees at its first set-abstraction layer.

pointnet2_cls_ssg SA1 is npoint=512, radius=0.2, nsample=32, and it consumes the raw
(re-normalised) input cloud, so its statistics are measurable directly from the data.

models/pointnet2_utils.py :: query_ball_point behaves as follows.
  * more than nsample points inside the ball -> the surplus is DISCARDED, and the kept
    ones are the lowest *storage indices*, i.e. file order, not proximity.
  * fewer than nsample inside the ball  -> the empty slots are filled by REPEATING the
    first in-ball point.

So the two ways a cloud can hurt the network are measurable:
  saturation  = share of in-ball points thrown away
  padding     = share of the 32 neighbour slots holding a repeated point

Both are driven by how evenly the points are distributed, not by how many there are,
which is why they must be compared inside a fixed point-count bucket.
"""
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
DS = ROOT / "datasets"
NPOINT, RADIUS, NSAMPLE = 512, 0.2, 32

# (bucket, label, root) -- root/<split>/<class>/<shape>.npy
VARIANTS = [
    (256, "Downsampled ×4", DS / "modelnet40_downsampled_x4"),
    (256, "Mesh-ref 256", DS / "modelnet40_mesh_ref_256"),
    (1024, "Original", DS / "modelnet40_original"),
    (1024, "EAR", DS / "lineB_downsampled_x4_up" / "strict_N" / "ear"),
    (1024, "PDANS", DS / "lineB_downsampled_x4_up" / "strict_N" / "pdans"),
    (1024, "PU-Net", DS / "lineB_downsampled_x4_up" / "strict_N" / "pu_net"),
    (1024, "PU-GCN", DS / "lineB_downsampled_x4_up" / "strict_N" / "pu_gcn"),
    (1024, "PU-EdgeFormer", DS / "lineB_downsampled_x4_up" / "strict_N" / "pu_edgeformer"),
    (4096, "Mesh-ref 4096", DS / "modelnet40_mesh_ref_4096"),
    (4096, "EAR", DS / "lineA_original_up" / "strict_4N" / "ear"),
    (4096, "PDANS", DS / "lineA_original_up" / "strict_4N" / "pdans"),
    (4096, "PU-Net", DS / "lineA_original_up" / "strict_4N" / "pu_net"),
    (4096, "PU-GCN", DS / "lineA_original_up" / "strict_4N" / "pu_gcn"),
    (4096, "PU-EdgeFormer", DS / "lineA_original_up" / "strict_4N" / "pu_edgeformer"),
]


def pc_normalize(pc):
    """Exactly the loader's normalisation (ModelNetDataLoader.py:17)."""
    pc = pc - np.mean(pc, axis=0)
    return pc / np.max(np.sqrt(np.sum(pc ** 2, axis=1)))


def fps(pts, npoint, seed):
    """Farthest point sampling, as in pointnet2_utils (random first centroid)."""
    n = pts.shape[0]
    k = min(npoint, n)
    rng = np.random.default_rng(seed)
    idx = np.empty(k, dtype=np.int64)
    dist = np.full(n, np.inf)
    far = int(rng.integers(0, n))
    for i in range(k):
        idx[i] = far
        d = np.sum((pts - pts[far]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        far = int(np.argmax(dist))
    return pts[idx]


def manifest(split="test"):
    rows = []
    path = ROOT / "datasets" / "modelnet40_original" / "metadata" / f"{split}_manifest.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r.get("class_name") or r.get("class"), r["shape_id"]))
    return rows


def one(args):
    cls, sid, seed = args
    out = {}
    for bucket, label, root in VARIANTS:
        p = root / "test" / cls / f"{sid}.npy"
        pts = pc_normalize(np.load(p).astype(np.float64))
        cent = fps(pts, NPOINT, seed)
        tree = cKDTree(pts)
        cnt = np.asarray(tree.query_ball_point(cent, r=RADIUS, return_length=True),
                         dtype=np.float64)
        # what truncation actually keeps: the 32 lowest STORAGE indices in the ball.
        # compare their spatial spread with that of 32 uniformly random in-ball points.
        rng2 = np.random.default_rng(seed)
        ratios = []
        for c in cent[:128]:
            ib = tree.query_ball_point(c, r=RADIUS)
            if len(ib) <= NSAMPLE:
                continue
            ib = np.sort(np.asarray(ib))
            kept = pts[ib[:NSAMPLE]]
            rand = pts[rng2.choice(ib, NSAMPLE, replace=False)]
            sk = np.sqrt(np.mean(np.sum((kept - kept.mean(0)) ** 2, axis=1)))
            sr = np.sqrt(np.mean(np.sum((rand - rand.mean(0)) ** 2, axis=1)))
            if sr > 1e-9:
                ratios.append(sk / sr)
        spread_ratio = float(np.mean(ratios)) if ratios else 1.0
        over = np.maximum(cnt - NSAMPLE, 0.0)       # points the layer discards
        under = np.maximum(NSAMPLE - cnt, 0.0)      # slots filled by a repeated point
        out[(bucket, label)] = np.array([
            cnt.mean(),
            over.sum() / max(cnt.sum(), 1e-9),      # discarded share of in-ball points
            under.sum() / (len(cnt) * NSAMPLE),     # padded share of neighbour slots
            float((cnt > NSAMPLE).mean()),          # share of balls saturated
            float((cnt < NSAMPLE).mean()),          # share of balls starved
            cnt.std() / max(cnt.mean(), 1e-9),      # CV of local density
            spread_ratio,                           # spread(kept 32) / spread(random 32)
        ])
    return out


def main():
    items = manifest("test")
    if "--smoke" in sys.argv:
        items = items[:16]
    tasks = [(c, s, 42 + i) for i, (c, s) in enumerate(items)]
    acc, n = {}, 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, t) for t in tasks]
        for fu in as_completed(futs):
            r = fu.result()
            n += 1
            for k, v in r.items():
                acc[k] = acc.get(k, np.zeros(7)) + v
    res = {f"{b}|{l}": (v / n).tolist() for (b, l), v in acc.items()}
    print(f"shapes = {n}   SA1: npoint={NPOINT} radius={RADIUS} nsample={NSAMPLE}\n")
    hdr = (f"{'bucket':>7}  {'variant':<16}{'mean in ball':>13}{'discarded':>11}"
           f"{'padded':>9}{'saturated':>11}{'starved':>9}{'CV':>8}{'kept/rand':>11}")
    print(hdr); print("-" * len(hdr))
    for bucket in (256, 1024, 4096):
        for b, l in [(b, l) for b, l, _ in VARIANTS if b == bucket]:
            m, disc, pad, sat, star, cv, sr = res[f"{b}|{l}"]
            print(f"{b:>7}  {l:<16}{m:>13.1f}{disc*100:>10.1f}%{pad*100:>8.1f}%"
                  f"{sat*100:>10.1f}%{star*100:>8.1f}%{cv:>8.3f}{sr:>11.3f}")
        print()
    json.dump({"shapes": n, "npoint": NPOINT, "radius": RADIUS, "nsample": NSAMPLE,
               "results": res}, open(Path(__file__).with_name("sa1_probe.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
