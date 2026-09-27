#!/usr/bin/env python3
"""Why unequal-cardinality CD is biased: forward/backward decomposition on Line A.

For every Line A variant (5 real methods + a degenerate "repeat x4" control) we
compute, over the 2,468 test shapes:

  * CD against Original 1024   -- the OLD protocol. Note Original 1024 is also the
                                  *input* to Line A, so this asks "how little did
                                  you move your own input?"
      - cd_fwd = mean over output points  of dist(out, Original)
      - cd_bwd = mean over Original points of dist(orig, output)
  * CD against Mesh-ref 4096   -- the equal-N protocol, an independent draw from
                                  the same surface.

The repeat-x4 control tiles the Original cloud four times: it performs zero real
upsampling, and should expose the metric's bias if the bias exists.
"""
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/home/woody/iwnt/iwnt189h/thesis_pointcloud/modelnet40_pointnet2_upsampling")
ORIGINAL = ROOT / "datasets" / "modelnet40_original"
MESHREF4096 = ROOT / "datasets" / "modelnet40_mesh_ref_4096"
LINEA = ROOT / "datasets" / "lineA_original_up" / "strict_4N"
METHODS = ("ear", "pdans", "pu_net", "pu_gcn", "pu_edgeformer")


def nn(a, b):
    d, _ = cKDTree(b).query(a, k=1, workers=1)
    return np.asarray(d, dtype=np.float64)


def pair(a, b):
    fwd, bwd = nn(a, b), nn(b, a)
    return float(np.mean(fwd)), float(np.mean(bwd)), float(max(fwd.max(), bwd.max()))


def manifest(split="test"):
    rows = []
    with open(ORIGINAL / "metadata" / f"{split}_manifest.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r.get("class_name") or r.get("class"), r["shape_id"]))
    return rows


def one(item):
    cls, sid = item
    orig = np.load(ORIGINAL / "test" / cls / f"{sid}.npy").astype(np.float64)
    ref = np.load(MESHREF4096 / "test" / cls / f"{sid}.npy").astype(np.float64)
    out = {}
    variants = {m: LINEA / m / "test" / cls / f"{sid}.npy" for m in METHODS}
    for name, path in variants.items():
        arr = np.load(path).astype(np.float64)
        f, b, h = pair(arr, orig)          # old protocol: 4096 vs input 1024
        fe, be, he = pair(arr, ref)        # equal-N: 4096 vs mesh-ref 4096
        out[name] = (f, b, f + b, h, fe + be, he)
    rep = np.repeat(orig, 4, axis=0)       # degenerate control: no real upsampling
    f, b, h = pair(rep, orig)
    fe, be, he = pair(rep, ref)
    out["repeat_x4"] = (f, b, f + b, h, fe + be, he)
    return out


def main():
    items = manifest("test")
    if "--smoke" in sys.argv:
        items = items[:24]
    acc = {}
    n = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(one, it) for it in items]
        for fu in as_completed(futs):
            r = fu.result()
            n += 1
            for k, v in r.items():
                if k not in acc:
                    acc[k] = np.zeros(6)
                acc[k] += np.asarray(v)
    res = {k: (v / n).tolist() for k, v in acc.items()}
    print(f"shapes = {n}\n")
    hdr = f"{'variant':<16}{'CDold':>9}{'fwd':>9}{'bwd':>9}{'bwd/CD':>9}{'HDold':>9}{'CDeqN':>9}{'HDeqN':>9}"
    print(hdr)
    print("-" * len(hdr))
    for k in sorted(res, key=lambda x: res[x][2]):
        f, b, cd, h, cde, hde = res[k]
        print(f"{k:<16}{cd:>9.4f}{f:>9.4f}{b:>9.4f}{(b/cd*100 if cd>0 else 0.0):>8.1f}%{h:>9.4f}{cde:>9.4f}{hde:>9.4f}")
    json.dump({"shapes": n, "results": res},
              open(Path(__file__).with_name("cardinality_probe.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
