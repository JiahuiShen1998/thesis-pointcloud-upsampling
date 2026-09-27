#!/usr/bin/env python
"""How many val frames does a KITTI Car AP_R40 comparison actually need?

Pure re-analysis of already-completed full-3769-frame PointRCNN runs. Draws
random frame subsets of several sizes and re-evaluates AP on each, so the
spread of the estimator itself can be read off. Touches no detector input,
no GT-based selection -- prediction files and labels are read as-is.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = '/home/ra87racy/projects/baseline_detectors/PointRCNN'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import tools.kitti_object_eval_python.kitti_common as kitti  # noqa: E402
from tools.kitti_object_eval_python.eval import eval_class  # noqa: E402

LABEL_DIR = os.path.join(ROOT, 'data/KITTI/object/training/label_2')
VAL_SPLIT = os.path.join(ROOT, 'data/KITTI/ImageSets/val.txt')
EVAL_BASE = os.path.join(
    ROOT, 'results/kitti_unified_x4_pointrcnn_eval_20260716_201225/eval_outputs')
PRED_TAIL = 'inference/eval/epoch_no_number/val/final_result/data'

SOURCES = {
    'original_baseline': 'original_baseline',
    'downsampled_x4_baseline': 'downsampled_x4_baseline',
    'downsampled_x4_pdans': 'downsampled_x4_pdans',
}

# Car only, [num_minoverlap=2, metric=3, num_class=1]; row 0 is the 0.7 IoU set.
MIN_OVERLAPS = np.stack([
    np.array([[0.7], [0.7], [0.7]]),
    np.array([[0.7], [0.5], [0.5]]),
], axis=0)

METRIC_3D = 2
DIFFICULTIES = [0, 1, 2]


def mAP_r40(prec):
    """R40: mean of the 40 non-zero recall sample points (KITTI AP_R40)."""
    return prec[..., 1:41].sum(-1) / 40.0 * 100


def car_ap_r40_3d(gt_annos, dt_annos):
    """-> (easy, moderate, hard) Car 3D AP_R40 at IoU 0.7."""
    # the default 50 partitions leaves empty parts on small subsets
    num_parts = max(1, min(50, len(gt_annos) // 4))
    ret = eval_class(gt_annos, dt_annos, [0], DIFFICULTIES, METRIC_3D,
                     MIN_OVERLAPS, num_parts=num_parts)
    ap = mAP_r40(ret['precision'])  # [cls, diff, minoverlap]
    return ap[0, 0, 0], ap[0, 1, 0], ap[0, 2, 0]


def count_car_gt(gt_annos, idx):
    """Car GT boxes in a subset, and how many clear the moderate cut."""
    total = 0
    moderate = 0
    for i in idx:
        a = gt_annos[i]
        names = a['name']
        for k, nm in enumerate(names):
            if nm != 'Car':
                continue
            total += 1
            h = a['bbox'][k][3] - a['bbox'][k][1]
            occ = a['occluded'][k]
            trunc = a['truncated'][k]
            if h >= 25 and occ <= 1 and trunc <= 0.30:
                moderate += 1
    return total, moderate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', type=int, nargs='+',
                    default=[20, 50, 100, 256, 512])
    ap.add_argument('--reps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(VAL_SPLIT) as f:
        val_ids = [int(x) for x in f.read().split()]
    n = len(val_ids)
    print(f'val frames: {n}', flush=True)

    print('loading GT ...', flush=True)
    gt_annos = kitti.get_label_annos(LABEL_DIR, val_ids)

    dt = {}
    for tag, sub in SOURCES.items():
        d = os.path.join(EVAL_BASE, sub, PRED_TAIL)
        print(f'loading predictions: {tag}', flush=True)
        dt[tag] = kitti.get_label_annos(d, val_ids)

    print('full-set reference AP (all 3769 frames):', flush=True)
    full = {}
    for tag in SOURCES:
        e, m, h = car_ap_r40_3d(gt_annos, dt[tag])
        full[tag] = {'easy': e, 'moderate': m, 'hard': h}
        print(f'  {tag:26s} 3D E/M/H = {e:.4f} / {m:.4f} / {h:.4f}', flush=True)

    rng = np.random.default_rng(args.seed)
    out = {'n_val': n, 'reps': args.reps, 'seed': args.seed,
           'full_set': full, 'by_size': {}}

    for size in args.sizes:
        print(f'\n=== subset size {size} ({args.reps} draws) ===', flush=True)
        rec = {tag: [] for tag in SOURCES}
        gtc, gtm = [], []
        for r in range(args.reps):
            idx = rng.choice(n, size=size, replace=False)
            gsub = [gt_annos[i] for i in idx]
            c, mo = count_car_gt(gt_annos, idx)
            gtc.append(c)
            gtm.append(mo)
            for tag in SOURCES:
                dsub = [dt[tag][i] for i in idx]
                rec[tag].append(car_ap_r40_3d(gsub, dsub))
            if (r + 1) % 50 == 0:
                print(f'  {r + 1}/{args.reps}', flush=True)

        entry = {'car_gt_total_mean': float(np.mean(gtc)),
                 'car_gt_moderate_mean': float(np.mean(gtm)),
                 'car_gt_moderate_p05': float(np.percentile(gtm, 5)),
                 'car_gt_moderate_p95': float(np.percentile(gtm, 95)),
                 'sources': {}}
        arrs = {}
        for tag in SOURCES:
            a = np.array(rec[tag])  # [reps, 3]
            arrs[tag] = a
            mod = a[:, 1]
            entry['sources'][tag] = {
                'moderate_mean': float(mod.mean()),
                'moderate_std': float(mod.std(ddof=1)),
                'moderate_p05': float(np.percentile(mod, 5)),
                'moderate_p95': float(np.percentile(mod, 95)),
                'moderate_min': float(mod.min()),
                'moderate_max': float(mod.max()),
                'full_set_moderate': full[tag]['moderate'],
            }
            print(f'  {tag:26s} mod mean {mod.mean():7.3f}  sd {mod.std(ddof=1):6.3f}'
                  f'  [p5 {np.percentile(mod, 5):7.3f}, p95 {np.percentile(mod, 95):7.3f}]',
                  flush=True)

        # paired deltas: same frames for both arms, exactly how a real
        # side-by-side comparison would be run
        entry['paired_deltas'] = {}
        pairs = [('original_baseline', 'downsampled_x4_baseline'),
                 ('original_baseline', 'downsampled_x4_pdans'),
                 ('downsampled_x4_baseline', 'downsampled_x4_pdans')]
        for a_tag, b_tag in pairs:
            d = arrs[b_tag][:, 1] - arrs[a_tag][:, 1]
            true_d = full[b_tag]['moderate'] - full[a_tag]['moderate']
            key = f'{b_tag}_minus_{a_tag}'
            entry['paired_deltas'][key] = {
                'true_delta_full_set': float(true_d),
                'mean': float(d.mean()),
                'std': float(d.std(ddof=1)),
                'p05': float(np.percentile(d, 5)),
                'p95': float(np.percentile(d, 95)),
                'sign_flip_rate': float(np.mean(np.sign(d) != np.sign(true_d))),
                # smallest true delta a single run at this size could call
                # non-zero with ~95% confidence
                'mde_95': float(1.96 * d.std(ddof=1)),
            }
            print(f'  delta {key[:44]:44s} true {true_d:7.3f} '
                  f'sd {d.std(ddof=1):6.3f} flip {np.mean(np.sign(d) != np.sign(true_d)):.3f}',
                  flush=True)

        out['by_size'][str(size)] = entry
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)

    print(f'\nwrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
