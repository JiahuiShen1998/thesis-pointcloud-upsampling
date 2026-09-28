"""Prepare local, source-bound tables; never run training or modify final/."""
from pathlib import Path
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / 'evidence/modelnet40_hpc'
K = ROOT / 'evidence/kitti_lab'
T = ROOT / 'tables'
T.mkdir(exist_ok=True)

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def tex_rows(name, rows):
    (T / name).write_text('\n'.join(' & '.join(map(str,r)) + r' \\' for r in rows)+'\n', encoding='utf-8')

matrix = []
for line in (K / 'FULL_VAL_MATRIX.md').read_text(encoding='utf-8-sig').splitlines():
    if line.startswith('| PointRCNN |') or line.startswith('| CenterPoint |'):
        v = [c.strip() for c in line.strip('|').split('|')]
        e,m,h = map(float, v[5].split('/'))
        matrix.append(dict(detector=v[0],line=v[1],input=v[2],weights=v[3],frames=int(v[4]),easy=e,moderate=m,hard=h,reported_delta=float(v[6]),status=v[7]))
assert len(matrix) == 20 and all(r['frames']==3769 and r['status']=='PASS' for r in matrix)
# Prefer original evaluator precision to the rounded Markdown transcription.
canonical = read_csv(K/'all_classes_ap_r40.csv')
for row in matrix:
    kind = 'baseline' if row['input'].startswith('baseline') else ('pugcn_direct' if 'direct' in row['input'] else 'pugcn_observed_first')
    arm = 'line_' + row['line'].lower() + '_' + kind + '_' + row['weights']
    record = next(x for x in canonical if x['detector']==row['detector'] and x['arm']==arm and x['class']=='Car' and x['metric']=='3d_ap_r40')
    for key in ['easy','moderate','hard']:
        assert abs(row[key]-float(record[key])) < 0.00011
        row[key] = float(record[key])
write_csv(K / 'full_val_ap_r40.csv', matrix)
for detector in ['PointRCNN', 'CenterPoint']:
    for line in ['A','B']:
        rows = []
        group = [r for r in matrix if r['detector']==detector and r['line']==line]
        for r in group:
            name = 'Baseline' if r['input'].startswith('baseline') else ('Direct' if 'direct' in r['input'] else 'Observed-first')
            base = next(b for b in group if b['input'].startswith('baseline') and (b['weights']==r['weights'] or line=='A'))
            delta = r['moderate']-base['moderate']
            rows.append([name, 'Adapted' if r['weights']=='adapted' else 'Frozen', *[f"{r[k]:.2f}" for k in ['easy','moderate','hard']],f'{delta:+.2f}'])
        tex_rows(f'kitti_{detector.lower()}_{line}.tex',rows)

for line in ['A','B']:
    cls=[r for r in read_csv(M/'data/classification_results.csv') if r['line']==line]
    tex_rows(f'modelnet_classification_{line}.tex',[[r['method'].replace('Downsampled x4 baseline','Sparse baseline'),f"{int(r['points']):,}",*[f"{float(r[k]):.2f}" for k in ['best_oa_percent','final_oa_percent','best_macc_percent','delta_vs_line_baseline_pp']]] for r in cls])
    g=read_csv(M/'data/geometry_equal_n_distribution_summary.csv')
    rows=[]
    for method in ['EAR','PDANS','PU-Net','PU-GCN','PU-EdgeFormer']:
        def val(metric):
            return next(r for r in g if r['line']==line and r['method']==method and r['metric']==metric)
        cd,hd,nuc=val('CD_L2'),val('HD'),val('abs_NUC_deviation')
        p2f = next(r for r in read_csv(M/'data/geometry_dense_gt_test_summary.csv')
                   if r['line']=='line'+line and r['method']==method and r['metric']=='p2f_mean')
        assert int(p2f['n']) == 2468
        rows.append([method,f"{float(cd['mean']):.4f}",f"[{float(cd['bootstrap_ci_low']):.4f}, {float(cd['bootstrap_ci_high']):.4f}]",f"{float(hd['mean']):.4f}",f"{float(nuc['mean']):.4f}",f"{float(p2f['mean']):.5f}"])
    tex_rows(f'modelnet_geometry_{line}.tex',rows)

# The old CenterPoint matrix is retained as a distinct integration generation.
record=(K/'WORK_RECORD.md').read_text(encoding='utf-8-sig')
part=record.split('### 5.2 ')[1].split('### 5.3 ')[0]
legacy=[]
for line in part.splitlines():
    if not line.startswith('| ') or line.startswith('|  Enter ') or line.startswith('|---'): continue
    v=[x.strip() for x in line.strip('|').split('|')]
    legacy.append(dict(input=v[0],car=float(v[1]),pedestrian=float(v[2]),cyclist=float(v[3]),scope='legacy frozen full-val',frames=3769))
assert len(legacy)==10
write_csv(K/'centerpoint_legacy_ap_r40.csv',legacy)

# No inferred AP values: these are the two recorded, explicitly scoped probes.
norm=[dict(configuration=n,pointrcnn=a,centerpoint=b,frames=256) for n,a,b in [
('Baseline',81.3033,79.8368),('Wrong / old',7.9823,15.9788),('Fixed / old',18.2742,41.2227),('Wrong / local',6.3008,16.4119),('Fixed / local',43.5424,46.2047)]]
write_csv(K/'normalization_patch_pilot256.csv',norm)
voxel=[dict(method=n,median_voxels=v,cap_frames=c,frames=32) for n,v,c in [
('Original',14944,0),('PDANS',36913,6),('PU-GCN',45108,29),('PU-EdgeFormer',46182,29),('PU-Net old',55384,32)]]
write_csv(K/'voxel_audit32.csv',voxel)

manifest=[]
for folder,origin in [(M,'HPC'),(K,'lab')]:
    for p in sorted(folder.rglob('*')):
        if p.is_file() and p.name!='manifest.json':
            manifest.append(dict(file=p.relative_to(ROOT).as_posix(),origin=origin,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(ROOT/'evidence/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(f'Prepared {len(matrix)} complete KITTI arms, native ModelNet tables, and {len(manifest)} source files.')
