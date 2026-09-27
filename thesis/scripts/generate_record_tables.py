"""Reproduce added manuscript tables directly from archived primary results."""
from pathlib import Path
import csv,json
R=Path(__file__).resolve().parents[1];K=R/'evidence/kitti_lab/primary_20260913'
def read(n):
 with (K/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
e2=[r for r in read('e1_e2_live_ap_summary.csv') if r['experiment']=='E2'];e3=read('e3_real_first_32768_live_ap_summary.csv')
assert len(e2)==len(e3)==10
assert all(r['status']=='PASS' and int(r['prediction_files'])==3769 for r in e2+e3)
names=['Baseline','PDANS','PU-GCN','PU-EF compat','PU-Net old']
def key(line,j):
 if j==0:return 'original_baseline' if line=='A' else 'downsampled_x4_baseline'
 return ('original_x4_' if line=='A' else 'downsampled_x4_')+['','pdans','pu_gcn','pu_edgeformer','pu_net'][j]
def get(rows,k):return next(r for r in rows if r['variant']==k)
parts=[r'''\subsubsection{PointRCNN E2 and E3 Full-Validation Controls}
\label{sec:k-e2-e3}

The earlier E2 protocol evaluated a deterministic 16,384-point input
constructed after field-of-view and range filtering, using 0.1~m voxel
representatives and depth-stratified quotas. Its four depth intervals
were 0--20, 20--40, 40--60 and 60--70.4~m; largest-remainder allocation
assigned quotas from candidate counts without labels or AP-based
selection. E3 tested a 32,768-point, observed-first input. Both completed
ten conditions on all 3,769 validation frames. The primary CSV and
runner identify these as C++ AP$_{R40}$ results.

Figure~\ref{fig:k-e2-e3} and Tables~\ref{tab:k-e2-3d}--\ref{tab:k-e3}
retain this independent comparison. E2's original and sparse Car
Moderate baselines were 82.26 and 65.75. Among generated inputs, PDANS
was highest in both lines at 67.22 and 44.08. These figures describe
the integrated pipelines, including the old PU-Net error and
PU-EdgeFormer compatibility path; they do not rank faithful original
implementations.

\thesisfigure{k_e2_e3}{PointRCNN input-budget sensitivity}{Car Moderate
3D AP$_{R40}$ for the historical E2 (16,384-point) and E3 (32,768-point,
observed-first) inputs, each evaluated on 3,769 frames. Markers share a
0--100 axis. The change combines input budget and selection policy;
it is not a pure point-count effect. Baselines belong to these protocols,
not the later PU-GCN adaptation matrix.}{fig:k-e2-e3}
''']
for metric,label in [('3d','3D'),('bev','BEV'),('bbox','image-box')]:
 parts.append(r'\begin{table}[!t]\centering\setlength{\tabcolsep}{5pt}'+'\n'+r'\begin{tabular}{llrrr}\toprule'+'\n'+r'Line & Input & Easy & Moderate & Hard\\\midrule')
 for line in 'AB':
  for j,n in enumerate(names):
   r=get(e2,key(line,j));parts.append(f'{line} & {n} & '+ ' & '.join(f'{float(r[metric+"_ap_"+d]):.2f}' for d in ['easy','moderate','hard'])+r'\\')
  if line=='A':parts.append(r'\midrule')
 parts.append(r'\bottomrule\end{tabular}'+'\n'+r'\caption{Historical PointRCNN E2 '+label+r' AP$_{R40}$ for Car. Each row covers all 3,769 frames; the integrated input protocol is shared within E2. Values are percentages.}\label{tab:k-e2-'+metric+r'}\end{table}')
parts.append(r'''\begin{table}[!t]\centering\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrr}\toprule
Line & Input & Easy & Moderate & Hard & $\Delta$ Mod.\\\midrule''')
for line in 'AB':
 for j,n in enumerate(names):
  a=get(e2,key(line,j));b=get(e3,key(line,j));parts.append(f'{line} & {n} & '+' & '.join(f'{float(b["3d_ap_"+d]):.2f}' for d in ['easy','moderate','hard'])+f' & {float(b["3d_ap_moderate"])-float(a["3d_ap_moderate"]):+.2f}'+r'\\')
 if line=='A':parts.append(r'\midrule')
parts.append(r'''\bottomrule\end{tabular}
\caption{Historical PointRCNN E3 Car 3D AP$_{R40}$. $\Delta$ Mod. is E3
minus E2 for the same input name, in AP points. All ten rows have 3,769
prediction files.}\label{tab:k-e3}\end{table}

E3 raised dense-input PDANS from 67.22 to 70.06 and PU-GCN from
58.28 to 64.95, while the original baseline fell from 82.26 to 80.38.
The stronger generated-input recovery therefore coexisted with a
changed baseline, and all Line~A generated inputs remained below it.
For Line~B, PDANS was approximately unchanged (44.08 to 44.04),
PU-GCN declined slightly (28.68 to 28.28), and the sparse baseline
fell from 65.75 to 64.14. Enlarging the budget with observed-first
selection did not remove the downstream deficit. Because both budget
and point selection changed, these results establish sensitivity of
the complete input adapter rather than a causal effect of cardinality
alone.

\FloatBarrier
''')

(R/'texfiles/05_kitti_protocol_history.tex').write_text('\n\n'.join(parts),encoding='utf-8')
d=json.loads((K/'subset_size_power_probe.json').read_text());keys=list(d['by_size']['20']['paired_deltas'])
parts=[r'''\subsubsection{Frame-Subset Size and AP Stability}
\label{sec:k-subset-scale}

A separate completed probe varied which validation frames were scored,
while keeping saved detector predictions fixed. It drew 200 subsets
without replacement at each size of 20, 50, 100, 256 and 512 frames,
using seed 0 and paired frame identities for the original, sparse and
sparse-PDANS inputs. Car Moderate 3D AP$_{R40}$ was recomputed from the
saved predictions. This is variation from frame selection, distinct
from the preceding 16-seed detector-input sampling probe and from
retraining variability.

Figure~\ref{fig:k-subset-scale} and Table~\ref{tab:k-subset-scale} show
the standard deviation of each paired AP contrast. For the sparse
minus original contrast, it fell from 6.02 points at 20 frames to
1.26 at 512 frames. For PDANS minus sparse, the standard deviation was
4.78 at 50 frames and 2.27 at 256 frames. Quoting the former contrast's
variability as uncertainty of the latter would misstate the experiment.

\thesisfigure{k_subset_scale}{Frame-subset size and paired AP variability}{
Standard deviation across 200 random frame subsets per size, calculated
from fixed historical PointRCNN predictions. Each curve is a different
paired Car Moderate 3D AP$_{R40}$ contrast. These are observed
subsampling standard deviations, not confidence intervals for the final
matrix and not results of additional detector training.}{fig:k-subset-scale}

\begin{table}[!t]\centering\setlength{\tabcolsep}{5pt}
\begin{tabular}{rrrr}\toprule
Frames & Sparse $-$ original & PDANS $-$ original & PDANS $-$ sparse\\\midrule''']
for n in ['20','50','100','256','512']:
 parts.append(n+' & '+' & '.join(f'{d["by_size"][n]["paired_deltas"][k]["std"]:.2f}' for k in keys)+r'\\')
parts.append(r'''\bottomrule\end{tabular}
\caption{Sample standard deviations of paired Car Moderate AP contrasts
in AP points. All three contrasts use the same sampled frames within
a draw.}\label{tab:k-subset-scale}\end{table}

At 20 frames, the original-input AP's empirical 5th--95th percentile
range was 51.96--89.95, despite fixed model predictions. The mean was
75.63 compared with 81.99 on the complete source set. Small-subset AP
therefore changed both variability and average value, reflecting the
composition and recall support of the selected frames. This is a
reason to distinguish case diagnosis from full-validation evaluation.
The archived probe also multiplies standard deviation by 1.96 under a
sensitivity label. That number is only a descriptive scale here; it is
not a formal minimum detectable effect established by a power analysis.
No repeated-training uncertainty or extrapolated full-set confidence
interval is inferred from this probe.
''')
(R/'texfiles/05_kitti_sampling_scale.tex').write_text('\n\n'.join(parts),encoding='utf-8')
print('Generated E2/E3 and subset-size tables from primary CSV/JSON.')
