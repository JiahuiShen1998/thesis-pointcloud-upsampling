"""Apply the source-audited September 13 revision once to original chapter files."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def replace(name,old,new):
 p=R/name;t=p.read_text(encoding='utf-8')
 if new in t:return
 assert t.count(old)==1,(name,t.count(old),old[:100])
 p.write_text(t.replace(old,new),encoding='utf-8')
def before(name,anchor,new):replace(name,anchor,new+'\n\n'+anchor)
replace('texfiles/02_fundamentals.tex','OA is the principal classification quantity used in the final comparison, while mAcc provides a class-balanced complementary measure. The exact training and reporting protocol for PointNet++ is given in Chapter~\\ref{chap:setup}.',r'These definitions describe population-level metrics. An implementation may instead average intermediate batch accuracies, which need not give identical sample weights. The aggregation actually used by the classifier in this thesis is stated explicitly in Chapter~\ref{chap:setup}.')
replace('texfiles/04_experimental_setup.tex','256 points obtained by the fixed fourfold decimation used by the final preprocessing pipeline','256 points selected without replacement by the fixed, object-specific random sampling rule of the final preprocessing pipeline')
replace('texfiles/04_experimental_setup.tex','For ModelNet40, the final sparse representation contains 256 points selected from the fixed 1,024-point object representation by the final deterministic fourfold decimation pipeline.',r'For ModelNet40, the final sparse representation contains 256 points selected without replacement from the stored 1,024-point object. The archived HPC implementation uses NumPy random choice with a stable per-object seed derived from global seed 42, the downsampling operation, split, class name, and shape identifier. The seed helper hashes this token with MD5; the operation does not retain every fourth row and does not use farthest-point sampling \cite{shen2026modelnetupdated}.')
before('texfiles/04_experimental_setup.tex',r'\subsection{KITTI}',r'''The initial mesh sampler used Python's built-in hash of class, split, and
shape identifier in its seed. Its output is therefore not guaranteed to be
identical across fresh processes without controlling Python hash randomisation.
The stored point sets are the reproducible inputs to the completed runs;
the later stable-seed downsampler does not retrospectively remove this
limitation of rebuilding the original clouds from meshes
\cite{shen2026modelnetupdated}.''')
replace('texfiles/04_experimental_setup.tex',r'EAR \cite{huang2013ear} is fully executed in both final ModelNet40 lines and serves as the optimisation-based reference in the object-level study.',r'The project EAR-style geometric baseline, labelled EAR in the result tables, is fully executed on HPC in both final ModelNet40 lines. Edge-aware resampling provides its methodological background \cite{huang2013ear}; the archived project wrapper calls a migrated EAR-style implementation and does not establish an exact reproduction of the published algorithm \cite{shen2026modelnetupdated,shen2026kittiupdated}.')
replace('texfiles/04_experimental_setup.tex',r'''TULIP and SPU-PMD exploratory runs did not enter the final unified method
matrix. Their execution history is retained in the lab record, but the
main tables contain only the completed comparable conditions
\cite{shen2026kitti}.''',r'''Exploratory execution also covered TULIP, SPU-PMD and a strict KITTI
EAR-style wrapper. TULIP completed native range-image full-validation
evaluations, whereas SPU-PMD remained a four-frame feasibility run and
the strict EAR-style wrapper was checked on ten frames. These activities
are reported with their actual scope in Chapter~\ref{chap:results}; they
are not additional conditions in the common exact-fourfold matrix
\cite{shen2026kittiupdated}.''')
replace('texfiles/04_experimental_setup.tex','Overall accuracy and mean class accuracy are computed on the full 2,468-object test split.',r'''All 2,468 test objects are evaluated, but the saved metrics are averages of
batch-level ratios rather than the population formula in
Equation~\eqref{eq:fc-2-12}. The archived evaluator accumulates
\begin{equation}
 \mathrm{OA}_{\mathrm{rep}}=\frac{1}{B}\sum_{b=1}^{B}\frac{C_b}{n_b},
 \qquad a_c=\frac{1}{|\mathcal B_c|}\sum_{b\in\mathcal B_c}
 \frac{C_{bc}}{n_{bc}},\qquad
 \mathrm{mAcc}_{\mathrm{rep}}=\frac{1}{K}\sum_{c=1}^{K}a_c.
 \label{eq:reported-classification}
\end{equation}
Here $B$ is the number of test batches, $n_b$ and $C_b$ are their object
and correct-prediction counts, $\mathcal B_c$ contains batches with class
$c$, and $n_{bc}$ and $C_{bc}$ are the corresponding class-specific
counts. All $K=40$ classes are present. With batch size 24 and no dropped
test samples there are 102 full batches and a final batch of 20 objects.
Each batch has equal weight in reported OA, including the smaller last
batch; the class scores also average the ratios of batches containing
that class. Multiplication by 100 gives the percentages in Chapter~5.
The table labels OA and mAcc retain these recorded implementation
statistics throughout; they have not been converted into object-count
accuracies \cite{shen2026modelnetupdated}.''')
replace('texfiles/04_experimental_setup.tex',r'''The final PU-GCN full-validation matrix and the recorded CenterPoint multi-method matrix use AP$_{R40}$. Historical PointRCNN result tables produced with an older local evaluator that effectively followed an AP$_{R11}$-style sampling scheme remain development evidence and are not numerically mixed with the final AP$_{R40}$ tables.''',r'''The final PU-GCN matrix and the earlier CenterPoint multi-method matrix
use AP$_{R40}$. The archived PointRCNN E2/E3 runner independently confirms
C++ R40 evaluation: it averages the last 40 entries of each 41-entry
precision row. These earlier full-validation experiments are reported
under their own input protocols. In contrast, the 64-frame adaptation
screen explicitly used the repository's legacy Python evaluator. Its AP
values are retained under that label and are not pooled with R40 results
\cite{shen2026kittiprimary}.''')
before('texfiles/04_experimental_setup.tex',r'\section{Geometric and Task Evaluation Protocol}',r'''The wider HPC inventory contains 22 completed classifier jobs: the
12 canonical branches, two mesh-reference controls and eight earlier
jobs. Three earlier/final pairs have identical values in all 79 saved
weight tensors at seed 42. They demonstrate repeatability for those
stored inputs, not eight extra independent confirmations of the final
comparison. The earlier point-count protocols and their results are
retained in Section~\ref{sec:mn-history} \cite{shen2026modelnetupdated}.''')
before('texfiles/04_experimental_setup.tex',r'\subsection{KITTI detection evaluation}',r'''The secondary uniformity tables report mean absolute deviation from the
reference, not a signed difference of aggregate uniformity scores:
\begin{equation}
 D_{\mathrm{NUC}}=\frac{1}{S}\sum_{i=1}^{S}
 |\mathrm{NUC}(Y_i)-\mathrm{NUC}(R_i)|.
 \label{eq:nuc-deviation}
\end{equation}
Here $S$ is the number of evaluated objects, $Y_i$ the method output and
$R_i$ its designated reference. The absolute value is taken per object
before averaging. The independently chosen query centres remain a
limitation of this secondary comparison \cite{shen2026modelnetupdated}.''')
before('texfiles/05_modelnet_results.tex',r'\subsection{Densifying the Original Observation}',r'''OA and mAcc below are the recorded batch-aggregated statistics defined
in Equation~\eqref{eq:reported-classification}. All result values are
preserved from the HPC runs. This distinction matters when interpreting
small differences or comparing against externally reported accuracies
\cite{shen2026modelnetupdated}.''')
replace('texfiles/05_modelnet_results.tex',r'''despite lower OA. OA weights objects, while mAcc weights categories, so
an improvement concentrated in small categories can coexist with a reduction
in aggregate accuracy.''',r'''despite lower OA. Reported OA averages batch accuracies, whereas reported
mAcc averages the class-specific batch ratios across categories. Their
weights therefore differ, and a gain in one aggregate need not produce
a gain in the other. Without retained object predictions, a sample-weighted
recalculation or attribution to particular misclassified objects is not
possible.''')
before('texfiles/05_modelnet_results.tex',r'\subsection{Equal-Cardinality Geometry}',r'\input{texfiles/05_modelnet_history}')
replace('texfiles/05_kitti_results.tex',r'''used for the HPC ModelNet40 branch. The historical PointRCNN tables
also have an AP$_{R11}$/AP$_{R40}$ labelling risk. They are retained in
the source records but excluded from a numerical ranking against the
final AP$_{R40}$ matrix.''',r'''used for the HPC ModelNet40 branch. The historical PointRCNN E2/E3
results can be identified more precisely: their archived runner uses
C++ AP$_{R40}$. They are therefore included below with their own paired
baselines, while the legacy adaptation screen retains its different
metric label \cite{shen2026kittiprimary}.''')
before('texfiles/05_kitti_results.tex',r'\subsection{Patch Locality and Normalisation}',r'\input{texfiles/05_kitti_protocol_history}')
before('texfiles/05_kitti_results.tex',r'The training losses decreased.',r'\input{texfiles/05_kitti_adaptation_stages}')
before('texfiles/05_kitti_results.tex',r'A 256-frame PU-Net ablation isolated normalisation',r'''The first local PDANS replacement produced opposite detector responses
(Figure~\ref{fig:k-patch-response}). On the same 256 frames, PointRCNN
Car Moderate 3D AP fell from 57.18 to 43.70, while CenterPoint Car
rose from 61.12 to 77.42. The saved paired report also records
CenterPoint Pedestrian and Cyclist increases. Its locality rule guaranteed
only 256 unique supporting points for a 2,048-row patch; repeated filling
and overlapping patches concentrated the generated output. This was a
development configuration, distinct from the later unique-2,048-support
PU-GCN pipeline \cite{shen2026kittiprimary}.

\thesisfigure{k_patch_response}{Paired PDANS patch responses}{Paired
Moderate 3D AP from the archived 256-frame PDANS patch experiment. The
source report labels these values AP$_{R40}$; they retain that report's
protocol and are not inserted into the final adaptation matrix. Each
segment connects the old and first local patch result for one detector
and class. The CenterPoint values use the original evaluation logs named
in the paired report, which differ slightly from an older summary CSV
\cite{shen2026kittiprimary}.}{fig:k-patch-response}

The retained input audit supports a concentration explanation. Across 20
frames, the median fraction of PointRCNN input coordinates unique at
1~mm fell from 99.90\% to 57.70\%; occupied diagnostic voxels fell from
11,584.5 to 2,124.5, and the share of points in the densest 10\% of voxels
rose from 27.23\% to 87.85\%. CenterPoint preserved the original measured
points before voxelisation, whereas PointRCNN sampled 16,384 input rows.
The Car matching audit changed from 539 to 402 matches for PointRCNN
and from 621 to 748 for CenterPoint. These counts corroborate the
opposite observed changes, but do not replace AP or isolate a single
causal operation: locality, duplication and effective support changed
together \cite{shen2026kittiprimary}.

A 256-frame PU-Net ablation isolated normalisation''')
# The previous insertion includes its anchor wording once by replacement rather than duplication.
p=R/'texfiles/05_kitti_results.tex';t=p.read_text();t=t.replace('A 256-frame PU-Net ablation isolated normalisation\n\nA 256-frame PU-Net ablation isolated normalisation','A 256-frame PU-Net ablation isolated normalisation');p.write_text(t,encoding='utf-8')
before('texfiles/05_kitti_results.tex',r'\input{texfiles/05_kitti_completed_interventions}',r'\input{texfiles/05_kitti_sampling_scale}'+'\n'+r'\FloatBarrier')
replace('texfiles/06_conclusion.tex','original 1,024-point classifier. Original best OA','original 1,024-point classifier. Original best reported OA')
replace('texfiles/06_conclusion.tex',r'''The ModelNet40 classifiers used one seed and selected their best checkpoint
on the test set.''',r'''The ModelNet40 classifiers used one seed and selected their best checkpoint
on the test set. Their saved OA and class scores average batch-level
ratios; they are not exact object-count proportions. Initial mesh sampling
also used Python's process-dependent hash in its seed. Reusing the archived
point sets supports repeatability of the completed runs, whereas exact
regeneration from meshes requires additional hash-state control
\cite{shen2026modelnetupdated}.''')
before('texfiles/06_conclusion.tex',r'The historical full-validation object-transition audit',r'''The earlier PointRCNN E2/E3 controls show that enlarging the input budget
helped some dense-input pipelines but did not restore their paired
baselines. The first local PDANS patch gave opposite responses in the
two detectors, together with a measured concentration of PointRCNN's
sampled support. These results limit a purely geometric explanation of
detection changes. The 200-draw frame-subsampling probe and the separate
16-seed evaluator probe also exposed different sources of diagnostic
variability; neither provides training-seed uncertainty for the final
matrix \cite{shen2026kittiprimary}.''')
replace('texfiles/abstract.tex','No tested method improves ModelNet40 Line~A best overall accuracy over','Using the recorded batch-aggregated accuracy, no tested method improves\nModelNet40 Line~A best overall accuracy over')
replace('texfiles/kurzfassung.tex','Keine getestete Methode übertrifft in ModelNet40-Linie~A die beste','Bei der protokollierten, über Batches gemittelten Genauigkeit übertrifft\nkeine getestete Methode in ModelNet40-Linie~A die beste')
bib=r'''
@unpublished{shen2026modelnetupdated,
 author={Shen, Jiahui},
 title={{ModelNet40}: Updated Configuration, Code, Modification and Result Record},
 year={2026},
 note={HPC record updated 13 September 2026, 22-job inventory and archived preprocessing and evaluation source. Stored in evidence/modelnet40\_hpc/WORK\_RECORD\_20260913.md and primary\_20260913. Unpublished research records}
}
@unpublished{shen2026kittiupdated,
 author={Shen, Jiahui},
 title={Thesis Experiment Master Inventory: Updated {KITTI} Lab Execution Record},
 year={2026},
 note={Updated record archived 13 September 2026 in evidence/kitti\_lab/MASTER\_INVENTORY\_UPDATED\_20260913.md. Unpublished research record}
}
@unpublished{shen2026kittiprimary,
 author={Shen, Jiahui},
 title={{KITTI}: Archived Protocol Controls, Adaptation Screen and Diagnostic Source Data},
 year={2026},
 note={Lab primary CSV, JSON, paired patch report, 64-frame adaptation report, evaluator runner and subsampling code. Retrieved 13 September 2026; stored in evidence/kitti\_lab/primary\_20260913 with source paths and hashes in audit/manuscript\_revision\_20260913. Unpublished research records}
}
'''
p=R/'bibfiles/references.bib';t=p.read_text(encoding='utf-8')
if '@unpublished{shen2026modelnetupdated' not in t:p.write_text(t+bib,encoding='utf-8')
print('Updated original chapter sources and bibliography.')
