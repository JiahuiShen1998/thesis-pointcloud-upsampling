"""Complete front/back matter and assemble cited evidence into the thesis."""
from pathlib import Path
import csv,json,re,shutil,hashlib
R=Path(__file__).resolve().parents[1];T=R/'texfiles';M=R/'evidence/modelnet40_hpc/data';K=R/'evidence/kitti_lab'
def put(name,text): (T/name).write_text(text.strip()+'\n',encoding='utf-8')
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

put('06_conclusion.tex',r'''
\chapter{Conclusion and Outlook}\label{chap:conclusion}

This thesis investigated whether point-cloud upsampling improves downstream
perception, using object classification on ModelNet40 and three-dimensional
object detection on KITTI. The two-line design separated densification of
an available observation from recovery after fourfold sparsification.
Five upsampling methods completed the HPC object-level study. The lab
scene-level study combined earlier frozen-detector comparisons with a
later, fully evaluated PU-GCN input and detector-adaptation experiment
\cite{shen2026modelnet,shen2026kitti,shen2026kittifinal}.
The combined evidence shows that added coordinates, geometric agreement,
and useful task information are distinct properties.

\section{Findings from the Completed Experiments}

On ModelNet40, none of the five upsampled Line~A representations exceeded
the original 1,024-point classifier. Original best OA was 91.95\%, while
the highest upsampled best OA was 91.63\% for PU-GCN. After sparsification
to 256 points, PU-Net alone improved the selected-checkpoint result,
from 90.85\% to 91.27\%. Its final-epoch gain was smaller, from 90.46\%
to 90.65\%. These are results from separately trained, single-seed
classifiers, not repeated-run estimates\cite{shen2026modelnet}.

The geometric ranking did not reproduce the classification ranking.
PU-GCN had the lowest mean equal-cardinality Chamfer distance in both
lines, while PU-Net had the best Line~B classification and Hausdorff
distance. The reference-sensitivity analysis also showed that changing
the target point set can reverse apparent method ordering. The repeat
control and independent mesh references made this issue explicit.
PointNet++ reception measurements provided a further distinction between
stored density and the support used by local feature aggregation
(Sections~\ref{sec:geometry-task} and~\ref{sec:modelnet-results}).

The final KITTI study evaluated 20 completed arms on all 3,769 validation
frames. Frozen direct PU-GCN inputs reduced Car Moderate 3D AP$_{R40}$
for both PointRCNN and CenterPoint. Preserving observed points helped,
and detector adaptation recovered additional performance, but neither
detector exceeded its corresponding reference. In Line~B, adapted
observed-first inputs reached 55.21 for PointRCNN and 61.66 for CenterPoint,
compared with adapted sparse baselines of 68.33 and 68.05, respectively
\cite{shen2026kittifinal}. The improvement after adaptation is consistent
with an input-distribution mismatch; the remaining losses show that the
completed training procedure did not remove that mismatch.

The integration diagnostics explain why a method name alone is insufficient
to interpret a scene-level result. Local patch construction, correct
normalisation, measurement preservation, and the detector's point or voxel
budget all changed the evaluated input. Correcting PU-Net normalisation
and patch locality substantially improved its diagnostic result. Enlarging
PointRCNN input or changing CenterPoint occupancy did not yield a universal
remedy. These findings support evaluation of the complete implemented
pipeline, with the scope of every control stated explicitly
\cite{shen2026kitti}.

\section{Limits of the Evidence}

The ModelNet40 classifiers used one seed and selected their best checkpoint
on the test set. Reporting epoch~200 alongside best OA reduces dependence
on that selection in the presentation, but does not create an independent
validation protocol. Four reused training objects limit the mesh-reference
classifier controls. P2F used independent cloud and mesh normalisation,
and NUC query centres were not perfectly shared across methods; these
remain secondary geometry diagnostics. Missing per-object classifier
predictions prevent paired prediction-level tests.

The KITTI adaptation used a fixed three-epoch schedule and retained the
final checkpoints. Decreasing losses demonstrate completed optimisation,
but neither convergence nor an optimal training duration was established.
Line~A has no adapted original baseline, so its adapted system-level
comparison changes both the input and detector weights. The earlier
PU-Net full-validation pipeline had a normalisation error, and the lab
PU-EdgeFormer execution used a compatibility path. Historical evaluator
variants and small diagnostic subsets consequently cannot be combined
with the final matrix as one architecture ranking. Development also
examined validation outcomes; these results are not a new blind test.

\section{Implications and Future Work}

The practical implication is to assess upsampling after the actual task
model and its input adapter. A useful evaluation records visible
measurements, generated candidates, strict output, and effective model
input separately. It also distinguishes geometry references and training
regimes. Such reporting makes a limited recovery gain interpretable
without promoting it into a general claim about point-cloud densification.

Further experiments could strengthen the conclusions by separating
classifier validation from the fixed test set, repeating training across
seeds, regenerating all mesh-reference training controls, and retaining
per-object predictions. For KITTI, longer adaptation with intermediate
validation and an adapted original control would test whether the remaining
deficit persists beyond the completed schedule. Native implementations
under corrected common patch processing would permit cleaner method
comparisons. These are proposed extensions, not experiments performed in
this thesis.

Within the executed scope, geometric improvement was insufficient to
establish downstream improvement. PU-Net provided a limited object-level
recovery benefit, while observation preservation and detector adaptation
partially recovered KITTI performance. The evidence supports these
conditional findings rather than a universal benefit or universal failure
of point-cloud upsampling.
''')

put('abstract.tex',r'''
\chapter*{Abstract}\addcontentsline{toc}{chapter}{Abstract}
Point-cloud upsampling increases the number of spatial samples, but its
benefit for downstream perception cannot be inferred from geometric
distance alone. This thesis evaluates densification of an original input
and recovery after fourfold sparsification on two complementary tasks.
The ModelNet40 study compares EAR, PDANS, PU-Net, PU-GCN, and PU-EdgeFormer
using separately trained PointNet++ classifiers and equal-cardinality
geometric references. The KITTI study evaluates point-based PointRCNN
and voxel-based CenterPoint, with a final 20-arm PU-GCN input and detector-
adaptation matrix covering 3,769 validation frames per arm.

No tested method improves ModelNet40 Line~A best overall accuracy over
the original 1,024-point representation at 91.95\%. After sparsification,
PU-Net raises best accuracy from 90.85\% to 91.27\%, while PU-GCN achieves
the lowest mean Chamfer distance in both lines. On KITTI, direct insertion
degrades frozen detection. Preserving measured points and adapting the
detectors recover performance but leave deficits relative to the
corresponding baselines. PU-GCN itself retains its released pretrained
weights throughout this later study.

Patch-normalisation, point-reception, and voxel-occupancy diagnostics show
why stored density differs from effective task input. The findings are
bounded by single-seed classifier training, test-based best-checkpoint
selection, and a three-epoch detector-adaptation schedule without a
demonstrated convergence plateau. The combined evidence establishes a
limited object-level recovery benefit and partial scene-level recovery,
while showing that geometric fidelity alone is insufficient to establish
downstream utility.
''')
put('kurzfassung.tex',r'''
\chapter*{Kurzfassung}\addcontentsline{toc}{chapter}{Kurzfassung}
Die Hochabtastung von Punktwolken erhöht die Anzahl räumlicher Abtastpunkte.
Ein Nutzen für nachgelagerte Wahrnehmungsaufgaben lässt sich jedoch nicht
allein aus geometrischen Abständen ableiten. Diese Arbeit untersucht die
Verdichtung einer ursprünglichen Eingabe sowie die Rekonstruktion nach
einer Reduktion auf ein Viertel der Punkte. Auf ModelNet40 werden EAR,
PDANS, PU-Net, PU-GCN und PU-EdgeFormer mit separat trainierten
PointNet++-Klassifikatoren und geometrischen Referenzen gleicher
Punktanzahl verglichen. Auf KITTI werden PointRCNN und CenterPoint
untersucht. Die abschließende PU-GCN-Eingabe- und Detektoranpassungsstudie
umfasst 20 Auswertungen mit jeweils 3.769 Validierungsframes.

Keine getestete Methode übertrifft in ModelNet40-Linie~A die beste
Gesamtgenauigkeit der ursprünglichen Darstellung mit 1.024 Punkten
von 91,95\%. Nach der Ausdünnung verbessert PU-Net diese Genauigkeit
von 90,85\% auf 91,27\%, während PU-GCN in beiden Linien den niedrigsten
mittleren Chamfer-Abstand erreicht. Auf KITTI verschlechtern direkte
PU-GCN-Eingaben die Ergebnisse der unveränderten Detektoren. Der Erhalt
gemessener Punkte und die Anpassung der Detektoren verbessern die
Ergebnisse, übertreffen jedoch nicht die jeweiligen Referenzen. Das
PU-GCN-Netz selbst verwendet weiterhin seine veröffentlichten Gewichte.

Diagnostische Versuche zu Patch-Normalisierung, Punktverarbeitung und
Voxelbelegung zeigen, warum gespeicherte Punktdichte und wirksame
Modelleingabe verschieden sind. Die Aussagekraft wird durch nur einen
Trainingsseed, die Auswahl des besten Klassifikators anhand des Testsatzes
und eine Detektoranpassung über drei Epochen ohne nachgewiesenes
Konvergenzplateau begrenzt. Die Ergebnisse belegen einen begrenzten
Rekonstruktionsnutzen auf Objektebene und eine teilweise Erholung auf
Szenenebene. Geometrische Qualität allein reicht nicht aus, um einen
Nutzen für nachgelagerte Aufgaben nachzuweisen.
''')

put('symbols.tex',r'''
\chapter*{Notation}\addcontentsline{toc}{chapter}{Notation}
Symbols are defined at their first use. The following list collects the
principal quantities; method-specific indices are defined locally.
\begin{longtable}{ll}
\toprule Symbol & Meaning\\\midrule
$\mathcal P,\mathcal X$ & Input point sets\\
$\mathbf p_i,\mathbf x_i$ & Three-dimensional coordinates\\
$\iota_i$ & Reflectance associated with a measured point\\
$N_s,M_s$ & Original and sparse point counts in frame $s$\\
$U_m$ & Upsampling method $m$\\
$\mathcal C,\mathcal Y$ & Candidate and strict output sets\\
$\mathcal R$ & Geometric reference point set\\
$S_4,A_d$ & Cardinality operator and downstream adapter\\
$\boldsymbol\mu,s$ & Patch centroid and normalisation radius\\
$\phi_\theta,f_\theta$ & Learned feature map and classifier\\
$\mathcal N_j(R)$ & Radius-$R$ neighbourhood of centroid $j$\\
$y_m,\hat y_m$ & True and predicted class of test object $m$\\
$B_{\mathrm{pred}},B_{\mathrm{gt}}$ & Predicted and ground-truth boxes\\
$\Delta$ & Difference from the stated reference\\
\bottomrule\end{longtable}
''')
put('acronyms.tex',r'''
\chapter*{Abbreviations}\addcontentsline{toc}{chapter}{Abbreviations}
\begin{longtable}{ll}\toprule Abbreviation & Meaning\\\midrule
AP & Average precision\\
BEV & Bird's-eye view\\
CAD & Computer-aided design\\
CD & Chamfer distance\\
EAR & Edge-Aware Resampling\\
FPS & Farthest Point Sampling\\
HD & Hausdorff distance\\
HPC & High-performance computing\\
IoU & Intersection over union\\
LiDAR & Light Detection and Ranging\\
mAcc & Mean class accuracy\\
NUC & Local non-uniformity coefficient used in the evaluation\\
OA & Overall accuracy\\
P2F & Point-to-face distance\\
RCNN & Region-based convolutional neural network\\
RPN & Region proposal network\\
SSG & Single-scale grouping\\
XYZI & Cartesian coordinates with intensity\\
\bottomrule\end{longtable}
''')

# Archive-specific references identify original work without presenting it as
# external peer-reviewed literature.
bib=R/'bibfiles/references.bib';s=bib.read_text(encoding='utf-8')
if '@unpublished{shen2026modelnet' not in s:
    s+=r'''

%%%% Primary records of experiments performed for this thesis %%%%
@unpublished{shen2026modelnet,
  author={Shen, Jiahui},
  title={{ModelNet40}: Complete Experiment and Modification Record},
  year={2026},
  note={HPC experiment record, 10 September 2026. Archived with exported data in the thesis evidence/modelnet40\_hpc directory. Unpublished research record}
}
@unpublished{shen2026kitti,
  author={Shen, Jiahui},
  title={Executed Work, Experiments, Changes, and Results: {KITTI} Lab Record},
  year={2026},
  note={Lab work record, 10 September 2026. Archived in thesis evidence/kitti\_lab/WORK\_RECORD.md. Unpublished research record}
}
@unpublished{shen2026kittifinal,
  author={Shen, Jiahui},
  title={{PU-GCN} Detector Adaptation: Full-Validation Matrix and Protocol},
  year={2026},
  note={Lab primary records: 20 evaluation arms, 3769 validation frames per arm; full matrix, training protocol, convergence audit and class-wise AP CSV archived in thesis evidence/kitti\_lab. Unpublished research records}
}
@misc{kittieval2026,
  author={{KITTI Vision Benchmark Suite}},
  title={Object Detection Evaluation: {3D} Object Detection},
  howpublished={\url{https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d}},
  note={Evaluation rules, accessed 13 September 2026}
}
'''
s=s.replace('  Year      = {2025}\n}', '  Year      = {2025},\n  Pages     = {16987--16996}\n}')
bib.write_text(s,encoding='utf-8')

for name,key in [('05_modelnet_results.tex','shen2026modelnet'),('05_kitti_results.tex','shen2026kittifinal'),('05_joint_discussion.tex','shen2026modelnet,shen2026kitti,shen2026kittifinal')]:
    p=T/name;s=p.read_text(encoding='utf-8')
    # Add an explicit primary-source anchor to each results subsection.
    parts=re.split(r'(\\subsection\{[^}]+\}(?:\n\\label\{[^}]+\})?\n)',s)
    for i in range(2,len(parts),2):
        body=parts[i];end=body.find('\n\n',2)
        source='shen2026kitti' if name=='05_kitti_results.tex' and 'Diagnostics' in ''.join(parts[:i]) else key
        if end>=0 and '\\cite{'+source+'}' not in body[:end]:
            body=body[:end].rstrip()+r'\cite{'+source+'}'+body[end:]
        parts[i]=body
    s=''.join(parts).replace('input, mean\nRPN loss','input, median\nRPN loss')
    p.write_text(s,encoding='utf-8')
p=T/'05_results.tex';s=p.read_text(encoding='utf-8').replace('exported arrays, training logs, and tables.',r'exported arrays, training logs, and tables\cite{shen2026modelnet}.').replace('and diagnostic subsets retain their own protocol and evaluation scope.',r'and diagnostic subsets retain their own protocol and evaluation scope\cite{shen2026kitti,shen2026kittifinal}.');p.write_text(s,encoding='utf-8')

# Detailed detection results: exact values come from the lab CSV, not prose.
rows=read(K/'all_classes_ap_r40.csv');out=[r'\chapter{Detailed KITTI Detection Results}\label{app:kitti-full}',r'The following tables retain the full recorded class and difficulty results for the final lab matrix\cite{shen2026kittifinal}. Values are percent AP$_{R40}$. F and A denote frozen and adapted detector weights. Direct and observed-first identify the PU-GCN input construction. All rows evaluate 3,769 frames. The CSV in the evidence directory retains full precision.']
for det in ['PointRCNN','CenterPoint']:
    for metric,display in [('3d_ap_r40','3D'),('bev_ap_r40','BEV'),('bbox_ap_r40','Image-Box')]:
        subset=[r for r in rows if r['detector']==det and r['metric']==metric]
        out += [r'\section{'+det+' '+display+r' AP}',r'\begin{longtable}{llrrr}',r'\caption{'+det+' '+display+r' AP$_{R40}$ on the final validation set.}\\',r'\toprule Input / weights & Class & Easy & Moderate & Hard\\\midrule\endfirsthead',r'\toprule Input / weights & Class & Easy & Moderate & Hard\\\midrule\endhead',r'\bottomrule\endfoot']
        for r in subset:
            arm=r['arm'];line=arm.split('_')[1].upper();label='Baseline' if 'baseline' in arm else ('Observed' if 'observed' in arm else 'Direct');weights='A' if 'adapted' in arm else 'F'
            out.append(f"{line} / {label} / {weights} & {r['class']} & "+' & '.join(f"{float(r[k]):.2f}" for k in ['easy','moderate','hard'])+r' \\')
        out += [r'\end{longtable}']
put('appendix_a_quantitative_results.tex','\n'.join(out))

put('appendix_b_kitti_visualizations.tex',r'''
\chapter{Source and Geometry Audit}\label{app:geometry-audit}
The source package keeps the two execution branches separate. ModelNet40
tables, training curves, and point arrays are copied from the HPC chapter
package. KITTI work records and detector results are copied from the lab
records. Lab ModelNet40 smoke figures are not used. The file manifest
records SHA-256 hashes for the copied evidence, and the figure manifest
identifies the source and transformation for every plotted item
\cite{shen2026modelnet,shen2026kitti,shen2026kittifinal}.

\section{Secondary Geometry Results}
The archived ModelNet40 dense-reference audit computes point-to-face
distance after independently centring and normalising the point cloud
and mesh. Its nonzero baseline reflects this coordinate convention.
These values do not estimate absolute distance to the original surface
in a shared frame. Full per-object values and paired summaries are retained
in the evidence package, alongside equal-cardinality CD and HD, local
density summaries, reference sensitivity, and training curves. They are
provided for traceability, without pooling these different geometric
questions into one method ranking.
\input{tables/p2f_audit}

\section{Visualisation Scope}
The ModelNet40 plates use the supplied HPC airplane\_0627 and chair\_0890
arrays. One projection and common limits are used within each plate, with
every supplied point rendered. The KITTI plates use lab frame 006833.
The overview windows retain all points satisfying the stated metric
crop; the near/far example consists of spatial regions, not isolated
vehicles. Original and sparse scans are measured inputs. The two
generated arrays belong to the earlier direct integration and are not
presented as final detector-adaptation outputs. No detection boxes,
classification labels, or successful/failed predictions were generated
for these illustrations.

\section{Reproduction Files}
The thesis package contains the plotting source, table-generation source,
input CSV files and displayed point arrays. Figures are exported as vector
PDF and SVG with PNG previews. Each PDF uses the final thesis text width;
its text is 12 TeX points after inclusion. Native data are preserved in
the evidence directories, and generated tables retain more precise values
than the rounded prose where needed. The accompanying revision report
maps teacher comments to the revised chapters and records checks and
remaining experimental limitations.
''')

out=[r'\chapter{Reported ModelNet40 Class Scores}\label{app:class-results}',r'These tables reproduce the HPC exported class scores\cite{shen2026modelnet}. They are retained as reported aggregates. Several fractional values do not correspond to integer correct counts at the nominal category size; the retained exports do not justify reconstructing a confusion matrix, object-level predictions, or binomial confidence intervals. These tables are secondary to the logged overall accuracy. PU-EF abbreviates PU-EdgeFormer; the baseline is Original in Line A and Sparse in Line B.']
for line in 'AB':
    rr=read(M/f'classification_per_class_line{line}_wide.csv');base='Original' if line=='A' else 'Downsampled x4'
    out += [r'\section{Line '+line+'}',r'{\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{lrrrrrr}',r'\caption{Reported class scores (\%), Line '+line+r'.}\\',r'\toprule Class & Base & EAR & PDANS & PU-Net & PU-GCN & PU-EF\\\midrule\endfirsthead',r'\toprule Class & Base & EAR & PDANS & PU-Net & PU-GCN & PU-EF\\\midrule\endhead',r'\bottomrule\endfoot']
    for r in rr:out.append(r['class_name'].replace('_',' ')+ ' & '+' & '.join(f"{float(r[k]):.1f}" for k in [base,'EAR','PDANS','PU-Net','PU-GCN','PU-EdgeFormer'])+r' \\')
    out += [r'\end{longtable}}']
put('appendix_c_modelnet40_results.tex','\n'.join(out))

# Copy only literature actually cited in the compiled manuscript.
tex='\n'.join(p.read_text(encoding='utf-8') for p in T.glob('*.tex') if p.name not in ['cv.tex','appendix_d_supplementary_methods.tex','appendix_e_environment.tex','appendix_f_commands.tex'])
keys=sorted({k.strip() for group in re.findall(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}',tex) for k in group.split(',')})
index=(R.parent/'Literature/00_REFERENCE_INDEX.md').read_text(encoding='utf-8');papers=R/'bibfiles/papers';papers.mkdir(exist_ok=True)
manifest=[]
for key in keys:
    row=next((line for line in index.splitlines() if '| `'+key+'` |' in line),None)
    if not row:continue
    cells=row.split('|');names=re.findall(r'`([^`]+\.pdf)`',cells[3])
    if not names:continue
    name=names[0];matches=list((R.parent/'Literature').glob(name.replace('...','*')))
    if not matches:continue
    src=matches[0];dst=papers/(key+'.pdf');shutil.copy2(src,dst)
    manifest.append(dict(key=key,source=str(src.relative_to(R.parent)),copied=str(dst.relative_to(R)),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
(R/'bibfiles/reference_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(R/'audit/revision_20260913/citation_keys.json').write_text(json.dumps(keys,indent=2),encoding='utf-8')
print('Completed manuscript sections;',len(keys),'cited keys;',len(manifest),'local reference PDFs copied.')
