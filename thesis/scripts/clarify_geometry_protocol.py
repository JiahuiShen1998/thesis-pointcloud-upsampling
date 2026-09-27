"""Clarify the audited geometry protocol without changing experiment values."""
from pathlib import Path
import shutil
R=Path(__file__).resolve().parents[1];Q=R/'audit/geometry_protocol_20260914'
def edit(name,old,new):
 p=R/name;t=p.read_text(encoding='utf-8')
 if new in t:return
 assert t.count(old)==1,(name,t.count(old))
 b=Q/'before'/name;b.parent.mkdir(parents=True,exist_ok=True)
 if not b.exists():shutil.copy2(p,b)
 p.write_text(t.replace(old,new),encoding='utf-8')
E=R/'evidence/modelnet40_hpc/primary_20260914';E.mkdir(exist_ok=True)
for p in (Q/'primary_hpc').iterdir():shutil.copy2(p,E/p.name)
edit('texfiles/02_fundamentals.tex',r'The executed convention and references are given in Chapter~\ref{chap:setup}.',r'''These nearest-neighbour definitions can compare nonempty sets of different
cardinalities; equal point counts are an experimental control, not a
mathematical precondition. Sampling density, reference choice and coordinate
normalisation still affect the interpretation. In particular, reproducing
input coordinates can yield zero distance to that input without adding
surface information. The executed convention and references are given in
Chapter~\ref{chap:setup} \cite{fan2017pointsetgen}.''')
edit('texfiles/04_experimental_setup.tex','followed by the same unit-sphere normalisation. These references are evaluation-side data;',r'''followed by the same unit-sphere normalisation algorithm applied independently
to each newly sampled point set. They do not reuse the centroid and radius
of Original 1024. These references are evaluation-side data;''')
anchor='The final CD implementation uses the non-squared Euclidean nearest-neighbour distance defined in Equation~\\eqref{eq:fc-2-10}.'
edit('texfiles/04_experimental_setup.tex',anchor,r'''Table~\ref{tab:geometry-counts} states the actual point-count controls.
The HPC evaluator resolves output and reference by the same split, class
and shape identifier, checks both array shapes against $(n,3)$, and
rejects a mismatched point count before computing distances. It uses the
stored arrays without evaluation-time resampling or alignment. The
archived test audit contains 14 conditions with 2,468 objects each,
34,552 rows in total, and zero failed rows \cite{shen2026geometryaudit}.

\begin{table}[!t]\centering\setlength{\tabcolsep}{4pt}
\begin{tabular}{L{0.35\linewidth}rL{0.30\linewidth}r}\toprule
Evaluated condition & Points & Reference & Points\\\midrule
Line A method output & 4,096 & Mesh-ref 4,096 & 4,096\\
Line B method output & 1,024 & Original 1,024 & 1,024\\
Line B sparse control & 256 & Mesh-ref 256 & 256\\\bottomrule
\end{tabular}
\caption{Equal-cardinality CD/HD comparisons on 2,468 test objects per
condition. The sparse-control comparison uses a different reference and
is not pooled with the restored-output ranking. Each shape's saved
reference is shared across the methods in its comparison
\cite{shen2026geometryaudit}.}\label{tab:geometry-counts}\end{table}

Equal cardinality does not remove every source of geometric discrepancy.
Original 1024 and the mesh-reference clouds were independently sampled,
centred and scaled using their own sample centroids and maximum radii.
Although each comparison uses the same saved reference for all methods,
the mesh-reference comparisons can include sampling-dependent translation
and scale differences. Their values describe the stored normalised point
sets; they are not absolute surface errors under a shared mesh-derived
transform. No common-transform reevaluation was performed in this thesis
\cite{shen2026geometryaudit}.

'''+anchor)
edit('texfiles/04_experimental_setup.tex','Query centres were not perfectly shared by object across methods. P2F',r'''Query centres were not perfectly shared by object across methods; the
seed also includes the method name and Python's built-in hash. For a
zero mean neighbour count, the implementation returns zero rather than
an undefined coefficient. NUC is therefore a project-specific diagnostic,
not an unconditional measure of surface quality. P2F''')
edit('texfiles/05_modelnet_results.tex','The rankings indicate a trade-off between\naverage fidelity and extreme local errors.',r'''The rankings indicate a trade-off between average residuals and extreme
local residuals under the retained normalised-reference protocol. Equal
point counts control cardinality, but the independently normalised mesh
reference can still contribute translation and scale differences
(Section~\ref{sec:fc-4-10-1}). These results are not presented as a
common-transform surface-error ranking.''')
edit('texfiles/06_conclusion.tex','classifier controls. P2F used independent cloud and mesh normalisation,',r'''classifier controls. Equal-cardinality mesh-reference comparisons also
use independently normalised point samples rather than one shared
mesh-derived transform. Their distance rankings are conditional on that
coordinate convention \cite{shen2026geometryaudit}. P2F used independent cloud and mesh normalisation,''')
edit('texfiles/06_conclusion.tex','seeds, regenerating all mesh-reference training controls, and retaining',r'''seeds, regenerating all mesh-reference training controls, evaluating
geometry with a shared mesh-derived transform, and retaining''')
p=R/'bibfiles/references.bib';t=p.read_text(encoding='utf-8')
if '@unpublished{shen2026geometryaudit' not in t:
 b=Q/'before/bibfiles/references.bib';b.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,b)
 t+=r'''
@unpublished{shen2026geometryaudit,
 author={Shen, Jiahui},
 title={{ModelNet40}: Equal-Cardinality Geometry Protocol, Evaluator and Build Audit},
 year={2026},
 note={HPC protocol and audit dated 3 August 2026; source snapshots retrieved 14 September 2026. Original geometry values unchanged. Archived in evidence/modelnet40\_hpc/primary\_20260914 with point-count and formula checks in audit/geometry\_protocol\_20260914. Unpublished research records}
}
''';p.write_text(t,encoding='utf-8')
print('Clarified geometry definitions, exact point-count checks, normalisation and diagnostic limits.')
