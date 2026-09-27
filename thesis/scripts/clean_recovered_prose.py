"""Repair PDF extraction artefacts while retaining the Formula_Clear prose."""
from pathlib import Path
import re
T=Path(__file__).resolve().parents[1]/'texfiles'

def replace_between(text,start,end,new):
    a=text.index(start); b=text.index(end,a)
    return text[:a]+new.strip()+'\n\n'+text[b:]

p=T/'02_fundamentals.tex';s=p.read_text(encoding='utf-8')
s=replace_between(s,'A sensor point may','A common characteristic',r'''
Here $N$ is the number of points, $i$ indexes a point, and $(x_i,y_i,z_i)$
are Cartesian coordinates. A sensor point may additionally carry attributes
such as reflectance, colour, or a surface normal. These attributes are
separate from the coordinate vector $\mathbf p_i$; for example, an XYZI
record stores $(x_i,y_i,z_i,\iota_i)$ with reflectance $\iota_i$.
The order of stored points is generally not part of their geometric meaning.
Point-cloud networks consequently use permutation-invariant or
permutation-equivariant operations\cite{qi2017pointnet}.
''')
s=replace_between(s,'For a rotating LiDAR','\\subsection{Object-Level',r'''
For a rotating LiDAR, fixed angular increments correspond to increasing
physical separation as range increases. Surface orientation, occlusion,
reflectance, and the sensor's scan pattern further affect the returns
\cite{geiger2013kitti}. Thus equally sized objects can have very different
point support. A single viewpoint also observes only visible surfaces;
missing returns are not equivalent to uniform removal from a complete
surface. These properties motivate the distinction below.
''')
s=replace_between(s,'The thesis uses one object classifier','\\subsection{Point-cloud classification',r'''
PointNet++ aggregates local metric neighbourhoods, PointRCNN directly
processes selected points, and CenterPoint constructs a voxel representation
\cite{qi2017pointnetpp,shi2019pointrcnn,yin2021centerpoint}.
These architectures illustrate different mechanisms by which additional
points can affect a downstream representation.
''')
s=replace_between(s,'For an object point cloud','PointNet \\cite',r'''
For an object point cloud $\mathcal P$, a classifier $f_\theta$ with learned
parameters $\theta$ produces class logits $\mathbf z=f_\theta(\mathcal P)
\in\mathbb R^C$, where $C$ is the number of classes. The predicted class
has the largest logit. Training commonly uses categorical cross-entropy
\cite{qi2017pointnet,qi2017pointnetpp}. Additional points influence the
prediction through the learned representation and relative class scores;
their number alone does not determine the output.
''')
s=replace_between(s,'where \\ensuremath{\\phi}','PointNet++ \\cite',r'''
Here $\phi_\theta$ is a shared multilayer perceptron, $\mathbf g$ is the
global feature, and $\max$ takes the maximum in each feature channel.
The symmetric maximum makes the representation invariant to point order.
''')
s=s.replace(r'\ensuremath{\mathcal{M}} \ensuremath{t}',r'$\mathcal M_t$')
s=replace_between(s,'If \\ensuremath{\\mathcal{N}}','\\begin{equation}\n\\mathbf{f}',r'''
For centroid $\mathbf c_j$ and radius $R$, let
$\mathcal N_j(R)=\{\mathbf p_i\in\mathcal P:
\|\mathbf p_i-\mathbf c_j\|_2\le R\}$ be its ball-query neighbourhood.
With input feature $\mathbf f_i$, one local feature is
''')
s=s.replace('The relative coordinate',r'Here $\mathbf f^{\prime}_j$ is the pooled region feature and brackets denote concatenation. The relative coordinate')
s=replace_between(s,'where (\\ensuremath{x}','Performance is therefore',r'''
Here $(x_c,y_c,z_c)$ is the box centre, $(l,w,h)$ are its length, width,
and height, and $\psi$ is its yaw angle. The detector additionally predicts
a class and confidence score\cite{shi2019pointrcnn}.
''')
s=replace_between(s,'The first-stage segmentation','\\begin{equation}\n\\mathbf{p}^{\\mathrm{can}}',r'''
PointRCNN uses focal-loss-style supervision for foreground--background
imbalance\cite{shi2019pointrcnn,lin2017focal}. Proposal regression divides
horizontal offsets and orientation into bins and within-bin residuals.
For a proposal centre $\mathbf c$ and yaw $\psi$, canonical coordinates are
''')
s=s.replace('This canonical transformation',r'Here $R_z(-\psi)$ rotates about the vertical axis by $-\psi$. This canonical transformation')
s=replace_between(s,'For a ground-truth centre projected','For an input point',r'''
CenterPoint supervises a centre heatmap with Gaussian targets in bird's-eye
view. At detected peaks, it predicts centre offsets, height, dimensions,
and orientation\cite{yin2021centerpoint}. Occupied spatial cells are
therefore a relevant intermediate representation, alongside raw point count.
''')
s=s.replace('where o is the lower spatial bound and v the voxel size.',r'Here $\mathbf k_i$ is the integer voxel index, $\mathbf o$ is the lower spatial bound, and $\mathbf v$ contains the voxel dimensions. The symbol $\oslash$ denotes component-wise division and the floor is applied in each coordinate.')
s=replace_between(s,'Given an input point set','\\begin{equation}\n|\\mathcal{Y}|',r'''
Given $\mathcal X=\{\mathbf x_i\}_{i=1}^N$, an upsampler $U$ constructs
a denser set $\mathcal Y=U(\mathcal X)$. For nominal ratio $r$, its idealised
cardinality is
''')
s=replace_between(s,'Conceptually, if h','\\subsection{PU-GCN}',r'''
Feature expansion maps $N$ sparse features to $rN$ output features before
coordinate reconstruction. Repulsion discourages clustered neighbours,
while reconstruction encourages agreement with the target surface
\cite{yu2018punet}. Such object-patch processing requires an explicit
adaptation when the input is a complete metric LiDAR scene.
''')
s=replace_between(s,'A typical graph feature update','\\subsection{PU-EdgeFormer}',r'''
The Inception DenseGCN feature extractor combines local graph features
at multiple neighbourhood scales. NodeShuffle expands feature channels
and rearranges them into additional point features, which a coordinate
head maps to dense points\cite{qian2021pugcn}. Figure~\ref{fig:pugcn-principle}
summarises these components. Patch selection and coordinate normalisation
determine the neighbourhoods supplied to this architecture.

\thesisfigure{pugcn_principle}{PU-GCN principle}{Schematic of feature
extraction, NodeShuffle expansion, and coordinate reconstruction, based on
the published method\cite{qian2021pugcn}.}{fig:pugcn-principle}
''')
s=s.replace('This describes the published architecture. The distinction between its native\nHPC execution and the lab compatibility path is an implementation issue and\nis documented in Chapter~\\ref{chap:setup}.','')
s=s.replace('the symmetric nonsquared form used in the executed ModelNet40 evaluation','a symmetric unsquared form')
s=replace_between(s,'The implementation used in the final equal-cardinality','Hausdorff distance',r'''
Here $\mathcal Y$ and $\mathcal R$ are the output and reference sets;
$|\cdot|$ denotes cardinality and $\|\cdot\|_2$ Euclidean distance.
The two terms measure output deviation and reference coverage.
Chamfer conventions vary: squared and unsquared distances must not be
numerically interchanged\cite{fan2017pointsetgen,qian2021pugcn}.
The executed convention and references are given in Chapter~\ref{chap:setup}.
''')
s=s.replace('Its interpretation requires the point cloud and surface to share one coordinate frame.',r'Its interpretation requires the point cloud and surface to share one coordinate frame\cite{yu2018ecnet,qian2021pugcn}. Local density statistics provide a complementary view of point distribution; their neighbourhood scale and reference must be specified\cite{li2019pugan}.')
s=s.replace('Mean class accuracy (mAcc)',r'Here $M$ is the number of test objects, $y_m$ and $\hat y_m$ are their true and predicted labels, and $\mathbb I$ equals one when its condition holds. Mean class accuracy (mAcc)')
s=s.replace(r'For two 3D boxes \ensuremath{B} pred and \ensuremath{B} gt',r'For predicted box $B_{\mathrm{pred}}$ and ground-truth box $B_{\mathrm{gt}}$')
s=s.replace('Predictions are sorted by confidence',r'Here $\operatorname{Vol}$ denotes volume, $\cap$ intersection, and $\cup$ union. Predictions are sorted by confidence')
s=s[:s.index('In Equation~\\eqref{eq:fc-2-1}')].rstrip()+'\n'
pos=s.index('\\subsection{Three-dimensional object detection}')
s=s[:pos]+r'''
Figure~\ref{fig:pointnet-principle} summarises the feature hierarchy.
\thesisfigure{pointnet_principle}{PointNet++ principle}{Local grouping,
shared feature extraction, and hierarchical pooling, redrawn from the
published method\cite{qi2017pointnetpp}.}{fig:pointnet-principle}

'''+s[pos:]
p.write_text(s,encoding='utf-8')

p=T/'03_concept.tex';s=p.read_text(encoding='utf-8')
s=replace_between(s,'The experimental object','\\begin{equation}',r'''
The experimental object is the combination of an upsampling method and
its integration protocol. Let $\mathcal X_s^{(b,l)}$ be the visible input
for sample $s$, branch $b\in\{M,K\}$ (ModelNet40 or KITTI), and line
$l\in\{A,B\}$. For method $m$ with fixed inference parameters $U_m$,
the method-native candidate set is
''')
s=s.replace(r'Here \ensuremath{d} geom denotes a geometric error and \ensuremath{\mu} task',r'Here $d_{\mathrm{geom}}$ denotes a geometric error and $\mu_{\mathrm{task}}$')
s=s.replace(r'Let \ensuremath{n_s} = |\ensuremath{\mathcal{X}} \ensuremath{s} | be the number of points visible to the method. A common assembly operator \ensuremath{S} 4',r'Let $n_s=|\mathcal X_s|$ be the visible point count. A common assembly operator $S_4$')
s=s.replace('No upsampling network is trained on the KITTI validation set or tuned using downstream AP.', 'No upsampling network is trained on the KITTI validation set. Wrapper revisions and diagnostic configurations were evaluated during development; the final comparison is therefore not a previously untouched external test.')
s=s.replace('This prevents an implicit oracle in which points are chosen because they happen to improve the test result.', 'The strict primary selector therefore does not use test outcomes to choose individual points. Detector-aware diagnostic selectors are separate experiments, identified as such in Chapter~\\ref{chap:setup}.')
s=s.replace('The work tests the following hypotheses.',r'The executed protocols\cite{shen2026modelnet,shen2026kitti,shen2026kittifinal} address the following hypotheses.')
p.write_text(s,encoding='utf-8')

p=T/'04_experimental_setup.tex';s=p.read_text(encoding='utf-8')
s=replace_between(s,'TULIP and SPU-PMD are not included','\\section{Software',r'''
TULIP and SPU-PMD exploratory runs did not enter the final unified method
matrix. Their execution history is retained in the lab record, but the
main tables contain only the completed comparable conditions
\cite{shen2026kitti}.
''')
s=s.replace('The main text must therefore not show an EAR value in a ``strict 4\\ensuremath{\\times}  full KITTI\'\' table. Earlier EAR detector results were obtained under historical protocols and can only be discussed as legacy evidence if required.', 'EAR therefore has no entry in the strict full-validation KITTI tables. Earlier detector measurements used a different protocol and are retained only in the archived work record.')
s=s.replace('The same object identity, class index, and split',r'Here $\boldsymbol\mu$ is the sample centroid, $s$ its maximum centred radius, and $\mathbf x_i^{\mathrm{norm}}$ the normalised coordinate. The same object identity, class index, and split')
s=s.replace('Each patch stores its own centre and scale',r'Here $\boldsymbol\mu_j$ and $s_j$ are the centroid and maximum radius of patch $\mathcal Q_j$; $\hat{\mathbf q}$ is a normalised input, $\hat{\mathbf q}_{\mathrm{out}}$ a network prediction, and $\mathbf q_{\mathrm{back}}$ its restored metric coordinate. Each patch stores its own centre and scale')
s=s.replace('The rule is deterministic, local',r'Here $\mathcal X_{\mathrm{obs}}$ is the visible measured set, $\mathbf g$ a generated coordinate, and $\iota$ reflectance. The rule is deterministic, local')
s=s.replace('a fixed 3,712/3,769 train/validation split.',r'a fixed 3,712/3,769 train/validation split\cite{chen2017mv3d,shen2026kitti}.')
s=s.replace('For PointRCNN, each adapted arm',r'The complete training and evaluation manifest is archived with the final matrix\cite{shen2026kittifinal}. For PointRCNN, each adapted arm')
s=s.replace('The frozen-detector experiments revealed that the input distribution created by upsampling differs from the detector\'s training distribution.', 'The frozen-detector experiments measured substantial degradation on upsampled inputs, motivating an input-distribution mismatch hypothesis.')
# Put completed-study parameters before the chapter summary, not after it.
a=s.index('\\section{Chapter Summary}');b=s.index('\\section{Additional Parameters',a)
summary=s[a:b];s=s[:a]+s[b:]+'\n'+summary
p.write_text(s,encoding='utf-8')

for p in T.glob('0[1-4]*.tex'):
    s=p.read_text(encoding='utf-8')
    corrections={'threedimensional':'three-dimensional','Pointcloud':'Point-cloud','trianglearea':'triangle-area','classbalanced':'class-balanced','perobject':'per-object','observedfirst':'observed-first','unitscale':'unit-scale','uniquesupport':'unique-support','enlargedinput':'enlarged-input','downstreaminput':'downstream-input','generateddose':'generated-dose','1024point':'1,024-point','im- portant':'important','512->2048':r'$512\rightarrow2,048$','256->1024':r'$256\rightarrow1,024$'}
    for a,b in corrections.items():s=s.replace(a,b)
    p.write_text(s,encoding='utf-8')
print('Repaired recovered Chapters 1--4.')
