"""Targeted restoration from the archived author sources, with a review trail.

Run once after reviewing the adjacent restoration audit. Chapter 3--6 are frozen.
"""
from pathlib import Path
import re, json, hashlib

R = Path(__file__).resolve().parents[1]
A = R / 'audit/chapters_1_2_restoration_20260914'
original = (A/'archived_02_fundamentals.tex').read_text(encoding='utf-8-sig')

def between(start, end):
    return original[original.index(start):original.index(end)]

intro = (A/'archived_01_introduction.tex').read_text(encoding='utf-8-sig')
changes = []
def fix_intro(old, new, reason):
    global intro
    assert old in intro, old
    intro = intro.replace(old, new, 1)
    changes.append(dict(original=old, replacement=new, reason=reason))

fix_intro('preservation of the measured sensor points, and a frozen downstream model,',
          'accounting of measured and generated points, and explicitly specified downstream training conditions,',
          'Executed direct outputs do not universally retain observations; classification and detection use different training conditions.')
fix_intro('points for the same input, all measured points are preserved, and every\n    point is labelled as either observed or generated. This eliminates the\n    three most common confounding factors in such comparisons: inconsistent\n    output sizes, destroyed sensor measurements, and untraceable point\n    origins.',
          'points for the same input within a comparison line. Direct method\n    outputs and observation-preserving diagnostic inputs are distinguished\n    explicitly. The latter retain measured points and add generated rows\n    to the remaining budget. Output cardinality and measurement retention\n    can therefore be examined as separate conditions.',
          'Correct the unexecuted universal measurement-preservation claim; retain the contribution heading.')
fix_intro('eliminates the domain gap to which any negative result on \\ac{LiDAR}\n    data could otherwise be attributed.',
          'provides a controlled surface-sampling setting in which to examine\n    whether the same behaviour also occurs outside real sensor scenes.',
          'Object-level experiments do not mathematically eliminate every domain gap.')
fix_intro('so that adding points forces a selection rather than\n    adding information.',
          'so that adding points beyond that budget forces a selection.',
          'State the input-budget condition rather than a universal implication.')
fix_intro('Section~\\ref{sec:comparison_lines}', 'Section~\\ref{sec:fc-3-3}',
          'Repair the cross-reference to the existing comparison-line section.')
fix_intro('density, voxel occupancy, the point and voxel budget, and the consistency of\nthe generated points with the true measurements.',
          'density, voxel occupancy, the point and voxel budget, and the consistency of\nthe generated points with the true measurements~\\cite{shi2019pointrcnn,yin2021centerpoint}.',
          'Place the detector-method sources beside the corresponding external claims.')
training = r'''The primary detection comparison keeps the downstream detectors frozen.
A later, separate experiment adapts PointRCNN and CenterPoint to PU-GCN
inputs while retaining a fixed PU-GCN upsampler. This is detector adaptation,
not retraining of the upsampling network. The object-classification branch
uses separately trained PointNet++ SSG classifiers under the common training
recipe described in Chapter~\ref{chap:setup}; it is distinct from this later
detector-adaptation experiment.

'''
intro = intro.replace('\\section*{Outline}', training+'\\section*{Outline}')
changes.append(dict(reason='Preserve the user-requested distinction between KITTI detector adaptation and the ModelNet SSG classification protocol.', addition=training))
intro=re.sub(r'\\section\*\{[^{}]*\}\s*', '', intro)
(R/'texfiles/01_introduction.tex').write_text(intro,encoding='utf-8')
(A/'introduction_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')

# Restore the author's detailed classification explanation, then make local
# mathematical and attribution corrections. Other sections are reviewed fragments.
cls = between('\\section{Point Cloud Classification}', '\\section{3D Object Detection from Point Clouds}')
cls = cls.replace('single object and the output is its category $y$.',
    'single object and the output is its category $y$~\\cite{qi2017pointnet,qi2017pointnetpp}. Here $y\\in\\{1,\\ldots,C\\}$ and $C$ is the number of classes. The subscript $\\theta$ denotes learned parameters.')
cls = cls.replace('After the softmax operation,', 'Here $\\mathbf z$ is a vector of real-valued class scores, or logits. After the softmax operation,')
cls = cls.replace('Given the ground-truth label $y$,', 'Here $z_c$ is the score of class $c$, $j$ indexes all candidate classes, and $\\exp$ is the exponential function. The denominator makes the probabilities sum to one. Given the ground-truth label $y$,')
cls = cls.replace('and for a dataset containing $M$ samples the empirical risk is',
    'where $\\log$ is the natural logarithm and $\\hat p_y$ is the probability assigned to the true class. A confident correct prediction has a small loss; assigning little probability to the true class gives a large loss. For $M$ training samples the empirical risk is')
cls = cls.replace('The classification decision is', 'The superscript $(m)$ indexes a sample and $\\mathcal L_{\\mathrm{train}}$ is its sample-averaged training objective. These equations express the classification objective of PointNet-family networks~\\cite{qi2017pointnet,qi2017pointnetpp}. The classification decision is')
start = cls.index('the predicted class remains unchanged')
end = cls.index('\\subsection{PointNet}',start)
cls = cls[:start]+r'''a positive margin means that the true class has the largest score. Here
$z_y$ is the true-class logit and the maximum selects the strongest competing
class; a zero margin is a tie. For a fixed classifier, a change in point
count alone imposes no constraint on this margin. This observation follows
from the decision rule and is not an additional measured result. The
following architectures explain how the point coordinates become the scores
used by that rule.

'''+cls[end:]
cls = cls.replace('the feature\n', 'the feature $\\mathbf h_i\\in\\mathbb R^F$, where $F$ is the feature dimension,\n',1)
cls = cls.replace('Since the maximum is insensitive', 'Here $\\rho_\\theta$ maps the pooled feature $\\mathbf g$ to the class logits. Since the maximum is insensitive')
cls = cls.replace('map the points or the intermediate features into a more stable canonical', 'map the points or the intermediate features into a learned canonical')
cls = cls.replace('The advantages of PointNet are', r'''Here $\mathbf I$ is the identity matrix of the same size as the learned
square matrix $\mathbf A$, $\mathsf T$ denotes transpose, and
$\|\mathbf B\|_F^2=\sum_{u,v}B_{uv}^2$ is the squared Frobenius norm.
The regulariser penalises deviation from orthogonality; it does not impose
an exact rigid rotation~\cite{qi2017pointnet}.

The advantages of PointNet are''')
cls = cls.replace('determine that channel.', 'determine that channel~\\cite{qi2017pointnet}. Here $h_{i,d}$ is channel $d$ of point feature $\\mathbf h_i$.')
start=cls.index('Because it lacks hierarchical local modelling')
end=cls.index('\\subsection{PointNet++}',start)
cls=cls[:start]+r'''The shared mapping and global pooling do not explicitly build a hierarchy
of local neighbourhoods. PointNet++ addresses this limitation by applying
PointNet within nested metric regions~\cite{qi2017pointnetpp}. The progression
from global set aggregation to local geometric aggregation is therefore the
link between the two architectures.

'''+cls[end:]
cls=cls.replace('\\ac{FPS}~\\cite{eldar1997fps}', '\\ac{FPS}, as used in PointNet++~\\cite{qi2017pointnetpp}')
cls=cls.replace('Under a fixed random seed, \\ac{FPS} yields a deterministic and\nwell-distributed coverage of the input.',
    'Here $t$ counts selection steps and $\\mathcal M_t$ contains the centroids already selected. The inner minimum is the distance to the nearest selected centroid; maximising it chooses the least-covered remaining point. An initial centroid and a rule for resolving equal distances are needed to specify the selection completely.')
cls=cls.replace('and a fixed number of $k$ nearest neighbours may be used as well.',
    'Here $R>0$ is a radius in the coordinate units of the input; $j$ indexes centroids. A $k$-nearest-neighbour query instead fixes the number $k$ of neighbours and permits the neighbourhood radius to vary. The two alternatives are discussed in PointNet++~\\cite{qi2017pointnetpp}.')
cls=cls.replace('physical meaning in regions of differing\ndensity.', 'spatial scale in regions of differing\ndensity. For normalised coordinates this scale is relative to the normalisation radius, not a distance in metres.')
cls=cls.replace('where $\\mathbf{f}_i$ is the point feature', 'where $\\mathbf f^{\\prime}_j$ is the output feature, $[\\,\\cdot,\\cdot\\,]$ denotes concatenation, $\\mathcal N_j$ is the ball-query set, and $\\mathbf{f}_i$ is the point feature')
cls=cls.replace('where $\\Vert$ denotes feature concatenation.', 'Here $L$ is the number of scales, $\\ell$ indexes a scale, $R_\\ell$ is its radius, and $\\theta_\\ell$ denotes the parameters of that branch; $\\Vert$ denotes feature concatenation. Single-scale grouping (SSG) uses one radius per layer, whereas multi-scale grouping (MSG) concatenates several radii~\\cite{qi2017pointnetpp}.')
cls=cls.replace('Upsampling directly changes', 'From these sampling and grouping operations, upsampling can change')
cls+=r'''
Figure~\ref{fig:pointnet-principle} connects these operations in the feature hierarchy.
\thesisfigure{pointnet_principle}{PointNet++ principle}{Sampling, local
grouping and shared feature aggregation in PointNet++, redrawn from the
published architecture~\cite{qi2017pointnetpp}.}{fig:pointnet-principle}

'''

pieces=[(A/'draft_point_clouds.tex').read_text(encoding='utf-8'),cls,
        (A/'draft_detectors.tex').read_text(encoding='utf-8'),
        (A/'draft_upsampling.tex').read_text(encoding='utf-8'),
        (A/'draft_metrics.tex').read_text(encoding='utf-8')]
chapter='\n\\FloatBarrier\n'.join(pieces)
# Preserve labels referenced by the unchanged implementation chapter.
aliases={'eq:cd':'eq:fc-2-10','eq:hd':'eq:fc-2-11','eq:oa':'eq:fc-2-12'}
for old,new in aliases.items():chapter=chapter.replace('{'+old+'}', '{'+new+'}')
chapter=re.sub(r'\\paragraph\*\{[^{}]*\}\s*', '', chapter).replace('\\operatorname*{MAX}', '\\max')
(R/'texfiles/02_fundamentals.tex').write_text(chapter,encoding='utf-8')
master=(R/'thesis.tex').read_text(encoding='utf-8')
if '\\input{texfiles/cv}' not in master:
    master=master.replace('\\end{document}', '    \\cleardoublepage\n    \\input{texfiles/cv}\n\n\\end{document}')
    (R/'thesis.tex').write_text(master,encoding='utf-8')
# The user explicitly requested restoration of the existing CV placeholder.
assert (R/'texfiles/cv.tex').read_bytes()==(A/'before/texfiles/cv.tex').read_bytes()
expected=json.loads((A/'frozen_chapters_3_6.json').read_text())
assert all(hashlib.sha256((R/'texfiles'/n).read_bytes()).hexdigest()==v for n,v in expected.items())
print('Restored Chapters 1 and 2; CV placeholder included; Chapters 3--6 unchanged.')
