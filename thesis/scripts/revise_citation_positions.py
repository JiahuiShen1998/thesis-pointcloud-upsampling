"""Apply the reviewed citation-placement corrections to the original thesis."""
from pathlib import Path
import json

R = Path(__file__).resolve().parents[1]
A = R / 'audit/citation_positions_20260914'
changes = []

def replace(file, old, new, reason):
    p = R / 'texfiles' / file
    text = p.read_text(encoding='utf-8')
    assert text.count(old) == 1, (file, old[:100], text.count(old))
    p.write_text(text.replace(old, new), encoding='utf-8')
    changes.append(dict(file='texfiles/' + file, reason=reason, before=old, after=new))

replace('01_introduction.tex',
    'it responds to local neighbourhoods, point density, sampling patterns, foreground-background boundaries, occupied voxels, and finite input budgets. Additional generated points can therefore have several qualitatively different effects.',
    r'local grouping, point selection, foreground prediction, and spatial-cell aggregation mediate its input representation \cite{qi2017pointnetpp,shi2019pointrcnn,yin2021centerpoint}. These mechanisms motivate several possible effects of generated points, which this thesis tests rather than assumes.',
    'Attach network sources to their mechanisms and distinguish the thesis hypotheses.')
replace('01_introduction.tex',
    'These cases can obtain similar geometric scores while producing different task outcomes.',
    r'Whether these cases produce similar geometric scores but different task outcomes is examined in Chapter~\ref{chap:results}.',
    'Make the geometry-task proposition an explicit research question.')
replace('01_introduction.tex',
    'A real KITTI frame contains ground, multiple objects, background structures, occlusions, scan-line effects, metric scale, and abrupt depth discontinuities.',
    r'A real KITTI frame contains ground, multiple objects, background structures, occlusions, scan-line effects, metric scale, and abrupt depth discontinuities \cite{geiger2013kitti}.',
    'Place the dataset source beside the scene-characteristics claim.')
replace('01_introduction.tex', 'the voxel-based detector CenterPoint',
    'a voxel-based configuration of CenterPoint', 'Avoid treating every CenterPoint encoder as voxel-based.')

replace('02_fundamentals.tex',
    'A point cloud is an unordered finite set of samples in three-dimensional space. When only coordinates are considered, a point cloud with N points is written as',
    r'A point cloud is an unordered finite set of samples in three-dimensional space \cite{qi2017pointnet}. When only coordinates are considered, a point cloud with $N$ points is written as',
    'Cite the point-set representation at its introduction and typeset N.')
replace('02_fundamentals.tex',
    'The distance between neighbouring samples varies across the observed surface, and the number of points representing a physical object can vary strongly with sensor distance, object size, visibility, and sampling strategy. This irregularity is precisely why simply increasing the number of stored coordinates does not necessarily increase the amount of independent information available to a downstream model.',
    r'The distance between neighbouring samples varies across the observed surface \cite{qi2017pointnetpp}, and the number of points representing an object depends on the sensor and observation geometry \cite{geiger2013kitti}. In this thesis, stored cardinality and useful spatial support are therefore treated as separate quantities.',
    'Separate sourced sampling properties from the author evaluation rationale.')
replace('02_fundamentals.tex',
    'processes selected points, and CenterPoint constructs a voxel representation',
    'processes selected points, and a voxel-based CenterPoint configuration constructs spatial-cell features',
    'Qualify the CenterPoint encoding variant.')
replace('02_fundamentals.tex',
    'This hierarchy is relevant to upsampling because the distribution of generated points can change which centroids are selected and how many points fall inside each local neighbourhood.',
    r'The FPS and radius-grouping operations described above \cite{qi2017pointnetpp} motivate the following hypothesis: generated-point placement can change which centroids are selected and how many points fall inside each neighbourhood.',
    'Ground the architectural mechanism without attributing an untested upsampling effect to the paper.')
replace('02_fundamentals.tex',
    'A three-dimensional object detector predicts multiple objects in a scene. An oriented box can be parameterised as',
    r'A three-dimensional object detector predicts multiple objects in a scene. Following the oriented-box representation used by PointRCNN \cite{shi2019pointrcnn}, a box can be parameterised as',
    'Cite the box parameterisation before the equation.')
replace('02_fundamentals.tex',
    'Performance is therefore sensitive both to semantic evidence and to small geometric errors that change the overlap between predicted and ground-truth boxes.',
    r'Overlap-based evaluation is sensitive to localisation errors in predicted boxes \cite{simonelli2019disentangling}. Section~\ref{sec:fc-2-4-3} defines this overlap measure.',
    'Support the localisation/overlap claim and connect it to the definition.')
replace('02_fundamentals.tex',
    r'features from occupied voxels \cite{zhou2018voxelnet,yan2018second}. This architectural distinction is important in the present study because',
    r'features from occupied voxels \cite{shi2019pointrcnn,zhou2018voxelnet,yan2018second}. These mechanisms motivate a possible difference in response to upsampling:',
    'Add the point-based source; identify the subsequent mechanism argument as an inference.')
replace('02_fundamentals.tex',
    'horizontal offsets and orientation into bins and within-bin residuals.',
    r'horizontal offsets and orientation into bins and within-bin residuals \cite{shi2019pointrcnn}.',
    'Place the bin/residual source beside the specific method claim.')
replace('02_fundamentals.tex',
    'Second, practical PointRCNN pipelines use a finite point-sampling stage before the backbone.',
    r'Second, PointRCNN uses point sampling before its backbone \cite{shi2019pointrcnn}.',
    'Support the published sampling stage while keeping the thesis budget in Chapter 4.')
replace('02_fundamentals.tex',
    r"CenterPoint \cite{yin2021centerpoint} is an anchor-free detector that represents objects by their centres in bird's-eye view. Before the detection backbone, the raw point cloud is voxelised. Points assigned to the same voxel are aggregated into a voxel feature; sparse three-dimensional convolutions and a bird's-eye-view backbone then produce features from which object-centre heatmaps and box parameters are predicted \cite{yin2021centerpoint,zhou2018voxelnet,yan2018second}.",
    r"CenterPoint is an anchor-free detector that represents objects by their centres in bird's-eye view. Its centre-based prediction head can be combined with voxel or pillar encoders \cite{yin2021centerpoint}. In a voxel-based configuration, points in each cell are aggregated into features, and sparse three-dimensional convolutions produce a bird's-eye-view representation \cite{zhou2018voxelnet,yan2018second}. The head predicts object-centre heatmaps and box parameters from these features \cite{yin2021centerpoint}.",
    'Correct the distinction between the CenterPoint head and its possible encoders.')
replace('02_fundamentals.tex',
    r'For an input point \ensuremath{\mathbf{p}_i} , voxelisation can be represented abstractly as',
    r'Following regular-grid voxelisation \cite{zhou2018voxelnet}, the cell index of input point $\mathbf p_i$ can be written as',
    'Cite the voxelisation principle before the index equation.')
replace('02_fundamentals.tex',
    r'a denser set $\mathcal Y=U(\mathcal X)$. For nominal ratio $r$, its idealised',
    r'a denser set $\mathcal Y=U(\mathcal X)$. Following the fixed-ratio formulation of PU-Net \cite{yu2018punet}, for nominal ratio $r$ its idealised',
    'Support the upsampling formulation where it is defined.')
replace('02_fundamentals.tex',
    'A common evaluation paradigm compares the output to a dense reference surface using nearest-neighbour distances.',
    r'Upsampling studies compare outputs with dense point or mesh references using nearest-neighbour or point-to-surface distances \cite{yu2018punet,qian2021pugcn}.',
    'Cite actual evaluation practice and distinguish discrete and mesh references.')
replace('02_fundamentals.tex',
    'Chamfer distance (CD) measures the average nearest-neighbour discrepancy between two point sets. For an output',
    r'Chamfer comparisons use nearest-neighbour discrepancies between two point sets \cite{fan2017pointsetgen,qian2021pugcn}. For an output',
    'Cite the metric family before the formula without attributing the unsquared convention to squared-loss papers.')
replace('02_fundamentals.tex',
    'The symmetric form is\n\n',
    r'The symmetric max--min form is \cite{huttenlocher1993hausdorff}' + '\n\n',
    'Add a primary source for the exact Hausdorff definition.')
replace('02_fundamentals.tex',
    'Point-to-surface distance (P2F) measures the distance from generated points to a reference mesh rather than to another discrete point sample.',
    r'Point-to-surface distance (P2F) measures the distance from generated points to a reference mesh rather than to another discrete point sample \cite{yu2018ecnet,qian2021pugcn}.',
    'Place P2F sources beside its definition.')
replace('02_fundamentals.tex',
    r'Its interpretation requires the point cloud and surface to share one coordinate frame \cite{yu2018ecnet,qian2021pugcn}.',
    'To interpret it as a physical surface error, the point cloud and mesh must be expressed in a common coordinate frame.',
    'Separate the mathematical coordinate-frame condition from the literature attribution.')
replace('02_fundamentals.tex',
    'For predicted box $B_{\\mathrm{pred}}$ and ground-truth box $B_{\\mathrm{gt}}$, the intersection over union is',
    r'Using the volumetric overlap evaluated in 3D detection \cite{shi2019pointrcnn,simonelli2019disentangling}, the intersection over union of predicted box $B_{\mathrm{pred}}$ and ground-truth box $B_{\mathrm{gt}}$ is',
    'Add the missing 3D IoU source immediately before the equation.')

replace('03_concept.tex',
    'after PointNet++ loading, PointRCNN sampling, or CenterPoint voxelisation.',
    r'after PointNet++ loading, PointRCNN sampling, or voxel-based CenterPoint encoding; the latter stages follow the respective network mechanisms \cite{qi2017pointnetpp,shi2019pointrcnn,yin2021centerpoint}.',
    'Support borrowed network mechanisms while preserving the original comparison design.')

replace('04_experimental_setup.tex',
    'PU-Net, PU-GCN, PU-EdgeFormer, and PDANS cannot consume a complete KITTI scene as one object-level tensor under their released inference formulation.',
    r'The learned methods were developed for object or local-patch inputs \cite{yu2018punet,qian2021pugcn,kim2023puedgeformer,zhang2025pdans}. Their integrated inference paths in this project do not directly consume a complete KITTI scene as one tensor.',
    'Separate published method scope from the limitations of the implemented wrappers.')
replace('04_experimental_setup.tex',
    r'PU-Net \cite{yu2018punet} required substantial compatibility work before stable inference.',
    r'PU-Net uses hierarchical point features and learned feature expansion \cite{yu2018punet}. Its integration in this project required substantial compatibility work before stable inference.',
    'Do not use the original paper as evidence for the author compatibility work.')
replace('04_experimental_setup.tex',
    'The two branches have different execution provenance. On HPC, the final',
    r'The published PU-EdgeFormer model uses an edge transformer \cite{kim2023puedgeformer}. The two project branches have different execution provenance. On HPC, the final',
    'Identify the published architecture separately from the compatibility-path evidence.')
replace('04_experimental_setup.tex',
    r'PDANS \cite{zhang2025pdans} is implemented in PyTorch and required compatibility changes in its point-operator utilities,',
    r'PDANS combines conditional diffusion and adaptive noise suppression \cite{zhang2025pdans}. The PyTorch implementation used here required compatibility changes in its point-operator utilities,',
    'Cite PDANS architecture rather than incorrectly sourcing the project code changes.')
replace('04_experimental_setup.tex',
    r'The final ModelNet40 classifier is \texttt{pointnet2\_cls\_ssg} from the integrated PointNet++ implementation \cite{qi2017pointnetpp}.',
    r'The final ModelNet40 classifier uses the PointNet++ single-scale grouping architecture \cite{qi2017pointnetpp}, implemented here as \texttt{pointnet2\_cls\_ssg}.',
    'Separate the published architecture from the local implementation identifier.')
replace('04_experimental_setup.tex',
    'For NUC, the HPC code used 128 query centres and radii 0.02, 0.05, and',
    r'PU-Net introduced normalised uniformity coefficient (NUC) evaluation \cite{yu2018punet}. The project uses a neighbourhood-count variant, whose values are not claimed to reproduce the original NUC protocol. The HPC code used 128 query centres and radii 0.02, 0.05, and',
    'Cite the NUC concept and distinguish the executed project-specific formula.')
replace('04_experimental_setup.tex',
    'The project also stores per-class accuracy summaries. Because complete per-object logits were not preserved for every final run, a complete calibration analysis, McNemar test, and probability-level significance test were not performed.',
    r'The project stores per-class accuracy summaries, but complete per-object predictions were not preserved for every final run. A McNemar comparison of correlated correctness proportions \cite{mcnemar1947correlated} would require paired object-level outcomes; aggregate class scores are insufficient. That test was not performed. Complete probability outputs were also unavailable, so calibration and probability-level analyses were not performed.',
    'Correct the data requirements of paired prediction testing; do not imply that logits are necessary for McNemar.')
replace('04_experimental_setup.tex',
    r'0.5 for Pedestrian and Cyclist \cite{geiger2012kitti,simonelli2019disentangling}.',
    r'0.5 for Pedestrian and Cyclist \cite{kittieval2026}.',
    'Use the actual official 3D class-threshold source.')
replace('04_experimental_setup.tex',
    "matching and ignore rules. Precision is $P=\\mathrm{TP}/(\\mathrm{TP}+\\mathrm{FP})$",
    r'matching and ignore rules. Using the precision, recall, and interpolated-envelope definitions described for detection evaluation \cite{everingham2010voc}, precision is $P=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$',
    'Source the standard precision/recall/envelope definitions separately from the KITTI recall grid.')
replace('04_experimental_setup.tex',
    'The final implementation averages the 40 positive recall-grid entries:',
    r'The final implementation uses the 40 positive recall-grid entries proposed for KITTI AP$_{R40}$ \cite{simonelli2019disentangling}:',
    'Place the R40 source at its formula; do not source it to the VOC eleven-point rule.')
replace('04_experimental_setup.tex',
    'The analysis is descriptive rather than causal: a rank correlation or consistent ordering can indicate association, but it does not prove that improving a particular geometric metric causes better classification.',
    r'The archived category-level analysis uses Spearman rank correlation \cite{spearman1904association}. The reported geometry intervals use bootstrap resampling \cite{efron1979bootstrap}: objects are the resampling units for aggregate geometric errors, and categories for geometry--score associations. These analyses are descriptive. They do not identify a causal effect of a geometric metric on classification.',
    'Provide primary statistical-method sources; retain the actual resampling units and inference limits.')

replace('05_modelnet_results.tex',
    'The first set-abstraction layer provides an architectural explanation for\nthe limited return. In the retained reception probe,',
    r'The fixed-radius grouping and finite neighbourhoods of PointNet++ \cite{qi2017pointnetpp} motivate a possible explanation for the limited return. In the retained reception probe,',
    'Separate the sourced grouping mechanism from the measured reception evidence and causal uncertainty.')
replace('05_modelnet_results.tex',
    'object-bootstrap intervals. These intervals describe variation across test',
    r'object-bootstrap intervals, using the resampling principle of Efron \cite{efron1979bootstrap}. These intervals describe variation across test',
    'Cite the statistical method where geometric intervals are first interpreted.')
replace('05_modelnet_results.tex',
    'correlations and their bootstrap intervals, covering five methods,',
    r'Spearman rank correlations \cite{spearman1904association} and their bootstrap intervals \cite{efron1979bootstrap}, covering five methods,',
    'Support the statistical tools without citing external literature as evidence for the observed correlations.')

(A / 'changes.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'Applied {len(changes)} reviewed changes.')
