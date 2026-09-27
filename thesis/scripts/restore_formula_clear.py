"""Recover the supplied revised prose, typeset equations, and apply traced edits.

The PDF is the author-confirmed Chapters 1--4 baseline. The untouched PDF and
block extraction are retained so paragraph preservation can be checked.
"""
from pathlib import Path
import sys,re,unicodedata,json,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.python-deps'))
import pymupdf
SOURCE=ROOT.parent/'final/Thesis_Chapters_1_4_Formula_Clear.pdf'
AUDIT=ROOT/'audit/revision_20260913'
shutil.copy2(SOURCE,AUDIT/'CHAPTERS_1_4_BASELINE.pdf')
doc=pymupdf.open(SOURCE)
CITES=['geiger2012kitti','geiger2013kitti','huang2013ear','yu2018punet','qian2021pugcn','kim2023puedgeformer','zhang2025pdans','fan2017pointsetgen','wu2015modelnet','qi2017pointnetpp','shi2019pointrcnn','yin2021centerpoint','qi2017pointnet','eldar1997fps','zhou2018voxelnet','yan2018second','lin2017focal','openpcdet2020','zhou2019objectsaspoints','li2019pugan','wang2019mpu','yu2018ecnet','lipman2007lop','huang2009wlop','alexa2003pss','wang2019dgcnn','vaswani2017attention','ho2020ddpm','luo2021diffusionpc','shu2019treegcn','song2021ddim','yang2024tulip','liu2021swin','everingham2010voc','simonelli2019disentangling','zhou2018open3d']

def cite(m):
    nums=[]
    for part in m.group(1).split(','):
        if '-' in part:
            a,b=map(int,part.split('-'));nums.extend(range(a,b+1))
        else:nums.append(int(part))
    return r'\cite{'+','.join(CITES[n-1] for n in nums)+'}'

MATH={'𝒫':r'\mathcal{P}','𝒳':r'\mathcal{X}','𝒴':r'\mathcal{Y}','𝒞':r'\mathcal{C}','ℛ':r'\mathcal{R}','𝒬':r'\mathcal{Q}','𝒩':r'\mathcal{N}','ℳ':r'\mathcal{M}','𝒮':r'\mathcal{S}','𝒯':r'\mathcal{T}','𝒪':r'\mathcal{O}','ℝ':r'\mathbb{R}','𝕀':r'\mathbb{I}','𝜃':r'\theta','𝜙':r'\phi','𝜌':r'\rho','𝛼':r'\alpha','𝛽':r'\beta','𝜇':r'\mu','𝜎':r'\sigma','𝜅':r'\kappa','𝜓':r'\psi','𝜄':r'\iota','𝜖':r'\epsilon','Δ':r'\Delta','Ω':r'\Omega','Λ':r'\Lambda','∈':r'\in','∼':r'\sim','∝':r'\propto','⊤':r'\top','×':r'\times','→':r'\rightarrow','⇒':r'\Rightarrow','⊎':r'\uplus','∣':r'\mid','‖':r'\Vert','⌊':r'\lfloor','⌋':r'\rfloor','∩':r'\cap','∪':r'\cup','⋅':r'\cdot','′':r"'",'−':'-','√':r'\sqrt{}','∑':r'\sum'}
for code in range(0x1d400,0x1d800):
    ch=chr(code);v=unicodedata.normalize('NFKC',ch)
    if len(v)==1 and v.isascii() and v.isalnum():MATH.setdefault(ch,v)
def tex(s):
    s=re.sub(r'(?<=\w)-\n(?=\w)','',s)
    s=re.sub(r'\s+',' ',s).strip()
    s=s.replace('2$×$2','2 × 2').replace('AP_R40',r'AP$_{R40}$').replace('AP_R11',r'AP$_{R11}$')
    s=s.replace('pointcloud','point-cloud').replace('PointRCNN','PointRCNN').replace('4x','4×')
    s=s.replace('’',"'").replace('“','``').replace('”',"''").replace('–','--').replace('—','---').replace('•','')
    s=s.replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
    # Preserve source variable meaning while restoring common lost subscript layout.
    for src,dst in [('𝑁𝑠','N_s'),('𝑀𝑠','M_s'),('𝑛𝑠','n_s'),('𝑈𝑚','U_m'),('𝐴𝑑','A_d'),('𝜇𝑗',r'\mu_j'),('𝑠𝑗','s_j'),('𝒬𝑗',r'\mathcal{Q}_j'),('p𝑖',r'\mathbf{p}_i'),('q𝑖',r'\mathbf{q}_i'),('x𝑖',r'\mathbf{x}_i'),('f𝑖',r'\mathbf{f}_i'),('c𝑗',r'\mathbf{c}_j'),('𝑥𝑖','x_i'),('𝑦𝑖','y_i'),('𝑧𝑖','z_i'),('𝑟𝑖','r_i')]:
        s=s.replace(src,r'\ensuremath{'+dst+'} ')
    for ch,rep in MATH.items():
        if ch in s:s=s.replace(ch,r'\ensuremath{'+rep+'} ')
    s=s.replace('̂','').replace('̃','').replace('̄','')
    s=re.sub(r'\[([0-9]+(?:[-,][0-9]+)*)\]',cite,s)
    s=re.sub(r'Equation \(([234])\.(\d+)\)',lambda m:'Equation~\\eqref{eq:fc-'+m[1]+'-'+m[2]+'}',s)
    s=re.sub(r'Section ([234])\.(\d+)(?:\.(\d+))?',lambda m:'Section~\\ref{sec:fc-'+m[1]+'-'+m[2]+('-'+m[3] if m[3] else '')+'}',s)
    s=re.sub(r'Chapter ([1-6])',lambda m:'Chapter~\\ref{chap:'+{'1':'introduction','2':'fundamentals','3':'concept','4':'setup','5':'results','6':'conclusion'}[m[1]]+'}',s)
    s=s.replace('fps_ball_cover_knn_v3',r'\texttt{fps\_ball\_cover\_knn\_v3}')
    return s

# Equation spans are explicitly restored from the author-confirmed PDF.
EQ={
(8,6,7):('2-1',r'\mathcal{P}=\{\mathbf{p}_i\}_{i=1}^{N},\qquad \mathbf{p}_i=(x_i,y_i,z_i)^{\top}\in\mathbb{R}^{3}.'),
(9,1,1):('2-2',r'n(\rho)\propto\frac{A\cos\theta}{\rho^2\Delta\alpha\Delta\beta}.'),
(9,8,8):('2-3',r'\mathcal{P}_{\mathrm{obj}}\sim p(\mathcal{P}\mid\mathcal{S},\text{surface sampling}).'),
(9,10,10):('2-4',r'\mathcal{P}_{\mathrm{LiDAR}}\sim p(\mathcal{P}\mid\mathcal{S},\mathcal{T},\Omega,\mathcal{O},\Lambda).'),
(10,9,9):('2-5',r'\mathbf{g}=\max_{\mathbf{p}_i\in\mathcal{P}}\phi_{\theta}(\mathbf{p}_i).'),
(11,2,5):('2-6',r'\mathbf{c}_{t+1}=\operatorname*{arg\,max}_{\mathbf{p}_i\in\mathcal{P}}\min_{\mathbf{c}\in\mathcal{M}_{t}}\|\mathbf{p}_i-\mathbf{c}\|_2.'),
(11,8,9):('pool',r'\mathbf{f}^{\prime}_j=\max_{\mathbf{p}_i\in\mathcal{N}_j(R)}\phi_{\theta}([\mathbf{p}_i-\mathbf{c}_j,\mathbf{f}_i]).'),
(11,15,15):('2-7',r'\mathbf{b}=(x_c,y_c,z_c,l,w,h,\psi)^{\top}.'),
(12,5,6):('canonical',r'\mathbf{p}^{\mathrm{can}}_i=R_z(-\psi)(\mathbf{p}_i-\mathbf{c}).'),
(13,3,5):('2-8',r'\mathbf{k}_i=\left\lfloor(\mathbf{p}_i-\mathbf{o})\oslash\mathbf{v}\right\rfloor.'),
(13,13,13):('2-9',r'|\mathcal{Y}|=r|\mathcal{X}|.'),
(14,3,5):('omit',''),(15,7,9):('omit',''),(16,2,2):('omit',''),
(16,11,19):('2-10',r'\begin{split}\operatorname{CD}(\mathcal{Y},\mathcal{R})={}&\frac{1}{|\mathcal{Y}|}\sum_{\mathbf{y}\in\mathcal{Y}}\min_{\mathbf{r}\in\mathcal{R}}\|\mathbf{y}-\mathbf{r}\|_2\\&+\frac{1}{|\mathcal{R}|}\sum_{\mathbf{r}\in\mathcal{R}}\min_{\mathbf{y}\in\mathcal{Y}}\|\mathbf{r}-\mathbf{y}\|_2.\end{split}'),
(17,2,8):('2-11',r'\operatorname{HD}(\mathcal{Y},\mathcal{R})=\max\left\{\max_{\mathbf{y}\in\mathcal{Y}}\min_{\mathbf{r}\in\mathcal{R}}\|\mathbf{y}-\mathbf{r}\|_2,\ \max_{\mathbf{r}\in\mathcal{R}}\min_{\mathbf{y}\in\mathcal{Y}}\|\mathbf{r}-\mathbf{y}\|_2\right\}.'),
(17,14,18):('2-12',r'\operatorname{OA}=\frac{1}{M}\sum_{m=1}^{M}\mathbb{I}[\hat y_m=y_m].'),
(17,22,24):('2-13',r'\operatorname{IoU}_{3D}=\frac{\operatorname{Vol}(B_{\mathrm{pred}}\cap B_{\mathrm{gt}})}{\operatorname{Vol}(B_{\mathrm{pred}}\cup B_{\mathrm{gt}})}.'),
(19,6,8):('3-1',r'\mathcal{C}_{s,m}^{(b,l)}=U_m(\mathcal{X}_{s}^{(b,l)}).'),
(19,12,12):('3-2',r'\Delta d_{\mathrm{geom}}<0\quad\Longrightarrow\quad\Delta\mu_{\mathrm{task}}>0\;?'),
(21,9,10):('3-3',r'N^{A}_{\mathrm{target},s}=4N_s.'),
(22,4,5):('3-4',r'M_s=\lfloor N_s/4\rfloor.'),
(22,7,8):('3-5',r'N^{B}_{\mathrm{target},s}=4M_s.'),
(22,16,16):('3-6',r'\mathcal{Y}_{s,m}=S_4(\mathcal{C}_{s,m}),\qquad |\mathcal{Y}_{s,m}|=4n_s.'),
(23,2,4):('3-7',r'\mathcal{Y}^{\mathrm{obs}}_{s,m}=\mathcal{X}_s\uplus S(\mathcal{Y}_{s,m},3n_s),\qquad |\mathcal{Y}^{\mathrm{obs}}_{s,m}|=4n_s.'),
(23,11,12):('3-8',r'\widetilde{\mathcal{Y}}^{(d)}_{s,m}=A_d(\mathcal{Y}_{s,m}).'),
(28,0,10):('4-1',r'\boldsymbol{\mu}=\frac1N\sum_{i=1}^{N}\mathbf{x}_i,\qquad s=\max_i\|\mathbf{x}_i-\boldsymbol{\mu}\|_2,\qquad \mathbf{x}_i^{\mathrm{norm}}=\frac{\mathbf{x}_i-\boldsymbol{\mu}}{s}.'),
(30,4,8):('4-2',r'\boldsymbol{\mu}_j=\frac1{|\mathcal{Q}_j|}\sum_{\mathbf{q}\in\mathcal{Q}_j}\mathbf{q},\qquad s_j=\max_{\mathbf{q}\in\mathcal{Q}_j}\|\mathbf{q}-\boldsymbol{\mu}_j\|_2.'),
(30,9,12):('4-3',r'\hat{\mathbf{q}}=\frac{\mathbf{q}-\boldsymbol{\mu}_j}{s_j},\qquad\mathbf{q}_{\mathrm{back}}=s_j\hat{\mathbf{q}}_{\mathrm{out}}+\boldsymbol{\mu}_j.'),
(33,3,5):('4-4',r'\iota(\mathbf{g})=\iota\!\left(\operatorname*{arg\,min}_{\mathbf{x}_i\in\mathcal{X}_{\mathrm{obs}}}\|\mathbf{g}-\mathbf{x}_i\|_2\right).')}
starts={(p,a):(lab,eq) for (p,a,b),(lab,eq) in EQ.items()}
skip={(p,i) for p,a,b in EQ for i in range(a,b+1)}
chapters={i:[] for i in range(1,5)}; blocks=[]; chapter=0
titles={1:'Introduction',2:'Fundamentals and Related Work',3:'Idea and Concept',4:'Experimental Setup'}
for pi in range(3,39):
    page=doc[pi];bb=page.get_text('blocks')
    for bi,b in enumerate(bb):
        raw=b[4];ss=raw.strip();blocks.append(dict(page=pi+1,block=bi,text=raw))
        if re.fullmatch(r'Chapter [1-4]',ss):chapter=int(ss[-1]);chapters[chapter].append('\\chapter{'+titles[chapter]+'}\n\\label{chap:'+['','introduction','fundamentals','concept','setup'][chapter]+'}');continue
        if ss==titles.get(chapter) or not chapter:continue
        if (pi+1,bi) in starts:
            label,eq=starts[(pi+1,bi)]
            if eq:chapters[chapter].append('\\begin{equation}\n'+eq+'\n\\label{eq:fc-'+label+'}\n\\end{equation}')
            continue
        if (pi+1,bi) in skip:continue
        if re.fullmatch(r'\d+',ss) or b[1]>790:continue
        if 'placement note.' in raw or (92<b[0]<105):continue
        # Notes in this PDF are indented; body paragraphs are at 70.9 pt.
        heading=re.match(r'^([234]\.\d+(?:\.\d+)?)\s*\n(.+)',ss,re.S)
        if heading:
            number,title=heading.groups();level='subsection' if number.count('.')==2 else 'section';chapters[chapter].append('\\'+level+'{'+tex(title)+'}\n\\label{sec:fc-'+number.replace('.','-')+'}');continue
        if ss=='and':continue
        # Merge only cross-page paragraph continuations, not independent paragraphs.
        s=tex(raw)
        if bi==0 and chapters[chapter] and not chapters[chapter][-1].endswith(('.',':','?','}')):chapters[chapter][-1]+=' '+s
        else:chapters[chapter].append(s)

(AUDIT/'formula_clear_blocks.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2),encoding='utf-8')
text={i:'\n\n'.join(v)+'\n' for i,v in chapters.items()}
def replace_section(ch,num,new):
    pattern=r'\\(?:subsection|section)\{[^\n]*\}\n\\label\{sec:fc-'+re.escape(num)+r'\}.*?(?=\n\\(?:subsection|section)\{|\Z)'
    text[ch],count=re.subn(pattern,lambda _:new.strip()+'\n',text[ch],flags=re.S)
    assert count==1,(num,count)

# Retain the introduction's argument and contributions; remove revision history
# and express research questions as prose, as requested by the supervisor.
text[1]=re.sub(r'Earlier drafts described.*?assumed\.', 'The direct strict output contains method predictions after candidate aggregation and exact-cardinality control. A separate observed-first input retains all measured rows and fills the remaining budget from predicted rows. Measurement preservation is therefore an experimental condition, rather than a universal property of upsampling.',text[1],flags=re.S)
text[1]=re.sub(r'The work is organised around.*?observed effect of upsampling\?', 'The study asks whether upsampling improves object classification and scene detection when the original input is available, and whether it recovers performance after controlled sparsification. It also asks whether geometric improvements correspond to task improvements, and how patch construction, normalisation, input budgets, observation preservation, and downstream adaptation influence the outcome.',text[1],flags=re.S)
text[1]=text[1].replace('determine how much of the degradation is attributable to a distribution mismatch','test whether training on the modified representation reduces the degradation')
text[1]+=r'''
The later PU-GCN study used a fixed upsampler and adapted the detectors on
3,712 training frames before evaluating 3,769 validation frames. It tests
the effect of downstream adaptation, not a newly trained PU-GCN network
\cite{shen2026kittifinal}. Figure~\ref{fig:overview} summarises the two branches,
while Figure~\ref{fig:sparsity} illustrates the observed sensing variation.
\thesisfigure{overview}{Experimental overview}{The two-line design and the
different downstream training regimes. Counts and scope follow the executed
HPC and lab protocols\cite{shen2026modelnet,shen2026kittifinal}.}{fig:overview}
\thesisfigure{k_sparsity}{Variation in observed support}{Two equal-size spatial
windows from lab frame 006833. They are scene regions, not segmented vehicles;
the example illustrates sampling variation without estimating a universal
range law\cite{shen2026kitti}.}{fig:sparsity}
'''

replace_section(2,'2-1-2',r'''
\subsection{Object-Level Point Clouds and LiDAR Scenes}\label{sec:fc-2-1-2}
Object-level surface samples and automotive LiDAR scenes arise from different
observation processes. Sampling a CAD surface can provide a single foreground
shape in a centred coordinate system, including surfaces that one external
sensor viewpoint would not observe. A LiDAR frame instead contains multiple
objects, road surfaces, vegetation, buildings, and background clutter. Its
metric samples depend on scanning geometry, occlusion, and reflectance
\cite{wu2015modelnet,geiger2013kitti}. Missing sensor returns therefore differ
from uniformly removing points from a complete object surface. Dataset
construction and the actual splits are specified in Chapter~\ref{chap:setup}.
''')
replace_section(2,'2-3-2',r'''
\subsection{Edge-Aware Resampling (EAR)}\label{sec:fc-2-3-2}
EAR is an optimisation-based method for resampling point sets while retaining
sharp features\cite{huang2013ear}. It uses spatial and normal information to
distinguish smooth surface neighbourhoods from points across an edge.
Edge-aware processing and progressive insertion improve point distribution
without treating every nearby surface as one smooth patch. Unlike a learned
upsampler, it does not require a pretrained generator. Its dependence on
local geometric support and normal estimates is particularly relevant when
the input is sparse, noisy, or composed of intersecting scene surfaces.
''')
replace_section(2,'2-3-5',r'''
\subsection{PU-EdgeFormer}\label{sec:fc-2-3-5}
PU-EdgeFormer combines local graph relationships with attention-based feature
processing for dense coordinate prediction\cite{kim2023puedgeformer}.
Its edge transformer uses relationships between neighbouring points to
represent local structure while attention supports broader feature interaction.
The resulting features feed the point-expansion and reconstruction stages.
This describes the published architecture. The distinction between its native
HPC execution and the lab compatibility path is an implementation issue and
is documented in Chapter~\ref{chap:setup}.
''')
replace_section(2,'2-3-6',r'''
\subsection{PDANS}\label{sec:fc-2-3-6}
PDANS uses conditional diffusion to generate dense points from sparse input
\cite{zhang2025pdans}. Its Adaptive Noise Suppression module assigns weights
using a point and its neighbours, then uses those weights to limit the
influence of noisy support. The TreeTrans module combines features across
encoder levels and attention-based interactions. Sparse-input features
condition the denoising process. Thus it combines point generation with a
specific treatment of noisy observations, rather than simply expanding a
fixed feature vector. Its surface reconstruction objective does not itself
establish a downstream classification or detection gain.
''')
replace_section(2,'2-4-4',r'''
\subsection{Geometric and Task-Level Evidence}\label{sec:fc-2-4-4}
Geometric distances, classification accuracy, and detection AP quantify
different properties. Geometry measures agreement with a reference; task
metrics measure predictions against labels. Evaluating both is necessary
to test an association between them. The next chapter turns this distinction
into the comparison design, and Chapter~\ref{chap:setup} specifies the
implemented metrics and references.
''')
# Move study-specific geometry choices to the existing setup sections.
text[2]=re.sub(r'The executed ModelNet40 evaluation.*?Chapter~\\ref\{chap:setup\}\.', '',text[2],flags=re.S)
text[2]=text[2].replace('In this thesis it is retained as a secondary ModelNet40 geometric audit rather than as the sole ranking criterion.','Its interpretation requires the point cloud and surface to share one coordinate frame.')
text[2]=text[2].replace('A second defining characteristic is irregular sampling.','A common characteristic is irregular sampling.')
text[2]=text[2].replace('p\\ensuremath{i}',r'\mathbf{p}_i')
text[2]=text[2].replace('MAX','max')
text[2]+=r'''
In Equation~\eqref{eq:fc-2-1}, $N$ is the point count and $i$ the point
index; $(x_i,y_i,z_i)$ are its Cartesian coordinates. Attributes are stored
separately from this coordinate vector. In Equation~\eqref{eq:fc-2-2},
$n(\rho)$ is the approximate return count at range $\rho$, and
$\Delta\alpha,\Delta\beta$ are angular sampling increments. The relation
assumes a locally planar visible surface and is an approximation.
For pooling, $\phi_\theta$ is a shared learned feature map with parameters
$\theta$, $\mathbf g$ the pooled feature, and $\mathbf f_i$ a point feature.
In the voxel expression, $\oslash$ denotes component-wise division and
the floor is taken in each coordinate. For OA, $M$ is the number of test
objects, $y_m$ and $\hat y_m$ are true and predicted labels, and
$\mathbb I$ equals one when its condition is true.

Figure~\ref{fig:pointnet-principle} illustrates the set-abstraction sequence,
and Figure~\ref{fig:pugcn-principle} shows the PU-GCN components.
\thesisfigure{pointnet_principle}{PointNet++ principle}{Schematic of local
grouping, shared feature extraction, and hierarchical pooling, redrawn from
the method description\cite{qi2017pointnetpp}.}{fig:pointnet-principle}
\thesisfigure{pugcn_principle}{PU-GCN principle}{Schematic of the published
feature extractor, NodeShuffle expansion, and coordinate reconstruction
\cite{qian2021pugcn}. Experimental checkpoint and patch settings are given
in Chapter~\ref{chap:setup}.}{fig:pugcn-principle}
'''

text[3]=text[3].replace('This directly addresses RQ3','This directly addresses the geometry--task question')
text[3]=text[3].replace('This is necessary because the final experiments compare native point counts','This is the executed training design: the final experiments compare native point counts')
text[3]=text[3].replace('to consume incompatible inputs','to evaluate all representations under one checkpoint')
text[3]=text[3].replace('an upper reference','a pre-sparsification reference').replace('as an upper reference','as a pre-sparsification reference')
text[3]=text[3].replace('They support causal interpretation','They inform mechanism interpretation within their own controls')
text[3]+=r'''
The retained comparison lines are summarised in Table~\ref{tab:lines}.
Their completed outputs and training provenance are documented in the two
experiment records\cite{shen2026modelnet,shen2026kitti}. The final PU-GCN
adaptation matrix extends the frozen comparison with 11 PointRCNN and
9 CenterPoint arms\cite{shen2026kittifinal}. Adapted Line~B arms have a
separately adapted sparse baseline; Line~A has no adapted original baseline.
\begin{table}[t]\centering
\begin{tabular}{llll}\toprule
Branch & Line & Native baseline & Upsampling \\
\midrule
ModelNet40 & A & 1,024 & $1,024\rightarrow4,096$ \\
ModelNet40 & B & 256 & $256\rightarrow1,024$ \\
KITTI & A & $N_s$ & $N_s\rightarrow4N_s$ \\
KITTI & B & $M_s=\lfloor N_s/4\rfloor$ & $M_s\rightarrow4M_s$ \\
\bottomrule\end{tabular}
\caption{Executed comparison lines\cite{shen2026modelnet,shen2026kittifinal}.
ModelNet40 uses one separately trained classifier per representation;
KITTI separates frozen inference and detector adaptation.}\label{tab:lines}
\end{table}
'''

# Correct the inherited conflation of the two compute sources and method paths.
text[4]=re.sub(r'The principal PyTorch-based experimental environment.*?LMS local workspace\.',r'ModelNet40 generation and PointNet++ training were executed as Slurm jobs on the FAU HPC system. The recorded PU-GCN recovery used RTX 2080 Ti nodes after failures on RTX 3080 nodes. KITTI results in this thesis come exclusively from the lab workspace. Both branches required method-specific PyTorch or legacy TensorFlow environments and custom CUDA operators; no single software or GPU configuration is assumed to describe every run. The archived source records identify the actual environments and recovery operations\\cite{shen2026modelnet,shen2026kitti}.',text[4],flags=re.S)
text[4]=text[4].replace('using the corrected integration path.','under the earlier integration path, whose PU-Net normalisation error is explicitly flagged. Corrected subset experiments are reported separately.')
replace_section(4,'4-5-6',r'''
\subsection{PU-EdgeFormer}\label{sec:fc-4-5-6}
The two branches have different execution provenance. On HPC, the final
ModelNet40 record verifies native EdgeTransformer inference, its epoch-100
checkpoint, complete outputs in both lines, and independent PointNet++
training\cite{shen2026modelnet}. On lab, the original environment failed
and the completed KITTI compatibility path reused PU-GCN-compatible
operators and checkpoint inference components\cite{shen2026kitti}.
The lab results are therefore labelled as compatibility-pipeline evidence,
not as an unmodified reproduction of the published PU-EdgeFormer model.
''')
text[4]=text[4].replace('The final unified and adaptation matrices use', 'The final PU-GCN full-validation matrix and the recorded CenterPoint multi-method matrix use')
text[4]=text[4].replace('The thesis must therefore describe the procedure as', 'The procedure is described as').replace('and must not claim that the detectors converged in three epochs.','; convergence in three epochs was not established.')
text[4]=text[4].replace('The main text must therefore not show an EAR value in a ``strict 4\\ensuremath{\\times} full KITTI\'\' table.','EAR is excluded from the strict full-validation KITTI matrix.')
text[4]=text[4].replace('The final thesis must state clearly which quantity is used in a particular table.','Each result table identifies its checkpoint statistic.')
text[4]+=r'''
\section{Additional Parameters of the Completed PU-GCN Study}
\label{sec:pugcn-completed}
The final study retained a fixed PU1K \texttt{model-100} upsampler;
``retraining'' refers to PointRCNN and CenterPoint adaptation
\cite{shen2026kittifinal}. Primary patch centres required 2,048 supporting
points within 2~m for Line~A or 4~m for Line~B. Supplemental coverage
considered uncovered points with at least 32 neighbours inside 6~m,
then used unique nearest neighbours to reach 2,048 rows. The 6~m cover
radius is therefore not a hard bound on every supplemental patch.

Line~B sampled $M_s=\lfloor N_s/4\rfloor$ indices without replacement
using seed $20260702+\mathrm{frame\_id}$ and sorted the indices to retain
relative row order. Strict output sampling used the same seed base after
merging patch predictions. Observed-first construction used a separately
SHA-256-derived seed with base 20260718 and drew $3N_s$ or $3M_s$
prediction rows without replacement. No confidence, curvature, surface
reference, or ground-truth box guided that selection. Distinct row indices
do not guarantee distinct coordinates.

For the final PointRCNN comparison, the dataset reader reset the sampling
seed for every frame to $(20260908+\mathrm{frame\_id})\bmod2^{32}$.
The same input consequently produced the same sampled tensor under frozen
and adapted weights. Actual tensor shapes and SHA-256 hashes were retained.
CenterPoint evaluation used no test-time shuffling, range
$[0,70.4]\times[-40,40]\times[-3,1]$~m, voxel dimensions
$(0.05,0.05,0.1)$~m, at most five points per voxel, and 40,000 test voxels.

Figure~\ref{fig:patch-pipeline} locates patch normalisation and reconstruction;
Figure~\ref{fig:input-budget} distinguishes the two detector adapters;
Figure~\ref{fig:adaptation-design} identifies which weights were updated.
\thesisfigure{patch_pipeline}{Local patch processing}{The final lab PU-GCN
patch and strict-output pipeline\cite{shen2026kittifinal}. Centres and scales
are stored per patch; metric coordinates are restored before merging.}{fig:patch-pipeline}
\thesisfigure{input_budget}{Effective detector input}{Alternative PointRCNN
and CenterPoint input constraints after scene-level filtering
\cite{shen2026kittifinal}. File cardinality does not equal model capacity.}{fig:input-budget}
\thesisfigure{adaptation_design}{Detector adaptation}{The fixed PU-GCN
upsampler supplies inputs for representation-specific detector training.
The completed study uses 3,712 training and 3,769 validation frames
\cite{shen2026kittifinal}.}{fig:adaptation-design}

\section{Metric and Classifier Reporting Details}\label{sec:classifier-setup}
The HPC classifier was PointNet++ SSG with XYZ-only input, batch size 24,
four loader workers, Adam, learning rate 0.001, weight decay 0.0001,
StepLR period 20 and factor 0.7, 200 epochs, and seed 42
\cite{shen2026modelnet}. Its first abstraction layer used 512 centroids,
radius 0.2 and 32 neighbours; the second used 128 centroids, radius 0.4
and 64 neighbours; the last grouped globally. Each branch retained its
native point count with loader resampling disabled. Best OA selected the
checkpoint on the test set, with ties retaining the latest epoch. Final OA
uses epoch 200; neither statistic is a multi-seed estimate.

For NUC, the HPC code used 128 query centres and radii 0.02, 0.05, and
0.10, averaging the coefficient of variation of neighbourhood counts.
Query centres were not perfectly shared by object across methods. P2F
independently normalised cloud and mesh and is therefore a secondary
normalisation-conditioned audit. Four reused mesh-reference training
shapes limit the classifier controls but do not affect test geometry.

KITTI uses class-specific overlap and difficulty rules\cite{kittieval2026}.
Car requires 3D IoU 0.70; Pedestrian and Cyclist require 0.50. Easy,
Moderate, and Hard require image-box heights of at least 40, 25, and
25 pixels, maximum truncations of 0.15, 0.30, and 0.50, and maximum
occlusion levels of 0, 1, and 2, respectively. Let TP, FP, and FN be
true positives, false positives, and false negatives under the evaluator's
matching and ignore rules. Precision is $P=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$
and recall is $R=\mathrm{TP}/(\mathrm{TP}+\mathrm{FN})$. The interpolated
precision envelope is $P_{\mathrm{int}}(r)=\max_{\tilde r\ge r}P(\tilde r)$.
The final implementation averages the 40 positive recall-grid entries:
\begin{equation}
\operatorname{AP}_{R40}=\frac1{40}\sum_{j=1}^{40}P_{\mathrm{int}}(j/40).
\end{equation}
Here $j$ indexes recall-grid positions and $\tilde r$ an achieved recall.
Percent AP multiplies this fraction by 100. Historical AP$_{R11}$ values
use another sampling rule and are not interchanged with AP$_{R40}$.
'''

names={1:'01_introduction',2:'02_fundamentals',3:'03_concept',4:'04_experimental_setup'}
for ch,s in text.items():
    # Remove residual writing instructions, preserve source paragraph anchors.
    s=s.replace('the thesis should not claim that a complete calibration analysis, McNemar test, or probability-level significance test was performed when those outputs are unavailable.','a complete calibration analysis, McNemar test, and probability-level significance test were not performed.')
    (ROOT/f'texfiles/{names[ch]}.tex').write_text(s,encoding='utf-8')
print('Restored author-confirmed prose; applied supervisor and provenance corrections to Chapters 1--4.')
