"""Final targeted edits following the first compilation and evidence check."""
from pathlib import Path
import re,shutil
R=Path(__file__).resolve().parents[1];T=R/'texfiles'
p=T/'04_experimental_setup.tex';s=p.read_text(encoding='utf-8')
s=s.replace('pointnet2_cls_ssg',r'\texttt{pointnet2\_cls\_ssg}').replace('sampling/grouping/interpolation','sampling, grouping, and interpolation').replace('filelevel','file-level')
s=s.replace('using Adam, an initial',r'using Adam\cite{kingma2015adam}, an initial')
s=s.replace('This procedure should not be described as a realistic low-beam LiDAR simulation.', 'This procedure is a controlled decimation rather than a physical low-beam LiDAR simulation.')
s=s.replace('the ModelNet40 result should not be presented as a multi-seed mean or accompanied by a fabricated standard deviation.', 'the reported values are individual-run results without a cross-seed standard deviation.')
s=s.replace(r'\ensuremath{M_s} = \ensuremath{\lfloor} \ensuremath{N_s} /4\ensuremath{\rfloor}',r'$M_s=\lfloor N_s/4\rfloor$')
s=s.replace(r'\ensuremath{N_s} \ensuremath{\times}  4',r'$N_s\times4$')
s=s.replace(r'(\ensuremath{x} , \ensuremath{y} , \ensuremath{z} , \ensuremath{r} )',r'$(x,y,z,\iota)$').replace(r'\ensuremath{r} is reflectance',r'$\iota$ is reflectance')
s=s.replace(r'\ensuremath{\mathcal{Q}_j} = {\ensuremath{\mathbf{q}_i} }',r'$\mathcal Q_j=\{\mathbf q_i\}$').replace(r'\ensuremath{Q} \ensuremath{\times}  4',r'$Q\times4$, where $Q$ is the output row count')
# Anchor implemented details to original records, not only method papers.
s=s.replace('This separation was essential during development.',r'The completed implementation and audits are documented in the two work records and final detector matrix\cite{shen2026modelnet,shen2026kitti,shen2026kittifinal}. This separation was essential during development.')
s=s.replace('The data loader is required to expose the actual point count of the condition rather than silently truncating or padding all branches to one common count.', 'The final data loader exposed each condition at its native point count.')
# Place the three distinct setup figures beside their subjects.
figs={}
for name in ['patch_pipeline','input_budget','adaptation_design']:
    pat=r'\\thesisfigure\{'+name+r'\}.*?\}\{fig:[^}]+\}\n'
    m=re.search(pat,s,re.S);assert m,name;figs[name]=m.group(0);s=s[:m.start()]+s[m.end():]
s=s.replace('Figure~\\ref{fig:patch-pipeline} locates patch normalisation and reconstruction;\nFigure~\\ref{fig:input-budget} distinguishes the two detector adapters;\nFigure~\\ref{fig:adaptation-design} identifies which weights were updated.\n','')
for anchor,name,lead in [('\\subsection{EAR}','patch_pipeline',r'Figure~\ref{fig:patch-pipeline} locates patch normalisation and reconstruction.'),('\\section{ModelNet40 Classification Integration}','input_budget',r'Figure~\ref{fig:input-budget} compares the two detector input adapters.'),('\\section{Executed Diagnostic','adaptation_design',r'Figure~\ref{fig:adaptation-design} identifies the detector weights updated during adaptation.')]:
    if anchor not in s and name=='input_budget':anchor='\\subsection{PointNet++ Classification Integration}'
    assert anchor in s,anchor
    at=s.index(anchor);s=s[:at]+lead+'\n'+figs[name]+'\n'+s[at:]
p.write_text(s,encoding='utf-8')
p=T/'03_concept.tex';s=p.read_text(encoding='utf-8').replace(r'4\ensuremath{\lfloor} \ensuremath{N_s} /4\ensuremath{\rfloor}',r'$4\lfloor N_s/4\rfloor$').replace(r'\ensuremath{S} (\ensuremath{\cdot} , 3\ensuremath{n_s} )',r'$S(\cdot,3n_s)$');p.write_text(s,encoding='utf-8')
# Introduction has continuous prose; move examples before the roadmap.
p=T/'01_introduction.tex';s=p.read_text(encoding='utf-8');at=s.index('The later PU-GCN study');tail=s[at:];s=s[:at];at=s.index('The remainder of the thesis');s=s[:at]+tail+'\n'+s[at:];p.write_text(s,encoding='utf-8')
src=R.parent/'Literature/kingma2015_Adam.pdf';shutil.copy2(src,R/'bibfiles/papers/kingma2015adam.pdf')
print('Applied layout and source-anchor fixes.')
