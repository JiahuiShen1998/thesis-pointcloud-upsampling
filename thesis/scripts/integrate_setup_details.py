"""Place recovered additions beside their existing experimental subsections."""
from pathlib import Path
import re
T=Path(__file__).resolve().parents[1]/'texfiles'
p=T/'04_experimental_setup.tex';s=p.read_text(encoding='utf-8')
a=s.index('\\section{Additional Parameters');b=s.index('\\section{Metric and Classifier',a);c=s.index('\\section{Chapter Summary}',b)
pu=s[a:b];metrics=s[b:c];s=s[:a]+s[c:]
patch=pu[pu.index('The final study'):pu.index('Line~B sampled')].strip()
seeds=pu[pu.index('Line~B sampled'):pu.index('For the final PointRCNN')].strip()
adapters=pu[pu.index('For the final PointRCNN'):].strip()
classifier=metrics[metrics.index('The HPC classifier'):metrics.index('For NUC')].strip()
geometry=metrics[metrics.index('For NUC'):metrics.index('KITTI uses')].strip()
detection=metrics[metrics.index('KITTI uses'):].strip()
for anchor,body in [('Figure~\\ref{fig:patch-pipeline}',patch),('\\section{KITTI Format Conversion',seeds),('\\section{Classifier Integration}',adapters),('\\section{Geometric and Task Evaluation Protocol}',r'\label{sec:classifier-setup}'+'\n'+classifier),('\\subsection{ModelNet40 classification evaluation}',geometry),('\\subsection{Geometry-task relationship}',detection)]:
    assert anchor in s,anchor
    i=s.index(anchor);s=s[:i]+body+'\n\n'+s[i:]
p.write_text(s,encoding='utf-8')
p=T/'03_concept.tex';s=p.read_text(encoding='utf-8');a=s.index('The retained comparison lines');block=s[a:];s=s[:a];i=s.index('\\subsection{Line A');s=s[:i]+block+'\n'+s[i:];p.write_text(s,encoding='utf-8')
p=T/'01_introduction.tex';s=p.read_text(encoding='utf-8')
for name,anchor,lead in [('k_sparsity','A natural response',r'Figure~\ref{fig:sparsity} illustrates this variation using two spatial windows from a recorded lab scan.'),('overview','The study asks',r'Figure~\ref{fig:overview} summarises the two data branches and comparison lines.')]:
    m=re.search(r'\\thesisfigure\{'+name+r'\}.*?\}\{fig:[^}]+\}\n',s,re.S);assert m,name
    fig=m.group(0);s=s[:m.start()]+s[m.end():];i=s.index(anchor);s=s[:i]+lead+'\n'+fig+'\n'+s[i:]
s=s.replace(r'Figure~\ref{fig:overview} summarises the two branches,'+'\n'+r'while Figure~\ref{fig:sparsity} illustrates the observed sensing variation.','')
p.write_text(s,encoding='utf-8')
print('Integrated setup additions and placed introductory figures beside their discussion.')
