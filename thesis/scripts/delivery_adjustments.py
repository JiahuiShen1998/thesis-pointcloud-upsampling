"""Corrections from reviewing the first complete manuscript PDF."""
from pathlib import Path
import re,json,hashlib
R=Path(__file__).resolve().parents[1];T=R/'texfiles'
p=T/'appendix_a_quantitative_results.tex';s=p.read_text(encoding='utf-8')
s=s[:s.index('\\section{CenterPoint Image-Box AP}')]
s=s.replace('The CSV in the evidence directory retains full precision.', 'The CSV in the evidence directory retains full precision. The final CenterPoint export contains 3D and BEV AP; image-box AP is not reported for that detector.')
p.write_text(s,encoding='utf-8')
# Prevent figures from crossing the two dataset branches or the joint discussion.
for name in ['05_kitti_results.tex','05_joint_discussion.tex']:
    p=T/name;s=p.read_text(encoding='utf-8')
    if not s.startswith('\\FloatBarrier'):s='\\FloatBarrier\n'+s
    s=s.replace('Representation-specific detector training recovered', 'Training detectors for each input representation recovered')
    if name=='05_kitti_results.tex':
        s=s.replace('\\section{KITTI Integration and Mechanism Diagnostics}', '\\FloatBarrier\n\\section{KITTI Integration and Mechanism Diagnostics}')
    p.write_text(s,encoding='utf-8')
for p in T.glob('*.tex'):
    if p.name in ['cv.tex','appendix_d_supplementary_methods.tex','appendix_e_environment.tex','appendix_f_commands.tex']:continue
    s=p.read_text(encoding='utf-8');s=re.sub(r'(?<![\s~])\\cite',r' \\cite',s)
    p.write_text(s,encoding='utf-8')
p=T/'01_introduction.tex';s=p.read_text(encoding='utf-8').replace('different downstream training regimes. Counts and scope follow the executed',r'different downstream training regimes; $M=\lfloor N/4\rfloor$. Counts and scope follow the executed');p.write_text(s,encoding='utf-8')
manifest=[]
for p in sorted((R/'bibfiles/papers').glob('*.pdf')):
    manifest.append(dict(key=p.stem,copied=p.relative_to(R).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(R/'bibfiles/reference_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Removed the unreported empty CenterPoint metric; constrained floats; corrected citation spacing.')
