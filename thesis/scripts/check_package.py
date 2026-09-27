"""Evidence-derived secondary table and deterministic package/figure checks."""
from pathlib import Path
import csv,json,re,sys,subprocess,os
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'.python-deps'))
import fitz
Q=R/'audit/revision_20260913/qa';Q.mkdir(exist_ok=True)
with (R/'evidence/modelnet40_hpc/data/geometry_dense_gt_test_summary.csv').open(encoding='utf-8',newline='') as f:rr=list(csv.DictReader(f))
lines=[r'\begin{table}[!t]\centering',r'\begin{tabular}{llrr}\toprule Line & Input & Mean P2F & Median P2F\\\midrule']
for r in rr:
    if r['metric']!='p2f_mean':continue
    line={'baseline':'Reference','lineB_baseline':'B','lineA':'A','lineB':'B'}[r['line']]
    method=r['method'].replace('Downsampled x4','Sparse')
    lines.append(f"{line} & {method} & {float(r['mean']):.5f} & {float(r['median']):.5f}"+r' \\')
lines += [r'\bottomrule\end{tabular}',r'\caption{Secondary P2F audit, 2,468 test objects per row. Clouds and meshes were independently normalised. The median is over the per-object mean distances. Values retain the recorded coordinate convention.}\label{tab:p2f-audit}\end{table}']
(R/'tables/p2f_audit.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# Detect unsupported glyphs and unresolved source-level references before TeX.
active=[p for p in (R/'texfiles').glob('*.tex') if p.name not in ['cv.tex','appendix_d_supplementary_methods.tex','appendix_e_environment.tex','appendix_f_commands.tex']]
alltext='\n'.join(p.read_text(encoding='utf-8') for p in active)
bib=(R/'bibfiles/references.bib').read_text(encoding='utf-8')
keys={s for s in re.findall(r'@\w+\{([^,]+),',bib)}
cites={k.strip() for g in re.findall(r'\\cite\{([^}]+)\}',alltext) for k in g.split(',')}
labels=re.findall(r'\\label\{([^}]+)\}',alltext+'\n'+'\n'.join(p.read_text(encoding='utf-8') for p in (R/'tables').glob('*.tex')))
labels+=re.findall(r'\}\{(fig:[^}]+)\}',alltext)
refs=set(re.findall(r'\\(?:eq)?ref\{([^}]+)\}',alltext))
report=dict(missing_citations=sorted(cites-keys),missing_labels=sorted(refs-set(labels)),duplicate_labels=sorted({l for l in labels if labels.count(l)>1}),unsupported_unicode=[])
for p in active:
    for n,line in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
        weird=sorted({c for c in line if ord(c)>255 and c not in '–—’‘“”−×→'})
        if weird:report['unsupported_unicode'].append(dict(file=p.name,line=n,characters=''.join(weird)))
(Q/'source_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Source checks:',report)

# Extract local primary-paper title pages for human bibliography review.
pages={}
for p in (R/'bibfiles/papers').glob('*.pdf'):
    d=fitz.open(p);pages[p.stem]=dict(pages=len(d),first_page=d[0].get_text(),metadata=d.metadata)
(Q/'reference_title_pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')

skill=Path.home()/'.codex/skills/nature-figure/scripts'
env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONPATH']=str(R/'.python-deps')+os.pathsep+env.get('PYTHONPATH','')
summary=[]
for p in sorted((R/'figures/final').glob('*.pdf')):
    q=Q/(p.stem+'.collisions.json')
    c=subprocess.run([sys.executable,str(skill/'audit_figure_collisions.py'),str(p),'--json-out',str(q)],env=env,capture_output=True,text=True,encoding='utf-8')
    ft=subprocess.run([sys.executable,str(skill/'audit_pdf_text.py'),str(p),'--min-pt','11.95','--json'],env=env,capture_output=True,text=True,encoding='utf-8')
    (Q/(p.stem+'.text.json')).write_text(ft.stdout,encoding='utf-8')
    summary.append(dict(name=p.stem,collision_exit=c.returncode,text_exit=ft.returncode,collision_output=c.stdout[-900:],error=c.stderr[-500:]))
(Q/'figure_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Figure checks:',[(r['name'],r['collision_exit'],r['text_exit']) for r in summary])
