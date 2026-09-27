"""Build, verify and publish the detector-convergence thesis revision."""
from pathlib import Path
import sys,os,subprocess,json,hashlib,re,shutil,textwrap
R=Path(__file__).resolve().parents[1]
A=R/'audit/convergence_revision_20260917'
sys.path.insert(0,str(R/'.python-deps'))
import pymupdf
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if '--publish-only' not in sys.argv:
 env=os.environ.copy();env['PATH']=r'E:\texlive\2025\bin\windows'+';'+env.get('PATH','')
 with (A/'compile.txt').open('w',encoding='utf-8') as log:
  run=subprocess.run([r'E:\texlive\2025\bin\windows\latexmk.EXE','-pdf','-outdir=build/current','-interaction=nonstopmode','-halt-on-error','thesis.tex'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
 assert run.returncode==0
 for fn in ['check_teacher_revision_citations.py','check_teacher_revision_layout.py']:
  source=(R/'scripts'/fn).read_text(encoding='utf-8').replace('audit/teacher_email_revision_20260917','audit/convergence_revision_20260917')
  target=R/'scripts'/fn
  wrapper="import sys;sys.argv=["+repr(str(target))+",'--built'];exec(compile("+repr(source)+","+repr(str(target))+",'exec'),{'__file__':"+repr(str(target))+",'__name__':'__main__'})"
  run=subprocess.run([sys.executable,'-X','utf8','-c',wrapper],capture_output=True,text=True,encoding='utf-8')
  (A/(fn+'.console.txt')).write_text(run.stdout+run.stderr,encoding='utf-8')
  assert run.returncode==0,(fn,run.stdout,run.stderr)
audit=json.loads((A/'audit.json').read_text(encoding='utf-8'))
before=json.loads((A/'before_hashes.json').read_text(encoding='utf-8'))
changed=[name for name,h in before.items() if digest(R/name)!=h]
assert not any(name.startswith('figure/') or name in ['texfiles/01_introduction.tex','texfiles/02_fundamentals.tex','texfiles/cv.tex','bibfiles/references.bib','thesis.tex'] or 'modelnet' in name for name in changed),changed
d=pymupdf.open(R/'build/current/thesis.pdf')
toc=d.get_toc();starts={t:p-1 for l,t,p in toc};names=d.resolve_names()
old=pymupdf.open(A/'before/thesis.pdf');oldstarts={t:p-1 for l,t,p in old.get_toc()}
protected={}
for begin,end in [('Introduction','Fundamentals and Related Work'),('Fundamentals and Related Work','Idea and Concept')]:
 oldtext='\n'.join(old[i].get_text() for i in range(oldstarts[begin],oldstarts[end]))
 newtext='\n'.join(d[i].get_text() for i in range(starts[begin],starts[end]))
 protected[begin]=oldtext==newtext
 assert protected[begin],begin
assert len(d[starts['Kurzfassung']+1].get_text().strip())<10,'German abstract spills onto another page'
old.close()
# Extend the portable active manifest with new verified assets.
manifest=json.loads((R/'figure/active_manifest.json').read_text(encoding='utf-8'))
for filename,name in [('figure_manifest.json','k_convergence_gains'),('terminal_figure_source.json','k_convergence_terminal')]:
 row=json.loads((A/'qa'/filename).read_text(encoding='utf-8'));row['name']=name
 row['plotting_script']='scripts/make_convergence_figures.py'
 manifest=[x for x in manifest if x['name']!=name]+[row]
(R/'figure/active_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
run=subprocess.run([sys.executable,str(R/'scripts/update_active_figure_manifest.py')],capture_output=True,text=True,encoding='utf-8')
assert run.returncode==0,run.stderr
print(run.stdout,flush=True)
published=False;publication_error=''
try:
 shutil.copy2(R/'build/current/thesis.pdf',R/'thesis.pdf');published=True
except PermissionError as ex:
 publication_error=str(ex)
# Reuse the established portable excerpt/link-preservation function.
source=(R/'audit/teacher_email_revision_20260917/finalize_delivery.py').read_text(encoding='utf-8')
helper=textwrap.dedent(source[source.index(' def position('):source.index("\n bib=list(range(")])
reviews=[]
scope=dict(R=R,A=A,d=d,names=names,toc=toc,pymupdf=pymupdf,shutil=shutil,reviews=reviews)
if not published:helper=helper.replace("file='thesis.pdf'","file='build/current/thesis.pdf'")
helper=helper.replace("out.set_toc([[l,t,mapping[p-1]+1]", "out.set_toc([[(1 if filename=='review_convergence_update.pdf' else l),t,mapping[p-1]+1]")
exec(helper,scope)
extract=scope['extract']
bib=list(range(starts['Bibliography'],starts['Curriculum Vitae']))
if published:
 extract('review_chapter_1.pdf',list(range(starts['Introduction'],starts['Fundamentals and Related Work']))+bib,'Chapter 1 unchanged; current bibliography')
 extract('review_chapter_2.pdf',list(range(starts['Fundamentals and Related Work'],starts['Idea and Concept']))+bib,'Chapter 2 unchanged; current bibliography')
 extract('review_chapters_1_2.pdf',list(range(starts['Introduction'],starts['Idea and Concept']))+bib+[len(d)-1],'Chapters 1 and 2 unchanged, bibliography and academic CV')
 extract('review_chapters_3_4.pdf',list(range(starts['Idea and Concept'],starts['Results and Evaluation']))+bib,'Chapters 3 and 4 including extended detector protocol')
 extract('review_chapters_5_6.pdf',list(range(starts['Results and Evaluation'],starts['List of Figures']))+bib,'Chapters 5 and 6 including extended detector results')
 extract('CV_review.pdf',[starts['Curriculum Vitae']],'Academic Curriculum Vitae')

ids=[starts['Kurzfassung'],starts['Abstract']]
ids+=list(range(starts['Detector Retraining with PU-GCN Inputs'],starts['Executed Diagnostic and Mechanism Experiments']+1))
ids+=list(range(starts['PU-GCN Inputs and Detector Retraining'],starts['KITTI Integration and Mechanism Diagnostics']+1))
ids+=list(range(starts['Joint Discussion'],starts['List of Figures']))
ids=sorted(set(ids))
extract('review_convergence_update.pdf',ids,'Extended detector adaptation: changed thesis sections')
new_files=['texfiles/04_kitti_extended_adaptation.tex','texfiles/05_kitti_extended_adaptation.tex','tables/kitti_extended_comparison.tex','tables/kitti_extended_stages.tex','scripts/make_convergence_figures.py']
status={'pages':len(d),'canonical_published':published,'publication_error':publication_error,'sha256':digest(R/'build/current/thesis.pdf'),
 'only_expected_manuscript_changes':changed,'new_files':new_files,'protected_chapter_text_equal':protected,
 'existing_figure_pdfs_unchanged':sum(name.startswith('figure/') for name in before),
 'scientific_figures':audit['figures'],'references':audit['cited_sources'],'citation_link_errors':audit['pdf_link_errors'],
 'new_figure_font_pdf_pt':12*72/72.27,'reviews':reviews,
 'raw_lab_epoch_files_retrieved':False,'new_results_source':'Author-provided lab records, with original baselines and three-epoch values cross-checked against archived lab CSV',
 'complete_epoch_curves_included':False,'terminal_curve':'Only six explicitly reported final epochs; preceding epochs not reconstructed',
 'chapter_start_pdf_pages':{t:p for l,t,p in toc if l==1},
 'section_pdf_pages':{k:starts[k]+1 for k in ['Detector Retraining with PU-GCN Inputs','PU-GCN Inputs and Detector Retraining','Joint Discussion']}}
(A/'delivery_verification.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
audit['canonical_matches_build']=published
(A/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(status,ensure_ascii=False),flush=True)
