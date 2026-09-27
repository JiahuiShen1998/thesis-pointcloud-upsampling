"""Validate active source, final figures and archived-case provenance."""
from pathlib import Path
import sys,os,re,json,csv,subprocess,hashlib,collections
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'.python-deps'))
import pymupdf
Q=R/'audit/manuscript_revision_20260913/qa';Q.mkdir(parents=True,exist_ok=True)
skill=Path.home()/'.codex/skills/nature-figure/scripts'
env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONPATH']=str(R/'.python-deps')+os.pathsep+env.get('PYTHONPATH','')
seen=set();missing=[]
def walk(p):
 if p in seen:return
 if not p.exists():missing.append(str(p));return
 seen.add(p);t=p.read_text(encoding='utf-8')
 for v in re.findall(r'\\(?:input|include|tableinput)\{([^}]+)\}',t):
  q=R/v;walk(q if q.suffix else q.with_suffix('.tex'))
walk(R/'thesis.tex')
alltext='\n'.join(p.read_text(encoding='utf-8') for p in seen)
labels=re.findall(r'\\label\{([^}]+)\}',alltext)+re.findall(r'\}\s*\{(fig:[^}]+)\}',alltext)
refs=set(re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}',alltext))
cites={k.strip() for m in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}',alltext) for k in m.split(',')}
keys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',(R/'bibfiles/references.bib').read_text(encoding='utf-8')))
figs=re.findall(r'\\thesisfigure\{([^}]+)\}',alltext)
source={'active_files':len(seen),'figures':len(figs),'missing_files':missing,'missing_labels':sorted(refs-set(labels)),'duplicate_labels':[k for k,v in collections.Counter(labels).items() if v>1],'missing_cites':sorted(cites-keys),'unreferenced_figures':[l for l in labels if l.startswith('fig:') and l not in refs],'appendix_present':bool(re.search(r'\\appendix\b|Appendix~',alltext)),'placeholders':[]}
for p in seen:
 for n,l in enumerate(p.read_text(encoding='utf-8').splitlines(),1):
  if re.search(r'\bTODO\b|\bTBD\b|\bFIXME\b',l):source['placeholders'].append([p.name,n,l])
(Q/'source_check.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
print('SOURCE',source,flush=True)
p=subprocess.run([sys.executable,str(skill/'validate_figure.py'),str(R/'scripts/make_record_figures.py')],env=env,capture_output=True,text=True,encoding='utf-8')
(Q/'figure_source_check.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
print('STATIC FIGURE SOURCE',p.returncode,flush=True)
summary=[]
for name in sorted(figs):
 f=R/'figure'/f'{name}.pdf'
 a=subprocess.run([sys.executable,str(skill/'audit_pdf_text.py'),str(f),'--min-pt','11.95','--json'],env=env,capture_output=True,text=True,encoding='utf-8')
 c=subprocess.run([sys.executable,str(skill/'audit_figure_collisions.py'),str(f),'--json-out',str(Q/(name+'.collisions.json'))],env=env,capture_output=True,text=True,encoding='utf-8')
 (Q/(name+'.text.json')).write_text(a.stdout,encoding='utf-8')
 doc=pymupdf.open(f);fonts=[];outside=[]
 for page in doc:
  for b in page.get_text('dict').get('blocks',[]):
   for line in b.get('lines',[]):
    for s in line.get('spans',[]):
     if s.get('text','').strip():
      fonts.append(s['size'])
      if not page.rect.contains(pymupdf.Rect(s['bbox'])):outside.append(s['text'])
 summary.append(dict(name=name,text_exit=a.returncode,collision_exit=c.returncode,minimum_font=min(fonts) if fonts else None,outside_text=outside,collision_output=c.stdout[-1400:],stderr=c.stderr[-500:]))
 print('FIGURE',name,a.returncode,c.returncode,'outside',outside,flush=True)
(Q/'figure_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
# Preserve a complete hash manifest for all copied experiment evidence.
manifest=[]
for p in sorted((R/'evidence').rglob('*')):
 if p.is_file() and p.name!='manifest.json':
  origin='HPC' if 'modelnet40_hpc' in p.parts else 'lab' if 'kitti_lab' in p.parts else 'derived'
  manifest.append({'file':p.relative_to(R).as_posix(),'origin':origin,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(R/'evidence/manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('MANIFEST',len(manifest),flush=True)
# Create readable contact sheets from the final PDFs.
from PIL import Image,ImageDraw
for start in range(0,len(figs),8):
 canvas=Image.new('RGB',(1800,1400),'#e9ecef');draw=ImageDraw.Draw(canvas)
 for j,name in enumerate(sorted(figs)[start:start+8]):
  doc=pymupdf.open(R/'figure'/f'{name}.pdf');pix=doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False)
  im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((430,650))
  xx=(j%4)*450+(450-im.width)//2;yy=(j//4)*700+30
  canvas.paste(im,(xx,yy));draw.text(((j%4)*450+12,(j//4)*700+10),name,fill='black')
 canvas.save(Q/f'figure_contact_{start//8+1}.jpg',quality=90)
