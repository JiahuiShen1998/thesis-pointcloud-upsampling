"""Audit the compiled manuscript and render every page for visual review."""
from pathlib import Path
import sys,json,re,collections
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'.python-deps'))
import pymupdf
from PIL import Image,ImageDraw
Q=R/'audit/manuscript_revision_20260913/qa'
O=Q/'pages';O.mkdir(exist_ok=True)
doc=pymupdf.open(R/'build/current/thesis.pdf')
log=(R/'build/current/thesis.log').read_text(encoding='utf-8',errors='replace')
pages=[];outside=[];fontbad=[]
for i,page in enumerate(doc):
 text=page.get_text();spans=[]
 for b in page.get_text('dict')['blocks']:
  for line in b.get('lines',[]):
   for s in line.get('spans',[]):
    if not s['text'].strip():continue
    if not page.rect.contains(pymupdf.Rect(s['bbox'])):outside.append({'page':i+1,'text':s['text'],'bbox':s['bbox']})
    if 'DejaVu' in s['font'] and s['size']<11.94:fontbad.append({'page':i+1,'text':s['text'],'size':s['size']})
    if 70<s['bbox'][1]<756:spans.append(s)
 yend=max((s['bbox'][3] for s in spans),default=0)
 pages.append({'physical_page':i+1,'label':page.get_label(),'chars':len(text),'last_content_y':round(yend,2),'figure_captions':re.findall(r'Figure\s+\d+\.\d+:[^\n]*',text),'first_lines':text.splitlines()[:5]})
 (O/f'page_{i+1:03d}.txt').write_text(text,encoding='utf-8')
for start in range(0,len(doc),12):
 canvas=Image.new('RGB',(1480,1608),'#dde1e5');draw=ImageDraw.Draw(canvas)
 for j in range(min(12,len(doc)-start)):
  i=start+j;page=doc[i];pix=page.get_pixmap(matrix=pymupdf.Matrix(.65,.65),alpha=False)
  im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples);im.thumbnail((354,501))
  x=(j%4)*370+(370-im.width)//2;y=(j//4)*536+26
  canvas.paste(im,(x,y));draw.text(((j%4)*370+10,(j//4)*536+8),f'PDF {i+1} / {page.get_label()}',fill='black')
 canvas.save(O/f'contact_{start//12+1:02d}.jpg',quality=90)
selected_labels=['45','51','52','56','61','67','68','70','72','75','78','84','86','92','94','95','98','101','104','106','108','110','111','112','115']
for page in doc:
 if page.get_label() in selected_labels:
  page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False).save(O/f'label_{page.get_label()}.png')
summary={'pdf_pages':len(doc),'pdf_bytes':Path(doc.name).stat().st_size,'overfull_count':log.count('Overfull'),'undefined_references':bool(re.search(r'undefined references|Citation .*undefined|Reference .*undefined',log)),'multiply_defined':bool(re.search(r'multiply defined',log)),'float_too_large':log.count('Float too large'),'outside_page_text':outside,'figure_fonts_below_body_size':fontbad,'appendix_heading':bool(re.search(r'\nAppendix\b','\n'.join(p.get_text() for p in doc))),'bibliography_entries':len(re.findall(r'\\bibitem', (R/'build/current/thesis.bbl').read_text(encoding='utf-8'))),'pages':pages}
(Q/'compiled_pdf_check.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='pages'},indent=2),flush=True)
