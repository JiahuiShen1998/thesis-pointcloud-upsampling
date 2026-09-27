"""Check actual page placements of thesis figure PDF forms after a build."""
from pathlib import Path
import sys,re,json
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'.python-deps'))
import pymupdf
d=pymupdf.open(R/'thesis.pdf')
rows=[]
number=r'[-+]?(?:\d*\.\d+|\d+\.?\d*)'
op=re.compile(r'(?m)^\s*(?:(q|Q)|('+number+r'(?:\s+'+number+r'){5})\s+cm|/(\w+)\s+Do)\s*$')
for p in d:
    if not re.search(r'Figure \d+\.\d+:',p.get_text()):continue
    forms={x[1]:pymupdf.Rect(x[3]) for x in p.get_xobjects() if x[2]==0}
    matrix=pymupdf.Matrix(1,1);stack=[]
    for m in op.finditer(p.read_contents().decode('latin-1')):
        if m.group(1)=='q':stack.append(pymupdf.Matrix(matrix))
        elif m.group(1)=='Q':matrix=stack.pop()
        elif m.group(2):matrix=pymupdf.Matrix(*map(float,m.group(2).split()))*matrix
        elif m.group(3) in forms:
            box=forms[m.group(3)]*matrix*p.transformation_matrix
            if box.width<350 or box.height<50:continue
            before=[b[4].strip() for b in p.get_text('blocks') if b[4].strip() and b[1]>80 and b[3]<box.y0-1]
            font_sizes=sorted({round(s['size'],3) for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if box.contains(pymupdf.Rect(s['bbox'])) and s['text'].strip()})
            rows.append(dict(page=p.number+1,box=list(box),text_before=before,font_sizes=font_sizes))
out=dict(figure_count=len(rows),figures=rows,not_at_page_top=[r for r in rows if r['text_before']],
         unexpected_font_sizes=[r for r in rows if any(abs(s-11.9552)>0.04 for s in r['font_sizes'])])
(R/'audit/chapters_1_2_restoration_20260914/layout_check.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='figures'}),flush=True)
assert len(rows)==51 and not out['not_at_page_top'] and not out['unexpected_font_sizes']
