"""Create an accessible PDF copy of the thesis change list."""
from pathlib import Path
import sys,re,html
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'.python-deps'))
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
A=R/'audit/convergence_revision_20260917'
for path in [Path(r'C:\Windows\Fonts\msyh.ttc'),Path(r'C:\Windows\Fonts\simsun.ttc')]:
 try:
  pdfmetrics.registerFont(TTFont('Chinese',str(path),subfontIndex=0));break
 except Exception:continue
else:raise RuntimeError('No usable CJK TrueType font')
styles={
 'title':ParagraphStyle('title',fontName='Chinese',fontSize=18,leading=25,spaceAfter=14,wordWrap='CJK'),
 'h':ParagraphStyle('h',fontName='Chinese',fontSize=12.5,leading=18,spaceBefore=12,spaceAfter=7,wordWrap='CJK',keepWithNext=True),
 'body':ParagraphStyle('body',fontName='Chinese',fontSize=10.5,leading=16,spaceAfter=7,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='Chinese',fontSize=9.5,leading=14,wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName='Chinese',fontSize=10,leading=15,textColor=colors.HexColor('#213a50'),wordWrap='CJK'),
}
def plain(s):
 s=re.sub(r'\[([^]]+)\]\([^)]*\)',r'\1',s)
 return html.escape(s.replace('**','').replace(chr(96),''))
def par(s,style='body'):return Paragraph(plain(s),styles[style])
lines=(A/'CHANGES_CN.md').read_text(encoding='utf-8').splitlines()
flow=[];i=0
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line == '##  Modify File ':
  flow.append(par(line[3:],'h'))
  i+=1
  entries=[]
  while i<len(lines) and not lines[i].startswith('## '):
   value=lines[i].strip()
   if value.startswith('- '):entries.append(value[2:])
   i+=1
  rows=[[par(entries[j],'cell'),par(entries[j+1],'cell') if j+1<len(entries) else ''] for j in range(0,len(entries),2)]
  files=Table(rows,colWidths=[257.5,257.5],hAlign='LEFT')
  files.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
  flow.append(files)
  continue
 if line.startswith('|'):
  cells=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   row=[s.strip() for s in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch(r':?-+:?',s) for s in row):
    cells.append([par(s,'head' if not cells else 'cell') for s in row])
   i+=1
  table=Table(cells,colWidths=[125,235,155],repeatRows=1,hAlign='LEFT')
  table.setStyle(TableStyle([
   ('VALIGN',(0,0),(-1,-1),'TOP'),
   ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f0f4')),
   ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#60798a')),
   ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d9e0e6')),
   ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
   ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
  flow.extend([table,Spacer(1,8)]);continue
 if line.startswith('# '):flow.append(par(line[2:],'title'))
 elif line.startswith('## '):flow.append(par(line[3:],'h'))
 elif line.startswith('- '):flow.append(par('• '+line[2:]))
 else:flow.append(par(line))
 i+=1
def footer(canvas,doc):
 canvas.saveState();canvas.setFont('Chinese',9)
 canvas.setFillColor(colors.HexColor('#65717a'))
 canvas.drawString(40,25,' Revised comparison of papers  · 2026-09-17')
 canvas.drawRightString(A4[0]-40,25,str(doc.page))
 canvas.restoreState()
dest=R/'thesis_changes.pdf'
doc=SimpleDocTemplate(str(dest),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=38,bottomMargin=42,title=' Thesis amends the list item by item ',author='Jiahui Shen')
doc.build(flow,onFirstPage=footer,onLaterPages=footer)
print('CREATED',dest)
