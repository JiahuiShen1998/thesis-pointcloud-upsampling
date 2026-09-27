"""Render the compiled PDF and evidence figures for visual review."""
from pathlib import Path
import sys,json,collections
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'.python-deps'))
import pymupdf as fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
Q=R/'audit/revision_20260913/visual';Q.mkdir(exist_ok=True)
def arr(page,scale):
    pix=page.get_pixmap(matrix=fitz.Matrix(scale,scale),alpha=False)
    return np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)
d=fitz.open(R/'thesis.pdf');pages=[]
for i,p in enumerate(d):
    text=p.get_text();spans=[s for b in p.get_text('dict')['blocks'] if 'lines'in b for l in b['lines'] for s in l['spans']]
    fonts=collections.Counter(round(s['size'],3) for s in spans)
    pages.append(dict(page=i+1,label=p.get_label(),words=len(text.split()),fonts=dict(fonts),figures=[l for l in text.splitlines() if l.startswith('Figure ')],text=text))
    if 'Figure ' in text or ('Conclusion and Outlook' in text and i>30):
        p.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(Q/f'page_{i+1:03}.png')
for start in range(0,len(d),12):
    fig,axs=plt.subplots(3,4,figsize=(12,12.7));fig.subplots_adjust(left=.01,right=.99,top=.975,bottom=.01,hspace=.09,wspace=.05)
    for j,ax in enumerate(axs.flat):
        idx=start+j;ax.axis('off')
        if idx>=len(d):continue
        ax.imshow(arr(d[idx],.55));ax.set_title(f'PDF {idx+1} / {d[idx].get_label()}',fontsize=10,pad=2)
    fig.savefig(Q/f'pages_{start+1:03}_{min(start+12,len(d)):03}.png',dpi=150);plt.close(fig)
ff=sorted((R/'figures/final').glob('*.pdf'))
for start in range(0,len(ff),4):
    fig,axs=plt.subplots(2,2,figsize=(12,12));fig.subplots_adjust(left=.015,right=.985,top=.97,bottom=.01,hspace=.13,wspace=.07)
    for j,ax in enumerate(axs.flat):
        idx=start+j;ax.axis('off')
        if idx>=len(ff):continue
        f=fitz.open(ff[idx]);ax.imshow(arr(f[0],1.6));ax.set_title(ff[idx].stem,fontsize=12,pad=8)
    fig.savefig(Q/f'figures_{start+1:02}.png',dpi=150);plt.close(fig)
(Q/'page_inventory.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
print('PDF pages:',len(d),'Sparse nonblank pages:',[(p['page'],p['label'],p['words']) for p in pages if 4<p['words']<80])
print('Figures:',[(p['page'],p['figures']) for p in pages if p['figures']])
