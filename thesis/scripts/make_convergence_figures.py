"""Plot author-recorded detector adaptation gains at the existing thesis size."""
import os
from pathlib import Path
import sys, csv, json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.python-deps'))
sys.path.insert(0,str(Path(os.environ.get('FIGURE_HELPERS_DIR', Path(__file__).resolve().parent/'external_helpers'))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
E=ROOT/'evidence/kitti_lab/convergence_20260917'
QA=ROOT/'audit/convergence_revision_20260917/qa'
QA.mkdir(parents=True,exist_ok=True)
WIDTH=439.2395/72.27
FONT=12*72/72.27
BLUE='#35678a'; GRAY='#858585'; INK='#292929'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
 'font.size':11.955168119551681,'axes.labelsize':FONT,'axes.titlesize':FONT,
 'xtick.labelsize':FONT,'ytick.labelsize':FONT,'legend.fontsize':FONT,
 'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.65,'legend.frameon':False,
 'savefig.bbox':None,'savefig.pad_inches':0})
with (E/'comparison_from_author_records.csv').open(encoding='utf-8') as f:
 rows=list(csv.DictReader(f))
assert len(rows)==4 and all(int(r['frames'])==3769 for r in rows)
labels=[r['detector']+' '+r['line'] for r in rows]
yy=np.arange(4)
fig,axs=plt.subplots(2,1,figsize=(WIDTH,6.0))
fig.subplots_adjust(left=.35,right=.965,top=.93,bottom=.18,hspace=1.10)
def clean(ax):
 ax.set_yticks(yy,labels);ax.set_ylim(3.45,-.45)
 ax.grid(axis='x',lw=.35,color='#d8d8d8');ax.set_axisbelow(True)
 ax.set_xlabel('Difference in AP points')
a=axs[0]
a.plot([float(r['gain_from_unadapted']) for r in rows],yy,'o',color=BLUE,ms=5,label='Over frozen weights')
a.plot([float(r['gain_from_three_epoch']) for r in rows],yy,'s',mfc='white',mec=INK,ms=5,label='Over 3 epochs')
a.set_xlim(-1,25);a.set_xticks([0,5,10,15,20,25])
a.set_title('(a) Recovery after extended adaptation',loc='left',pad=12)
clean(a)
a.legend(loc='upper left',bbox_to_anchor=(-.52,-.36),ncol=2,handlelength=.8,columnspacing=1,handletextpad=.5)
a=axs[1]
a.plot([float(r['gap_to_original']) for r in rows],yy,'D',mfc='white',mec=GRAY,ms=7,label='To original N')
a.plot([float(r['gap_to_line_reference']) for r in rows],yy,'o',color=BLUE,ms=4,label='To line reference')
a.axvline(0,color=GRAY,lw=.7,ls='--')
a.set_xlim(-28,2);a.set_xticks([-25,-20,-15,-10,-5,0])
a.set_title('(b) Remaining baseline deficit',loc='left',pad=12)
clean(a)
a.legend(loc='upper left',bbox_to_anchor=(-.52,-.36),ncol=2,handlelength=.8,columnspacing=1,handletextpad=.5)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=QA/'k_convergence_gains.alignment.json',strict=True)
fig.savefig(ROOT/'figure/k_convergence_gains.pdf')
fig.savefig(ROOT/'figure/k_convergence_gains.svg')
fig.savefig(ROOT/'figure/k_convergence_gains.png',dpi=300)
plt.close(fig)
manifest={'name':'k_convergence_gains','claim':'Extended adaptation improves frozen-input AP but remains below original and line-specific references.',
'width_inches':WIDTH,'height_inches':6.0,'font_pdf_pt':FONT,
'source':'evidence/kitti_lab/convergence_20260917/comparison_from_author_records.csv',
'uncertainty':'No interval: single validation-selected run per schedule, not independent repeated trials.',
'baseline':'Original N official for both lines; line-specific reference is original official for A and sparse 3-epoch adapted for B.',
'data_integrity':'All four conditions included; subtraction only; no inferred epochs.'}
(QA/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('CREATED k_convergence_gains')

# The source report explicitly supplies this complete six-epoch terminal window.
# Earlier epochs are not reconstructed from maxima or schedule summaries.
with (E/'pointrcnn_B_reported_tail.csv').open(encoding='utf-8') as f:
 tail=list(csv.DictReader(f))
epochs=[int(r['epoch']) for r in tail]
aps=[float(r['car_moderate_ap_r40']) for r in tail]
assert epochs==list(range(19,25))
fig,ax=plt.subplots(figsize=(WIDTH,3.15))
fig.subplots_adjust(left=.15,right=.97,bottom=.23,top=.84)
ax.plot(epochs,aps,'o-',color=GRAY,lw=1,ms=4)
ax.plot(epochs[0],aps[0],'o',color=BLUE,ms=6)
ax.set_title('PointRCNN B: terminal RCNN validation',loc='left',pad=12)
ax.set_xlabel('Epoch in the 24-epoch schedule')
ax.set_ylabel('Car 3D AP (R40, %)')
ax.set_xlim(18.8,24.2);ax.set_xticks(epochs)
ax.set_ylim(54.9,57.5);ax.set_yticks([55,56,57])
ax.grid(axis='y',lw=.35,color='#d8d8d8');ax.set_axisbelow(True)
fig.canvas.draw()
require_matplotlib_panel_alignment(fig,json_out=QA/'k_convergence_terminal.alignment.json',strict=True)
fig.savefig(ROOT/'figure/k_convergence_terminal.pdf')
fig.savefig(ROOT/'figure/k_convergence_terminal.svg')
fig.savefig(ROOT/'figure/k_convergence_terminal.png',dpi=300)
plt.close(fig)
(QA/'terminal_figure_source.json').write_text(json.dumps({
 'source':'evidence/kitti_lab/convergence_20260917/pointrcnn_B_reported_tail.csv',
 'selection':'All six terminal epochs explicitly tabulated in author record; preceding 18 epochs unavailable locally and not inferred.',
 'claim':'No terminal score exceeded the selected epoch-19 checkpoint.',
 'frames_per_epoch':3769,'metric':'Car 3D Moderate AP_R40','font_pdf_pt':FONT,'width_inches':WIDTH
},indent=2),encoding='utf-8')
