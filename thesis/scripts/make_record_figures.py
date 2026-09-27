"""Plot completed historical controls from archived HPC/lab sources (Python backend)."""
from pathlib import Path
import sys,csv,json
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'.python-deps'))
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_panel_alignment import require_matplotlib_panel_alignment
K=R/'evidence/kitti_lab/primary_20260913';O=R/'figure';Q=R/'audit/manuscript_revision_20260913/qa'
O.mkdir(exist_ok=True);Q.mkdir(parents=True,exist_ok=True)
W=439.2395/72.27;FONT=12*72/72.27
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
 'font.size':FONT,'axes.labelsize':FONT,'axes.titlesize':FONT,'xtick.labelsize':FONT,
 'ytick.labelsize':FONT,'legend.fontsize':FONT,'figure.titlesize':FONT,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.65,'savefig.bbox':None,
 'legend.frameon':False,'savefig.pad_inches':0})
BLUE='#286b91';RED='#a45154';GRAY='#737b83';MAN=[]
def read(n):
 with (K/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(fig,name,claim,source):
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=Q/(name+'.alignment.json'),strict=True)
 fig.savefig(O/(name+'.pdf'),dpi=600)
 fig.savefig(O/(name+'.svg'),dpi=600)
 fig.savefig(O/(name+'.png'),dpi=600)
 MAN.append({'name':name,'claim':claim,'source':source,'width_inches':W,'height_inches':float(fig.get_size_inches()[1]),'font_pdf_pt':FONT})
 plt.close(fig);print('EXPORTED',name,flush=True)
def grid(ax):
 ax.grid(axis='x',color='#d9dde0',lw=.5);ax.set_axisbelow(True)
e2=[r for r in read('e1_e2_live_ap_summary.csv') if r['experiment']=='E2'];e3=read('e3_real_first_32768_live_ap_summary.csv')
fig,axs=plt.subplots(2,1,figsize=(W,5.8));fig.subplots_adjust(left=.29,right=.96,top=.87,bottom=.10,hspace=.48)
for ax,line in zip(axs,'AB'):
 prefix='original_x4_' if line=='A' else 'downsampled_x4_'
 keys=[('original_baseline' if line=='A' else 'downsampled_x4_baseline')]+[prefix+x for x in ['pdans','pu_gcn','pu_edgeformer','pu_net']]
 a=[float(next(r for r in e2 if r['variant']==k)['3d_ap_moderate']) for k in keys];b=[float(next(r for r in e3 if r['variant']==k)['3d_ap_moderate']) for k in keys]
 for i,(x,y) in enumerate(zip(a,b)):ax.plot([x,y],[i,i],color=GRAY,lw=1)
 ax.scatter(a,range(5),c=GRAY,marker='o',s=27,label='E2 / 16,384')
 ax.scatter(b,range(5),c=BLUE,marker='D',s=27,label='E3 / 32,768')
 ax.set(yticks=range(5),yticklabels=['Baseline','PDANS','PU-GCN','PU-EF compat','PU-Net old'],xlim=(0,100),xticks=[0,25,50,75,100],ylim=(4.5,-.5))
 ax.set_title('Line '+line,loc='left');grid(ax)
axs[1].set_xlabel('Car Moderate 3D AP R40 (%)')
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.57,1),columnspacing=.8,handletextpad=.4)
save(fig,'k_e2_e3','Larger observed-first input recovers some dense-input AP but does not restore paired baselines',['evidence/kitti_lab/primary_20260913/e1_e2_live_ap_summary.csv','evidence/kitti_lab/primary_20260913/e3_real_first_32768_live_ap_summary.csv'])
rows=[]
for line in (K/'PDANS_DETECTOR_DISAGREEMENT_ROOT_CAUSE.md').read_text(encoding='utf-8').splitlines():
 fields=[x.strip() for x in line.strip('|').split('|')]
 if len(fields)==5 and fields[1]=='3D AP_R40':rows.append(fields)
assert len(rows)==4
# Labels translate the four source rows; values are parsed from the source table.
labels=['PointRCNN / Car','CenterPoint / Car','CenterPoint / Ped.','CenterPoint / Cyc.']
a=[float(r[2]) for r in rows];b=[float(r[3]) for r in rows]
fig,ax=plt.subplots(figsize=(W,3.55));fig.subplots_adjust(left=.35,right=.96,top=.81,bottom=.22)
for i,(x,y) in enumerate(zip(a,b)):ax.plot([x,y],[i,i],color=GRAY,lw=1.2)
ax.scatter(a,range(4),c=GRAY,s=32,label='Old patches');ax.scatter(b,range(4),c=BLUE,marker='D',s=32,label='First local patches')
ax.set(yticks=range(4),yticklabels=labels,xlim=(0,100),xticks=[0,25,50,75,100],ylim=(3.5,-.5),xlabel='Recorded Moderate 3D AP (%)');grid(ax)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=2,bbox_to_anchor=(.51,1),columnspacing=.7,handletextpad=.3)
save(fig,'k_patch_response','First local PDANS patches produced opposite detector responses on the same 256 frames',['evidence/kitti_lab/primary_20260913/PDANS_DETECTOR_DISAGREEMENT_ROOT_CAUSE.md'])
d=json.loads((K/'subset_size_power_probe.json').read_text());sizes=[20,50,100,256,512]
keys=list(d['by_size']['20']['paired_deltas'])
fig,ax=plt.subplots(figsize=(W,4.35));fig.subplots_adjust(left=.13,right=.97,top=.73,bottom=.16)
for k,label,col,mark in zip(keys,['Sparse − original','PDANS − original','PDANS − sparse'],[GRAY,RED,BLUE],['o','s','D']):
 y=[d['by_size'][str(n)]['paired_deltas'][k]['std'] for n in sizes]
 ax.plot(range(5),y,marker=mark,color=col,lw=1.2,ms=5,label=label)
ax.set(xticks=range(5),xticklabels=[str(n) for n in sizes],xlabel='Frames in each subset',ylabel='SD of paired AP difference (pp)',ylim=(0,10),yticks=[0,2,4,6,8,10])
ax.grid(axis='y',color='#d9dde0',lw=.5);ax.set_axisbelow(True)
fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=1,bbox_to_anchor=(.58,1.01),labelspacing=.25)
save(fig,'k_subset_scale','AP contrast variability decreases with frame-subset size and differs by contrast',['evidence/kitti_lab/primary_20260913/subset_size_power_probe.json','evidence/kitti_lab/primary_20260913/subset_size_power_probe.py'])
(Q/'new_figure_manifest.json').write_text(json.dumps(MAN,indent=2),encoding='utf-8')
print('All figures exported at 12 TeX pt on the final thesis text width.',flush=True)
