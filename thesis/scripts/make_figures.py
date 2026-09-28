"""Replot audited observations at final thesis width; no synthetic data."""
import os
from pathlib import Path
import sys, json, csv
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'.python-deps'))
sys.path.insert(0, str(Path(os.environ.get('FIGURE_HELPERS_DIR', Path(__file__).resolve().parent/'external_helpers'))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT=ROOT/'figure'; OUT.mkdir(parents=True,exist_ok=True)
QA=ROOT/'audit/revision_20260913/qa'; QA.mkdir(parents=True,exist_ok=True)
M=ROOT/'evidence/modelnet40_hpc/data'; K=ROOT/'evidence/kitti_lab'
WIDTH=439.2395/72.27
FONT=11.955168119551681 # 12 TeX points expressed in PDF/PostScript points.
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':FONT,
 'axes.labelsize':FONT,'axes.titlesize':FONT,'xtick.labelsize':FONT,
 'ytick.labelsize':FONT,'legend.fontsize':FONT,'figure.titlesize':FONT,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.65,
 'savefig.bbox':None,'savefig.pad_inches':0,'legend.frameon':False})
BLUE='#35678a'; INK='#292929'; GRAY='#858585'
METHODS=['EAR','PDANS','PU-Net','PU-GCN','PU-EdgeFormer']
COLORS=['#777777','#3c6a8c','#976745','#536c48','#806281']
MAN=[]
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(fig,name,claim,source,**opts):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=QA/(name+'.alignment.json'),strict=True,**opts)
    fig.savefig(OUT/(name+'.pdf'))
    fig.savefig(OUT/(name+'.svg'))
    fig.savefig(OUT/(name+'.png'),dpi=300)
    MAN.append(dict(name=name,claim=claim,source=source,width_inches=WIDTH,
                    font_pdf_pt=FONT,height_inches=float(fig.get_size_inches()[1])))
    plt.close(fig)
def clean(ax,x=True):
    ax.grid(axis='x' if x else 'y',lw=.35,color='#d8d8d8');ax.set_axisbelow(True)
def rows2(height=5.5,left=.30):
    fig,ax=plt.subplots(2,1,figsize=(WIDTH,height))
    fig.subplots_adjust(left=left,right=.96,bottom=.11,top=.94,hspace=.65)
    return fig,ax

cls=read(M/'classification_results.csv')
fig,axs=rows2()
for ax,line in zip(axs,'AB'):
    rows=[r for r in cls if r['line']==line]; yy=np.arange(len(rows))
    best=[float(r['best_oa_percent']) for r in rows];end=[float(r['final_oa_percent']) for r in rows]
    ax.hlines(yy,end,best,color=GRAY,lw=1)
    ax.plot(best,yy,'s',color=INK,ms=5,label='Best OA')
    ax.plot(end,yy,'o',mfc='white',mec=BLUE,ms=5,label='Epoch 200')
    ax.set_yticks(yy,[r['method'].replace('Original baseline','Original').replace('Downsampled x4 baseline','Sparse') for r in rows]);ax.invert_yaxis()
    ax.set_xlim(87,93);ax.set_xticks([88,90,92]);ax.set_title('Line '+line,loc='left');clean(ax)
axs[-1].set_xlabel('Overall accuracy (%)');fig.subplots_adjust(top=.84)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.995),ncol=2)
save(fig,'mn_accuracy','No Line A gain; only PU-Net recovers Line B accuracy','HPC classification_results.csv')

curves=read(M/'training_curves.csv');fig,axs=rows2(5.8,.15)
for ax,line in zip(axs,'AB'):
    for i,method in enumerate(['Original' if line=='A' else 'Sparse']+METHODS):
        rows=[r for r in curves if r['line']==line and (r['method']==method or (method=='Sparse' and r['method'] in ['Downsampled x4','Sparse 256','Downsampled x4 baseline']))]
        if not rows and method=='Sparse':
            rows=[r for r in curves if r['line']==line and int(r['points'])==256]
        assert len(rows)==200,(line,method,len(rows))
        ax.plot([int(r['epoch']) for r in rows],[float(r['test_oa_percent']) for r in rows],lw=.65,color=INK if i==0 else COLORS[i-1],label='Baseline' if i==0 else method)
    ax.set(xlim=(1,200),ylim=(45,95),ylabel='Test OA (%)');ax.set_title('Line '+line,loc='left');clean(ax,False)
axs[-1].set_xlabel('Epoch');fig.subplots_adjust(top=.79)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,handlelength=1,columnspacing=.8)
save(fig,'mn_training','Single-seed training trajectories contextualize best-epoch selection','HPC training_curves.csv')

fig,ax=plt.subplots(figsize=(WIDTH,3.1));fig.subplots_adjust(left=.36,right=.96,bottom=.23,top=.87)
labels=['Mesh-ref 256','Sparse 256','Original 1,024','Mesh-ref 4,096'];best=[90.88,90.85,91.95,92];last=[90.38,90.46,91.31,91.18]
ax.hlines(range(4),last,best,color=GRAY);ax.plot(best,range(4),'s',color=INK,label='Best');ax.plot(last,range(4),'o',mfc='white',mec=BLUE,label='Final')
ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set(xlim=(90,92.3),xlabel='Overall accuracy (%)');clean(ax);ax.legend(ncol=2,loc='lower right',bbox_to_anchor=(1,1.03))
save(fig,'mn_controls','Genuine density gives a small selected-checkpoint difference above 1024','HPC WORK_RECORD.md; four reused training shapes limit controls')

sa=read(M/'sa1_reception_results.csv');fig,ax=plt.subplots(figsize=(WIDTH,5.8));fig.subplots_adjust(left=.43,right=.96,bottom=.12,top=.89)
yy=np.arange(len(sa));ax.plot([float(r['discarded_percent']) for r in sa],yy,'s',color=INK,label='Discarded');ax.plot([float(r['repeated_slots_percent']) for r in sa],yy,'o',mfc='white',mec=BLUE,label='Repeated slots')
ax.set_yticks(yy,[r['method'].replace('Downsampled x4','Sparse')+' / '+r['bucket_points'] for r in sa]);ax.invert_yaxis();ax.set(xlim=(-2,102),xlabel='Local grouping rate (%)');clean(ax);ax.legend(ncol=2,loc='lower right',bbox_to_anchor=(1,1.03),handlelength=.8)
save(fig,'mn_sa1','Local input budgets limit reception of dense points','HPC sa1_reception_results.csv')

geom=read(M/'geometry_equal_n_distribution_summary.csv');fig,axs=plt.subplots(2,2,figsize=(WIDTH,5.6));fig.subplots_adjust(left=.28,right=.97,bottom=.11,top=.92,hspace=.60,wspace=.36)
for i,line in enumerate('AB'):
    for j,metric in enumerate(['CD_L2','HD']):
        ax=axs[i,j];rows=[next(r for r in geom if r['line']==line and r['metric']==metric and r['method']==m) for m in METHODS]
        mean=np.array([float(r['mean']) for r in rows]);low=np.array([float(r['bootstrap_ci_low']) for r in rows]);high=np.array([float(r['bootstrap_ci_high']) for r in rows])
        ax.errorbar(mean,range(5),xerr=[mean-low,high-mean],fmt='o',color=BLUE,ms=4,capsize=2)
        ax.set_yticks(range(5),METHODS if j==0 else ['']*5);ax.invert_yaxis();ax.set_title('Line '+line+' / '+('CD' if j==0 else 'HD'),loc='left');clean(ax)
        ax.set_xlim((.035,.085) if j==0 else (.075,.21));ax.set_xticks([.04,.06,.08] if j==0 else [.1,.2])
save(fig,'mn_geometry','PU-GCN leads CD but PU-Net leads Line B HD','HPC geometry_equal_n_distribution_summary.csv')

dirs=read(M/'geometry_directional_multiscale_summary.csv');fig,axs=rows2(5.0)
for ax,line in zip(axs,'AB'):
    rows=[next(r for r in dirs if r['line']==line and r['method']==m) for m in METHODS];a=np.array([float(r['cd_forward']) for r in rows]);b=np.array([float(r['cd_backward']) for r in rows])
    ax.barh(range(5),a,color=GRAY,label='Output to reference');ax.barh(range(5),b,left=a,color=BLUE,label='Reference to output')
    ax.set_yticks(range(5),METHODS);ax.invert_yaxis();ax.set_xlim(0,.085);ax.set_title('Line '+line,loc='left');clean(ax)
axs[-1].set_xlabel('Mean unsquared distance');fig.subplots_adjust(top=.79)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.995),ncol=1)
save(fig,'mn_directional','CD combines coverage and deviation','HPC geometry_directional_multiscale_summary.csv')

refs=[r for r in read(M/'geometry_reference_sensitivity.csv') if r['metric']=='CD_L2'];fig,ax=plt.subplots(figsize=(WIDTH,3.5));fig.subplots_adjust(left=.10,right=.66,bottom=.20,top=.89)
for m,c in zip(METHODS,COLORS):
    r=next(r for r in refs if r['method']==m);a,b=float(r['rank_original1024']),float(r['rank_equalN']);ax.plot([0,1],[a,b],'o-',color=c,lw=1,ms=4);ax.text(1.07,b,m,va='center')
ax.set(xlim=(-.1,1.05),ylim=(5.4,.6));ax.set_xticks([0,1],['Original 1,024','Mesh-ref 4,096']);ax.set_yticks(range(1,6));ax.set_ylabel('CD rank');clean(ax,False)
save(fig,'mn_reference','Changing the reference changes the apparent ranking','HPC geometry_reference_sensitivity.csv')

def project(points):
    a=np.deg2rad(35);b=np.deg2rad(17);x,y,z=points[:,:3].T
    u=np.cos(a)*x+np.sin(a)*z;v=-np.sin(b)*(-np.sin(a)*x+np.cos(a)*z)+np.cos(b)*y
    return np.c_[u,v]
for line,obj,name in [('A','airplane_0627','mn_cloud_a'),('B','chair_0890','mn_cloud_b')]:
    variants=[('input','Input'),('reference','Reference'),('ear','EAR'),('pdans','PDANS'),('punet','PU-Net'),('pugcn','PU-GCN'),('puedgeformer','PU-EdgeFormer')]
    arrays=[np.load(M/f'pointcloud_parallel/line{line}_{obj}_{v}.npy') for v,_ in variants];ps=[project(p) for p in arrays];allp=np.concatenate(ps);lo=allp.min(0);hi=allp.max(0);center=(lo+hi)/2;span=max(hi-lo)*1.1
    fig,axs=plt.subplots(4,2,figsize=(WIDTH,6.2));fig.subplots_adjust(left=.025,right=.98,top=.94,bottom=.015,wspace=.18,hspace=.40)
    for j,(ax,p,(_,title)) in enumerate(zip(axs.flat,ps,variants)):
        ax.scatter(p[:,0],p[:,1],s=.75,c=INK if j<2 else BLUE,linewidths=0,rasterized=True)
        ax.set_xlim(center[0]-span/2,center[0]+span/2);ax.set_ylim(center[1]-span/2,center[1]+span/2)
        ax.set_aspect('equal',adjustable='box');ax.axis('off');ax.set_title(f'({chr(97+j)}) {title} / {len(p):,}',pad=4)
    axs.flat[-1].set_visible(False)
    save(fig,name,'Identical views compare measured, reference, and generated geometry',f'HPC pointcloud_parallel/{obj}; all array rows rendered')

matrix=read(K/'full_val_ap_r40.csv')
for detector,name in [('PointRCNN','k_pointrcnn'),('CenterPoint','k_centerpoint')]:
    fig,axs=rows2(6.0,.36)
    for ax,line in zip(axs,'AB'):
        rows=[r for r in matrix if r['detector']==detector and r['line']==line]
        for j,r in enumerate(rows):ax.plot(float(r['moderate']),j,'s' if r['weights']=='official' else 'o',color=INK if r['weights']=='official' else BLUE,ms=5)
        labels=[('Baseline' if r['input'].startswith('baseline') else ('Direct' if 'direct' in r['input'] else 'Observed-first'))+' / '+('F' if r['weights']=='official' else 'A') for r in rows]
        ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.set(xlim=(0,100),xticks=[0,25,50,75,100]);ax.set_title('Line '+line+' (F: frozen; A: adapted)',loc='left');clean(ax)
    axs[-1].set_xlabel('Car Moderate 3D AP_R40 (%)')
    save(fig,name,'Adaptation recovers performance without exceeding the reference','lab full_val_ap_r40.csv; 3769 frames per arm')

allcls=read(K/'all_classes_ap_r40.csv');fig,axs=rows2(4.6,.24)
for ax,line in zip(axs,'ab'):
    target=[r for r in allcls if r['detector']=='CenterPoint' and r['metric']=='3d_ap_r40' and r['arm']==f'line_{line}_pugcn_observed_first_adapted']
    if not target:
        candidates=sorted({r['arm'] for r in allcls if r['detector']=='CenterPoint'});print('ARM KEYS',candidates)
        target=[r for r in allcls if r['detector']=='CenterPoint' and r['metric']=='3d_ap_r40' and f'line_{line}_' in r['arm'] and 'observed' in r['arm'] and 'adapted' in r['arm']]
    base=[r for r in allcls if r['detector']=='CenterPoint' and r['metric']=='3d_ap_r40' and r['arm']==f'line_{line}_baseline_'+('official' if line=='a' else 'adapted')]
    classes=['Car','Pedestrian','Cyclist']
    for j,c in enumerate(classes):
        a=float(next(r for r in base if r['class']==c)['moderate']);b=float(next(r for r in target if r['class']==c)['moderate']);ax.plot([a,b],[j,j],color=GRAY);ax.plot(a,j,'s',color=INK,label='Reference' if j==0 else None);ax.plot(b,j,'o',color=BLUE,label='Observed-first adapted' if j==0 else None)
    ax.set_yticks(range(3),classes);ax.invert_yaxis();ax.set(xlim=(20,90),xticks=[20,40,60,80]);ax.set_title('Line '+line.upper(),loc='left');clean(ax)
axs[-1].set_xlabel('Moderate 3D AP_R40 (%)');fig.subplots_adjust(top=.77)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.995),ncol=1)
save(fig,'k_classes','Final adaptation leaves a deficit in all three Moderate class results','lab all_classes_ap_r40.csv; 3769 frames per arm')

legacy=read(K/'centerpoint_legacy_ap_r40.csv');fig,axs=rows2(5.6,.39)
for ax,line in zip(axs,'AB'):
    rows=[r for r in legacy if r['input'].startswith('Line '+line)]
    base=next(r for r in legacy if r['input']==('Original baseline' if line=='A' else 'Downsampled baseline'));rows=[base]+rows
    for key,m,c in [('car','s',INK),('pedestrian','o',BLUE),('cyclist','^','#976745')]:ax.plot([float(r[key]) for r in rows],range(5),m,color=c,label=key.capitalize(),ms=5)
    labels=[r['input'].replace('Line '+line+' ','').replace('Original baseline','Original').replace('Downsampled baseline','Sparse').replace('PU-EdgeFormer','PU-EF compat') for r in rows]
    ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set(xlim=(0,100),xticks=[0,25,50,75,100]);ax.set_title('Line '+line+' / earlier integration',loc='left');clean(ax)
axs[-1].set_xlabel('Moderate 3D AP_R40 (%)');fig.subplots_adjust(top=.84)
fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,handlelength=.7,columnspacing=.8)
save(fig,'k_legacy','Earlier integration has class-specific exceptions and known implementation faults','lab centerpoint_legacy_ap_r40.csv')

norm=read(K/'normalization_patch_pilot256.csv');fig,ax=plt.subplots(figsize=(WIDTH,3.6));fig.subplots_adjust(left=.32,right=.96,bottom=.20,top=.81)
for key,c,m in [('pointrcnn',INK,'s'),('centerpoint',BLUE,'o')]:ax.plot([float(r[key]) for r in norm],range(5),m,color=c,label='PointRCNN' if key=='pointrcnn' else 'CenterPoint',ms=5)
ax.set_yticks(range(5),[r['configuration'] for r in norm]);ax.invert_yaxis();ax.set(xlim=(0,100),xlabel='Recorded Moderate 3D AP (%)');clean(ax);ax.legend(loc='lower right',bbox_to_anchor=(1,1.03),ncol=2)
save(fig,'k_normalization','Normalisation and locality both affect the flawed PU-Net integration','lab diagnostic record; 256 frames; historical evaluator scope')

vox=read(K/'voxel_audit32.csv');fig,ax=plt.subplots(figsize=(WIDTH,3.8));fig.subplots_adjust(left=.34,right=.95,bottom=.21,top=.88)
ax.barh(range(5),[float(r['median_voxels'])/1000 for r in vox],color=GRAY);ax.axvline(40,color=BLUE,ls='--',lw=1)
for j,r in enumerate(vox):ax.text(65,j,r['cap_frames']+'/32',va='center')
ax.set_yticks(range(5),[r['method'].replace('PU-EdgeFormer','PU-EF compat') for r in vox]);ax.invert_yaxis();ax.set(xlim=(0,78),xlabel='Median occupied voxels (thousands)')
ax.set_title('Labels: frames reaching the cap',loc='left')
save(fig,'k_voxels','Dense outputs frequently exceed the voxel cap in the 32-frame audit','lab voxel_audit32.csv')

clouds=[np.fromfile(K/n,dtype=np.float32).reshape(-1,4) for n in ['original_006833.bin','sparse_006833.bin','legacy_direct_a_006833.bin','legacy_direct_b_006833.bin']]
fig,axs=plt.subplots(2,2,figsize=(WIDTH,5.3));fig.subplots_adjust(left=.13,right=.98,bottom=.13,top=.91,wspace=.3,hspace=.53)
for j,(ax,p,title) in enumerate(zip(axs.flat,clouds,['Original','Sparse','Earlier direct A','Earlier direct B'])):
    keep=(p[:,0]>=0)&(p[:,0]<=60)&(p[:,1]>=-20)&(p[:,1]<=20);q=p[keep]
    ax.scatter(q[:,0],q[:,1],s=.13,c=INK if j<2 else BLUE,linewidths=0,rasterized=True)
    ax.set(xlim=(0,60),ylim=(-20,20),xticks=[0,30,60],yticks=[-20,0,20]);ax.set_title(f'({chr(97+j)}) {title}',loc='left');ax.set_xlabel('Forward x (m)')
    if j%2==0:ax.set_ylabel('Lateral y (m)')
    MAN.append(dict(crop=name if False else title,before=len(p),after=int(keep.sum()),rule='0 <= x <= 60; -20 <= y <= 20; all heights; all in-window rows'))
save(fig,'k_clouds','A shared view separates native measurements from generated cardinality','lab 006833 arrays matching the supplied core figure manifest')

# Real sensing motivation: retain all points in two identically sized crops.
fig,axs=plt.subplots(2,1,figsize=(WIDTH,4.4));fig.subplots_adjust(left=.14,right=.97,bottom=.14,top=.92,hspace=.60)
for ax,(cx,cy,title) in zip(axs,[(8, -1,'Near region'),(42,-1,'Far region')]):
    p=clouds[0];sel=(abs(p[:,0]-cx)<=3)&(abs(p[:,1]-cy)<=2)&(p[:,2]>=-2.5)&(p[:,2]<=1.5);q=p[sel]
    ax.scatter(q[:,0]-cx,q[:,2],s=1,c=INK,linewidths=0,rasterized=True);ax.set(xlim=(-3,3),ylim=(-2.5,1.5),ylabel='Height z (m)');ax.set_title(f'{title}: {len(q):,} measured returns',loc='left')
axs[-1].set_xlabel('Local forward position (m)')
save(fig,'k_sparsity','Equal-size spatial windows can have very different observed support','lab original_006833.bin; windows centered at (8,-1) and (42,-1), half widths (3,2), z in [-2.5,1.5]; counts are region examples, not isolated vehicles')

def diagram(name,boxes,claim,source,height=4.0):
    fig,ax=plt.subplots(figsize=(WIDTH,height));fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.975);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ys=np.linspace(.88,.12,len(boxes))
    for i,(y,txt) in enumerate(zip(ys,boxes)):
        ax.add_patch(FancyBboxPatch((.08,y-.063),.84,.126,boxstyle='round,pad=0.006',facecolor='#f5f5f5',edgecolor=GRAY,lw=.7));ax.text(.5,y,txt,ha='center',va='center')
        if i<len(boxes)-1:ax.annotate('',xy=(.5,ys[i+1]+.076),xytext=(.5,y-.075),arrowprops=dict(arrowstyle='->',color=INK,lw=.9))
    save(fig,name,claim,source)
diagram('pointnet_principle',['Unordered XYZ points','Farthest-point sampling and local grouping','Shared point features and local max pooling','Repeated set abstraction and global aggregation','Object-class prediction'],'Hierarchical grouping explains density-dependent reception','Schematic adapted from PointNet++ mechanism; no experiment-specific parameters',4.0)
diagram('pugcn_principle',['Sparse XYZ patch','Inception DenseGCN: local multiscale features','NodeShuffle: expand features into more point slots','Coordinate reconstruction','Dense XYZ patch'],'PU-GCN expands local graph features into coordinates','Schematic based on Qian et al. 2021',4.0)
diagram('patch_pipeline',['Measured scene in metric coordinates','Unique local supports: 2,048 points per patch','Centre and scale each patch; fixed PU-GCN','Invert transform; merge 8,192-point outputs','Strict selection; transfer intensity from observations'],'Local adaptation must preserve coordinate and support meaning','lab final protocol',4.3)
def branches(name,top,left,right,bottom,claim):
    fig,ax=plt.subplots(figsize=(WIDTH,3.8));fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.975);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    for x,y,w,h,txt in [(.05,.77,.90,.17,top),(.015,.33,.46,.29,left),(.525,.33,.46,.29,right),(.05,.06,.90,.14,bottom)]:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.004',facecolor='#f5f5f5',edgecolor=GRAY,lw=.7));ax.text(x+w/2,y+h/2,txt,ha='center',va='center',linespacing=1.45)
    for x in [.245,.755]:
        ax.annotate('',xy=(x,.63),xytext=(.5,.76),arrowprops=dict(arrowstyle='->',color=INK,lw=.9))
        ax.annotate('',xy=(.5,.21),xytext=(x,.32),arrowprops=dict(arrowstyle='->',color=INK,lw=.9))
    save(fig,name,claim,'lab final protocol; alternative detector branches')
branches('input_budget','Strict scene: 4N or 4M rows','PointRCNN\nField-of-view filtering\nSample 16,384 rows','CenterPoint\nSpatial-range filtering\n5 points / voxel\n40,000 test voxels','Detection after the input adapter','File cardinality differs from model input capacity')
branches('adaptation_design','Fixed PU-GCN: PU1K model-100\nDirect or observed-first inputs','PointRCNN\n3,712 training frames\nRPN: 3 epochs\nRCNN: 3 epochs','CenterPoint\n3,712 training frames\n3 epochs','Evaluate all 3,769 validation frames','Later training updates detectors, not the upsampler')
branches('overview','Line A: densification\nLine B: recovery after sparsification','ModelNet40 (HPC)\nA: 1,024 to 4,096\nB: 256 to 1,024\nTrain PointNet++','KITTI (lab)\nA: N to 4N\nB: M to 4M\nFrozen / adapted detectors','Geometry, task scores, and input diagnostics','Two independently executed domains test density at different levels')
(OUT/'figure_manifest.json').write_text(json.dumps(MAN,indent=2),encoding='utf-8')
print('Exported',len([r for r in MAN if 'name' in r]),'figures at',FONT,'PDF points.')
