"""Rebuild manuscript figures from archived HPC/lab observations."""
import os
from pathlib import Path
import sys,json,csv,math
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'.python-deps'))
sys.path.insert(0,str(Path(os.environ.get('FIGURE_HELPERS_DIR', Path(__file__).resolve().parent/'external_helpers'))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from audit_panel_alignment import require_matplotlib_panel_alignment
W=439.2395/72.27
FONT=12*72/72.27
O=R/'figure';Q=R/'audit/results_expansion_20260913/qa'
Q.mkdir(parents=True,exist_ok=True)
M=R/'evidence/modelnet40_hpc/data'
K=R/'evidence/kitti_lab'
V=K/'visual_audit'
D=V/'reports/dual_detector_three_frame_root_cause_20260730'
METHODS=['EAR','PDANS','PU-Net','PU-GCN','PU-EdgeFormer']
COLORS=['#797979','#8e7248','#a45154','#286b91','#688960']
LAB_METHODS=['PDANS','PU-GCN','PU-EdgeFormer','PU-Net']
SLUG={'PDANS':'pdans','PU-GCN':'pu_gcn','PU-EdgeFormer':'pu_edgeformer','PU-Net':'pu_net'}
INK='#252c33';BLUE='#286b91';GRAY='#9c9fa2';RED='#a45154';GREEN='#518069'
MAN=[];STATS={}
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
 'font.size':FONT,'axes.labelsize':FONT,'axes.titlesize':FONT,'xtick.labelsize':FONT,
 'ytick.labelsize':FONT,'legend.fontsize':FONT,'figure.titlesize':FONT,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
 'axes.spines.right':False,'axes.linewidth':.65,'savefig.bbox':None,
 'legend.frameon':False,'savefig.pad_inches':0})
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(fig,name,claim,source,**kw):
 fig.canvas.draw()
 require_matplotlib_panel_alignment(fig,json_out=Q/(name+'.alignment.json'),strict=True,**kw)
 fig.savefig(O/(name+'.pdf'),dpi=600)
 fig.savefig(O/(name+'.svg'),dpi=600)
 fig.savefig(O/(name+'.png'),dpi=600)
 MAN.append({'name':name,'claim':claim,'source':source,'width_inches':W,
 'height_inches':float(fig.get_size_inches()[1]),'font_pdf_pt':FONT})
 plt.close(fig);print('EXPORTED',name,flush=True)
def clean(ax,axis='x'):
 ax.grid(axis=axis,color='#d9dde0',lw=.45);ax.set_axisbelow(True)
def short(m):
 return {'PU-EdgeFormer':'PU-EF','PU-Net':'PU-Net*'}.get(m,m)
def rows2(h=5.7,left=.18):
 fig,axs=plt.subplots(2,1,figsize=(W,h))
 fig.subplots_adjust(left=left,right=.96,top=.90,bottom=.11,hspace=.63)
 return fig,axs

def modelnet_stats():
 delta=read(M/'classification_per_class_delta.csv')
 classes=sorted({r['class_name'] for r in delta})
 for line in 'AB':
  for half in range(2):
   cc=classes[half*20:(half+1)*20]
   a=np.array([[float(next(r['delta_pp'] for r in delta if r['line']==line and r['class_name']==c and r['method']==m)) for m in METHODS] for c in cc])
   fig,ax=plt.subplots(figsize=(W,6.4));fig.subplots_adjust(left=.26,right=.97,top=.92,bottom=.17)
   im=ax.imshow(a,cmap='RdBu',vmin=-25,vmax=25,aspect='auto',interpolation='none')
   ax.set_yticks(range(20),[x.replace('_',' ') for x in cc])
   ax.set_xticks(range(5),['EAR','PDANS','PU-Net','PU-GCN','PU-EF'])
   ax.xaxis.tick_top();ax.tick_params(length=0,pad=6)
   cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.065,pad=.06,aspect=35)
   cb.set_ticks([-25,0,25]);cb.set_label('Class-score change from baseline (pp)')
   save(fig,f'mn_categories_{line}_{half+1}','Category gains and losses coexist; all 40 categories retained',str(M/'classification_per_class_delta.csv'))
 winners=read(M/'geometry_objectwise_winner_share.csv')
 fig,axs=rows2(5.4,.15)
 for ax,metric in zip(axs,['CD_L2','HD']):
  for j,line in enumerate('AB'):
   vals=[float(next(r['winner_share_percent'] for r in winners if r['line']==line and r['metric']==metric and r['method']==m)) for m in METHODS]
   left=0
   for m,val,c in zip(METHODS,vals,COLORS):
    ax.barh(j,val,left=left,height=.45,color=c,label=m if j==0 else None)
    if val>15:ax.text(left+val/2,j,f'{val:.1f}%',ha='center',va='center',color='white')
    left+=val
  ax.set(yticks=[0,1],yticklabels=['Line A','Line B'],xlim=(0,100),xticks=[0,25,50,75,100])
  ax.set_title('Lowest CD per object' if metric=='CD_L2' else 'Lowest HD per object',loc='left')
  ax.invert_yaxis()
 axs[1].set_xlabel('Share of 2,468 test objects (%)')
 fig.subplots_adjust(top=.78)
 fig.legend(*axs[0].get_legend_handles_labels(),ncol=3,loc='upper center',bbox_to_anchor=(.5,1),handlelength=1.1,columnspacing=.7)
 save(fig,'mn_object_winners','Geometry rankings persist at the object level but depend on the metric',str(M/'geometry_objectwise_winner_share.csv'))
 rows=read(M/'geometry_equal_n_per_sample.csv')
 fig,axs=plt.subplots(2,2,figsize=(W,5.5));fig.subplots_adjust(left=.12,right=.97,top=.90,bottom=.12,wspace=.39,hspace=.62)
 stat=[]
 for i,line in enumerate('AB'):
  pp={r['shape_id']:r for r in rows if r['line']==line and r['display']=='PU-Net'}
  pg={r['shape_id']:r for r in rows if r['line']==line and r['display']=='PU-GCN'}
  ids=sorted(pp);assert set(ids)==set(pg) and len(ids)==2468
  for j,col in enumerate(['cd','hd']):
   values=np.array([float(pg[k][col])-float(pp[k][col]) for k in ids]);x=np.sort(values)
   axs[i,j].step(x,np.arange(1,len(x)+1)/len(x),where='post',color=BLUE,lw=1.4)
   axs[i,j].axvline(0,color=GRAY,ls='--',lw=.8);axs[i,j].set(ylim=(0,1),yticks=[0,.5,1],xlabel='PU-GCN − PU-Net')
   axs[i,j].set_title(f'Line {line} / {col.upper()}',loc='left')
   clean(axs[i,j], 'y')
   if j==0:axs[i,j].set_ylabel('Object fraction')
   stat.append(dict(line=line,metric=col,n=len(values),median=float(np.median(values)),gcn_lower=int((values<0).sum()),net_lower=int((values>0).sum())))
 STATS['paired_geometry']=stat
 save(fig,'mn_paired_geometry','Paired CD and HD differences reveal the metric-dependent trade-off',str(M/'geometry_equal_n_per_sample.csv'))
 corr=read(M/'category_geometry_classification_alignment.csv')
 fig,axs=rows2(6.1,.30)
 for ax,line in zip(axs,'AB'):
  for j,(metric,marker,col) in enumerate([('cd','o',BLUE),('hd','s',RED),('nuc_abs_dev','^',GRAY)]):
   rr=[next(r for r in corr if r['line']==line and r['method']==m and r['geometry_quality']==metric) for m in METHODS]
   x=np.array([float(r['spearman_rho']) for r in rr]);lo=np.array([float(r['bootstrap_ci_low']) for r in rr]);hi=np.array([float(r['bootstrap_ci_high']) for r in rr])
   ax.errorbar(x,np.arange(5)+(j-1)*.22,xerr=[x-lo,hi-x],fmt=marker,ms=4,capsize=1.5,lw=.7,color=col,label=['CD','HD','NUC'][j])
  ax.set(yticks=range(5),yticklabels=METHODS,xlim=(-.65,.65),xticks=[-.5,0,.5])
  ax.axvline(0,color='#92999d',ls='--',lw=.8);ax.invert_yaxis();ax.set_title('Line '+line,loc='left')
 axs[-1].set_xlabel('Category Spearman correlation')
 fig.subplots_adjust(top=.84)
 fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.58,1),ncol=3,handlelength=1)
 save(fig,'mn_category_association','All 30 archived associations show the uncertainty of category-level alignment',str(M/'category_geometry_classification_alignment.csv'))
 curves=read(M/'training_curves.csv')
 fig,axs=rows2(5.8,.15)
 for ax,line in zip(axs,'AB'):
  for i,m in enumerate(['Original' if line=='A' else 'Sparse']+METHODS):
   rr=[r for r in curves if r['line']==line and r['method']==m and int(r['epoch'])>=100]
   assert len(rr)==101
   ax.plot([int(r['epoch']) for r in rr],[float(r['test_oa_percent']) for r in rr],color=INK if i==0 else COLORS[i-1],lw=1.05,label='Baseline' if i==0 else m)
  ax.set(xlim=(100,200),ylim=(87,92.3),ylabel='Test OA (%)');ax.set_title('Line '+line,loc='left');clean(ax,'y')
 axs[-1].set_xlabel('Epoch');fig.subplots_adjust(top=.78)
 fig.legend(*axs[0].get_legend_handles_labels(),ncol=3,loc='upper center',bbox_to_anchor=(.5,1),handlelength=1,columnspacing=.7)
 save(fig,'mn_late_training','The later training record separates checkpoint variation from method ordering',str(M/'training_curves.csv'))

def project(points,az=35,el=17):
 a=np.deg2rad(az);b=np.deg2rad(el);x,y,z=points[:,:3].T
 return np.c_[np.cos(a)*x+np.sin(a)*z,-np.sin(b)*(-np.sin(a)*x+np.cos(a)*z)+np.cos(b)*y]
def nearest(points,ref):
 out=[]
 for start in range(0,len(points),128):
  dd=((points[start:start+128,None,:3].astype(float)-ref[None,:,:3])**2).sum(2)
  out.extend(np.sqrt(dd.min(1)))
 return np.array(out)
def modelnet_clouds():
 from make_teacher_revision_figures import main
 main()

def remake_controls():
 data=read(M/'training_stability_summary.csv')
 labels=['Mesh-ref 256','Sparse','Original','Mesh-ref 4096']
 rows=[next(r for r in data if r['method']==m) for m in labels]
 fig,ax=plt.subplots(figsize=(W,3.2));fig.subplots_adjust(left=.34,right=.95,bottom=.24,top=.75)
 yy=np.arange(4);best=[float(r['best_oa_percent']) for r in rows];last=[float(r['final_oa_percent']) for r in rows]
 ax.hlines(yy,last,best,color=GRAY,lw=1)
 ax.plot(best,yy,'s',color=INK,label='Best')
 ax.plot(last,yy,'o',mfc='white',mec=BLUE,label='Final')
 ax.set(yticks=yy,yticklabels=['Mesh-ref 256','Sparse 256','Original 1,024','Mesh-ref 4,096'],xlim=(90,92.3),xlabel='Overall accuracy (%)')
 ax.invert_yaxis();clean(ax)
 fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.64,.99),ncol=2)
 save(fig,'mn_controls','Genuine-density controls remain qualified by four reused training shapes',str(M/'training_stability_summary.csv'))

def kitti_stats():
 rows=read(D/'analysis/transition_summary_full_val.csv')
 for detector,title in [('pointrcnn','PointRCNN'),('centerpoint','CenterPoint')]:
  fig,axs=rows2(5.5,.23)
  for ax,line in zip(axs,'AB'):
   rr=[next(r for r in rows if r['detector']==detector and r['line']==line and r['class']=='Car' and r['method']==m) for m in LAB_METHODS]
   yy=np.arange(4)
   ax.barh(yy-.17,[int(r['lost_after_upsampling']) for r in rr],height=.30,color=RED,label='Lost')
   ax.barh(yy+.17,[int(r['recovered_after_upsampling']) for r in rr],height=.30,color=GREEN,label='Recovered')
   ax.set(yticks=yy,yticklabels=[short(m) for m in LAB_METHODS]);ax.invert_yaxis()
   ax.set_title(title+' / Line '+line,loc='left');clean(ax);ax.set_xlim(0,10000)
  axs[-1].set_xlabel('Matched Car objects across validation frames')
  fig.subplots_adjust(top=.84,right=.93)
  fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.58,1),ncol=2)
  save(fig,'k_transitions_'+detector,'Losses outnumber recoveries in the retained historical matching audit',str(D/'analysis/transition_summary_full_val.csv'))
 dist=read(D/'analysis/transition_summary_by_distance.csv')
 for detector,title in [('pointrcnn','PointRCNN'),('centerpoint','CenterPoint')]:
  fig,axs=rows2(5.5,.16)
  for ax,line in zip(axs,'AB'):
   for j,m in enumerate(LAB_METHODS):
    rr=[next(r for r in dist if r['detector']==detector and r['line']==line and r['class']=='Car' and r['method']==m and r['distance_bin']==bb) for bb in ['0-20m','20-40m','40m+']]
    ax.plot(range(3),[100*float(r['lost_rate_of_baseline_tp']) for r in rr],marker=['o','s','^','D'][j],ms=4,lw=1,color=[BLUE,INK,'#688960',RED][j],label=short(m))
   ax.set(xticks=range(3),xticklabels=['0–20','20–40','≥40'],ylim=(0,100),yticks=[0,25,50,75,100],ylabel='Baseline TP lost (%)')
   ax.set_title(title+' / Line '+line,loc='left');clean(ax,'y')
  axs[-1].set_xlabel('Planar range (m)')
  fig.subplots_adjust(top=.79)
  fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.55,1),ncol=2)
  save(fig,'k_distance_'+detector,'Distance stratification localises the historical loss of matched Cars',str(D/'analysis/transition_summary_by_distance.csv'))

def get_manifest(frame,line,method,detector='centerpoint'):
 return json.loads((D/f'frames/{frame}/line_{line.lower()}/{SLUG[method]}/{detector}/case_manifest.json').read_text(encoding='utf-8'))
def cloud(raw):
 rel=raw.replace('\\','/').split('/PointRCNN/')[1]
 p=V/'project_data'/rel
 a=np.fromfile(p,dtype=np.float32).reshape(-1,4)
 assert np.isfinite(a).all()
 return a
def split_cloud(arr,obs):
 if len(arr)==len(obs) and np.array_equal(arr,obs):return arr,np.empty((0,4),np.float32)
 if np.array_equal(arr[:len(obs)],obs):return arr[:len(obs)],arr[len(obs):]
 dt=np.dtype((np.void,arr.dtype.itemsize*4))
 mask=np.isin(arr.copy().view(dt).ravel(),obs.copy().view(dt).ravel())
 return arr[mask],arr[~mask]
def metriccrop(a,x0,x1,y0,y1,z0=-np.inf,z1=np.inf):
 return a[(a[:,0]>=x0)&(a[:,0]<=x1)&(a[:,1]>=y0)&(a[:,1]<=y1)&(a[:,2]>=z0)&(a[:,2]<=z1)]
def kitti_scene():
 frame='000590'
 for line in 'AB':
  md=get_manifest(frame,line,'PDANS');obs=cloud(md['paths']['observed'])
  states=[('Baseline',obs)]+[(short(m),cloud(get_manifest(frame,line,m)['paths']['upsampled_e1'])) for m in LAB_METHODS]
  fig,axs=plt.subplots(3,2,figsize=(W,6.1));fig.subplots_adjust(left=.15,right=.97,top=.94,bottom=.18,wspace=.25,hspace=.44)
  stats=[]
  for j,(ax,(name,arr)) in enumerate(zip(axs.flat,states)):
   measured,generated=split_cloud(arr,obs)
   measured=metriccrop(measured,0,60,-18,18);generated=metriccrop(generated,0,60,-18,18)
   ax.scatter(generated[:,0],generated[:,1],s=.15,c=BLUE,lw=0,rasterized=True)
   ax.scatter(measured[:,0],measured[:,1],s=.18,c='#777777',lw=0,rasterized=True)
   ax.set(xlim=(0,60),ylim=(-18,18),xticks=[0,30,60],yticks=[-15,0,15])
   ax.set_aspect('equal',adjustable='datalim');ax.set_title(f'({chr(97+j)}) {name}',loc='left',pad=6)
   if j%2==0:ax.set_ylabel('Lateral (m)')
   stats.append(dict(method=name,total=len(arr),visible_observed=len(measured),visible_generated=len(generated)))
  ax=axs.flat[-1];ax.axis('off')
  ax.set_title('(f) Point counts',loc='left',pad=6)
  for j,ss in enumerate(stats):ax.text(0,.88-j*.20,f'{ss["method"]}: {ss["visible_observed"]+ss["visible_generated"]:,}')
  fig.text(.53,.075,'Forward (m)',ha='center',va='center')
  fig.legend(handles=[Line2D([],[],marker='o',ls='',color='#777777',label='Observed',ms=4),Line2D([],[],marker='o',ls='',color=BLUE,label='Generated',ms=4)],loc='lower center',bbox_to_anchor=(.5,.005),ncol=2)
  save(fig,'k_scene_'+line,'Full historical multi-method clouds reveal added density at a common metric scale',f'lab frame {frame}; crop x=[0,60], y=[-18,18], all heights; no random display sampling')
  STATS['k_scene_'+line]=stats

def localproj(a,az=-35,el=22):
 a=np.asarray(a);az=np.deg2rad(az);el=np.deg2rad(el)
 return np.c_[np.cos(az)*a[:,0]-np.sin(az)*a[:,1],np.sin(el)*(np.sin(az)*a[:,0]+np.cos(az)*a[:,1])+np.cos(el)*a[:,2]]
def local_basis(gt):
 co=np.asarray(gt['corners_lidar']);center=co.mean(0)
 e0=co[1]-co[0];e1=co[3]-co[0];eh=co[4]-co[0]
 if np.linalg.norm(e0)<np.linalg.norm(e1):e0,e1=e1,e0
 basis=np.array([e0/np.linalg.norm(e0),e1/np.linalg.norm(e1),eh/np.linalg.norm(eh)])
 if basis[2,2]<0:basis[2]*=-1
 box=(co-center)@basis.T
 return center,basis,box
def crop_to_local(a,c,b,limits):
 loc=(a[:,:3]-c)@b.T
 return loc[np.all(np.abs(loc)<=limits,axis=1)]
EDGES=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
def boxlines(ax,co,color,ls='--',view='oblique'):
 xy=localproj(co) if view=='oblique' else co[:,:2]
 for i,j in EDGES if view=='oblique' else [(0,1),(1,2),(2,3),(3,0)]:
  ax.plot(xy[[i,j],0],xy[[i,j],1],color=color,ls=ls,lw=.85)
def kitti_local():
 frame='005625';gi=10
 for line in 'AB':
  md=get_manifest(frame,line,'PDANS');obs=cloud(md['paths']['observed'])
  gt=md['baseline_audit']['gt'][gi];center,basis,box=local_basis(gt)
  limits=np.max(np.abs(box),axis=0)+[.6,.6,.4]
  states=[('Baseline',obs)]+[(short(m),cloud(get_manifest(frame,line,m)['paths']['upsampled_e1'])) for m in LAB_METHODS]
  fig,axs=plt.subplots(3,2,figsize=(W,5.9));fig.subplots_adjust(left=.035,right=.965,top=.94,bottom=.09,wspace=.18,hspace=.32)
  stats=[]
  for j,(ax,(name,arr)) in enumerate(zip(axs.flat,states)):
   measured,generated=split_cloud(arr,obs)
   mm=crop_to_local(measured,center,basis,limits);gg=crop_to_local(generated,center,basis,limits)
   pm=localproj(mm);pg=localproj(gg)
   ax.scatter(pg[:,0],pg[:,1],s=2.0,c=BLUE,lw=0,rasterized=True)
   ax.scatter(pm[:,0],pm[:,1],s=2.2,c=INK,lw=0,rasterized=True)
   boxlines(ax,box,GRAY)
   ax.set(xlim=(-3.5,3.5),ylim=(-1.8,1.8));ax.set_aspect('equal',adjustable='datalim')
   ax.axis('off');ax.set_title(f'({chr(97+j)}) {name}',loc='left',pad=5)
   stats.append(dict(method=name,observed=len(mm),generated=len(gg)))
  ax=axs.flat[-1];ax.axis('off');ax.text(0,.91,'Observed + generated',va='top')
  for j,ss in enumerate(stats):ax.text(0,.70-.155*j,f'{ss["method"]}: {ss["observed"]} + {ss["generated"]}')
  fig.legend(handles=[Line2D([],[],marker='o',ls='',color=INK,label='Observed',ms=4),Line2D([],[],marker='o',ls='',color=BLUE,label='Generated',ms=4),Line2D([],[],color=GRAY,ls='--',label='GT box')],loc='lower center',bbox_to_anchor=(.5,0),ncol=3,handlelength=1)
  save(fig,'k_local_'+line,'Common object crops distinguish retained measurements from generated neighbourhoods',f'lab frame {frame}, audit GT index {gi}, Car; local box + [0.6,0.6,0.4] m padding; all points')
  STATS['k_local_'+line]=stats

def kitti_cases():
 import source_lost_car_gallery as old
 specs=[
 ('pr_lost','000590','A','PDANS','pointrcnn',2,'lost_after_upsampling'),
 ('pr_recovered','005625','B','PDANS','pointrcnn',7,'recovered_after_upsampling'),
 ('cp_lost','005625','B','PU-GCN','centerpoint',10,'lost_after_upsampling'),
 ('cp_recovered','005625','B','PU-GCN','centerpoint',4,'recovered_after_upsampling')]
 for name,frame,line,method,detector,gi,event in specs:
  md=get_manifest(frame,line,method,detector)
  tr=next(x for x in md['transition_rows'] if int(x['gt_index'])==gi)
  assert tr['transition']==event,(name,tr)
  gt=md['baseline_audit']['gt'][gi];center,basis,box=local_basis(gt)
  limits=np.max(np.abs(box),0)+[.9,.8,.5]
  _,baseline,up=old.effective_inputs(V,md)
  assert len(baseline)==md['baseline_effective_meta']['retained_points']
  assert len(up)==md['upsampled_effective_meta']['retained_points']
  pts=[crop_to_local(x,center,basis,limits) for x in [baseline,up]]
  fig,axs=plt.subplots(2,2,figsize=(W,5.0));fig.subplots_adjust(left=.05,right=.96,top=.85,bottom=.10,wspace=.15,hspace=.34)
  matches=[]
  limits_by_view={}
  geometry=[box]+pts
  for state in ['baseline','upsampled']:
   match=md[state+'_audit']['match_by_gt'].get(str(gi))
   if match is not None:
    geometry.append((np.array(md[state+'_audit']['pred'][int(match['pred_idx'])]['corners_lidar'])-center)@basis.T)
  for view in ['oblique','bev']:
   xy=np.vstack([localproj(x) if view=='oblique' else x[:,:2] for x in geometry])
   low=xy.min(0);high=xy.max(0);mid=(low+high)/2
   span=np.maximum(high-low,.5)*1.12
   pos=axs[0,0].get_position();ratio=(pos.width*W)/(pos.height*5.0)
   if span[0]/span[1]>ratio:span[1]=span[0]/ratio
   else:span[0]=span[1]*ratio
   limits_by_view[view]=(mid-span/2,mid+span/2)
  for j,(state,pp) in enumerate(zip(['baseline','upsampled'],pts)):
   audit=md[state+'_audit'];mt=audit['match_by_gt'].get(str(gi));matches.append(mt)
   pred=None if mt is None else (np.array(audit['pred'][int(mt['pred_idx'])]['corners_lidar'])-center)@basis.T
   for i,view in enumerate(['oblique','bev']):
    ax=axs[i,j];xy=localproj(pp) if i==0 else pp[:,:2]
    ax.scatter(xy[:,0],xy[:,1],s=3,c=INK if j==0 else BLUE,lw=0,rasterized=True)
    boxlines(ax,box,INK,'--',view)
    if pred is not None:boxlines(ax,pred,GREEN,'-',view)
    low,high=limits_by_view[view];ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]));ax.set_aspect('equal',adjustable='box');ax.axis('off')
    if i==0:
     result='FN: no qualifying match' if mt is None else f'TP: 3D IoU {float(mt["iou3d"]):.3f}'
     ax.set_title(('Baseline' if j==0 else 'Upsampled')+'\n'+result,pad=7)
    else:ax.set_title('BEV; '+str(len(pp))+' input points',pad=7)
  fig.legend(handles=[Line2D([],[],color=INK,ls='--',label='GT box'),Line2D([],[],color=GREEN,label='Qualifying prediction')],loc='lower center',bbox_to_anchor=(.5,0),ncol=2,handlelength=1.4)
  save(fig,'k_case_'+name,'A retained qualifying match is lost or recovered for the same labelled object',f'lab case manifest {frame}/{line}/{method}/{detector}; event={event}; GT audit index={gi}')
  STATS['k_case_'+name]={'frame':frame,'line':line,'method':method,'detector':detector,'gt_index':gi,'class':gt['class'],'transition':event,'crop_points':[len(x) for x in pts],'baseline_match':matches[0],'upsampled_match':matches[1]}

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['all','stats','clouds','kitti','cases'],default='all');args=parser.parse_args()
 if args.part in ['all','stats']:modelnet_stats();remake_controls();kitti_stats()
 if args.part in ['all','clouds']:modelnet_clouds()
 if args.part in ['all','kitti']:kitti_scene();kitti_local()
 if args.part in ['all','cases']:kitti_cases()
 p=R/'figure/expansion_manifest.json'
 prev=json.loads(p.read_text(encoding='utf-8')) if p.exists() else []
 names={x['name'] for x in MAN};prev=[x for x in prev if x['name'] not in names]+MAN
 p.write_text(json.dumps(prev,indent=2),encoding='utf-8')
 p=R/'audit/results_expansion_20260913/derived_statistics.json'
 old=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {};old.update(STATS)
 p.write_text(json.dumps(old,indent=2),encoding='utf-8')
