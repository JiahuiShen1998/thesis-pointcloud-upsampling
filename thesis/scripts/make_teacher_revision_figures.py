"""Teacher-requested two-case, seven-panel ModelNet evidence plates.
Contract: localise output-to-reference residuals for all five methods.
Python structural adaptation of the existing orthographic point-cloud renderer.
No new inference, no display subsampling, no per-object classification claims.
"""
import os
from pathlib import Path
import sys,json,csv,hashlib
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'.python-deps'))
sys.path.insert(0,str(Path(os.environ.get('FIGURE_HELPERS_DIR', Path(__file__).resolve().parent/'external_helpers'))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from audit_panel_alignment import require_matplotlib_panel_alignment
A=R/'audit/teacher_email_revision_20260917'
Q=A/'figure_qa';Q.mkdir(parents=True,exist_ok=True)
M=R/'evidence/modelnet40_hpc/data'
W=439.2395/72.27
FONT=12*72/72.27
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
'font.size':11.9551681196,
'axes.titlesize':FONT,'axes.labelsize':FONT,'xtick.labelsize':FONT,
'ytick.labelsize':FONT,'legend.fontsize':FONT,'pdf.fonttype':42,
'ps.fonttype':42,'svg.fonttype':'none','savefig.bbox':None})
def project(a):
 az=np.deg2rad(35);el=np.deg2rad(17);x,y,z=a[:,:3].T
 return np.c_[np.cos(az)*x+np.sin(az)*z,
 -np.sin(el)*(-np.sin(az)*x+np.cos(az)*z)+np.cos(el)*y]
def nearest(a,r):
 d=[]
 for i in range(0,len(a),128):
  sq=((a[i:i+128,None,:3].astype(float)-r[None,:,:3])**2).sum(2)
  d.extend(np.sqrt(sq.min(1)))
 return np.asarray(d)
def main():
 manifest=[]
 geom=list(csv.DictReader((M/'geometry_equal_n_per_sample.csv').open(encoding='utf-8-sig')))
 variants=[('input','Input'),('reference','Reference'),('ear','EAR'),
 ('pdans','PDANS'),('punet','PU-Net'),('pugcn','PU-GCN'),('puedgeformer','PU-EF')]
 for line,obj in [('A','airplane_0627'),('B','chair_0890')]:
  paths={title:M/f'pointcloud_parallel/line{line}_{obj}_{slug}.npy' for slug,title in variants}
  arrays={title:np.load(path) for title,path in paths.items()}
  nn=4096 if line=='A' else 1024
  assert len(arrays['Input'])==(1024 if line=='A' else 256)
  assert all(len(v)==nn for k,v in arrays.items() if k!='Input')
  assert all(np.isfinite(v).all() for v in arrays.values())
  xy={k:project(v) for k,v in arrays.items()}
  errors={k:nearest(v,arrays['Reference']) for k,v in arrays.items() if k not in ['Input','Reference']}
  maxerr=max(float(v.max()) for v in errors.values());norm=Normalize(0,maxerr)
  diagnostics=[]
  for k,v in arrays.items():
   if k not in errors:continue
   back=nearest(arrays['Reference'],v)
   cd=float(errors[k].mean()+back.mean());hd=float(max(errors[k].max(),back.max()))
   display='PU-EdgeFormer' if k=='PU-EF' else k
   rec=next(r for r in geom if r['line']==line and r['shape_id']==obj and r['display']==display)
   assert abs(cd-float(rec['cd']))<1e-6,(obj,k,cd,rec['cd'])
   assert abs(hd-float(rec['hd']))<1e-6
   diagnostics.append({'method':display,'cd':cd,'hd':hd,'colour_max':float(errors[k].max())})
  fig,grid=plt.subplots(4,2,figsize=(W,6.65))
  fig.subplots_adjust(left=.045,right=.955,top=.946,bottom=.03,wspace=.16,hspace=.39)
  axes=list(grid.flat[:7]);legend=grid.flat[7];legend.set_axis_off()
  both=np.concatenate(list(xy.values()));mid=(both.min(0)+both.max(0))/2;span=np.ptp(both,axis=0)*1.10
  pos=axes[0].get_position();ratio=pos.width*W/(pos.height*6.65)
  dx=max(span[0],span[1]*ratio);dy=dx/ratio
  for j,(ax,(_,name)) in enumerate(zip(axes,variants)):
   if j<2:
    ax.scatter(xy[name][:,0],xy[name][:,1],s=1.65,c='#343f4a',linewidths=0,rasterized=True)
   else:
    sc=ax.scatter(xy[name][:,0],xy[name][:,1],s=1.65,c=errors[name],cmap='viridis',norm=norm,linewidths=0,rasterized=True)
   ax.set(xlim=(mid[0]-dx/2,mid[0]+dx/2),ylim=(mid[1]-dy/2,mid[1]+dy/2))
   ax.set_aspect('equal',adjustable='box');ax.axis('off')
   ax.set_title(f'({chr(97+j)}) {name}',loc='left',pad=6)
  # Final eighth grid cell holds the common legend, not another data panel.
  pos=legend.get_position()
  legend.text(0,.92,'Distance to reference',transform=legend.transAxes,va='top')
  legend.text(0,.65,'Normalised coordinates',transform=legend.transAxes,va='top')
  cbax=fig.add_axes([pos.x0,pos.y0+.17*pos.height,pos.width,.14*pos.height])
  cb=fig.colorbar(sc,cax=cbax,orientation='horizontal')
  cb.set_ticks([0,maxerr]);cb.set_ticklabels(['0',f'{maxerr:.3f}'])
  cb.outline.set_linewidth(.5)
  fig.canvas.draw()
  name=f'mn_atlas_{line}_{obj.split("_")[0]}'
  require_matplotlib_panel_alignment(fig,json_out=Q/(name+'.alignment.json'),
     exclude_axes=[legend,cbax],strict=True)
  for suffix in ['.pdf','.svg','.png']:fig.savefig(R/'figure'/(name+suffix),dpi=600)
  plt.close(fig)
  manifest.append({'name':name,'shape_id':obj,'line':line,'all_points_shown':True,
   'points':{k:len(v) for k,v in arrays.items()},'source_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
   'reference_colour_range':[0,maxerr],'method_metrics_verified':diagnostics,'width_inches':W,'height_inches':6.65,'font_pdf_pt':FONT,
   'selection':'Two of eight previously retained examples, one per line; agreement vs trade-off of CD and HD; not classification successes or failures.'})
  print('EXPORTED',name,flush=True)
 (A/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
if __name__=='__main__':main()
