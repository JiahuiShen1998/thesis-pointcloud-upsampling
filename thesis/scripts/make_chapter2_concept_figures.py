"""Python-only conceptual illustrations; no experimental observations are used.

Random cloud: one unordered coordinate set with all 192 generated rows shown.
Detector schematic: original vector drawing of published processing stages.
Final text is 12 TeX pt at the thesis text width; outputs are not cropped/scaled.
"""
from pathlib import Path
import sys, json
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'.python-deps'))
sys.path.insert(0,str(Path.home()/'.codex/skills/nature-figure/scripts'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from audit_panel_alignment import require_matplotlib_panel_alignment

O=R/'figure'; A=R/'audit/chapters_1_2_restoration_20260914/figure_qa'
A.mkdir(parents=True,exist_ok=True)
(O/'source_data').mkdir(exist_ok=True)
W=439.2395/72.27
FONT=12*72/72.27
BLUE='#35678a'; INK='#292929'; LIGHT='#eef3f6'; ORANGE='#a75b28'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':11.955168119551681,
 'axes.labelsize':FONT,'axes.titlesize':FONT,'xtick.labelsize':FONT,
 'ytick.labelsize':FONT,'legend.fontsize':FONT,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.bbox':None,
 'axes.spines.top':False,'axes.spines.right':False})
manifest=[]
def save(fig,name,contract):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=A/(name+'.alignment.json'),strict=True)
    fig.savefig(O/(name+'.pdf'))
    fig.savefig(O/(name+'.svg'))
    fig.savefig(O/(name+'.png'),dpi=300)
    manifest.append(dict(name=name,width_inches=W,font_pdf_pt=FONT,**contract))
    plt.close(fig)

# Seeded synthetic data requested solely for a point-cloud definition.
rng=np.random.default_rng(20260914)
points=rng.uniform(0,1,size=(192,3))
np.savetxt(O/'source_data/random_point_cloud.csv',points,delimiter=',',
           header='x,y,z',comments='',fmt='%.9f')
fig=plt.figure(figsize=(W,4.6))
ax=fig.add_axes([.08,.09,.82,.88],projection='3d')
idx=int(np.argmax(points[:,2]-.5*points[:,1]))
ax.scatter(*points.T,s=15,color=BLUE,alpha=.72,depthshade=False,linewidths=0)
ax.scatter(*points[idx],s=70,color=ORANGE,edgecolors='white',linewidths=.8,depthshade=False)

ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1.14),xlabel='x',ylabel='y',zlabel='z')
for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
    axis.set_ticks([0,.5,1]);axis.pane.fill=False
    axis._axinfo['grid'].update(color='#dedede',linewidth=.55)
ax.view_init(elev=23,azim=-58);ax.set_box_aspect((1,1,1))
ax.yaxis.set_ticks([.5,1])
from mpl_toolkits.mplot3d import proj3d
fig.canvas.draw()
px,py,_=proj3d.proj_transform(*points[idx],ax.get_proj())
position=fig.transFigure.inverted().transform(ax.transData.transform((px,py)))
ax.annotate('Point i',xy=position,xycoords=fig.transFigure,
            xytext=(.73,.94),textcoords=fig.transFigure,ha='left',va='center',
            arrowprops=dict(arrowstyle='-',color=ORANGE,lw=.8))
save(fig,'random_point_cloud',dict(archetype='schematic-led composite',
 claim='A point cloud is an unordered set of 3D coordinates, independent of an object category.',
 source='Explicitly synthetic uniform samples in the unit cube; seed 20260914.',
 observations=192,shown=192,excluded=0,highlighted_row=idx,
 statistics='Not applicable: conceptual example, no experimental inference.',
 alignment='Not applicable: single 3D axes.'))

fig,axs=plt.subplots(1,2,figsize=(W,4.6))
fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.965,wspace=.18)
stages=[['Input points','PointNet++\npointwise features','Foreground scores\nand point proposals','Region pooling and\ncanonical refinement','3D boxes and scores'],
        ['Input points','Voxel encoding','Sparse backbone\nand BEV features','Centre heatmap\nand box regression','3D boxes and scores']]
for ax,label,title,steps in zip(axs,['a','b'],['PointRCNN','CenterPoint'],stages):
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    ax.text(.01,.985,label,fontweight='bold',va='top')
    ax.text(.50,.985,title,ha='center',va='top')
    centers=[.845,.665,.485,.305,.125]
    for j,(y,t) in enumerate(zip(centers,steps)):
        ax.add_patch(FancyBboxPatch((.025,y-.06),.95,.12,
            boxstyle='round,pad=0.008,rounding_size=0.015',
            facecolor=LIGHT if 0<j<4 else 'white',edgecolor=BLUE,linewidth=.85))
        ax.text(.50,y,t,ha='center',va='center',linespacing=1.12)
        if j<4:ax.annotate('',xy=(.5,centers[j+1]+.074),xytext=(.5,y-.071),
            arrowprops=dict(arrowstyle='-|>',lw=.8,color=INK))
save(fig,'detector_principles',dict(archetype='schematic-led composite',
 claim='PointRCNN and voxel-based CenterPoint form object candidates from different intermediate representations.',
 source='Shi et al., CVPR 2019, section 3; Yin et al., CVPR 2021, section 3.',
 statistics='Not applicable: architecture schematic; no measured quantities.',
 fidelity='Simplified representations, no layer widths or experimental budgets.'))
(A/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Exported 2 concept figures with PDF, SVG, PNG and source-data/QA records.')
