"""Render frozen C1 route estimates; no fitting, testing, or data selection.

Publication style follows the scientific-figure-making skill: portable sans
font, minimal spines, consistent palette, editable SVG and PDF/300dpi PNG.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 analysis=a.root/'analysis';out=a.root/'figures';out.mkdir(exist_ok=True)
 assert read(analysis/'TRAIN_COMPLETE.json')['status']=='PASS' and read(analysis/'VALIDATION_COMPLETE.json')['status']=='PASS'
 sources=[analysis/'TRAIN_CONTRAST_TABLES.json',analysis/'VALIDATION_CONTRAST_TABLES.json']
 tables=sum([read(path) for path in sources],[])
 selected=[x for x in tables if x['cohort'] in ['train1030','validation299'] and x['group']=='rescue'
           and x['sensitivity']=='all' and x['metric']=='meanwrong']
 fonts={font.name for font in font_manager.fontManager.ttflist}
 font=next((name for name in ['Arial','Helvetica','DejaVu Sans'] if name in fonts),'sans-serif')
 plt.rcParams.update({'font.family':font,'font.size':13,'axes.linewidth':1.8,
   'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
   'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
 series=[('', 'Raw contrast','#0F4D92','o',-.17),
         ('_vs_norm','Minus norm-only control','#767676','s',0),
         ('_vs_unrelated','Minus unrelated control','#B64342','^',.17)]
 fig,axes=plt.subplots(2,2,figsize=(12.4,8.3),sharey=True)
 plotted=[]
 for row,(branch,title) in enumerate([('X','Residual bypass X'),('A','Attention increment A')]):
  ranges=[]
  for col,(cohort,label) in enumerate([('train1030','Remaining Train1030'),('validation299','Validation299')]):
   ax=axes[row,col];ax.axvline(0,color='#272727',linewidth=1,linestyle='--',zorder=0)
   populations=set()
   for y,direction in enumerate(['restore','loss']):
    for suffix,legend,color,marker,offset in series:
     name=branch+'_'+direction+suffix
     values=[x for x in selected if x['cohort']==cohort and x['contrast']==name]
     assert len(values)==1,(cohort,name);value=values[0];populations.add(value['n_population'])
     plotted.append(value)
     if value['mean'] is None:
      ax.text(0,y+offset,' undefined',fontsize=10,color=color,va='center');continue
     lo,hi=value['paired95'];mean=value['mean'];assert lo<=hi
     ax.hlines(y+offset,lo,hi,color=color,linewidth=1.7,zorder=2)
     ax.vlines([lo,hi],y+offset-.045,y+offset+.045,color=color,linewidth=1.2,zorder=2)
     ax.plot(mean,y+offset,marker=marker,color=color,markersize=6.5,linestyle='none',markeredgewidth=1.1,zorder=3)
     ranges.extend([lo,hi,mean,0])
   assert len(populations)==1
   ax.set_title(f'{title}\n{label}; natural corrections n={next(iter(populations))}',fontsize=13,pad=12)
   ax.set_yticks([0,1],['Restoration in U','Reciprocal loss in N'])
   ax.set_ylim(1.48,-.48)
   ax.tick_params(axis='both',length=4,width=1.2)
   ax.set_xlabel('Gold–mean-wrong log-probability gap change\n(nats)',labelpad=8)
   if col==1:ax.tick_params(labelleft=False)
  minimum=min(ranges) if ranges else -1;maximum=max(ranges) if ranges else 1
  padding=max((maximum-minimum)*.08,.01)
  for ax in axes[row]:ax.set_xlim(minimum-padding,maximum+padding)
 handles=[Line2D([0],[0],color=color,marker=marker,linestyle='none',markersize=7,label=label)
          for suffix,label,color,marker,offset in series]
 fig.legend(handles=handles,loc='lower center',ncol=3,bbox_to_anchor=(.5,.015),fontsize=11)
 fig.tight_layout(pad=1.8,rect=[0,.08,1,1])
 fig.canvas.draw()
 canvas=fig.bbox;clipped=[]
 for ax in axes.flat:
  for artist in [ax.title,ax.xaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]:
   if not artist.get_visible() or not artist.get_text():continue
   b=artist.get_window_extent(fig.canvas.get_renderer())
   if b.x0<canvas.x0-1 or b.y0<canvas.y0-1 or b.x1>canvas.x1+1 or b.y1>canvas.y1+1:clipped.append(artist.get_text())
 assert not clipped,('Clipped figure labels',clipped)
 paths=[]
 for suffix in ['png','pdf','svg']:
  path=out/('C1_route_control_contrasts.'+suffix);fig.savefig(path,dpi=300,bbox_inches='tight',pad_inches=.12);paths.append(path)
 plt.close(fig)
 caption=('C1 conditional route tests on fresh native K1-wrong/K2-correct questions. Columns separate the previously exposed remaining Train1030 and Validation299 cohorts; their samples are never pooled. Rows separate the residual bypass X and normalized attention increment A. Markers show frozen raw restoration/reciprocal-loss contrasts and those contrasts minus the corresponding norm-only or unrelated control. Whiskers are ordinary question-paired95% bootstrap intervals,10000 resamples. The strong-route decision instead uses the separately reported one-sided12-test family bounds in both cohorts and no natural-no-tie sign reversal; no significance stars are inferred from these whiskers. The x scale is shared across cohorts within each branch and differs between X and A. Undefined controls retain explicit coverage in the source tables. U is the Q9-conditioned L10–L16 reference-update removal; it is not a natural K2−K1 difference. Joint restoration is a structural identity and is not plotted as a mechanistic discovery.\n')
 (out/'C1_route_control_contrasts.caption.md').write_text(caption,encoding='utf-8')
 (out/'C1_route_control_contrasts.values.json').write_text(json.dumps(plotted,indent=2)+'\n')
 (out/'C1_FIGURE_QA.json').write_text(json.dumps({'status':'RENDERED_LABEL_EXTENTS_PASS_PENDING_VISUAL_INSPECTION',
    'source_sha256':sha(__file__),'data_sources':{path.name:sha(path) for path in sources},
    'font':font,'whiskers':'Ordinary paired95%, not family-adjusted support gate',
    'files':{path.name:sha(path) for path in paths},'clipped_labels':clipped},indent=2)+'\n')
 print('C1_ROUTE_FIGURE_RENDERED',str(paths[0]),flush=True)
if __name__=='__main__':main()
