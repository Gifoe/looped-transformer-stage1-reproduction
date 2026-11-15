"""Publication plots from completed audited scalar data; no model execution."""
import hashlib,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
R=pathlib.Path(__file__).resolve().parents[1]
O=R/'publication/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_latent_mechanism_completion/figures'
assert (O.parent/'FINAL_RESEARCH_REPORT.md').exists()
O.mkdir(parents=True,exist_ok=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.3,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
read=lambda p:json.loads(pathlib.Path(p).read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
blue='#0F4D92';gray='#767676';red='#B64342'
source={};exports=[]
def save(fig,name):
 for extension in ['png','pdf','svg']:
  path=O/(name+'.'+extension);assert not path.exists();fig.savefig(path,dpi=300,bbox_inches='tight');exports.append(dict(file=path.name,sha256=sha(path)))
 fig.savefig(O/(name+'_preview.png'),dpi=82,bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11,7.2),sharey=True)
for i,stage in enumerate(['train1030','validation299']):
 path=R/'analysis_results'/stage/'E2_DESCRIPTIVE.json';source[str(path.relative_to(R))]=sha(path);s=read(path)['summary']['all']
 for j,site in enumerate(['l18_input','post_norm']):
  ax=axs[i,j];categories=[('raw','real'),('unit_l2','real'),('raw','permuted'),('unit_l2','permuted')]
  for x,(normalization,control) in enumerate(categories):
   head=f'{site}_{normalization}_trainK1_{control}';a=s['K1|'+head]['accuracy'];b=s['H11|'+head]['accuracy']
   ax.plot([x-.12,x+.12],[a,b],color='#B7B7B7',lw=1.5,zorder=1)
   ax.scatter(x-.12,a,color=gray,marker='s',s=35,zorder=2);ax.scatter(x+.12,b,color=blue,s=40,zorder=3)
  ax.set_xticks(range(4),['Real\nraw','Real\nunit-L2','Permuted\nraw','Permuted\nunit-L2']);ax.set_ylim(0,1);ax.set_yticks(np.linspace(0,1,6));ax.set_title(stage+' | '+('L18 R input' if site=='l18_input' else 'Final normalized R'))
  if j==0:ax.set_ylabel('Frozen-head accuracy')
fig.legend(handles=[Line2D([],[],marker='s',color=gray,ls='',label='Actual K1'),Line2D([],[],marker='o',color=blue,ls='',label='Actual K2')],loc='upper center',bbox_to_anchor=(.5,1.01),ncol=2)
fig.text(.5,.01,'Identical original TrainK1 heads; pooled Train scaler includes both loops. Decoding is not native causal-use proof.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.96),pad=2);save(fig,'A_frozen_cross_loop')
fig,axs=plt.subplots(3,2,figsize=(14,13),gridspec_kw={'height_ratios':[1,1.8,1]})
families=['attention','L18_branch','MLP_secondary'];ss={}
for stage in ['train1030','validation299']:
 path=R/'b2_analysis'/stage/'summary.json';source[str(path.relative_to(R))]=sha(path);ss[stage]=read(path)
for row,family in enumerate(families):
 allcells=[c for stage in ss for c in ss[stage]['rescue']['families'][family]['contrasts'].values() if c['mean'] is not None]
 lo=min([0]+[c['ci95'][0] for c in allcells]);hi=max([0]+[c['ci95'][1] for c in allcells]);pad=max(.08,(hi-lo)*.1)
 for col,stage in enumerate(['train1030','validation299']):
  ax=axs[row,col];cells=ss[stage]['rescue']['families'][family]['contrasts'];names=list(cells)
  for y,name in enumerate(names):
   c=cells[name]
   if c['mean'] is None:continue
   ax.plot(c['ci95'],[y,y],color=blue,lw=1.6)
   ax.scatter(c['mean'],y,s=43,edgecolor=blue,facecolor=blue if c['positive_pass'] else 'white',zorder=3)
  labels=[n.replace('_vs_',' vs ').replace('specific_','specific ').replace('_',' ') for n in names]
  ax.set_yticks(range(len(names)),labels);ax.invert_yaxis();ax.axvline(0,color='#4D4D4D',lw=.9,ls='--');ax.set_xlim(lo-pad,hi+pad)
  ax.set_title(stage+' | '+family.replace('_',' ')+' | natural rescue n='+str(ss[stage]['rescue']['n']));ax.set_xlabel('Gold-minus-mean-wrong gap contrast')
fig.text(.5,.01,'Bars: ordinary95% paired bootstrap CI. Filled: original declared positive contrast criterion. Families corrected separately; no global FWER claim.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.03,1,1),pad=2.3,w_pad=3);save(fig,'B2_controlled_route_contrasts')
fig,ax=plt.subplots(figsize=(12,3.8));names=['Not discretely blocked','Blocked; neither restores','Blocked; X only','Blocked; A only','Blocked; both restore'];colors=['#CFCECE','#B64342','#3775BA','#42949E','#AADCA9'];totals=[]
for y,stage in enumerate(['train1030','validation299']):
 path=R/'posthoc_review'/stage/'BINARY_PATHS.json';source[str(path.relative_to(R))]=sha(path);s=read(path)['rescue'];x=s['binary_paths']['N_correct__T_wrong__TX_correct'];a=s['binary_paths']['N_correct__T_wrong__TA_correct'];blocked=set(x['susceptible_ids']);xs=set(x['recovered_ids']);ats=set(a['recovered_ids']);values=[s['n']-len(blocked),len(blocked-xs-ats),len(xs-ats),len(ats-xs),len(xs&ats)];assert sum(values)==s['n'];totals.append(dict(stage=stage,n=s['n'],counts=dict(zip(names,values))))
 left=0
 for count,color in zip(values,colors):
  ax.barh(y,count,left=left,color=color,edgecolor='white',height=.55)
  if count>=8:ax.text(left+count/2,y,str(count),ha='center',va='center',color='white' if color in ('#B64342','#3775BA','#42949E') else '#272727',fontsize=10)
  left+=count
 ax.text(left+4,y,'n='+str(s['n']),va='center')
ax.set_yticks([0,1],['Train1030 native rescue','Validation299 native rescue']);ax.invert_yaxis();ax.set_xlabel('Questions (counts, not explained-reasoning percentages)');ax.set_xlim(0,max(x['n'] for x in totals)+35)
fig.legend(handles=[plt.Rectangle((0,0),1,1,color=c,label=n) for c,n in zip(colors,names)],loc='upper center',bbox_to_anchor=(.5,1.06),ncol=3,fontsize=9)
fig.text(.5,.01,'Mutually exclusive observed transition sets on the specified receivers. X/A overlap split explicitly. TXA structural identity excluded.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.04,1,.87),pad=1.8);save(fig,'C_natural_correction_sets')
(O/'FIGURE_SOURCE.json').write_text(json.dumps(dict(status='RENDERED_AWAITING_VISUAL_QA',source_sha256=source,exports=exports,correction_sets=totals,script_sha256=sha(__file__)),indent=2))
print('FIGURES_RENDERED',len(exports))
