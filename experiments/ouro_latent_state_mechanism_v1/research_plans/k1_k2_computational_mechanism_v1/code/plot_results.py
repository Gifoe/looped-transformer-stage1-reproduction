import pathlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=pathlib.Path('/root/ouro-computation-sol-20261116');O=R/'figures';O.mkdir(exist_ok=True);a=json.loads((R/'analysis/A_SUMMARY.json').read_text());b=json.loads((R/'analysis/B_SUMMARY.json').read_text());c=json.loads((R/'analysis/C_SUMMARY.json').read_text())
plt.rcParams.update({'font.family':['DejaVu Sans','sans-serif'],'font.size':12,'axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':1.6,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,axes=plt.subplots(1,3,figsize=(18,5.6),gridspec_kw={'width_ratios':[1,1.25,1]})
ax=axes[0];names=['K1','native','freeze_all','freeze_Q','freeze_P'];labels=['K1','K2','All B','Q B','P B'];colors=['#CFCECE','#0F4D92','#B64342','#E9A6A1','#F6CFCB']
for k,cohort in enumerate(['train1030','validation299']):
 for j,(name,color) in enumerate(zip(names,colors)):
  v=a[cohort]['groups']['all']['conditions'][name]['accuracy'];x=k*6+j;mean=100*v['mean'];lo,hi=[100*t for t in v['interval']];ax.bar(x,mean,width=.85,color=color,edgecolor='#272727',linewidth=.7,label=labels[j] if k==0 else None);ax.errorbar(x,mean,yerr=[[mean-lo],[hi-mean]],fmt='none',color='#272727',capsize=2,lw=.8)
ax.set_xticks([2,8],['Train1030','Validation299']);ax.set_ylim(0,103);ax.set_ylabel('Correct (%)');ax.set_title('a  Context reuse on ARC',loc='left',fontweight='bold');ax.legend(ncol=3,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.12));ax.text(.02,.97,'R executes real K2 in all conditions',transform=ax.transAxes,va='top',fontsize=10)
ax=axes[1];names=['text_edit_K2','source_boundary','unrelated','dose','readout_copy','descendant_edit'];labels=['Text edit','Source','Unrelated','Dose','R copy','Descendants'];colors=['#8BCF8B','#0F4D92','#CFCECE','#CFCECE','#CFCECE','#42949E']
for j,(name,color) in enumerate(zip(names,colors)):
 v=b['heldout']['conditions'][name];m=100*v['cross_accuracy']['mean'];lo,hi=[100*t for t in v['cross_accuracy']['interval']];ax.bar(j,m,color=color,edgecolor='#272727',lw=.7);ax.errorbar(j,m,yerr=[[m-lo],[hi-m]],fmt='none',color='#272727',capsize=2,lw=.8);ax.bar(j,100*v['novel_cross_neither_native_output']/512,color='none',edgecolor='#272727',hatch='///',lw=.7);ax.text(j,m+4,f"{v['crossed']}/512",ha='center',fontsize=9)
ax.set_ylim(0,108);ax.set_xticks(range(6),labels,rotation=32,ha='right');ax.set_ylabel('Predicted crossed answer (%)');ax.set_title('b  Counterfactual recomputation',loc='left',fontweight='bold');ax.text(.02,.98,'Hatched: differs from both native outputs',transform=ax.transAxes,va='top',fontsize=9)
ax=axes[2];names=['restore_attention','attention_norm_control','attention_dose_control','restore_early_R','restore_L18_R'];labels=['L10–16 attention','Norm preserved','Dose permuted','Early L2 R','Late L18 R'];colors=['#0F4D92','#3775BA','#767676','#767676','#42949E']
for j,(name,color) in enumerate(zip(names,colors)):
 v=c['all']['conditions'][name]['source_minus_treatment_margin'];m=v['mean'];lo,hi=v['interval'];ax.errorbar(m,j,xerr=[[m-lo],[hi-m]],fmt='o',color=color,capsize=4,markersize=7,lw=2)
ax.axvline(0,color='#767676',ls='--',lw=1);ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlim(-.045,.49);ax.set_xlabel('Loss of crossed margin (logits)');ax.set_title('c  Limited causal routing',loc='left',fontweight='bold');ax.text(.02,-.22,'All512 units; C reuses exposed B data',transform=ax.transAxes,fontsize=10)
fig.tight_layout(pad=2,w_pad=3)
for ext in ['png','pdf','svg']:fig.savefig(O/f'COMPUTATIONAL_MECHANISM.{ext}',dpi=300,facecolor='white',bbox_inches='tight')
fig.savefig(O/'COMPUTATIONAL_MECHANISM_preview.png',dpi=90,facecolor='white',bbox_inches='tight')
(O/'CAPTION.md').write_text('A: same-device ARC accuracy; bars carry descriptive95% question-bootstrap intervals. All/Q/P B means the actual K1 normalized region is reset at each K2 layer input. B: crossed-answer rates with95% base-pair bootstrap intervals; hatched portion is a crossed output unlike both actual original and donor native predictions. Descendant replacement can contain precomputed answers. C: source-patch minus route-intervention crossed margin over all512 directed units;98.333% intervals for the3prespecified restoration sites and95% for compatibility/dose controls. C follows B and reuses exposed cases. No error bar is a global multiplicity-adjusted generalization statement.\n')
print('FIGURES_RENDERED')
