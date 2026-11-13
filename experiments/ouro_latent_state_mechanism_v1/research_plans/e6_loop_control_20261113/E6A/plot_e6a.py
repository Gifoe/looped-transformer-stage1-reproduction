"""Plot supplied frozen results, without selecting new policies."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=Path(__file__).resolve().parent
fig,axes=plt.subplots(1,2,figsize=(10.8,4.0),layout='constrained')
colors=['#2864A0','#C56C30']
methods=['weighted4','loopcd_adaptive_cap1','loopcd_fixed_0.5']
labels=['Weighted fusion\n(ZL)','Joint-selected Cal\nLoopCD','Fixed LoopCD\ncontrol']
for j,(folder,protocol) in enumerate([('joint5090','Joint option letters'),('fas_char5090','Full answer / characters')]):
 rows=json.loads((R/folder/'TEST_SUMMARY.json').read_text());d={r['method']:r for r in rows};x=np.arange(3)+(j-.5)*.25
 y=[d[m]['pp_vs_K4'] for m in methods];err=np.array([[d[m]['paired_bootstrap95_pp'][0],d[m]['paired_bootstrap95_pp'][1]] for m in methods]).T
 axes[0].errorbar(x,y,yerr=[np.array(y)-err[0],err[1]-np.array(y)],fmt='o',capsize=3,color=colors[j],label=protocol)
 early=[r for r in rows if r['method'].startswith('cal_early') or r['method'] in ['early_fixed','K4']]
 axes[1].scatter([r['mean_loops'] for r in early],[100*r['accuracy'] for r in early],s=24,color=colors[j],label=protocol)
 axes[1].scatter([4],[100*d['K4']['accuracy']],marker='*',s=95,color=colors[j],edgecolor='black',linewidth=.5)
axes[0].axhline(0,c='#666666',lw=.8);axes[0].set_xticks(range(3),labels,fontsize=9);axes[0].set_ylabel('Paired accuracy difference vs native K4 (pp)');axes[0].set_title('Frozen primaries and existing controls')
axes[1].set_xlabel('Replay mean loops per candidate');axes[1].set_ylabel('Test accuracy (%)');axes[1].set_title('Predeclared exit rules; stars: native K4')
for ax in axes:
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
out=R/'figures';out.mkdir(exist_ok=True)
for suffix in ['svg','pdf','png']:fig.savefig(out/('E6A_FROZEN_RESULTS.'+suffix),dpi=220)
(out/'README.md').write_text('Figures use only frozen TEST_SUMMARY.json files. Paired intervals:10000 bootstrap draws; no multiplicity correction. Cal identity selected on Joint Validation and transferred unchanged. Right panel is replay cost, not actual Full-Answer speed. All predeclared exit rules are shown; points are not a Test-selected policy.\n')
