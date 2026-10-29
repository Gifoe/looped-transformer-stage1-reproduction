"""Render only measured development results; no efficacy claim in the figure."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
r=json.loads((ROOT/'results/ANALYSIS.json').read_text())['full400']
fig,axes=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
keys=['probe','confidence','terminal_only','fixed_-1.0','fixed_-0.5','fixed_0.5']
labels=['Input-probe controller','Confidence controller','Terminal-only control','Fixed rollback','Fixed partial rollback','Fixed extrapolation']
for i,(key,label) in enumerate(zip(keys,labels)):
    x=r['summary_vs_adaptive'][key];lo,hi=x['ci95_pp'];color='#1768AC' if key=='probe' else '#747C86'
    axes[0].plot([lo,hi],[i,i],color=color,lw=2);axes[0].scatter([x['delta_pp']],[i],color=color,zorder=3)
axes[0].set_yticks(range(len(labels)),labels);axes[0].invert_yaxis();axes[0].axvline(0,color='black',ls='--',lw=.8)
axes[0].set_xlabel('Accuracy change vs Adaptive (pp), 95% CI')
axes[0].set_title('Exposed OpenBookQA (n=400)')
for i,(a,label) in enumerate([('-1.0','Rollback'),('-0.5','Partial rollback'),('0.5','Extrapolation')]):
    x=r['signals'][a]
    if x['auc'] is not None:
        lo,hi=x['auc_ci95'];axes[1].plot([lo,hi],[i,i],color='#1768AC',lw=2);axes[1].scatter([x['auc']],[i],color='#1768AC')
    axes[1].text(.01,i+.22,f"{x['beneficial']} beneficial / {x['harmful']} harmful",fontsize=8)
axes[1].set_yticks(range(3),['Rollback','Partial rollback','Extrapolation']);axes[1].invert_yaxis();axes[1].axvline(.5,color='black',ls='--',lw=.8)
axes[1].set_ylim(2.55,-.3);axes[1].set_xlim(0,1);axes[1].set_xlabel('AUC on changed-correctness events, 95% CI');axes[1].set_title('Probe value vs intervention benefit')
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('Development mechanism evidence — not independent confirmation',fontsize=12)
(ROOT/'figures').mkdir(exist_ok=True)
fig.savefig(ROOT/'figures/mechanism_development.png',dpi=180);fig.savefig(ROOT/'figures/mechanism_development.pdf')
