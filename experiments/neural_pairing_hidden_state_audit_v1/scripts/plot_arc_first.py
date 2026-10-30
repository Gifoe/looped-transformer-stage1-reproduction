"""Render completed ARC findings only, including paired changes and margins."""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from linux_runtime import ROOT


def main():
    out=ROOT/'fresh/arc_first'
    d=json.loads((out/'ANALYSIS.json').read_text())
    n=d['n']
    plt.rcParams.update({'font.family':['DejaVu Sans'],'font.size':11,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    for ax in axes:
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    for j,(cond,color) in enumerate([('cut','#767676'),('link','#0F4D92')]):
        rows=[next(r for r in d['baseline'] if r['condition']==cond and r['receiver']==x) for x in ['quill','ember','probability_mean']]
        x=np.arange(3)+(j-.5)*.32;y=[100*r['accuracy'] for r in rows]
        axes[0].bar(x,y,.32,color=color,label=cond.upper(),edgecolor='black',lw=.6)
        for a,b in zip(x,y):axes[0].text(a,b+1,f'{b:.1f}',ha='center',fontsize=9)
    axes[0].set(xticks=np.arange(3),xticklabels=['Quill','Ember','Prob. mean'],ylim=(0,100),ylabel='Accuracy (%)',title='A  Same released weights')
    axes[0].legend()
    for j,r in enumerate(d['paired']):
        axes[1].bar(j-.18,r['corrections'],.36,color='#42949E',label='Corrected' if j==0 else None)
        axes[1].bar(j+.18,r['degradations'],.36,color='#B64342',label='Damaged' if j==0 else None)
        axes[1].text(j,max(r['corrections'],r['degradations'])+2,f"Net {r['net']:+d}",ha='center')
    axes[1].set(xticks=[0,1],xticklabels=['Quill','Ember'],ylabel='Questions',title='B  Paired ON versus CUT changes',
        ylim=(0,1.28*max(max(r['corrections'],r['degradations']) for r in d['paired'])))
    axes[1].legend()
    with (out/'per_question.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    for side,color in [('quill','#0F4D92'),('ember','#42949E')]:
        values=np.array([float(r['delta_margin']) for r in rows if r['receiver']==side])
        axes[2].hist(values,bins=40,histtype='step',lw=1.8,color=color,label=side.title())
    axes[2].axvline(0,ls='--',lw=1,color='#767676')
    axes[2].set(xlabel='Change in correct-answer logit margin',ylabel='Questions',title='C  Margin movement')
    axes[2].legend()
    fig.suptitle(f'Fresh final nl33w on complete four-choice ARC-Challenge (n = {n}); not QA-trained nl23',fontsize=13)
    fig.tight_layout(pad=1.5,rect=[0,0,1,.94])
    for ext in ['png','pdf','svg']:fig.savefig(out/('arc_first_results.'+ext),dpi=300,facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
