"""Five report figures, with discovery, validation and absent held-out separated."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quality_core import ROOT,put

BLUE='#0F4D92';GREEN='#42949E';RED='#B64342';NEUTRAL='#CFCECE';VIOLET='#9A4D8E'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':1.3,'legend.frameon':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})


def finish(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=240,bbox_inches='tight',facecolor='white')
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    f=pd.read_csv(ROOT/'data/discovery_features.csv');l=pd.read_csv(ROOT/'data/discovery_labels.csv')
    x=f.merge(l[['id','gain']],on='id',validate='one_to_one');s=pd.read_csv(ROOT/'results/policy_summary.csv')
    u=pd.read_csv(ROOT/'results/univariate_predictability.csv');p=pd.read_csv(ROOT/'results/predictor_predictability.csv')
    # Distributions are ECDFs: no arbitrary density bandwidth, all observations visible.
    signals=[('choice_kl_euler_original','Choice KL (Euler || Original)'),('choice_confidence_delta','Choice confidence delta'),
             ('choice_entropy_delta','Choice entropy delta'),('choice_margin_delta','Choice probability margin delta'),
             ('hidden_output_l2','Window relative L2'),('step_l2_slope','Six-step L2 slope')]
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for ax,(name,label) in zip(axes.flat,signals):
        for gain,color,cl in [(1,GREEN,'Beneficial (n=32)'),(-1,RED,'Harmful (n=15)')]:
            vals=np.sort(x.loc[x.gain==gain,name].dropna());ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',color=color,lw=2,label=cl)
        ax.set(xlabel=label,ylabel='Empirical cumulative fraction',ylim=(0,1.04))
        if name=='choice_kl_euler_original':
            ax.set_xscale('symlog',linthresh=1e-4);ax.set_xlim(left=0)
    axes[0,0].legend(fontsize=9);fig.suptitle('Discovery only: beneficial and harmful updates overlap',fontsize=16)
    finish(fig,'figure1_feature_distributions')
    # Predefined signals, rather than a top-AUC cherry-picked subset.
    selected=[('choice_confidence_delta','Confidence delta'),('choice_entropy_delta','Entropy delta'),('choice_margin_delta','Margin delta'),
              ('choice_kl_euler_original','Choice KL'),('choice_js','Choice JS'),('hidden_output_l2','Hidden L2'),
              ('hidden_output_cosine','Hidden cosine'),('step_l2_slope','Update trend'),('random_feature','Random feature')]
    values=[]
    for name,label in selected:
        r=u[u.feature==name].iloc[0];values.append((label,r,'Discovery'))
    for scope,label in [('nested_cv','Linear: nested CV'),('cross_dataset','Linear: leave-task-out')]:
        r=p[(p.scope==scope)&(p.subset=='all')&(p.policy=='linear')].iloc[0];values.append((label,r,'Validation'))
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained',sharey=True);y=np.arange(len(values))
    for ax,metric,baseline in zip(axes,['auroc','auprc'],[.5,32/47]):
        for j,(label,r,scope) in enumerate(values):
            ax.errorbar(r[metric],j,xerr=[[max(0,r[metric]-r[metric+'_low'])],[max(0,r[metric+'_high']-r[metric])]],
                        fmt='o',color=BLUE if scope=='Validation' else GREEN,capsize=3)
        ax.axvline(baseline,color='grey',ls='--')
        ax.set(xlim=(0,1),xlabel=metric.upper()+' (beneficial = positive)',title=f'Chance reference = {baseline:.3f}')
    axes[0].set_yticks(y,[v[0] for v in values]);axes[0].invert_yaxis()
    fig.suptitle('Predictability: fixed signal orientation, stratified bootstrap 95% intervals',fontsize=14)
    finish(fig,'figure2_direction_predictability')
    policies=['always_original','always_euler','confidence_gate','kl_gate','linear','oracle'];labels=['Original','Euler','Confidence','KL','Linear','Oracle*']
    fig,axes=plt.subplots(1,4,figsize=(17,5),layout='constrained',sharey=True)
    for ax,scope,title in zip(axes[:3],['discovery_fit','nested_cv','cross_dataset'],['Same-data fit (exploratory)','Nested CV (600 questions)','Leave-task-out (600 questions)']):
        sub=s[(s.scope==scope)&(s.subset=='all')].set_index('policy').loc[policies]
        ax.bar(np.arange(6),sub.accuracy_percent,color=[RED,NEUTRAL,GREEN,GREEN,BLUE,VIOLET],edgecolor='black',linewidth=.6)
        for i,r in enumerate(sub.itertuples()):ax.text(i,r.accuracy_percent+.4,f'{r.accuracy_percent:.2f}',ha='center',fontsize=9)
        ax.axhline(314/6,color='grey',ls='--',lw=1);ax.set_xticks(np.arange(6),labels,rotation=55,ha='right');ax.set_title(title,fontsize=11);ax.set_ylim(0,63)
    axes[0].set_ylabel('Accuracy (%)');axes[3].axis('off');axes[3].text(.5,.5,'Independent test\nNOT RUN\n\nDeferred by user\nNo held-out claim',ha='center',va='center',transform=axes[3].transAxes,fontsize=13,color=RED)
    fig.suptitle('Selective update accuracy — *Oracle uses gold labels',fontsize=16);finish(fig,'figure3_selective_accuracy')
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    for ax,scope,title in zip(axes,['discovery_fit','nested_cv','cross_dataset'],['Discovery fit','Nested CV','Leave-task-out']):
        sub=s[(s.scope==scope)&(s.subset=='all')].set_index('policy')
        for k,name,color,marker,offset in [('always_euler','Euler',NEUTRAL,'s',(4,-14)),('kl_gate','KL',GREEN,'^',(4,5)),
                                          ('linear','Linear',BLUE,'o',(-50,5)),('oracle','Oracle*',VIOLET,'*',(-60,-18))]:
            r=sub.loc[k];xx=r.beneficial_retained/r.beneficial_total*100;yy=r.harmful_avoided/r.harmful_total*100
            ax.scatter(xx,yy,c=color,marker=marker,s=85,edgecolors='black',linewidth=.6,label=f'{name}: {r.accuracy_percent:.2f}%')
        ax.set(xlim=(-5,110),ylim=(-12,110),xlabel='Beneficial corrections retained (%)',ylabel='Harmful degradations avoided (%)',title=title)
        ax.legend(loc='center left',fontsize=9)
    fig.suptitle('Rejecting an update helps only when avoided harm exceeds lost corrections',fontsize=14)
    finish(fig,'figure4_retention_avoidance')
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained',sharey=True)
    for ax,task,title in zip(axes,['arc_challenge','openbookqa','mmlu'],['ARC-Challenge (n=200)','OpenBookQA (n=200)','MMLU, 5 subjects (n=200)']):
        sub=s[(s.scope=='cross_dataset')&(s.subset==task)].set_index('policy').loc[policies]
        ax.bar(np.arange(6),sub.accuracy_percent,color=[RED,NEUTRAL,GREEN,GREEN,BLUE,VIOLET],edgecolor='black',linewidth=.6)
        for i,r in enumerate(sub.itertuples()):ax.text(i,r.accuracy_percent+.5,f'{r.accuracy_percent:.1f}',ha='center',fontsize=9)
        ax.set_xticks(np.arange(6),labels,rotation=50,ha='right');ax.set_title(title,fontsize=11);ax.set_ylim(0,70)
    axes[0].set_ylabel('Accuracy (%)');fig.suptitle('Cross-benchmark validation within discovery data — independent test pending',fontsize=14)
    finish(fig,'figure5_cross_benchmark_validation')
    put('figures/FIGURE_MANIFEST.json',dict(figures=5,formats=['png','pdf'],independent_results_shown=False,
          notes=['Figure2 CIs condition on fitted/OOF scores, not refitting uncertainty',
                 'Figure3 bar axis begins at zero; accuracy CIs are tabulated in policy_summary.csv',
                 'Oracle is diagnostic gold-label upper bound; all five figures use discovery samples']))


if __name__=='__main__':main()
