"""Descriptive plot of four frozen signals; no selection or new forward pass."""
import csv
import numpy as np
from r15_common import ROOT,read,sha


def main():
    checked=read('results/r15_EXPERIMENT_VERIFIED.json');assert checked['status']=='verified_complete_observational_probe'
    result=read('results/r15_ASSOCIATION_COMPLETE.json');assert sha('results/r15_question_features.csv')==result['question_features_sha256']
    with (ROOT/'results/r15_question_features.csv').open(encoding='utf8',newline='') as f:rows=[r for r in csv.DictReader(f) if r['outcome'] in ['rescue','damage']]
    labels=np.asarray([r['outcome']=='rescue' for r in rows]);p=np.flatnonzero(labels);n=np.flatnonzero(~labels);assert (len(p),len(n))==(7,15)
    keys=['primary','static','energy','old_margin'];scores=np.asarray([[float(r[k]) for r in rows] for k in keys]);rng=np.random.default_rng(2021);samples=[]
    for _ in range(5000):
        pp=rng.choice(p,len(p),replace=True);nn=rng.choice(n,len(n),replace=True);a=scores[:,pp,None];b=scores[:,None,nn]
        samples.append(((a>b).astype(float)+.5*(a==b)).mean(axis=(1,2)))
    low,high=np.quantile(samples,[.025,.975],axis=0);assert np.array_equal([low[0],high[0]],result['primary_stratified_bootstrap_95_CI'])
    data=[dict(signal=k,AUC=result['metrics'][k]['AUC'],lower95=float(lo),upper95=float(hi),positive_questions=7,negative_questions=15) for k,lo,hi in zip(keys,low,high)]
    with (ROOT/'results/r15_AUC_plot_data.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(8.3,3.9));labels=['Dynamic context change','Static context contrast','Correction energy','Old R6 margin']
    for i,row in enumerate(data):
        value=row['AUC'];ax.errorbar(value,i,xerr=[[value-row['lower95']],[row['upper95']-value]],fmt='o',color='#0F4D92' if i==0 else '#767676',capsize=3,markersize=6)
    ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.axvline(.5,c='#999999',lw=1,ls='--');ax.set_xlim(0,1)
    ax.set_xlabel('AUC; stratified question-bootstrap 95% CI')
    ax.set_title('Exploratory development association: 7 rescues, 15 damages')
    fig.tight_layout(pad=1.4)
    for extension in ['png','pdf']:fig.savefig(ROOT/f'results/r15_context_AUC.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig);print('R15_AUC_PLOT_WRITTEN',len(data))


if __name__=='__main__':main()
