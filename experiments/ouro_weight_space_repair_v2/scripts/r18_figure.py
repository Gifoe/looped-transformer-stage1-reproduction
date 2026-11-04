"""Descriptive geometry, all prefixes and question means; no fitted decoder."""
import csv
import numpy as np
from r18_common import ROOT,read,jsonlines


def main():
    assert read('results/r18_EXPERIMENT_VERIFIED.json')['status']=='passed_read_only_independent_replay'
    rows=list(jsonlines('results/r18_pilot.jsonl'));assert len(rows)==64;rank=[];data=[]
    names=['native','native_shuffle','uniform','uniform_shuffle']
    for row in rows:
        by={k:[] for k in names};ranks=[];item=dict(id=row['id'])
        for c in row['candidates']:
            m=c['metrics'];ranks.extend(x for x,valid in zip(m['Fisher_effective_rank'],m['Fisher_rank_defined']) if valid)
            for k in names:by[k].extend(x*x for x,valid in zip(m[k+'_alignment'],m[k+'_alignment_defined']) if valid)
        rank.extend(ranks);item['positions']=sum(c['tokens'] for c in row['candidates']);item['median_effective_rank']=float(np.median(ranks)) if ranks else None
        for k in names:item[k+'_R2_mean']=float(np.mean(by[k])) if by[k] else None
        data.append(item)
    assert len(rank)==1540 and all(d[k+'_R2_mean'] is not None and 0<d[k+'_R2_mean']<=1 for d in data for k in names)
    with (ROOT/'results/r18_GEOMETRY_PLOT_DATA.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.8),gridspec_kw={'width_ratios':[1,1.15]});ax=axes[0];upper=2**int(np.ceil(np.log2(max(rank))))
    ax.hist(rank,bins=np.geomspace(1,upper,25),color='#547A92',edgecolor='white',linewidth=.4);ax.set_xscale('log',base=2);ax.set_xlim(1,upper)
    ax.set_xticks([2**k for k in range(int(np.log2(upper))+1)]);ax.xaxis.set_major_formatter(ScalarFormatter());ax.axvline(np.median(rank),color='#A34747',linestyle='--',linewidth=1.3)
    ax.set_xlabel('Effective Fisher rank');ax.set_ylabel('Prefix positions (descriptive count)');ax.set_title('A  Native probability geometry',loc='left',weight='bold')
    ax.text(.98,.94,f'1,540 positions\nMedian = {np.median(rank):.2f}',transform=ax.transAxes,ha='right',va='top')
    ax=axes[1];rng=np.random.default_rng(2021);colors=['#225580','#989898','#387A6D','#BBA780']
    for j,k in enumerate(names):
        v=np.asarray([d[k+'_R2_mean'] for d in data]);ax.scatter(j+rng.uniform(-.16,.16,len(v)),v,color=colors[j],s=13,alpha=.55,linewidths=0)
        ax.plot([j-.24,j+.24],[v.mean(),v.mean()],color='black',lw=1.6)
    ax.set_yscale('log');minimum=min(d[k+'_R2_mean'] for d in data for k in names);ax.set_ylim(min(1e-7,minimum*.5),1)
    ax.set_xticks(range(4),['Fisher\noriginal','Fisher\nshuffled','Uniform\noriginal','Uniform\nshuffled']);ax.set_ylabel('Mean prefix cosine squared per question')
    ax.set_title('B  Original g/e alignment; 64 questions',loc='left',weight='bold');ax.text(.02,.03,'Dots: question means; bars: mean of questions\nFixed shuffle is not an isotropic null.',transform=ax.transAxes,va='bottom',fontsize=8)
    fig.tight_layout(pad=1.2,w_pad=2.1)
    for suffix in ['png','pdf']:fig.savefig(ROOT/f'results/r18_geometry_diagnostic.{suffix}',dpi=300,bbox_inches='tight')
    plt.close(fig);print('R18_GEOMETRY_PLOT_WRITTEN',len(data),len(rank))


if __name__=='__main__':main()
