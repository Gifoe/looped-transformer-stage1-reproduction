"""Plot only complete, independently audited cross-loop experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cohort-title',required=True)
    a=p.parse_args()
    audit=json.loads(a.audit.read_text());geo=json.loads(a.geometry.read_text())
    assert audit['status']=='PASS_COMPLETE' and not audit['failures']
    assert geo['status']=='DESCRIPTIVE_TENSORS_AUDITED' and not geo['failures']
    n=audit['n_observed'];assert n==audit['n_expected']==geo['n']
    pairs=['1-2','2-3','3-4'];x=np.arange(3)
    states={'x_R':'L18 residual input','O':'SDPA output','a_R':'Normalized attention increment','final_norm_R':'Loop-end normalized state'}
    colors=['#0F4D92','#42949E','#9A4D8E','#B64342']
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.linewidth':1.6,'legend.frameon':False,
                         'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12.8,8.6))
    fig.suptitle('Ouro-1.4B | '+a.cohort_title,fontsize=19,y=.985)
    ax=axes[0,0]
    for (state,label),color in zip(states.items(),colors):
        data=[next(r for r in geo['summary'] if r['pair']==pair and r['state']==state) for pair in pairs]
        ax.plot(x,[r['relative_delta_l2']['mean'] for r in data],'-o',label=label,color=color,lw=2,ms=6)
    ax.set_title('A  Native states continue to change',loc='left',fontweight='bold')
    ax.set_ylabel('Mean relative L2 update at readout R')
    ax.set_xticks(x,['K1 / K2','K2 / K3','K3 / K4']);ax.set_ylim(bottom=0)
    ax.legend(fontsize=9.5,loc='upper right')
    series=[('target','Same-question KV','#0F4D92','o','-'),
            ('norm_only','Endpoint norm only','#767676','s','--'),
            ('unrelated_delta','Unrelated matched delta','#B64342','^',':')]
    bound=[]
    def find(pair,direction,condition):
        rs=[r for r in audit['summary'] if r['pair']==pair and r['direction']==direction and r['condition']==condition and r['stratum']=='all']
        assert len(rs)==1 and rs[0]['n']==n
        return rs[0]
    for ax,direction,panel,title in [(axes[0,1],'restore','B','Restore later KV in earlier receiver'),(axes[1,0],'rollback','C','Roll back KV in natural later receiver')]:
        for condition,label,color,marker,ls in series:
            values=[find(pair,direction,condition)['meanwrong_delta']['mean'] for pair in pairs]
            bound+=values
            ax.plot(x,values,label=label,color=color,marker=marker,ls=ls,lw=2,ms=6)
        ax.axhline(0,color='#4D4D4D',lw=.9,alpha=.6)
        ax.set_title(panel+'  '+title,loc='left',fontweight='bold')
        ax.set_xticks(x,['K1 / K2','K2 / K3','K3 / K4'])
        ax.set_ylabel('Mean change in gold–meanwrong gap')
    limit=max(abs(v) for v in bound)*1.25
    for ax in (axes[0,1],axes[1,0]):ax.set_ylim(-limit,limit)
    axes[0,1].legend(fontsize=10,loc='lower left')
    ax=axes[1,1];rows=[];labels=[]
    for pair in pairs:
        early,late=pair.split('-')
        for direction in ('restore','rollback'):
            rows.append(find(pair,direction,'target'))
            labels.append((early+'←'+late) if direction=='restore' else (late+'←'+early))
    xx=np.arange(len(rows));rescue=[r['rescue'] for r in rows];damage=[-r['damage'] for r in rows]
    ax.bar(xx,rescue,color='#8BCF8B',edgecolor='#272727',lw=1,label='Rescue')
    ax.bar(xx,damage,color='#E9A6A1',edgecolor='#272727',lw=1,hatch='//',label='Damage')
    for i,(r,d) in enumerate(zip(rescue,damage)):
        if r:ax.text(i,r+.3,str(r),ha='center',va='bottom',fontsize=10)
        if d:ax.text(i,d-.3,str(-d),ha='center',va='top',fontsize=10)
    ax.axhline(0,color='#4D4D4D',lw=1)
    ax.set_xticks(xx,labels);ax.set_xlabel('Receiver loop ← KV donor loop')
    ax.set_ylabel('Questions changed versus native receiver')
    ax.set_title('D  Target KV patch: decision flips',loc='left',fontweight='bold')
    ax.legend(fontsize=10,ncol=2,loc='upper right')
    lo=min(damage+[0]);hi=max(rescue+[0]);span=max(hi-lo,1);ax.set_ylim(lo-.2*span,hi+.25*span)
    fig.text(.5,.034,f'n = {n}. Paired within the RTX 4090 run; BF16 model, FP32 score arithmetic. Means describe this cohort.',ha='center',fontsize=9.5,color='#4D4D4D')
    fig.text(.5,.012,'KV perturbation magnitude varies with depth; panels B/C do not compare equal-size perturbations.',ha='center',fontsize=9.5,color='#4D4D4D')
    fig.tight_layout(rect=(0,.06,1,1),pad=1.4)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for fmt in ('png','pdf','svg'):fig.savefig(a.out.with_suffix('.'+fmt),dpi=300,facecolor='white')
    plt.close(fig)
    receipt={'cohort_title':a.cohort_title,'n':n,'audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),'geometry_sha256':hashlib.sha256(a.geometry.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,'numpy':np.__version__,'panels':['native readout state updates','KV restore with controls','KV rollback with controls','target rescue and damage'],'uncertainty':'No generalization intervals; deterministic within-cohort descriptive estimates.'}
    a.out.with_suffix('.source.json').write_text(json.dumps(receipt,indent=2))
    a.out.with_suffix('.caption.md').write_text('Native readout-state changes and matched causal KV effects across recurrent depth. At physical layer 18, later-loop KV is restored into an earlier receiver, or earlier-loop KV is rolled back into the later receiver, preserving the receiver query. Endpoint-norm and unrelated matched-delta controls use the same SDPA readout-output site. Lines are cohort means; the state panel measures selected states at R, not convergence of the whole model. Decision bars show rescue and damage separately, relative to the recipient’s current native prediction. KV perturbation magnitudes vary across depth: the declining effects combine update size, direction and receiver compatibility, rather than showing sensitivity to an equal-size update. '+a.cohort_title+'.\n',encoding='utf8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
