"""Render the audited natural-K2 Query/joint-KV discriminator."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--query-audit',type=Path,required=True);p.add_argument('--kv-audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cohort-title',default='Validation');a=p.parse_args()
    q=json.loads(a.query_audit.read_text());kv=json.loads(a.kv_audit.read_text())
    assert q['status']==kv['status']=='PASS_COMPLETE' and not q['failures'] and not kv['failures']
    n=q['n_observed'];assert n==q['n_expected']==kv['n_observed']==kv['n_expected']
    qs={r['condition']:r for r in q['summary'] if r['stratum']=='all'}
    reduced_scope='NATIVE_K1_K4_IDENTITY_K1_K4_K2_KV_TARGET_NORM_UNRELATED_ONLY'
    if kv.get('scope')==reduced_scope:
        ks={r['condition']:r for r in kv['summary']}
        assert set(ks)=={'target','norm_only','unrelated_delta'}
        assert all(r['n']==n for r in ks.values())
    else:
        ks={r['condition']:r for r in kv['summary'] if r['stratum']=='all' and r['pair']=='1-2' and r['direction']=='rollback'}
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.6,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(17,5.4));fig.suptitle(f'Natural K2 receiver | Query and joint-KV rollback | {a.cohort_title}, n = {n}',fontsize=19,y=.98)
    series=[('Target','#0F4D92',None),('Endpoint norm','#767676','//'),('Unrelated delta','#B64342','..')]
    rows=[[qs['q_only'],qs['q_only_norm_only'],qs['q_only_unrelated_delta']],[ks['target'],ks['norm_only'],ks['unrelated_delta']],[qs['joint_qkv'],qs['joint_qkv_norm_only'],qs['joint_qkv_unrelated_delta']]]
    ax=axes[0];x=np.arange(3);width=.23
    for i,(label,color,hatch) in enumerate(series):
        values=[row[i]['meanwrong_delta']['mean'] for row in rows]
        ax.bar(x+(i-1)*width,values,width,label=label,color=color,edgecolor='#272727',lw=.8,hatch=hatch)
    ax.axhline(0,color='#4D4D4D',lw=.8);ax.set_xticks(x,['Q only','Joint KV','Joint QKV']);ax.set_ylabel('Mean change in gold–meanwrong gap');ax.set_title('A  Component rollback with controls',loc='left',fontweight='bold');ax.legend(fontsize=10,loc='lower left')
    ax=axes[1];names=['q_only','q_direction_headnorm_matched','q_temperature_headnorm'];labels=['K1 query','K1 direction\nK2 head norms','K2 direction\nK1 head norms'];colors=['#0F4D92','#42949E','#CFCECE']
    values=[qs[k]['meanwrong_delta']['mean'] for k in names]
    bars=ax.bar(np.arange(3),values,color=colors,edgecolor='#272727',lw=.8)
    for bar,name,value in zip(bars,names,values):
        row=qs[name];ax.annotate(f'{value:+.3f}\nR/D {row["rescue"]}/{row["damage"]}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,-8 if value<0 else 8),textcoords='offset points',ha='center',va='top' if value<0 else 'bottom',fontsize=10)
    extent=max(abs(v) for v in values);ax.set_ylim(min(values)-.55*extent,max(0,max(values))+.25*extent)
    ax.axhline(0,color='#4D4D4D',lw=.8);ax.set_xticks(np.arange(3),labels);ax.set_ylabel('Mean change in gold–meanwrong gap');ax.set_title('B  Query direction versus magnitude',loc='left',fontweight='bold')
    ax=axes[2];groups=['rescue','damage','stable_correct','stable_wrong'];labels=['Natural\nrescue','Natural\ndamage','Stable\ncorrect','Stable\nwrong'];values=[[r['meanwrong_exact_raw'] for r in q['factorial_rows'] if r['group']==g] for g in groups]
    assert sum(map(len,values))==n
    bp=ax.boxplot(values,tick_labels=[s+f'\n(n={len(v)})' for s,v in zip(labels,values)],patch_artist=True,showmeans=True,medianprops={'color':'#272727','linewidth':1.4},meanprops={'marker':'D','markerfacecolor':'#B64342','markeredgecolor':'#B64342','markersize':4},flierprops={'marker':'.','markersize':4,'markerfacecolor':'#767676','alpha':.5})
    for patch in bp['boxes']:patch.set_facecolor('#D5E2EF')
    ax.axhline(0,color='#4D4D4D',lw=.8);ax.set_ylabel('Q × joint-KV finite difference (gap)');ax.set_title('C  Heterogeneous end-to-end interaction',loc='left',fontweight='bold');ax.tick_params(axis='x',labelsize=10)
    fig.text(.5,.03,'All conditions use the natural K2 residual and downstream network. R/D: rescue/damage versus current native K2.',ha='center',fontsize=10,color='#4D4D4D')
    fig.tight_layout(rect=(0,.07,1,.98),pad=1.3);a.out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(a.out.with_suffix('.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    receipt={'n':n,'cohort_title':a.cohort_title,'query_audit_sha256':hashlib.sha256(a.query_audit.read_bytes()).hexdigest(),'kv_audit_sha256':hashlib.sha256(a.kv_audit.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,'numpy':np.__version__}
    a.out.with_suffix('.source.json').write_text(json.dumps(receipt,indent=2));a.out.with_suffix('.caption.md').write_text('Query and joint-KV rollback in the natural K2 receiver at L18 SDPA_R. A compares targets with endpoint-norm and unrelated matched-delta controls. B separates per-head Query direction and norm, using FP32 rescaling and BF16 writeback. C shows the full downstream finite difference F(Q1,K1V1)−F(Q1,K2V2)−F(Q2,K1V1)+F(Q2,K2V2), using exact raw-logit gap arithmetic, split by current-native K1/K2 outcomes; diamonds are means. This Q×joint-KV contrast is distinct from the historical K×V attention interaction and is not a mediated proportion. All '+str(n)+' questions in '+a.cohort_title+' are retained.\n',encoding='utf8');print(json.dumps(receipt))


if __name__=='__main__':main()
