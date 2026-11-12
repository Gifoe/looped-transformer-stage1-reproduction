"""Matched R-state→actual-SDPA path comparison on the same full Validation IDs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--audits',nargs=3,type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cohort-title',default='Validation');a=p.parse_args()
    data=[json.loads(p.read_text()) for p in a.audits]
    n=data[0]['n_observed']
    for i,m in enumerate(data):
        assert m['status']=='PASS_COMPLETE' and not m['failures'] and m['n_observed']==m['n_expected']==n
        assert m.get('receiver_loop',2)==i+2
    ids=[{(r['split'],r['id']) for r in m['contrast_rows']} for m in data];assert len(ids[0])==n and ids[0]==ids[1]==ids[2]
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.6,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(17,5.7));fig.suptitle(f'Matched residual-state → attention path across loop depth | {a.cohort_title}, same {n} questions',fontsize=18,y=.98)
    x=np.arange(3);pairlabels=['K2 ← K1','K3 ← K2','K4 ← K3']
    summary=[{(r['stratum'],r['condition']):r for r in m['summary']} for m in data]
    contrasts=[{(r['group'],r['contrast']):r for r in m['contrasts']} for m in data]
    ax=axes[0]
    for condition,label,color,marker in [('U','State block U','#0F4D92','o'),('U_restore','State block + native attention','#42949E','s'),('A_only','Block-induced attention only','#9A4D8E','^')]:
        values=[s[('all',condition)]['meanwrong_delta']['mean'] for s in summary];ax.plot(x,values,color=color,marker=marker,label=label,lw=2,ms=6)
    ax.axhline(0,color='#4D4D4D',lw=.8);ax.set_xticks(x,pairlabels);ax.set_ylabel('Mean gap change relative to native N');ax.set_title('A  Same four-corner intervention',loc='left',fontweight='bold');ax.legend(fontsize=9.5,loc='lower right')
    ax=axes[1]
    for name,label,color,marker,style in [('restore_effect','Native attention','#0F4D92','o','-'),('norm_restore_effect','Norm control','#767676','s','--'),('unrelated_restore_effect','Unrelated delta','#B64342','^',':')]:
        values=[c[('all',name)]['meanwrong']['mean'] for c in contrasts];ax.plot(x,values,color=color,marker=marker,ls=style,label=label,lw=2,ms=6)
    ax.axhline(0,color='#4D4D4D',lw=.8);ax.set_xticks(x,pairlabels);ax.set_ylabel('Mean restoration gap change relative to U');ax.set_title('B  Attention restoration under state block',loc='left',fontweight='bold');ax.legend(fontsize=9,loc='lower left')
    loss=[];restored=[];strict=[];opportunities=[]
    required={'group','restore_correct','mediator_norm_correct','mediator_unrelated_correct','state_norm_correct','state_unrelated_correct','native_no_tie','U_tie','restore_tie'}
    for m,s in zip(data,summary):
        cases=m['upstream_damage_cases'];assert all(required<=set(c) for c in cases)
        cases=[c for c in cases if c['group']=='rescue'];loss.append(len(cases));restored.append(sum(c['restore_correct'] for c in cases));opportunities.append(s.get(('rescue','identity'),{'n':0})['n'])
        strict.append(sum(c['restore_correct'] and not c['mediator_norm_correct'] and not c['mediator_unrelated_correct'] and c['state_norm_correct'] and c['state_unrelated_correct'] and c['native_no_tie'] and not c['U_tie'] and not c['restore_tie'] for c in cases))
    ax=axes[2];width=.24
    for i,(values,label,color,hatch) in enumerate([(loss,'Natural corrections lost under U','#E9A6A1','//'),(restored,'Lost corrections restored by attention','#8BCF8B',None),(strict,'Strict recovery with all four controls','#0F4D92',None)]):
        bars=ax.bar(x+(i-1)*width,values,width,color=color,label=label,edgecolor='#272727',lw=.8,hatch=hatch)
        for bar,v in zip(bars,values):
            if v:ax.annotate(str(v),(bar.get_x()+bar.get_width()/2,v),xytext=(0,3),textcoords='offset points',ha='center',fontsize=10)
    ax.set_xticks(x,[p+f'\n(n = {n})' for p,n in zip(pairlabels,opportunities)]);ax.set_ylim(0,max(loss+restored+strict+[1])*1.35);ax.set_ylabel('Questions');ax.set_title('C  Paired decision losses and recoveries',loc='left',fontweight='bold');ax.legend(fontsize=8.7,loc='upper right')
    fig.text(.5,.047,'C: n denotes native adjacent-loop corrections at risk. Update sizes and risk sets change with depth; these are not equal-amplitude sensitivity tests.',ha='center',fontsize=10,color='#4D4D4D')
    fig.text(.5,.017,'All cases and counterexamples retained. Recovery counts are controlled-intervention outcomes, not fractions of natural gains explained.',ha='center',fontsize=10,color='#4D4D4D')
    fig.tight_layout(rect=(0,.09,1,.98),pad=1.3);a.out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(a.out.with_suffix('.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    receipt={'n':n,'cohort_title':a.cohort_title,'audit_files':[str(p) for p in a.audits],'audit_sha256':[hashlib.sha256(p.read_bytes()).hexdigest() for p in a.audits],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,'numpy':np.__version__,'adjacent_native_corrections':opportunities,'U_losses':loss,'attention_recoveries':restored,'strict_all_four_control_recoveries':strict}
    a.out.with_suffix('.source.json').write_text(json.dumps(receipt,indent=2));a.out.with_suffix('.caption.md').write_text('Matched intervention at L18 readout R in natural K2, K3 and K4, rolling back R residual to the same question’s immediately preceding loop and measuring actual A(U). A compares state block U, block with native attention restored, and block-induced attention transferred alone, all relative to N. B holds the block and compares native-attention restoration with norm and unrelated controls, relative to U. C counts native adjacent-loop corrections destroyed by U and recovered by native attention; strict cases additionally require no native/U/restore top ties, neither mediator-control recovery, and both input-state controls preserving correctness. Correction groups differ by transition even though the '+str(n)+' question cohort is identical. Differences combine natural update amplitude, direction and receiver compatibility; they do not establish global convergence, equal-amplitude sensitivity, or a mediated percentage. Cohort: '+a.cohort_title+'.\n',encoding='utf8');print(json.dumps(receipt))


if __name__=='__main__':main()
