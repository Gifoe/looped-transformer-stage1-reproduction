"""Audited four-corner residual-state/SDPA mediation contrasts, without fractions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cohort-title',required=True);a=p.parse_args()
    m=json.loads(a.audit.read_text());assert m['status']=='PASS_COMPLETE' and not m['failures'] and m['n_expected']==m['n_observed']
    n=m['n_observed'];rows={(r['stratum'],r['condition']):r for r in m['summary']};contrast={(r['group'],r['contrast']):r for r in m['contrasts']}
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':1.6,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(16.6,5.8));fig.suptitle('L18 readout state → attention mediation | '+a.cohort_title,fontsize=19,y=.98)
    cells=[['identity','A_only'],['U_restore','U']]
    matrices={g:np.array([[rows[(g,c)]['meanwrong_delta']['mean'] for c in row] for row in cells]) for g in ['all','rescue']}
    limit=max(np.abs(v).max() for v in matrices.values());limit=max(limit,.001)
    cmap=LinearSegmentedColormap.from_list('causal_gap',['#B64342','#FFFFFF','#0F4D92'])
    labels={'identity':'N','A_only':'A only','U_restore':'U + A2','U':'U'}
    for ax,g,letter,title in zip(axes[:2],['all','rescue'],['A','B'],['All questions','Natural K1→K2 rescues']):
        z=matrices[g];im=ax.imshow(z,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto')
        for i in range(2):
            for j in range(2):
                key=cells[i][j];r=rows[(g,key)];text=f'{labels[key]}\nΔgap {z[i,j]:+.3f}\ncorrect {r["correct"]}/{r["n"]}'
                ax.text(j,i,text,ha='center',va='center',fontsize=12,color='white' if abs(z[i,j])>.55*limit else '#272727')
        ax.set_xticks([0,1],['A2\nnative SDPA','A(U)\nSDPA under block']);ax.set_yticks([0,1],['X2\nnative residual','X1\nrolled-back residual']);ax.set_title(letter+'  '+title,loc='left',fontweight='bold');ax.tick_params(length=0)
        for s in ax.spines.values():s.set_visible(False)
    ax=axes[2];names=['restore_effect','norm_restore_effect','unrelated_restore_effect'];labels3=['Restore A2','Match A2 norm','Unrelated\nmatched delta'];values=[contrast[('all',c)]['meanwrong']['mean'] for c in names]
    bars=ax.bar(range(3),values,color=['#0F4D92','#767676','#B64342'],edgecolor='#272727',lw=.9)
    for bar,value in zip(bars,values):ax.annotate(f'{value:+.3f}',(bar.get_x()+bar.get_width()/2,value),xytext=(0,5 if value>=0 else -5),textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=11)
    ax.axhline(0,color='#4D4D4D',lw=.8);ax.set_xticks(range(3),labels3);ax.set_title('C  Attention restoration under X1',loc='left',fontweight='bold');ax.set_ylabel('Mean gap change relative to U')
    extent=max(max(abs(v) for v in values),.01);ax.set_ylim(min(0,min(values))-.22*extent,max(0,max(values))+.35*extent)
    fig.text(.5,.048,'A/B: mean gap change versus native N and number correct. C: restoration effects versus U, with the X1 block retained.',ha='center',fontsize=10,color='#4D4D4D')
    fig.text(.5,.018,'A(U) is measured in the actual upstream-block forward; it is not natural K1 attention. No mediated fraction is estimated.',ha='center',fontsize=10,color='#4D4D4D')
    fig.tight_layout(rect=(0,.09,1,.99),pad=1.4)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:fig.savefig(a.out.with_suffix('.'+ext),dpi=300,facecolor='white')
    plt.close(fig)
    receipt={'n':n,'cohort_title':a.cohort_title,'audit_sha256':hashlib.sha256(a.audit.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,'numpy':np.__version__,'baselines':{'A/B':'N','C':'U'}}
    a.out.with_suffix('.source.json').write_text(json.dumps(receipt,indent=2));a.out.with_suffix('.caption.md').write_text('Controlled L18 readout-state/attention-output mediation. X1 is the same-question K1 L18 input residual at final token R, X2 the natural K2 residual. U replaces X2 with X1 and measures the actual resulting SDPA row A(U); A2 is natural K2 SDPA. The four cells retain the indicated residual/SDPA combination and execute the remaining model normally. A/B show paired mean gold-minus-meanwrong changes relative to N and classification counts; B conditions on the native K1-wrong/K2-correct group. C compares restoring A2 with endpoint-norm and unrelated-delta restoration controls while retaining X1, relative to U. Effects bypassing this L18 attention-output mediator may still pass through L18 MLP and later attention/MLP; they are not a pure-MLP or globally attention-free effect. No generating upstream computation or mediated proportion is inferred. '+a.cohort_title+'.\n',encoding='utf8');print(json.dumps(receipt))


if __name__=='__main__':main()
