"""Measured margins, layer lesions and write diagnostics; PNG/PDF/editable SVG."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from linux_runtime import ROOT


def main():
    out=ROOT/'fresh/causal_original_shape';figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':['DejaVu Sans'],'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    cases=[]
    for p in sorted(out.glob('batch_*.json')):
        d=json.loads(p.read_text(encoding='utf8'))
        cases += [(c,i,d) for i,c in enumerate(d['cases']) if c['selected']]
    layers=[6,9,12,15,18,21,24,27]
    data=np.array([[d['conditions'][f'drop_layer_{l}'][c['target_receiver']][i]['dot64_margin']-d['conditions']['link'][c['target_receiver']][i]['dot64_margin'] for l in layers] for c,i,d in cases])
    bound=max(float(abs(data).max()),.01)
    fig,ax=plt.subplots(figsize=(10.6,4.6))
    im=ax.imshow(data,cmap='RdBu',norm=TwoSlopeNorm(vcenter=0,vmin=-bound,vmax=bound),aspect='auto')
    labels=[c['question_id']+' / '+c['target_receiver'][0].upper() for c,i,d in cases]
    ax.set(xticks=np.arange(8),xticklabels=layers,yticks=np.arange(len(cases)),yticklabels=labels,xlabel='Bridge layer removed (both directions)',title='Original-batch causal lesions: change in gold-answer margin')
    for i in range(len(cases)):
        for j in range(8):ax.text(j,i,f'{data[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(data[i,j])>.65*bound else 'black')
    fig.colorbar(im,ax=ax,label='FP64 readout delta; same BF16 hidden states',fraction=.035,pad=.025)
    fig.tight_layout();export(fig,figdir/'layer_lesion_margins')
    c,i,d=next(t for t in cases if t[0]['question_id']=='NCEOGA_2013_8_57')
    x=c['target_receiver'];cond=d['conditions']
    names=['cut','link','ember_to_quill_only','clean_kv_all_ember','clean_gate_all_ember','clean_kv_normmatched_all_ember']
    labels=['CUT','ON','One-way E→Q','CUT K/V','CUT gates','CUT K/V\nON write norms']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),gridspec_kw={'width_ratios':[1.3,1]})
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.axhline(0,color='.45',lw=.8)
    vals=[cond[n][x][i]['dot64_margin'] for n in names]
    axes[0].bar(np.arange(len(vals)),vals,color=['#767676','#B64342','#42949E','#0F4D92','#B07C45','#0F4D92'],edgecolor='black',lw=.5)
    for j,v in enumerate(vals):axes[0].text(j,v+(.04 if v>=0 else -.04),f'{v:+.3f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=9)
    axes[0].set(xticks=np.arange(len(vals)),xticklabels=labels,ylim=(min(vals)-.14,max(vals)+.10),ylabel='Gold-answer logit margin',title='A  Correct-peer feedback candidate')
    for prefix,label,color in [('drop_reverse_ember_read_','Cut reverse at one layer','#42949E'),('clean_kv_ember_','CUT K/V at one layer','#0F4D92'),('clean_gate_ember_','CUT gate at one layer','#B07C45'),('clean_kv_normmatched_ember_','CUT K/V, ON write norm','#8B5C8E')]:
        values=[cond[prefix+str(l)][x][i]['dot64_margin'] for l in layers]
        axes[1].plot(layers,values,marker='o',ms=4,lw=1.3,label=label,color=color)
    axes[1].axhline(cond['link'][x][i]['dot64_margin'],color='#B64342',ls='--',lw=1,label='ON')
    axes[1].set(xticks=layers,xlabel='Single intervened bridge layer',ylabel='Gold-answer logit margin',title='B  Where the reverse path matters')
    axes[1].legend(fontsize=8,loc='best')
    fig.suptitle('NCEOGA_2013_8_57 / Quill; original batch shape\nFP64 output dot products on the same BF16 hidden states',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.95]);export(fig,figdir/'feedback_path_dissection')
    with (out/'layerwise_analysis.csv').open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(9,3.7))
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    for x,color in [('quill','#0F4D92'),('ember','#42949E')]:
        rr=[r for r in rows if r['receiver']==x and r['condition']=='link']
        q=np.array([np.quantile([float(r['write_hidden_ratio']) for r in rr if int(r['layer'])==l],[.1,.5,.9]) for l in layers])
        axes[0].plot(layers,q[:,1],marker='o',ms=4,color=color,label=x.title());axes[0].fill_between(layers,q[:,0],q[:,2],color=color,alpha=.15)
        mass=[max(float(r['null_attention_mass']),1e-12) for r in rr]
        axes[1].hist(np.log10(mass),bins=np.linspace(-12,0,31),histtype='step',lw=1.6,color=color,label=x.title())
    axes[0].set(xticks=layers,xlabel='Bridge layer',ylabel='Write L2 / hidden-state L2',title='A  ON write amplitude (median, 10–90%)')
    axes[1].set(xlabel='log10 mean null-attention mass',ylabel='Question × layer',title='B  Null attention is usually small')
    for ax in axes:ax.legend(fontsize=9)
    fig.tight_layout();export(fig,figdir/'bridge_diagnostics')
    validation=ROOT/'fresh/feedback_validation/validation_cases.csv'
    if validation.exists():
        with validation.open(encoding='utf8',newline='') as f:vv=list(csv.DictReader(f))
        fig,axes=plt.subplots(1,2,figsize=(12,7),gridspec_kw={'width_ratios':[1.45,1]})
        for ax in axes:ax.spines[['top','right']].set_visible(False)
        labels=[v['question_id']+' / '+v['receiver'][0].upper() for v in vv]
        delta=np.array([float(v['clean_kv_normmatched_dot64_margin'])-float(v['dot64_on_margin']) for v in vv])
        native=np.array([float(v['clean_kv_normmatched_margin'])-float(v['on_margin']) for v in vv])
        colors=['#0F4D92' if v['receiver']=='quill' else '#42949E' for v in vv]
        axes[0].barh(np.arange(len(vv)),delta,color=colors,height=.65)
        axes[0].plot(native,np.arange(len(vv)),'k|',ms=10,label='Native BF16 delta')
        axes[0].axvline(0,color='.45',lw=.8)
        axes[0].set(yticks=np.arange(len(vv)),yticklabels=labels,xlabel='Change in gold margin; fixed FP32 write norms',title='A  Positive margin shifts do not imply answer repair')
        axes[0].tick_params(axis='y',labelsize=8);axes[0].invert_yaxis();axes[0].legend(fontsize=8)
        for v,color in zip(vv,colors):
            stable=v['transition_survives_dot64']=='True'
            axes[1].scatter(float(v['dot64_on_margin']),float(v['clean_kv_normmatched_dot64_margin']),s=50,
                marker='o' if v['receiver']=='quill' else '^',edgecolor=color,facecolor=color if stable else 'none',lw=1.2)
        bounds=[min(float(v['dot64_on_margin']) for v in vv)-.1,max(float(v['clean_kv_normmatched_dot64_margin']) for v in vv)+.15]
        axes[1].plot(bounds,bounds,color='.5',ls='--',lw=1)
        axes[1].axhline(0,color='.5',lw=.8);axes[1].axvline(0,color='.5',lw=.8)
        axes[1].set(xlim=bounds,ylim=bounds,xlabel='ON gold margin (FP64 output dot)',ylabel='Norm-matched CUT K/V gold margin',title='B  Threshold and precision controls')
        axes[1].text(.04,.96,'16 new target cases\n13 positive shifts\n0 repairs in both readouts\n\nOpen: ON/CUT transition\nchanges under head-dot check',transform=axes[1].transAxes,va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.94,pad=3))
        fig.suptitle('Correct-peer harm validation; selected failures, not a benchmark gain estimate',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.95]);export(fig,figdir/'validation_margin_effects')


def export(fig,path):
    for ext in ['png','pdf','svg']:fig.savefig(path.with_suffix('.'+ext),dpi=300,facecolor='white')
    plt.close(fig)


if __name__=='__main__':main()
