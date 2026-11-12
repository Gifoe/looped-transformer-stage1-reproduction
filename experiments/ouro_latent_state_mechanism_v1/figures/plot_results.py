import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P=Path(__file__).resolve().parent
with (P/'layer_curves.csv').open(newline='',encoding='utf-8') as f: layers=list(csv.DictReader(f))
with (P/'l18_swaps.csv').open(newline='',encoding='utf-8') as f: swaps=list(csv.DictReader(f))
colors={'train':'#275D8C','validation':'#D27A2C','test':'#20806F'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
fig=plt.figure(figsize=(11.2,7.4),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.05]); a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
for split in ('train','validation','test'):
    rows=[r for r in layers if r['split']==split];x=[int(r['layer_ordinal_1_based']) for r in rows]
    label=('Train non-demo' if split=='train' else split.title())+f" (n={rows[0]['n']})"
    a.plot(x,[float(r['competition_k2_minus_k1']) for r in rows],color=colors[split],label=label,lw=1.8)
    for name,style in [('attention_k2_minus_k1','-'),('mlp_k2_minus_k1','--')]: b.plot(x,[float(r[name]) for r in rows],color=colors[split],ls=style,lw=1.65)
for ax in (a,b):
    ax.axhline(0,color='#999999',lw=.7);ax.set_xlim(1,24);ax.set_xticks([1,6,12,18,24]);ax.set_xlabel('Layer ordinal (1–24)');ax.grid(axis='y',alpha=.16)
a.set_title('A  Natural candidate competition · FP64 lens',loc='left',weight='bold');a.set_ylabel('FP64 lens gold–wrong gap: K2 − K1');a.legend(frameon=False,fontsize=8,loc='best')
b.set_title('B  Natural branch lens changes',loc='left',weight='bold');b.set_ylabel('Mean branch lens change: K2 − K1');b.legend(handles=[Line2D([0],[0],color='#333',ls='-',label='Attention'),Line2D([0],[0],color='#333',ls='--',label='MLP')],frameon=False,fontsize=8)
for i,r in enumerate(swaps):
    y=5-i; n=int(r['n']);base=int(r['baseline_correct']);end=int(r['swap_correct']); col=colors[r['split']]
    c.plot([100*base/n,100*end/n],[y,y],color=col,lw=2.2);c.scatter([100*base/n],[y],facecolor='white',edgecolor=col,s=48,zorder=3);c.scatter([100*end/n],[y],color=col,s=48,zorder=3)
    c.text(101,y,f"{base} → {end}  |  +{r['wrong_to_correct']} / −{r['correct_to_wrong']}  (net {int(r['net_correct']):+d})",va='center',fontsize=8)
c.set_yticks(range(5,-1,-1),[('Train non-demo' if r['split']=='train' else r['split'].title())+'  '+r['condition'] for r in swaps]);c.set_xlim(0,160);c.set_xticks([0,25,50,75,100]);c.set_xlabel('Correct answers (% of split)');c.grid(axis='x',alpha=.16)
c.set_title('C  L18 activation swap · native scoring independently replayed on CUDA',loc='left',weight='bold');c.text(101,5.75,'Correct count; rescues / losses',fontsize=8,weight='bold');c.text(0,-1.15,'A–B: offline normalized readout differences, not intermediate-layer probabilities; each question and wrong option is equally weighted.\nC: T12 transfers the K2 L18 R attention increment to K1; T21 reverses it, after input_layernorm_2 and before residual addition.\nOpen = baseline; filled = swap. Displayed layer ordinal: 1–24. Train excludes demonstrations.',fontsize=8,transform=c.transData,va='top')
fig.savefig(P/'natural_l18_results.png',bbox_inches='tight');fig.savefig(P/'natural_l18_results.pdf',bbox_inches='tight');plt.close(fig)
