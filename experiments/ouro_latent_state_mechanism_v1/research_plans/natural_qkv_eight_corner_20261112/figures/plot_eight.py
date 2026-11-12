import csv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parent
with (P/'effects.csv').open(encoding='utf8',newline='') as f: rows=list(csv.DictReader(f))
with (P/'corners.csv').open(encoding='utf8',newline='') as f: corners=list(csv.DictReader(f))
colors={'train':'#275D8C','validation':'#D27A2C','test':'#20806F'};offset={'train':-.18,'validation':0,'test':.18}
panels=[('A  Key edges',[('K_Q1V1','Q1, V1'),('K_Q1V2','Q1, V2'),('K_Q2V1','Q2, V1'),('K_Q2V2','Q2, V2')]),('B  Value edges',[('V_Q1K1','Q1, K1'),('V_Q1K2','Q1, K2'),('V_Q2K1','Q2, K1'),('V_Q2K2','Q2, K2')]),('C  Query edges',[('Q_K1V1','K1, V1'),('Q_K1V2','K1, V2'),('Q_K2V1','K2, V1'),('Q_K2V2','K2, V2')]),('D  Interactions',[('I_Q1','K×V | Q1'),('I_Q2','K×V | Q2'),('J_QKV','I(Q2) − I(Q1)')])]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
fig,axes=plt.subplots(1,4,figsize=(14.5,4.4),layout='constrained',sharex=True)
lookup={(r['split'],r['contrast']):float(r['mean_score_gap_change']) for r in rows}
for ax,(title,items) in zip(axes,panels):
 for i,(key,label) in enumerate(items):
  y=len(items)-1-i
  for split in ('train','validation','test'):ax.scatter(lookup[(split,key)],y+offset[split],color=colors[split],s=27,zorder=3)
 ax.set_yticks(range(len(items)-1,-1,-1),[x[1] for x in items]);ax.axvline(0,color='#999',lw=.8);ax.grid(axis='x',alpha=.15);ax.set_title(title,loc='left',weight='bold');ax.set_xlabel('Mean score-gap contrast')
fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=colors[s],markeredgecolor=colors[s],label=lab,markersize=5) for s,lab in [('train','Train non-demo (n=1,094)'),('validation','Validation (n=299)'),('test','Test (n=1,172)')]],loc='outside lower center',ncol=3,frameon=False)
fig.suptitle('Q × K × V at L18: mean score-gap contrasts, fixed K1 receiver',weight='bold',x=.01,ha='left')
fig.savefig(P/'eight_corner_edges.png',bbox_inches='tight');fig.savefig(P/'eight_corner_edges.pdf',bbox_inches='tight');plt.close(fig)
order=['G111','G112','G121','G122','G211','G212','G221','G222'];lines=['# 八角正确题数','', '| 数据 | N | '+' | '.join(order)+' |','|---|---:|'+'---:|'*8]
for split,label in [('train','Train non-demo'),('validation','Validation'),('test','Test')]:
 group=[r for r in corners if r['split']==split];d={r['corner']:r for r in group};lines.append('| '+label+' | '+group[0]['n']+' | '+' | '.join(d[k]['correct'] for k in order)+' |')
lines+=['','正确数使用首索引 argmax；不是自然 K2 准确率表。Validation 全体在固定 Q1、K1 下的 V 边 G112−G111 为 149−150 = −1，未满足该预注册净正确数为正的判据；平均 gap 与正确数是不同指标。完整逐题分数、平分与反例见各 split rows 和分析。']
(P/'EIGHT_CORNER_TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
