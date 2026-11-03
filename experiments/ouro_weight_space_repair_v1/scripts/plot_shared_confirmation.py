"""Plot closed discovery-only scope controls; never reads held-out results."""
import csv,json,os
from runtime import ROOT,configure
configure();os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')

def main():
 folder=ROOT/'results/shared_confirmation_original';complete=json.loads((folder/'COMPLETE.json').read_text());assert complete['status']=='complete'
 rows=[json.loads(l) for l in (folder/'scores.jsonl').read_text().splitlines()]
 native={r['id']:r for r in map(json.loads,(ROOT/'results/discovery_original/scores.jsonl').read_text().splitlines())}
 ids=json.loads((folder/'FROZEN.json').read_text())['ids'];assert len(ids)==8 and all('/train/' in uid for uid in ids)
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained');summary=[]
 for ax,module in zip(axes[:2],complete['modules']):
  ks=[3,4,5,6];baseline=[sum(native[uid]['methods'][f'native_k{k}']['correct'] for uid in ids) for k in ks]
  ax.plot(ks,baseline,color='#555555',marker='o',linestyle='--',label='Native')
  for scope,label,color in [('loop5','Only loop5','#388A80'),('all_loops','All loops','#C96354')]:
   part=[r for r in rows if r['layer']==module['layer'] and r['branch']==module['branch'] and r['scope']==scope];assert {r['id'] for r in part}==set(ids)
   values=[sum(r['methods'][f'native_k{k}']['correct'] for r in part) for k in ks]
   ax.plot(ks,values,color=color,marker='o',label=label)
   for k,c,b in zip(ks,values,baseline):
    rescues=sum(not native[r['id']]['methods'][f'native_k{k}']['correct'] and r['methods'][f'native_k{k}']['correct'] for r in part)
    damages=sum(native[r['id']]['methods'][f'native_k{k}']['correct'] and not r['methods'][f'native_k{k}']['correct'] for r in part)
    summary.append(dict(layer=module['layer'],branch=module['branch'],scope=scope,k=k,n=8,native_correct=b,intervened_correct=c,rescues=rescues,regressions=damages,net_correct=c-b,mean_delta_margin=sum(r['delta_margin_vs_native'][f'k{k}'] for r in part)/8))
  ax.set(xlabel='Recurrent loops K',ylabel='Correct questions (out of8)',xticks=ks,yticks=[0,2,4,6,8],ylim=(0,8),title=f"Physical layer{module['layer']}: {module['branch']}");ax.legend(fontsize=8,loc='lower right')
 targets=['arc_challenge/train/Mercury_7180758','arc_challenge/train/Mercury_SC_415737']
 x=np.arange(2);width=.24
 baseline=[native[uid]['methods']['native_k5']['margin'] for uid in targets]
 axes[2].bar(x-width,baseline,width,color='#777777',label='Native K5')
 for offset,scope,label,color in [(0,'loop5','MLP0, only loop5','#388A80'),(width,'all_loops','MLP0, all loops','#C96354')]:
  values=[next(r for r in rows if r['id']==uid and r['layer']==0 and r['branch']=='mlp' and r['scope']==scope)['methods']['native_k5']['margin'] for uid in targets]
  axes[2].bar(x+offset,values,width,color=color,label=label)
 axes[2].axhline(0,color='black',linewidth=.7);axes[2].set(xticks=x,xticklabels=['Target harm\nMercury_7180758','Native rescue\nMercury_SC_415737'],ylabel='Gold margin (nats / character)',title='Single-case benefit and collateral harm');axes[2].legend(fontsize=7)
 fig.suptitle('Selected discovery cases (n=8): post-norm scalar activation controls; no weight edit',fontsize=11)
 (ROOT/'figures').mkdir(exist_ok=True)
 for ext in ['png','svg','pdf']:fig.savefig(ROOT/'figures'/('shared_vs_loop5.'+ext),dpi=200)
 plt.close(fig)
 with (ROOT/'shared_confirmation_summary.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 print('shared_confirmation_figure_complete',len(summary))
if __name__=='__main__':main()
