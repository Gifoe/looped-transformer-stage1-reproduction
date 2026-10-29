"""Exploratory plots and uncertainty for the completed 400-question sample."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analysis_core import ACTIONS,targets
ROOT=Path(__file__).resolve().parents[1]
def main():
 with (ROOT/'results/discovery.jsonl').open(encoding='utf8') as f:rows=[json.loads(l) for l in f if l.strip()]
 assert len(rows)==400
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axes=plt.subplots(1,3,figsize=(13,4.5),sharey=True,layout='constrained');report={};rng=np.random.default_rng(2021)
 for ai,(a,ax) in enumerate(zip(ACTIONS[1:],axes)):
  y=np.array([targets(r)[ai] for r in rows]);v=np.array([r['views']['official']['probe_nll']['adaptive']-r['views']['official']['probe_nll'][a] if r['views']['official']['probe_nll']['adaptive'] is not None else np.nan for r in rows]);mask=(y!=0)&np.isfinite(v);pos=v[mask&(y>0)];neg=v[mask&(y<0)]
  labels=np.r_[np.ones(len(pos)),np.zeros(len(neg))];auc=roc_auc_score(labels,np.r_[pos,neg]);bs=[]
  for _ in range(2000):bs.append(roc_auc_score(labels,np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]))
  ci=np.quantile(bs,[.025,.975]);report[a]=dict(auc=float(auc),ci95=ci.tolist(),positive=len(pos),negative=len(neg),interpretation='Exploratory; prespecified sign; stratified bootstrap2000')
  for x,data,color in [(0,pos,'#42949E'),(1,neg,'#B64342')]:ax.scatter(x+rng.uniform(-.12,.12,len(data)),data,color=color,alpha=.65,s=24);ax.plot([x-.2,x+.2],[np.median(data)]*2,color='black',linewidth=2)
  ax.axhline(0,color='#777777',linestyle='--',linewidth=.8);ax.set_xticks([0,1],[f'Beneficial\nn={len(pos)}',f'Harmful\nn={len(neg)}']);ax.set_title(f'{a.capitalize()} vs Adaptive\nAUC={auc:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]')
 axes[0].set_ylabel('Question-token NLL reduction vs Adaptive');fig.suptitle('Discovery only: lower question loss is not a quality certificate')
 for ext in ['png','pdf']:fig.savefig(ROOT/'figures'/('discovery_proxy.'+ext),dpi=240,bbox_inches='tight')
 plt.close(fig);(ROOT/'results/DISCOVERY_PROXY_UNCERTAINTY.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
