"""Secondary uncertainty audit, never used for controller/model selection.

Stratified paired bootstrap over nonzero gain events, 2000 resamples, seed2021.
Reuses frozen heads and keeps proxy signs unchanged. Resampling the same event
indices for two heads estimates their paired AUC difference. These intervals
are exploratory and unadjusted, not a new confirmation gate.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from analysis_core import ACTIONS,features,targets
ROOT=Path(__file__).resolve().parents[1]
def auc(y,v):
 n=int(y.sum());m=len(y)-n
 return float((rankdata(v)[y].sum()-n*(n+1)/2)/(n*m))
def main():
 models=json.loads((ROOT/'configs/heads.json').read_text());report={}
 for split in ['validation','final_sciq','final_piqa']:
  with (ROOT/'results'/(split+'.jsonl')).open(encoding='utf8') as f:rows=[json.loads(l) for l in f if l.strip()]
  Y=np.stack([targets(r) for r in rows]);values={};xs={}
  for kind in ['controls','new','combined']:xs[kind]=np.stack([features(r,kind)[0] for r in rows])
  for name,m in models.items():values[name]=((xs[m['kind']]-m['mean'])/m['scale'])@np.array(m['coef']).T+m['intercept']
  out={}
  for i,action in enumerate(ACTIONS[1:]):
   mask=Y[:,i]!=0;y=Y[mask,i]>0;pos=np.flatnonzero(y);neg=np.flatnonzero(~y);v={k:x[mask,i] for k,x in values.items()};stats={};diff={}
   if len(pos) and len(neg):
    rng=np.random.default_rng(2021);sample={k:[] for k in v}
    for _ in range(2000):
     ix=np.r_[rng.choice(pos,len(pos)),rng.choice(neg,len(neg))]
     for k in v:sample[k].append(auc(y[ix],v[k][ix]))
    for k in v:stats[k]=dict(auc=auc(y,v[k]),ci95=np.quantile(sample[k],[.025,.975]).tolist())
    for target in ['accuracy','logweight']:
     for kind in ['new','combined']:
      a='head_'+kind+'_'+target;b='head_controls_'+target
      if a in v and b in v:diff[a+'_minus_controls']=dict(delta_auc=stats[a]['auc']-stats[b]['auc'],ci95=np.quantile(np.array(sample[a])-sample[b],[.025,.975]).tolist())
   out[action]=dict(positive=len(pos),negative=len(neg),heads=stats,paired_auc_differences=diff)
  report[split]=out
 result=dict(method='2000 stratified paired event bootstrap; fixed signs; seed2021; exploratory unadjusted intervals; conditional on nonzero correctness gains',datasets=report)
 (ROOT/'results/SIGNAL_UNCERTAINTY.json').write_text(json.dumps(result,indent=2),encoding='utf8');print('Wrote SIGNAL_UNCERTAINTY.json')
if __name__=='__main__':main()
