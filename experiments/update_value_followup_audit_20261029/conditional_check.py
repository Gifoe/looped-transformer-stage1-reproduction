"""Two fixed-capacity development checks; never load final task data."""
import json,hashlib,sys,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'frozen_loop_update_value_seed2021_v1'
sys.path.insert(0,str(BASE/'scripts'))
from analysis_core import features,paired,holm,ACTIONS
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text(encoding='utf8'))
def read(split):
 assert split in ['head_train','validation']
 with (BASE/'results'/(split+'.jsonl')).open(encoding='utf8') as f:return [json.loads(l) for l in f]
def predictions(rows):return np.array([[np.argmax(np.asarray(r['views']['official']['sum_scores'][a])/r['choice_char_lengths']) for a in ACTIONS] for r in rows])
def main():
 freeze=load(ROOT/'CONDITIONAL_LOCK.json')
 for rel,h in freeze['files'].items():assert sha(ROOT/rel)==h
 for rel,h in freeze['original_inputs'].items():assert sha(BASE/rel)==h
 assert not (ROOT/'CONDITIONAL_RESULTS.json').exists(),'No refitting an already evaluated development check'
 train=read('head_train');val=read('validation');tp=predictions(train);vp=predictions(val);tg=np.array([r['gold'] for r in train]);vg=np.array([r['gold'] for r in val]);base=vp[:,0]==vg;models={};summaries={};records=[];start=time.perf_counter()
 for kind in ['controls','new']:
  X=np.stack([features(r,kind)[0] for r in train]);V=np.stack([features(r,kind)[0] for r in val]);values=np.zeros((len(val),3));model=[]
  for i,a in enumerate(ACTIONS[1:]):
   mask=tp[:,i+1]!=tp[:,0];Y=(tp[:,i+1]==tg).astype(int)-(tp[:,0]==tg).astype(int);mean=X[mask].mean(0);scale=X[mask].std(0);scale[scale<1e-8]=1
   fit=Ridge(alpha=100).fit((X[mask]-mean)/scale,Y[mask]);raw=(V-mean)/scale@fit.coef_+fit.intercept_;values[:,i]=np.where(vp[:,i+1]!=vp[:,0],raw,0)
   model.append(dict(action=a,training_disagreements=int(mask.sum()),positive=int((Y[mask]>0).sum()),negative=int((Y[mask]<0).sum()),neutral=int((Y[mask]==0).sum()),mean=mean.tolist(),scale=scale.tolist(),coef=fit.coef_.tolist(),intercept=float(fit.intercept_),features=features(train[0],kind)[1]))
  best=values.argmax(1);action=np.where(values[np.arange(len(val)),best]>.05,best+1,0);pred=vp[np.arange(len(val)),action];name='conditional_'+kind;summaries[name]=paired(pred==vg,base)
  parameters=sum(len(m['coef'])+1 for m in model);assert parameters==({'controls':96,'new':345}[kind]);models[name]=dict(parameters=parameters,actions=model,alpha=100,threshold=.05)
  for i,r in enumerate(val):records.append(dict(policy=name,id=r['id'],gold=int(vg[i]),prediction=int(pred[i]),action=ACTIONS[action[i]],values=values[i].tolist()))
 original=load(BASE/'results/validation_analysis.json');old_candidates=load(BASE/'configs/FINAL_LOCK.json')['adjusted_p'];p={k:original['summary'][k]['mcnemar_p'] for k in old_candidates};p.update({k:v['mcnemar_p'] for k,v in summaries.items()});adjusted=holm(p)
 eligible=[k for k,x in summaries.items() if x['net']>=5 and x['ci95_pp'][0]>0 and adjusted[k]<.05]
 result=dict(scope='Exploratory reuse of development data; never independent confirmation',n=800,summary=summaries,holm_13_p=adjusted,passes_numerical_development_screen=eligible,cpu_seconds=time.perf_counter()-start,additional_gpu_seconds=0,final_tasks_evaluated=False,settings_changed=False)
 for name,x in [('CONDITIONAL_MODELS.json',models),('CONDITIONAL_PREDICTIONS.json',records),('CONDITIONAL_RESULTS.json',result)]:(ROOT/name).write_text(json.dumps(x,indent=2),encoding='utf8')
 print(json.dumps(result))
if __name__=='__main__':main()
