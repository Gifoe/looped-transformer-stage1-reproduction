"""Staged analysis: discovery first, bounded heads second, final only after lock."""
import runtime_bootstrap
import json,sys,time,hashlib
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from analysis_core import *
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):
 with (ROOT/'results'/(name+'.jsonl')).open(encoding='utf8') as f:return [json.loads(l) for l in f if l.strip()]
def dump(name,x):
 p=ROOT/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf8')
def assess(rows,extra=None):
 records=[]
 for row in rows:
  pred,act=fixed_policies(row)
  if extra:
   for name,model in extra.items():
    a,values=head_action(row,model);pred[name]=int(view_scores(row)[a].argmax());act[name]=a
  gold=row['gold'];correct={k:v==gold for k,v in pred.items()};correct['oracle']=any(correct[a] for a in ACTIONS)
  records.append(dict(id=row['id'],task=row['task'],gold=gold,predictions=pred,actions=act,correct=correct))
 keys=list(records[0]['correct']);summary={k:paired([r['correct'][k] for r in records],[r['correct']['adaptive'] for r in records]) for k in keys}
 return records,summary
def mechanism(rows):
 out={};scores={a:np.array([view_scores(r)[a].argmax()==r['gold'] for r in rows]) for a in ALL}
 for ref in ['keep','adaptive']:
  out['against_'+ref]={a:paired(scores[a],scores[ref]) for a in ALL}
 masks={format(sum((1<<i) for i,a in enumerate(ACTIONS) if scores[a][j]),'04b'):0 for j in range(len(rows))}
 for j in range(len(rows)):masks[format(sum((1<<i) for i,a in enumerate(ACTIONS) if scores[a][j]),'04b')]+=1
 out['oracle_masks_adaptive_keep_forward_rollback_lsb_first']=masks;out['unreachable_errors']=int((~np.logical_or.reduce([scores[a] for a in ACTIONS])).sum())
 out['rollback_recovers_final_errors']=int((scores['rollback']&~scores['keep']).sum())
 out['earlier_depth_any_recovers_final_errors']=int(((scores['depth1']|scores['depth2']|scores['rollback'])&~scores['keep']).sum())
 return out
def discovery():
 rows=read('discovery');assert len(rows)==400
 rec,summary=assess(rows);signals={}
 # Signed, prespecified proxies: positive = hypothesized action improvement.
 for a in ['keep','forward','rollback']:
  y=np.array([targets(r)[ACTIONS[1:].index(a)] for r in rows]);mask=y!=0
  proxies={'prompt_residual':[r['views']['official']['probe_nll']['adaptive']-r['views']['official']['probe_nll'][a] if r['views']['official']['probe_nll']['adaptive'] is not None else np.nan for r in rows], 'alignment':[r['views']['official']['geometry']['align34']*(1 if a=='forward' else -1 if a=='rollback' else 0) for r in rows]}
  auc={}
  for k,v in proxies.items():
   valid=mask&np.isfinite(v);auc[k]=dict(n=int(valid.sum()),auc=float(roc_auc_score(y[valid]>0,np.array(v)[valid])) if len(set(y[valid]))==2 else None)
  signals[a]=dict(positive=int((y>0).sum()),negative=int((y<0).sum()),neutral=int((y==0).sum()),auc=auc)
 report=dict(stage='discovery_exploratory',n=len(rows),summary=summary,mechanism=mechanism(rows),signed_signal_auc=signals,training_free_first=True,head_next='Bounded head permitted to test whether genuine update supervision adds information; no controller claim from discovery')
 dump('results/discovery_analysis.json',report);dump('results/discovery_predictions.json',rec)
 print(json.dumps(report),flush=True)
def train_head(rows,kind,target):
 start=time.perf_counter();X=np.stack([features(r,kind)[0] for r in rows]);names=features(rows[0],kind)[1];Y=np.stack([targets(r,target) for r in rows]);mean=X.mean(0);sd=X.std(0);sd[sd<1e-8]=1
 model=Ridge(alpha=100).fit((X-mean)/sd,Y);parameters=model.coef_.size+model.intercept_.size;assert parameters<=2048
 return dict(kind=kind,target=target,features=names,mean=mean.tolist(),scale=sd.tolist(),coef=model.coef_.tolist(),intercept=model.intercept_.tolist(),alpha=100,threshold=.05,parameters=parameters,train_seconds=time.perf_counter()-start,n=len(rows))
def head_action(row,m):
 x,names=features(row,m['kind']);assert names==m['features'];values=np.array(m['coef'])@((x-np.array(m['mean']))/np.array(m['scale']))+np.array(m['intercept']);i=int(values.argmax());return (ACTIONS[i+1] if values[i]>.05 else 'adaptive'),values
def train():
 assert (ROOT/'results/discovery_analysis.json').exists()
 assert not (ROOT/'configs/FINAL_LOCK.json').exists(),'Never refit after final policy lock'
 train=read('head_train');val=read('validation');assert len(train)==3200 and len(val)==800
 Y=np.stack([targets(r) for r in train]);events=dict(positive=int((Y>0).sum()),negative=int((Y<0).sum()),neutral=int((Y==0).sum()),by_action={a:dict(positive=int((Y[:,i]>0).sum()),negative=int((Y[:,i]<0).sum()),neutral=int((Y[:,i]==0).sum())) for i,a in enumerate(ACTIONS[1:])})
 models={}
 if min(events['positive'],events['negative'])>=50:
  for kind in ['controls','new','combined']:
   for target in ['accuracy','logweight']:models['head_'+kind+'_'+target]=train_head(train,kind,target)
 dump('configs/heads.json',models)
 rec,summary=assess(val,models);candidates=['proxy','geometry','confidence','support','format_support']+list(models);adj=holm({k:summary[k]['mcnemar_p'] for k in candidates});eligible=[k for k in candidates if summary[k]['net']>=5 and summary[k]['ci95_pp'][0]>0 and adj[k]<.05]
 winner=max(eligible,key=lambda k:summary[k]['net']) if eligible else 'adaptive'
 depth=max(['depth1','depth2','rollback','keep'],key=lambda k:summary[k]['correct'])
 lock=dict(selected_policy=winner,best_fixed_depth=depth,eligible=eligible,adjusted_p=adj,training_events=events,head_count=len(models),validation_n=800,reason='Validation selection gate passed' if eligible else 'No candidate passed the precommitted validation gate; retain Adaptive, final comparisons of other policies remain diagnostic',frozen_files={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'configs/protocol.json',ROOT/'PROTOCOL.md',ROOT/'configs/heads.json',ROOT/'scripts/analysis_core.py',Path(__file__),ROOT/'scripts/score.py',ROOT/'scripts/finalize_analysis.py',ROOT/'scripts/runtime_bootstrap.py',ROOT/'scripts/windows_task_affinity.py']})
 lock.update(frozen_unix=time.time(),training_predictions_sha256=sha(ROOT/'results/head_train.jsonl'),validation_predictions_sha256=sha(ROOT/'results/validation.jsonl'))
 dump('results/validation_analysis.json',dict(summary=summary,adjusted_p=adj,events=events));dump('results/validation_predictions.json',rec);dump('configs/FINAL_LOCK.json',lock);print(json.dumps(lock),flush=True)
if __name__=='__main__':
 {'discovery':discovery,'train':train}[sys.argv[1]]()
