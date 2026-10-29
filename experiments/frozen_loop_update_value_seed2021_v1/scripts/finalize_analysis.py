"""Locked final analysis; diagnostics cannot change the selected policy."""
import runtime_bootstrap
import json,time,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from analysis_core import *
from analyze import ROOT,read,dump,assess,mechanism,head_action,sha
def head_diagnostics(rows,models):
 result={};Y=np.stack([targets(r) for r in rows]);margins=np.array([margin(view_scores(r)['adaptive']) for r in rows])
 for name,m in models.items():
  begin=time.perf_counter();values=np.stack([head_action(r,m)[1] for r in rows]);seconds=time.perf_counter()-begin
  per={'_inference':dict(cpu_seconds=seconds,milliseconds_per_question=1000*seconds/len(rows),includes='Answer-free feature construction, scaling and head matrix multiply; no GPU backbone passes')}
  for i,a in enumerate(ACTIONS[1:]):
   y=Y[:,i];mask=y!=0
   auc=float(roc_auc_score(y[mask]>0,values[mask,i])) if len(set(y[mask]))==2 else None
   groups=[]
   for low,high in zip([0,.25,.5,.75],[.25,.5,.75,1.01]):
    lo,hi=np.quantile(margins,[low,min(high,1)]);mm=mask&(margins>=lo)&((margins<hi) if high<1 else (margins<=hi))
    groups.append(dict(n=int(mm.sum()),auc=float(roc_auc_score(y[mm]>0,values[mm,i])) if len(set(y[mm]))==2 else None))
   mse=float(np.mean((values[:,i]-y)**2)) if m['target']=='accuracy' else None
   per[a]=dict(positive=int((y>0).sum()),negative=int((y<0).sum()),neutral=int((y==0).sum()),auc=auc,confidence_quartiles=groups,accuracy_gain_mse=mse,target=m['target'])
  result[name]=per
 return result
def correction_accounting(records,policy):
 keep=np.array([r['correct']['keep'] for r in records]);base=np.array([r['correct']['adaptive'] for r in records]);out=np.array([r['correct'][policy] for r in records]);benefit=base&~keep;harm=~base&keep
 return dict(beneficial_corrections=int(benefit.sum()),beneficial_retained=int((benefit&out).sum()),beneficial_rejected=int((benefit&~out).sum()),harmful_updates=int(harm.sum()),harmful_avoided=int((harm&out).sum()),harmful_accepted=int((harm&~out).sum()))
def random_control(records,policy):
 acts=np.array([r['actions'].get(policy,policy) for r in records]);assert all(a in ACTIONS for a in acts)
 rng=np.random.default_rng(2021);scores=np.array([[r['correct'][a] for a in ACTIONS] for r in records]);idx=np.array([ACTIONS.index(a) for a in acts]);observed=sum(r['correct'][policy] for r in records);random_counts=np.array([scores[np.arange(len(idx)),rng.permutation(idx)].sum() for _ in range(1000)])
 return dict(action_counts={a:int((acts==a).sum()) for a in ACTIONS},random_correct_mean=float(random_counts.mean()),random_correct_interval95=np.quantile(random_counts,[.025,.975]).tolist(),observed_correct=int(observed),randomization_p_ge=(1+int((random_counts>=observed).sum()))/1001)
def main():
 lock=json.loads((ROOT/'configs/FINAL_LOCK.json').read_text());models=json.loads((ROOT/'configs/heads.json').read_text())
 for p,h in lock['frozen_files'].items():assert sha(ROOT/p)==h,('Locked file changed',p)
 selected=lock['selected_policy'];reports={};primary={};allpred=[]
 for split in ['final_sciq','final_piqa']:
  rows=read(split);audit=json.loads((ROOT/'data/manifest_audit.json').read_text());assert len(rows)==audit['manifests'][split]['n']
  ids=[r['id'] for r in rows];assert ids==audit['manifests'][split]['ids'] and len(set(ids))==len(ids)
  begin=time.perf_counter();records,summary=assess(rows,models);cpu=time.perf_counter()-begin
  baselines={}
  for base in ['adaptive','ensemble',lock['best_fixed_depth'],lock['best_fixed_action']]:baselines[base]=paired([r['correct'][selected] for r in records],[r['correct'][base] for r in records])
  for base in ['adaptive','ensemble',lock['best_fixed_action']]:primary[split+'/'+base]=baselines[base]['mcnemar_p']
  compare={k:correction_accounting(records,k) for k in summary if k!='oracle'}
  randoms={k:random_control(records,k) for k in [selected,'proxy','geometry','confidence','support','format_support']+list(models)}
  raw={a:float(np.mean([view_scores(r,normalized=False)[a].argmax()==r['gold'] for r in rows]))*100 for a in ALL}
  report=dict(n=len(rows),summary=summary,raw_accuracy_percent=raw,selected_policy=selected,selected_vs_baselines=baselines,mechanism=mechanism(rows),correction_accounting=compare,random_controls=randoms,head_diagnostics=head_diagnostics(rows,models),all_policy_holm_p=holm({k:v['mcnemar_p'] for k,v in summary.items() if k not in ALL+['oracle']}),cost=dict(all_policy_analysis_cpu_seconds=cpu,official_view_total_seconds=sum(r['views']['official']['elapsed_s'] for r in rows),format_view_total_seconds=sum(r['views']['format']['elapsed_s'] for r in rows),scope='View times include all depth readouts and probes; historical isolated LoopCD timing is not substituted for matched timing'))
  if split=='final_sciq':
   supp=[view_scores(r)['adaptive'].argmax()==r['gold'] for r in rows];no=[view_scores(r,'no_support')['adaptive'].argmax()==r['gold'] for r in rows]
   report['support_evidence_diagnostic']=dict(with_support_vs_without=paired(supp,no),no_support_accuracy_percent=100*float(np.mean(no)),third_view_seconds=sum(r['views']['no_support']['elapsed_s'] for r in rows),interpretation='Evidence-removal diagnostic on the same questions; not an independent verifier and not CoDR')
  reports[split]=report;dump('results/'+split+'_predictions.json',records);allpred+=records
 adj=holm(primary)
 success=selected!='adaptive' and all(reports[s]['selected_vs_baselines'][b]['ci95_pp'][0]>0 and adj[s+'/'+b]<.05 for s in reports for b in ['adaptive','ensemble',lock['best_fixed_action']])
 outcome=dict(selected_policy=selected,primary_adjusted_p=adj,success=success,decision='Proceed to separately frozen Qwen transfer' if success else 'No-Go for a method-paper claim from this round; no reliable controller superiority under the locked criteria',reports=reports,validation_head_diagnostics=head_diagnostics(read('validation'),models),training_head_parameters={k:v['parameters'] for k,v in models.items()},training_cpu_seconds=sum(v['train_seconds'] for v in models.values()))
 dump('results/FINAL_ANALYSIS.json',outcome);print(json.dumps(dict(selected_policy=selected,success=success,decision=outcome['decision'],accuracy={s:{k:round(v['accuracy_percent'],3) for k,v in r['summary'].items()} for s,r in reports.items()})),flush=True)
if __name__=='__main__':main()
