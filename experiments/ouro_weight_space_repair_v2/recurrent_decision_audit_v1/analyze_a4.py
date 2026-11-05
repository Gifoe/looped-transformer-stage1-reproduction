"""Complete-clean numeric validation precedes development labels and predictions."""
from pathlib import Path
import json,hashlib,sys,csv
import numpy as np
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
H=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def validate_rows(plan,rows):
 assert len(rows)==len(plan['cohort64'])==64 and [x['id'] for x in rows]==[x['id'] for x in plan['cohort64']]
 for q,p in zip(rows,plan['cohort64']):
  assert len(q['rotations'])==len(p['rotations'])==p['nchoices']
  for r,t in zip(q['rotations'],p['rotations']):
   assert r['shift']==t['shift'] and r['order']==t['order'] and r['task_keys']==[x['key'] for x in t['tasks']]
   assert set(r['native_sum_scores'])=={'1','2','3','4'}
   for z in r['native_sum_scores'].values():assert len(z)==p['nchoices'] and np.isfinite(z).all() and np.all(np.asarray(z)<=0)
   assert r['native_sum_scores']['1']==t['expected_K1'] and r['native_sum_scores']['4']==t['expected_K4'] and r['fixed_sum_scores']==t['expected_Fixed']
 return True

def main():
 assert not (OUT/'a4_ANALYSIS.json').exists();s=read(OUT/'a4_SUPERVISOR.json');b=read(OUT/'a4_gpu_budget.json');d=read(OUT/'a4_COMPLETE.json');plan=read(OUT/'a4_CPU_TASK_PLAN.json');lock=read(OUT/'a4_EXECUTION_LOCK.json')
 assert s['status']=='complete' and s['exit_code']==0 and not s['timed_out'] and not s['overrun'] and 0<s['gpu_worker_seconds']<=500 and s['old_ledgers_unchanged']
 assert b['used_seconds']==s['gpu_worker_seconds'] and b['total_used_seconds']==20834.389865819365+b['used_seconds']<=21600
 assert d['cleanup_exact'] and d['parameter_hashes_before']==d['parameter_hashes_after'] and len(d['parameter_hashes_before'])==269
 assert d['protocol_sha256']==H((OUT/'a4_EXECUTION_LOCK.json').read_bytes())==s['protocol_sha256'] and d['plan_sha256']==H((OUT/'a4_CPU_TASK_PLAN.json').read_bytes())
 assert H((OUT/'a4_CPU_TASK_PLAN.json').read_bytes())==lock['plan_sha256']
 for rel,h in plan['source_inputs'].items():assert H((ROOT/rel).read_bytes())==h
 for rel,h in lock['source_sha256'].items():assert H((OUT/rel).read_bytes())==h
 assert d['vector_npz_sha256']==H((OUT/'a4_NATIVE_VECTORS.npz').read_bytes());index=read(OUT/'a4_VECTOR_INDEX.json');z=np.load(OUT/'a4_NATIVE_VECTORS.npz',allow_pickle=False);assert set(z.files)==set(index)
 for key,meta in index.items():
  a=z[key];assert list(a.shape)==meta['shape'];assert H((meta['dtype']+str(a.shape)).encode()+a.tobytes())==key
  if meta['dtype']=='BF16':assert a.dtype==np.uint16 and a.shape==(1,1,2048);values=(a.astype(np.uint32)<<16).view(np.float32)
  else:assert meta['dtype']=='FP32' and a.dtype==np.float32 and a.shape==(1,49152);values=a
  assert np.isfinite(values).all()
 parity=read(OUT/'a4_PARITY.json');assert parity['status']=='all_exact' and len(parity['independent_checks'])==16
 expected={(q['id'],r['shift'],j,k) for q in plan['parity'] for r in q['rotations'] for j in range(len(r['tasks'])) for k in [2,3]};assert {(p['id'],p['shift'],p['task'],p['depth']) for p in parity['independent_checks']}==expected
 for p in parity['independent_checks']:assert p['state_key']==p['independent_state_key'] in index and p['head_key']==p['independent_head_key'] in index and p['exact']
 if d['status']!='complete64':
  assert d['status']=='parity_only_runtime_not_admitted' and not d['completed_ids'] and d['body_calls']==24
  (OUT/'a4_ANALYSIS.json').write_text(json.dumps(dict(status='parity_only_no_development_effects',GPU_worker_seconds=b['used_seconds']),indent=2)+'\n');return
 rows=[json.loads(x) for x in (OUT/'a4_RAW.jsonl').read_text(encoding='utf8').splitlines()];assert H((OUT/'a4_RAW.jsonl').read_bytes())==d['raw_sha256'];validate_rows(plan,rows)
 assert d['body_calls']==279 and d['head_calls']==1092 and d['completed_ids']==plan['selection']['ids']
 for q in rows:
  for r in q['rotations']:
   for keys in r['state_keys'].values():assert len(keys)==4 and all(k in index for k in keys)
 # All numerical/lifecycle checks passed. Historical development labels first opened here.
 authority={}
 for split in ['discovery','selection']:
  for x in (ROOT/('data/slcd_'+split+'.jsonl')).read_text(encoding='utf8').splitlines():
   v=json.loads(x);authority[v['id']]=v['gold']
 predrows=[];trans=[];stability=[];correct={};qsummary=[]
 for q in rows:
  gold=authority[q['id']]
  for family in ['joint','cyclic_mean']:
   pred=[];marg=[]
   for k in range(1,5):
    matrix=np.asarray([r['native_sum_scores'][str(k)] for r in q['rotations']]);v=matrix[0] if family=='joint' else matrix.mean(0);p=int(v.argmax());pred.append(p);marg.append(float(v[gold]-np.delete(v,gold).max()));correct.setdefault((family,k),[]).append(p==gold)
   firstcorrect=next((k+1 for k,p in enumerate(pred) if p==gold),None);firstchange=next((k+1 for k in range(1,4) if pred[k]!=pred[k-1]),None);stable=next((k+1 for k in range(1,4) if pred[k]==pred[k-1]),None);laterchanges=sum(pred[k]!=pred[k-1] for k in range(stable,4)) if stable else None
   persistent=next((j+1 for j in range(3) if len(set(pred[j:]))==1),None);allchanges=sum(pred[k]!=pred[k-1] for k in range(1,4));returns=sum(pred[k]!=pred[k-1] and pred[k] in pred[:k-1] for k in range(2,4))
   qsummary.append(dict(stable_start_K=stable-1 if stable else None,persistent_stable_suffix_start_K=persistent,all_answer_changes=allchanges,return_to_prior_answer_count=returns,id=q['id'],family=family,predictions=pred,gold=gold,first_correct_K=firstcorrect,first_change_K=firstchange,first_two_consecutive_stability_confirmation_K=stable,later_answer_changes_after_confirmation=laterchanges))
   for k,p in enumerate(pred,1):predrows.append(dict(id=q['id'],family=family,K=k,prediction=p,gold=gold,correct=p==gold,gold_margin=marg[k-1],first_correct_K=firstcorrect,first_change_K=firstchange,stability_confirmation_K=stable,later_answer_changes=laterchanges))
   for k in range(1,4):trans.append(dict(id=q['id'],family=family,K_from=k,K_to=k+1,rescue=pred[k-1]!=gold and pred[k]==gold,damage=pred[k-1]==gold and pred[k]!=gold,changed=pred[k-1]!=pred[k]))
  for k in range(1,5):
   matrix=np.asarray([r['native_sum_scores'][str(k)] for r in q['rotations']]);pred=[r['order'][int(matrix[i,r['order']].argmax())] for i,r in enumerate(q['rotations'])];n=len(pred);stability.append(dict(id=q['id'],K=k,unanimous=len(set(pred))==1,pairwise_content_agreement=sum(pred[i]==pred[j] for i in range(n) for j in range(i+1,n))/(n*(n-1)/2),score_variance=float(matrix.var(0).mean())))
 def csvout(name,rows):
  with (OUT/name).open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 csvout('development64_loopwise_predictions.csv',predrows);csvout('development64_rescue_damage.csv',trans);csvout('development64_permutation_stability.csv',stability)
 result=dict(status='complete64_clean_numeric_then_descriptive_labels',questions=64,readout_K=[1,2,3,4],correct={f'{f}_K{k}':sum(x) for (f,k),x in correct.items()},transitions=[dict(family=f,K_from=k,rescue=sum(x['rescue'] for x in trans if x['family']==f and x['K_from']==k),damage=sum(x['damage'] for x in trans if x['family']==f and x['K_from']==k)) for f in ['joint','cyclic_mean'] for k in [1,2,3]],trajectories=qsummary,GPU_worker_seconds=b['used_seconds'],scope='Fixed seed64 historical training/dev subset; not full test intermediate trajectory and not a new decoding method')
 (OUT/'a4_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='trajectories'},indent=2))
if __name__=='__main__':main()
