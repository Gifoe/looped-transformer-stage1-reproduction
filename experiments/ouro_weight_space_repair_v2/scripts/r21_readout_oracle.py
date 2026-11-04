"""Complete576 all-label H6/H7 CPU oracle. Not a model or deployed selector."""
import os,time,json
import numpy as np
from r21_common import ROOT,LOCK,sha,read,save,gate
from r21_lp import solve_label,TAU

def rows(path):return [json.loads(x) for x in (ROOT/path).read_text(encoding='utf8').splitlines()]
def main():
 s=gate();cohort=[];input_audit=[]
 for split,n in [('discovery',384),('selection',192)]:
  r2=rows('results/slcd_r2_'+split+'_development.jsonl');r7=rows('results/r7_'+split+'_development.jsonl');manifest=rows('data/slcd_'+split+'.jsonl');assert len(r2)==len(r7)==len(manifest)==n
  receipts=[]
  for prefix,records in [('slcd_r2_',r2),('r7_',r7)]:
   path='results/'+prefix+split+'_development';done=read(path+'_COMPLETE.json');assert done['status']=='complete' and done['n']==n and done['score_file_sha256']==sha(path+'.jsonl') and done['provenance']['data_sha256']==sha('data/slcd_'+split+'.jsonl')
   assert all(r['signature']==done['signature'] for r in records);receipts.append(done)
  a={r['id']:r for r in r2};b={r['id']:r for r in r7};assert set(a)==set(b)=={r['id'] for r in manifest}
  for item in manifest:
   x=a[item['id']];y=b[item['id']];assert x['gold']==y['gold']==item['gold'] and x['prompt_sha256']==y['prompt_sha256']==item['prompt_sha256']
   assert x['choice_char_lengths']==y['choice_char_lengths']==item['choice_char_lengths'] and all(v>0 for v in x['choice_char_lengths'])
   assert x['choice_token_counts']==y['token_counts'] and len(item['request_pairs'])==y['candidate_count']==len(x['choice_char_lengths'])
   for k in range(1,7):
    value=x['original_methods']['native_k'+str(k)];assert np.array_equal(np.asarray(value['sum_scores'],dtype=np.float64)/np.asarray(item['choice_char_lengths'],dtype=np.float64),value['normalized_scores'])
   for k in [1,4,5]:
    for field in ['sum_scores','normalized_scores']:assert x['original_methods']['native_k'+str(k)][field]==y['original_methods']['native_k'+str(k)][field]
   S=np.array([x['original_methods']['native_k'+str(k)]['normalized_scores'] for k in range(1,7)],dtype=np.float64).T;assert S.shape==(len(item['request_pairs']),6)
   assert all(pair[0]==item['prompt'] for pair in item['request_pairs'])
   assert __import__('hashlib').sha256(item['prompt'].encode()).hexdigest()==item['prompt_sha256']
   input_audit.append(dict(id=item['id'],request_pairs_sha256=__import__('hashlib').sha256(json.dumps(item['request_pairs'],ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),candidate_order_evidence='Both split COMPLETE dataSHA and scoreSHA signatures verified; frozen R2 worker enumerates manifest pairs; candidate index/target IDs asserted against cached states. Frozen R7 inference enumerates same manifest and asserts state target IDs/index. Shared k1/k4/k5 full vectors exact. No per-candidate request hash exists in historical score rows.',six_depth_sum_divide_original_chars_exact=True,shared_depth_vectors_exact=True))
   cohort.append((split,item['id'],x['gold'],S,np.asarray(y['baseline_scores']['M0_R6'],dtype=np.float64)))
 assert len(cohort)==576 and len({x[1] for x in cohort})==576
 save('results/r21_INPUT_AUDIT.json',dict(status='passed_source_bound_candidate_order_audit',n=576,rows=input_audit,provenance_source_sha256={p:sha('scripts/'+p) for p in ['slcd_r2_worker.py','r7_worker.py','r7_inference.py','runtime.py']}))
 output=[]
 with (ROOT/'results/r21_oracle.jsonl').open('x',encoding='utf8') as f:
  for split,qid,gold,S,r6 in cohort:
   assert time.monotonic()<float(os.environ['R21_CPU_DEADLINE_MONOTONIC'])
   h6=[solve_label(S,j) for j in range(len(S))];h7=[solve_label(np.column_stack([S,r6]),j) for j in range(len(S))]
   for x,y in zip(h6,h7):
    assert y['original_margin_upper']>=x['original_margin_lower'],'H7 cannot be smaller than H6'
   discrete=any(int(np.argmax(S[:,k]))==gold for k in range(6));strict_native=any(np.min(S[gold,k]-np.delete(S[:,k],gold))>0 for k in range(6))
   native_normalized_margin=max(float(np.min(S[gold,k]-np.delete(S[:,k],gold))) for k in range(6))/h6[gold]['scale']
   if native_normalized_margin>TAU+h6[gold]['roundoff_delta']:assert h6[gold]['status']=='strict_reachable','Certified strict native winner failed hull certification'
   category='discrete_reachable' if discrete else 'only_convex_reachable' if h6[gold]['status']=='strict_reachable' else 'hull_unreachable' if h6[gold]['status']=='strict_unreachable' else 'numerically_undetermined'
   row=dict(id=qid,split=split,gold=gold,R6_correct=int(np.argmax(r6))==gold,category=category,discrete_reachable=discrete,native_strict_winner=strict_native,tie_only=discrete and not strict_native,H6=h6,H7=h7,H6_reachable_label_count=sum(x['status']=='strict_reachable' for x in h6),H7_reachable_label_count=sum(x['status']=='strict_reachable' for x in h7))
   f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno());output.append(row)
 summaries={}
 for split in ['discovery','selection','combined']:
  subset=[r for r in output if split=='combined' or r['split']==split];by={}
  for group in ['all','R6_correct','R6_wrong']:
   rs=[r for r in subset if group=='all' or r['R6_correct']==(group=='R6_correct')];n=len(rs);strict=sum(r['H6'][r['gold']]['status']=='strict_reachable' for r in rs)
   categories={k:sum(r['category']==k for r in rs) for k in ['discrete_reachable','only_convex_reachable','hull_unreachable','numerically_undetermined']};cross={}
   for r in rs:
    key=r['H6'][r['gold']]['status']+' / '+r['H7'][r['gold']]['status'];cross[key]=cross.get(key,0)+1
   by[group]=dict(n=n,categories=categories,category_proportions={k:v/n if n else None for k,v in categories.items()},H6_strict_gold_reachable=strict,H6_strict_proportion=strict/n if n else None,tie_only=sum(r['tie_only'] for r in rs),H6_H7_gold_cross_table=cross)
  summaries[split]=by
 u=summaries['combined']['all']['categories']['numerically_undetermined'];passed=summaries['combined']['R6_wrong']['H6_strict_proportion']>=.2 and all(summaries[k]['R6_wrong']['H6_strict_proportion']>=.1 for k in ['discovery','selection']) and u/576<=.01
 save('results/r21_oracle_COMPLETE.json',dict(status='complete_CPU_oracle',n=576,protocol_sha256=sha(LOCK),raw_sha256=sha('results/r21_oracle.jsonl'),summaries=summaries,capacity_diagnostic_gate_passed=passed,GPU_worker_seconds=0,oracle_not_deployable=True,heldout_accessed=False,interpretation='Finite score-hull capacity only; no semantic selector or hidden-knowledge conclusion; H7 separate upperbound not a method',uncertain_count=u))
 print('R21_CPU_ORACLE_COMPLETE',576)
if __name__=='__main__':main()
