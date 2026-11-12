"""Bind three independently audited cohorts to full-results index."""
import json,hashlib,datetime,os
from pathlib import Path
B=Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_eight_corner_20261112')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
if not __debug__:raise RuntimeError('optimized Python forbidden')
f=B/'ROUND1_FULL_RESULTS.json';full=json.loads(f.read_text());N={f'G{i}{j}{k}' for i in (1,2) for j in (1,2) for k in (1,2)};idx={(q['split'],q['id']):q for q in full['questions']};assert len(idx)==len(full['questions'])==2590 and full['status']=='AUDITED_FULL_COHORT_DATA_COMPLETE';splitsha={};total=0
for split,count in [('train',1119),('validation',299),('test',1172)]:
 cp=B/(split+'_independent_cuda_coverage.json');sp=B/(split+'_analysis/summary.json');rp=B/split/'receipt.json';c=json.loads(cp.read_text());s=json.loads(sp.read_text());r=json.loads(rp.read_text());exp='TRAIN_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED' if split=='train' else 'SPLIT_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED';assert c['status']==exp and c['count']==s['count']==r['count']==count and r['split']==split and c['collector_receipt_sha256']==s['collector_receipt_sha256']==sha(rp) and s['coverage_sha256']==sha(cp) and len(c['questions'])==len({q['id'] for q in c['questions']})==count
 for q in c['questions']:
  z=idx[(split,q['id'])];assert z['source_ledger']==q and set(z['scores'])==set(z['target_logits'])==set(z['predictions'])==set(z['top_ties'])==N and all(len(v)==len(z['targets']) for v in z['scores'].values())
 splitsha[split]={'count':count,'independent_cuda_coverage_sha256':sha(cp),'summary_sha256':sha(sp),'collector_receipt_sha256':sha(rp),'rows_sha256':s['rows_sha256'],'audit_script_sha256':c['script_sha256'],'coverage_status':c['status']};total+=count
assert total==2590
out={'status':'ALL_SPLITS_AUDIT_LINKS_VERIFIED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'count':total,'split_evidence':splitsha,'full_results_sha256':sha(f),'runtime_sha256':sha(B/'ROUND1_RUNTIME_AND_MEMORY.json'),'factorial_report_sha256':sha(B/'ROUND1_FACTORIAL_ANALYSIS.md'),'counterexamples_sha256':sha(B/'ROUND1_COUNTEREXAMPLES.md'),'forecast_sha256':sha(B/'ROUND1_TRAIN_INFORMED_PROSPECTIVE_CHECKS.md'),'scope':'Links three independent CUDA logit-score replays and source audits including bitwise old four-corner checks; intervention execution itself is evidenced by frozen collector and receipts. Missing natural source tensors were not reread; their hashes are historical links to earlier coverage, with branch R anchor read and hashed in this round.'}
p=B/'ROUND1_COMBINED_AUDIT_LINK.json'
with p.open('x') as g:json.dump(out,g,sort_keys=True);g.write('\n');g.flush();os.fsync(g.fileno())
print('combined',sha(p),flush=True)
