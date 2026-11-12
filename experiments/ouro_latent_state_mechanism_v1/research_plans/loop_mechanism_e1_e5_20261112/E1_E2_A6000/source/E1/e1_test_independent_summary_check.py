"""Recompute train remainder displayed means from completed audited row records."""
import json,gzip,hashlib,math,os
from pathlib import Path
B=Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112');D=B/'E1/test_analysis';S=D/'summary.json';R=D/'rows.jsonl.gz';A=B/'E1/test_cuda_audit.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
s=json.loads(S.read_text());a=json.loads(A.read_text());assert s['status']=='E1_SPLIT_AUDITED_ANALYSIS_COMPLETE' and s['audit_sha256']==sha(A) and s['rows_sha256']==sha(R) and a['status']=='E1_SPLIT_INDEPENDENT_CUDA_SCORE_AND_SOURCE_AUDIT_PASSED'
with gzip.open(R,'rt') as f:rows=[json.loads(x) for x in f]
assert len(rows)==1172 and len({r['id'] for r in rows})==1172 and {r['id'] for r in rows}=={r['id'] for r in a['questions']}
out={}
for g in ('rescue','damage','stable_correct','stable_wrong'):
 rr=[r for r in rows if not r['demo'] and r['natural_group']==g];out[g]={}
 for q in ('1','2'):
  rec=[r['q'][q] for r in rr];n=len(rec);v=lambda k:sum(x[k] for x in rec)/n;summary=s['groups'][g][q];assert summary['N']==n
  for k,source in [('mean_removed','I_removed_allwrong'),('mean_suffix','I_suffix_allwrong'),('mean_terminal_I','I_terminal_allwrong')]:assert math.isclose(summary[k],v(source),rel_tol=1e-13,abs_tol=1e-13)
  for x in rec:assert math.isclose(x['I_terminal_allwrong'],x['I_removed_allwrong']+x['I_suffix_allwrong'],rel_tol=0,abs_tol=1e-10)
  unrelated=sum(x['conditions']['UNRELATED']['mean_loss_vs_ID'] for x in rec)/n;assert math.isclose(summary['mean_unrelated_loss'],unrelated,rel_tol=1e-13,abs_tol=1e-13);out[g][q]={'N':n,'removed':v('I_removed_allwrong'),'unrelated':unrelated,'paired':v('I_removed_allwrong')-unrelated}
out={'status':'E1_TEST_INDEPENDENT_SUMMARY_CHECK_PASSED','summary_sha256':sha(S),'rows_sha256':sha(R),'audit_sha256':sha(A),'script_sha256':sha(Path(__file__)),'scope':'Independent aggregation and score accounting of audited saved row records; no additional model forward or raw-logit rerun.','groups':out};p=D/'independent_summary_check.json'
with p.open('x') as f:json.dump(out,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(out,sort_keys=True));print('SHA',sha(p))
