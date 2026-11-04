"""Post-run read-only E2 complete/parity/accounting replay; no model execution."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def main():
 s=read('configs/r20_e2_VALUE_ABLATION_PILOT_LOCK.json');d=read('results/r20_e2_pilot_COMPLETE.json');u=read('logs/r20_e2_SUPERVISOR.json');b=read('logs/r20_e2_gpu_budget.json');a=read('results/r20_e2_PILOT_ANALYSIS.json')
 assert d['status']==u['status']=='complete' and d['n']==24 and u['exit_code']==0 and not u['overrun'] and not u.get('timed_out',False)
 assert d['all_parameter_hashes_before']==d['all_parameter_hashes_after'] and not d['weights_modified']
 assert all(sha('scripts/'+p)==v for p,v in s['source_sha256'].items()) and all(sha('scripts/'+p)==v for p,v in s['inherited_source_sha256'].items())
 assert all(sha(p)==v for p,v in s['budget']['inherited_ledger_sha256'].items())
 hashes={'V0':sha('results/r20_pilot.jsonl'),'E1':sha('results/r20_e1_pilot.jsonl'),'E2':sha('results/r20_e2_pilot.jsonl')}
 for tag in ['V0','E1']:
  p=read('results/r20_e2_OVERLAP_'+tag+'.json');assert p['status']=='passed_strict_zero' and not p['failures'] and p['all_score_sha256']==hashes and all(v['max_abs_diff']==0 for v in p['per_field'].values())
 rows=[json.loads(x) for x in (R/'results/r20_e2_pilot.jsonl').read_text().splitlines()];assert [r['id'] for r in rows]==[c['id'] for c in s['cohort']]
 first=rows[0]['candidates'][0];assert all(first['first_candidate_exact_checks'].values()) and len(first['actual_forward_audit'])==5
 assert all(c['native_main_parity_error']==c['native_N0_parity_error']==c['old_M2_score_and_alpha_parity_error']==0 for r in rows for c in r['candidates'])
 total=sum(read(p)['used_seconds'] for p in s['budget']['inherited_ledger_sha256'])+b['used_seconds'];assert total==b['total_used_seconds'] and b['used_seconds']==u['gpu_worker_seconds']
 out=dict(status='complete24_double_overlap_exact_scientific_gate_failed',source_and_old11_ledgers_unchanged=True,complete24=True,double_overlap_strict_zero=True,all_parameter_hashes_equal=True,first_fullV_parity_and_actual_HF_audit_passed=True,all_native_N0_M2_parity_exact=True,counts=a['counts'],paired_vs_R6=a['paired']['R6'],R6_unique_total=a['R6_unique_total'],R6_unique_retained=a['R6_unique_retained'],eligible_for_feasibility_review=False,attempt_worker_seconds={'v0':read('logs/r20_gpu_budget.json')['used_seconds'],'E1':read('logs/r20_e1_gpu_budget.json')['used_seconds'],'E2':b['used_seconds']},aggregate_worker_seconds=total,remaining_worker_seconds=21600-total,peak_allocator_bytes=d['peak_cuda_allocated_bytes'],memory_scope=d['memory_scope'],protocol_sha256=sha('configs/r20_e2_VALUE_ABLATION_PILOT_LOCK.json'),score_sha256=hashes,heldout_accessed=False,external_benchmarks_started=False)
 p=R/'results/r20_e2_EXECUTION_VERIFIED.json'
 if p.exists():assert read(str(p.relative_to(R)))==out
 else:
  with p.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
