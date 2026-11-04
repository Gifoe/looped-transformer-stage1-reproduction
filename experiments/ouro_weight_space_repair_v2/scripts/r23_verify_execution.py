"""Post-run R23 execution/numerical/accounting inventory, no model or new method."""
import json,numpy as np
from r23_common import ROOT,read,sha,save

def main():
 s=read('configs/r23_SLED_PILOT_LOCK.json');d=read('results/r23_COMPLETE.json');u=read('logs/r23_SUPERVISOR.json');a=read('results/r23_ANALYSIS.json');b=read('logs/r23_gpu_budget.json');rows=[json.loads(x) for x in (ROOT/'results/r23_pilot.jsonl').read_text().splitlines()]
 assert d['status']==u['status']=='complete' and len(rows)==d['n']==24 and not u['overrun'] and not u.get('timed_out',False) and d['all_parameter_hashes_before']==d['all_parameter_hashes_after']
 assert [r['id'] for r in rows]==[r['id'] for r in s['cohort']]
 assert all(sha('scripts/'+p)==h for p,h in s['source_sha256'].items()) and all(sha('scripts/'+p)==h for p,h in s['inherited_source_sha256'].items()) and all(sha(p)==h for p,h in s['budget']['inherited_ledger_sha256'].items())
 checks=[x for r in rows for c in r['candidates'] for x in c['readout_checks']];assert len(checks)==8 and all(x['fullV_exact'] for x in checks)
 assert all(r['original_baseline_vectors_parity_error']==0 and all(c['baseline_parity_error']==c['native2_native3_parity_error']==0 for c in r['candidates']) for r in rows)
 ds=[c['metrics'] for r in rows for c in r['candidates']];flat=lambda k:np.array([x for z in ds for x in z[k]])
 keys=['Q_support','Q_TV','Q_u_P_norm2','c_P_norm2']+[f'N{j}_{k}' for j in range(3) for k in ['effective_support','TV','u_P_norm2']]
 diagnostics={k:dict(mean=float(flat(k).mean()),min=float(flat(k).min()),max=float(flat(k).max())) for k in keys}
 total=sum(read(p)['used_seconds'] for p in s['budget']['inherited_ledger_sha256'])+b['used_seconds'];assert total==b['total_used_seconds']
 out=dict(status='complete24_exact_checks_scientific_gate_failed',protocol_sha256=sha('configs/r23_SLED_PILOT_LOCK.json'),scores_sha256=sha('results/r23_pilot.jsonl'),all_parameter_hashes_equal=True,independent_depth_checks=checks,all_original_baseline_native2_native3_M2_M3_alpha_scores_exact=True,source_and12_old_ledgers_unchanged=True,norm_max_error=a['norm_max_error'],unmatched_positions=a['unmatched_positions'],positions=len(flat('c_zero')),c_zero_positions=int(flat('c_zero').sum()),diagnostics=diagnostics,diagnostic_semantics='degenerate_gradients counts combinedinvalidcos mask; see r23_DIAGNOSTIC_SEMANTICS.md',prediction_disagreements={k:sum(r['primary_prediction']!=r[k+'_prediction'] for r in a['question_results']) for k in ['M3','M2','R6','Fixed','A1','q_only','reference_only','probability_contrast','leaveout_0','leaveout_1','leaveout_2']},counts=a['counts'],paired_R6=a['paired']['R6'],R6_unique_total=a['R6_unique_total'],R6_unique_retained=a['R6_unique_retained'],eligible_for_feasibility_review=False,worker_seconds=b['used_seconds'],aggregate_worker_seconds=total,remaining_worker_seconds=21600-total,peak_allocator_bytes=d['peak_cuda_allocated_bytes'],heldout_accessed=False,external_benchmarks_started=False)
 save('results/r23_EXECUTION_VERIFIED.json',out);print(json.dumps({k:out[k] for k in ['positions','c_zero_positions','aggregate_worker_seconds','remaining_worker_seconds']}))
if __name__=='__main__':main()
