"""Post-run read-only independent execution audit; never prediction/gold access."""
import json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def main():
    lock=read('configs/r20_VALUE_ABLATION_PILOT_LOCK.json');sup=read('logs/r20_SUPERVISOR.json');ledger=read('logs/r20_gpu_budget.json');progress=read('results/r20_pilot_PROGRESS.json')
    rows=[json.loads(x) for x in (R/'results/r20_pilot.jsonl').read_text().splitlines()];assert len(rows)==progress['completed']==19
    assert [r['id'] for r in rows]==[r['id'] for r in lock['cohort'][:19]]
    assert sup['status']=='failed' and sup['exit_code']==1 and not sup['overrun'] and not (R/'results/r20_pilot_COMPLETE.json').exists()
    first=rows[0]['candidates'][0];assert all(first['first_candidate_exact_checks'].values()) and len(first['actual_forward_audit'])==5
    cn=dn=0.;unmatched=0;positions=0
    for row in rows:
        for c in row['candidates']:
            assert c['native_main_parity_error']==c['native_N0_parity_error']==c['old_M2_score_and_alpha_parity_error']==0
            assert c['question_hook']['enabled'] and c['demo_hook']['enabled']
            assert set(c['question_hook']['calls_per_physical_layer'])=={4} and set(c['demo_hook']['calls_per_physical_layer'])=={4}
            d=c['metrics'];mask=~np.array(d['unmatched'],dtype=bool);unmatched+=int((~mask).sum());positions+=len(mask)
            cn=max(cn,float(np.abs(np.array(d['matched_norm2'])-d['main_norm2'])[mask].max(initial=0)));dn=max(dn,float(np.abs(np.array(d['shuffle_norm2'])-d['main_norm2']).max()))
    assert cn<=1e-10 and dn<=1e-10
    inherited=sum(read(p)['used_seconds'] for p in lock['budget']['inherited_ledger_sha256']);total=inherited+ledger['used_seconds'];assert ledger['used_seconds']==sup['gpu_worker_seconds'] and abs(total-ledger['total_used_seconds'])<1e-6
    out=dict(status='incomplete_budget_stopped_no_effect_analysis',completed_questions=19,required_questions=24,new_accuracy_computed=False,partial_effects_analyzed=False,eligible_for_feasibility_review=False,reason='Original runtime reserve check stopped before next question; no automatic retry',first_fullV_exact_checks=first['first_candidate_exact_checks'],first_actual_forward_audit=first['actual_forward_audit'],completed_prefix_native_N0_M2_parity_exact=True,completed_prefix_hook_calls_exact=True,completed_positions=positions,completed_C_matched_norm_max_error=cn,completed_D_norm_max_error=dn,unmatched_positions_in_completed_prefix=unmatched,all_parameter_hash_after_unavailable=True,peak_memory_unavailable=True,limitation='Incomplete worker did not reach final all-parameter hash or allocator peak receipt; no full24 scientific conclusion.',worker_seconds=ledger['used_seconds'],aggregate_worker_seconds=total,remaining_worker_seconds=21600-total,protocol_sha256=sha('configs/r20_VALUE_ABLATION_PILOT_LOCK.json'),raw_partial_records_sha256=sha('results/r20_pilot.jsonl'),heldout_accessed=False,external_benchmarks_started=False)
    target=R/'results/r20_EXECUTION_VERIFIED.json'
    if target.exists():assert read('results/r20_EXECUTION_VERIFIED.json')==out
    else:
        with target.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
