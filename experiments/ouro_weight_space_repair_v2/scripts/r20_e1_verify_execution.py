"""E1 terminal execution audit only; never read candidate numeric contents."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads((R/p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
def main():
    spec=read('configs/r20_e1_VALUE_ABLATION_PILOT_LOCK.json');sup=read('logs/r20_e1_SUPERVISOR.json');book=read('logs/r20_e1_gpu_budget.json');progress=read('results/r20_e1_pilot_PROGRESS.json')
    assert sup['status']=='failed' and sup['exit_code']==1 and not sup['overrun'] and progress['completed']==23 and progress['target']==24
    assert not any((R/p).exists() for p in ['results/r20_e1_pilot_COMPLETE.json','results/r20_e1_OVERLAP_PARITY.json','results/r20_e1_PILOT_ANALYSIS.json'])
    assert len((R/'results/r20_e1_pilot.jsonl').read_bytes().splitlines())==23
    assert all(sha('scripts/'+p)==v for p,v in spec['source_sha256'].items()) and all(sha('scripts/'+p)==v for p,v in spec['inherited_source_sha256'].items())
    assert all(sha(p)==v for p,v in spec['budget']['inherited_ledger_sha256'].items()) and sha('results/r20_pilot.jsonl')==spec['v0_partial_sha256']
    inherited=sum(read(p)['used_seconds'] for p in spec['budget']['inherited_ledger_sha256']);total=inherited+book['used_seconds'];assert book['used_seconds']==sup['gpu_worker_seconds'] and total==book['total_used_seconds']
    out=dict(status='incomplete23of24_stop_no_overlap_or_effects',completed_questions=23,required_questions=24,reason='Original runtime reserve5 check before final question',new_accuracy_computed=False,partial_effects_analyzed=False,overlap_program_run=False,effect_analyzer_run=False,eligible_for_feasibility_review=False,all_parameter_hash_after_unavailable=True,allocator_peak_receipt_unavailable=True,source_and_all10_old_ledgers_unchanged=True,v0_partial_bytes_unchanged=True,worker_seconds=book['used_seconds'],aggregate_worker_seconds=total,remaining_worker_seconds=21600-total,protocol_sha256=sha('configs/r20_e1_VALUE_ABLATION_PILOT_LOCK.json'),e1_partial_sha256=sha('results/r20_e1_pilot.jsonl'),scope='Read terminal metadata, file hashes and raw line count only; no candidate numeric contents, no old/new overlap, no labels/predictions',heldout_accessed=False,external_benchmarks_started=False)
    p=R/'results/r20_e1_EXECUTION_VERIFIED.json'
    if p.exists():assert read(str(p.relative_to(R)))==out
    else:
        with p.open('x',encoding='utf8') as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
