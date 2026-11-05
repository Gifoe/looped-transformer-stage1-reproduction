"""Independent CPU-only A4 lifecycle/state-byte review. No parity gold opened."""
from pathlib import Path
import collections
import hashlib
import json

import numpy as np

P=Path(__file__).resolve().parent
ROOT=P.parent
H=lambda b:hashlib.sha256(b).hexdigest()
read=lambda name:json.loads((P/name).read_text(encoding='utf8'))
lock=read('a4_EXECUTION_LOCK.json')
plan=read('a4_CPU_TASK_PLAN.json')
sup=read('a4_SUPERVISOR.json')
ledger=read('a4_gpu_budget.json')
complete=read('a4_COMPLETE.json')
parity=read('a4_PARITY.json')
admission=read('a4_RUNTIME_ADMISSION.json')
timings=read('a4_TIMINGS.json')
index=read('a4_VECTOR_INDEX.json')
analysis=read('a4_ANALYSIS.json')
locksha=H((P/'a4_EXECUTION_LOCK.json').read_bytes())
assert locksha==complete['protocol_sha256']==sup['protocol_sha256']=='814e68b2286944bf774d37ed61c6e43f4d901ca4767c81df32c0e89a5478042c'
assert H((P/'a4_CPU_TASK_PLAN.json').read_bytes())==lock['plan_sha256']==complete['plan_sha256']
for name,h in lock['source_sha256'].items():assert H((P/name).read_bytes())==h,name
for name,h in lock['input_sha256'].items():assert H((ROOT/name).read_bytes())==h,name
total=0.
for name,h in lock['inherited_ledgers'].items():
    raw=(ROOT/name).read_bytes()
    assert H(raw)==h
    total+=json.loads(raw)['used_seconds']
assert len(lock['inherited_ledgers'])==19 and total==20834.389865819365
assert sup['status']=='complete' and sup['exit_code']==0 and not sup['timed_out'] and not sup['overrun']
assert sup['old_ledgers_unchanged'] and 0<sup['gpu_worker_seconds']<500
assert ledger['used_seconds']==sup['gpu_worker_seconds']
assert ledger['total_used_seconds']==total+ledger['used_seconds']<21600
assert ledger['remaining_seconds']==21600-ledger['total_used_seconds']
assert complete['parameter_hashes_before']==complete['parameter_hashes_after'] and len(complete['parameter_hashes_before'])==269 and complete['cleanup_exact']
assert complete['status']=='parity_only_runtime_not_admitted' and complete['completed_ids']==[]
assert complete['body_calls']==24 and complete['head_calls']==72 and len(timings)==24
depth_counts=collections.Counter(x['depth'] for x in timings)
assert depth_counts=={2:8,3:8,4:8}
assert not (P/'a4_RAW.jsonl').exists() and not admission['admitted']
assert analysis['status']=='parity_only_no_development_effects'

npzraw=(P/'a4_NATIVE_VECTORS.npz').read_bytes()
assert H(npzraw)==complete['vector_npz_sha256']
vectors=np.load(P/'a4_NATIVE_VECTORS.npz',allow_pickle=False)
assert set(vectors.files)==set(index)
for key,meta in index.items():
    a=vectors[key]
    assert list(a.shape)==meta['shape']
    assert H((meta['dtype']+str(a.shape)).encode()+a.tobytes())==key
    if meta['dtype']=='BF16':
        assert a.dtype==np.uint16 and a.shape==(1,1,2048)
        values=(a.astype(np.uint32)<<16).view(np.float32)
    else:
        assert meta['dtype']=='FP32' and a.dtype==np.float32 and a.shape==(1,49152)
        assert ((a.view(np.uint32)&0xffff)==0).all(), 'Head payload not exact BF16-to-FP32'
        values=a
    assert np.isfinite(values).all()

expected={(q['id'],r['shift'],j,k) for q in plan['parity'] for r in q['rotations'] for j in range(len(r['tasks'])) for k in (2,3)}
observed={(x['id'],x['shift'],x['task'],x['depth']) for x in parity['independent_checks']}
assert parity['status']=='all_exact' and expected==observed and len(observed)==16
for x in parity['independent_checks']:
    assert x['exact'] and x['state_key']==x['independent_state_key'] in index
    assert x['head_key']==x['independent_head_key'] in index
for q,p in zip(parity['questions'],plan['parity']):
    assert q['id']==p['id'] and len(q['rotations'])==len(p['rotations'])==4
    for r,t in zip(q['rotations'],p['rotations']):
        assert r['order']==t['order'] and r['shift']==t['shift']
        assert set(r['native_sum_scores'])=={'1','2','3','4'}
        assert r['native_sum_scores']['1']==t['expected_K1']
        assert r['native_sum_scores']['4']==t['expected_K4']
        assert r['fixed_sum_scores']==t['expected_Fixed']
        for state_keys in r['state_keys'].values():assert len(state_keys)==4 and all(k in index for k in state_keys)

native=[x for x in timings if x['purpose']=='parity_native4']
rate=max(x['seconds']/x['length']**2 for x in native)
tasks=[t for q in plan['cohort64'] for r in q['rotations'] for t in r['tasks']]
forecast=2*rate*sum(len(t['input_ids'])**2 for t in tasks)+60
assert forecast==admission['forecast_seconds_including_60s_finalization']
assert forecast>ledger['remaining_seconds'] and forecast+5>admission['remaining_to_soft_deadline']
result=dict(status='passed_Root_independent_terminal_numeric_and_budget_review',
    fresh_parity_questions=2,cyclic_rotations=8,independent_depth2_depth3_checks=16,
    body_calls=24,native_head_calls=72,actual_vectors_verified=len(index),
    actual_npz_bytes=len(npzraw),actual_npz_sha256=H(npzraw),
    all_K1_K4_Fixed_historical_scores_exact=True,all_depth2_depth3_state_and_fullV_keys_exact=True,
    parameter_hashes_unchanged=269,new_GPU_worker_seconds=ledger['used_seconds'],
    original_20_ledger_total_seconds=ledger['total_used_seconds'],remaining_seconds=ledger['remaining_seconds'],
    fixed64_future_forecast_seconds_including_finalization=forecast,
    forecast_type='Predeclared conservative estimate, not a measured completion time or a mathematical lower bound',
    fixed64_forwards_started=0,development64_effects_opened=False,
    source_inputs_and_historical_ledgers_unchanged=True,
    limits='Two parity questions certify native-readout extraction only; do not identify where most corrections occur in a cohort.')
(P/'root_a4_terminal_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
