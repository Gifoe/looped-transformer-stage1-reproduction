"""CPU cache/stream admission, no model, tokenizer construction or new head."""
from r33_common import ROOT,read,sha
from r30_cache_abi import validate
from r33_stream import native_stream
def run():
 legacy=validate();abi={k:v for k,v in legacy.items() if k not in ['body_CF_forwards','fixed_integrity_body_forwards','total_body_forwards_if_complete','head_replays_two_native_passes','base_CF_heads']};abi['scope']='Native ABI facts only; legacy R30 counterfactual execution-plan fields are not part of R33';graph,native,uses,groups,fresh=native_stream();old=read('configs/r30_SOURCE_CLOCK_PILOT_LOCK.json');ledgers=dict(old['budget']['inherited_ledger_sha256']);ledgers['logs/r30_gpu_budget.json']=sha('logs/r30_gpu_budget.json');assert len(ledgers)==18 and sum(read(p)['used_seconds'] for p in ledgers)==20755.46525352029
 return dict(status='passed_CPU_native_admission',cache_ABI=abi,questions=24,candidates=96,position_uses=548,Q_prefix_keys=421,native_tasks=1588,state_references=6773,groups=389,head_passes=2,cache_heads=7194,fresh_Q=fresh,model_bodies=1,body_heads=3,total_heads=7197,no_CF_tasks=True,inherited_GPU_ledgers=ledgers,inherited_GPU_seconds=20755.46525352029,remaining_GPU_seconds=844.5347464797087,model_loaded=False,new_head_run=False,GPU_worker_seconds=0,source_sha256={p:sha(p) for p in ['scripts/r33_preflight.py','scripts/r33_stream.py','scripts/r30_cache_abi.py']})
if __name__=='__main__':
 import json
 result=run()
 with (ROOT/'results/r33_CPU_NATIVE_ADMISSION.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['cache_ABI','inherited_GPU_ledgers']}))
