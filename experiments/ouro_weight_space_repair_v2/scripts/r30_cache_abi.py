"""CPU-only validation of R28 native state ABI; no model/head/effect evaluation."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def read(p):return json.loads((ROOT/p).read_text(encoding='utf8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def validate():
 complete=read('results/r28_COMPLETE.json');lock=read('configs/r28_SOURCE_PRIOR_PILOT_LOCK.json');index=read('results/r28_NATIVE_STATE_INDEX.json');graph=read('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json');tasks={t['key']:t for t in graph['tasks']};native={k:t for k,t in tasks.items() if t['mode']=='native'}
 assert complete['status']=='complete' and complete['protocol_sha256']==sha('configs/r28_SOURCE_PRIOR_PILOT_LOCK.json') and complete['native_states_sha256']==sha('results/r28_NATIVE_SINGLE_POSITION_STATES.npz') and complete['native_state_index_sha256']==sha('results/r28_NATIVE_STATE_INDEX.json')
 assert len(complete['all_parameter_hashes_before'])==269 and complete['all_parameter_hashes_before']==complete['all_parameter_hashes_after'] and index['parameter_hash_digest']==digest(complete['all_parameter_hashes_after'])
 assert index['weights_sha256']==lock['model_identity']['safetensors_sha256'] and index['cache_contract_sha256']==lock['cache_contract_sha256'] and lock['cache_contract']['graph_sha256']==sha('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json')
 assert lock['cache_contract']['primitive']=='true_prefix_original_uncached_single_position_head'
 expected={k:t for k,t in native.items()};records=index['tasks'];assert len(records)==len({r['task'] for r in records})==1588 and {r['task'] for r in records}==set(expected);refs={};states=0;heads=0
 for r in records:
  t=expected[r['task']];ids=graph['prefix_pool'][t['prefix_ref']]['ids']+t['consumed_ids'];assert digest(ids)==t['input_ids_sha256']==r['input_ids_sha256'] and len(ids)==t['length']
  assert t['read_position']==len(ids)-1 and r['primitive']=='true_prefix_original_uncached_sdpa_BF16_single_position_head'
  assert r['context']==t['context'] and r['depth']==t['depth'] and len(r['states'])==t['depth'] and r['head_input_shape']==[1,1,2048] and r['weights_sha256']==index['weights_sha256'] and r['source_binding_sha256']==index['cache_contract_sha256'];states+=len(r['states']);heads+=3 if r['context']=='Q' else 2
  for ref in r['states']:
   assert ref['dtype']=='torch.bfloat16' and ref['shape']==[1,1,2048]
   if ref['key'] in refs:assert refs[ref['key']]==ref
   refs[ref['key']]=ref
 with np.load(ROOT/'results/r28_NATIVE_SINGLE_POSITION_STATES.npz',allow_pickle=False) as z:
  assert set(z.files)==set(refs)
  for key,r in refs.items():
   a=z[key];assert a.dtype==np.uint16 and list(a.shape)==r['shape'];b=a.tobytes();assert hashlib.sha256(b).hexdigest()==r['raw_sha256'] and hashlib.sha256((r['dtype']+str(tuple(r['shape']))).encode()+b).hexdigest()==key;assert np.isfinite((a.astype(np.uint32)<<16).view(np.float32)).all()
 assert states==6773 and heads==3597
 return dict(status='passed_CPU_native_cache_ABI_only',native_tasks=1588,state_references=6773,unique_state_arrays=len(refs),required_native_heads_per_pass=heads,body_CF_forwards=778,fixed_integrity_body_forwards=7,total_body_forwards_if_complete=785,head_replays_two_native_passes=7194,base_CF_heads=778,model_loaded=False,GPU_worker_seconds=0,old_native_cache_contract_sha256=index['cache_contract_sha256'],graph_sha256=sha('configs/r28_TRUE_PREFIX_TASK_INVENTORY.json'),native_states_sha256=complete['native_states_sha256'],native_state_index_sha256=complete['native_state_index_sha256'],scope='Only old native ABI admitted; old spatial-CF task identity is NOT the new clock-CF identity; no head parity or functional claim until explicit model authorization')
if __name__=='__main__':
 result=validate();p=ROOT/'results/r30_CPU_NATIVE_CACHE_ABI_v2.json'
 with p.open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
