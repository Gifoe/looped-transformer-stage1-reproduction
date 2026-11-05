"""CPU scalar-buffer, stream completeness and analyzer ordering checks; no model."""
from pathlib import Path
import json,hashlib,ast
import torch,numpy as np
from r28_common import ROOT,PositionBuffers,read,save,sha,assert_fresh,RUNTIME_OUTPUTS
from r28_stream import stream_contract,source_contract,METHODS

def main():
 graph,tasks,uses,checks=stream_contract();positions,permutation,mapsha=source_contract();assert len(positions)==143 and len(graph['usages'])==548 and len(tasks)==2366 and checks['total_forwards_if_complete']==2371
 rng=np.random.default_rng(2021);order=list(tasks);rng.shuffle(order);assert set(order)==set(tasks) and all(t['source_map_sha256']==mapsha for t in tasks.values() if t['mode']!='native')
 fixture=[dict(id='synthetic',candidates=[dict(Q_target_ids=list(range(7)),reference_target_ids=dict(N0=list(range(9))))])];names=['primary','N0_native4'];original=PositionBuffers(fixture,names,'cpu');shuffled=PositionBuffers(fixture,names,'cpu');values={name:-torch.as_tensor(rng.random(9 if name.startswith('N0_') else 7),dtype=torch.float32) for name in names}
 for name,v in values.items():
  for i,x in enumerate(v):original.put('synthetic',0,name,i,x)
  for i in rng.permutation(len(v)):shuffled.put('synthetic',0,name,int(i),v[i])
 assert original.finish()==shuffled.finish()
 for name in names:
  try:shuffled.put('synthetic',0,name,0,torch.tensor(-1.,dtype=torch.float32))
  except AssertionError:pass
  else:raise AssertionError('duplicate buffer overwrite')
 missing=PositionBuffers(fixture,names,'cpu')
 try:missing.finish()
 except AssertionError:pass
 else:raise AssertionError('missing positions summed')
 assert len(original.buffers[('synthetic',0,'N0_native4')])==9
 fixture_path=ROOT/'transport/r28_CPU_ONCE_GUARD_PRIVATE.txt';assert not fixture_path.exists()
 try:
  fixture_path.write_text('CPU guard fixture only')
  try:assert_fresh([fixture_path.relative_to(ROOT).as_posix()])
  except AssertionError:pass
  else:raise AssertionError('preexisting output accepted')
 finally:fixture_path.unlink()
 assert_fresh([fixture_path.relative_to(ROOT).as_posix()]);assert len(RUNTIME_OUTPUTS)==14
 analyzer=(ROOT/'scripts/r28_analyze.py').read_text();worker=(ROOT/'scripts/r28_worker.py').read_text();assert analyzer.index('numeric_integrity(spec)')<analyzer.index('def analyze_effects') and "c,scores,rows,unmatched=numeric_integrity(spec)" in analyzer and 'if unmatched:save' in analyzer
 assert worker.index('audit_runtime(tok)')<worker.index('tok,model=load_model()') and 'rawz.dtype==torch.bfloat16' in worker and "forward_body/" in worker and "head/" in worker and "component_seconds=timing" in worker
 for p in ['scripts/r28_common.py','scripts/r28_stream.py','scripts/r28_worker.py','scripts/r28_preflight.py','scripts/r28_analyze.py','scripts/r28_supervise.py']:ast.parse((ROOT/p).read_text());assert '\\' not in p
 save('results/r28_CPU_STREAM_CHECKS.json',dict(status='passed_CPU_only',checks=['Full2366 node stream and2371 explicit count','all548uses and complete N0 mapping','fixed143 source bijection','execution order permutation leaves task set unchanged','original-position FP32 one-sum buffer replay exact','duplicate/missing buffer guard','synthetic N0 trajectory may be longer than Q without truncation','source-bound full integrity before gold/argmax and unmatched stops effect unsealing','pre-model tokenizer audit and BF16head/synchronized diagnostic timing source','single-run output guard rejects temporary CPU fixture, official paths untouched'],native_cache_expected_tasks=1588,native_cache_expected_states=6773,native_cache_raw_upper_bytes=6773*2048*2,model_loaded=False,GPU_worker_seconds=0,source_sha256={p:sha(p) for p in ['scripts/r28_common.py','scripts/r28_stream.py','scripts/r28_worker.py','scripts/r28_preflight.py','scripts/r28_analyze.py','scripts/r28_supervise.py','scripts/r28_stream_checks.py']}))
 print('R28_CPU_STREAM_CHECKS_PASS')
if __name__=='__main__':main()
