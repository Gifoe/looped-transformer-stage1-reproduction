"""Two legal native-head passes, one fresh Q, no reference/body intervention."""
import time,os,inspect,importlib,contextlib
from r33_common import ROOT,read,sha,digest,save,gate,Vectors,PositionBuffers,pack_diag
from r33_stream import native_stream,METHODS,candidate_readouts
from r33_preflight import run as cache_admission
from r30_native_replay import NativeCache,BASE_METHODS,fill_native_group,verify_scalar_parity
from r28_stream import task_ids
from r33_math import torch_readouts,numpy_readouts
def main():
 started=time.monotonic();timing={};spec=gate();admission=cache_admission();graph,tasks,uses,groups,fresh_key=native_stream();rows=read('configs/r28_SANITIZED_REQUESTS.json')['rows'];choices={(r['id'],c['candidate']):c for r in rows for c in r['candidates']}
 from transformers import AutoTokenizer
 from r28_preflight import audit_runtime
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True);save('results/r33_INPUT_AUDIT.json',dict(cache_admission=admission,tokenizer_input_audit=audit_runtime(tok)))
 import torch,numpy as np,importlib.metadata
 from runtime import load_model,check_budget
 from ouro_cached_scoring import OuroCachedScorer
 from weight_edits import tensor_sha256
 torch.set_grad_enabled(False)
 @contextlib.contextmanager
 def measure(category):
  torch.cuda.synchronize();t=time.monotonic()
  try:yield
  finally:torch.cuda.synchronize();timing[category]=timing.get(category,0.)+time.monotonic()-t
 def budget():check_budget(5);assert time.monotonic()<float(os.environ['R33_SOFT_MONOTONIC'])-5
 def raw(a,b):return (a.contiguous().view(torch.uint8)==b.contiguous().view(torch.uint8)).all()
 timing['CPU/pre_model']=time.monotonic()-started
 with measure('model_load'):tok,model=load_model()
 identity=spec['runtime_identity']
 with measure('hash_and_identity/before'):
  assert sha(inspect.getfile(importlib.import_module(type(model).__module__)))==identity['model_source_sha256']
  import transformers.integrations.sdpa_attention as sdpa,transformers.masking_utils as masks
  assert sha(inspect.getfile(sdpa))==identity['sdpa_sha256'] and sha(inspect.getfile(masks))==identity['masking_sha256'] and all(importlib.metadata.version(k)==identity[k] for k in ['torch','transformers'])
  assert model.config._attn_implementation=='sdpa' and model.config.attention_dropout==0 and model.config.sliding_window is None and all(l.self_attn.is_causal and l.self_attn.sliding_window is None for l in model.model.layers) and not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
  before={n:tensor_sha256(p) for n,p in model.named_parameters()};assert len(before)==269 and before==read('results/r28_COMPLETE.json')['all_parameter_hashes_after']
 device=model.lm_head.weight.device;cache=NativeCache(device);evidence=Vectors(timing,'IO/evidence');diagnostics={};checks=[];fatal=None;completed_groups=0;body_calls=[];body_heads=0;buffers=PositionBuffers(rows,METHODS,device);saved_checked=set();parity_passes=[]
 oldaudit=read('results/r28_EXECUTION_AUDIT.json');oldraw=np.load(ROOT/'results/r28_REPRESENTATIVE_VECTORS.npz',allow_pickle=False);expected={}
 for item in oldaudit['checks']:
  if item['type']=='N0_representative':source=item['raw']
  elif item['type']=='no_hook_capture_exact':source=item['native']
  elif item['type']=='first_actual_duplicate_exact':source=item['A']
  else:continue
  for k,ref in source['heads'].items():expected[(item['task'],int(k))]=ref
 assert len(expected)==9;original_heads=cache.heads
 def heads(task,model,measure):
  budget();states,z=original_heads(task,model,measure)
  with measure('checks/saved_native_fullV'):
   for k,v in z.items():
    if (task['key'],k) in expected:
     ref=expected[(task['key'],k)];a=torch.from_numpy(oldraw[ref['key']].copy()).to(device);assert a.shape==(1,1,49152) and bool(raw(v.reshape(1,1,-1),a).item());saved_checked.add((task['key'],k))
  return states,z
 cache.heads=heads
 def snap(states,z):return dict(states=[evidence.keep(s) for s in states],heads={str(k):evidence.keep(v.reshape(1,1,-1)) for k,v in z.items()})
 try:
  first=PositionBuffers(rows,BASE_METHODS,device)
  for group in groups:fill_native_group(group,tasks,cache,model,measure,uses,choices,first)
  assert cache.head_calls==3597 and len(saved_checked)==9
  with measure('checks/first_native_scalar_parity'):parity=verify_scalar_parity(rows,first.finish(),read('results/r28_SCORES.json'))
  parity.update(native_head_calls=3597,historical_saved_fullV_heads_checked=9,all_historical_fullV_available=False);parity_passes.append(parity);save('results/r33_NATIVE_PARITY.json',parity);del first
  budget();task=tasks[fresh_key];ids=task_ids(graph,task);scorer=OuroCachedScorer(model,5)
  with measure('forward_body/fresh_Q'):fresh=scorer._run(ids,positions=[len(ids)-1],use_cache=False)[1]
  assert len(fresh)==5;z={}
  for k in [1,4,5]:
   h=fresh[k-1];assert h.dtype==torch.bfloat16 and h.shape==(1,1,2048)
   with measure('head/body/fresh_Q'):rz=model.lm_head(h);assert rz.dtype==torch.bfloat16 and rz.shape==(1,1,49152);z[k]=rz.float()[0,0];body_heads+=1
  with measure('checks/fresh_Q'):
   cached=cache.states(task);oldz={k:torch.from_numpy(oldraw[expected[(fresh_key,k)]['key']].copy()).to(device)[0,0] for k in [1,4,5]};assert bool(torch.stack([raw(a,b) for a,b in zip(fresh,cached)]+[raw(z[k],oldz[k]) for k in [1,4,5]]).all().item())
  body_calls.append(dict(task=fresh_key,input_ids_sha256=task['input_ids_sha256'],depth=5,context='Q',length=len(ids),use_cache=False,head_input_shape=[1,1,2048],heads=[1,4,5],no_hooks=True));checks.append(dict(type='fresh_Q_exact',fresh=snap(fresh,z),cached=snap(cached,oldz),exact=True));del fresh,z,cached,oldz
  vp=np.random.default_rng(2021).permutation(49152);assert vp.dtype==np.int64 and vp.shape==(49152,) and np.array_equal(np.sort(vp),np.arange(49152)) and digest(vp.tolist())==spec['vocabulary_permutation_sha256'];vp=torch.as_tensor(vp,device=device);real_math=False;h0=cache.head_calls
  for group in groups:
   refs={}
   for name in ['N0','N1','N2']:
    _,refs[name]=cache.heads(tasks[group['reference_tasks'][name+'_native']],model,measure)
   with measure('projection/reference_setup'):
    n1=torch.stack([refs[n][1] for n in ['N0','N1','N2']]);n4=torch.stack([refs[n][4] for n in ['N0','N1','N2']]);prior={name:v.reshape(1,-1).log_softmax(-1)[0] for name,v in dict(N0_native1=refs['N0'][1],N0_native4=refs['N0'][4],N0_Fixed=refs['N0'][4]+.5*(refs['N0'][4]-refs['N0'][1])).items()}
   for qkey in group['Q_consumers']:
    _,q=cache.heads(tasks[qkey],model,measure)
    with measure('projection/common_gain_and_controls'):out,diag=torch_readouts(q[1],q[4],n1,n4,vp)
    if not real_math:
     with measure('checks/representative_numpy_math'):
      independent,nd=numpy_readouts(q[1].cpu().numpy(),q[4].cpu().numpy(),n1.cpu().numpy(),n4.cpu().numpy(),vp.cpu().numpy());error=max(float(np.max(np.abs(independent[k]-v.cpu().numpy()))) for k,v in out.items());packed=pack_diag(diag);numeric=[k for k,v in nd.items() if not isinstance(v,(str,bool,np.bool_))];de=max(float(abs(nd[k]-packed[k])) for k in numeric);assert error<=1e-6 and de<=1e-10
     checks.append(dict(type='first_real_common_gain_math',task=qkey,max_FP32_output_error=error,max_FP64_diag_error=de,q1=evidence.keep(q[1].reshape(1,1,-1)),q4=evidence.keep(q[4].reshape(1,1,-1)),n1=evidence.keep(n1),n4=evidence.keep(n4),outputs={k:evidence.keep(v.reshape(1,1,-1)) for k,v in out.items()},diagnostics=packed));real_math=True
    with measure('projection/normalization_buffers'):
     top=q[4].reshape(1,-1).softmax(-1).topk(2,dim=-1).values[0];out.update(native1=q[1],native4=q[4],native5=q[5],adaptive=q[4]+(1-(top[0]-top[1]))*(q[4]-q[1]));logs={name:v.reshape(1,-1).log_softmax(-1)[0] for name,v in out.items()};diagnostics[qkey]=pack_diag(diag)
     for u in uses[qkey]:
      c=choices[(u['id'],u['candidate'])];t=u['Q_position']
      for name,v in logs.items():buffers.put(u['id'],u['candidate'],name,t,v[c['Q_target_ids'][t]])
      for name,v in prior.items():buffers.put(u['id'],u['candidate'],name,t,v[c['reference_target_ids']['N0'][t]])
    del q,out,logs
   completed_groups+=1;save('results/r33_PROGRESS.json',dict(groups=completed_groups,target_groups=389,native_head_calls=cache.head_calls,body_calls=len(body_calls),body_heads=body_heads,elapsed_seconds=time.monotonic()-started));del refs,n1,n4,prior
  assert real_math and completed_groups==389 and cache.head_calls-h0==3597 and cache.head_calls==7194 and len(body_calls)==1 and body_heads==3 and len(diagnostics)==421
 except BaseException as e:fatal=repr(e)
 finally:
  cleanup=dict(no_hooks_installed=True,all_model_hooks_empty=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()))
  with measure('hash_and_identity/after'):after={n:tensor_sha256(p) for n,p in model.named_parameters()}
  if before!=after or not cleanup['all_model_hooks_empty']:fatal=fatal or 'Parameter or hook integrity failure'
  cache.close();oldraw.close();evidence.write('results/r33_REPRESENTATIVE_VECTORS.npz')
  if fatal is None:
   with measure('projection/final_FP32_sums_R6'):
    finished=buffers.finish();parity_passes.append(verify_scalar_parity(rows,{k:v for k,v in finished.items() if k[2] in BASE_METHODS},read('results/r28_SCORES.json')));result=candidate_readouts(rows,finished)
   save('results/r33_SCORES.json',dict(rows=result,diagnostics=diagnostics,scope='Label-free fixed24 coordinate common gain; no gold/argmax'))
  save('results/r33_EXECUTION_AUDIT.json',dict(body_calls=body_calls,body_heads=body_heads,native_head_calls=cache.head_calls,total_heads=cache.head_calls+body_heads,groups=completed_groups,checks=checks,native_parity_passes=parity_passes,cleanup=cleanup,fatal=fatal))
  elapsed=time.monotonic()-started;measured=sum(timing.values());assert measured<=elapsed+1e-6
  save('results/r33_COMPLETE.json',dict(status='complete' if fatal is None else 'invalid',n=24 if fatal is None else None,body_calls=len(body_calls),body_heads=body_heads,native_head_calls=cache.head_calls,total_heads=cache.head_calls+body_heads,groups=completed_groups,protocol_sha256=sha('configs/r33_COMMON_GAIN_PILOT_LOCK.json'),scores_sha256=sha('results/r33_SCORES.json') if fatal is None else None,audit_sha256=sha('results/r33_EXECUTION_AUDIT.json'),representative_vectors_sha256=sha('results/r33_REPRESENTATIVE_VECTORS.npz'),native_parity_sha256=sha('results/r33_NATIVE_PARITY.json') if (ROOT/'results/r33_NATIVE_PARITY.json').exists() else None,all_parameter_hashes_before=before,all_parameter_hashes_after=after,cleanup=cleanup,fatal=fatal,component_seconds=timing,elapsed_seconds=elapsed,measured_components_sum=measured,unclassified_worker_residual=elapsed-measured,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),heldout_accessed=False,timing_scope='Diagnostic head replay/math/check/IO pipeline; actual supervisor lifecycle is the sole debit, not deployment speed'))
 if fatal is not None:raise RuntimeError(fatal)
if __name__=='__main__':main()
