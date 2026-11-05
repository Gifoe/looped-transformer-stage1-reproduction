"""Two native head passes precede/stream one fixed source-clock CF per prefix."""
import time,os,inspect,importlib,contextlib
from r30_common import ROOT,read,sha,digest,save,gate,Vectors,PositionBuffers
from r30_protocol import build
from r30_cache_abi import validate
from r30_native_replay import NativeCache,BASE_METHODS,fill_native_group,verify_scalar_parity
from r30_clock_hooks import SourceClock
from r28_stream import task_ids,METHODS,candidate_readouts
from r28_math_v2 import torch_readouts,numpy_readouts
def pack_diag(diag):
 import torch
 return {k:(v.detach().cpu().item() if isinstance(v,torch.Tensor) else v) for k,v in diag.items()}
def main():
 started=time.monotonic();timing={};spec=gate();abi=validate();graph,tasks,uses,plan=build();assert plan==read('configs/r30_CLOCK_TASK_PLAN.json');rows=read('configs/r28_SANITIZED_REQUESTS.json')['rows'];choices={(r['id'],c['candidate']):c for r in rows for c in r['candidates']};positions=tuple(plan['source_positions']);assert len(positions)==143
 from transformers import AutoTokenizer
 from r28_preflight import audit_runtime
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True);audit=audit_runtime(tok);save('results/r30_INPUT_AUDIT.json',dict(original_true_prefix_audit=audit,native_ABI=abi,new_task_plan_sha256=sha('configs/r30_CLOCK_TASK_PLAN.json')))
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
 def budget():check_budget(5);assert time.monotonic()<float(os.environ['R30_SOFT_MONOTONIC'])-5
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
 device=model.lm_head.weight.device;cache=NativeCache(device);evidence=Vectors(timing,'IO/evidence');calls=[];checks=[];count=0;fatal=None;controller=None;groups=0;diagnostics={};saved_heads_checked=set();base_done=set();native_state_refs={};buffers=PositionBuffers(rows,METHODS,device)
 # Historical fullV exists for precisely nine representative native heads.
 oldaudit=read('results/r28_EXECUTION_AUDIT.json');oldraw=np.load(ROOT/'results/r28_REPRESENTATIVE_VECTORS.npz',allow_pickle=False);expected_heads={}
 for item in oldaudit['checks']:
  if item['type']=='N0_representative':source=item['raw']
  elif item['type']=='no_hook_capture_exact':source=item['native']
  elif item['type']=='first_actual_duplicate_exact':source=item['A']
  else:continue
  for k,ref in source['heads'].items():expected_heads[(item['task'],int(k))]=ref
 assert len(expected_heads)==9
 original_heads=cache.heads
 def heads(task,model,measure):
  budget();states,z=original_heads(task,model,measure)
  with measure('checks/native_representative_head'):
   for k,v in z.items():
    if (task['key'],k) in expected_heads:
     ref=expected_heads[(task['key'],k)];a=torch.from_numpy(oldraw[ref['key']].copy()).to(device);assert a.dtype==torch.float32 and bool(raw(v.reshape(1,1,-1),a).item());saved_heads_checked.add((task['key'],k))
  return states,z
 cache.heads=heads
 def snap(states,z,full=False):return dict(states=[evidence.keep(s if full else s[:,-1:,:]) for s in states],heads={str(k):evidence.keep(v.reshape(1,1,-1)) for k,v in z.items()})
 def body(task,mode,purpose,full=False):
  nonlocal count
  budget();ids=task_ids(graph,task);scorer=OuroCachedScorer(model,task['depth'])
  if mode!='no_hook':controller.bind(scorer);controller.configure(mode,ids,task['context'],positions)
  with measure('forward_body/'+task['context']+'/'+mode+'/'+purpose):states=scorer._run(ids,positions=range(len(ids)) if full else [len(ids)-1],use_cache=False)[1];count+=1
  with measure('checks/clock_finite'):
   assert len(states)==task['depth'] and all(s.dtype==torch.bfloat16 for s in states) and bool(torch.stack([torch.isfinite(s).all() for s in states]).all().item());clock=None if mode=='no_hook' else controller.verify()
  z={}
  for k in ([1,4,5] if task['context']=='Q' else [4]):
   h=states[k-1][:,-1:,:];assert h.shape==(1,1,2048)
   with measure('head/body/'+task['context']+'/'+mode+'/'+purpose):v=model.lm_head(h);assert v.dtype==torch.bfloat16 and v.shape==(1,1,49152);z[k]=v.float()[0,0]
   assert torch.isfinite(z[k]).all()
  record=dict(number=count,task=task['key'],native_task=task.get('native_task',task['key']),mode=mode,purpose=purpose,input_ids_sha256=task['input_ids_sha256'],context=task['context'],depth=task['depth'],length=len(ids),clock=clock,head_input_shape=[1,1,2048]);calls.append(record)
  return states,z,record
 try:
  # Complete native-only pass and scalar gate before any model-body CF feature.
  first=PositionBuffers(rows,BASE_METHODS,device);h0=cache.head_calls
  for g in plan['groups']:fill_native_group(dict(reference_tasks={n+'_native':k for n,k in g['reference_native'].items()},Q_consumers=g['Q_consumers']),tasks,cache,model,measure,uses,choices,first)
  assert cache.head_calls-h0==3597 and len(saved_heads_checked)==9
  with measure('checks/native_scalar_parity'):parity=verify_scalar_parity(rows,first.finish(),read('results/r28_SCORES.json'))
  parity.update(native_head_calls=3597,historical_saved_fullV_heads_checked=9,all_historical_fullV_available=False);save('results/r30_NATIVE_PARITY.json',parity);del first
  qtask=tasks[plan['fresh_Q_task']];fresh,qz,_=body(qtask,'no_hook','fresh_Q');cached=cache.states(qtask);cz={k:torch.from_numpy(oldraw[expected_heads[(qtask['key'],k)]['key']].copy()).to(device)[0,0] for k in [1,4,5]}
  with measure('checks/fresh_Q'):assert bool(torch.stack([raw(a,b) for a,b in zip(fresh,cached)]+[raw(qz[k],cz[k]) for k in [1,4,5]]).all().item())
  checks.append(dict(type='fresh_Q_exact',task=qtask['key'],fresh=snap(fresh,qz),cached=snap(cached,cz),exact=True));del fresh,qz,cached,cz
  vp=np.random.default_rng(2021).permutation(49152);assert vp.dtype==np.int64 and np.array_equal(np.sort(vp),np.arange(49152)) and digest(vp.tolist())==spec['vocabulary_permutation_sha256'];vp=torch.as_tensor(vp,device=device)
  real_math=False;h1=cache.head_calls
  with SourceClock(model) as controller:
   for gi,g in enumerate(plan['groups']):
    refs={};cf={}
    for name in ['N0','N1','N2']:
     nt=tasks[g['reference_native'][name]];native,nz=cache.heads(nt,model,measure);refs[name]=nz
     if name=='N0':continue
     ct=tasks[g['reference_CF'][name]];representative=nt['key']==plan['representative_native'][name];normal=None;normalraw=None
     if representative:
      normal,nh,_=body(nt,'capture_only','native_capture_only',full=True);donors=controller.captured()
      with measure('checks/capture_only_native'):
       assert bool(torch.stack([raw(s[:,-1:,:],a) for s,a in zip(normal,native)]+[raw(nh[4],nz[4])]).all().item())
       # Full old native/no-hook evidence, not only cache-last-position.
       old=next(x['native'] for x in oldaudit['checks'] if x['type']=='no_hook_capture_exact' and x['task']==nt['key'])
       assert bool(torch.stack([raw(s,torch.from_numpy(oldraw[r['key']].copy()).view(torch.bfloat16).to(device)) for s,r in zip(normal,old['states'])]).all().item())
      normalraw=snap(normal,nh,True);checks.append(dict(type='capture_only_native_exact',task=nt['key'],raw=normalraw,donors=[evidence.keep(s) for s in donors],exact=True))
     cs,ch,rec=body(ct,'clock_CF','base',full=representative);early=controller.captured()
     with measure('checks/first3_last_position'):
      assert bool(torch.stack([raw(s[:,-1:,:],a) for s,a in zip(cs[:3],native[:3])]).all().item())
     rec.update(first3_last_position_exact=True,states=[evidence.keep(s[:,-1:,:]) for s in cs]);base_done.add(ct['key']);cf[name]=ch[4]
     if representative:
      with measure('checks/representative_full_sequence_and_source'):
       assert bool(torch.stack([raw(a,b) for a,b in zip(cs[:3],normal[:3])]+[raw(a,b) for a,b in zip(early,donors)]).all().item())
      cfsnap=snap(cs,ch,True);checks.append(dict(type='CF_first3_full_and_loop1_source_exact',task=ct['key'],native=normalraw,CF=cfsnap,donors=[evidence.keep(s) for s in early],exact=True))
      si,zi,_=body(nt,'identity','identity',full=True)
      with measure('checks/identity'):assert bool(torch.stack([raw(a,b) for a,b in zip(si,normal)]+[raw(zi[4],nz[4])]).all().item())
      checks.append(dict(type='identity_exact',task=nt['key'],native=normalraw,identity=snap(si,zi,True),exact=True));del si,zi
      repeat,rh,_=body(ct,'clock_CF','repeat',full=False)
      with measure('checks/CF_repeat'):assert bool(torch.stack([raw(a[:,-1:,:],b) for a,b in zip(cs,repeat)]+[raw(ch[4],rh[4])]+[raw(a,b) for a,b in zip(early,controller.captured())]).all().item())
      checks.append(dict(type='CF_repeat_exact',task=ct['key'],first=snap(cs,ch),repeat=snap(repeat,rh),donors=[evidence.keep(s) for s in controller.captured()],exact=True));del repeat,rh,normal,donors
     del cs,ch,native
    with measure('projection/reference_setup'):
     n1=torch.stack([refs[n][1] for n in ['N0','N1','N2']]);n4=torch.stack([refs[n][4] for n in ['N0','N1','N2']]);b4=torch.stack([cf[n] for n in ['N1','N2']]);prior={n:z.reshape(1,-1).log_softmax(-1)[0] for n,z in dict(N0_native1=refs['N0'][1],N0_native4=refs['N0'][4],N0_Fixed=refs['N0'][4]+.5*(refs['N0'][4]-refs['N0'][1])).items()}
    for qkey in g['Q_consumers']:
     qs,q=cache.heads(tasks[qkey],model,measure)
     with measure('projection/estimators_controls'):out,diag=torch_readouts(q[1],q[4],n1,n4,b4,vp)
     if not real_math:
      with measure('checks/representative_math'):
       independent,nd=numpy_readouts(q[1].cpu().numpy(),q[4].cpu().numpy(),n1.cpu().numpy(),n4.cpu().numpy(),b4.cpu().numpy(),vp.cpu().numpy());error=max(float(np.max(np.abs(independent[k]-z.cpu().numpy()))) for k,z in out.items());assert error<=1e-6;de=max(float(np.max(np.abs(np.asarray(nd[k])-diag[k].detach().cpu().numpy()))) for k in ['alpha','numerator','denominator','main_norm2','scale_g2','scale_r2','scale_rho']);assert de<=1e-10
      checks.append(dict(type='first_real_math',task=qkey,max_FP32_output_error=error,max_FP64_diag_error=de,q1=evidence.keep(q[1].reshape(1,1,-1)),q4=evidence.keep(q[4].reshape(1,1,-1)),n1=evidence.keep(n1),n4=evidence.keep(n4),b4=evidence.keep(b4)));real_math=True
     with measure('projection/normalization_buffers'):
      top=q[4].reshape(1,-1).softmax(-1).topk(2,dim=-1).values[0];out.update(native1=q[1],native4=q[4],native5=q[5],adaptive=q[4]+(1-(top[0]-top[1]))*(q[4]-q[1]));logs={n:z.reshape(1,-1).log_softmax(-1)[0] for n,z in out.items()};diagnostics[qkey]=pack_diag(diag)
      for u in uses[qkey]:
       c=choices[(u['id'],u['candidate'])];t=u['Q_position']
       for name,z in logs.items():buffers.put(u['id'],u['candidate'],name,t,z[c['Q_target_ids'][t]])
       for name,z in prior.items():buffers.put(u['id'],u['candidate'],name,t,z[c['reference_target_ids']['N0'][t]])
     del qs,q,out,logs
    groups+=1;save('results/r30_PROGRESS.json',dict(groups=groups,body_forwards=count,CF_completed=len(base_done),native_heads=cache.head_calls,elapsed_seconds=time.monotonic()-started));del refs,cf,n1,n4,b4,prior
   assert count==785 and groups==389 and len(base_done)==778 and len(diagnostics)==421 and real_math and cache.head_calls-h1==3597
 except BaseException as e:fatal=repr(e)
 finally:
  cleanup=getattr(controller,'cleanup',dict(handles_removed=True,original_hooks_restored=True,errors=[]));cleanup['all_model_hooks_empty']=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
  with measure('hash_and_identity/after'):after={n:tensor_sha256(p) for n,p in model.named_parameters()}
  if before!=after or not cleanup['all_model_hooks_empty'] or not cleanup['original_hooks_restored'] or cleanup['errors']:fatal=fatal or 'Cleanup or parameter integrity failure'
  cache.close();oldraw.close();evidence.write('results/r30_REPRESENTATIVE_VECTORS.npz');save('results/r30_EXECUTION_AUDIT.json',dict(calls=calls,checks=checks,cleanup=cleanup,fatal=fatal,groups=groups,CF_tasks=list(sorted(base_done)),native_head_calls=cache.head_calls))
  if fatal is None:
   with measure('projection/final_FP32_sums_R6'):finished=buffers.finish();verify_scalar_parity(rows,{k:v for k,v in finished.items() if k[2] in BASE_METHODS},read('results/r28_SCORES.json'));result=candidate_readouts(rows,finished)
   save('results/r30_SCORES.json',dict(rows=result,diagnostics=diagnostics,scope='Label-free fixed24 source-clock outputs; no gold/predictions'))
  elapsed=time.monotonic()-started;measured=sum(timing.values());assert measured<=elapsed+1e-6
  save('results/r30_COMPLETE.json',dict(status='complete' if fatal is None else 'invalid',n=24 if fatal is None else None,body_forwards=count,CF_forwards=len(base_done),groups=groups,protocol_sha256=sha('configs/r30_SOURCE_CLOCK_PILOT_LOCK.json'),scores_sha256=sha('results/r30_SCORES.json') if fatal is None else None,audit_sha256=sha('results/r30_EXECUTION_AUDIT.json'),representative_vectors_sha256=sha('results/r30_REPRESENTATIVE_VECTORS.npz'),native_parity_sha256=sha('results/r30_NATIVE_PARITY.json') if (ROOT/'results/r30_NATIVE_PARITY.json').exists() else None,all_parameter_hashes_before=before,all_parameter_hashes_after=after,cleanup=cleanup,fatal=fatal,component_seconds=timing,elapsed_seconds=elapsed,measured_components_sum=measured,unclassified_worker_residual=elapsed-measured,native_head_calls=cache.head_calls,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),heldout_accessed=False,timing_scope='Diagnostic pipeline including head replay and hooks; supervisor actual is sole total debit; not deployment throughput'))
 if fatal is not None:raise RuntimeError(fatal)
if __name__=='__main__':main()
