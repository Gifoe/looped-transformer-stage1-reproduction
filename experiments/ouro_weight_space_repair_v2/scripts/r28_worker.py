"""One fixed24 true-prefix stream; no historical scores/labels/model training."""
import time,os,inspect,importlib
from r28_common import ROOT,LOCK,read,sha,digest,save,gate,Vectors,PositionBuffers
from r28_stream import stream_contract,task_ids,source_contract,METHODS,candidate_readouts
from r28_hooks import AnswerReplay
from r28_math_v2 import torch_readouts,numpy_readouts

def main():
 started=time.monotonic();timing={};spec=gate()
 from transformers import AutoTokenizer
 from r28_preflight import audit_runtime
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True);audit=audit_runtime(tok);save('results/r28_INPUT_AUDIT.json',audit)
 graph,tasks,uses,checks=stream_contract();rows=read('configs/r28_SANITIZED_REQUESTS.json')['rows'];choices={(r['id'],c['candidate']):c for r in rows for c in r['candidates']};positions,permutation,mapsha=source_contract();assert mapsha==spec['source_map_sha256'] and checks==spec['execution_checks']
 import numpy as np
 vp=np.random.default_rng(2021).permutation(49152);assert vp.dtype==np.int64 and vp.shape==(49152,) and np.array_equal(np.sort(vp),np.arange(49152)) and digest(vp.tolist())==spec['vocabulary_permutation_sha256']
 from runtime import load_model,check_budget
 from ouro_cached_scoring import OuroCachedScorer
 from weight_edits import tensor_sha256
 import torch,importlib.metadata
 torch.set_grad_enabled(False);timing['CPU/pre_model']=time.monotonic()-started;load_begin=time.monotonic();tok,model=load_model();torch.cuda.synchronize();timing['model_load']=time.monotonic()-load_begin;hash_begin=time.monotonic();identity=spec['runtime_identity'];assert sha(inspect.getfile(importlib.import_module(type(model).__module__)))==identity['model_source_sha256']
 import transformers.integrations.sdpa_attention as sdpa,transformers.masking_utils as masks
 assert sha(inspect.getfile(sdpa))==identity['sdpa_sha256'] and sha(inspect.getfile(masks))==identity['masking_sha256'] and all(importlib.metadata.version(k)==identity[k] for k in ['torch','transformers'])
 assert model.config._attn_implementation=='sdpa' and model.config.attention_dropout==0 and model.config.sliding_window is None and all(l.self_attn.is_causal and l.self_attn.sliding_window is None for l in model.model.layers) and not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules())
 before={n:tensor_sha256(p) for n,p in model.named_parameters()};assert len(before)==269 and before==read('results/r27_EXEC_COMPLETE.json')['all_parameter_hashes_after'];torch.cuda.synchronize();timing['hash_and_identity/before']=time.monotonic()-hash_begin;vp=torch.as_tensor(vp,device=model.lm_head.weight.device,dtype=torch.long)
 evidence=Vectors(timing,'IO/representative');native_cache=Vectors(timing,'IO/native_states');native_index=[];buffers=PositionBuffers(rows,METHODS,model.lm_head.weight.device);calls=[];done=set();diagnostics={};checks_evidence=[];count=0;controller=None;fatal=None;completed_groups=0
 def begin():torch.cuda.synchronize();return time.monotonic()
 def end(category,t):
  torch.cuda.synchronize();timing[category]=timing.get(category,0.)+time.monotonic()-t
 def raw_equal(a,b):return (a.contiguous().view(torch.uint8)==b.contiguous().view(torch.uint8)).all()
 def forward(task,mode,donor=None,identity_map=False,plain=False,purpose='base'):
  nonlocal count
  check_budget(5);assert time.monotonic()<float(os.environ['R28_SOFT_MONOTONIC']);ids=task_ids(graph,task);setup_begin=time.monotonic();scorer=OuroCachedScorer(model,task['depth'])
  if not plain:
   controller.bind(scorer);controller.configure(mode,ids,task['context'],positions if mode!='off' else (),tuple(range(143)) if identity_map else permutation if mode!='off' else (),donor,task['depth'])
  timing['scorer_setup_and_configure']=timing.get('scorer_setup_and_configure',0.)+time.monotonic()-setup_begin
  body_begin=begin()
  full=task['context'] in ['N1','N2'];states=scorer._run(ids,positions=range(len(ids)) if full else [len(ids)-1],use_cache=False)[1];count+=1;assert len(states)==task['depth'];end('forward_body/'+task['context']+'/'+mode+'/'+purpose,body_begin);verify_begin=begin();clock=None if plain else controller.verify();end('checks/hook_clock_and_flags',verify_begin);heads={}
  wanted=[1,4,5] if task['context']=='Q' else [4] if mode=='patch' else [1,4]
  for k in wanted:
   h=states[k-1][:,-1:,:];assert h.shape==(1,1,2048) and h.dtype==torch.bfloat16;head_begin=begin();rawz=model.lm_head(h);assert rawz.dtype==torch.bfloat16 and rawz.shape==(1,1,49152);z=rawz.float();end('head/'+task['context']+'/'+mode+'/'+purpose,head_begin);finite_begin=begin();assert torch.isfinite(z).all();end('checks/finite',finite_begin);heads[k]=z[0,0]
  finite_begin=begin();assert bool(torch.stack([torch.isfinite(s).all() for s in states]).all().item());end('checks/finite',finite_begin)
  if clock is not None:clock=dict(clock,events_sha256=digest(clock['events']));clock.pop('events')
  record=dict(number=count,task=task['key'],purpose=purpose,context=task['context'],mode=mode,depth=task['depth'],input_ids_sha256=task['input_ids_sha256'],length=len(ids),clock=clock,head_input_shape=[1,1,2048])
  calls.append(record)
  if purpose=='base':
   assert task['key'] not in done;done.add(task['key'])
   if task['mode']=='native':native_index.append(dict(task=task['key'],input_ids_sha256=task['input_ids_sha256'],context=task['context'],depth=task['depth'],states=[native_cache.keep(s[:,-1:,:]) for s in states],weights_sha256=spec['model_identity']['safetensors_sha256'],primitive='true_prefix_original_uncached_sdpa_BF16_single_position_head',head_input_shape=[1,1,2048],source_binding_sha256=spec['cache_contract_sha256']))
  return states,heads,record
 def snap(states,heads,full=True):return dict(states=[evidence.keep(s if full else s[:,-1:,:]) for s in states],heads={str(k):evidence.keep(z.reshape(1,1,-1)) for k,z in heads.items()})
 def lp(z):return z.reshape(1,-1).log_softmax(-1)[0]
 try:
  nohook={}
  for key in checks['no_hook_native_tasks']:
   task=tasks[key];states,heads,record=forward(task,'off',plain=True,purpose='no_hook_native');nohook[key]=(states,heads,snap(states,heads))
  with AnswerReplay(model) as controller:
   duplicate_checked=False;real_math_checked=False
   for gi,group in enumerate(graph['stream_groups']):
    refs={};group_checks=[]
    for name in ['N0','N1','N2']:
     task=tasks[group['reference_tasks'][name+'_native']];states,heads,record=forward(task,'off' if name=='N0' else 'capture');refs[name]=(heads,None)
     if name=='N0':
      if gi==0:checks_evidence.append(dict(type='N0_representative',task=task['key'],raw=snap(states,heads,False)))
      del states;continue
     donor=controller.take_donor();assert all(z.shape==(1,143,spec['projection_width']) for z in donor.values)
     representative=task['key'] in checks['no_hook_native_tasks'];native_snap=None
     if representative:
      old,oldheads,oldraw=nohook.pop(task['key']);check_begin=begin();flags=[raw_equal(a,b) for a,b in zip(old,states)]+[raw_equal(oldheads[k],heads[k]) for k in [1,4]];assert bool(torch.stack(flags).all().item());end('checks/no_hook_parity',check_begin);native_snap=snap(states,heads);checks_evidence.append(dict(type='no_hook_capture_exact',task=task['key'],no_hook=oldraw,native=native_snap,donors=[evidence.keep(z) for z in donor.values],exact=True));del old,oldheads
     cf_task=tasks[group['reference_tasks'][name+'_CF']];cf,cfheads,cfrecord=forward(cf_task,'patch',donor);check_begin=begin();flags=[raw_equal(a,b) for a,b in zip(states[:3],cf[:3])];assert bool(torch.stack(flags).all().item());end('checks/full_first3',check_begin);cfrecord['first3_full_sequence_exact']=True;cfrecord['native_task']=task['key'];cfrecord['donor_native_input_sha256']=donor.input_sha256;cfrecord['source_map_sha256']=mapsha
     if representative:checks_evidence.append(dict(type='first3_full_representative',task=cf_task['key'],native=[evidence.keep(s) for s in states[:3]],CF=[evidence.keep(s) for s in cf[:3]],CF_head=evidence.keep(cfheads[4].reshape(1,1,-1)),exact=True))
     refs[name]=(heads,cfheads);del cf
     if representative:
      same,sameheads,samerecord=forward(task,'patch',donor,identity_map=True,purpose='identity_replay');check_begin=begin();flags=[raw_equal(a,b) for a,b in zip(states,same)]+[raw_equal(heads[4],sameheads[4])];assert bool(torch.stack(flags).all().item());end('checks/identity',check_begin);checks_evidence.append(dict(type='identity_exact',task=task['key'],native=native_snap,identity=snap(same,sameheads),exact=True));del same,sameheads
     del states,donor
    scoring_begin=begin();prior={name:lp(z) for name,z in dict(N0_native1=refs['N0'][0][1],N0_native4=refs['N0'][0][4],N0_Fixed=refs['N0'][0][4]+.5*(refs['N0'][0][4]-refs['N0'][0][1])).items()}
    n1=torch.stack([refs[n][0][1] for n in ['N0','N1','N2']]);n4=torch.stack([refs[n][0][4] for n in ['N0','N1','N2']]);b4=torch.stack([refs[n][1][4] for n in ['N1','N2']]);end('projection_and_scoring/reference_setup',scoring_begin)
    for qkey in group['Q_consumers']:
     task=tasks[qkey];states,q,record=forward(task,'off');projection_begin=begin();out,diag=torch_readouts(q[1],q[4],n1,n4,b4,vp);end('projection_and_scoring/fullV_estimators_controls',projection_begin)
     if not real_math_checked:
      check_begin=begin();independent,nd=numpy_readouts(q[1].cpu().numpy(),q[4].cpu().numpy(),n1.cpu().numpy(),n4.cpu().numpy(),b4.cpu().numpy(),vp.cpu().numpy());error=max(float(np.max(np.abs(independent[k]-z.cpu().numpy()))) for k,z in out.items());assert error<=1e-6;diag_error=max(float(np.max(np.abs(np.asarray(nd[k])-diag[k].detach().cpu().numpy()))) for k in ['alpha','numerator','denominator','main_norm2','scale_g2','scale_r2','scale_rho']);assert diag_error<=1e-10;end('checks/independent_math',check_begin);checks_evidence.append(dict(type='first_real_math',task=qkey,max_FP32_output_error=error,max_FP64_diag_error=diag_error,q1=evidence.keep(q[1].reshape(1,1,-1)),q4=evidence.keep(q[4].reshape(1,1,-1)),n1=evidence.keep(n1),n4=evidence.keep(n4),b4=evidence.keep(b4)));real_math_checked=True
     if qkey==checks['exact_duplicate_Q_task']:
      same,sameq,samerecord=forward(task,'off',purpose='exact_duplicate');check_begin=begin();flags=[raw_equal(a,b) for a,b in zip(states,same)]+[raw_equal(q[k],sameq[k]) for k in [1,4,5]];assert bool(torch.stack(flags).all().item());end('checks/duplicate',check_begin);checks_evidence.append(dict(type='first_actual_duplicate_exact',task=qkey,A=snap(states,q,False),B=snap(same,sameq,False),exact=True));duplicate_checked=True;del same,sameq
     scoring_begin=begin();top=q[4].reshape(1,-1).softmax(-1).topk(2,dim=-1).values[0];out.update(native1=q[1],native4=q[4],native5=q[5],adaptive=q[4]+(1-(top[0]-top[1]))*(q[4]-q[1]));logs={k:lp(z) for k,z in out.items()};packed={k:(v.detach().cpu().item() if isinstance(v,torch.Tensor) else v) for k,v in diag.items()};diagnostics[qkey]=packed
     for use in uses[qkey]:
      c=choices[(use['id'],use['candidate'])];t=use['Q_position'];target=c['Q_target_ids'][t]
      for method,z in logs.items():buffers.put(use['id'],use['candidate'],method,t,z[target])
      for method,z in prior.items():buffers.put(use['id'],use['candidate'],method,t,z[c['reference_target_ids']['N0'][t]])
     end('projection_and_scoring/normalization_buffers_diagnostics',scoring_begin);del states,q,out,logs
    completed_groups+=1;io_begin=time.monotonic();save('results/r28_PROGRESS.json',dict(completed_groups=completed_groups,target_groups=389,base_tasks=len(done),forwards=count,elapsed_seconds=time.monotonic()-started));timing['IO/progress_json']=timing.get('IO/progress_json',0.)+time.monotonic()-io_begin;del refs,n1,n4,b4,prior
   assert duplicate_checked and real_math_checked and not nohook and done==set(tasks) and count==2371
 except BaseException as error:fatal=repr(error)
 finally:
  cleanup=getattr(controller,'cleanup',dict(handles_removed=True,original_hooks_restored=True,errors=[]));cleanup['all_model_hooks_empty']=not any(m._forward_hooks or m._forward_pre_hooks for m in model.modules());hash_begin=begin();after={n:tensor_sha256(p) for n,p in model.named_parameters()};end('hash_and_identity/after',hash_begin)
  if not cleanup['all_model_hooks_empty'] or not cleanup['original_hooks_restored'] or cleanup['errors'] or before!=after:fatal=fatal or 'Cleanup or parameter integrity failure'
  evidence.write('results/r28_REPRESENTATIVE_VECTORS.npz');native_cache.write('results/r28_NATIVE_SINGLE_POSITION_STATES.npz');io_begin=time.monotonic();save('results/r28_NATIVE_STATE_INDEX.json',dict(tasks=native_index,weights_sha256=spec['model_identity']['safetensors_sha256'],parameter_hash_digest=digest(before),cache_contract_sha256=spec['cache_contract_sha256'],future_reuse_requires_separate_review=True))
  save('results/r28_EXECUTION_AUDIT.json',dict(calls=calls,checks=checks_evidence,cleanup=cleanup,fatal=fatal,base_tasks_completed=len(done),groups_completed=completed_groups,source_map_sha256=mapsha));timing['IO/final_metadata_json']=time.monotonic()-io_begin
  if fatal is None:
   reduce_begin=begin();finished=buffers.finish();result=candidate_readouts(rows,finished);end('projection_and_scoring/final_FP32_sums_R6',reduce_begin);io_begin=time.monotonic();save('results/r28_SCORES.json',dict(rows=result,diagnostics=diagnostics,scope='Label-free full24 true-prefix scalar outputs; no gold/predictions'));timing['IO/scores_json']=time.monotonic()-io_begin
  measured=sum(timing.values());elapsed=time.monotonic()-started;assert measured<=elapsed+1e-6
  save('results/r28_COMPLETE.json',dict(status='complete' if fatal is None else 'invalid',n=24 if fatal is None else None,forwards=count,base_tasks=len(done),groups=completed_groups,protocol_sha256=sha(LOCK),scores_sha256=sha('results/r28_SCORES.json') if fatal is None else None,audit_sha256=sha('results/r28_EXECUTION_AUDIT.json'),representative_vectors_sha256=sha('results/r28_REPRESENTATIVE_VECTORS.npz'),native_states_sha256=sha('results/r28_NATIVE_SINGLE_POSITION_STATES.npz'),native_state_index_sha256=sha('results/r28_NATIVE_STATE_INDEX.json'),all_parameter_hashes_before=before,all_parameter_hashes_after=after,cleanup=cleanup,fatal=fatal,elapsed_seconds=elapsed,component_seconds=timing,measured_components_sum=measured,unclassified_worker_residual_before_final_receipt=elapsed-measured,timing_scope='Mutually exclusive diagnostic pipeline categories; final supervisor actual is the total, not sum plus ledger. Forward includes required hooks; not uninstrumented deployment throughput. Unclassified residual includes Python dispatch/cleanup and source-record bookkeeping; final receipt/exit/supervisor overhead remains outside this pre-receipt snapshot.',peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),precision='All per-target log_softmax/gather and original-position buffers/one Torch sum FP32; original character denominator once FP64; unchanged Q4-Q5',source_bound_runtime_identity=identity,heldout_accessed=False))
 if fatal is not None:raise RuntimeError(fatal)
if __name__=='__main__':main()
