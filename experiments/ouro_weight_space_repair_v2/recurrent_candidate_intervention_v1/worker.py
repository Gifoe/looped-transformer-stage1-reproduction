"""Native loop readouts under frozen input interventions; no labels or argmax."""
import sys,time,os,inspect
from pathlib import Path
from core import ROOT,OUT,H,read,save,gate,scalar_scores
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'recurrent_decision_audit_v1'))
from collect_a4 import all_heads

def main():
 started=time.monotonic();spec=gate();plan=read(OUT/'INFERENCE_TASK_PLAN.json');texts=read(OUT/'INFERENCE_REQUEST_TEXTS.json');assert H((OUT/'INFERENCE_REQUEST_TEXTS.json').read_bytes())==plan['request_texts_sha256']
 def goldfree(x):
  if isinstance(x,dict):
   assert not set(x)&{'gold','correct','expected_index','expected_display_index','expected_original_content_index','condition','semantic_decision'}
   for v in x.values():goldfree(v)
  elif isinstance(x,list):
   for v in x:goldfree(v)
 goldfree(plan);goldfree(texts)
 for n,h in plan['model_identity']['metadata_and_tokenizer_sha256'].items():assert H((ROOT/'models/Ouro-1.4B'/n).read_bytes())==h,n
 assert H((ROOT/'models/Ouro-1.4B/model.safetensors').read_bytes())==plan['model_identity']['safetensors_sha256']
 from runtime import load_model,encode_pair,check_budget
 from ouro_cached_scoring import OuroCachedScorer
 from weight_edits import tensor_sha256
 import torch,numpy as np
 tok,model=load_model();assert H(Path(inspect.getfile(type(model))).read_bytes())=='c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413';scorer=OuroCachedScorer(model,4)
 for task in plan['tasks']:
  assert len(task['positions'])>0 and all(0<=i<len(task['input_ids']) for i in task['positions'])
  for c in task['consumers']:
   text=texts[c['consumer_id']];p,t=encode_pair(tok,text['context'],text['continuation']);assert p+t[:-1]==task['input_ids'] and t==c['target_ids'] and task['positions']==list(range(len(p)-1,len(p)+len(t)-1))
 save('INPUT_AUDIT.json',dict(status='all784runtime_HF_encode_pair_exact',consumers=sum(len(t['consumers']) for t in plan['tasks']),tasks=len(plan['tasks']),plan_sha256=H((OUT/'INFERENCE_TASK_PLAN.json').read_bytes()),request_texts_sha256=H((OUT/'INFERENCE_REQUEST_TEXTS.json').read_bytes()),gold_or_sidecar_deserialized=False))
 def params():return {n:tensor_sha256(p) for n,p in model.named_parameters()}
 before=params();assert len(before)==269;assert before==read(ROOT/'recurrent_decision_audit_v1/a4_COMPLETE.json')['parameter_hashes_after'];hooks={n:(len(m._forward_hooks),len(m._forward_pre_hooks)) for n,m in model.named_modules()};torch.cuda.reset_peak_memory_stats();arrays={};index={};timings=[];headcalls=0;bodycalls=0;checks=[]
 def vector(t,kind):
  raw=t.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy().copy();key=H((kind+str(raw.shape)).encode()+raw.tobytes());arrays.setdefault(key,raw);index[key]=dict(kind=kind,dtype='BF16_raw_uint16',shape=list(raw.shape));return key
 def run(task,fh):
  nonlocal headcalls,bodycalls
  check_budget(5);scorer.kmax=4;scorer._set_depth(4);torch.cuda.synchronize();start=time.monotonic();_,states,_=scorer._run(task['input_ids'],positions=task['positions'],use_cache=False);assert len(states)==4 and all(s.shape==(1,len(task['positions']),2048) for s in states);z=all_heads(model,states);torch.cuda.synchronize();forwardtime=time.monotonic()-start;bodycalls+=1;headcalls+=4
  normalstart=time.monotonic();lp=[v.log_softmax(-1) for v in z];norm=[v.double().logsumexp(-1).detach().cpu().tolist() for v in z];scores={}
  for c in task['consumers']:
   cid=c['consumer_id'];values=[scalar_scores(a,b,c['target_ids'],c['char_denominator']) for a,b in zip(lp,z)];scores[cid]=values
   if cid in plan['historical_endpoint_references']:
    ref=plan['historical_endpoint_references'][cid];fixed=z[3]+.5*(z[3]-z[0]);flp=fixed.log_softmax(-1);fixedsum=scalar_scores(flp,fixed,c['target_ids'],c['char_denominator'])['sum_logprob'];assert values[0]['sum_logprob']==ref['K1'] and values[3]['sum_logprob']==ref['K4'] and fixedsum==ref['Fixed'],('historical native1/4/Fixed parity',cid);checks.append(dict(consumer_id=cid,K1_exact=True,K4_exact=True,Fixed_exact=True))
  torch.cuda.synchronize();scoretime=time.monotonic()-normalstart;io=time.monotonic();statekeys=[vector(s,'normalized_state') for s in states];headkeys=[vector(v,'head_logits') for v in z];ioreftime=time.monotonic()-io
  rec=dict(task_key=task['task_key'],scores=scores,state_keys=statekeys,head_keys=headkeys,fullV_log_normalizer_FP64=norm);fh.write(__import__('json').dumps(rec,allow_nan=False)+'\n');fh.flush();os.fsync(fh.fileno());timings.append(dict(task_key=task['task_key'],length=len(task['input_ids']),prediction_positions=len(task['positions']),forward_and_four_heads_seconds=forwardtime,scoring_diagnostics_seconds=scoretime,vector_export_seconds=ioreftime));save('PROGRESS.json',dict(completed_tasks=bodycalls,planned_tasks=len(plan['tasks']),native_heads=headcalls,phase='unchanged_baseline' if bodycalls<=24 else 'interventions'))
 tasks=plan['tasks'];assert [t['task_key'] for t in tasks[:24]]==plan['baseline_task_keys'] and len(tasks)==218
 with (OUT/'SCORES.jsonl').open('x',encoding='utf8') as fh:
  for task in tasks[:24]:run(task,fh)
  assert len(checks)==96;save('PARITY.json',dict(status='all24_P0_native1_native4_Fixed_exact_before_interventions',checks=checks.copy(),body_calls=24))
  warm=timings[1:24];rate=max(t['forward_and_four_heads_seconds']/t['length']**2 for t in warm);remaining_tasks=tasks[24:];extra=sum(max(0,len(t['positions'])-1) for t in remaining_tasks);targetreserve=20.+.02*4*extra;forecast=1.5*rate*sum(len(t['input_ids'])**2 for t in remaining_tasks)+60.+targetreserve;remaining=float(os.environ['GPU_DEADLINE_UNIX'])-time.time();admitted=forecast+5<remaining
  save('RUNTIME_ADMISSION.json',dict(admitted=admitted,warm_rate_seconds_per_L2=rate,safety=1.5,forecast_remaining_seconds=forecast,remaining_to_soft=remaining,final_reserve=60.,head_target_reserve=targetreserve,remaining_tasks=len(remaining_tasks),rule='predeclared max of P0warm23 seconds/L² *1.5 *remaining sumL² +final60 +headtarget20+0.02perextraheadposition; require5 additional sec'))
  if admitted:
   for task in remaining_tasks:run(task,fh)
 if admitted:assert bodycalls==218 and headcalls==872 and len(checks)==192
 after=params();assert before==after and hooks=={n:(len(m._forward_hooks),len(m._forward_pre_hooks)) for n,m in model.named_modules()};np.savez(OUT/'VECTORS.npz',**arrays);assert (OUT/'VECTORS.npz').stat().st_size<=160*1024**2;save('VECTOR_INDEX.json',index);save('TIMINGS.json',timings);gate()
 save('COMPLETE.json',dict(status='complete_all218_tasks' if admitted else 'P0_only_runtime_not_admitted_no_effects',tasks=bodycalls,native_heads=headcalls,head_prediction_positions=4*sum(len(t['positions']) for t in tasks[:bodycalls]),historical_parity_checks=checks,parameter_hashes_before=before,parameter_hashes_after=after,cleanup_exact=True,SCORES_sha256=H((OUT/'SCORES.jsonl').read_bytes()),VECTORS_sha256=H((OUT/'VECTORS.npz').read_bytes()),VECTOR_INDEX_sha256=H((OUT/'VECTOR_INDEX.json').read_bytes()),protocol_sha256=H((OUT/'EXECUTION_LOCK.json').read_bytes()),plan_sha256=H((OUT/'INFERENCE_TASK_PLAN.json').read_bytes()),worker_elapsed_seconds=time.monotonic()-started,peak_allocator_bytes=int(torch.cuda.max_memory_allocated()),allocator_scope='GB10 unified memory diagnostic pipeline, not deployment throughput or independent GPU memory capacity',labels_read=False))
 print('B_TERMINAL', 'complete' if admitted else 'P0_only_no_interventions',bodycalls,headcalls,flush=True)
if __name__=='__main__':main()
