"""Independent CPU cross-check of saved full-FP32 outputs and backend receipts."""
import json,pathlib,hashlib,math,statistics,torch
from safetensors.torch import load_file
R=pathlib.Path('/home/luansiyu/loop-latent');B=R/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';P=B/'E1/full_model_fp32_three_train';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pend=json.loads((P/'PENDING.json').read_text());run=json.loads((P/'RUNTIME.json').read_text());assert pend['status']=='THREE_TRAIN_FULL_FP32_PENDING_INDEPENDENT_AUDIT' and pend['ids']==['ACTAAP_2010_7_7','AKDE&ED_2008_4_41','AKDE&ED_2008_8_32'] and run['checkpoint_tensors_verified_exact_bf16_to_fp32']==269 and not run['tf32_matmul'] and not run['tf32_cudnn'];assert all(sha(pathlib.Path(k))==v for k,v in pend['source_sha256'].items());out=[];maxerr=0
for case in pend['cases']:
 key=hashlib.sha256(case['id'].encode()).hexdigest();mp=P/(key+'.json');tp=P/(key+'.safetensors');assert sha(mp)==case['metadata_sha256'] and sha(tp)==case['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));assert m['id']==case['id'];targets=m['targets'];count=0
 def check(tag,entry):
  global maxerr,count
  z=t[tag+'_logits'];assert z.dtype==torch.float32 and z.shape==(1,49152) and torch.isfinite(z).all();assert hashlib.sha256(z.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==entry['logits_sha256'];x=z.double();lse=float(torch.logsumexp(x,dim=-1)[0]);expected=[float(x[0,i])-lse for i in targets];err=max(abs(a-b) for a,b in zip(expected,entry['scores']));maxerr=max(maxerr,err);assert err<1e-4;count+=1
 for q in (1,2):
  for k in (1,2):
   for v in (1,2):tag=f'G{q}{k}{v}';check(tag,m['corners'][str(q)][tag])
  for name,entry in m['controls'][str(q)]['conditions'].items():
   if entry['status']=='OK':check(f'q{q}_{name}',entry)
 assert count>=16;assert torch.equal(t['G111_logits'],t['native1_logits']);assert torch.equal(t[f'q1_ID_logits'],t['G122_logits']) and torch.equal(t[f'q1_restore_logits'],t['G122_logits']) and torch.equal(t[f'q2_ID_logits'],t['G222_logits']) and torch.equal(t[f'q2_restore_logits'],t['G222_logits']);out.append({'id':case['id'],'conditions_checked':count,'profiler_ops':m['actual_sdpa']['profiler_ops']})
assert all(sha(pathlib.Path(k))==v for k,v in pend['source_sha256'].items());res={'status':'FULL_MODEL_FP32_SAVED_OUTPUT_CPU_AUDIT_PASSED','scope':'Independent saved FP32 logits hash/FP64 arithmetic score and identity/restore check; no independent GPU full-forward replay, no causal or rescue-group inference.','cases':out,'max_fp64_reference_vs_fp32_score_absolute_error':maxerr,'source_pending_sha256':sha(P/'PENDING.json'),'script_sha256':sha(pathlib.Path(__file__))};p=P/'CPU_SAVED_OUTPUT_AUDIT.json';assert not p.exists();p.write_text(json.dumps(res,indent=2)+'\n');print('FP32_CPU_AUDIT_PASS',sha(p),maxerr)
