"""Independent bare-hook replay of frozen 64 R-producer upstream cuts."""
import os,json,gzip,pathlib,hashlib,fcntl
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';P=E/'state_intervention/r_producer/discovery64';PR=E/'state_intervention/discovery64';CAT=B/'results/compact_input_catalog.jsonl.gz';MODEL=ROOT/'model/model.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def dh(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
 pp=P/'PENDING.json';pending=json.loads(pp.read_text());sel=json.loads((B/'orders/E1_DISCOVERY64_FROZEN.json').read_text());expected={('train',v['id']) for group in sel['groups'].values() for v in group};assert pending['status']=='R_PRODUCER_DISCOVERY64_PENDING_INDEPENDENT_REPLAY' and len(pending['cases'])==len(expected)==64 and {(c['split'],c['id']) for c in pending['cases']}==expected;sources=pending['source_sha256'];assert all(sha(pathlib.Path(k))==v for k,v in sources.items());assert torch.__version__==pending['torch'] and torch.version.cuda==pending['cuda'] and torch.cuda.get_device_name()==pending['gpu']
 with gzip.open(CAT,'rt') as f:cat={(r['split'],r['id']):r for r in map(json.loads,f)}
 oldpending=json.loads((PR/'PENDING_INDEPENDENT_AUDIT.json').read_text());links={(x['split'],x['id']):x for x in oldpending['cases']};lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);model=None;old=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model=AutoModelForCausalLM.from_pretrained(MODEL.parent,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;old=model.model.total_ut_steps;model.model.total_ut_steps=2;cases=[]
  with torch.inference_mode():
   for q in pending['cases']:
    key=hashlib.sha256((q['split']+'\0'+q['id']).encode()).hexdigest();mp=P/(key+'.json');tp=P/(key+'.safetensors');assert sha(mp)==q['metadata_sha256'] and sha(tp)==q['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));r=cat[(q['split'],q['id'])];assert (m['id'],m['split'],m['R'],m['targets'],m['input_ids_sha256'])==(r['id'],r['split'],r['R'],r['targets'],r['input_ids_sha256']) and set(m['cuts'])=={'0','8','16'};oldp=PR/(key+'.safetensors');oldm=PR/(key+'.json');assert sha(oldp)==m['prior_tensor_sha256']==links[(q['split'],q['id'])]['tensor_sha256'] and sha(oldm)==m['prior_metadata_sha256']==links[(q['split'],q['id'])]['metadata_sha256'];prior=load_file(str(oldp));assert bits(t['native2_logits'],prior['native2_logits']) and bits(t['target17_logits'],prior['rollback_full_logits']);ids=torch.tensor([r['input_ids']],device='cuda');R=r['R'];results={}
    for cut in (0,8,16):
     prefix=f'cut{cut}';calls=[];targetcalls=[];active=[];got=[]
     def cut_pre(mod,args,kw):
      u=kw['current_ut'];assert u==len(calls);calls.append(u)
      if u!=1:return
      x=args[0];assert dh(x)==m['cuts'][str(cut)]['cut_input_full_sha256'] and bits(x[...,R,:].cpu(),t[prefix+'_cut_original_R']);v=x.clone();v[...,R,:]=t[prefix+'_cut_after_R'].cuda();mask=torch.arange(x.shape[-2],device=x.device)!=R;assert bits(x[...,mask,:],v[...,mask,:]);return (v,)+args[1:],kw
     def target_pre(mod,args,kw):
      u=kw['current_ut'];assert u==len(targetcalls);targetcalls.append(u)
      if u!=1:return
      x=args[0];assert dh(x)==m['cuts'][str(cut)]['target_input_full_sha256'] and bits(x[...,R,:].cpu(),t[prefix+'_target_R']);clean=prior['native2_input_full'].to(x.device);mask=torch.arange(x.shape[-2],device=x.device)!=R;assert bits(x[...,mask,:],clean[...,mask,:]);active.append(1)
     def target_norm(mod,args,out):
      if active:assert not got and bits(out[...,R,:].cpu(),t[prefix+'_a_R']);got.append(1)
     def target_post(mod,args,kw,out):
      if kw['current_ut']==1:assert active.pop()==1
     layers=model.model.layers;hs=[layers[cut].register_forward_pre_hook(cut_pre,with_kwargs=True),layers[17].register_forward_pre_hook(target_pre,with_kwargs=True),layers[17].input_layernorm_2.register_forward_hook(target_norm),layers[17].register_forward_hook(target_post,with_kwargs=True)]
     try:z=model(ids,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:]
     finally:
      for h in hs:h.remove()
     assert calls==targetcalls==[0,1] and not active and got==[1] and bits(z.cpu(),t[prefix+'_logits']);scores=[float(z.float().log_softmax(-1).gather(-1,torch.tensor([[i]],device=z.device)).sum()) for i in r['targets']];assert scores==m['cuts'][str(cut)]['candidate_full_vocab_logprob'];results[str(cut)]='FULL_LOGITS_A_SCORES_AND_FULL_INPUT_HASH_MATCH'
    cases.append({'id':q['id'],'split':q['split'],'cuts':results});print('R_PRODUCER_REPLAY',len(cases),q['id'],flush=True)
  assert all(sha(pathlib.Path(k))==v for k,v in sources.items());rec={'status':'R_PRODUCER_DISCOVERY64_INDEPENDENT_REPLAY_PASSED','scope':'Independently redoes model forwards, verifies full input hashes and R/non-R checks; full inputs were not stored for offline reconstruction.','cases':cases,'pending_sha256':sha(pp),'script_sha256':sha(pathlib.Path(__file__))};out=P/'CUDA_REPLAY.json'
  with out.open('x') as f:json.dump(rec,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  print('R_PRODUCER_64_REPLAY_PASS',sha(out),flush=True)
 finally:
  if model is not None and old is not None:model.model.total_ut_steps=old
  os.close(lock)
if __name__=='__main__':main()
