"""Independent CUDA replay of saved E2.3 injected states and full logits."""
import os,json,gzip,hashlib,pathlib,fcntl,time
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';OUT=E/'state_intervention/technical_three';CAT=B/'results/compact_input_catalog.jsonl.gz';MODEL=ROOT/'model/model.safetensors'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def main():
 assert (OUT/'PENDING_INDEPENDENT_AUDIT.json').exists() and not (OUT/'CUDA_REPLAY.json').exists();pending=json.loads((OUT/'PENDING_INDEPENDENT_AUDIT.json').read_text());assert pending['status']=='TECHNICAL_PENDING_INDEPENDENT_AUDIT';assert sha(CAT)=='a38218bfe0c9940f326663b8ecc0447f875a18c07f0091813ddfef4a3d12342d' and sha(MODEL)=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
 with gzip.open(CAT,'rt') as f:cat={(r['split'],r['id']):r for r in map(json.loads,f)}
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);model=None;old=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;old=model.model.total_ut_steps;cases=[]
  with torch.inference_mode():
   for item in pending['cases']:
    name=hashlib.sha256((item['split']+'\0'+item['id']).encode()).hexdigest();mp=OUT/(name+'.json');tp=OUT/(name+'.safetensors');assert sha(mp)==item['metadata_sha256'] and sha(tp)==item['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));r=cat[(item['split'],item['id'])];ids=torch.tensor([r['input_ids']],device='cuda');R=r['R'];res={}
    for cond,record in m['conditions'].items():
     if record['status']!='OK':res[cond]='UNDEFINED';continue
     ut=0 if cond=='forward_add' else 1;model.model.total_ut_steps=ut+1;calls=[]
     def pre(module,args,kw):
      u=kw['current_ut'];calls.append(u)
      if u!=ut:return
      x=args[0];expected=t[f'native{ut+1}_input_full'].to(x.device);assert bits(x,expected);patched=x.clone();patched[...,R,:]=t[cond+'_injected_R'].to(x.device);outside=torch.arange(x.shape[-2],device=x.device)!=R;assert bits(x[...,outside,:],patched[...,outside,:]);return (patched,)+args[1:],kw
     h=model.model.layers[17].register_forward_pre_hook(pre,with_kwargs=True)
     try:z=model(ids,labels=None,use_cache=False,exit_at_step=ut).logits[:,R,:]
     finally:h.remove()
     assert calls==list(range(ut+1)) and bits(z.cpu(),t[cond+'_logits']);res[cond]='BITWISE_FULL_LOGITS_MATCH'
    cases.append({'id':item['id'],'split':item['split'],'conditions':res})
  receipt={'status':'CUDA_INDEPENDENT_SAVED_STATE_LOGITS_REPLAY_PASSED','scope':'Independent bare prehook replays saved R vectors and checks complete [1,V] logits exactly; same hardware and library, not an independent scientific validation.','cases':cases,'model_sha256':sha(MODEL),'catalog_sha256':sha(CAT),'script_sha256':sha(pathlib.Path(__file__)),'pending_sha256':sha(OUT/'PENDING_INDEPENDENT_AUDIT.json')};p=OUT/'CUDA_REPLAY.json'
  with p.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  print('CUDA_REPLAY_PASS',sha(p),flush=True)
 finally:
  if model is not None and old is not None:model.model.total_ut_steps=old
  os.close(lock)
if __name__=='__main__':main()
