"""Independent bare-hook disk replay for E2.4 technical cases; no experiment module imported."""
import os,json,gzip,pathlib,hashlib,fcntl
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';PRIOR=E/'state_intervention/technical_three';OUT=E/'state_intervention/mediation/technical_three';CAT=B/'results/compact_input_catalog.jsonl.gz';MODEL=ROOT/'model/model.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def main():
 P=OUT/'PENDING.json';pending=json.loads(P.read_text());assert pending['status']=='THREE_CASES_PENDING_INDEPENDENT_BARE_DOUBLEHOOK_REPLAY' and len(pending['cases'])==3 and {(x['split'],x['id']) for x in pending['cases']}=={('test','ACTAAP_2013_7_16'),('test','Mercury_405804'),('test','MSA_2012_8_30')};source=pending['source_sha256'];assert all(sha(pathlib.Path(k))==v for k,v in source.items());assert pending['torch']==torch.__version__ and pending['cuda']==torch.version.cuda and pending['gpu']==torch.cuda.get_device_name()
 with gzip.open(CAT,'rt') as f:cat={(r['split'],r['id']):r for r in map(json.loads,f)}
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);model=None;old=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;old=model.model.total_ut_steps;model.model.total_ut_steps=2;cases=[]
  with torch.inference_mode():
   for item in pending['cases']:
    key=hashlib.sha256((item['split']+'\0'+item['id']).encode()).hexdigest();mp=OUT/(key+'.json');tp=OUT/(key+'.safetensors');assert sha(mp)==item['metadata_sha256'] and sha(tp)==item['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));priorp=PRIOR/(key+'.safetensors');priorm=PRIOR/(key+'.json');prior_receipt=json.loads((PRIOR/'PENDING_INDEPENDENT_AUDIT.json').read_text());link=next(x for x in prior_receipt['cases'] if (x['id'],x['split'])==(item['id'],item['split']));assert sha(priorp)==m['source_tensor_sha256']==link['tensor_sha256'] and sha(priorm)==m['source_metadata_sha256']==link['metadata_sha256'];prior=load_file(str(priorp));r=cat[(item['split'],item['id'])];assert (r['id'],r['split'],r['R'],r['targets'])==(m['id'],m['split'],m['R'],m['targets']);assert set(m['conditions'])=={'C00','C11','C10','C01','full_norm','full_unrelated'};ids=torch.tensor([r['input_ids']],device='cuda');R=r['R'];res={}
    for name,record in m['conditions'].items():
     if record['status']!='OK':assert name in ('full_norm','full_unrelated') and record['status']=='UNDEFINED';res[name]='UNDEFINED';continue
     events=[];active=[];matched=[]
     def pre(mod,args,kw):
      u=kw['current_ut'];assert u==len(events);events.append(u)
      if u!=1:return
      x=args[0];assert bits(x,prior['native2_input_full'].cuda());v=t[name+'_x_injected_R'].cuda() if name.startswith('C') else t[name+'_injected_R'].cuda();out=x.clone();out[...,R,:]=v;mask=torch.arange(x.shape[-2],device=x.device)!=R;assert bits(x[...,mask,:],out[...,mask,:]);active.append(1);return (out,)+args[1:],kw
     def anorm(mod,args,out):
      if not active or not name.startswith('C'):return
      assert bits(out[...,R,:].cpu(),t[name+'_a_receiver_R']);v=t[name+'_a_injected_R'].to(out.device);patched=out.clone();patched[...,R,:]=v;mask=torch.arange(out.shape[-2],device=out.device)!=R;assert bits(out[...,mask,:],patched[...,mask,:]);matched.append(1);return patched
     def post(mod,args,kw,out):
      if kw.get('current_ut')==1:assert active.pop()==1
     hooks=[model.model.layers[17].register_forward_pre_hook(pre,with_kwargs=True),model.model.layers[17].input_layernorm_2.register_forward_hook(anorm),model.model.layers[17].register_forward_hook(post,with_kwargs=True)]
     try:z=model(ids,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:]
     finally:
      for h in hooks:h.remove()
     assert events==[0,1] and not active and (len(matched)==1 if name.startswith('C') else not matched) and bits(z.cpu(),t[name+'_logits']);scores=[float(z.float().log_softmax(-1).gather(-1,torch.tensor([[a]],device=z.device)).sum()) for a in r['targets']];assert scores==record['candidate_full_vocab_logprob'];res[name]='BITWISE_FULL_LOGITS_AND_CANDIDATES_MATCH'
    cases.append({'id':item['id'],'split':item['split'],'conditions':res})
  assert all(sha(pathlib.Path(k))==v for k,v in source.items());receipt={'status':'E2_4_THREE_INDEPENDENT_DOUBLEHOOK_CUDA_REPLAY_PASSED','scope':'Independent bare prehook and attention-output hook, full logits and candidate score replay; technical evidence only.','cases':cases,'source_pending_sha256':sha(P),'script_sha256':sha(pathlib.Path(__file__))};out=OUT/'CUDA_REPLAY.json'
  with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  print('E2_4_REPLAY_PASS',sha(out),flush=True)
 finally:
  if model is not None and old is not None:model.model.total_ut_steps=old
  os.close(lock)
if __name__=='__main__':main()
