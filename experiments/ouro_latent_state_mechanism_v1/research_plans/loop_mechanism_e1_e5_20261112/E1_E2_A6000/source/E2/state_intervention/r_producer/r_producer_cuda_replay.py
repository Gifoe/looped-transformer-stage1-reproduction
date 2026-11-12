"""Independent bare-hook R-producer technical disk replay."""
import os,json,gzip,pathlib,hashlib,fcntl
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=pathlib.Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';E=B/'E2';P=E/'state_intervention/r_producer/technical_three';CAT=B/'results/compact_input_catalog.jsonl.gz';MODEL=ROOT/'model/model.safetensors';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def main():
 pp=P/'PENDING.json';pending=json.loads(pp.read_text());assert pending['status']=='R_PRODUCER_THREE_PENDING_INDEPENDENT_REPLAY' and len(pending['cases'])==3 and {(c['split'],c['id']) for c in pending['cases']}=={('test','ACTAAP_2013_7_16'),('test','Mercury_405804'),('test','MSA_2012_8_30')};sources=pending['source_sha256'];assert all(sha(pathlib.Path(k))==v for k,v in sources.items());assert torch.__version__==pending['torch'] and torch.version.cuda==pending['cuda'] and torch.cuda.get_device_name()==pending['gpu']
 with gzip.open(CAT,'rt') as f:cat={(r['split'],r['id']):r for r in map(json.loads,f)}
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600);model=None;old=None
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);model=AutoModelForCausalLM.from_pretrained(MODEL.parent,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;old=model.model.total_ut_steps;model.model.total_ut_steps=2;cases=[]
  with torch.inference_mode():
   for q in pending['cases']:
    key=hashlib.sha256((q['split']+'\0'+q['id']).encode()).hexdigest();mp=P/(key+'.json');tp=P/(key+'.safetensors');assert sha(mp)==q['metadata_sha256'] and sha(tp)==q['tensor_sha256'];m=json.loads(mp.read_text());t=load_file(str(tp));r=cat[(q['split'],q['id'])];assert (m['id'],m['split'],m['targets'],m['R'])==(r['id'],r['split'],r['targets'],r['R']);assert set(m['cuts'])=={'0','8','16','17'};ids=torch.tensor([r['input_ids']],device='cuda');R=r['R'];result={}
    for cut in (0,8,16,17):
     for name in ('clean','producer','restore_target','target_only'):
      prefix=f'cut{cut}_{name}';events=[];target_events=[];active=[];got=[]
      def put(x,patch):
       y=x.clone();y[...,R,:]=patch.to(x.device);mask=torch.arange(x.shape[-2],device=x.device)!=R;assert bits(x[...,mask,:],y[...,mask,:]);return y
      def pre(mod,args,kw):
       u=kw['current_ut'];assert u==len(events);events.append(u)
       if u!=1:return
       x=args[0];assert bits(x.cpu(),t[prefix+'_cut_before_full']);y=put(x,t[prefix+'_cut_after_R']) if name in ('producer','restore_target') else x
       if cut==17:
        target_events.append(1);assert bits(y.cpu(),t[prefix+'_target_before_full']);y=put(y,t[prefix+'_target_after_R']) if name in ('restore_target','target_only') else y;active.append(1)
       if name in ('producer','restore_target') or (cut==17 and name=='target_only'):return (y,)+args[1:],kw
      def target_pre(mod,args,kw):
       u=kw['current_ut'];assert u==len(target_events);target_events.append(u)
       if u!=1:return
       x=args[0];assert bits(x.cpu(),t[prefix+'_target_before_full']);active.append(1)
       if name in ('restore_target','target_only'):return (put(x,t[prefix+'_target_after_R']),)+args[1:],kw
      def anorm(mod,args,out):
       if active:assert not got and bits(out[...,R,:].cpu(),t[prefix+'_a_R']);got.append(1)
      def post(mod,args,kw,out):
       if kw['current_ut']==1:assert active.pop()==1
      layers=model.model.layers;hs=[layers[cut].register_forward_pre_hook(pre,with_kwargs=True),layers[17].input_layernorm_2.register_forward_hook(anorm),layers[17].register_forward_hook(post,with_kwargs=True)]
      if cut!=17:hs.append(layers[17].register_forward_pre_hook(target_pre,with_kwargs=True))
      try:z=model(ids,labels=None,use_cache=False,exit_at_step=1).logits[:,R,:]
      finally:
       for h in hs:h.remove()
      assert events==[0,1] and target_events==([1] if cut==17 else [0,1]) and not active and got==[1] and bits(z.cpu(),t[prefix+'_logits']);scores=[float(z.float().log_softmax(-1).gather(-1,torch.tensor([[i]],device=z.device)).sum()) for i in r['targets']];assert scores==m['cuts'][str(cut)]['conditions'][name]['candidate_full_vocab_logprob'];result[prefix]='BITWISE_FULL_LOGITS_A_AND_SCORE_MATCH'
    cases.append({'id':q['id'],'split':q['split'],'conditions':result})
  assert all(sha(pathlib.Path(k))==v for k,v in sources.items());rec={'status':'R_PRODUCER_THREE_INDEPENDENT_REPLAY_PASSED','scope':'Independent bare hooks; checks saved full logits/a/scores and actual full cut/target inputs. Structural, not natural path identification.','cases':cases,'pending_sha256':sha(pp),'script_sha256':sha(pathlib.Path(__file__))};out=P/'CUDA_REPLAY.json'
  with out.open('x') as f:json.dump(rec,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
  print('R_PRODUCER_THREE_REPLAY_PASS',sha(out),flush=True)
 finally:
  if model is not None and old is not None:model.model.total_ut_steps=old
  os.close(lock)
if __name__=='__main__':main()
