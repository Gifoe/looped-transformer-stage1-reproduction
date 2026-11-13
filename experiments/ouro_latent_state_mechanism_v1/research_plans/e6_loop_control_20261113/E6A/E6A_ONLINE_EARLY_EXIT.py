"""Actual prefix execution, with native endpoint hashes checked on each query."""
import argparse,json,time,hashlib,fcntl,os
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class StopAtPrefix(Exception):pass
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--native',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 import torch,transformers
 transformers.utils.import_utils._torchvision_available=False
 from transformers import AutoModelForCausalLM
 torch.set_grad_enabled(False);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False
 lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(Path(a.model)/'model.safetensors')=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);data=[json.loads(l) for l in Path(a.inputs).read_text().splitlines()];assert all('gold' not in q for q in data)
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;model.model.total_ut_steps=4
 versions=[(n,id(v),v._version) for n,v in model.named_parameters()];rows=[]
 def th(x):return hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 for i,q in enumerate(data):
  key=q['split']+'_'+hashlib.sha256(q['id'].encode()).hexdigest();native=json.loads((Path(a.native)/(key+'.json')).read_text());assert q['input_ids_sha256']==native['input_ids_sha256'];ids=torch.tensor([q['ids']],device='cuda');R=q['R']
  # Fixed order alternation limits systematic warm-order confounding.
  policies=[('native4',None),('early_fixed',(.9,.3)),('cal_early_0.99',(.99,.5))]
  if i%2:policies=policies[::-1]
  for name,threshold in policies:
   prefix=[];endpoint={};calls=[0]
   def count_layer(m,args,kwargs):calls[0]+=1
   layer_handle=model.model.layers[0].register_forward_pre_hook(count_layer,with_kwargs=True)
   def callback(m,args,h):
    z=model.lm_head(h)[0,R].detach().clone();s=z.float().log_softmax(-1)[q['targets']].cpu().numpy();prefix.append(s);d=len(prefix);p=np.exp(s.astype(float)-float(max(s)));p/=p.sum();pred=int(p.argmax());entropy=float(-np.sum(p*np.log(np.maximum(p,1e-300)))/np.log(len(p)))
    stop=d==4 or (d in [2,3] and int(np.argmax(prefix[-2]))==pred and p.max()>=threshold[0] and entropy<=threshold[1])
    if stop:endpoint.update(depth=d,prediction=pred,scores=s.tolist(),full_logits_sha256=th(z));raise StopAtPrefix()
   h=None
   if threshold is not None:h=model.model.norm.register_forward_hook(callback)
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
   try:
    if threshold is None:
     z=model(ids,use_cache=False,exit_at_step=3).logits[0,R].detach().clone();s=z.float().log_softmax(-1)[q['targets']];endpoint={'depth':4,'prediction':int(s.argmax()),'scores':s.cpu().tolist(),'full_logits_sha256':th(z)}
    else:
     try:model(ids,use_cache=False,exit_at_step=3)
     except StopAtPrefix:pass
   finally:
    if h is not None:h.remove()
    layer_handle.remove()
   torch.cuda.synchronize();seconds=time.perf_counter()-t;d=endpoint['depth'];assert calls[0]==d and endpoint['full_logits_sha256']==native['native'][d-1]['full_logits_sha256'],(q['id'],name,'native/prefix identity')
   rows.append({'id':q['id'],'split':q['split'],'method':name,'seconds':seconds,'first_layer_calls':calls[0],'actual_executed_loops':d,'prefix_only':True,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),**endpoint})
  print('ONLINE_PREFIX_VALIDATED',i+1,len(data),q['id'],flush=True)
 assert versions==[(n,id(v),v._version) for n,v in model.named_parameters()]
 (out/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(out/'COMPLETE.json').write_text(json.dumps({'status':'COMPLETE','n':len(data),'source_sha256':sha(__file__),'input_sha256':sha(a.inputs),'model_sha256':sha(Path(a.model)/'model.safetensors'),'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'policies':'fixed before5090 outcomes: ZL early.9 entropy.3; historical Joint development calibrated.99 entropy.5; no latency tuning','scope':'Actual terminating forward; all selected endpoints equal separately captured native prefixes. Alternating policy order, cold first question retained separately.'},indent=2))
if __name__=='__main__':main()
