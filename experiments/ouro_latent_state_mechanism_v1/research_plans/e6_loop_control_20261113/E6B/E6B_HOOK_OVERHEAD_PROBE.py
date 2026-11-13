"""Fixed Train3 paired hook-overhead resource probe; not method accuracy evidence."""
import argparse,json,hashlib,time,fcntl
from pathlib import Path
import torch,transformers
transformers.utils.import_utils._torchvision_available=False
from transformers import AutoModelForCausalLM

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def th(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def forward(model,q,observed):
 R=q['R'];ids=torch.tensor([q['ids']],device='cuda');states={};active={};events=[];handles=[]
 if observed:
  for j in sorted(set(range(1,8))|set(range(9,16))|{17}):
   layer=model.model.layers[j]
   def pre(m,a,k,j=j):active[j]=k['current_ut'];events.append((active[j],j));states[(active[j],j,'x')]=a[0][:,R,:].detach().clone()
   def norm(m,a,y,j=j):states[(active[j],j,'a')]=y[:,R,:].detach().clone()
   def post(m,a,k,y,j=j):states[(active[j],j,'y')]=y[:,R,:].detach().clone()
   handles.extend([layer.register_forward_pre_hook(pre,with_kwargs=True),layer.input_layernorm_2.register_forward_hook(norm),layer.register_forward_hook(post,with_kwargs=True)])
 model.model.total_ut_steps=4;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 try:_,hs,gs=model.model(ids,use_cache=False,past_key_values=None)
 finally:
  for h in handles:h.remove()
 hashes=[];scores=[];early=None
 for d,h in enumerate(hs,1):
  z=model.lm_head(h)[0,R].detach().clone();hashes.append(th(z));scores.append(z.float().log_softmax(-1)[q['targets']].cpu().tolist())
  if d==1:early=z.clone()
  if d==4:
   prob=z.float().softmax(-1).topk(2).values;strength=1-(prob[0]-prob[1])
   for amount in [strength,.5,0.]:scores.append((z.float()+amount*(z.float()-early.float())).log_softmax(-1)[q['targets']].cpu().tolist())
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 if observed:assert events==[(u,j) for u in range(4) for j in sorted(set(range(1,8))|set(range(9,16))|{17})] and len(states)==180
 return {'observed':observed,'seconds':seconds,'full_vocab_hashes':hashes,'scores':scores,'max_allocated_bytes':torch.cuda.max_memory_allocated(),'max_reserved_bytes':torch.cuda.max_memory_reserved()}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_grad_enabled(False);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False;lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert sha(Path(a.model)/'model.safetensors')=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af';rows=[json.loads(s) for s in Path(a.inputs).read_text().splitlines()];assert len(rows)==3 and all(q['split']=='train' and 'gold' not in q for q in rows)
 model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;versions=[(n,id(v),v._version) for n,v in model.named_parameters()];result=[]
 for q in rows:
  warm=[forward(model,q,False),forward(model,q,True)];assert warm[0]['full_vocab_hashes']==warm[1]['full_vocab_hashes'] and warm[0]['scores']==warm[1]['scores']
  measurements=[]
  for order in [[False,True],[True,False]]:
   pair=[forward(model,q,mode) for mode in order];assert pair[0]['full_vocab_hashes']==pair[1]['full_vocab_hashes'] and pair[0]['scores']==pair[1]['scores'];measurements.extend(pair)
  result.append({'id':q['id'],'warmups':warm,'measurements':measurements})
 assert versions==[(n,id(v),v._version) for n,v in model.named_parameters()]
 Path(a.out).write_text(json.dumps({'status':'PASS','scope':'Fixed Train3 warmed paired resource measurements, not population latency or method accuracy. GPU window covers body, four full heads, identical LoopCD confidence/scoring and endpoint hashing; CPU extraction/storage excluded.','source_sha256':sha(__file__),'inputs_sha256':sha(a.inputs),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'model_body_forwards':18,'native_head_calls':72,'rows':result},indent=2));print('HOOK_OVERHEAD_COMPLETE',flush=True)
if __name__=='__main__':main()
