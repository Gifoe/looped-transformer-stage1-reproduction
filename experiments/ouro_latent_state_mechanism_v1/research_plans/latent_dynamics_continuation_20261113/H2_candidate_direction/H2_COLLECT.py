"""Natural per-layer candidate-direction lens; no gold or state intervention."""
import argparse,json,hashlib,time,fcntl,os,inspect
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(x,allow_nan=False));os.replace(t,p)
def direction(b,scores):
 out=[None];zeros=0
 for k in range(1,4):
  d=b[k]-b[k-1];order=np.argsort(-np.asarray(scores[k]),kind='stable');t,u=order[:2];v=d[:,t]-d[:,u];f=[]
  for i in range(7):f.extend([float(v[i]),float(np.sqrt(np.mean(d[i]**2)))])
  ds=d.sum(0);ss=np.array(scores[k])-np.array(scores[k-1]);ss-=ss.mean();den=np.linalg.norm(ds)*np.linalg.norm(ss)
  if den==0:zeros+=1
  out.append(f+[float(np.mean(v>0)),float(np.dot(ds,ss)/den) if den else 0.])
 return out,zeros
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);p.add_argument('--model',required=True);p.add_argument('--native-root',required=True);p.add_argument('--technical',action='store_true');p.add_argument('--budget-seconds',type=float,default=1800);a=p.parse_args()
 import torch,transformers
 transformers.utils.import_utils._torchvision_available=False
 from transformers import AutoModelForCausalLM
 torch.set_grad_enabled(False);torch.set_num_threads(1);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False
 lock=open('/root/autodl-tmp/ouro-e6a/gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.monotonic();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);data=[json.loads(l) for l in Path(a.inputs).read_text().splitlines()];assert all(q['split'] in ['train','validation'] and 'gold' not in q for q in data)
 assert sha(Path(a.model)/'model.safetensors')=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
 model=AutoModelForCausalLM.from_pretrained(a.model,trust_remote_code=True,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False
 identity={'collector_sha256':sha(__file__),'input_sha256':sha(a.inputs),'model_sha256':sha(Path(a.model)/'model.safetensors'),'loaded_model_source_sha256':sha(inspect.getfile(model.__class__)),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'lens':'CPU FP32 checkpoint final RMSnorm and candidate output rows; centered post-minus-pre; actual BF16 residual-add output observed','layers':{'target':list(range(10,17)),'control':list(range(2,9))},'loop1':'unavailable None; diagnostics only depth2+'}
 assert identity['loaded_model_source_sha256']=='c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413'
 write(out/'IDENTITY.json',identity);versions=[(n,id(v),v._version) for n,v in model.named_parameters()];norm=model.model.norm.weight.detach().float().cpu().numpy();eps=np.float32(model.model.norm.variance_epsilon);alltargets=sorted({t for q in data for t in q['targets']});head={t:model.lm_head.weight[t].detach().float().cpu().numpy() for t in alltargets};write(out/'LENS_OPERATOR.json',{'norm_weight_sha256':hashlib.sha256(norm.tobytes()).hexdigest(),'epsilon':float(eps),'candidate_row_sha256':{str(t):hashlib.sha256(v.tobytes()).hexdigest() for t,v in head.items()},'arithmetic':'FP32 x / sqrt(mean(x*x)+eps) * normweight; candidate matrix product; output stored FP32; temporal differences FP32; scalar summaries float64 output-score cosine','source':identity})
 def th(z):return hashlib.sha256(z.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
 times=[];records=[]
 for i,q in enumerate(data):
  if time.monotonic()-start>a.budget_seconds:raise RuntimeError('Explicit cumulative GPU task wall budget reached; no automatic restart')
  key=q['split']+'_'+hashlib.sha256(q['id'].encode()).hexdigest();file=out/(key+'.json');assert not file.exists(),'Never silently overwrite or restart a completed question'
  native=json.loads((Path(a.native_root)/q['split']/(key+'.json')).read_text());assert native['input_ids_sha256']==q['input_ids_sha256'];R=q['R'];ids=torch.tensor([q['ids']],device='cuda');states={};active={};handles=[]
  for li in list(range(1,8))+list(range(9,16)):
   layer=model.model.layers[li]
   def pre(m,args,kwargs,j=li):
    u=kwargs['current_ut'];active[j]=u;states[u,j,'x']=args[0][:,R,:].detach().clone()
   def att(m,args,y,j=li):states[active[j],j,'a']=y[:,R,:].detach().clone()
   def added(m,args,j=li):states[active[j],j,'y']=args[0][:,R,:].detach().clone()
   handles += [layer.register_forward_pre_hook(pre,with_kwargs=True),layer.input_layernorm_2.register_forward_hook(att),layer.post_attention_layernorm.register_forward_pre_hook(added)]
  model.model.total_ut_steps=4;torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
  try:_,hidden,gates=model.model(ids,use_cache=False,past_key_values=None)
  finally:
   for h in handles:h.remove()
  assert len(states)==4*14*3;actual=[]
  for k,h in enumerate(hidden):
   z=model.lm_head(h)[0,R].detach().clone();assert th(z)==native['native'][k]['full_logits_sha256'],(q['id'],k,'matched capture endpoint');actual.append(th(z))
  torch.cuda.synchronize();seconds=time.perf_counter()-t;times.append(seconds)
  if a.technical:
   for k in range(1,5):
    model.model.total_ut_steps=k;z=model(ids,use_cache=False,exit_at_step=k-1).logits[0,R];assert th(z)==actual[k-1],('unhooked independent depth',q['id'],k)
  cput=time.perf_counter();arrays={};features={};zero={};w=np.stack([head[t] for t in q['targets']]);scores=[n['candidate_logits'] for n in native['native']]
  for fam,layers in [('candidate',list(range(9,16))),('unrelated',list(range(1,8)))]:
   for slot in ['x','a','y']:arrays[fam+'_'+slot]=np.stack([np.stack([states[u,j,slot].float().cpu().numpy().ravel() for j in layers]) for u in range(4)])
   # Check the recorded actual add in original BF16 arithmetic, never regenerate it for lens.
   for u in range(4):
    for j in layers:assert torch.equal(states[u,j,'x']+states[u,j,'a'],states[u,j,'y'])
   def lens(x):return ((x*np.reciprocal(np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)))*norm)@w.T
   b=lens(arrays[fam+'_y'])-lens(arrays[fam+'_x']);b-=b.mean(-1,keepdims=True);arrays[fam+'_b']=b;features[fam],zero[fam]=direction(b,scores)
  np.savez_compressed(out/(key+'.npz'),**arrays)
  row={'id':q['id'],'split':q['split'],'input_ids_sha256':q['input_ids_sha256'],'native_source_sha256':sha(Path(a.native_root)/q['split']/(key+'.json')),'targets':q['targets'],'native_full_logits_sha256':actual,'features':features,'zero_cosines':zero,'states_sha256':sha(out/(key+'.npz')),'states_bytes':(out/(key+'.npz')).stat().st_size,'gpu_window_seconds':seconds,'cpu_lens_storage_seconds':time.perf_counter()-cput,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'technical_independent_native':a.technical,'source_identity':identity}
  write(file,row);records.append({'id':q['id'],'split':q['split'],'file':str(file),'sha256':sha(file)});print('H2_CAPTURE',i+1,len(data),q['id'],round(seconds,3),flush=True);del hidden,gates,states,z
 assert versions==[(n,id(v),v._version) for n,v in model.named_parameters()]
 write(out/'COMPLETE.json',{'status':'COMPLETE','n':len(records),'body_forwards':len(records)*(5 if a.technical else 1),'full_shaped_head_calls':len(records)*(8 if a.technical else 4),'physical_loop_applications':len(records)*(14 if a.technical else 4),'task_wall_seconds':time.monotonic()-start,'production_gpu_window_seconds':sum(times),'production_max_seconds':max(times),'identity':identity,'records':records,'weights_modified':False})
 print('H2_CAPTURE_COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
