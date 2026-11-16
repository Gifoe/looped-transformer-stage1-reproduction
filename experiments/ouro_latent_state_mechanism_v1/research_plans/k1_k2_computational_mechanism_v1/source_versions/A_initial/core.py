"""Full-shaped native Ouro forwards. No gold, fitting, layer search, or custom attention."""
import sys,pathlib,json,hashlib,time,contextlib
ROOT=pathlib.Path('/root/ouro-computation-sol-20261116')
sys.path.insert(0,str(ROOT/'prior/code'))
import torch
from ab1_loader_draft import load_model,verify_source
from ab1_metrics import score
from ab1_boundary_hooks import bits,_hidden,_replace,actual_args
MODEL='/root/autodl-tmp/ouro-e4-e5/model'
def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def jsave(p,obj):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False));t.replace(p)
def tsave(p,obj):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');torch.save(obj,t);t.replace(p)
def cpu(x):return x.detach().cpu().contiguous()
def initialize():
 import subprocess,fcntl
 lock=open(ROOT/'gpu.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid=='GPU-d4e36609-350a-a3be-5024-3eefc3ad01fc'
 torch.set_num_threads(4);torch.manual_seed(20261116);torch.backends.cuda.matmul.allow_tf32=False
 model,receipt=load_model(MODEL,authorized=True,device='cuda:0')
 assert model.config.sliding_window is None and not model.model.has_sliding_layers
 receipt.update(gpu_uuid=uuid,python=sys.version,full_vocabulary_head_shape='[1,T,49152] before R slicing',source_runtime_difference='torch2.5.1+cu124 on this device; historical13580 torch2.7.1+cu128; endpoints never pooled')
 jsave(ROOT/'SOURCE_RUNTIME.json',receipt);return model,lock
@contextlib.contextmanager
def intervene(model,q,*,mode='native',B=None,history=None,boundary=None,span=None,donor=None,layer=0,audit=False):
 R=q['R'];P=q.get('prefix_len',0);rec={'events':[],'stages':{},'dose':[]};handles=[];normcalls=[];active=[]
 seed=int.from_bytes(hashlib.sha256(('COMPUTE-v1|'+q['id']).encode()).digest()[:8],'big')&((1<<63)-1)
 g=torch.Generator().manual_seed(seed);perm=torch.randperm(2048,generator=g).to('cuda')
 def before(i):
  def fn(m,args,kw):
   ut=kw['current_ut'];assert kw['use_cache'] is False and kw.get('past_key_value') is None
   active.append((ut,i));x=_hidden(args,kw);assert x.shape==(1,R+1,2048);rec['events'].append([ut,i])
   if ut==0:return
   if i==0:
    assert bits(x,rec['B']), 'K2 input must equal true K1 final norm'
    rec['native_boundary_sha256']=digest(x)
   if audit:
    if i==0:rec['mask_args']=actual_args(kw)
    assert model.model.layers[i].self_attn.is_causal and model.model.layers[i].self_attn.sliding_window is None
    if kw.get('attention_mask') is not None:
     mask=kw['attention_mask'];assert mask.shape[-2:]==(R+1,R+1)
     if q.get('source_span'):
      lo,hi=q['source_span'];later=q['mapping_start'];assert bool((mask[...,lo:hi,later:]<0).all())
   if history is None and mode=='capture':rec.setdefault('history',{})[i]=x.detach().clone()
   z=None
   if mode=='replay':z=x.clone();z[:,:R]=history[i][:,:R]
   elif mode.startswith(('freeze_','norm_','dose_')):
    region=mode.split('_',1)[1];lo,hi={'all':(0,R),'Q':(P,R),'P':(0,P)}[region];assert hi>lo
    target=rec['B'][:,lo:hi];current=x[:,lo:hi]
    if mode.startswith('norm_'):
     target=(target.float()*current.float().norm(dim=-1,keepdim=True)/target.float().norm(dim=-1,keepdim=True).clamp_min(1e-30)).to(x.dtype)
    elif mode.startswith('dose_'):
     delta=target.float()-current.float();target=(current.float()+delta[...,perm]).to(x.dtype)
    z=x.clone();z[:,lo:hi]=target
    wanted=(rec['B'][:,lo:hi].float()-current.float()).norm(dim=-1);real=(target.float()-current.float()).norm(dim=-1)
    rec['dose'].append(torch.stack([wanted.mean(),real.mean(),(real-wanted).abs().mean(),current.float().norm(dim=-1).mean(),target.float().norm(dim=-1).mean()]).detach())
   elif mode=='boundary' and i==layer:
    z=x.clone();lo,hi=span;assert donor.shape==z[:,lo:hi].shape;z[:,lo:hi]=donor
   elif boundary is not None and i==0:z=boundary.clone()
   if z is not None:
    assert bits(z[:,R],x[:,R]) or (mode=='boundary' and span==(R,R+1))
    if audit:rec.setdefault('applied',{})[i]=cpu(z)
    return _replace(args,kw,z)
  return fn
 def after(i):
  def fn(m,args,kw,out):
   assert active.pop()==(kw['current_ut'],i)
   if kw['current_ut']==1 and i in (0,8,9,15,17,23):rec['stages'][f'L{i+1}_out_R']=out[:,R].detach().clone()
  return fn
 def norm(m,args,out):
  normcalls.append(1)
  if len(normcalls)==1:rec['B']=out.detach().clone()
  elif len(normcalls)==2:rec['K2_final_R']=out[:,R].detach().clone()
  else:raise RuntimeError('norm calls')
 try:
  handles.append(model.model.norm.register_forward_hook(norm))
  for i,l in enumerate(model.model.layers):handles.extend([l.register_forward_pre_hook(before(i),with_kwargs=True),l.register_forward_hook(after(i),with_kwargs=True)])
  yield rec
  assert rec['events']==[[u,i] for u in (0,1) for i in range(24)] and len(normcalls)==2 and not active
 finally:
  for h in handles:h.remove()
@torch.inference_mode()
def forward(model,q,**kwargs):
 assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256']
 ids=torch.tensor([q['ids']],device='cuda');R=q['R'];start=time.monotonic()
 with intervene(model,q,**kwargs) as rec:
  _,states,_=model.model(ids,use_cache=False,past_key_values=None)
  z2=model.lm_head(states[1])[:,R].detach().clone()
  z1=model.lm_head(states[0])[:,R].detach().clone() if kwargs.get('mode','native') in ('native','capture') else None
 torch.cuda.synchronize();rec['elapsed_seconds']=time.monotonic()-start
 assert z2.shape==(1,49152) and z2.dtype==torch.bfloat16 and torch.isfinite(z2).all()
 return z1,z2,rec
def record(q,condition,z,rec):return {'id':q['id'],'split':q['split'],'condition':condition,'input_ids_sha256':q['input_ids_sha256'],'sequence_length':len(q['ids']),'prefix_len':q.get('prefix_len',0),'source_span':q.get('source_span'),'stage_R_sha256':{k:digest(v) for k,v in rec['stages'].items()},'logits_sha256':digest(z),'seconds':rec['elapsed_seconds'],**score(z,q['targets']),'dose_layer_means':torch.stack(rec['dose']).cpu().tolist() if rec['dose'] else []}
