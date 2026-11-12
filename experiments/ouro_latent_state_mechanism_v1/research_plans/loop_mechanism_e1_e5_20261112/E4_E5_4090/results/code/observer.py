"""L18, explicit recurrent call, complete-shape SDPA with R-only replacement."""
from contextlib import contextmanager
import hashlib
import torch
import torch.nn.functional as F
if not __debug__:raise RuntimeError('optimized Python forbidden')

def bits(a,b):
 return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

def thash(x):
 return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def description(x):
 if torch.is_tensor(x):return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=thash(x),stride=list(x.stride()))
 return x

@contextmanager
def observe(model,depth,R,patch=None,capture=False):
 assert not model.training and depth in (1,2,3,4)
 layer=model.model.layers[17];old=F.scaled_dot_product_attention;events=[];active=[];attn=[];handles=[];rec={}
 def pre(m,args,kw):
  u=kw['current_ut'];assert u==len(events) and not active and kw['use_cache'] is False and kw.get('past_key_value') is None
  events.append(u);active.append(u)
  if u==depth-1:rec['x_R']=args[0][:,R,:].detach().clone()
 def ap(m,args,kw):
  assert len(active)==1 and not attn and kw['current_ut']==active[0];attn.append(active[0])
 def az(m,args,kw,out):
  assert len(attn)==1;attn.pop()
 def post(m,args,kw,out):
  assert len(active)==1 and not attn;active.pop()
 def norm(m,args,out):
  if active==[depth-1]:rec['a_R']=out[:,R,:].detach().clone()
 def endnorm(m,args,out):
  if len(events)==depth:rec['final_norm_R']=out[:,R,:].detach().clone()
 def wrapper(*args,**kw):
  if active!=[depth-1]:return old(*args,**kw)
  assert attn==[depth-1] and 'O' not in rec
  q,k,v=args[:3];assert all(t.ndim==4 and torch.isfinite(t).all() for t in (q,k,v))
  assert R<q.shape[-2]
  out=old(*args,**kw);rec['O']=out[...,R,:].detach().clone()
  rec['sdpa_args']={'q':description(q),'k':description(k),'v':description(v),'tail':[description(x) for x in args[3:]],'kwargs':{n:description(x) for n,x in kw.items()}}
  if capture:rec['qkv']=(q.detach().clone(),k.detach().clone(),v.detach().clone(),args[3:],kw.copy())
  if patch is None:rec['injected']=rec['O'].clone();rec['outside_R_unchanged']=True;return out
  p=patch.to(out.device);assert p.shape==rec['O'].shape and p.dtype==out.dtype and torch.isfinite(p).all()
  result=out.clone();result[...,R,:]=p
  mask=torch.arange(out.shape[-2],device=out.device)!=R
  assert bits(result[...,mask,:],out[...,mask,:]);rec['outside_R_unchanged']=True;rec['injected']=p.clone()
  return result
 try:
  handles=[layer.register_forward_pre_hook(pre,with_kwargs=True),layer.self_attn.register_forward_pre_hook(ap,with_kwargs=True),layer.self_attn.register_forward_hook(az,with_kwargs=True),layer.register_forward_hook(post,with_kwargs=True),layer.input_layernorm_2.register_forward_hook(norm),model.model.norm.register_forward_hook(endnorm)]
  F.scaled_dot_product_attention=wrapper
  yield rec
  assert events==list(range(depth)) and not active and not attn and 'O' in rec and 'a_R' in rec
  rec['loop_events']=events
 finally:
  F.scaled_dot_product_attention=old
  for h in handles:h.remove()

def hybrid(receiver,donor,R,fp32=False):
 q,k,v,tail,kw=receiver;_,dk,dv,dtail,dkw=donor
 assert q.shape==donor[0].shape and k.shape==dk.shape and v.shape==dv.shape
 assert len(tail)==len(dtail) and set(kw)==set(dkw)
 for a,b in list(zip(tail,dtail))+[(kw[n],dkw[n]) for n in kw]:
  assert bits(a,b) if torch.is_tensor(a) else a==b
 if fp32:
  q,dk,dv=q.float(),dk.float(),dv.float()
  tail=tuple(x.float() if torch.is_tensor(x) and x.dtype.is_floating_point else x for x in tail)
  kw={n:x.float() if torch.is_tensor(x) and x.dtype.is_floating_point else x for n,x in kw.items()}
 return F.scaled_dot_product_attention(q,dk,dv,*tail,**kw)[...,R,:].to(receiver[0].dtype).clone()
