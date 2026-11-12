"""Full-dtype observer; actual post-RoPE q/k/v, explicit R-only SDPA replacement."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
if not __debug__:raise RuntimeError('optimized Python forbidden')
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
@contextmanager
def observe(model,ut,R,patch=None,profile=False):
 assert not model.training and ut in (0,1);layer=model.model.layers[17];old=F.scaled_dot_product_attention;events=[];active=[];attn=[];rec={};hs=[]
 def pre(m,args,kw):
  u=kw['current_ut'];assert u==len(events) and not active and kw.get('use_cache') is False and kw.get('past_key_value') is None;events.append(u);active.append(u)
 def post(m,args,kw,out):assert active and not attn;active.pop()
 def ap(m,args,kw):assert active and not attn and kw['current_ut']==active[0];attn.append(1)
 def az(m,args,kw,out):assert attn.pop()==1
 def sdpa(*args,**kw):
  if not active or active[0]!=ut:return old(*args,**kw)
  assert attn==[1] and 'q' not in rec;q,k,v=args[:3];assert all(x.dtype==torch.float32 and x.ndim==4 and torch.isfinite(x).all() for x in (q,k,v));rec.update(q=q.detach().clone(),k=k.detach().clone(),v=v.detach().clone(),tail=args[3:],kwargs=kw.copy())
  if profile:
   with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:out=old(*args,**kw)
   rec['profiler_ops']=[x.key for x in prof.key_averages() if 'scaled_dot_product' in x.key.lower() or 'flash_attention' in x.key.lower()]
  else:out=old(*args,**kw)
  assert torch.isfinite(out).all() and out.dtype==torch.float32;rec['native_sdpa_R']=out[...,R,:].detach().clone()
  if patch is None:rec['injected_sdpa_R']=rec['native_sdpa_R'].clone();return out
  assert patch.dtype==out.dtype and patch.shape==out[...,R,:].shape and torch.isfinite(patch).all();result=out.clone();result[...,R,:]=patch;mask=torch.arange(out.shape[-2],device=out.device)!=R;assert bits(out[...,mask,:],result[...,mask,:]);rec['injected_sdpa_R']=patch.detach().clone();return result
 try:
  hs=[layer.register_forward_pre_hook(pre,with_kwargs=True),layer.register_forward_hook(post,with_kwargs=True),layer.self_attn.register_forward_pre_hook(ap,with_kwargs=True),layer.self_attn.register_forward_hook(az,with_kwargs=True)];F.scaled_dot_product_attention=sdpa;yield rec;assert events==list(range(ut+1)) and not active and not attn and 'native_sdpa_R' in rec
 finally:
  F.scaled_dot_product_attention=old
  for h in hs:h.remove()
