"""CPU draft only: scope actual SDPA call at natural L18; no production approval."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
if not __debug__:raise RuntimeError('optimized Python forbidden')
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def equivalent(a,b):
 if torch.is_tensor(a) or torch.is_tensor(b):return torch.is_tensor(a) and torch.is_tensor(b) and bits(a,b)
 return a==b
@contextmanager
def observe_qkv(model,depth,R,donor=None,q2=False,k2=False,v2=False):
 if model.training or depth not in (1,2) or type(R)!=int or R<0 or type(q2)!=bool or type(k2)!=bool or type(v2)!=bool:raise ValueError('mode')
 core=model.model
 if len(core.layers)!=24 or any(x.self_attn.layer_idx!=i for i,x in enumerate(core.layers)):raise ValueError('physical layers')
 if (q2 or k2 or v2) and (depth!=1 or donor is None):raise ValueError('missing K2 donor')
 layer=core.layers[17];target=depth-1;active=[];attn=[];events=[];record={'calls':0};handles=[];old=F.scaled_dot_product_attention
 def pre(m,args,kw):
  u=kw.get('current_ut');x=args[0] if args else kw.get('hidden_states')
  if type(u)!=int or u!=len(events) or active or not torch.is_tensor(x) or x.ndim!=3 or R>=x.shape[1] or kw.get('past_key_value') is not None or kw.get('use_cache') is not False:raise ValueError('layer')
  active.append(u);events.append(u)
  if u==target:record['receiver_x']=x[:,R,:].detach().clone()
 def attn_pre(m,args,kw):
  if len(active)!=1 or attn or kw.get('current_ut')!=active[0] or kw.get('past_key_value') is not None:raise ValueError('attention scope')
  attn.append(active[0])
 def attn_post(m,args,kw,out):
  if len(attn)!=1:raise ValueError('attention scope close')
  attn.pop()
 def post(m,args,kw,out):
  if len(active)!=1 or attn:raise ValueError('layer scope close')
  active.pop()
 def wrapper(*args,**kwargs):
  if not active or active[0]!=target:return old(*args,**kwargs)
  if not attn:raise ValueError('SDPA outside target attention scope')
  if record['calls'] or len(args)<3:raise ValueError('extra/missing SDPA')
  q,k,v=args[:3];tail=args[3:]
  if any(not torch.is_tensor(x) or x.ndim!=4 or not torch.isfinite(x).all() for x in (q,k,v)) or R>=q.shape[-2]:raise ValueError('QKV')
  record['calls']+=1;record['q']=q.detach().clone();record['k']=k.detach().clone();record['v']=v.detach().clone();record['tail']=tuple(z.detach().clone() if torch.is_tensor(z) else z for z in tail);record['kwargs']={n:(z.detach().clone() if torch.is_tensor(z) else z) for n,z in kwargs.items()}
  baseline=old(*args,**kwargs)
  if not torch.is_tensor(baseline) or not torch.isfinite(baseline).all():raise ValueError('nonfinite SDPA baseline')
  record['baseline_sdpa_R']=baseline[...,R,:].detach().clone()
  if not (q2 or k2 or v2):record['injected_sdpa_R']=record['baseline_sdpa_R'].clone();record['outside_R_unchanged']=True;record['completed']=True;return baseline
  for n,x in [('q',q),('k',k),('v',v)]:
   z=donor[n]
   if not torch.is_tensor(z) or z.shape!=x.shape or z.dtype!=x.dtype or z.device!=x.device or not torch.isfinite(z).all():raise ValueError('donor QKV shape/device/finite')
  if len(tail)!=len(donor['tail']) or any(not equivalent(a,b) for a,b in zip(tail,donor['tail'])) or set(kwargs)!=set(donor['kwargs']) or any(not equivalent(kwargs[n],donor['kwargs'][n]) for n in kwargs):raise ValueError('mask/kwargs mismatch')
  mixedq=q.clone()
  if q2:mixedq[...,R,:]=donor['q'][...,R,:]
  hybrid=old(mixedq,donor['k'] if k2 else k,donor['v'] if v2 else v,*tail,**kwargs)
  if not torch.is_tensor(hybrid) or not torch.isfinite(hybrid).all():raise ValueError('nonfinite hybrid SDPA')
  out=baseline.clone();out[...,R,:]=hybrid[...,R,:]
  mask=torch.ones(out.shape[-2],dtype=torch.bool,device=out.device);mask[R]=False
  if not bits(out[...,mask,:],baseline[...,mask,:]) or not bits(out[...,R,:],hybrid[...,R,:]):raise ValueError('output mask')
  record['hybrid_sdpa_R']=hybrid[...,R,:].detach().clone();record['injected_sdpa_R']=out[...,R,:].detach().clone();record['outside_R_unchanged']=True;record['completed']=True;return out
 try:
  handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True));handles.append(layer.self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True));handles.append(layer.self_attn.register_forward_hook(attn_post,with_kwargs=True));handles.append(layer.register_forward_hook(post,with_kwargs=True));F.scaled_dot_product_attention=wrapper
  yield record
  if events!=list(range(depth)) or active or attn or record['calls']!=1 or record.get('completed') is not True:raise ValueError('incomplete observation')
 finally:
  F.scaled_dot_product_attention=old
  for h in handles:h.remove()
