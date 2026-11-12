"""Patch only actual layer18/ut0 SDPA R output and record unmodified suffix stages."""
from contextlib import contextmanager
import torch,torch.nn.functional as F

def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.device==b.device and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
@contextmanager
def inject_sdpa_R(model,R,value,expected_x=None,capture_stages=True):
 if model.training or type(R)!=int or R<0 or value.dtype!=torch.bfloat16 or not torch.isfinite(value).all():raise ValueError('mode/value')
 layers=model.model.layers
 if len(layers)!=24 or any(l.self_attn.layer_idx!=i for i,l in enumerate(layers)):raise ValueError('physical layer identity')
 active=[];attn=[];events=[];rec={'calls':0,'stages':{}};hooks=[];old=F.scaled_dot_product_attention
 def pre(m,args,kw):
  u=kw.get('current_ut');x=args[0] if args else kw.get('hidden_states')
  if u!=0 or active or not torch.is_tensor(x) or R>=x.shape[1] or kw.get('past_key_value') is not None or kw.get('use_cache') is not False:raise ValueError('layer/ut')
  active.append(u);events.append(u);rec['receiver_x']=x[:,R,:].detach().clone()
  if expected_x is not None and not bits(rec['receiver_x'],expected_x):raise ValueError('receiver x mismatch')
 def post(m,args,kw,out):
  if len(active)!=1 or attn:raise ValueError('layer scope')
  active.pop()
 def ap(m,args,kw):
  if len(active)!=1 or attn or kw.get('current_ut')!=0 or kw.get('past_key_value') is not None:raise ValueError('attention scope')
  attn.append(0)
 def az(m,args,kw,out):
  if len(attn)!=1:raise ValueError('attention scope close')
  attn.pop()
 def wrapper(*args,**kw):
  if not active:return old(*args,**kw)
  if not attn or rec['calls'] or len(args)<3:raise ValueError('SDPA scope/count')
  q,k,v=args[:3]; rec['calls']+=1
  if any(not torch.is_tensor(x) or x.ndim!=4 or not torch.isfinite(x).all() for x in (q,k,v)):raise ValueError('actual QKV')
  rec['q']=q.detach().clone();rec['k']=k.detach().clone();rec['v']=v.detach().clone();rec['tail']=tuple(x.detach().clone() if torch.is_tensor(x) else x for x in args[3:]);rec['kwargs']={n:(x.detach().clone() if torch.is_tensor(x) else x) for n,x in kw.items()}
  base=old(*args,**kw)
  if not torch.is_tensor(base) or not torch.isfinite(base).all() or R>=base.shape[-2] or value.shape!=base[...,R,:].shape or value.device!=base.device:raise ValueError('actual SDPA/value mismatch')
  out=base.clone();out[...,R,:]=value;mask=torch.ones(out.shape[-2],dtype=torch.bool,device=out.device);mask[R]=False
  if not bits(out[...,mask,:],base[...,mask,:]) or not bits(out[...,R,:],value):raise ValueError('outside R changed')
  rec['baseline_sdpa_R']=base[...,R,:].detach().clone();rec['injected_sdpa_R']=out[...,R,:].detach().clone();rec['outside_R_unchanged']=True;rec['completed']=True;return out
 def cap(name):
  def fn(m,args,out):
   if name in rec['stages'] or not torch.is_tensor(out) or out.ndim!=3 or out.shape[0]!=1 or out.shape[-1]!=2048 or out.dtype!=torch.bfloat16 or R>=out.shape[1] or not torch.isfinite(out[:,R,:]).all():raise ValueError('stage '+name)
   rec['stages'][name]=out[:,R,:].detach().clone()
  return fn
 def cap_in(name):
  def fn(m,args):
   x=args[0]
   if name in rec['stages'] or not torch.is_tensor(x) or x.ndim!=3 or x.shape[0]!=1 or x.shape[-1]!=2048 or x.dtype!=torch.bfloat16 or not torch.isfinite(x[:,R,:]).all():raise ValueError('stage '+name)
   rec['stages'][name]=x[:,R,:].detach().clone()
  return fn
 try:
  hooks.extend([layers[17].register_forward_pre_hook(pre,with_kwargs=True),layers[17].register_forward_hook(post,with_kwargs=True),layers[17].self_attn.register_forward_pre_hook(ap,with_kwargs=True),layers[17].self_attn.register_forward_hook(az,with_kwargs=True)])
  if capture_stages:
   l=layers[17];hooks.extend([l.self_attn.o_proj.register_forward_hook(cap('oproj')),l.input_layernorm_2.register_forward_hook(cap('attention_localnorm')),l.post_attention_layernorm.register_forward_pre_hook(cap_in('residual_after_attention')),l.mlp.register_forward_hook(cap('mlp_raw')),l.post_attention_layernorm_2.register_forward_hook(cap('mlp_localnorm'))])
   for i in range(17,24):hooks.append(layers[i].register_forward_hook(cap(f'layer{i+1:02d}_output')))
   hooks.append(model.model.norm.register_forward_hook(cap('loop_postnorm')))
  F.scaled_dot_product_attention=wrapper;yield rec
  expected={'oproj','attention_localnorm','residual_after_attention','mlp_raw','mlp_localnorm','loop_postnorm'}|{f'layer{i+1:02d}_output' for i in range(17,24)}
  if events!=[0] or active or attn or rec['calls']!=1 or not rec.get('completed') or (capture_stages and set(rec['stages'])!=expected):raise ValueError('incomplete hook events')
 finally:
  F.scaled_dot_product_attention=old
  for h in hooks:h.remove()
