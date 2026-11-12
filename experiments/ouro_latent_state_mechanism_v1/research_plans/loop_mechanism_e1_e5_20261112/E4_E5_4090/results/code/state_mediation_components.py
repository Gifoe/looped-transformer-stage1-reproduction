"""Prepared L18 x_R rollback instrumentation; GPU execution requires a new Astra order."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from observer import bits,thash
if not __debug__:raise RuntimeError('optimized Python forbidden')

@contextmanager
def input_r_block(model,depth,R,replacement=None):
 """Enter before runner.run()/observe(), so x_R changes precede attention observation.

 The nested observer can independently inject A2 or a previously captured A(U).
 This context records the actual unmodified SDPA result generated from its input.
 It never substitutes natural K1 attention for A(U).
 """
 assert depth in (1,2,3,4) and not model.training
 layer=model.model.layers[17];events=[];active=[];record={};handles=[];old=F.scaled_dot_product_attention
 def pre(m,args,kw):
  u=kw['current_ut'];assert u==len(events) and not active and kw['use_cache'] is False and kw.get('past_key_value') is None
  active.append(u);events.append(u)
  if u!=depth-1:return
  x=args[0] if args else kw['hidden_states'];assert x.ndim==3 and R==x.shape[1]-1
  non_r=torch.arange(x.shape[1],device=x.device)!=R
  record['original_x_R']=x[:,R,:].detach().clone();record['non_R_x_sha256']=thash(x[:,non_r,:]);record['sequence_length']=x.shape[1]
  if replacement is None:record['applied_x_R']=record['original_x_R'].clone();record['non_R_x_unchanged']=True;return
  value=replacement.to(x.device);assert value.shape==x[:,R,:].shape and value.dtype==x.dtype and torch.isfinite(value).all()
  changed=x.clone();changed[:,R,:]=value;assert bits(changed[:,non_r,:],x[:,non_r,:]);record['applied_x_R']=value.clone();record['non_R_x_unchanged']=True
  if args:return (changed,*args[1:]),kw
  nextkw=kw.copy();nextkw['hidden_states']=changed;return args,nextkw
 def post(m,args,kw,out):
  assert len(active)==1
  if active==[depth-1]:
   value=out if torch.is_tensor(out) else out[0];assert value.ndim==3
   record['after_mlp_R']=value[:,R,:].detach().clone()
  active.pop()
 def after_attention(m,args,kw):
  if active==[depth-1]:
   value=args[0] if args else kw['hidden_states'];assert value.ndim==3
   record['after_attention_R']=value[:,R,:].detach().clone()
 def wrapped(*args,**kw):
  if active!=[depth-1]:return old(*args,**kw)
  assert 'actual_sdpa_R' not in record
  q,k,v=args[:3];assert q.shape[-2]==k.shape[-2]==v.shape[-2]==record['sequence_length']
  mask=torch.arange(q.shape[-2],device=q.device)!=R
  for name,value in [('Q',q),('K',k),('V',v)]:
   record[name+'_R']=value[...,R,:].detach().clone();record['non_R_'+name+'_sha256']=thash(value[...,mask,:])
  output=old(*args,**kw);record['actual_sdpa_R']=output[...,R,:].detach().clone();record['non_R_sdpa_sha256']=thash(output[...,mask,:]);return output
 try:
  handles=[layer.register_forward_pre_hook(pre,with_kwargs=True),layer.register_forward_hook(post,with_kwargs=True)]
  if hasattr(layer,'post_attention_layernorm'):handles.append(layer.post_attention_layernorm.register_forward_pre_hook(after_attention,with_kwargs=True))
  F.scaled_dot_product_attention=wrapped
  yield record
  assert events==list(range(depth)) and not active and 'actual_sdpa_R' in record
  record['loop_events']=events
 finally:
  F.scaled_dot_product_attention=old
  for handle in handles:handle.remove()
