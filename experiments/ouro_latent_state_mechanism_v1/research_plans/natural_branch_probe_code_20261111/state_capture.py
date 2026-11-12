"""Read-only x/y/z residual capture, CPU toy staging only."""
from contextlib import contextmanager
import torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
@contextmanager
def observe(model,R):
 core=model.model
 if len(core.layers)!=24 or any(l.self_attn.layer_idx!=i for i,l in enumerate(core.layers)):raise ValueError('physical layers')
 if type(R)!=int or R<0:raise ValueError('R')
 events=[];active=[];out={};norms=[];handles=[]
 def pre(i):
  def hook(m,args,kw):
   u=kw.get('current_ut');expected=(len(events)//24,len(events)%24)
   if type(u)!=int or (u,i)!=expected or active or u>1:raise RuntimeError('layer sequence')
   x=args[0] if args else kw.get('hidden_states')
   if not torch.is_tensor(x) or x.ndim!=3 or x.shape[0]!=1 or R>=x.shape[1] or not torch.isfinite(x[:,R,:]).all():raise ValueError('input')
   if kw.get('past_key_value') is not None or kw.get('use_cache') is not False:raise ValueError('cache mode')
   events.append((u,i));active.append((u,i));out[(u,i)]={'x':x[:,R,:].detach().clone()}
  return hook
 def mid(i):
  def hook(m,args):
   if len(active)!=1 or active[0][1]!=i or len(args)!=1:raise RuntimeError('attention residual scope')
   y=args[0];key=active[0]
   if 'y' in out[key] or not torch.isfinite(y[:,R,:]).all():raise ValueError('bad y')
   out[key]['y']=y[:,R,:].detach().clone()
  return hook
 def post(i):
  def hook(m,args,kw,z):
   if len(active)!=1 or active[0][1]!=i or not torch.is_tensor(z) or not torch.isfinite(z[:,R,:]).all():raise ValueError('output scope')
   key=active.pop()
   if 'y' not in out[key] or 'z' in out[key]:raise RuntimeError('missing y')
   out[key]['z']=z[:,R,:].detach().clone()
  return hook
 def branch(i,name):
  def hook(m,args,value):
   if len(active)!=1 or active[0][1]!=i or name in out[active[0]] or not torch.is_tensor(value) or not torch.isfinite(value[:,R,:]).all():raise ValueError('branch scope')
   out[active[0]][name]=value[:,R,:].detach().clone()
  return hook
 def norm(m,args,z):
  if len(events)!=24*(len(norms)+1) or active or not torch.isfinite(z[:,R,:]).all():raise RuntimeError('norm sequence')
  norms.append(z[:,R,:].detach().clone())
 try:
  for i,l in enumerate(core.layers):
   handles.append(l.register_forward_pre_hook(pre(i),with_kwargs=True));handles.append(l.input_layernorm_2.register_forward_hook(branch(i,'a')));handles.append(l.post_attention_layernorm.register_forward_pre_hook(mid(i)));handles.append(l.post_attention_layernorm_2.register_forward_hook(branch(i,'b')));handles.append(l.register_forward_hook(post(i),with_kwargs=True))
  handles.append(core.norm.register_forward_hook(norm));yield {'states':out,'norms':norms,'events':events}
  if events!=[(u,i) for u in range(2) for i in range(24)] or len(norms)!=2 or active or any(set(v)!={'x','a','y','b','z'} for v in out.values()):raise RuntimeError('incomplete trace')
 finally:
  for h in handles:h.remove()
