"""Capture actual recurrent layer-local attention/MLP outputs, without intervention."""
from contextlib import contextmanager
import torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
@contextmanager
def capture_real_branches(model,R,*,expected_length,indices=tuple(range(8,16))):
 assert indices==tuple(range(8,16)) and len(model.model.layers)==24 and not model.training and type(R)==int and 0<=R<expected_length
 state={i:[] for i in indices};active=[];records={};hooks=[]
 def pre(i):
  def f(m,args,kw):
   u=kw.get('current_ut');assert type(u)==int and u==len(state[i]) and u in (0,1) and not active and kw.get('use_cache') is False and kw.get('past_key_value') is None;x=args[0] if args else kw.get('hidden_states');assert torch.is_tensor(x) and x.ndim==3 and x.shape[0]==1 and x.shape[1]==expected_length and x.dtype==torch.bfloat16 and torch.isfinite(x).all();state[i].append(u);active.append((i,u))
  return f
 def post(i):
  def f(m,args,kw,out):assert active.pop()==(i,kw.get('current_ut'))
  return f
 def capture(i,branch):
  def f(m,args,out):
   assert active and active[-1][0]==i;u=active[-1][1]
   if u!=1:return
   key=(i,branch);assert key not in records and torch.is_tensor(out) and out.ndim==3 and out.shape[0]==1 and out.shape[1]==expected_length and out.dtype==torch.bfloat16 and torch.isfinite(out).all();records[key]=out[:,R,:].detach().clone()
  return f
 try:
  for i in indices:
   l=model.model.layers[i];hooks.extend([l.register_forward_pre_hook(pre(i),with_kwargs=True),l.input_layernorm_2.register_forward_hook(capture(i,'a')),l.post_attention_layernorm_2.register_forward_hook(capture(i,'b')),l.register_forward_hook(post(i),with_kwargs=True)])
  yield records
  assert not active and all(state[i]==[0,1] for i in indices) and set(records)=={(i,k) for i in indices for k in ('a','b')}
 finally:
  for h in hooks:h.remove()
def reference_deltas(clean,perturbed):
 assert set(clean)==set(perturbed)=={(i,k) for i in range(8,16) for k in ('a','b')};out={}
 for key in clean:
  n,p=clean[key],perturbed[key];assert n.shape==p.shape and n.dtype==p.dtype==torch.bfloat16 and n.device==p.device and torch.isfinite(n).all() and torch.isfinite(p).all();out[key]=n.double()-p.double()
 return out
