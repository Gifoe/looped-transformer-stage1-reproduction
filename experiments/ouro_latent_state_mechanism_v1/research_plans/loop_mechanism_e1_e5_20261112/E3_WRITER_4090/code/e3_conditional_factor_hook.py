"""Apply fixed reference increments to actual L9–L16 residual branch outputs at R."""
from contextlib import contextmanager
import torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
class UndefinedNormControl(RuntimeError):pass
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def add_site(v,delta,sign,norm_preserve=False):
 assert v.ndim==2 and v.dtype==torch.bfloat16 and delta.shape==v.shape and delta.dtype==torch.float64 and sign in (-1,1) and torch.isfinite(v).all() and torch.isfinite(delta).all();before=v.double();proposal=before+sign*delta
 if norm_preserve:
  n0=before.norm();n1=proposal.norm()
  if n1==0 and n0!=0:raise UndefinedNormControl('zero proposed norm with nonzero receiver norm')
  exact=proposal if n1==0 else proposal*(n0/n1)
 else:exact=proposal
 after=exact.to(torch.bfloat16);assert torch.isfinite(after).all();return after,{'before':v.detach().clone(),'after':after.detach().clone(),'reference_delta':delta.detach().clone(),'effective_delta':after.double()-before,'cast_residual':after.double()-exact,'exact':exact,'reference_dose':float(delta.norm()),'effective_dose':float((after.double()-before).norm()),'before_norm':float(n0) if norm_preserve else float(before.norm()),'after_norm':float(after.double().norm())}
@contextmanager
def intervene_attention_factor(model,R,*,expected_length,factor,reference_deltas,sign,norm_preserve=False):
 indices={'E':(8,),'L':tuple(range(9,16))}[factor];branch='a'
 assert branch in ('a','b') and sign in (-1,1) and set(reference_deltas)==set(indices) and len(model.model.layers)==24 and not model.training and type(R)==int and 0<=R<expected_length
 events={i:[] for i in indices};active=[];records={};handles=[]
 def pre(i):
  def hook(m,args,kw):
   u=kw.get('current_ut');assert type(u)==int and u==len(events[i]) and u in (0,1) and not active and kw.get('use_cache') is False and kw.get('past_key_value') is None;events[i].append(u);active.append((i,u))
  return hook
 def post(i):
  def hook(m,args,kw,out):assert active.pop()==(i,kw['current_ut'])
  return hook
 def patch(i):
  def hook(m,args,out):
   assert active and active[-1][0]==i
   if active[-1][1]!=1:return
   assert i not in records and out.dtype==torch.bfloat16 and out.ndim==3 and out.shape[0]==1 and out.shape[1]==expected_length and torch.isfinite(out).all();delta=reference_deltas[i].to(device=out.device)
   try:value,receipt=add_site(out[:,R,:],delta,sign,norm_preserve)
   except UndefinedNormControl as exc:raise UndefinedNormControl(f'layer{i:02d}_{branch}: {exc}') from exc
   changed=out.clone();changed[:,R,:]=value;mask=torch.arange(expected_length,device=out.device)!=R;assert bits(out[:,mask,:],changed[:,mask,:]);receipt['outside_R_unchanged']=True;records[i]=receipt;return changed
  return hook
 try:
  for i in indices:
   l=model.model.layers[i];site=l.input_layernorm_2 if branch=='a' else l.post_attention_layernorm_2;handles.extend([l.register_forward_pre_hook(pre(i),with_kwargs=True),site.register_forward_hook(patch(i)),l.register_forward_hook(post(i),with_kwargs=True)])
  yield records
  assert not active and all(events[i]==[0,1] for i in events) and set(records)==set(indices)
 finally:
  for h in handles:h.remove()
