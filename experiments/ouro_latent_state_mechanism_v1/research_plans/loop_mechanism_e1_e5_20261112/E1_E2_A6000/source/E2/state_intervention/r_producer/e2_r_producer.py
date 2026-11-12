"""R-only coarse producer→L18 state path engine; technical preparation, no cut chosen."""
from contextlib import contextmanager
import torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
@contextmanager
def producer_transfer(layers,*,cut_index,R,expected_id,current_id,input_sha,expected_input_sha,producer_patch=None,target_patch=None,expected_cut_full=None,expected_target_full=None,capture_attention=True):
 assert len(layers)==24 and cut_index in (0,8,16,17) and expected_id==current_id and input_sha==expected_input_sha
 rec={'cut_events':[],'target_events':[]};handles=[];target_active=[]
 def install_patch(x,patch,label):
  if patch is None:return x
  assert patch.dtype==x.dtype==torch.bfloat16 and patch.shape==x[...,R,:].shape and torch.isfinite(patch).all();y=x.clone();y[...,R,:]=patch.to(x.device);other=torch.arange(x.shape[-2],device=x.device)!=R;assert bits(x[...,other,:],y[...,other,:]);rec[label+'_outside_R_unchanged']=True;return y
 def cut_hook(mod,args,kw):
  u=kw['current_ut'];assert u==len(rec['cut_events']);rec['cut_events'].append(u)
  if u!=1:return
  x=args[0];assert x.ndim==3 and x.dtype==torch.bfloat16 and 0<=R<x.shape[-2] and torch.isfinite(x).all();assert expected_cut_full is None or bits(x,expected_cut_full.to(x.device));rec['cut_before_full']=x.detach().clone();y=install_patch(x,producer_patch,'cut');rec['cut_after_R']=y[...,R,:].detach().clone()
  if cut_index==17:
   rec['target_events'].append(1);target_active.append(1);assert expected_target_full is None or bits(y,expected_target_full.to(y.device));rec['target_before_full']=y.detach().clone();y=install_patch(y,target_patch,'target');rec['target_after_R']=y[...,R,:].detach().clone()
  if producer_patch is not None or (cut_index==17 and target_patch is not None):return (y,)+args[1:],kw
 def target_hook(mod,args,kw):
  u=kw['current_ut'];assert u==len(rec['target_events']);rec['target_events'].append(u)
  if u!=1:return
  target_active.append(1);x=args[0];assert torch.isfinite(x).all();assert expected_target_full is None or bits(x,expected_target_full.to(x.device));rec['target_before_full']=x.detach().clone();y=install_patch(x,target_patch,'target');rec['target_after_R']=y[...,R,:].detach().clone()
  if target_patch is not None:return (y,)+args[1:],kw
 def norm_hook(mod,args,out):
  if target_active:assert 'a_R' not in rec;rec['a_R']=out[...,R,:].detach().clone()
 def target_post(mod,args,kw,out):
  if kw.get('current_ut')==1:assert target_active.pop()==1
 try:
  handles.append(layers[cut_index].register_forward_pre_hook(cut_hook,with_kwargs=True));handles.append(layers[17].register_forward_hook(target_post,with_kwargs=True))
  if capture_attention:handles.append(layers[17].input_layernorm_2.register_forward_hook(norm_hook))
  if cut_index!=17:handles.append(layers[17].register_forward_pre_hook(target_hook,with_kwargs=True))
  yield rec
  assert rec['cut_events']==[0,1] and rec['target_events']==([1] if cut_index==17 else [0,1]) and not target_active and all(k in rec for k in ('cut_before_full','cut_after_R','target_before_full','target_after_R'))
  if capture_attention:assert 'a_R' in rec and torch.isfinite(rec['a_R']).all()
 finally:
  for h in handles:h.remove()
