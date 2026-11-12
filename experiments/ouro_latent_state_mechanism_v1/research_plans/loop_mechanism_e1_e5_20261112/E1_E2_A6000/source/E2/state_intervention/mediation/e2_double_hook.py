"""Technical double-site hook for K2 layer-18 input and its actual local attention norm output.
No intervention on an unobserved future state. Not evidence of natural mediation on its own.
"""
from contextlib import contextmanager
import torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
@contextmanager
def double_intervene(layer,*,physical_index,ut,R,expected_full,expected_id,current_id,expected_input_ids_sha,input_ids_sha,x_patch,a_patch):
 assert physical_index==17 and ut==1 and expected_id==current_id and expected_input_ids_sha==input_ids_sha
 assert expected_full.ndim==3 and expected_full.dtype==torch.bfloat16 and 0<=R<expected_full.shape[-2]
 assert x_patch.dtype==a_patch.dtype==torch.bfloat16 and x_patch.shape==a_patch.shape==expected_full[...,R,:].shape and torch.isfinite(x_patch).all() and torch.isfinite(a_patch).all()
 rec={'events':[]};active=[];hs=[]
 def pre(m,args,kw):
  u=kw.get('current_ut');assert isinstance(u,int) and u==len(rec['events']);rec['events'].append(u)
  if u!=ut:return
  x=args[0];assert bits(x,expected_full.to(x.device)) and torch.isfinite(x).all() and not active;active.append(u);rec['x_receiver_R']=x[...,R,:].detach().clone();result=x.clone();result[...,R,:]=x_patch.to(x.device);mask=torch.arange(x.shape[-2],device=x.device)!=R;assert bits(x[...,mask,:],result[...,mask,:]);rec['x_injected_R']=result[...,R,:].detach().clone();rec['x_outside_R_unchanged']=True;return (result,)+args[1:],kw
 def attention_norm(m,args,out):
  if not active:return
  assert 'a_receiver_R' not in rec and out.dtype==torch.bfloat16 and out.shape==expected_full.shape and torch.isfinite(out).all();rec['a_receiver_R']=out[...,R,:].detach().clone();y=out.clone();y[...,R,:]=a_patch.to(out.device);mask=torch.arange(out.shape[-2],device=out.device)!=R;assert bits(out[...,mask,:],y[...,mask,:]);rec['a_injected_R']=y[...,R,:].detach().clone();rec['a_outside_R_unchanged']=True;return y
 def ypre(m,args):
  if active:assert 'y_R' not in rec;rec['y_R']=args[0][...,R,:].detach().clone();assert bits(rec['y_R'],(rec['x_injected_R']+rec['a_injected_R']).to(torch.bfloat16))
 def post(m,args,kw,out):
  if kw.get('current_ut')==ut:
   assert active==[ut] and 'z_R' not in rec;rec['z_R']=(out[0] if isinstance(out,tuple) else out)[...,R,:].detach().clone();active.pop()
 try:
  hs=[layer.register_forward_pre_hook(pre,with_kwargs=True),layer.input_layernorm_2.register_forward_hook(attention_norm),layer.post_attention_layernorm.register_forward_pre_hook(ypre),layer.register_forward_hook(post,with_kwargs=True)]
  yield rec
  assert rec['events']==[0,1] and not active and all(k in rec and torch.isfinite(rec[k]).all() for k in ('x_receiver_R','x_injected_R','a_receiver_R','a_injected_R','y_R','z_R'))
 finally:
  for h in hs:h.remove()
