"""E2.3 L18 residual intervention primitives. Technical use only; no scientific target selected."""
from contextlib import contextmanager
import hashlib, math
import numpy as np
import torch
import torch.nn.functional as F
if not __debug__: raise RuntimeError('optimized Python forbidden')

def bits(a,b):
 return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

def raw_probe_span(w,std):
 """Float64 SVD in original input coordinates. Four class contrasts against class 0; no bias."""
 w=np.asarray(w,dtype=np.float64);std=np.asarray(std,dtype=np.float64)
 assert w.ndim==2 and w.shape[0]==len(std)+1 and w.shape[1]>=2 and np.isfinite(w).all() and np.isfinite(std).all() and (std>0).all()
 directions=(w[:-1,1:]-w[:-1,:1])/std[:,None]
 U,s,_=np.linalg.svd(directions,full_matrices=False);tol=max(directions.shape)*np.finfo(np.float64).eps*(s[0] if s.size else 0.);rank=int(np.sum(s>tol));return U[:,:rank],s,tol

def project(delta,basis):
 d=np.asarray(delta,dtype=np.float64);b=np.asarray(basis,dtype=np.float64);assert b.ndim==2 and d.shape==(b.shape[0],)
 return b@(b.T@d)

def conditions(x1,x2,other,projection=None,random_basis=None):
 """FP64 deltas and BF16 writeback; unavailable controls represented explicitly."""
 assert x1.shape==x2.shape==other.shape and x1.dtype==x2.dtype==torch.bfloat16 and other.dtype in (torch.bfloat16,torch.float64)
 a=x1.double();b=x2.double();od=other.double();delta=b-a;d=delta if projection is None else torch.as_tensor(project(delta.cpu().numpy().ravel(),projection),device=x1.device,dtype=torch.float64).reshape_as(delta)
 rollback=(b-d).to(x2.dtype);q=rollback.double();qn=torch.linalg.vector_norm(q);bn=torch.linalg.vector_norm(b)
 matched=(x2.clone() if bn.item()==0 else None) if qn.item()==0 else (q*(bn/qn)).to(x2.dtype)
 on=torch.linalg.vector_norm(od);dn=torch.linalg.vector_norm(d);unrelated=(x2.clone() if dn.item()==0 else None) if on.item()==0 else (b-od*(dn/on)).to(x2.dtype)
 random_delta=None if random_basis is None else torch.as_tensor(project(delta.cpu().numpy().ravel(),random_basis),device=x1.device,dtype=torch.float64).reshape_as(delta)
 rn=None if random_delta is None else torch.linalg.vector_norm(random_delta)
 random_patch=None if rn is None else ((x2.clone() if dn.item()==0 else None) if rn.item()==0 else (b-random_delta*(dn/rn)).to(x2.dtype))
 complement=(b-(delta-d)).to(x2.dtype);full=(b-delta).to(x2.dtype)
 exact={'rollback':b-d,'rollback_complement':b-(delta-d),'rollback_full':b-delta,'forward_add':a+d}
 if matched is not None:exact['norm_matched']=b.clone() if qn.item()==0 else q*(bn/qn)
 if unrelated is not None:exact['unrelated']=b.clone() if on.item()==0 else b-od*(dn/on)
 if random_patch is not None:exact['random_rank4']=b.clone() if rn.item()==0 else b-random_delta*(dn/rn)
 return {'exact_fp64':exact,'random_rank4':random_patch,'random_status':'UNDEFINED_ZERO_RANDOM_PROJECTION' if random_patch is None else 'OK','rollback_complement':complement,'rollback_full':full,'delta_fp64':delta,'component_fp64':d,'identity':x2.clone(),'rollback':rollback,'norm_matched':matched,'norm_status':'UNDEFINED_ZERO_ROLLBACK' if matched is None else 'OK','unrelated':unrelated,'unrelated_status':'UNDEFINED_ZERO_OTHER_DELTA' if unrelated is None else 'OK','restore':x2.clone(),'forward_add':(a+d).to(x1.dtype),'rollback_cast_error':float(torch.linalg.vector_norm(rollback.double()-(b-d)).item())}

def fp64_attention(q,k,v,R,tail,kw):
 """Reconstruct one true SDPA call at R; supports causal, additive/bool masks, GQA."""
 assert q.ndim==k.ndim==v.ndim==4 and k.shape[:-1]==v.shape[:-1] and q.shape[0]==k.shape[0] and q.shape[-1]==k.shape[-1] and 0<=R<q.shape[-2] and q.shape[-2]>0 and k.shape[-2]>0
 q=q.double();k=k.double();v=v.double(); mask=kw.get('attn_mask',tail[0] if len(tail)>0 else None);drop=kw.get('dropout_p',tail[1] if len(tail)>1 else 0.);causal=kw.get('is_causal',tail[2] if len(tail)>2 else False);scale=kw.get('scale',None);gqa=kw.get('enable_gqa',False);assert drop==0 and not (causal and mask is not None)
 if causal:assert q.shape[-2]==k.shape[-2], 'causal offset unsupported without cache alignment'
 if q.shape[-3]!=k.shape[-3]:
  assert gqa and q.shape[-3]%k.shape[-3]==0;k=k.repeat_interleave(q.shape[-3]//k.shape[-3],dim=-3);v=v.repeat_interleave(q.shape[-3]//v.shape[-3],dim=-3)
 logits=(q[...,R:R+1,:]@k.transpose(-2,-1))*(scale if scale is not None else 1/math.sqrt(q.shape[-1]))
 if causal:
  valid=torch.arange(k.shape[-2],device=k.device)<=R;logits=logits.masked_fill(~valid,float('-inf'))
 if mask is not None:
  assert 1<=mask.ndim<=4
  m=torch.broadcast_to(mask,(q.shape[0],q.shape[-3],q.shape[-2],k.shape[-2]))[...,R:R+1,:]
  logits=logits.masked_fill(~m,float('-inf')) if m.dtype==torch.bool else logits+m.double()
 A=torch.softmax(logits,dim=-1);return A,A@v

@contextmanager
def intervene(layer,*,physical_index,ut,R,expected_id,current_id,expected_shape,expected_input_ids_sha,input_ids_sha,patch=None,capture_sdpa=True,expected_full=None):
 """One receiver call; patch only selected R at specific loop, restore hooks and SDPA on exit."""
 assert physical_index==17 and ut in (0,1) and expected_id==current_id and expected_input_ids_sha==input_ids_sha
 rec={'events':[],'outside_R_unchanged':None};active=[];attn=[];old=F.scaled_dot_product_attention;hs=[]
 def pre(m,args,kw):
  actual=kw.get('current_ut');assert isinstance(actual,int) and actual==len(rec['events']);rec['events'].append(actual)
  if actual!=ut:return
  assert not active and args and tuple(args[0].shape)==tuple(expected_shape) and args[0].dtype==torch.bfloat16 and 0<=R<args[0].shape[-2]
  active.append(actual);x=args[0];assert torch.isfinite(x).all();assert expected_full is None or bits(x,expected_full.to(x.device));rec['original_full']=x.detach().clone();rec['original_R']=x[...,R,:].detach().clone()
  if patch is None:rec['injected_R']=rec['original_R'].clone();rec['outside_R_unchanged']=True;return
  assert patch.shape==x[...,R,:].shape and patch.dtype==x.dtype and torch.isfinite(patch).all();y=x.clone();y[...,R,:]=patch.to(x.device);mask=torch.arange(x.shape[-2],device=x.device)!=R;assert bits(y[...,mask,:],x[...,mask,:]);rec['outside_R_unchanged']=True;rec['injected_R']=y[...,R,:].detach().clone();return (y,)+args[1:],kw
 def post(m,args,kw,out):
  actual=kw.get('current_ut')
  if actual==ut:
   rec['z_R']=(out[0] if isinstance(out,tuple) else out)[...,R,:].detach().clone();
   if 'a_R' in rec and 'y_R' in rec:assert bits(rec['y_R'],(rec['injected_R']+rec['a_R']).to(rec['injected_R'].dtype))
   active.pop()
 def ap(m,args,kw):
  if active:attn.append(ut)
 def az(m,args,kw,out):
  if active:assert attn.pop()==ut
 def norm(m,args,out):
  if active:assert 'a_R' not in rec;rec['a_R']=out[...,R,:].detach().clone()
 def ypre(m,args):
  if active:assert 'y_R' not in rec;rec['y_R']=args[0][...,R,:].detach().clone()
 def norm2(m,args,out):
  if active:assert 'mlp_norm_R' not in rec;rec['mlp_norm_R']=out[...,R,:].detach().clone()
 def wrap(*args,**kw):
  out=old(*args,**kw)
  if active and attn:
   assert 'sdpa_R' not in rec;q,k,v=args[:3];rec['Q_R']=q[...,R,:].detach().clone();rec['sdpa_R']=out[...,R,:].detach().clone();rec['QKV']=[a.detach().clone() for a in (q,k,v)];rec['sdpa_tail']=args[3:];rec['sdpa_kwargs']=kw.copy()
   A,recon=fp64_attention(q,k,v,R,args[3:],kw);rec['A_R_fp64']=A;rec['sdpa_R_fp64']=recon.squeeze(-2)
  return out
 try:
  hs=[layer.register_forward_pre_hook(pre,with_kwargs=True),layer.register_forward_hook(post,with_kwargs=True)]
  if capture_sdpa:
   hs += [layer.self_attn.register_forward_pre_hook(ap,with_kwargs=True),layer.self_attn.register_forward_hook(az,with_kwargs=True),layer.input_layernorm_2.register_forward_hook(norm),layer.post_attention_layernorm.register_forward_pre_hook(ypre),layer.post_attention_layernorm_2.register_forward_hook(norm2)]
   F.scaled_dot_product_attention=wrap
  yield rec
  assert rec['events']==list(range(ut+1)) and not active and 'original_R' in rec
  if capture_sdpa:
   assert all(k in rec for k in ('Q_R','sdpa_R','A_R_fp64','sdpa_R_fp64','a_R','mlp_norm_R','y_R','z_R')) and all(torch.isfinite(rec[k]).all() for k in ('Q_R','sdpa_R','A_R_fp64','sdpa_R_fp64','a_R','mlp_norm_R','y_R','z_R'))
 finally:
  F.scaled_dot_product_attention=old
  for h in hs:h.remove()

