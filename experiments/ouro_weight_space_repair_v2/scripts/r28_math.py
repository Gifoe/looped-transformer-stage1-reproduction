"""Fixed source-defined J2 prior; original R7 projection and FP32 outputs."""
import numpy as np
from r7_math import token_logits,token_logits_numpy

def numpy_readouts(q1,q4,n1,n4,b4,permutation):
 assert q1.dtype==q4.dtype==n1.dtype==n4.dtype==b4.dtype==np.float32 and n1.shape==n4.shape==(3,)+q4.shape and b4.shape==(2,)+q4.shape
 primary,d=token_logits_numpy(q1,q4,b4,n4[1:]);guidance,gd=token_logits_numpy(q1,q4,n1[1:],n4[1:]);m2,_=token_logits_numpy(q1,q4,n1[:1],n4[:1]);m3,_=token_logits_numpy(q1,q4,n1,n4)
 fixed=np.float32(q4+np.float32(.5)*(q4-q1));p=q4.astype(np.float64);p=np.exp(p-p.max(axis=-1,keepdims=True));p/=p.sum(axis=-1,keepdims=True)
 def center(x):s=x-x[..., :1];return s-np.sum(p*s,axis=-1,keepdims=True)
 ec=center((np.float32(.5)*(n4[1:]-b4)).astype(np.float64));h=d['alpha'][...,None]*ec.mean(axis=0);main2=np.sum(p*h*h,axis=-1)
 eg=center((np.float32(.5)*(n4[1:]-n1[1:])).astype(np.float64)).mean(axis=0)
 def matched(direction):
  v=center(direction);n2=np.sum(p*v*v,axis=-1);threshold=64*np.finfo(np.float64).eps*np.max(np.abs(v),axis=-1)**2;bad=(main2>0)&(n2<=threshold);zero=main2==0;scale=np.sqrt(main2/np.where(zero|bad,1.,n2));v=v*scale[...,None];v=np.where((zero|bad)[...,None],0.,v);z=np.float32(fixed.astype(np.float64)-v);z=np.where((zero|bad)[...,None],fixed,z);actual=np.sum(p*v*v,axis=-1);error=np.abs(np.sqrt(actual)-np.sqrt(main2));return z,bad,error,n2,threshold
 gm,g_bad,g_error,graw,gthr=matched(eg);null,n_bad,n_error,nraw,nthr=matched(h[...,permutation]);loo=[token_logits_numpy(q1,q4,b4[j:j+1],n4[j+1:j+2])[0] for j in range(2)]
 out=dict(primary=primary,J2_guidance=guidance,M2=m2,M3=m3,G_matched=gm,vocabulary_null=null,LOO_N1=loo[0],LOO_N2=loo[1],Fixed=fixed)
 assert all(np.isfinite(x).all() for x in out.values())
 diag=dict(d,main_norm2=main2,G_unmatched=g_bad,null_unmatched=n_bad,G_norm_error=g_error,null_norm_error=n_error,G_raw_norm2=graw,null_raw_norm2=nraw,G_degenerate_threshold=gthr,null_degenerate_threshold=nthr,J2_alpha=gd['alpha'])
 return out,diag

def torch_readouts(q1,q4,n1,n4,b4,permutation):
 import torch
 with torch.no_grad():
  assert q1.dtype==q4.dtype==n1.dtype==n4.dtype==b4.dtype==torch.float32 and n1.shape==n4.shape==(3,)+q4.shape and b4.shape==(2,)+q4.shape
  primary,d=token_logits(q1,q4,b4,n4[1:]);guidance,gd=token_logits(q1,q4,n1[1:],n4[1:]);m2,_=token_logits(q1,q4,n1[:1],n4[:1]);m3,_=token_logits(q1,q4,n1,n4)
  fixed=q4+.5*(q4-q1);p=torch.softmax(q4.double(),dim=-1)
  def center(x):s=x-x[..., :1];return s-(p*s).sum(dim=-1,keepdim=True)
  ec=center((.5*(n4[1:]-b4)).double());h=d['alpha'][...,None]*ec.mean(dim=0);main2=(p*h*h).sum(dim=-1);eg=center((.5*(n4[1:]-n1[1:])).double()).mean(dim=0)
  def matched(direction):
   v=center(direction);n2=(p*v*v).sum(dim=-1);threshold=64*torch.finfo(torch.float64).eps*v.abs().amax(dim=-1).square();bad=(main2>0)&(n2<=threshold);zero=main2==0;scale=(main2/torch.where(zero|bad,torch.ones_like(n2),n2)).sqrt();v=v*scale[...,None];v=torch.where((zero|bad)[...,None],torch.zeros_like(v),v);z=(fixed.double()-v).float();z=torch.where((zero|bad)[...,None],fixed,z);actual=(p*v*v).sum(dim=-1);error=(actual.sqrt()-main2.sqrt()).abs();return z,bad,error,n2,threshold
  gm,g_bad,g_error,graw,gthr=matched(eg);null,n_bad,n_error,nraw,nthr=matched(h[...,permutation]);loo=[token_logits(q1,q4,b4[j:j+1],n4[j+1:j+2])[0] for j in range(2)]
  out=dict(primary=primary,J2_guidance=guidance,M2=m2,M3=m3,G_matched=gm,vocabulary_null=null,LOO_N1=loo[0],LOO_N2=loo[1],Fixed=fixed)
  if not all(bool(torch.isfinite(x).all()) for x in out.values()):raise ValueError('Nonfinite R28 output')
  diag=dict(d,main_norm2=main2,G_unmatched=g_bad,null_unmatched=n_bad,G_norm_error=g_error,null_norm_error=n_error,G_raw_norm2=graw,null_raw_norm2=nraw,G_degenerate_threshold=gthr,null_degenerate_threshold=nthr,J2_alpha=gd['alpha'])
  return out,diag
