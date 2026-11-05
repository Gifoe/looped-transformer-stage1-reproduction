"""Coordinatewise actual normalized common gain, anchored to literal FP32 Fixed."""
import numpy as np
from r7_math import token_logits_numpy,token_logits
class NumericDomainError(ValueError):pass
def np_center(p,x):
 shifted=x-x[..., :1];return shifted-np.sum(p*shifted,axis=-1,keepdims=True)
def np_checked_norm(p,h):
 if not(np.isfinite(p).all() and np.all(p>0) and np.isfinite(h).all()):raise NumericDomainError('Finite strictly positive native measure required')
 n2=np.sum(p*h*h);zero=bool(np.all(h==0))
 if not(np.isfinite(n2) and n2>=0 and (n2==0)==zero):raise NumericDomainError('Nonzero correction has zero/nonfinite weighted norm')
 return n2,zero
def np_logp(z):
 x=z.astype(np.float64);x=x-np.max(x,axis=-1,keepdims=True);return x-np.log(np.exp(x).sum(axis=-1,keepdims=True))
def np_signed_min(vectors):
 return np.maximum(vectors,0.).min(axis=0)-np.maximum(-vectors,0.).min(axis=0)
def numpy_readouts(q1,q4,n1,n4,permutation):
 assert q1.dtype==q4.dtype==n1.dtype==n4.dtype==np.float32 and q1.ndim==q4.ndim==1 and q1.shape==q4.shape and n1.shape==n4.shape==(3,len(q4)) and permutation.shape==q4.shape
 if not all(np.isfinite(z).all() for z in [q1,q4,n1,n4]):raise NumericDomainError('Nonfinite logits')
 g=np.float32(.5)*(q4-q1);fixed=np.float32(q4+g);ng=np.float32(.5)*(n4-n1);nf=np.float32(n4+ng)
 if not all(np.isfinite(z).all() for z in [g,fixed,ng,nf]):raise NumericDomainError('Nonfinite literal Fixed arithmetic')
 G=np_logp(fixed)-np_logp(q4);E=np_logp(nf)-np_logp(n4);p=np.exp(q4.astype(np.float64)-float(np.max(q4)));p/=p.sum()
 if not(np.isfinite(p).all() and np.all(p>0)):raise NumericDomainError('Native FP64 measure underflow/nonfinite')
 c=np_signed_min(np.concatenate([G[None],E],axis=0));h=np_center(p,c);main2,hzero=np_checked_norm(p,h);primary=fixed.copy() if hzero else np.float32(fixed.astype(np.float64)-h)
 residual=G-c
 if not(np.all(np.where(G>=0,(residual>=0)&(residual<=G),(residual>=G)&(residual<=0)))):raise NumericDomainError('Local common-gain clamp failed')
 def matched(direction):
  v=np_center(p,direction);n2=np.sum(p*v*v);threshold=64*np.finfo(np.float64).eps*np.max(np.abs(v))**2
  if not(np.isfinite(v).all() and np.isfinite(n2) and n2>=0 and np.isfinite(threshold)):raise NumericDomainError('Nonfinite matched-control direction')
  bad=bool(not hzero and n2<=threshold)
  if hzero or bad:return fixed.copy(),bad,0. if hzero else float(np.sqrt(main2)),n2,threshold
  scale=np.sqrt(main2/n2);v=v*scale;actual=np.sum(p*v*v);z=np.float32(fixed.astype(np.float64)-v);error=abs(np.sqrt(actual)-np.sqrt(main2))
  if not(np.isfinite(scale) and np.isfinite(v).all() and np.isfinite(z).all() and np.isfinite(error)):raise NumericDomainError('Nonfinite norm matching')
  return z,bad,error,n2,threshold
 generic=np_center(p,ng.astype(np.float64)).mean(axis=0);consensus=np_signed_min(E)
 gm,gb,ge,gn,gt=matched(generic);null,nb,ne,nn,nt=matched(h[permutation]);cons,cb,ce,cn,ct=matched(consensus)
 gc=np_center(p,g.astype(np.float64));g2=np.sum(p*gc*gc);r2=np.sum(p*(gc-h)*(gc-h));scale_bad=bool(not hzero and g2==0)
 if not(np.isfinite(g2) and g2>=0 and np.isfinite(r2) and r2>=0):raise NumericDomainError('Nonfinite residual scale norm')
 rho=1. if hzero or scale_bad else float(np.sqrt(r2/g2));scaled=fixed.copy() if hzero or scale_bad else np.float32(q4.astype(np.float64)+rho*g.astype(np.float64));scale_error=0. if hzero or scale_bad else abs(np.sqrt(rho*rho*g2)-np.sqrt(r2))
 loo=[];loonorm=[]
 for j in range(3):
  cl=np_signed_min(np.concatenate([G[None],np.delete(E,j,axis=0)],axis=0));hl=np_center(p,cl);l2,lzero=np_checked_norm(p,hl);loo.append(fixed.copy() if lzero else np.float32(fixed.astype(np.float64)-hl));loonorm.append(l2)
 m2,_=token_logits_numpy(q1,q4,n1[:1],n4[:1]);m3,_=token_logits_numpy(q1,q4,n1,n4);j2,_=token_logits_numpy(q1,q4,n1[1:],n4[1:])
 out=dict(primary=primary,Fixed=fixed,M2=m2,M3=m3,J2_guidance=j2,G_matched=gm,vocabulary_null=null,matched_scale=scaled,reference_consensus_matched=cons,LOO_N0=loo[0],LOO_N1=loo[1],LOO_N2=loo[2]);diag=dict(mode='coordinate_actual_common_gain',n_references=3,main_norm2=main2,h_exact_zero=hzero,c_exact_zero=bool(np.all(c==0)),positive_coordinates=int(np.sum(c>0)),negative_coordinates=int(np.sum(c<0)),scale_g2=g2,scale_r2=r2,scale_rho=rho,scale_unmatched=scale_bad,scale_norm_error=scale_error,G_unmatched=gb,G_norm_error=ge,G_raw_norm2=gn,G_threshold=gt,null_unmatched=nb,null_norm_error=ne,null_raw_norm2=nn,null_threshold=nt,consensus_unmatched=cb,consensus_norm_error=ce,consensus_raw_norm2=cn,consensus_threshold=ct,LOO_N0_norm2=loonorm[0],LOO_N1_norm2=loonorm[1],LOO_N2_norm2=loonorm[2])
 if not all(np.isfinite(v).all() for v in out.values()) or not all(np.isfinite(v) for v in diag.values() if not isinstance(v,str)):raise NumericDomainError('Nonfinite output/diagnostic')
 return out,diag

def torch_center(p,x):
 s=x-x[..., :1];return s-(p*s).sum(dim=-1,keepdim=True)
def torch_checked_norm(p,h):
 import torch
 if not bool((torch.isfinite(p).all()&(p>0).all()&torch.isfinite(h).all()).item()):raise NumericDomainError('Finite strictly positive native measure required')
 n2=(p*h*h).sum();zero=(h==0).all()
 if not bool((torch.isfinite(n2)&(n2>=0)&((n2==0)==zero)).item()):raise NumericDomainError('Nonzero correction has zero/nonfinite weighted norm')
 return n2,zero
def torch_signed_min(vectors):
 return vectors.clamp_min(0).amin(dim=0)-(-vectors).clamp_min(0).amin(dim=0)
def torch_readouts(q1,q4,n1,n4,permutation):
 import torch
 with torch.no_grad():
  assert q1.dtype==q4.dtype==n1.dtype==n4.dtype==torch.float32 and q1.ndim==q4.ndim==1 and q1.shape==q4.shape and n1.shape==n4.shape==(3,len(q4)) and permutation.shape==q4.shape
  if not all(bool(torch.isfinite(z).all()) for z in [q1,q4,n1,n4]):raise NumericDomainError('Nonfinite logits')
  g=.5*(q4-q1);fixed=q4+g;ng=.5*(n4-n1);nf=n4+ng
  if not all(bool(torch.isfinite(z).all()) for z in [g,fixed,ng,nf]):raise NumericDomainError('Nonfinite literal Fixed arithmetic')
  G=fixed.double().log_softmax(-1)-q4.double().log_softmax(-1);E=nf.double().log_softmax(-1)-n4.double().log_softmax(-1);p=q4.double().softmax(-1)
  if not bool((torch.isfinite(p).all()&(p>0).all()).item()):raise NumericDomainError('Native FP64 measure underflow/nonfinite')
  c=torch_signed_min(torch.cat([G[None],E],dim=0));h=torch_center(p,c);main2,hzero=torch_checked_norm(p,h);primary=fixed if bool(hzero) else (fixed.double()-h).float();residual=G-c
  if not bool(torch.where(G>=0,(residual>=0)&(residual<=G),(residual>=G)&(residual<=0)).all()):raise NumericDomainError('Local common-gain clamp failed')
  def matched(direction):
   v=torch_center(p,direction);n2=(p*v*v).sum();threshold=64*torch.finfo(torch.float64).eps*v.abs().amax().square()
   if not bool((torch.isfinite(v).all()&torch.isfinite(n2)&(n2>=0)&torch.isfinite(threshold)).item()):raise NumericDomainError('Nonfinite matched-control direction')
   bad=(~hzero)&(n2<=threshold)
   if bool(hzero|bad):return fixed,bad,torch.where(hzero,torch.zeros_like(main2),main2.sqrt()),n2,threshold
   scale=(main2/n2).sqrt();v=v*scale;actual=(p*v*v).sum();z=(fixed.double()-v).float();error=(actual.sqrt()-main2.sqrt()).abs()
   if not bool((torch.isfinite(scale)&torch.isfinite(v).all()&torch.isfinite(z).all()&torch.isfinite(error)).item()):raise NumericDomainError('Nonfinite norm matching')
   return z,bad,error,n2,threshold
  generic=torch_center(p,ng.double()).mean(dim=0);consensus=torch_signed_min(E);gm,gb,ge,gn,gt=matched(generic);null,nb,ne,nn,nt=matched(h[permutation]);cons,cb,ce,cn,ct=matched(consensus)
  gc=torch_center(p,g.double());g2=(p*gc*gc).sum();r2=(p*(gc-h).square()).sum();scale_bad=(~hzero)&(g2==0)
  if not bool((torch.isfinite(g2)&(g2>=0)&torch.isfinite(r2)&(r2>=0)).item()):raise NumericDomainError('Nonfinite residual scale norm')
  rho=torch.ones_like(g2) if bool(hzero|scale_bad) else (r2/g2).sqrt();scaled=fixed if bool(hzero|scale_bad) else (q4.double()+rho*g.double()).float();scale_error=torch.zeros_like(g2) if bool(hzero|scale_bad) else ((rho*rho*g2).sqrt()-r2.sqrt()).abs()
  loo=[];loonorm=[]
  for j in range(3):
   cl=torch_signed_min(torch.cat([G[None],E[[i for i in range(3) if i!=j]]],dim=0));hl=torch_center(p,cl);l2,lzero=torch_checked_norm(p,hl);loo.append(fixed if bool(lzero) else (fixed.double()-hl).float());loonorm.append(l2)
  m2,_=token_logits(q1,q4,n1[:1],n4[:1]);m3,_=token_logits(q1,q4,n1,n4);j2,_=token_logits(q1,q4,n1[1:],n4[1:]);out=dict(primary=primary,Fixed=fixed,M2=m2,M3=m3,J2_guidance=j2,G_matched=gm,vocabulary_null=null,matched_scale=scaled,reference_consensus_matched=cons,LOO_N0=loo[0],LOO_N1=loo[1],LOO_N2=loo[2]);diag=dict(mode='coordinate_actual_common_gain',n_references=3,main_norm2=main2,h_exact_zero=hzero,c_exact_zero=(c==0).all(),positive_coordinates=(c>0).sum(),negative_coordinates=(c<0).sum(),scale_g2=g2,scale_r2=r2,scale_rho=rho,scale_unmatched=scale_bad,scale_norm_error=scale_error,G_unmatched=gb,G_norm_error=ge,G_raw_norm2=gn,G_threshold=gt,null_unmatched=nb,null_norm_error=ne,null_raw_norm2=nn,null_threshold=nt,consensus_unmatched=cb,consensus_norm_error=ce,consensus_raw_norm2=cn,consensus_threshold=ct,LOO_N0_norm2=loonorm[0],LOO_N1_norm2=loonorm[1],LOO_N2_norm2=loonorm[2])
  if not all(bool(torch.isfinite(v).all()) for v in out.values()) or not all(bool(torch.isfinite(v).all()) for v in diag.values() if isinstance(v,torch.Tensor)):raise NumericDomainError('Nonfinite output/diagnostic')
  return out,diag
