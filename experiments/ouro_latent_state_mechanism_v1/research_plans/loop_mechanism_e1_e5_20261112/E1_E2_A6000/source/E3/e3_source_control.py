"""Full-sequence post-RoPE K/V source region controls; no model execution."""
import torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
def _match(donor,target):
 a=donor.float();b=target.float();dn=torch.linalg.vector_norm(a,dim=-1,keepdim=True);tn=torch.linalg.vector_norm(b,dim=-1,keepdim=True);bad=(dn==0)&(tn!=0)
 if bad.any():return None,{'status':'UNDEFINED_ZERO_DONOR_NONZERO_TARGET','undefined_elements':int(bad.sum())}
 ratio=torch.where(dn!=0,tn/dn,torch.zeros_like(dn));out=(a*ratio).to(torch.bfloat16);return out,{'status':'OK','max_abs_norm_error':float((torch.linalg.vector_norm(out.float(),dim=-1,keepdim=True)-tn).abs().max()) if out.numel() else 0.0,'zero_both_count':int(((dn==0)&(tn==0)).sum())}
def build(k1,k2,v1,v2,R,prefix_length,condition):
 values=(k1,k2,v1,v2)
 if any(not torch.is_tensor(x) or x.dtype!=torch.bfloat16 or x.ndim!=4 or x.shape!=k1.shape or x.device!=k1.device or not torch.isfinite(x).all() for x in values):raise ValueError('source K/V')
 L=k1.shape[-2]
 if type(R)!=int or not(0<prefix_length<R==L-1):raise ValueError('P/Q/R boundary')
 allowed={'ID','K_Q','V_Q','KV_Q','KV_P','K_Q_NORM','V_Q_NORM','V_Q_ROLL'}
 if condition not in allowed:raise ValueError('unknown condition')
 lo,hi=(0,prefix_length) if condition=='KV_P' else (prefix_length,R);K=k2.clone();V=v2.clone();meta={'status':'OK','condition':condition,'region':[lo,hi]}
 if condition in ('K_Q','KV_Q','KV_P'):K[...,lo:hi,:]=k1[...,lo:hi,:]
 if condition in ('V_Q','KV_Q','KV_P'):V[...,lo:hi,:]=v1[...,lo:hi,:]
 if condition=='K_Q_NORM':
  matched,info=_match(k1[...,lo:hi,:],k2[...,lo:hi,:]);meta.update(info)
  if matched is None:return None,None,meta
  K[...,lo:hi,:]=matched
 if condition=='V_Q_NORM':
  matched,info=_match(v1[...,lo:hi,:],v2[...,lo:hi,:]);meta.update(info)
  if matched is None:return None,None,meta
  V[...,lo:hi,:]=matched
 if condition=='V_Q_ROLL':V[...,lo:hi,:]=torch.roll(v2[...,lo:hi,:],shifts=1,dims=-2);meta['roll_shift']=1
 outside=torch.ones(L,dtype=torch.bool,device=K.device);outside[lo:hi]=False
 if not torch.equal(K[...,outside,:],k2[...,outside,:]) or not torch.equal(V[...,outside,:],v2[...,outside,:]):raise ValueError('outside region changed')
 return K,V,meta
