"""Prepared only: alternate Q_R rollback semantics, pending a separate Astra GPU order."""
import torch
import torch.nn.functional as F
from observer import bits
if not __debug__:raise RuntimeError('optimized Python forbidden')

def derive_q_patch(receiver,donor,R,mode='q_only',fp32_sdpa=False,return_receipt=False):
 """Complete-shape SDPA, donor Q only at R; return only the R result for injection."""
 assert mode in ('q_only','joint_qkv','receiver_q_to_donor_headnorm','donor_q_to_receiver_headnorm')
 q,k,v,tail,kw=receiver;dq,dk,dv,dtail,dkw=donor
 assert all(a.shape==b.shape and a.dtype==b.dtype and a.device==b.device for a,b in zip((q,k,v),(dq,dk,dv)))
 assert len(tail)==len(dtail) and set(kw)==set(dkw)
 for a,b in list(zip(tail,dtail))+[(kw[n],dkw[n]) for n in kw]:
  assert bits(a,b) if torch.is_tensor(a) else a==b
 receiver_r=q[...,R,:];donor_r=dq[...,R,:]
 rn=receiver_r.float().norm(dim=-1,keepdim=True);dn=donor_r.float().norm(dim=-1,keepdim=True)
 zero=(rn==0)|(dn==0)
 receipt={'mode':mode,'receiver_headnorms':rn.squeeze(-1).tolist(),'donor_headnorms':dn.squeeze(-1).tolist(),'zero_head_indices':torch.nonzero(zero.squeeze(-1)).tolist(),'norm_rescale_arithmetic':'FP32_PER_HEAD_THEN_CAST_TO_Q_DTYPE'}
 if mode in ('receiver_q_to_donor_headnorm','donor_q_to_receiver_headnorm'):
  if zero.any():raise ZeroDivisionError('unavailable zero Q head norms: '+str(receipt))
  replacement=(receiver_r.float()*(dn/rn) if mode=='receiver_q_to_donor_headnorm' else donor_r.float()*(rn/dn)).to(q.dtype)
 else:replacement=donor_r
 receipt['injected_headnorms_after_cast']=replacement.float().norm(dim=-1).tolist()
 mixed_q=q.clone();mixed_q[...,R,:]=replacement
 mask=torch.arange(q.shape[-2],device=q.device)!=R
 assert bits(mixed_q[...,mask,:],q[...,mask,:])
 use_k,use_v=(dk,dv) if mode=='joint_qkv' else (k,v)
 if fp32_sdpa:
  mixed_q,use_k,use_v=mixed_q.float(),use_k.float(),use_v.float()
  tail=tuple(x.float() if torch.is_tensor(x) and x.dtype.is_floating_point else x for x in tail)
  kw={n:x.float() if torch.is_tensor(x) and x.dtype.is_floating_point else x for n,x in kw.items()}
 result=F.scaled_dot_product_attention(mixed_q,use_k,use_v,*tail,**kw)[...,R,:]
 assert torch.isfinite(result).all()
 patch=result.to(q.dtype).clone()
 return (patch,receipt) if return_receipt else patch
