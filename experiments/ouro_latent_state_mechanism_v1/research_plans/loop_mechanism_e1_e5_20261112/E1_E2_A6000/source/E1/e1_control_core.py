"""Frozen E1 ADD/identity controls at actual SDPA R, no model calls here."""
import torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
def controls(s11,s12,s21,s22,donor_delta=None):
 ss=(s11,s12,s21,s22)
 if any(not torch.is_tensor(x) or x.dtype!=torch.bfloat16 or x.shape!=s22.shape or x.device!=s22.device or not torch.isfinite(x).all() for x in ss):raise ValueError('four BF16 corners finite, same shape/device required')
 x11,x12,x21,x22=[x.to(torch.float64) for x in ss]; exact=x21+x12-x11; add=exact.to(torch.bfloat16); delta=x22-add.to(torch.float64); restored=(add.to(torch.float64)+delta).to(torch.bfloat16)
 if not torch.equal(restored,s22):raise ValueError('restore not bitwise equal')
 a=add.to(torch.float64); na=torch.linalg.vector_norm(a); n22=torch.linalg.vector_norm(x22); nd=torch.linalg.vector_norm(delta)
 if na==0:
  norm=s22.clone() if n22==0 else None
 else:norm=(a*(n22/na)).to(torch.bfloat16)
 if nd==0:unrelated=s22.clone()
 else:
  if donor_delta is None or donor_delta.dtype!=torch.float64 or donor_delta.shape!=delta.shape or donor_delta.device!=delta.device or not torch.isfinite(donor_delta).all():raise ValueError('invalid donor delta')
  donor_norm=torch.linalg.vector_norm(donor_delta)
  if donor_norm==0:raise ValueError('zero donor delta')
  unrelated=(x22-donor_delta*(nd/donor_norm)).to(torch.bfloat16)
 return {'ID':s22.clone(),'ADD':add,'ADD_NORM':norm,'UNRELATED':unrelated,'restore':restored,'ADD_exact_fp64':exact,'cast_residual_fp64':add.to(torch.float64)-exact,'delta_fp64':delta,'delta_norm':float(nd),'target_norm':float(n22),'add_norm':float(na),'actual_norm':float(torch.linalg.vector_norm(norm.to(torch.float64))) if norm is not None else None}
