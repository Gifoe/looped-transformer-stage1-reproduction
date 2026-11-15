import torch
from branch_interfaces_v2 import UndefinedControl
def control(base,target,donor_delta,kind,sign):
 # All branch-control arithmetic runs on these archived CPU tensors in FP64, one writeback cast.
 b=base.double();t=target.double();delta=t-b;n0=b.norm();n1=t.norm();dose=delta.norm();scale=torch.ones((),dtype=torch.float64)
 if kind=='norm':
  if float(n0)==0 and float(n1)!=0:raise UndefinedControl('norm control zero base and nonzero target norm')
  scale=n1/n0 if float(n0)!=0 else scale;exact=b*scale
 else:
  direction=donor_delta.double();dn=direction.norm()
  if float(dose)==0:scale=torch.zeros((),dtype=torch.float64);exact=b.clone()
  elif float(dn)==0:raise UndefinedControl('nonzero requested dose but frozen donor direction is zero')
  else:scale=dose/dn;exact=b+sign*scale*direction
 after=exact.to(base.dtype);stats={'math_device':'CPU','math_dtype':'torch.float64','cast_dtype':str(base.dtype),'requested_delta_norm':float(dose),'actual_after_cast_delta_norm':float((after.double()-b).norm()),'base_norm':float(n0),'target_norm':float(n1),'after_norm':float(after.double().norm()),'scale_fp64':float(scale),'zero_requested_dose':float(dose)==0,'kind':kind,'sign':sign};return after,{'control_before':base,'control_exact_before_cast':exact,'control_after':after,'control_cast_residual':after.double()-exact,'control_scale_fp64':scale},stats
