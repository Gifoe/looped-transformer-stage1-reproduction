"""CPU-only independent arithmetic checks and implementation freeze."""
import datetime,json
from pathlib import Path
import numpy as np
import torch
import C1_CONTROL_TENSOR_ARITHMETIC_AUDIT as a

assert not torch.cuda.is_initialized() and np.finfo(np.longdouble).nmant>52
assert a.norm(np.array([3,4],dtype=np.longdouble))==5
error,bound=a.close(torch.tensor([6.,8.],dtype=torch.float64),np.array([3,4],dtype=np.longdouble)*2)
assert error==0 and bound>0
try:a.close(torch.tensor([1.],dtype=torch.float64),np.array([0],dtype=np.longdouble))
except AssertionError:rejected=True
else:rejected=False
assert rejected
proposal=torch.tensor([.1,-.3,1.125],dtype=torch.float64);after=proposal.to(torch.bfloat16)
residual=after.double()-proposal;assert torch.equal(proposal+residual,after.double())
root=Path(__file__).parent;path=root/'analysis/C1_CONTROL_AUDIT_SOURCE_FREEZE.json';assert not path.exists()
a.write(path,{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'FROZEN',
  'source_sha256':a.sha(a.__file__),'tensor_reader_sha256':a.sha(root/'C1_AUDIT_TENSOR_IO.py'),
  'extended_precision_mantissa_bits':int(np.finfo(np.longdouble).nmant),'known_3_4_norm':5,
  'known_scale_exact':True,'incorrect_proposal_rejected':rejected,'BF16_cast_residual_identity':True,
  'FP64_envelope':'32*max(2048,n)*eps64*(1+maxabs(expected)); includes source-vector reductions for scalar norm/scale metadata.',
  'chronology':'C1 collection is already underway. This independent audit implementation/bound is frozen before its control-tensor results are inspected; no experimental, sample, estimator or decision-gate change.'})
print('C1_CONTROL_AUDIT_FROZEN',a.sha(path),flush=True)
