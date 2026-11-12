import torch
from e1_control_core import controls
torch.manual_seed(33);s=[torch.randn(1,4,8).to(torch.bfloat16) for _ in range(4)];d=torch.randn(1,4,8,dtype=torch.float64);o=controls(*s,d);assert torch.equal(o['restore'],s[3]);assert torch.equal(o['ADD'],(s[2].double()+s[1].double()-s[0].double()).to(torch.bfloat16));assert torch.allclose(o['ADD_exact_fp64']+o['cast_residual_fp64'],o['ADD'].double(),rtol=0,atol=0)
z=torch.zeros(1,4,8,dtype=torch.bfloat16);o=controls(z,z,z,z);assert torch.equal(o['ID'],o['ADD_NORM']) and torch.equal(o['UNRELATED'],z)
o=controls(z,z,z,torch.ones_like(z),torch.ones_like(z,dtype=torch.float64));assert o['ADD_NORM'] is None
try:controls(*s,torch.zeros_like(d))
except ValueError:pass
else:raise AssertionError('zero unrelated donor accepted')
print('E1_ADD_RESTORE_NORM_CONTROL_CPU_PASS')
