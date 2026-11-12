import torch
from e3_source_control_v2 import build
torch.manual_seed(52);k1=torch.randn(1,4,9,3).bfloat16();k2=torch.randn_like(k1);v1=torch.randn_like(k1);v2=torch.randn_like(k1);R=8;P=4
for c in ('ID','K_Q','V_Q','KV_Q','KV_P','K_Q_NORM','V_Q_NORM','V_Q_ROLL'):
 k,v,m=build(k1,k2,v1,v2,R,P,c);assert m['status']=='OK';lo,hi=m['region'];assert torch.equal(k[...,:lo,:],k2[...,:lo,:]) and torch.equal(k[...,hi:,:],k2[...,hi:,:]);assert torch.equal(v[...,:lo,:],v2[...,:lo,:]) and torch.equal(v[...,hi:,:],v2[...,hi:,:])
 if c=='K_Q':assert torch.equal(k[...,P:R,:],k1[...,P:R,:]) and torch.equal(v,v2)
 if c=='V_Q':assert torch.equal(v[...,P:R,:],v1[...,P:R,:]) and torch.equal(k,k2)
 if c=='KV_Q':assert torch.equal(k[...,P:R,:],k1[...,P:R,:]) and torch.equal(v[...,P:R,:],v1[...,P:R,:])
 if c=='KV_P':assert torch.equal(k[...,:P,:],k1[...,:P,:]) and torch.equal(v[...,:P,:],v1[...,:P,:])
 if c=='V_Q_ROLL':assert torch.equal(v[...,P:R,:],torch.roll(v2[...,P:R,:],1,-2))
z=torch.zeros_like(k1);k,v,m=build(z,k2,v1,v2,R,P,'K_Q_NORM');assert k is None and m['status']=='UNDEFINED_ZERO_DONOR_NONZERO_TARGET'
k,v,m=build(z,z,v1,v2,R,P,'K_Q_NORM');assert m['status']=='OK' and torch.equal(k,z)
print('E3_SOURCE_REGION_NORM_PERMUTATION_CPU_PASS')
