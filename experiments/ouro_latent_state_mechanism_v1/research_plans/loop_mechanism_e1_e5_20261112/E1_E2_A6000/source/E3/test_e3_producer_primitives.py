import torch
from e3_producer_primitives import splice_hidden_Q,splice_postrope_K_Q
torch.manual_seed(41);h1=torch.randn(1,9,8).bfloat16();h2=torch.randn_like(h1);k1=torch.randn(1,4,9,2).bfloat16();k2=torch.randn_like(k1);p=4;r=8
h=splice_hidden_Q(h2,h1,r,p);k=splice_postrope_K_Q(k2,k1,r,p)
assert torch.equal(h[:,p:r],h1[:,p:r]) and torch.equal(h[:,:p],h2[:,:p]) and torch.equal(h[:,r:],h2[:,r:])
assert torch.equal(k[...,p:r,:],k1[...,p:r,:]) and torch.equal(k[...,:p,:],k2[...,:p,:]) and torch.equal(k[...,r:,:],k2[...,r:,:])
assert torch.equal(splice_hidden_Q(h2,h2,r,p),h2) and torch.equal(splice_postrope_K_Q(k2,k2,r,p),k2)
for fn,a,b in ((splice_hidden_Q,h1,h2),(splice_postrope_K_Q,k1,k2)):
 try:fn(a,b,3,4)
 except ValueError:pass
 else:raise AssertionError('crossed boundary')
print('E3_PRODUCER_Q_AND_POSTROPE_KEY_CPU_PRIMITIVES_PASS')
