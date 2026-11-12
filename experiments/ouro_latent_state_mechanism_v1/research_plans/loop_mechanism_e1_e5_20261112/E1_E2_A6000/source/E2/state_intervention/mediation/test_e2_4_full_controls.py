import torch
x1=torch.tensor([[1.,2.,3.]],dtype=torch.bfloat16);x2=torch.tensor([[4.,-2.,2.]],dtype=torch.bfloat16);o=torch.tensor([[.25,-.5,1.]],dtype=torch.float64);a=x1.double();b=x2.double();d=b-a;n=(a*(b.norm()/a.norm())).to(torch.bfloat16);u=(b-o*(d.norm()/o.norm())).to(torch.bfloat16);assert abs(float(n.double().norm()-b.norm()))<.03 and abs(float((b-u.double()).norm()-d.norm()))<.03
for original,exact in ((n,a*(b.norm()/a.norm())),(u,b-o*(d.norm()/o.norm()))):assert torch.equal((original.double()-exact)+exact,original.double())
print('PASS full-delta norm and unrelated scaling, BF16 cast residual')
