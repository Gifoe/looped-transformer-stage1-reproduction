import torch
from e2_r_producer import producer_transfer,bits
torch.manual_seed(42);torch.set_num_threads(1)
class Layer(torch.nn.Module):
 def __init__(self,i):super().__init__();self.i=i;self.w=torch.nn.Parameter(torch.randn(7,7)*.04,requires_grad=False)
 def forward(self,x,current_ut=0):return (x.float()+torch.tanh(x.float().cumsum(1)@self.w*.2+current_ut*.02+self.i*.001)).to(torch.bfloat16)
class M(torch.nn.Module):
 def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([Layer(i) for i in range(24)])
 def forward(self,x):
  for u in range(2):
   for l in self.layers:x=l(x,current_ut=u)
  return x
m=M().eval();x=torch.randn(1,6,7).to(torch.bfloat16);R=4;before=[p.clone() for p in m.parameters()]
for cut in (0,8,16,17):
 full={};h=x.clone()
 for u in range(2):
  for i,l in enumerate(m.layers):
   if i==cut:full[u]=h.clone()
   h=l(h,current_ut=u)
 baseline=h; x1=full[0][...,R,:];x2full=full[1];kw=dict(cut_index=cut,R=R,expected_id='a',current_id='a',input_sha='h',expected_input_sha='h',expected_cut_full=x2full,capture_attention=False)
 with producer_transfer(m.layers,**kw) as clean:out_clean=m(x)
 assert bits(out_clean,baseline);clean_target=clean['target_before_full'][...,R,:].clone()
 with producer_transfer(m.layers,**kw,producer_patch=x1) as disturbed:out_dist=m(x)
 disturbed_target=disturbed['target_before_full'][...,R,:].clone()
 def oracle(prod,target):
  h=x.clone()
  for u in range(2):
   for i,l in enumerate(m.layers):
    if u==1 and i==cut and prod is not None:h=h.clone();h[...,R,:]=prod
    if u==1 and i==17 and target is not None:h=h.clone();h[...,R,:]=target
    h=l(h,current_ut=u)
  return h
 for prod in (None,x1):
  for target in (clean_target,disturbed_target):
   with producer_transfer(m.layers,**kw,producer_patch=prod,target_patch=target) as rec:y=m(x)
   assert bits(y,oracle(prod,target)) and rec['target_outside_R_unchanged'] and (prod is None or rec['cut_outside_R_unchanged'])
 if cut==17:assert bits(out_dist,oracle(x1,None))
assert all(bits(a,b) for a,b in zip(before,m.parameters())) and all(not l._forward_pre_hooks for l in m.layers)
rejected=False
try:
 with producer_transfer(m.layers,cut_index=4,R=R,expected_id='a',current_id='a',input_sha='h',expected_input_sha='h',capture_attention=False):m(x)
except AssertionError:rejected=True
assert rejected
print('PASS full 24-layer two-loop independent oracle, four technical cut geometries, four transfer corners, same-site self-cut, only-R and cleanup; no cut scientifically selected')
