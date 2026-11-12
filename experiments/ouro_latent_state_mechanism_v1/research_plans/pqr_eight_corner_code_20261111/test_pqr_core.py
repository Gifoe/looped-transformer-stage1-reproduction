import torch
from torch import nn
from pqr_core import compose,CORNERS,pqr_boundary
r={'P':[0,1],'Q':[2,3],'R':[4]};a=torch.arange(10).reshape(1,5,2).float();b=a+100
for c in CORNERS:
 z=compose(a,b,r,c)
 for bit,name in zip(c,('P','Q','R')):assert torch.equal(z[:,r[name],:],(a if bit=='1' else b)[:,r[name],:])
for bad in ({'P':[],'Q':[0,1,2,3],'R':[4]},{'P':[0,1],'Q':[1,2,3],'R':[4]},{'P':[0,1],'Q':[2],'R':[4]}):
 try:compose(a,b,bad,'112')
 except ValueError:pass
 else:raise AssertionError('invalid region accepted')
class Attn:layer_idx=11
class Layer(nn.Module):
 def __init__(self,i):super().__init__();self.self_attn=type('A',(),{'layer_idx':i})()
 def forward(self,x,**kw):return x+1
class Core(nn.Module):
 def __init__(self):super().__init__();self.layers=nn.ModuleList([Layer(i) for i in range(24)]);self.total_ut_steps=4
class Model(nn.Module):
 def __init__(self):super().__init__();self.model=Core()
 def forward(self,x,labels=None,use_cache=False,exit_at_step=None):
  for u in range(self.model.total_ut_steps):
   for l in self.model.layers:x=l(x,current_ut=u)
  return x
m=Model();x=torch.zeros(1,5,2);k1=x+12;k2=x+36
for c in CORNERS:
 with pqr_boundary(m,k1,k2,r,c):out=m(x,labels=None,use_cache=False,exit_at_step=1)
 assert torch.equal(out,compose(k1,k2,r,c)+12) and m.model.total_ut_steps==4
try:
 with pqr_boundary(m,k1,k2+1,r,'122'):m(x,labels=None,use_cache=False,exit_at_step=1)
except RuntimeError:pass
else:raise AssertionError('K2 reference drift accepted')
assert m.model.total_ut_steps==4 and not m._forward_pre_hooks and not m.model.layers[11]._forward_pre_hooks and not m.model.layers[11]._forward_hooks
print('PQR_COMPOSITION_HOOK_SYNTHETIC_CPU_PASS')
