import torch
from torch import nn
from ab1_boundary_hooks import capture_and_patch,bits
from ab1_boundary_core import corner,q9_cross
class Layer(nn.Module):
 def forward(self,x,*,current_ut,use_cache,past_key_value=None):
  return x+torch.cumsum(x,dim=1)*0.002
class Body(nn.Module):
 def __init__(self):super().__init__();self.layers=nn.ModuleList([Layer() for _ in range(24)]);self.norm=nn.LayerNorm(5);self.total_ut_steps=2
class M(nn.Module):
 def __init__(self):super().__init__();self.model=Body()
 def forward(self,x):
  z=[]
  for i in range(2):
   for layer in self.model.layers:x=layer(x,current_ut=i,use_cache=False,past_key_value=None)
   x=self.model.norm(x);z.append(x.clone())
  return z
model=M().eval();torch.manual_seed(17);E=torch.randn(1,7,5)
with torch.no_grad():
 native=model(E)
 with capture_and_patch(model,7,6,2) as n:observed=model(E)
 assert bits(n['E'],E) and bits(n['B'],native[0]) and bits(n['B_at_L1'],native[0]) and bits(observed[1],native[1])
 with capture_and_patch(model,7,6,2,boundary_input=n['E']) as ea:allE=model(E)
 assert bits(allE[1],native[0])
 H=corner(n['E'],n['B'],2,6,'E','B')
 with capture_and_patch(model,7,6,2,boundary_input=H) as u:U=model(E)
 assert bits(u['K2_applied'],H) and not bits(u['Q9_actual'],n['Q9_actual'])
 with capture_and_patch(model,7,6,2,boundary_input=H,q9_donor=n['Q9_actual'][:,2:6]) as ux:model(E)
 assert bits(ux['Q9_applied'],q9_cross(ux['Q9_actual'],n['Q9_actual'],2,6))
 with capture_and_patch(model,7,6,2,q9_donor=u['Q9_actual'][:,2:6]) as nx:model(E)
 assert bits(nx['Q9_applied'],q9_cross(nx['Q9_actual'],u['Q9_actual'],2,6))
print('PASS independent 24-layer nn.Module full-boundary hooks, two-cycle B bridge, E-all replay, cross Q9 and cleanup; synthetic CPU only')
