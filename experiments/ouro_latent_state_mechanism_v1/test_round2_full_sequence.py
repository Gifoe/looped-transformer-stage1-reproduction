"""Small CPU structural checks; not real model parity or GPU acceptance."""
import types,torch
from torch import nn
from round2_full_sequence import compose_state,rescale_direction,full_boundary
class Layer(nn.Module):
 def __init__(self,i):super().__init__();self.self_attn=types.SimpleNamespace(layer_idx=i)
 def forward(self,x,current_ut=None):return x+0.002*x.cumsum(dim=1)
class Core(nn.Module):
 def __init__(self):super().__init__();self.layers=nn.ModuleList([Layer(i) for i in range(24)]);self.total_ut_steps=4
class Toy(nn.Module):
 def __init__(self):super().__init__();self.model=Core()
 def forward(self,x,labels=None,use_cache=None,past_key_values=None):
  for k in range(self.model.total_ut_steps):
   for layer in self.model.layers:x=layer(x,current_ut=k)
  return x
m=Toy();x=torch.arange(24,dtype=torch.float32).reshape(1,6,4)/10;R=[5]
with full_boundary(m,1) as a:y1=m(x,labels=None,use_cache=False)
k1=a[0];assert m.model.total_ut_steps==4
with full_boundary(m,2) as b:y2=m(x,labels=None,use_cache=False)
k2=b[1]
for condition in ('K2C_K2R','K1C_K2R','K2C_K1R','K1C_K1R'):
 with full_boundary(m,2,condition=condition,k1=k1,k2_reference=k2,readout=R):y=m(x,labels=None,use_cache=False)
 if condition=='K2C_K2R':assert torch.equal(y,y2)
 if condition=='K1C_K1R':assert torch.equal(y,y1)
 assert m.model.total_ut_steps==4
with full_boundary(m,1,layer_index=23) as a:m(x,labels=None,use_cache=False)
with full_boundary(m,2,layer_index=23) as b:y2last=m(x,labels=None,use_cache=False)
with full_boundary(m,2,condition='K1C_K2R',k1=a[0],k2_reference=b[1],readout=R,layer_index=23):y=m(x,labels=None,use_cache=False)
assert torch.equal(y[:,R,:],y2last[:,R,:])
try:
 with full_boundary(m,1):m(x,labels=None,use_cache=True)
 raise AssertionError('cache guard failed')
except RuntimeError:pass
assert m.model.total_ut_steps==4 and not m.model.layers[11]._forward_hooks and not m.model.layers[11]._forward_pre_hooks
u=torch.tensor([[[1,2,3],[4,5,6]]],dtype=torch.bfloat16);v=u*2;z,norm=rescale_direction(u,v);assert z.dtype==torch.bfloat16 and torch.equal(z,v) and torch.equal(norm['actual_bf16_norm'],norm['reference_norm'])
print('PASS synthetic 2x2 composition, identity/full-K1 anchors, last-layer C-only negative control, cache guard/cleanup and BF16 norm control; NOT GPU acceptance')
