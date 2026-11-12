import torch,torch.nn.functional as F
from fp32_observer import observe,bits
torch.manual_seed(5);torch.set_num_threads(1)
class Attention(torch.nn.Module):
 def forward(self,x,current_ut=0):
  q=x.unsqueeze(1);return F.scaled_dot_product_attention(q,q,q,is_causal=True).squeeze(1)
class Layer(torch.nn.Module):
 def __init__(self):super().__init__();self.self_attn=Attention()
 def forward(self,x,current_ut=0,use_cache=False,past_key_value=None):return x+self.self_attn(x,current_ut=current_ut)*.1
class Core(torch.nn.Module):
 def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([Layer() for _ in range(24)])
class M(torch.nn.Module):
 def __init__(self):super().__init__();self.model=Core()
 def forward(self,x,depth):
  for u in range(depth):
   for l in self.model.layers:x=l(x,current_ut=u,use_cache=False)
  return x
m=M().eval();x=torch.randn(1,5,8);base=m(x,2);old=F.scaled_dot_product_attention
with observe(m,1,4) as rec:y=m(x,2)
assert bits(y,base) and rec['q'].dtype==torch.float32
with observe(m,1,4,patch=rec['native_sdpa_R']) as rec2:y=m(x,2)
assert bits(y,base) and bits(rec2['injected_sdpa_R'],rec['native_sdpa_R']) and F.scaled_dot_product_attention is old
assert not m.model.layers[17]._forward_pre_hooks and not m.model.layers[17].self_attn._forward_pre_hooks
print('PASS full float32 24-layer two-loop nohook/observer/identity suffix and hook restore')
