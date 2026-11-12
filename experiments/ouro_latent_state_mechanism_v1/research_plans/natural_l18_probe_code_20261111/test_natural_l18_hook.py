import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path.home()/'.codex/natural_branch_full_review_v4_20261111'))
from natural_l18_hook import intervene,matched_norm
from state_capture import bits
torch.set_num_threads(1);torch.manual_seed(17)
class Attn(torch.nn.Module):
 def __init__(self,i):super().__init__();self.layer_idx=i;self.proj=torch.nn.Linear(8,8,bias=False)
 def forward(self,x):return self.proj(x.cumsum(1)/torch.arange(1,x.shape[1]+1,device=x.device)[None,:,None])
class Layer(torch.nn.Module):
 def __init__(self,i):
  super().__init__();self.self_attn=Attn(i);self.input_layernorm_2=torch.nn.LayerNorm(8);self.mlp=torch.nn.Sequential(torch.nn.Linear(8,16),torch.nn.SiLU(),torch.nn.Linear(16,8));self.post_attention_layernorm_2=torch.nn.LayerNorm(8)
 def forward(self,x,current_ut=None,use_cache=False,past_key_value=None):
  a=self.input_layernorm_2(self.self_attn(x));y=x+a;b=self.post_attention_layernorm_2(self.mlp(y));return y+b
class Core(torch.nn.Module):
 def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([Layer(i) for i in range(24)]);self.norm=torch.nn.LayerNorm(8)
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.model=Core()
 def forward(self,x,depth):
  for u in range(depth):
   for layer in self.model.layers:x=layer(x,current_ut=u,use_cache=False,past_key_value=None)
   x=self.model.norm(x)
  return x
@torch.inference_mode()
def manual(model,x,depth,R,replacement):
 for u in range(depth):
  for i,l in enumerate(model.model.layers):
   a=l.input_layernorm_2(l.self_attn(x))
   if u==depth-1 and i==17:a=a.clone();a[:,R,:]=replacement
   y=x+a;x=y+l.post_attention_layernorm_2(l.mlp(y))
  x=model.model.norm(x)
 return x
@torch.inference_mode()
def main():
 m=Model().to(torch.bfloat16).eval();x=torch.randn(1,5,8).bfloat16();R=3
 base1=m(x,1);base2=m(x,2)
 with intervene(m,1,R) as c1:z1=m(x,1)
 with intervene(m,2,R) as c2:z2=m(x,2)
 assert bits(base1,z1) and bits(base2,z2)
 for dep,own,other,base in [(1,c1,c2,base1),(2,c2,c1,base2)]:
  with intervene(m,dep,R,own['receiver_x'],own['receiver_a'],own['receiver_a']):z=m(x,dep)
  assert bits(z,base)
  with intervene(m,dep,R,own['receiver_x'],own['receiver_a'],other['receiver_a']):z=m(x,dep)
  assert bits(z,manual(m,x,dep,R,other['receiver_a']))
  try:
   with intervene(m,dep,R,own['receiver_x']+1,own['receiver_a'],other['receiver_a']):m(x,dep)
  except ValueError:pass
  else:raise AssertionError('wrong receiver accepted')
  assert not m.model.layers[17]._forward_hooks and not m.model.layers[17]._forward_pre_hooks and not m.model.layers[17].input_layernorm_2._forward_hooks
 for wrong_depth,wrong_R,wrong_x,wrong_a in [(2,R,c1['receiver_x'],c1['receiver_a']),(1,2,c1['receiver_x'],c1['receiver_a'])]:
  try:
   with intervene(m,wrong_depth,wrong_R,wrong_x,wrong_a,wrong_a):m(x,wrong_depth)
  except ValueError:pass
  else:raise AssertionError('wrong loop / R accepted')
 assert not m.model.layers[17]._forward_hooks and not m.model.layers[17]._forward_pre_hooks and not m.model.layers[17].input_layernorm_2._forward_hooks
 ctrl,rec=matched_norm(c1['receiver_a'],c2['receiver_a']);assert ctrl.dtype==torch.bfloat16 and rec['status']=='OK'
 ctrl,rec=matched_norm(torch.zeros_like(c1['receiver_a']),c2['receiver_a']);assert ctrl is None and rec['status']=='UNDEFINED_ZERO_DONOR'
 m.model.layers[17].self_attn.layer_idx=16
 try:
  with intervene(m,1,R):pass
 except ValueError:pass
 else:raise AssertionError('wrong layer accepted')
 print('CPU_L18_FULL_SUFFIX_AND_CLEANUP_PASS')
if __name__=='__main__':main()
