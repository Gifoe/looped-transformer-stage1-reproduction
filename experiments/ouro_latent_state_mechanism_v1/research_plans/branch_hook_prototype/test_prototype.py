import torch
from prototype import branch_hook as _branch_hook
import hashlib
def branch_hook(layer,idx,ut,branch,positions,donor=None):
 meta=None if donor is None else {"physical_idx":idx,"branch":branch,"donor_ut":0,"input_sha256":"0"*64,"donor_sha256":hashlib.sha256(donor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}
 return _branch_hook(layer,idx,ut,branch,positions,donor,meta)
from torch import nn
class Attention(nn.Module):
 def __init__(self): super().__init__();self.layer_idx=11
class Layer(nn.Module):
 def __init__(self): super().__init__();self.self_attn=Attention();self.input_layernorm_2=nn.Identity();self.post_attention_layernorm_2=nn.Identity()
 def forward(self,x,**kwargs):
  a=self.input_layernorm_2(x+x.mean(dim=1,keepdim=True)); x=x+a
  m=self.post_attention_layernorm_2(2*x); return x+m
l=Layer();x=torch.tensor([[[1.,2.],[3.,4.],[5.,6.]]]); baseline=l(x,current_ut=1)
for branch in ('attention','mlp'):
 with branch_hook(l,11,1,branch,[2]) as s: initial=l(x,current_ut=1)
 assert torch.equal(initial,baseline) and s['visits']==1
 donor=s['observed'].clone()
 with branch_hook(l,11,1,branch,[2],donor) as s: identity=l(x,current_ut=1)
 assert torch.equal(identity,baseline)
 donor[:,2,:]+=2
 with branch_hook(l,11,1,branch,[2],donor) as s: patched=l(x,current_ut=1)
 assert not torch.equal(patched[:,2,:],baseline[:,2,:]) and torch.equal(patched[:,:2,:],baseline[:,:2,:])
 with branch_hook(l,11,1,branch,[2],donor) as s:
  ignored=l(x,current_ut=0); assert torch.equal(ignored,baseline) and s['visits']==0
  l(x,current_ut=1)
with branch_hook(l,11,1,'mlp',[0,1],torch.zeros_like(x)) as s: cpatched=l(x,current_ut=1)
assert torch.equal(cpatched[:,2,:],baseline[:,2,:])
try:
 with branch_hook(l,11,1,'mlp',[2],torch.zeros(1)):
  l(x,current_ut=1)
except ValueError: pass
else: raise AssertionError('bad donor accepted')
assert not l._forward_pre_hooks and not l._forward_hooks and not l.input_layernorm_2._forward_hooks and not l.post_attention_layernorm_2._forward_hooks
try:
 with branch_hook(l,10,1,'attention',[2]): pass
except ValueError: pass
else: raise AssertionError('bad layer accepted')
print('SYNTHETIC_CPU_ONLY_PASS')

try:
 with branch_hook(l,11,1,"attention",[2]) as t: l(x if "receiver" not in globals() else receiver,current_ut=0)
except RuntimeError:pass
else:raise AssertionError("missing target accepted")
try:
 with branch_hook(l,11,1,"attention",[],torch.full_like(x,float("nan"))):l(x,current_ut=1)
except ValueError:pass
else:raise AssertionError("invalid donor accepted")
