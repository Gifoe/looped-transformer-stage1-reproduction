"""Synthetic CPU contract only; not a test on the production model."""
import torch
from torch import nn
from prototype import branch_hook as _branch_hook
import hashlib
def branch_hook(layer,idx,ut,branch,positions,donor=None):
 meta=None if donor is None else {"physical_idx":idx,"branch":branch,"donor_ut":0,"input_sha256":"0"*64,"donor_sha256":hashlib.sha256(donor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()}
 return _branch_hook(layer,idx,ut,branch,positions,donor,meta)
torch.set_num_threads(1);torch.manual_seed(18)
class Norm(nn.Module):
 def __init__(self,d):
  super().__init__();self.weight=nn.Parameter(torch.linspace(.6,1.4,d))
 def forward(self,x): return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*self.weight
class Attention(nn.Module):
 def __init__(self,i,d):
  super().__init__();self.layer_idx=i;self.q=nn.Linear(d,d,bias=False);self.k=nn.Linear(d,d,bias=False);self.v=nn.Linear(d,d,bias=False)
 def forward(self,x):
  t=x.shape[1]; scores=self.q(x)@self.k(x).transpose(-1,-2)/(x.shape[-1]**.5);mask=torch.triu(torch.ones(t,t,dtype=torch.bool),diagonal=1)
  return torch.softmax(scores.masked_fill(mask,float('-inf')),dim=-1)@self.v(x)
class SwiGLU(nn.Module):
 def __init__(self,d):super().__init__();self.a=nn.Linear(d,d*2,bias=False);self.b=nn.Linear(d,d*2,bias=False);self.c=nn.Linear(d*2,d,bias=False)
 def forward(self,x):return self.c(torch.nn.functional.silu(self.a(x))*self.b(x))
class Layer(nn.Module):
 def __init__(self,i,d):
  super().__init__();self.self_attn=Attention(i,d);self.mlp=SwiGLU(d);self.input_layernorm=Norm(d);self.input_layernorm_2=Norm(d);self.post_attention_layernorm=Norm(d);self.post_attention_layernorm_2=Norm(d)
 def forward(self,x,**kwargs):
  x=x+self.input_layernorm_2(self.self_attn(self.input_layernorm(x)))
  return x+self.post_attention_layernorm_2(self.mlp(self.post_attention_layernorm(x)))
def increments(l,x):
 a=l.input_layernorm_2(l.self_attn(l.input_layernorm(x)));y=x+a
 b=l.post_attention_layernorm_2(l.mlp(l.post_attention_layernorm(y)))
 return a,b
layers=nn.ModuleList([Layer(i,4) for i in range(24)]); assert len({id(l.self_attn.q.weight) for l in layers})==24
x=torch.randn(1,5,4); k1=x.clone();
for l in layers:k1=l(k1,current_ut=0)
k2=k1.clone()
for l in layers:k2=l(k2,current_ut=1)
l=layers[11];receiver=k2.clone();receiver[:,2,:]=k1[:,2,:]; assert not torch.equal(receiver,k1) and not torch.equal(receiver,k2);a,b=increments(l,receiver);da,db=increments(l,k1); natural=l(receiver,current_ut=1);assert torch.equal(natural,receiver+a+b)
for r in (0,2,4):
 with branch_hook(l,11,1,'attention',[r]) as cap: l(receiver,current_ut=1)
 with branch_hook(l,11,1,'attention',[r],cap['observed']) as cap2: assert torch.equal(l(receiver,current_ut=1),natural)
 assert torch.equal(cap['observed'],cap2['observed'])
 with branch_hook(l,11,1,'attention',[r],da): got=l(receiver,current_ut=1)
 mixed=a.clone();mixed[:,r,:]=da[:,r,:];y=receiver+mixed;expected=y+l.post_attention_layernorm_2(l.mlp(l.post_attention_layernorm(y)));assert torch.equal(got,expected)
 with branch_hook(l,11,1,'mlp',[r],db): got=l(receiver,current_ut=1)
 mixb=b.clone();mixb[:,r,:]=db[:,r,:];assert torch.equal(got,receiver+a+mixb)
 with branch_hook(l,11,1,'attention',[r],da):
  with branch_hook(l,11,1,'mlp',[r],db):got=l(receiver,current_ut=1)
 assert torch.equal(got,receiver+mixed+mixb)
 assert not torch.equal(got,l(k1,current_ut=0))
 with branch_hook(l,11,1,'attention',[r],da) as s:
  assert torch.equal(l(receiver,current_ut=0),natural) and s['visits']==0
  l(receiver,current_ut=1)
 if r<4:
  future=receiver.clone();future[:,r+1:,:]+=7;assert torch.equal(l(future,current_ut=1)[:,:r+1,:],natural[:,:r+1,:])
with branch_hook(l,11,1,'attention',[2],da) as s:
 untouched=layers[10](receiver,current_ut=1);assert s['visits']==0 and torch.equal(untouched,layers[10](receiver,current_ut=1));l(receiver,current_ut=1)
with branch_hook(l,11,1,'mlp',[0,1],db): c=l(receiver,current_ut=1)
assert torch.equal(c[:,2:,:],natural[:,2:,:])
last=layers[23];last_base=last(receiver,current_ut=1);last_donor=increments(last,k1)[1]
with branch_hook(last,23,1,'mlp',[0,1],last_donor):last_changed=last(receiver,current_ut=1)
assert torch.equal(last_changed[:,2:,:],last_base[:,2:,:])
wrong_raw=l.self_attn(l.input_layernorm(receiver));wrong_twice=l.input_layernorm_2(a);assert not torch.allclose(wrong_raw,a) and not torch.allclose(wrong_twice,a)
assert not torch.allclose(l.post_attention_layernorm(receiver+a),b)
try:
 with branch_hook(l,11,1,'attention',[4],torch.zeros(1)):l(receiver,current_ut=1)
except ValueError:pass
else:raise AssertionError('bad donor')
assert all(not x._forward_hooks and not x._forward_pre_hooks for x in (l,l.input_layernorm_2,l.post_attention_layernorm_2))

try:
 with branch_hook(l,11,1,"attention",[2]) as t: l(x if "receiver" not in globals() else receiver,current_ut=0)
except RuntimeError:pass
else:raise AssertionError("missing target accepted")
try:
 with branch_hook(l,11,1,"attention",[],torch.full_like(x,float("nan"))):l(x,current_ut=1)
except ValueError:pass
else:raise AssertionError("invalid donor accepted")

print("SINGLE_LAYER_CONTRACT_PASS after all checks")
