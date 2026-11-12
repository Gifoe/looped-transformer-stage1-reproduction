import torch
from q13_core import observe_branch,make_capture,matched,bits_equal
class Layer(torch.nn.Module):
 def __init__(self):super().__init__();self.self_attn=torch.nn.Identity();self.self_attn.layer_idx=12;self.input_layernorm_2=torch.nn.Identity()
 def forward(self,x,current_ut):
  a=self.input_layernorm_2(x.cumsum(dim=1));return x+a
layer=Layer();x1=torch.tensor([[[1.,2.],[3.,4.],[5.,6.],[7.,8.]]]);x2=x1.clone();x2[:,1:3,:]+=2
P=[0];R=[3];key='a'*64
def baseline(x):
 for u in range(2):out=layer(x,current_ut=u)
 return out
def capture(x,traj,donor=None):
 with observe_branch(layer,R,donor) as r:
  for u in range(2):out=layer(x,current_ut=u)
 return out,make_capture(traj,key,r)
y1,a=capture(x1,'212');y2,b=capture(x2,'222');assert bits_equal(y1,baseline(x1)) and bits_equal(y2,baseline(x2));assert matched(a,b,P,R)
y12,_=capture(x1,'212',b);y21,_=capture(x2,'222',a)
assert bits_equal(y12[:,R,:],x1[:,R,:]+b.branch[:,R,:]);assert bits_equal(y21[:,R,:],x2[:,R,:]+a.branch[:,R,:]);assert bits_equal(y12[:,:3,:],y1[:,:3,:])
yi,_=capture(x1,'212',a);assert bits_equal(yi,y1)
try:
 with observe_branch(layer,R):layer(x1,current_ut=0)
except RuntimeError:pass
else:raise AssertionError('missing loop accepted')
assert not layer._forward_hooks and not layer._forward_pre_hooks and not layer.input_layernorm_2._forward_hooks
try:
 b.branch[0,3,0]+=1;matched(a,b,P,R)
except ValueError:pass
else:raise AssertionError('donor tampering')
print('Q13_BRANCH_TOY_PASS; NOT PROVENANCE OR GPU')
