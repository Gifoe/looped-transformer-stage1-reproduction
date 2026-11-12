import torch
from e2_double_hook import double_intervene,bits
torch.set_num_threads(1);torch.manual_seed(23)
class Toy(torch.nn.Module):
 def __init__(self,i):
  super().__init__();self.i=i;self.w=torch.nn.Parameter(torch.randn(7,7)*.04,requires_grad=False);self.input_layernorm_2=torch.nn.Identity();self.post_attention_layernorm=torch.nn.Identity();self.post_attention_layernorm_2=torch.nn.Identity()
 def attention(self,x,ut):return torch.tanh((x.float().cumsum(dim=1)@self.w)*.2+ut*.1+self.i*.001).to(torch.bfloat16)
 def forward(self,x,current_ut=0):
  a=self.input_layernorm_2(self.attention(x,current_ut));y=(x+a).to(torch.bfloat16);z=self.post_attention_layernorm(y);b=self.post_attention_layernorm_2((z.float()*.011).to(torch.bfloat16));return (z+b).to(torch.bfloat16)
class M(torch.nn.Module):
 def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([Toy(i) for i in range(24)])
 def forward(self,x,depth=2):
  for u in range(depth):
   for l in self.layers:x=l(x,current_ut=u)
  return x
m=M().eval();x=torch.randn(1,6,7).to(torch.bfloat16);R=4;h=x.clone();full={};a={}
for u in range(2):
 for i,l in enumerate(m.layers):
  if i==17:full[u]=h.clone();a[u]=l.attention(h,u)[...,R,:].clone()
  h=l(h,current_ut=u)
base=h.clone();x1=full[0][...,R,:].clone();x2=full[1][...,R,:].clone();a2=a[1]
# a-prime from actual x1 receiver by explicit computation, not borrowed from a1.
y=full[1].clone();y[...,R,:]=x1;aprime=m.layers[17].attention(y,1)[...,R,:].clone()
def oracle(xpatch,apatch):
 h=x.clone()
 for u in range(2):
  for i,l in enumerate(m.layers):
   if u==1 and i==17:
    h=h.clone();h[...,R,:]=xpatch;av=l.attention(h,u);av[...,R,:]=apatch;yy=(h+av).to(torch.bfloat16);h=(yy+(yy.float()*.011).to(torch.bfloat16)).to(torch.bfloat16)
   else:h=l(h,current_ut=u)
 return h
params=[v.clone() for v in m.parameters()];hooks=[]
for xpatch,apatch in ((x2,a2),(x1,aprime),(x1,a2),(x2,aprime)):
 with double_intervene(m.layers[17],physical_index=17,ut=1,R=R,expected_full=full[1],expected_id='a',current_id='a',expected_input_ids_sha='h',input_ids_sha='h',x_patch=xpatch,a_patch=apatch) as rec:out=m(x)
 assert bits(out,oracle(xpatch,apatch)) and rec['x_outside_R_unchanged'] and rec['a_outside_R_unchanged']
 if bits(xpatch,x2) and bits(apatch,a2):assert bits(out,base)
if True:
 rejected=False
 try:
  with double_intervene(m.layers[17],physical_index=17,ut=1,R=R,expected_full=full[0],expected_id='a',current_id='a',expected_input_ids_sha='h',input_ids_sha='h',x_patch=x2,a_patch=a2):m(x)
 except AssertionError:rejected=True
 assert rejected and all(bits(p,q) for p,q in zip(params,m.parameters()))
assert not m.layers[17]._forward_pre_hooks and not m.layers[17]._forward_hooks and not m.layers[17].input_layernorm_2._forward_hooks
print('PASS four double-hook corners versus independent 24-layer two-loop oracle, only-R, input rejection, parameter/hook restoration')
