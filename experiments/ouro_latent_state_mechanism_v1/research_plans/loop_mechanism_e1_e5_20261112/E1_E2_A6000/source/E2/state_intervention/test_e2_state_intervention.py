import numpy as np,torch
from e2_state_intervention import *
torch.manual_seed(17);torch.set_num_threads(1)
class L(torch.nn.Module):
 def __init__(self,i):super().__init__();self.i=i;self.w=torch.nn.Parameter(torch.randn(7,7)*.03,requires_grad=False)
 def forward(self,x,current_ut=0):return (x.float()+torch.tanh(x.float()@self.w+self.i*.001+current_ut*.013)).to(torch.bfloat16)
class M(torch.nn.Module):
 def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([L(i) for i in range(24)])
 def forward(self,x,depth):
  for ut in range(depth):
   for layer in self.layers:x=layer(x,current_ut=ut)
  return x
m=M().eval();x=torch.randn(1,5,7).to(torch.bfloat16);before=[p.detach().clone() for p in m.parameters()];before_sdpa=F.scaled_dot_product_attention;expected={'physical_index':17,'R':3,'expected_id':'a','current_id':'a','expected_shape':tuple(x.shape),'expected_input_ids_sha':'h','input_ids_sha':'h','capture_sdpa':False}
def run(ut,patch=None):
 with intervene(m.layers[17],ut=ut,patch=patch,**expected) as rec:y=m(x,ut+1)
 return y,rec
b1,r1=run(0);b2,r2=run(1);x1=r1['original_R'];x2=r2['original_R'];other=torch.linspace(.1,.7,7).reshape_as(x1).to(torch.bfloat16);c=conditions(x1,x2,other);assert bits(c['rollback'],x1) and bits(c['forward_add'],x2)
for label in ('identity','restore'):
 y,r=run(1,c[label]);assert bits(y,b2) and r['outside_R_unchanged']
y,r=run(1,c['rollback']);h=x.clone()
for u in range(2):
 for i,layer in enumerate(m.layers):
  if u==1 and i==17:h=h.clone();h[...,3,:]=x1
  h=layer(h,current_ut=u)
assert bits(y,h) and not bits(y,b2)
y1,_=run(0,c['forward_add']);h=x.clone()
for i,layer in enumerate(m.layers):
 if i==17:h=h.clone();h[...,3,:]=x2
 h=layer(h,current_ut=0)
assert bits(y1,h)
for key in ('current_id','input_ids_sha','physical_index'):
 bad=dict(expected);bad[key]={'current_id':'wrong','input_ids_sha':'bad','physical_index':16}[key]
 rejected=False
 try:
  with intervene(m.layers[17],ut=1,**bad):pass
 except AssertionError:rejected=True
 assert rejected
rejected=False
try:
 with intervene(m.layers[17],ut=1,**expected):m(x,1)
except AssertionError:rejected=True
assert rejected
assert F.scaled_dot_product_attention is before_sdpa and all(bits(p,q) for p,q in zip(m.parameters(),before))
zero=conditions(x1,x2,other,projection=np.zeros((7,0)));assert bits(zero['rollback'],x2) and torch.linalg.vector_norm(zero['component_fp64'])==0
assert conditions(x1,torch.zeros_like(x2),torch.zeros_like(other))['unrelated_status']=='UNDEFINED_ZERO_OTHER_DELTA'
assert bits(conditions(torch.zeros_like(x1),torch.zeros_like(x2),other)['norm_matched'],torch.zeros_like(x2))
assert conditions(torch.zeros_like(x1),x2,other)['norm_status']=='UNDEFINED_ZERO_ROLLBACK'
rng=np.random.default_rng(4);w=rng.normal(size=(8,5));std=rng.uniform(.2,3,size=7);mu=rng.normal(size=7);basis,s,tol=raw_probe_span(w,std);d=rng.normal(size=7);pd=project(d,basis);assert np.allclose(pd+(d-pd),d) and np.allclose(basis.T@(d-pd),0,atol=1e-12)
z=rng.normal(size=7);diff=((z+d-mu)/std)@w[:-1]+w[-1]-(((z-mu)/std)@w[:-1]+w[-1]);assert np.allclose(diff[1:]-diff[0],((w[:-1,1:]-w[:-1,:1])/std[:,None]).T@d)
print('PASS 24-layer two-loop independent suffix oracle; identity/restore/rollback/add, only-R, rejection, state cleanup, FP64-BF16 cast, zero-control, probe span and projection.')

