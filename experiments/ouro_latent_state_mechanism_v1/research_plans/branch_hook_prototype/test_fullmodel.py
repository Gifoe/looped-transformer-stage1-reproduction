"""Synthetic full-sequence CPU-only test; no claim of production provenance."""
import torch
from test_contract import Layer,Norm,branch_hook,increments
from prototype import branch_hook as raw_hook
torch.set_num_threads(1);torch.manual_seed(412)
layers=torch.nn.ModuleList([Layer(i,4) for i in range(24)]);outer=Norm(4); x=torch.randn(1,5,4);captures={};events=[];handles=[]
for idx,l in enumerate(layers):
 def pre(m,args,kwargs,idx=idx):
  ut=kwargs['current_ut']; events.append((ut,idx));
  if idx==11:captures[(ut,'x')]=args[0].detach().clone()
 handles.append(l.register_forward_pre_hook(pre,with_kwargs=True))
 if idx==11:
  for name,module in (('a',l.input_layernorm_2),('b',l.post_attention_layernorm_2)):
   def capture(m,args,out,name=name):
    ut=events[-1][0];captures[(ut,name)]=out.detach().clone()
   handles.append(module.register_forward_hook(capture))
def run(hybrid=False,depth=2):
 h=x.clone()
 for ut in range(depth):
  if ut==1 and hybrid: h=h.clone();h[:,2,:]=x[:,2,:]
  for l in layers:h=l(h,current_ut=ut)
  h=outer(h)
 return h
natural=run();assert events==[(u,i) for u in (0,1) for i in range(24)]
for u in (0,1):
 a,b=increments(layers[11],captures[(u,'x')]);assert torch.equal(a,captures[(u,'a')]) and torch.equal(b,captures[(u,'b')])
x1=captures[(0,'x')].clone();a1=captures[(0,'a')].clone();b1=captures[(0,'b')].clone();assert torch.equal(layers[11](x1,current_ut=1),x1+a1+b1)
events.clear();final_k1=run(depth=1);events.clear();hybrid_base=run(True); assert not torch.equal(hybrid_base,natural);hybrid_x=captures[(1,'x')].clone();hybrid_a=captures[(1,'a')].clone();events.clear()
with branch_hook(layers[11],11,1,'attention',[2],hybrid_a) as state:hybrid_identity=run(True)
assert torch.equal(hybrid_identity,hybrid_base) and state['visits']==1 and events==[(u,i) for u in (0,1) for i in range(24)]
events.clear()
with branch_hook(layers[11],11,1,'attention',[2],a1) as state:hybrid_patch=run(True)
assert state['visits']==1 and events==[(u,i) for u in (0,1) for i in range(24)] and not torch.equal(hybrid_patch,hybrid_base)
manual_a=hybrid_a.clone();manual_a[:,2,:]=a1[:,2,:];manual_y=hybrid_x+manual_a;manual=manual_y+layers[11].post_attention_layernorm_2(layers[11].mlp(layers[11].post_attention_layernorm(manual_y)))
for ll in layers[12:]:manual=ll(manual,current_ut=1)
manual=outer(manual);assert torch.equal(manual,hybrid_patch)
def anchor(m,args,kwargs):
 if kwargs['current_ut']==1:return (x1, *args[1:]),kwargs
anchor_handle=layers[11].register_forward_pre_hook(anchor,with_kwargs=True)
try: anchored=run()
finally:anchor_handle.remove()
assert torch.equal(anchored,final_k1)
for h in handles:h.remove()
for invalid in (True,1.0):
 try:
  with raw_hook(layers[11],11,invalid,'attention',[2]):pass
 except ValueError:pass
 else:raise AssertionError('bad expected_ut')
 try:
  with raw_hook(layers[11],11,1,'attention',[2]):layers[11](x,current_ut=invalid)
 except ValueError:pass
 else:raise AssertionError('bad current_ut')
try:
 with branch_hook(layers[11],11,1,'attention',[2],torch.full_like(x,float('nan'))):layers[11](x,current_ut=1)
except ValueError:pass
else:raise AssertionError('nan donor')
try:
 with branch_hook(layers[11],11,1,'attention',[2]):layers[11](torch.full_like(x,float('nan')),current_ut=1)
except ValueError:pass
else:raise AssertionError('nan capture')
assert not layers[11]._forward_hooks and not layers[11]._forward_pre_hooks
print('FULLMODEL_CONTRACT_PASS synthetic CPU 24layers 2loops outernorm loop1 donor hybrid receiver')
