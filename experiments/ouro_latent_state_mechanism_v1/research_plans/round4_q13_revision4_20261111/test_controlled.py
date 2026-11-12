import tempfile,torch
from pathlib import Path
from types import SimpleNamespace
from test_pair_runner import Layer,Norm
from controlled import run_question,capture_natural,boundary,score_forward
from source_guard import FrozenSources,state_digest,tensor_sha,sha_file
torch.set_num_threads(1)
class Core(torch.nn.Module):
 def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([Layer(i) for i in range(24)]);self.norm=Norm();self.total_ut_steps=4
class Head(torch.nn.Module):
 def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.randn(10,4))
 def forward(self,x):return (x.unsqueeze(-2)*self.weight).sum(-1)
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.model=Core();self.emb=torch.nn.Embedding(10,4);self.lm_head=Head()
 def forward(self,ids,labels=None,use_cache=False,exit_at_step=None):
  x=self.emb(ids)
  for u in range(self.model.total_ut_steps):
   for l in self.model.layers:x=l(x,current_ut=u)
   x=self.model.norm(x)
  return SimpleNamespace(logits=self.lm_head(x))
m=Model().eval();ids=torch.tensor([[1,2,3,4,5]]);regions={'P':[0],'Q':[1,2,3],'R':[4]}
with tempfile.TemporaryDirectory() as td:
 p=Path(td)/'source';p.write_text('fixed');expected={'model':state_digest(m),'ids':tensor_sha(ids),'files':{str(p.resolve()):sha_file(p)}};guard=FrozenSources(m,ids,[p],expected)
 # Independent manual K1/K2 and hybrid suffix; no controlled boundary hooks.
 def manual(depth,k1=None,k2=None,corner=None):
  x=m.emb(ids);cut=None
  for u in range(depth):
   for i,layer in enumerate(m.model.layers):
    x=layer(x,current_ut=u)
    if i==11:
     cut=x.detach().clone()
     if corner is not None and u==1:
      x=k2.clone()
      for bit,name in zip(corner,('P','Q','R')):
       if bit=='1':x[:,regions[name],:]=k1[:,regions[name],:]
   x=m.model.norm(x)
  return cut,m.lm_head(x)[0,regions['R'],:]
 with torch.inference_mode():
  k1,z1=manual(1);k2,z2=manual(2);prior={corner:manual(2,k1,k2,corner)[1] for corner in ('211','221','212','222')}
 n,v=run_question(m,ids,regions,guard,prior)
 wrong=dict(prior);wrong['211']=prior['211'].clone()+1
 try:run_question(m,ids,regions,guard,wrong)
 except ValueError as e:assert 'prior corner' in str(e)
 else:raise AssertionError('wrong prior accepted')
 assert set(v)=={'1','2'} and m.model.total_ut_steps==4
 assert all(v[r][k].grad_fn is None and not v[r][k].requires_grad for r in ('1','2') for k in ('B1','B2','I1','I2','U1','U2'))
 m.train()
 try:run_question(m,ids,regions,guard,prior)
 except ValueError as e:assert 'eval' in str(e)
 else:raise AssertionError('train mode accepted')
 m.eval()
 p.write_text('changed')
 try:guard()
 except ValueError:pass
 else:raise AssertionError('file drift')
 p.write_text('fixed');guard=FrozenSources(m,ids,[p],expected);guard();ids[0,0]=9
 try:guard()
 except ValueError:pass
 else:raise AssertionError('input drift')
 ids[0,0]=1;guard()
 with torch.no_grad():m.emb.weight[0,0]+=1
 try:guard()
 except ValueError:pass
 else:raise AssertionError('model drift')
 try:guard.finish()
 except ValueError:pass
 else:raise AssertionError('finish failed to detect model drift')
assert not m._forward_pre_hooks and not m.model.layers[11]._forward_hooks and not m.model.layers[12]._forward_hooks
print('LOCAL_CLOSED_BOUNDARY_CPU_PASS; production sources / GPU unverified')
