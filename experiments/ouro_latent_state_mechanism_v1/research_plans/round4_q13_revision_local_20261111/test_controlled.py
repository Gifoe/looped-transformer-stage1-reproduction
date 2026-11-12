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
m=Model();ids=torch.tensor([[1,2,3,4,5]]);regions={'P':[0],'Q':[1,2,3],'R':[4]}
with tempfile.TemporaryDirectory() as td:
 p=Path(td)/'source';p.write_text('fixed');expected={'model':state_digest(m),'ids':tensor_sha(ids),'files':{str(p.resolve()):sha_file(p)}};guard=FrozenSources(m,ids,[p],expected)
 natural=capture_natural(m,ids,regions['R'],guard);k1,k2=natural[0][0],natural[1][0];prior={}
 for corner in ('211','221','212','222'):
  with boundary(m,2,k1,k2,regions,corner):prior[corner]=score_forward(m,ids,regions['R'])
 n,v=run_question(m,ids,regions,guard,prior)
 wrong=dict(prior);wrong['211']=prior['211'].clone()+1
 try:run_question(m,ids,regions,guard,wrong)
 except ValueError as e:assert 'prior corner' in str(e)
 else:raise AssertionError('wrong prior accepted')
 assert set(v)=={'1','2'} and m.model.total_ut_steps==4
 p.write_text('changed')
 try:guard()
 except ValueError:pass
 else:raise AssertionError('file drift')
 p.write_text('fixed');guard();ids[0,0]=9
 try:guard()
 except ValueError:pass
 else:raise AssertionError('input drift')
 ids[0,0]=1;guard()
 with torch.no_grad():m.emb.weight[0,0]+=1
 try:guard()
 except ValueError:pass
 else:raise AssertionError('model drift')
assert not m._forward_pre_hooks and not m.model.layers[11]._forward_hooks and not m.model.layers[12]._forward_hooks
print('LOCAL_CLOSED_BOUNDARY_CPU_PASS; production sources / GPU unverified')
