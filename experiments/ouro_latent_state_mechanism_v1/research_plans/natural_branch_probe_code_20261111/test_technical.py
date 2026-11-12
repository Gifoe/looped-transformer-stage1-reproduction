import sys,torch
from types import SimpleNamespace
sys.path.insert(0,r'C:\Users\gifoe\.codex\round4_q13_local_staging_20261111')
from test_controlled import Model
from test_pair_runner import Layer
from technical import verify,score
class L(Layer):
 def forward(self,x,current_ut,**kw):return super().forward(x,current_ut=current_ut)
class M(Model):
 def __init__(self):
  super().__init__();self.model.layers=torch.nn.ModuleList([L(i) for i in range(24)])
 def forward(self,ids,labels=None,use_cache=False,exit_at_step=None):
  x=self.emb(ids)
  for u in range(self.model.total_ut_steps):
   for l in self.model.layers:x=l(x,current_ut=u,past_key_value=None,use_cache=False)
   x=self.model.norm(x)
  return SimpleNamespace(logits=self.lm_head(x))
m=M().eval();ids=torch.tensor([[1,2,3,4,5]]);R=4;targets=[1,2];states={};norms=[];handles=[];events=[]
for i,l in enumerate(m.model.layers):
 def hook(mod,args,kw,out,i=i):states[(kw['current_ut'],i)]=out[:,R,:].detach().clone()
 handles.append(l.register_forward_hook(hook,with_kwargs=True))
def norm(mod,args,out):norms.append(out[:,R,:].detach().clone())
handles.append(m.model.norm.register_forward_hook(norm));m.model.total_ut_steps=2
with torch.inference_mode():m(ids,labels=None,use_cache=False,exit_at_step=1)
for h in handles:h.remove()
with torch.inference_mode():logits=[m.lm_head(x.unsqueeze(0))[0] for x in norms]
scores=[score(x,targets) for x in logits];m.model.total_ut_steps=4
obs,z=verify(m,ids,R,targets,states,norms,logits,scores);assert len(obs['states'])==48 and len(z)==2 and m.model.total_ut_steps==4
bad=dict(states);bad[(1,12)]=bad[(1,12)].clone()+1
try:verify(m,ids,R,targets,bad,norms,logits,scores)
except ValueError as e:assert 'state bridge' in str(e)
else:raise AssertionError('corrupt old state')
assert m.model.total_ut_steps==4 and all(not l._forward_hooks and not l._forward_pre_hooks and not l.input_layernorm_2._forward_hooks and not l.post_attention_layernorm._forward_pre_hooks for l in m.model.layers)
print('NATURAL_BRANCH_TECHNICAL_CPU_PASS; NOT GPU')
