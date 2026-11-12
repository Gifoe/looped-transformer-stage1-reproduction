import torch
from types import SimpleNamespace
from pair_runner import run_matched_pair,forward_capture
from q13_core import bits_equal
from contextlib import contextmanager
torch.set_num_threads(1);torch.manual_seed(9)
class Norm(torch.nn.Module):
 def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor([.8,1.,1.2,1.4]))
 def forward(self,x):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*self.w
class Attention(torch.nn.Module):
 def __init__(self,i):super().__init__();self.layer_idx=i;self.q=torch.nn.Linear(4,4,bias=False);self.k=torch.nn.Linear(4,4,bias=False);self.v=torch.nn.Linear(4,4,bias=False)
 def forward(self,x):
  v=self.q(x)@self.k(x).transpose(-1,-2);mask=torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool),diagonal=1);return v.masked_fill(mask,float('-inf')).softmax(-1)@self.v(x)
class Layer(torch.nn.Module):
 def __init__(self,i):
  super().__init__();self.self_attn=Attention(i);self.input_layernorm=Norm();self.input_layernorm_2=Norm();self.post_attention_layernorm=Norm();self.post_attention_layernorm_2=Norm();self.a=torch.nn.Linear(4,8);self.b=torch.nn.Linear(4,8);self.c=torch.nn.Linear(8,4)
 def forward(self,x,current_ut):
  x=x+self.input_layernorm_2(self.self_attn(self.input_layernorm(x)));h=self.post_attention_layernorm(x);return x+self.post_attention_layernorm_2(self.c(torch.nn.functional.silu(self.a(h))*self.b(h)))
class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.model=SimpleNamespace(layers=torch.nn.ModuleList([Layer(i) for i in range(24)]));self.outer=Norm();self.emb=torch.nn.Embedding(10,4);self.head=torch.nn.Linear(4,10)
 def forward(self,ids,labels=None,use_cache=False,past_key_values=None):
  x=self.emb(ids)
  for u in (0,1):
   for l in self.model.layers:x=l(x,current_ut=u)
   x=self.outer(x)
  return self.head(x)
m=Model();ids=torch.tensor([[1,2,3,4,5]]);P=[0];Q=[1,2,3];R=[4]
@contextmanager
def cut(q):
 def patch(m,a,k,out):
  if k['current_ut']==1:
   v=out.clone();v[:,Q,:]+=q;return v
 h=m.model.layers[11].register_forward_hook(patch,with_kwargs=True)
 try:yield
 finally:h.remove()
def run(q):
 def f():
  with cut(q):return m(ids,labels=None,use_cache=False)
 return f
def getcut(q):
 got=[]
 def cap(m,a,k,out):
  if k['current_ut']==1:got.append(out.detach().clone())
 with cut(q):
  h=m.model.layers[11].register_forward_hook(cap,with_kwargs=True)
  try:m(ids,labels=None,use_cache=False)
  finally:h.remove()
 return got[0]
a=getcut(0);b=getcut(.1);res=run_matched_pair(m,ids,P,R,run(0),run(.1),a,b,'212','222',lambda:None)
assert bits_equal(res['B1'],res['I1']) and bits_equal(res['B2'],res['I2'])
# Independent receiver suffix oracle uses the observed actual receiver layer input.
def oracle(receiver,donor):
 l=m.model.layers[12];x=receiver.layer_input;delta=l.input_layernorm_2(l.self_attn(l.input_layernorm(x))).clone();delta[:,R,:]=donor.branch[:,R,:];x=x+delta;h=l.post_attention_layernorm(x);x=x+l.post_attention_layernorm_2(l.c(torch.nn.functional.silu(l.a(h))*l.b(h)))
 for layer in m.model.layers[13:]:x=layer(x,current_ut=1)
 return m.head(m.outer(x))
assert bits_equal(oracle(res['capture1'],res['capture2']),res['U1']) and bits_equal(oracle(res['capture2'],res['capture1']),res['U2'])
try:forward_capture(m,ids,'212',a,P,R,run(.1),lambda:None)
except ValueError:pass
else:raise AssertionError('wrong cut accepted')
res['capture2'].layer_input[0,0,0]+=1
try:forward_capture(m,ids,'212',a,P,R,run(0),lambda:None,res['capture2'],res['capture1'])
except ValueError:pass
else:raise AssertionError('mutated capture accepted')
assert not m._forward_pre_hooks and not m.model.layers[11]._forward_hooks and not m.model.layers[12]._forward_hooks
print('LOCAL_PAIR_CPU_PASS; no production provenance / no GPU')
