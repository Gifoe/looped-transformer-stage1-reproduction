"""CPU mock for matched hybrid Q trajectories and complete suffix oracle."""
import sys,torch
sys.path.insert(0,'/home/luansiyu/loop-latent/research_plans/branch_hook_prototype')
from test_contract import Layer,Norm
from q13_core import observe_branch,make_capture,matched,bits_equal
torch.set_num_threads(1);torch.manual_seed(417)
layers=torch.nn.ModuleList([Layer(i,4) for i in range(24)]);outer=Norm(4);x=torch.randn(1,5,4);P=[0];Q=[1,2,3];R=[4];key='a'*64
# Two hybrid recipients differ only by controlled Q at the output of physical 11, ut1.
def run(q,donor=None,capture=True):
 def before(m,args,kw):
  if kw['current_ut']==1:
   out=args[0].clone();out[:,Q,:]+=q;return (out,*args[1:]),kw
 # mutate layer-12 output, not next layer input; a post-hook reproduces cutpoint semantics.
 def cut(m,args,kw,out):
  if kw['current_ut']==1:
   z=out.clone();z[:,Q,:]+=q;return z
 h=layers[11].register_forward_hook(cut,with_kwargs=True)
 try:
  if capture:
   with observe_branch(layers[12],R,donor) as record:
    v=x.clone()
    for u in range(2):
     for l in layers:v=l(v,current_ut=u)
     v=outer(v)
   return v,make_capture('212' if q==0 else '222',key,record)
  v=x.clone()
  for u in range(2):
   for l in layers:v=l(v,current_ut=u)
   v=outer(v)
  return v,None
 finally:h.remove()
b1,a=run(0);b2,b=run(.1);assert matched(a,b,P,R);native1,_=run(0,capture=False);assert bits_equal(native1,b1)
u1,_=run(0,b);u2,_=run(.1,a);self1,_=run(0,a);assert bits_equal(self1,b1)
# Independent unhooked full native suffix, recomputing the receiver MLP.
def oracle(receiver,donor):
 l=layers[12];delta=l.input_layernorm_2(l.self_attn(l.input_layernorm(receiver.layer_input)));delta=delta.clone();delta[:,R,:]=donor.branch[:,R,:];z=receiver.layer_input+delta;z=z+l.post_attention_layernorm_2(l.mlp(l.post_attention_layernorm(z)))
 for layer in layers[13:]:z=layer(z,current_ut=1)
 return outer(z)
assert bits_equal(oracle(a,b),u1) and bits_equal(oracle(b,a),u2)
assert not layers[12]._forward_hooks and not layers[12]._forward_pre_hooks and not layers[12].input_layernorm_2._forward_hooks
print('Q13_FULL_SUFFIX_CPU_PASS; provenance and GPU untested')
