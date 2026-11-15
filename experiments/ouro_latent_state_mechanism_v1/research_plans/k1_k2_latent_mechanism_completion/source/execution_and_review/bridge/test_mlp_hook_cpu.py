"""CPU hook acceptance, independent expected FP64 proposal; no model execution."""
import ast,pathlib,torch
from contextlib import contextmanager
s=pathlib.Path(__file__).with_name('b2_forward.py').read_text(encoding='utf8');tree=ast.parse(s);n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mlp_reference_operator')
class UndefinedControl(RuntimeError):pass
def bits(a,b):return torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
env=dict(torch=torch,contextmanager=contextmanager,bits=bits,UndefinedControl=UndefinedControl);exec('@contextmanager\n'+ast.get_source_segment(s,n),env);op=env['mlp_reference_operator']
class Layer(torch.nn.Module):
 def __init__(self):super().__init__();self.post_attention_layernorm_2=torch.nn.Identity()
 def forward(self,x,**kw):return self.post_attention_layernorm_2(x)
class Body:pass
class Model:
 def __init__(self):self.training=False;self.model=Body();self.model.total_ut_steps=2;self.model.layers=[Layer() for _ in range(24)]
m=Model();x=torch.linspace(-2,3,3*2048).reshape(1,3,2048).bfloat16();d={i:torch.linspace(-.3,.2,2048).reshape(1,2048).double() for i in range(9,16)}
for sign,norm in [(-1,False),(1,False),(-1,True),(1,True)]:
 with op(m,2,sequence_length=3,reference_delta=d,sign=sign,norm_preserve=norm) as rec:
  for ut in (0,1):
   for i,l in enumerate(m.model.layers):
    z=l(x,current_ut=ut,use_cache=False,past_key_value=None)
    if ut==0 or i not in d:assert bits(z,x)
    else:
     exact=x[:,2,:].double()+sign*d[i]
     if norm:exact=exact*(x[:,2,:].double().norm()/exact.norm())
     assert bits(z[:,:2,:],x[:,:2,:]) and bits(z[:,2,:],exact.bfloat16())
     assert bits(rec[i]['exact_before_cast'],exact)
     assert bits(rec[i]['cast_residual'],z[:,2,:].double()-exact)
 assert len(rec)==7
# Capture-only is identical and all seven before rows are present.
with op(m,2,sequence_length=3) as rec:
 for ut in (0,1):
  for l in m.model.layers:assert bits(l(x,current_ut=ut,use_cache=False,past_key_value=None),x)
assert all(bits(r['before'],x[:,2,:]) for r in rec.values())
print('PASS: 24-layer/2-loop MLP hook scope, +/-FP64 and norm arithmetic, BF16 cast residual, nonR, capture-only')

fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='complete_capture_identity')
exec(ast.get_source_segment(pathlib.Path(__file__).with_name('b2_forward.py').read_text(encoding='utf8'),fn),env)
old={i:{k:v.clone() if torch.is_tensor(v) else v for k,v in r.items()} for i,r in rec.items()}
env['complete_capture_identity'](rec)
for i,r in rec.items():
 assert bits(r['before'],old[i]['before']) and bits(r['after'],old[i]['after'])
 assert bits(r['exact_before_cast'],r['before'].double())
 assert float(r['normalization_scale_fp64'])==1 and not r['cast_residual'].any() and not r['effective_delta'].any()
 assert r['derived_capture_identity_fields'] is True
patched={0:{'before':x[:,2,:],'after':x[:,2,:],'exact_before_cast':x[:,2,:].double()}}
keys=set(patched[0]);env['complete_capture_identity'](patched);assert set(patched[0])==keys
print('PASS: derived identity receipts leave forward before/after and existing patched receipts unchanged')

# Exact norm-degenerate proposal: preserve witness and prove hooks clean up.
for func in ('attention_reference_operator','mlp_reference_operator'):
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==func)
 exec('@contextmanager\n'+ast.get_source_segment(pathlib.Path(__file__).with_name('b2_forward.py').read_text(encoding='utf8'),node),env)
 if func.startswith('attention'):
  for l in m.model.layers:l.input_layernorm_2=l.post_attention_layernorm_2
 dd={i:x[:,2,:].double().clone() for i in range(9,16)}
 try:
  with env[func](m,2,sequence_length=3,reference_delta=dd,sign=-1,norm_preserve=True):
   for ut in (0,1):
    for l in m.model.layers:l(x,current_ut=ut,use_cache=False,past_key_value=None)
  raise AssertionError('undefined proposal accepted')
 except UndefinedControl as e:
  w=e.witness;assert w['site']==9 and w['sign']==-1 and w['norm_preserve']
  assert bits(w['proposal_before_norm'],w['before'].double()-w['reference_delta'])
  assert not w['proposal_before_norm'].any() and w['before'].double().norm()>0
 assert all(not l._forward_pre_hooks and not l._forward_hooks and not l.post_attention_layernorm_2._forward_hooks for l in m.model.layers)
print('PASS: attention/MLP exact undefined witnesses and exception hook cleanup')
