"""Closed boundary/forward CPU staging; not approved for production."""
from contextlib import contextmanager
import torch
from q13_core import bits_equal
from pair_runner import run_matched_pair

def compose(k1,k2,regions,corner):
 if set(regions)!={'P','Q','R'} or corner not in ('211','221','212','222'):raise ValueError('region/corner')
 if k1.shape!=k2.shape or k1.ndim!=3 or k1.shape[0]!=1 or k1.dtype!=k2.dtype or k1.device!=k2.device:raise ValueError('state')
 values=[list(regions[k]) for k in ('P','Q','R')]
 if any(not x or any(type(i)!=int for i in x) or len(x)!=len(set(x)) for x in values) or set.union(*(set(x) for x in values))!=set(range(k1.shape[1])) or sum(map(len,values))!=k1.shape[1]:raise ValueError('partition')
 out=k2.clone()
 for bit,indices in zip(corner,values):
  if bit=='1':out[:,indices,:]=k1[:,indices,:]
 return out

@contextmanager
def boundary(model,depth,k1=None,k2=None,regions=None,corner=None):
 if depth not in (1,2) or len(model.model.layers)!=24 or model.model.layers[11].self_attn.layer_idx!=11:raise ValueError('identity')
 old=model.model.total_ut_steps;events=[];active=[];captured={};layer=model.model.layers[11];handles=[]
 def pre(m,a,k):
  u=k.get('current_ut')
  if type(u)!=int or u!=len(events) or u>=depth or active:raise ValueError('loop')
  events.append(u);active.append(u)
 def post(m,a,k,out):
  if len(active)!=1 or not torch.is_tensor(out) or not torch.isfinite(out).all():raise ValueError('cut state')
  u=active.pop();captured[u]=out.detach().clone()
  if corner is not None and u==1:
   if not bits_equal(out,k2):raise ValueError('K2 cutpoint drift')
   return compose(k1,out,regions,corner)
 try:
  model.model.total_ut_steps=depth;handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True));handles.append(layer.register_forward_hook(post,with_kwargs=True))
  yield captured
  if events!=list(range(depth)) or active or len(captured)!=depth:raise ValueError('incomplete boundary')
 finally:
  for h in handles:h.remove()
  model.model.total_ut_steps=old

def score_forward(model,ids,positions):
 norm=[]
 def h(m,a,out):norm.append(out[0,positions,:].detach().clone())
 handle=model.model.norm.register_forward_hook(h)
 try:
  y=model(ids,labels=None,use_cache=False,exit_at_step=model.model.total_ut_steps-1)
  z=model.lm_head(norm[-1].unsqueeze(0))[0];native=y.logits[0,positions,:]
  if len(norm)!=model.model.total_ut_steps or not bits_equal(z,native) or not torch.isfinite(z).all():raise ValueError('native scoring bridge')
  return z
 finally:handle.remove()

@torch.inference_mode()
def capture_natural(model,ids,positions,source_guard):
 vals=[]
 for depth in (1,2):
  source_guard()
  with boundary(model,depth) as states:z=score_forward(model,ids,positions)
  source_guard();vals.append((states[depth-1],z))
 return vals

@torch.inference_mode()
def run_question(model,ids,regions,source_guard,prior_logits):
 if model.training:raise ValueError('eval mode required')
 natural=capture_natural(model,ids,regions['R'],source_guard);k1,k2=natural[0][0],natural[1][0]
 if set(prior_logits)!={'211','221','212','222'}:raise ValueError('missing previous logits')
 result={}
 for r in ('1','2'):
  a='21'+r;b='22'+r;cut1=compose(k1,k2,regions,a);cut2=compose(k1,k2,regions,b)
  def make_run(corner):
   def run():
    with boundary(model,2,k1,k2,regions,corner):return score_forward(model,ids,regions['R'])
   return run
  plain1=make_run(a)();plain2=make_run(b)()
  if not bits_equal(plain1,prior_logits[a]) or not bits_equal(plain2,prior_logits[b]):raise ValueError('prior corner bridge')
  result[r]=run_matched_pair(model,ids,regions['P'],regions['R'],make_run(a),make_run(b),cut1,cut2,a,b,source_guard)
  if not bits_equal(plain1,result[r]['B1']) or not bits_equal(plain2,result[r]['B2']):raise ValueError('capture observer changed output')
 return natural,result
