"""Natural L18 residual attention hook, CPU draft only; no production entrypoint."""
from contextlib import contextmanager
import torch
from state_capture import bits
if not __debug__:raise RuntimeError('optimized Python forbidden')
@contextmanager
def intervene(model,depth,R,expected_x=None,expected_a=None,replacement=None):
 if model.training or depth not in (1,2) or type(R)!=int or R<0:raise ValueError('model/mode')
 core=model.model
 if len(core.layers)!=24 or any(layer.self_attn.layer_idx!=i for i,layer in enumerate(core.layers)):raise ValueError('physical layers')
 if (expected_x is None)!=(expected_a is None) or (replacement is not None and expected_x is None):raise ValueError('receiver validation missing')
 layer=core.layers[17];target=depth-1;active=[];events=[];result={'visits':0};handles=[]
 def pre(m,args,kwargs):
  u=kwargs.get('current_ut');x=args[0] if args else kwargs.get('hidden_states')
  if type(u)!=int or u!=len(events) or active or not torch.is_tensor(x) or x.ndim!=3 or x.shape[0]!=1 or R>=x.shape[1]:raise ValueError('loop/layer/input')
  if kwargs.get('use_cache') is not False or kwargs.get('past_key_value') is not None:raise ValueError('cache')
  events.append(u);active.append(u)
  if u==target:
   xr=x[:,R,:].detach().clone()
   if expected_x is not None and not bits(xr,expected_x.to(xr.device)):raise ValueError('receiver x mismatch')
   result['receiver_x']=xr
 def branch(m,args,out):
  if len(active)!=1:raise ValueError('branch scope')
  if active[0]!=target:return None
  if result['visits'] or not torch.is_tensor(out) or out.ndim!=3 or not torch.isfinite(out).all():raise ValueError('branch')
  ar=out[:,R,:].detach().clone();result['visits']+=1;result['receiver_a']=ar
  if expected_a is not None and not bits(ar,expected_a.to(ar.device)):raise ValueError('receiver a mismatch')
  if replacement is None:return None
  d=replacement.detach().to(out.device)
  if d.dtype!=out.dtype or d.shape!=ar.shape or not torch.isfinite(d).all():raise ValueError('donor')
  patched=out.clone();patched[:,R,:]=d
  if not bits(patched[:,R,:],d):raise ValueError('donor write mismatch')
  mask=torch.ones(out.shape[1],dtype=torch.bool,device=out.device);mask[R]=False
  if not bits(patched[:,mask,:],out[:,mask,:]):raise ValueError('outside R changed')
  return patched
 def post(m,args,kwargs,out):
  if len(active)!=1:raise ValueError('scope')
  active.pop()
 try:
  handles.append(layer.register_forward_pre_hook(pre,with_kwargs=True));handles.append(layer.input_layernorm_2.register_forward_hook(branch));handles.append(layer.register_forward_hook(post,with_kwargs=True));yield result
  if events!=list(range(depth)) or active or result['visits']!=1:raise ValueError('incomplete branch forward')
 finally:
  for h in handles:h.remove()

def matched_norm(donor,receiver):
 if donor.dtype!=torch.bfloat16 or receiver.dtype!=torch.bfloat16 or donor.shape!=receiver.shape or not torch.isfinite(donor).all() or not torch.isfinite(receiver).all():raise ValueError('norm control inputs')
 x=donor.to(torch.float32);y=receiver.to(torch.float32);dn=torch.linalg.vector_norm(x);rn=torch.linalg.vector_norm(y)
 if dn.item()==0:return None,{'status':'UNDEFINED_ZERO_DONOR','donor_norm':0.,'receiver_norm':rn.item()}
 d=(x*(rn/dn)).to(torch.bfloat16);actual=torch.linalg.vector_norm(d.to(torch.float32)).item()
 return d,{'status':'OK','donor_norm':dn.item(),'receiver_norm':rn.item(),'actual_norm':actual,'norm_error':actual-rn.item()}
