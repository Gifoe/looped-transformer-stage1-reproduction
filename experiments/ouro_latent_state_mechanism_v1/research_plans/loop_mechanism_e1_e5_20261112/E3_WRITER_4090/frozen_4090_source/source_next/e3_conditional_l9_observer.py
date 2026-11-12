"""Capture and optionally patch only Q-region post-RoPE KV at L9/R SDPA."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from e3_conditional_l9_kv_row import l9_qkv_row
@contextmanager
def observe_l9(model,R,prefix,*,donor_k=None,donor_v=None):
 layer=model.model.layers[8];old=F.scaled_dot_product_attention;active=[];attn=[];events=[];rec={};handles=[]
 def pre(m,args,kw):
  u=kw.get('current_ut');assert u==len(events) and not active and kw.get('past_key_value') is None and kw.get('use_cache') is False;events.append(u);active.append(u)
  if u==1:
   x=args[0] if args else kw['hidden_states'];rec['input_R']=x[:,R,:].detach().clone();rec['position_arguments']={k:v for k,v in kw.items() if k in ('position_ids','position_embeddings','attention_mask','cache_position')}
 def post(m,args,kw,out):
  assert active.pop()==kw['current_ut'] and not attn
  if kw['current_ut']==1:
   x=out[0] if isinstance(out,(tuple,list)) else out;assert torch.is_tensor(x) and x.ndim==3;rec['post_mlp_R']=x[:,R,:].detach().clone()
 def ap(m,args,kw):assert active and not attn;attn.append(1)
 def az(m,args,kw,out):assert attn.pop()==1
 def anorm(m,args,out):
  if active and active[-1]==1:rec['a_R']=out[:,R,:].detach().clone()
 def sdpa(*args,**kwargs):
  if not active or active[-1]!=1 or not attn:return old(*args,**kwargs)
  assert 'q' not in rec and len(args)>=3;q,k,v=args[:3];rec.update(q=q.detach().clone(),k=k.detach().clone(),v=v.detach().clone(),sdpa_args=tuple(args[3:]),sdpa_kwargs=dict(kwargs));baseline=old(*args,**kwargs);rec['sdpa_base_R']=baseline[...,R,:].detach().clone()
  if donor_k is None and donor_v is None:return baseline
  km=k.clone();vm=v.clone();km[...,prefix:R,:]=donor_k[...,prefix:R,:] if donor_k is not None else k[...,prefix:R,:];vm[...,prefix:R,:]=donor_v[...,prefix:R,:] if donor_v is not None else v[...,prefix:R,:];assert torch.equal(km[...,:prefix,:],k[...,:prefix,:]) and torch.equal(km[...,R:,:],k[...,R:,:]) and torch.equal(vm[...,:prefix,:],v[...,:prefix,:]) and torch.equal(vm[...,R:,:],v[...,R:,:]);alternate=old(q,km,vm,*args[3:],**kwargs);out=baseline.clone();out[...,R,:]=alternate[...,R,:];mask=torch.arange(q.shape[-2],device=q.device)!=R;assert torch.equal(out[...,mask,:],baseline[...,mask,:]);rec['injected_R']=out[...,R,:].detach().clone();return out
 try:
  handles=[layer.register_forward_pre_hook(pre,with_kwargs=True),layer.register_forward_hook(post,with_kwargs=True),layer.self_attn.register_forward_pre_hook(ap,with_kwargs=True),layer.self_attn.register_forward_hook(az,with_kwargs=True),layer.input_layernorm_2.register_forward_hook(anorm)];F.scaled_dot_product_attention=sdpa;yield rec;assert events==[0,1] and not active and not attn and all(k in rec for k in ('q','k','v','input_R','post_mlp_R','sdpa_base_R','a_R'))
 finally:
  F.scaled_dot_product_attention=old
  for h in handles:h.remove()
