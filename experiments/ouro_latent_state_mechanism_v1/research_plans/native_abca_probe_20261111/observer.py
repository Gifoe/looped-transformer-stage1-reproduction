"""Instrumented observation draft. Not approved for real GPU use."""
from contextlib import contextmanager
import torch
import torch.nn.functional as F

def copy(v):
 if v is None:return None
 if torch.is_tensor(v):return v.detach().cpu().clone()
 if isinstance(v,tuple):return tuple(copy(x) for x in v)
 raise TypeError('unexpected observation type')

@contextmanager
def observe(model,prefix_length):
 if type(prefix_length)!=int or prefix_length<1:raise ValueError('prefix')
 core=model.model
 if len(core.layers)!=24 or any(x.self_attn.layer_idx!=i for i,x in enumerate(core.layers)):raise ValueError('layer identity')
 saved=F.scaled_dot_product_attention;handles=[];active=[];layer_events=[];calls=[];states=[];rope=[];attention=[];attn_events=[]
 def make_pre(i):
  def pre(m,a,k):
   u=k.get('current_ut')
   if type(u)!=int or (u,i)!=(len(layer_events)//24,len(layer_events)%24) or active:raise RuntimeError('layer order')
   active.append((u,i));layer_events.append((u,i))
   if k.get('past_key_value') is not None or k.get('use_cache') is not False:raise RuntimeError('past/cache')
   if i==11: states.append({'loop':u,'mask':copy(k.get('attention_mask')),'position_embeddings':copy(k.get('position_embeddings')),'cache_position':copy(k.get('cache_position'))})
  return pre
 def make_post(i):
  def post(m,a,k,out):
   if not active or active[-1][1]!=i or not torch.is_tensor(out):raise RuntimeError('layer output')
   if i==11:
    if out.ndim!=3 or out.shape[0]!=1 or out.shape[1]<prefix_length:raise RuntimeError('shape')
    states[-1]['source_device']=str(out.device);states[-1]['prefix']=copy(out[:,:prefix_length,:])
   active.pop()
  return post
 def make_apre(i):
  def pre(m,a,k):
   if len(active)!=1 or active[0][1]!=i or attention: raise RuntimeError('attention scope')
   required=('attention_mask','position_embeddings','past_key_value','use_cache')
   if any(n not in k for n in required):raise RuntimeError('missing attention argument')
   if k['past_key_value'] is not None or k['use_cache'] is not False:raise RuntimeError('attention past/cache')
   attention.append({'layer_loop':active[-1],'attention_mask':copy(k['attention_mask']),'position_embeddings':copy(k['position_embeddings']),'past_is_none':True,'use_cache':k['use_cache'],'start':len(calls)})
  return pre
 def make_apost(i):
  def post(m,a,k,out):
   if len(attention)!=1 or attention[0]['layer_loop']!=active[-1] or attention[0]['layer_loop'][1]!=i or len(calls)!=attention[0]['start']+1 or not calls[-1]['completed']:raise RuntimeError('attention call count')
   attn_events.append(attention.pop())
  return post
 def ropepre(m,a,k):
  rope.append({'position_ids':copy(a[1] if len(a)>1 else k['position_ids'])})
 def ropepost(m,a,k,out):rope[-1]['source_devices']=[str(x.device) for x in out];rope[-1]['cos_sin']=copy(out)
 def api(*args,**kwargs):
  if len(active)!=1 or len(attention)!=1 or attention[0]['layer_loop']!=active[0]:raise RuntimeError('backend outside self_attention')
  if len(args)<3:raise RuntimeError('SDPA inputs missing')
  names=('attn_mask','dropout_p','is_causal','scale','enable_gqa')
  values={n:kwargs.get(n, args[3+j] if len(args)>3+j else 'UNSPECIFIED') for j,n in enumerate(names)}
  values['attn_mask']=copy(values['attn_mask']) if torch.is_tensor(values['attn_mask']) else values['attn_mask']
  call={'layer_loop':active[-1], 'qkv':[{'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device)} for x in args[:3]],'completed':False,**values}
  calls.append(call)
  result=saved(*args,**kwargs)
  call['completed']=True
  return result
 try:
  for i,layer in enumerate(core.layers):
   handles.append(layer.register_forward_pre_hook(make_pre(i),with_kwargs=True));handles.append(layer.register_forward_hook(make_post(i),with_kwargs=True));handles.append(layer.self_attn.register_forward_pre_hook(make_apre(i),with_kwargs=True));handles.append(layer.self_attn.register_forward_hook(make_apost(i),with_kwargs=True))
  handles.append(core.rotary_emb.register_forward_pre_hook(ropepre,with_kwargs=True));handles.append(core.rotary_emb.register_forward_hook(ropepost,with_kwargs=True))
  F.scaled_dot_product_attention=api
  yield {'layers':layer_events,'sdpa':calls,'attention':attn_events,'states':states,'rope':rope,'kernel':'UNKNOWN'}
  expected=[(u,i) for u in range(2) for i in range(24)]
  if layer_events!=expected or [x['layer_loop'] for x in calls]!=expected or active or len(attn_events)!=48 or attention or any(not x['completed'] for x in calls) or len(rope)!=1 or len(states)!=2:raise RuntimeError('incomplete event sequence')
 finally:
  F.scaled_dot_product_attention=saved
  for h in handles:h.remove()
