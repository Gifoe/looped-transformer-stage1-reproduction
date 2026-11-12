"""Producer splice observer: research candidate; requires independent technical validation."""
from contextlib import contextmanager
import torch,hashlib
import torch.nn.functional as F
from e3_producer_primitives import splice_hidden_Q,splice_postrope_K_Q

def bits(a,b):
 return a.shape==b.shape and a.dtype==b.dtype and a.device==b.device and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

def thash(x):return hashlib.sha256(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
@contextmanager
def producer_observer(model,cut,R,P,hidden_donor=None,key_donor=None,expected_clean_cut=None):
 if model.training or cut not in (0,8,16,17) or type(R)!=int:raise ValueError('mode/cut')
 layers=model.model.layers
 if len(layers)!=24 or any(x.self_attn.layer_idx!=i for i,x in enumerate(layers)):raise ValueError('physical identity')
 handles=[];active=[];attn=[];events=[];rec={'calls':0,'cut_events':[]};old=F.scaled_dot_product_attention
 def cut_pre(m,args,kw):
  u=kw.get('current_ut'); x=args[0] if args else kw.get('hidden_states')
  if type(u)!=int or u not in (0,1) or not torch.is_tensor(x) or x.ndim!=3 or not 0<P<R==x.shape[1]-1:raise ValueError('cut identity')
  rec['cut_events'].append(u)
  if u==0:rec['loop1_cut_hidden']=x.detach().clone();rec['loop1_cut_donor_Q_sha256']=thash(x[:,P:R,:])
  if u==1:
   rec['receiver_cut_hidden']=x.detach().clone();rec['cut_shape']=tuple(x.shape);rec['cut_input_sha256']=thash(x);rec['cut_input_Q_sha256']=thash(x[:,P:R,:]);rec['cut_input_R_sha256']=thash(x[:,R:,:])
   if expected_clean_cut is not None and not bits(x,expected_clean_cut):raise ValueError('clean cut input mismatch')
   if hidden_donor is not None:
    new=splice_hidden_Q(x,hidden_donor,R,P);rec['cut_outside_Q_unchanged']=bits(new[:,:P,:],x[:,:P,:]) and bits(new[:,R:,:],x[:,R:,:]);rec['cut_donor_Q_equal']=bits(new[:,P:R,:],hidden_donor[:,P:R,:]);rec['cut_patched_Q_sha256']=thash(new[:,P:R,:])
    if args:return (new,)+args[1:],kw
    kw=dict(kw);kw['hidden_states']=new;return args,kw
 def target_pre(m,args,kw):
  u=kw.get('current_ut');x=args[0] if args else kw.get('hidden_states')
  if type(u)!=int or u!=len(events) or active or kw.get('past_key_value') is not None or kw.get('use_cache') is not False:raise ValueError('target scope')
  events.append(u);active.append(u)
  if u==1:rec['receiver_x_R']=x[:,R,:].detach().clone()
 def target_post(m,args,kw,out):
  if len(active)!=1 or attn:raise ValueError('target close')
  active.pop()
 def attn_pre(m,args,kw):
  if len(active)!=1 or attn or kw.get('current_ut')!=active[0] or kw.get('past_key_value') is not None:raise ValueError('attention scope')
  attn.append(active[0])
 def attn_post(m,args,kw,out):
  if len(attn)!=1:raise ValueError('attention close')
  attn.pop()
 def wrapper(*args,**kw):
  if not active or active[0]!=1:return old(*args,**kw)
  if not attn or rec['calls'] or len(args)<3:raise ValueError('SDPA scope')
  q,k,v=args[:3]
  if any(not torch.is_tensor(x) or x.ndim!=4 or not torch.isfinite(x).all() for x in (q,k,v)):raise ValueError('nonfinite source')
  rec['calls']+=1
  for n,x in zip(('q','k','v'),(q,k,v)):rec[n]=x.detach().clone()
  rec['tail']=args[3:];rec['kwargs']=kw.copy()
  baseline=old(*args,**kw)
  if not torch.isfinite(baseline).all():raise ValueError('nonfinite baseline')
  rec['original_sdpa_R']=baseline[...,R,:].detach().clone()
  if key_donor is None:
   rec['mixed_K_Q_sha256']=thash(k[...,P:R,:]);rec['donor_K_Q_sha256']=None;rec['mixed_K_Q_shape']=tuple(k[...,P:R,:].shape);rec['mixed_K_Q_dtype']=str(k.dtype);rec['injected_sdpa_R']=rec['original_sdpa_R'].clone();rec['outside_R_unchanged']=True;return baseline
  mixed=splice_postrope_K_Q(k,key_donor,R,P);rec['mixed_K_Q_sha256']=thash(mixed[...,P:R,:]);rec['donor_K_Q_sha256']=thash(key_donor[...,P:R,:]);rec['mixed_K_Q_shape']=tuple(mixed[...,P:R,:].shape);rec['mixed_K_Q_dtype']=str(mixed.dtype);out2=old(q,mixed,v,*args[3:],**kw)
  if not torch.isfinite(out2).all():raise ValueError('nonfinite hybrid')
  result=baseline.clone();result[...,R,:]=out2[...,R,:];mask=torch.ones(result.shape[-2],dtype=torch.bool,device=result.device);mask[R]=False
  rec['outside_R_unchanged']=bits(result[...,mask,:],baseline[...,mask,:]);rec['injected_sdpa_R']=result[...,R,:].detach().clone()
  if not rec['outside_R_unchanged']:raise ValueError('outside R changed')
  return result
 try:
  handles.extend([layers[cut].register_forward_pre_hook(cut_pre,with_kwargs=True),layers[17].register_forward_pre_hook(target_pre,with_kwargs=True),layers[17].register_forward_hook(target_post,with_kwargs=True),layers[17].self_attn.register_forward_pre_hook(attn_pre,with_kwargs=True),layers[17].self_attn.register_forward_hook(attn_post,with_kwargs=True)]);F.scaled_dot_product_attention=wrapper;yield rec
  if rec['cut_events']!=[0,1] or events!=[0,1] or active or attn or rec['calls']!=1 or not rec.get('outside_R_unchanged'):raise ValueError('incomplete trace')
 finally:
  F.scaled_dot_product_attention=old
  for h in handles:h.remove()
