"""FP64 actual-entry R-row causal attention reconstruction and K/V finite difference."""
import numpy as np

def _repeat_heads(a,heads,enable_gqa):
 if a.shape[1]==heads:return a
 if not enable_gqa or heads%a.shape[1]:raise ValueError('head mismatch or GQA not enabled')
 return np.repeat(a,heads//a.shape[1],axis=1)

def attention_row(q,k,v,R,mask=None,is_causal=False,scale=None,enable_gqa=False):
 q=np.asarray(q,dtype=np.float64);k=np.asarray(k,dtype=np.float64);v=np.asarray(v,dtype=np.float64)
 if q.ndim!=4 or k.ndim!=4 or v.shape!=k.shape or q.shape[0]!=k.shape[0] or q.shape[-1]!=k.shape[-1] or not(0<=R<q.shape[-2]) or not all(np.isfinite(x).all() for x in (q,k,v)):raise ValueError('invalid actual QKV')
 k=_repeat_heads(k,q.shape[1],enable_gqa);v=_repeat_heads(v,q.shape[1],enable_gqa); L=k.shape[-2];alpha=float(scale) if scale is not None else q.shape[-1]**-.5
 if not np.isfinite(alpha):raise ValueError('scale')
 logits=np.einsum('bhd,bhld->bhl',q[:,:,R,:],k)*alpha
 if is_causal:
  if q.shape[-2]!=L:raise ValueError('unequal Q/K causal alignment is not identified')
  logits=np.where(np.arange(L)<=R,logits,-np.inf)
 if mask is not None:
  m=np.asarray(mask)
  if m.dtype!=np.bool_ and not np.issubdtype(m.dtype,np.floating):raise ValueError('mask dtype must be boolean or float')
  try:row=np.broadcast_to(m,(q.shape[0],q.shape[1],q.shape[-2],L))[:,:,R,:]
  except ValueError as e:raise ValueError('mask must broadcast to original full Q/K shape') from e
  if m.dtype==np.bool_:logits=np.where(row,logits,-np.inf)
  else:
   if np.isnan(row).any() or np.isposinf(row).any():raise ValueError('invalid mask')
   logits=logits+row
 valid=np.isfinite(logits).any(axis=-1); maxima=np.where(valid,np.max(logits,axis=-1),0);exp=np.exp(logits-maxima[...,None]);den=exp.sum(axis=-1);A=np.divide(exp,den[...,None],out=np.zeros_like(exp),where=den[...,None]!=0);o=np.einsum('bhl,bhld->bhd',A,v)
 return A,o,valid

def decompose(q,k1,v1,k2,v2,R,**kw):
 A1,O11,vld1=attention_row(q,k1,v1,R,**kw);A2,O21,vld2=attention_row(q,k2,v1,R,**kw);_,O12,_=attention_row(q,k1,v2,R,**kw);_,O22,_=attention_row(q,k2,v2,R,**kw)
 V1=_repeat_heads(np.asarray(v1,dtype=np.float64),q.shape[1],kw.get('enable_gqa',False));V2=_repeat_heads(np.asarray(v2,dtype=np.float64),q.shape[1],kw.get('enable_gqa',False));da=A2-A1;dv=V2-V1
 key=np.einsum('bhl,bhld->bhd',da,V1);val=np.einsum('bhl,bhld->bhd',A1,dv);inter=np.einsum('bhl,bhld->bhd',da,dv); residual=O22-O11-key-val-inter
 return {'A1':A1,'A2':A2,'O11':O11,'O12':O12,'O21':O21,'O22':O22,'key':key,'value':val,'interaction':inter,'accounting_residual':residual,'valid_A1':vld1,'valid_A2':vld2}
