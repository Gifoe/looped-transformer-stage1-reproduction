from contextlib import contextmanager
import torch
import torch.nn.functional as F
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.device==b.device and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def need(x,s):
 if not x:raise ValueError(s)
@contextmanager
def bare(model,cut,R,P,hidden=None,key=None):
 layers=model.model.layers;h=[];active=[];attn=[];seq=[];r={'calls':0};old=F.scaled_dot_product_attention
 def cut_hook(m,args,kw):
  u=kw.get('current_ut'); x=args[0] if args else kw['hidden_states']
  if u==0:r['donor_hidden']=x.detach().clone()
  if u==1:
   r['cut_input']=x.detach().clone()
   if hidden is not None:
    z=x.clone();z[:,P:R,:]=hidden[:,P:R,:];need(bits(z[:,:P,:],x[:,:P,:]) and bits(z[:,R:,:],x[:,R:,:]),'hidden outside Q')
    if args:return (z,)+args[1:],kw
    n=dict(kw);n['hidden_states']=z;return args,n
 def pre(m,args,kw):
  u=kw.get('current_ut');need(u==len(seq) and not active,'target event');seq.append(u);active.append(u)
  if u==1:r['x_R']=(args[0] if args else kw['hidden_states'])[:,R,:].detach().clone()
 def post(m,args,kw,out):need(len(active)==1 and not attn,'target close');active.pop()
 def ap(m,args,kw):need(len(active)==1 and not attn and kw.get('current_ut')==active[0],'attn event');attn.append(1)
 def az(m,args,kw,out):need(len(attn)==1,'attn close');attn.pop()
 def ar(m,args,out):
  if active and active[0]==1:r['a_R']=out[:,R,:].detach().clone()
 def sdpa(*args,**kw):
  if not active or active[0]!=1:return old(*args,**kw)
  need(len(attn)==1 and r['calls']==0 and len(args)>=3,'sdpa event');r['calls']+=1;q,k,v=args[:3];r['k']=k.detach().clone();base=old(*args,**kw)
  if key is not None:
   need(key.shape==k.shape and key.dtype==k.dtype and key.device==k.device,'key match');km=k.clone();km[...,P:R,:]=key[...,P:R,:];need(bits(km[...,:P,:],k[...,:P,:]) and bits(km[...,R:,:],k[...,R:,:]),'key outside Q');alt=old(q,km,v,*args[3:],**kw);z=base.clone();z[...,R,:]=alt[...,R,:];mask=torch.ones(z.shape[-2],device=z.device,dtype=torch.bool);mask[R]=False;need(bits(z[...,mask,:],base[...,mask,:]),'output outside R');base=z
  r['sdpa_R']=base[...,R,:].detach().clone();return base
 try:
  h=[layers[cut].register_forward_pre_hook(cut_hook,with_kwargs=True),layers[17].register_forward_pre_hook(pre,with_kwargs=True),layers[17].register_forward_hook(post,with_kwargs=True),layers[17].self_attn.register_forward_pre_hook(ap,with_kwargs=True),layers[17].self_attn.register_forward_hook(az,with_kwargs=True),layers[17].input_layernorm_2.register_forward_hook(ar)];F.scaled_dot_product_attention=sdpa;yield r;need(seq==[0,1] and r['calls']==1 and not active and not attn and 'a_R' in r,'incomplete')
 finally:
  F.scaled_dot_product_attention=old
  for v in h:v.remove()
