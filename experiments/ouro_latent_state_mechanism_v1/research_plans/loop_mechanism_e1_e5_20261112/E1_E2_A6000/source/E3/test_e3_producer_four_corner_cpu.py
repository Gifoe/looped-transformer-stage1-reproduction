"""CPU-only 24-layer, two-loop independent oracle for upstream Q splice x K_Q transfer."""
import sys,torch,torch.nn.functional as F
from pathlib import Path
from contextlib import contextmanager
sys.path.insert(0,str(Path.home()/'.codex/natural_qkv_probe_review_20261112'))
from test_qkv_observer import Model
from e3_producer_primitives import splice_hidden_Q,splice_postrope_K_Q
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
@contextmanager
def intervention(m,cut,donor_hidden,R,P,perturb=False,key_source=None):
 layers=m.model.layers;active=[];scope=[];rec={'calls':0};hooks=[];old=F.scaled_dot_product_attention
 def cutpre(mod,args,kw):
  if kw.get('current_ut')==0:rec['loop1_cut']=args[0].detach().clone()
  if kw.get('current_ut')==1 and perturb:
   if donor_hidden is None:raise ValueError('no hidden donor')
   return (splice_hidden_Q(args[0],donor_hidden,R,P),)+args[1:],kw
 def lp(mod,args,kw):
  if kw.get('current_ut')==1:active.append(1)
 def lpost(mod,args,kw,out):
  if kw.get('current_ut')==1:
   if scope:raise ValueError('open attention')
   active.pop()
 def ap(mod,args,kw):
  if active:scope.append(1)
 def az(mod,args,kw,out):
  if active:scope.pop()
 def wrapper(*args,**kw):
  if not active:return old(*args,**kw)
  if not scope or rec['calls']:raise ValueError('scope')
  q,k,v=args[:3];rec['calls']+=1;rec['target_k']=k.detach().clone();rec['target_q']=q.detach().clone();rec['target_v']=v.detach().clone();base=old(*args,**kw)
  if key_source is None:return base
  km=splice_postrope_K_Q(k,key_source,R,P);hyb=old(q,km,v,*args[3:],**kw);out=base.clone();out[...,R,:]=hyb[...,R,:];mask=torch.ones(out.shape[-2],dtype=torch.bool);mask[R]=False
  if not bits(out[...,mask,:],base[...,mask,:]):raise ValueError('outside R changed')
  return out
 try:
  hooks.extend([layers[cut].register_forward_pre_hook(cutpre,with_kwargs=True),layers[17].register_forward_pre_hook(lp,with_kwargs=True),layers[17].register_forward_hook(lpost,with_kwargs=True),layers[17].self_attn.register_forward_pre_hook(ap,with_kwargs=True),layers[17].self_attn.register_forward_hook(az,with_kwargs=True)]);F.scaled_dot_product_attention=wrapper;yield rec
  if active or scope or rec['calls']!=1:raise ValueError('incomplete')
 finally:
  F.scaled_dot_product_attention=old
  for h in hooks:h.remove()
@torch.inference_mode()
def manual(m,x,cut,h1,R,P,perturb,key_source):
 # Explicit no-tested-hooks reference, recomputes every residual/MLP in both loops.
 for u in range(2):
  for i,l in enumerate(m.model.layers):
   if u==1 and i==cut and perturb:
    y=x.clone();y[:,P:R,:]=h1[:,P:R,:];x=y
   h=l.input_layernorm(x);q,k,v=l.self_attn.qkv(h);base=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
   if u==1 and i==17 and key_source is not None:
    km=k.clone();km[...,P:R,:]=key_source[...,P:R,:];hyb=F.scaled_dot_product_attention(q,km,v,is_causal=True,dropout_p=0.);base=base.clone();base[...,R,:]=hyb[...,R,:]
   a=l.input_layernorm_2(l.self_attn.o(base.transpose(1,2).reshape(h.shape)));y=x+a;x=y+l.post_attention_layernorm_2(l.mlp(y))
  x=m.model.norm(x)
 return x
@torch.inference_mode()
def main():
 torch.set_num_threads(1);torch.manual_seed(143);m=Model().bfloat16().eval();x=torch.randn(1,9,8).bfloat16();R=8;P=4;orig=F.scaled_dot_product_attention
 for cut in (0,8,16,17):
  with intervention(m,cut,None,R,P) as clean:y0=m(x,2)
  h1=clean['loop1_cut'];k0=clean['target_k']
  with intervention(m,cut,h1,R,P,True) as changed:y1=m(x,2)
  k1=changed['target_k']
  if cut==17:
   # The splice is exactly upstream of the tokenwise K projection at the target layer.
   t=x.clone()
   for layer in m.model.layers[:17]:
    h=layer.input_layernorm(t); q,k,v=layer.self_attn.qkv(h)
    o=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.)
    a=layer.input_layernorm_2(layer.self_attn.o(o.transpose(1,2).reshape(h.shape)))
    y=t+a; t=y+layer.post_attention_layernorm_2(layer.mlp(y))
   _,native1_k,_=m.model.layers[17].self_attn.qkv(m.model.layers[17].input_layernorm(t))
   assert bits(k1[...,P:R,:],native1_k[...,P:R,:])
   assert bits(k1[...,:P,:],k0[...,:P,:]) and bits(k1[...,R:,:],k0[...,R:,:])
   assert bits(changed['target_q'][...,R,:],clean['target_q'][...,R,:])
  for perturb,ks in ((False,None),(False,k1),(True,None),(True,k0)):
   with intervention(m,cut,h1,R,P,perturb,ks) as rr:z=m(x,2)
   assert bits(z,manual(m,x,cut,h1,R,P,perturb,ks))
  assert bits(y0,manual(m,x,cut,h1,R,P,False,None)) and bits(y1,manual(m,x,cut,h1,R,P,True,None))
 assert F.scaled_dot_product_attention is orig
 print('E3_QKEY_PRODUCER_CLEAN_PERTURBED_FOUR_CORNERS_24_LAYER_CPU_PASS')
main()
