"""Independent AB1 model replay using explicit layer-0/layer-8 method wrappers.

Source draft only. Caller supplies same-question archived intervention operands and
must independently recalculate controls, verify actual source/model/runtime, and
compare full-vocab outputs and native no-hook baseline on three technical cases.
"""
import types,torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
@torch.inference_mode()
def replay(model,ids,R,prefix,*,boundary=None,q9_Q=None):
 if model.training or model.model.total_ut_steps!=2 or ids.ndim!=2 or ids.shape[0]!=1 or R!=ids.shape[1]-1 or not 0<prefix<R:raise ValueError('runtime/position')
 core=model.model;l0=core.layers[0];l9=core.layers[8];norm=core.norm
 for m in (l0,l9,norm):
  if m._forward_hooks or m._forward_pre_hooks:raise RuntimeError('replay requires clean layers')
 old0=l0.forward;old9=l9.forward;oldnorm=norm.forward;events0=[];events9=[];nevents=[];rec={};guard=[(p,p._version) for p in model.parameters()]
 def layer0(self,x,**kw):
  u=kw['current_ut'];assert u==len(events0) and u in (0,1) and kw['use_cache'] is False and kw.get('past_key_value') is None;events0.append(u)
  if u==0:rec['E']=x.detach().clone()
  else:
   assert 'B' in rec and bits(x,rec['B']);rec['B_at_L1']=x.detach().clone()
   if boundary is not None:
    v=boundary.to(x.device);assert v.shape==x.shape and v.dtype==x.dtype and torch.isfinite(v).all();x=v.clone()
   rec['applied_L1']=x.detach().clone()
  return old0(x,**kw)
 def layer9(self,x,**kw):
  u=kw['current_ut'];assert u==len(events9) and u in (0,1);events9.append(u)
  if u==1:
   rec['Q9_actual']=x.detach().clone()
   if q9_Q is not None:
    v=q9_Q.to(x.device);assert v.shape==x[:,prefix:R].shape and v.dtype==x.dtype and torch.isfinite(v).all();original=x;x=x.clone();x[:,prefix:R]=v;assert bits(x[:,:prefix],original[:,:prefix]) and bits(x[:,R:],original[:,R:]);rec['Q9_applied']=x.detach().clone()
  return old9(x,**kw)
 def normfn(self,x,*args,**kwargs):
  out=oldnorm(x,*args,**kwargs);nevents.append(len(nevents))
  if len(nevents)==1:rec['B']=out.detach().clone()
  return out
 try:
  l0.forward=types.MethodType(layer0,l0);l9.forward=types.MethodType(layer9,l9);norm.forward=types.MethodType(normfn,norm)
  _,states,_=core(ids,use_cache=False,past_key_values=None);z=model.lm_head(states[1])[:,R,:].detach().clone();k1=model.lm_head(states[0])[:,R,:].detach().clone()
  assert events0==events9==nevents==[0,1] and all(p._version==v for p,v in guard) and z.dtype==k1.dtype==torch.bfloat16 and z.shape==k1.shape==(1,49152)
  return {'logits':z.cpu(),'K1_logits':k1.cpu(),'E':rec['E'].cpu(),'B':rec['B'].cpu(),'Q9_actual':rec['Q9_actual'].cpu(),'Q9_applied':rec.get('Q9_applied',rec['Q9_actual']).cpu(),'implementation':'independent explicit layer-method boundary/Q9 wrapper; native model/score reused'}
 finally:
  l0.forward=old0;l9.forward=old9;norm.forward=oldnorm
  assert all(p._version==v for p,v in guard)
