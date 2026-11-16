"""AB1 site hooks; candidate implementation, requires true-model technical acceptance."""
from contextlib import contextmanager
import torch,hashlib

def describe(x):
 if torch.is_tensor(x):
  raw=x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes();return {"shape":list(x.shape),"stride":list(x.stride()),"dtype":str(x.dtype),"sha256":hashlib.sha256(raw).hexdigest()}
 if isinstance(x,(tuple,list)):return [describe(y) for y in x]
 if x is None or isinstance(x,(bool,int,float,str)):return x
 raise TypeError("unsupported attention argument type: "+type(x).__name__)

def actual_args(kw):return {k:describe(kw[k]) for k in ("attention_mask","position_ids","position_embeddings","cache_position","use_cache","current_ut") if k in kw}

def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

def _hidden(args,kw):
 return args[0] if args else kw['hidden_states']
def _replace(args,kw,x):
 if args:return (x,*args[1:]),kw
 k=kw.copy();k['hidden_states']=x;return args,k

@contextmanager
def capture_and_patch(model,seq_len,R,prefix,*,boundary_input=None,q9_donor=None):
 """Capture actual E, first norm B, native second input before splice, and Q9.

 The caller is responsible for source identity, root input IDs, full logits, model
 version, source freezing and pairing B/E/Q9 with same-question natural forwards.
 Returning a clone at a layer input is intended: the caller retains the original
 source tensor. No attempt is made to recreate norm internals or SDPA arguments.
 """
 if model.training or model.model.total_ut_steps!=2 or not 0<prefix<R==seq_len-1:raise ValueError('runtime/region')
 rec={};events=[];norm_count=[];q_events=[];handles=[]
 def l1(mod,args,kw):
  ut=kw['current_ut']
  if ut!=len(events) or ut not in (0,1) or kw.get('use_cache') is not False or kw.get('past_key_value') is not None:raise RuntimeError('L1 sequence/cache')
  events.append(ut);x=_hidden(args,kw)
  if x.ndim!=3 or x.shape[:2]!=(1,seq_len) or not torch.isfinite(x).all():raise RuntimeError('L1 input')
  if ut==0:rec['E']=x.detach().clone();return
  if 'B' not in rec or not bits(x,rec['B']):raise RuntimeError('K1 norm != K2 L1')
  rec['B_at_L1']=x.detach().clone();rec['L1_kwargs']=actual_args(kw);applied=x if boundary_input is None else boundary_input.to(x.device)
  if applied.shape!=x.shape or applied.dtype!=x.dtype or not torch.isfinite(applied).all():raise RuntimeError('boundary replacement')
  rec['K2_applied']=applied.detach().clone()
  if boundary_input is not None:return _replace(args,kw,applied.clone())
 def norm(mod,args,out):
  norm_count.append(len(norm_count))
  if len(norm_count)==1:rec['B']=out.detach().clone()
  elif len(norm_count)==2:rec['K2_final_R']=out[:,R,:].detach().clone()
  else:raise RuntimeError('unexpected final norm count')
 def l9(mod,args,kw):
  ut=kw['current_ut']
  if ut!=len(q_events) or ut not in (0,1):raise RuntimeError('L9 sequence')
  q_events.append(ut)
  if ut==1:
   x=_hidden(args,kw);rec['Q9_actual']=x.detach().clone();rec['L9_kwargs']=actual_args(kw)
   if q9_donor is not None:
    v=q9_donor.to(x.device)
    if v.shape!=x[:,prefix:R,:].shape or v.dtype!=x.dtype or not torch.isfinite(v).all():raise RuntimeError('Q9 source')
    z=x.clone();z[:,prefix:R,:]=v;mask=torch.ones(seq_len,dtype=torch.bool,device=x.device);mask[prefix:R]=False
    if not bits(z[:,mask,:],x[:,mask,:]):raise RuntimeError('Q9 non-Q modified')
    rec['Q9_applied']=z.detach().clone();return _replace(args,kw,z)
 try:
  handles=[model.model.layers[0].register_forward_pre_hook(l1,with_kwargs=True),model.model.layers[8].register_forward_pre_hook(l9,with_kwargs=True),model.model.norm.register_forward_hook(norm)]
  yield rec
  if events!=[0,1] or q_events!=[0,1] or norm_count!=[0,1]:raise RuntimeError('incomplete events')
 finally:
  for h in handles:h.remove()
