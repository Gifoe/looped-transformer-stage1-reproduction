"""Unlaunched hook framework. Scientific tensors/conditions require separately frozen order.

No model loader, cohort, dose choice, labels, CLI forward or GPU scheduler exists here.
"""
from contextlib import contextmanager
import hashlib
import torch
import torch.nn.functional as F

def bits(a,b):return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.detach().contiguous().reshape(-1).view(torch.uint8),b.detach().contiguous().reshape(-1).view(torch.uint8))
def thash(x):return hashlib.sha256(x.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
def description(x):
 if torch.is_tensor(x):return {'shape':list(x.shape),'stride':list(x.stride()),'dtype':str(x.dtype),'sha256':thash(x)}
 return x
def explicit_row(value,like):
 assert value.shape==like.shape and value.dtype==like.dtype and like.dtype in [torch.bfloat16,torch.float32] and torch.isfinite(value).all()
 return value.to(device=like.device)

@contextmanager
def attention_reference_operator(model,R,*,sequence_length,reference_delta=None,sign=-1,norm_preserve=False):
 """E3 exact FP64 site-add/BF16-writeback arithmetic; caller supplies frozen same-hardware refs.

Natural K2: current_ut1, physical L10..16. No Q9 transplant is applied by this operator.
Which reference, sign, controls or direction to run is not selected by this interface.
 """
 assert not model.training and model.model.total_ut_steps==2 and R==sequence_length-1 and sign in [-1,1]
 indices=tuple(range(9,16));assert reference_delta is None or set(reference_delta)==set(indices);events={i:[] for i in indices};active=[];records={};handles=[]
 def pre(i):
  def fn(m,args,kw):
   ut=kw['current_ut'];assert not active and ut==len(events[i]) and ut in [0,1] and kw['use_cache'] is False and kw.get('past_key_value') is None;events[i].append(ut);active.append((i,ut))
  return fn
 def post(i):
  def fn(m,args,kw,result):assert active.pop()==(i,kw['current_ut'])
  return fn
 def patch(i):
  def fn(m,args,result):
   assert active[-1][0]==i
   if active[-1][1]!=1:return
   assert i not in records and result.shape==(1,sequence_length,2048) and result.dtype in [torch.bfloat16,torch.float32];before=result[:,R,:].detach().clone()
   if reference_delta is None:records[i]={'before':before,'after':before.clone(),'non_R_unchanged':True,'capture_only':True};return
   delta=reference_delta[i];assert delta.shape==(1,2048) and delta.dtype==torch.float64 and torch.isfinite(delta).all();exact=before.double()+sign*delta.to(result.device);scale=torch.ones((),dtype=torch.float64,device=result.device)
   if norm_preserve:
    n0=before.double().norm();n1=exact.norm()
    if float(n1)==0 and float(n0)!=0:raise UndefinedControl('zero upstream proposal norm with nonzero receiver norm')
    if float(n1)!=0:scale=n0/n1;exact=exact*scale
   after=exact.to(result.dtype);assert torch.isfinite(after).all();changed=result.clone();changed[:,R,:]=after;mask=torch.arange(sequence_length,device=result.device)!=R;assert bits(changed[:,mask,:],result[:,mask,:]);records[i]={'before':before,'after':after.detach().clone(),'reference_delta':delta.detach().clone(),'effective_delta':after.double()-before.double(),'exact_before_cast':exact.detach().clone(),'normalization_scale_fp64':scale.detach().clone(),'cast_residual':after.double()-exact,'non_R_unchanged':True};return changed
  return fn
 try:
  for i in indices:
   l=model.model.layers[i];handles.extend([l.register_forward_pre_hook(pre(i),with_kwargs=True),l.input_layernorm_2.register_forward_hook(patch(i)),l.register_forward_hook(post(i),with_kwargs=True)])
  yield records
  assert not active and all(v==[0,1] for v in events.values()) and set(records)==set(indices)
 finally:
  for h in handles:h.remove()

@contextmanager
def actual_L18_branch_interface(model,R,*,sequence_length,x_branch=None,a_branch=None,capture_full_qkv=False,rewrite_identity=False):
 """Observe actual U attention first; compose exact BF16 x/a at the shared postadd tensor.

Does NOT replace L18 input x: QKV and attention always use the actual upstream receiver.
None uses the actual operand; explicitly provided BF16 R rows select a branch donor.
An in-place post_attention_layernorm prehook changes BOTH MLP input and its shared skip.
No SDPA-output or normalized-attention-output injection is done by this interface.
 """
 assert not model.training and model.model.total_ut_steps==2 and R==sequence_length-1;layer=model.model.layers[17];events=[];active=[];attn=[];norm_events=[];records={};handles=[];old=F.scaled_dot_product_attention
 def pre(m,args,kw):
  ut=kw['current_ut'];assert ut==len(events) and not active and ut in [0,1] and kw['use_cache'] is False and kw.get('past_key_value') is None;events.append(ut);active.append(ut)
  if ut==1:
   x=args[0] if args else kw['hidden_states'];assert x.shape==(1,sequence_length,2048) and x.dtype in [torch.bfloat16,torch.float32];mask=torch.arange(sequence_length,device=x.device)!=R;records['x_actual_R']=x[:,R,:].detach().clone();records['non_R_x_sha256']=thash(x[:,mask,:]);records['input_row_not_replaced']=True
 def post(m,args,kw,result):
  assert active==[kw['current_ut']] and not attn
  if active==[1]:
   records['after_mlp_R']=result[:,R,:].detach().clone();mask=torch.arange(sequence_length,device=result.device)!=R;records['non_R_after_mlp_sha256']=thash(result[:,mask,:]);assert bits(records['after_mlp_R'],records['y_branch_R']+records['b_actual_R']),'MLP skip/add BF16 bridge failed'
  active.pop()
 def attention_pre(m,args,kw):assert active and not attn;attn.append(active[-1])
 def attention_post(m,args,kw,result):assert attn.pop()==active[-1]
 def a_capture(m,args,result):
  if active==[1]:
   assert 'a_actual_R' not in records;records['a_actual_R']=result[:,R,:].detach().clone()
 def b_capture(m,args,result):
  if active==[1]:records['b_actual_R']=result[:,R,:].detach().clone()
 def postadd_alias(m,args,kw):
  if active!=[1]:return
  y=args[0] if args else kw['hidden_states'];assert y.shape==(1,sequence_length,2048) and y.dtype in [torch.bfloat16,torch.float32];mask=torch.arange(sequence_length,device=y.device)!=R;x=records['x_actual_R'];a=records['a_actual_R'];actual=y[:,R,:].detach().clone();assert bits(actual,x+a),'Native x+a bridge failed';xb=x if x_branch is None else explicit_row(x_branch,x);ab=a if a_branch is None else explicit_row(a_branch,a);used=xb+ab;assert used.dtype==y.dtype;records['y_actual_before_branch_R']=actual;records['x_branch_used_R']=xb.detach().clone();records['a_branch_used_R']=ab.detach().clone();records['y_branch_R']=used.detach().clone();nonR=thash(y[:,mask,:]);records['postadd_argument_object_id']=id(y);records['postadd_argument_data_ptr']=y.data_ptr()
  if x_branch is not None or a_branch is not None or rewrite_identity:y[:,R,:].copy_(used)
  assert bits(y[:,R,:],used) and thash(y[:,mask,:])==nonR;records['MLP_input_R']=y[:,R,:].detach().clone();records['MLP_skip_shared_alias_R']=y[:,R,:].detach().clone();records['postadd_non_R_unchanged']=True;records['inplace_preserves_outer_residual_alias']=True;records['MLP_skip_provenance']='Source-locked shared y object at modeling_ouro.py418–419; independently checked by explicit layer replay, not recovered by output subtraction'
  # Deliberately return None. A cloned returned argument would leave the outer MLP skip unchanged.
 def finalnorm(m,args,result):
  norm_events.append(len(norm_events))
  if len(norm_events)==2:
   records['final_norm_R']=result[:,R,:].detach().clone();mask=torch.arange(sequence_length,device=result.device)!=R;records['non_R_final_norm_sha256']=thash(result[:,mask,:])
 def sdpa(*args,**kw):
  if active!=[1] or attn!=[1]:return old(*args,**kw)
  assert 'O_actual_R' not in records;query,key,value=args[:3];assert all(t.ndim==4 and t.shape[-2]==sequence_length for t in [query,key,value]);mask=torch.arange(sequence_length,device=query.device)!=R
  for name,v in [('Q',query),('K',key),('V',value)]:records[name+'_actual_R']=v[...,R,:].detach().clone();records['non_R_'+name+'_sha256']=thash(v[...,mask,:])
  result=old(*args,**kw);records['O_actual_R']=result[...,R,:].detach().clone();records['non_R_O_sha256']=thash(result[...,mask,:]);records['sdpa_shapes_strides_dtypes']={n:{'shape':list(v.shape),'stride':list(v.stride()),'dtype':str(v.dtype)} for n,v in [('Q',query),('K',key),('V',value)]};records['sdpa_tail']=[description(v) for v in args[3:]];records['sdpa_kwargs']={k:description(v) for k,v in kw.items()}
  if capture_full_qkv:records['technical_only_full_qkv']=(query.detach().clone(),key.detach().clone(),value.detach().clone(),args[3:],kw.copy())
  return result
 try:
  handles=[layer.register_forward_pre_hook(pre,with_kwargs=True),layer.register_forward_hook(post,with_kwargs=True),layer.self_attn.register_forward_pre_hook(attention_pre,with_kwargs=True),layer.self_attn.register_forward_hook(attention_post,with_kwargs=True),layer.input_layernorm_2.register_forward_hook(a_capture),layer.post_attention_layernorm.register_forward_pre_hook(postadd_alias,with_kwargs=True),layer.post_attention_layernorm_2.register_forward_hook(b_capture),model.model.norm.register_forward_hook(finalnorm)];F.scaled_dot_product_attention=sdpa
  yield records
  assert events==[0,1] and not active and not attn and len(norm_events)==2 and all(k in records for k in ['x_actual_R','a_actual_R','O_actual_R','y_branch_R','after_mlp_R','final_norm_R']);records['loop_events']=events
 finally:
  F.scaled_dot_product_attention=old
  for h in handles:h.remove()

class UndefinedControl(RuntimeError):pass

@contextmanager
def q9_region(model,*,R,prefix_len,sequence_length,perturbed=False,expected_donor=None):
 """Same-run nativeK1 layer9 preinput Q donor; clone only Q, never archive full sequence."""
 assert 0<prefix_len<R==sequence_length-1;layer=model.model.layers[8];events=[];record={}
 def pre(m,args,kw):
  ut=kw['current_ut'];assert ut==len(events) and ut in [0,1] and kw['use_cache'] is False and kw.get('past_key_value') is None;events.append(ut);x=args[0] if args else kw['hidden_states'];assert x.shape==(1,sequence_length,2048)
  if ut==0:
   record['K1_Q']=x[:,prefix_len:R,:].detach().clone()
   if expected_donor is not None:assert bits(record['K1_Q'],expected_donor.to(x.device))
  else:
   outside=torch.ones(sequence_length,device=x.device,dtype=torch.bool);outside[prefix_len:R]=False;record['receiver_Q_sha256']=thash(x[:,prefix_len:R,:]);record['outside_Q_sha256']=thash(x[:,outside,:]);record['receiver_R']=x[:,R,:].detach().clone()
   if perturbed:
    changed=x.clone();changed[:,prefix_len:R,:]=record['K1_Q'];assert bits(changed[:,outside,:],x[:,outside,:]);record['applied_Q_sha256']=thash(changed[:,prefix_len:R,:]);record['outside_Q_unchanged']=True
    if args:return (changed,*args[1:]),kw
    kw=kw.copy();kw['hidden_states']=changed;return args,kw
 try:
  handle=layer.register_forward_pre_hook(pre,with_kwargs=True);yield record;assert events==[0,1] and 'K1_Q' in record
 finally:handle.remove()
