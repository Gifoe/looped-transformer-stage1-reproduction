"""AB1-derived same-receiver bridge technical candidate; not a scientific order.

No CLI, checkpoint access, source selection or launch side effects. The caller
must bind the same-site discovery source, IDs, q, checkpoint and all receipts.
Seven conditions are technical wiring checks only; no matched controls or
scientific confirmation are claimed. No labels enter this module.
"""
import hashlib,torch
from contextlib import contextmanager
from ab1_boundary_hooks import capture_and_patch,bits
from ab1_boundary_core import corner
from branch_interfaces_v2 import attention_reference_operator,actual_L18_branch_interface,UndefinedControl
from ab1_metrics import score

if not __debug__:raise RuntimeError('optimized Python forbidden')

def th(x):return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def immutable_versions(model):return tuple((n,id(x),x.data_ptr(),x._version) for n,x in list(model.named_parameters())+list(model.named_buffers()))
@contextmanager
def trace_nonr_layers(model,R):
 events={i:[] for i in range(9,16)};records={};handles=[]
 def before(i):
  def fn(m,args,kw):
   ut=kw['current_ut'];assert ut==len(events[i]) and ut in (0,1) and kw['use_cache'] is False and kw.get('past_key_value') is None;events[i].append(ut)
   if ut==1:
    x=args[0] if args else kw['hidden_states'];assert x.ndim==3 and x.shape[0]==1 and x.shape[1]==R+1;records[i]={'input_nonR_sha256':th(x[:,:R,:])}
  return fn
 def after(i):
  def fn(m,args,kw,out):
   if kw['current_ut']==1:
    assert torch.is_tensor(out) and out.ndim==3 and out.shape[1]==R+1;records[i]['output_nonR_sha256']=th(out[:,:R,:])
  return fn
 try:
  for i in range(9,16):
   handles.append(model.model.layers[i].register_forward_pre_hook(before(i),with_kwargs=True));handles.append(model.model.layers[i].register_forward_hook(after(i),with_kwargs=True))
  yield records
  assert all(v==[0,1] for v in events.values()) and all(set(records[i])=={'input_nonR_sha256','output_nonR_sha256'} for i in records) and set(records)==set(range(9,16))
 finally:
  for h in handles:h.remove()
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
    if float(n1)==0 and float(n0)!=0:
     e=UndefinedControl('zero upstream proposal norm with nonzero receiver norm')
     e.witness={'component':'attention','site':i,'sign':sign,'norm_preserve':True,'before':before.detach().clone(),'proposal_before_norm':exact.detach().clone(),'reference_delta':delta.detach().clone()}
     raise e
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
def mlp_reference_operator(model,R,*,sequence_length,reference_delta=None,sign=-1,norm_preserve=False):
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
    if float(n1)==0 and float(n0)!=0:
     e=UndefinedControl('zero upstream proposal norm with nonzero receiver norm')
     e.witness={'component':'mlp','site':i,'sign':sign,'norm_preserve':True,'before':before.detach().clone(),'proposal_before_norm':exact.detach().clone(),'reference_delta':delta.detach().clone()}
     raise e
    if float(n1)!=0:scale=n0/n1;exact=exact*scale
   after=exact.to(result.dtype);assert torch.isfinite(after).all();changed=result.clone();changed[:,R,:]=after;mask=torch.arange(sequence_length,device=result.device)!=R;assert bits(changed[:,mask,:],result[:,mask,:]);records[i]={'before':before,'after':after.detach().clone(),'reference_delta':delta.detach().clone(),'effective_delta':after.double()-before.double(),'exact_before_cast':exact.detach().clone(),'normalization_scale_fp64':scale.detach().clone(),'cast_residual':after.double()-exact,'non_R_unchanged':True};return changed
  return fn
 try:
  for i in indices:
   l=model.model.layers[i];handles.extend([l.register_forward_pre_hook(pre(i),with_kwargs=True),l.post_attention_layernorm_2.register_forward_hook(patch(i)),l.register_forward_hook(post(i),with_kwargs=True)])
  yield records
  assert not active and all(v==[0,1] for v in events.values()) and set(records)==set(indices)
 finally:
  for h in handles:h.remove()

def complete_capture_identity(records):
 """Derived receipts only; never modify hooked forward tensors or patched records."""
 for r in records.values():
  if not r.get('capture_only',False):continue
  before=r['before'];after=r['after'];assert bits(before,after)
  derived={'exact_before_cast':before.double(),'normalization_scale_fp64':torch.ones((),dtype=torch.float64,device=before.device),'cast_residual':after.double()-before.double(),'effective_delta':after.double()-before.double()}
  assert not any(k in r for k in derived),'capture-only receipt unexpectedly partly populated'
  r.update(derived);r['derived_capture_identity_fields']=True
 return records

@torch.inference_mode()
def forward(model,q,*,boundary=None,delta=None,sign=-1,x_branch=None,a_branch=None,norm_preserve=False,mlp_delta=None,mlp_sign=-1,mlp_norm_preserve=False):
 if set(q)!={'id','split','ids','R','targets','input_ids_sha256','prefix_len'} or q['split'] not in ('train','validation'):raise ValueError('gold-free input required')
 if model.training or model.model.total_ut_steps!=2 or q['R']!=len(q['ids'])-1 or not 0<q['prefix_len']<q['R']:raise ValueError('invalid runtime or regions')
 import json
 assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256']
 ids=torch.tensor([q['ids']],device=next(model.parameters()).device);R=q['R'];P=q['prefix_len'];T=len(q['ids']);guard=immutable_versions(model)
 with capture_and_patch(model,T,R,P,boundary_input=boundary) as bc, attention_reference_operator(model,R,sequence_length=T,reference_delta=delta,sign=sign,norm_preserve=norm_preserve) as sites, mlp_reference_operator(model,R,sequence_length=T,reference_delta=mlp_delta,sign=mlp_sign,norm_preserve=mlp_norm_preserve) as msites, actual_L18_branch_interface(model,R,sequence_length=T,x_branch=x_branch,a_branch=a_branch) as branch, trace_nonr_layers(model,R) as nonr:
  _,states,_=model.model(ids,use_cache=False,past_key_values=None);z=model.lm_head(states[1])[:,R,:].detach().clone()
 assert z.ndim==2 and z.shape[0]==1 and z.dtype==torch.bfloat16 and torch.isfinite(z).all() and immutable_versions(model)==guard
 complete_capture_identity(sites);complete_capture_identity(msites)
 for i,r in msites.items():sites[i].update({"mlp_"+k:v for k,v in r.items()})
 return z,bc,sites,branch,nonr

def _tensor(x):return x.detach().cpu().contiguous().clone()
def _same(a,b):return bits(a.detach().cpu(),b.detach().cpu())
def _expect_ref(z,bc,sites,branch,stored,name):
 assert _same(z,stored[name+'_logits']) and _same(bc['E'],stored['source_E']) and _same(bc['B'],stored['source_B'])
 for i in range(9,16):assert _same(sites[i]['before'],stored[f'{name}_L{i+1:02d}_a_R'])
 for key in ('x','a','y'):
  field={'x':'x_actual_R','a':'a_actual_R','y':'y_branch_R'}[key];assert _same(branch[field],stored[name+'_L18_'+key+'_R'])
 assert _same(bc['Q9_actual'],stored['native_Q9_full' if name=='H11' else 'U_Q9_full'])

def _nonr_equal(a,b):
 for key in ('non_R_x_sha256','non_R_Q_sha256','non_R_K_sha256','non_R_V_sha256','non_R_O_sha256','non_R_after_mlp_sha256','non_R_final_norm_sha256'):assert a[key]==b[key],key

