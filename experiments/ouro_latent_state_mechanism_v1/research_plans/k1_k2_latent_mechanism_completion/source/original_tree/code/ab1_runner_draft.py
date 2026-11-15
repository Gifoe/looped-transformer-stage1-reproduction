"""Draft AB1 real-model question runner; no CLI/model loading or launch side effects.

Requires imported, source-audited branch_interfaces_v2 and unchanged Ouro model.
This is NOT technically accepted or authorized for a GPU launch by its presence.
The caller must bind checkpoint/source/runtime/regions, serialize and audit results,
obtain storage budget and complete independent real-model replay before collection.
"""
import hashlib, time, torch
from ab1_boundary_hooks import capture_and_patch,bits
from ab1_conditions import build_boundary_endpoints
from ab1_metrics import score
from branch_interfaces_v2 import attention_reference_operator,actual_L18_branch_interface
if not __debug__:raise RuntimeError('optimized Python forbidden')

def th(x):return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def versions(model):return tuple((n,id(v),v.data_ptr(),v._version) for n,v in list(model.named_parameters())+list(model.named_buffers()))
def _as_cpu(t):return t.detach().cpu().contiguous().clone()
def json_safe(x):
 if torch.is_tensor(x):return {'sha256':th(x),'shape':list(x.shape),'dtype':str(x.dtype)}
 if isinstance(x,dict):return {k:json_safe(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [json_safe(v) for v in x]
 if x is None or isinstance(x,(str,int,float,bool)):return x
 raise TypeError('unexpected metadata type')

@torch.inference_mode()
def collect_case(model,q):
 """Run natural, 12 boundary conditions, 2 Q9 crosses, E-all anchor.

 q must contain *only* id/split/ids/R/targets/input_ids_sha256/prefix_len;
 selected 3 technical cases must additionally run independently implemented
 full-model replay and record masks/position/SDPA with full source receipts.
 """
 allow={'id','split','ids','R','targets','input_ids_sha256','prefix_len'}
 if set(q)!=allow or q['split'] not in ('train','validation'):raise ValueError('gold-free schema')
 import json
 if hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()!=q['input_ids_sha256']:raise ValueError('input hash')
 if model.training or model.model.total_ut_steps!=2 or q['R']!=len(q['ids'])-1:raise ValueError('runtime')
 P=q['prefix_len'];R=q['R'];T=len(q['ids']);ids=torch.tensor([q['ids']],device=next(model.parameters()).device);weights=versions(model)
 def run(boundary=None,donor=None,need_k1=False):
  if ids.device.type=='cuda':torch.cuda.synchronize()
  started=time.perf_counter();roots=[];heads=[];k1_l18=[]
  def before(m,args,kw):
   x=args[0] if args else kw.get('input_ids');assert torch.is_tensor(x) and bits(x,ids) and kw.get('use_cache') is False and kw.get('past_key_values') is None;roots.append(1)
  def head(m,args,out):
   assert tuple(out.shape)==(1,T,49152);heads.append(1)
  def k1_pre(m,args,kw):
   if kw.get('current_ut')==0:
    x=args[0] if args else kw['hidden_states'];k1_l18.append(x[:,R,:].detach().clone())
  h1=model.model.register_forward_pre_hook(before,with_kwargs=True);h2=model.lm_head.register_forward_hook(head);h3=model.model.layers[17].register_forward_pre_hook(k1_pre,with_kwargs=True)
  try:
   with capture_and_patch(model,T,R,P,boundary_input=boundary,q9_donor=donor) as bc,attention_reference_operator(model,R,sequence_length=T) as sites,actual_L18_branch_interface(model,R,sequence_length=T) as branch:
    _,states,_=model.model(ids,use_cache=False,past_key_values=None)
    z=model.lm_head(states[1])[:,R,:].detach().clone();k1=model.lm_head(states[0])[:,R,:].detach().clone() if need_k1 else None
  finally:h1.remove();h2.remove();h3.remove()
  assert len(roots)==1 and len(k1_l18)==1 and len(heads)==(2 if need_k1 else 1) and z.dtype==torch.bfloat16 and torch.isfinite(z).all() and versions(model)==weights
  if ids.device.type=='cuda':torch.cuda.synchronize()
  bc["synchronized_forward_wall_seconds_includes_hooks_head"]=time.perf_counter()-started;bc["K1_L18_x_R"]=k1_l18[0];return z,k1,bc,sites,branch
 _,bare_states,_=model.model(ids,use_cache=False,past_key_values=None);bare_K1=model.lm_head(bare_states[0])[:,R,:].detach().clone();bare_N=model.lm_head(bare_states[1])[:,R,:].detach().clone();assert versions(model)==weights
 N,K1,n,ns,nb=run(need_k1=True);assert bits(K1,bare_K1) and bits(N,bare_N);E=n['E'];B=n['B'];N9=n['Q9_actual'];endpoints=build_boundary_endpoints(E,B,P,R,split=q['split'],ident=q['id']);NQ=N9[:,P:R,:].clone();results={};stored={'source_E':_as_cpu(E),'source_B':_as_cpu(B),'source_K1_L18_x_R':_as_cpu(n['K1_L18_x_R']),'native_K1_logits':_as_cpu(K1),'native_K2_logits':_as_cpu(N),'native_Q9_full':_as_cpu(N9),'native_Q9_Q':_as_cpu(NQ)};UQ=None
 for name,item in endpoints.items():
  if item['status']!='OK':results[name]={'status':item['status'],'receipt':json_safe(item.get('receipt'))};continue
  z,_,bc,sites,branch=run(item['input']);assert bits(bc['E'],E) and bits(bc['B'],B) and bits(bc['K2_applied'],item['input'])
  if name=='H11':assert bits(z,N)
  if name=='H01':UQ=bc['Q9_actual'][:,P:R,:].clone();stored['U_Q9_full']=_as_cpu(bc['Q9_actual']);stored['U_Q9_Q']=_as_cpu(UQ)
  if 'receiver' in item:
   region=slice(P,R) if name.startswith('Q_') else slice(R,R+1);stored[name+'_injected_region']=_as_cpu(item['input'][:,region,:])
  stored[name+'_logits']=_as_cpu(z);stored[name+'_L1_R']=_as_cpu(bc['K2_applied'][:,R,:]);stored[name+'_L9_R']=_as_cpu(bc['Q9_actual'][:,R,:]);stored[name+'_L18_x_R']=_as_cpu(branch['x_actual_R']);stored[name+'_L18_a_R']=_as_cpu(branch['a_actual_R']);stored[name+'_L18_y_R']=_as_cpu(branch['y_branch_R']);stored[name+'_finalnorm_R']=_as_cpu(branch['final_norm_R'])
  for i in range(9,16):stored[f'{name}_L{i+1:02d}_a_R']=_as_cpu(sites[i]['before'])
  results[name]={'status':'OK','boundary_input_sha256':th(item['input']),'full_logits_sha256':th(z),'Q9_Q_sha256':th(bc['Q9_actual'][:,P:R,:]),'scores':score(z,q['targets']),'L18_non_R_hashes':{k:v for k,v in branch.items() if k.startswith('non_R_')},'control_receipt':json_safe(item.get('receipt')),'actual_L1_kwargs':bc['L1_kwargs'],'actual_L9_kwargs':bc['L9_kwargs'],'actual_L18_sdpa_kwargs':branch['sdpa_kwargs'],'actual_L18_sdpa_tail':branch['sdpa_tail'],'actual_L18_sdpa_shapes_strides_dtypes':branch['sdpa_shapes_strides_dtypes'],'synchronized_forward_wall_seconds_includes_hooks_head':bc['synchronized_forward_wall_seconds_includes_hooks_head']}
 assert UQ is not None
 for name,bound,don in [('U_Q9_N',endpoints['H01']['input'],NQ),('N_Q9_U',B,UQ)]:
  z,_,bc,sites,branch=run(bound,don);assert bits(bc['E'],E) and bits(bc['B'],B) and bits(bc['Q9_applied'][:,P:R,:],don);stored[name+'_logits']=_as_cpu(z);stored[name+'_L1_R']=_as_cpu(bc['K2_applied'][:,R,:]);
  for i in range(9,16):stored[f'{name}_L{i+1:02d}_a_R']=_as_cpu(sites[i]['before'])
  stored[name+'_L9_R']=_as_cpu(bc['Q9_actual'][:,R,:]);stored[name+'_L18_x_R']=_as_cpu(branch['x_actual_R']);stored[name+'_L18_a_R']=_as_cpu(branch['a_actual_R']);stored[name+'_L18_y_R']=_as_cpu(branch['y_branch_R']);stored[name+'_finalnorm_R']=_as_cpu(branch['final_norm_R']);results[name]={'status':'OK','full_logits_sha256':th(z),'scores':score(z,q['targets']),'receiver_Q9_R_sha256':th(bc['Q9_actual'][:,R,:]),'receiver_Q9_prefix_sha256':th(bc['Q9_actual'][:,:P,:]),'applied_Q9_R_sha256':th(bc['Q9_applied'][:,R,:]),'applied_Q9_prefix_sha256':th(bc['Q9_applied'][:,:P,:]),'donor_Q9_Q_sha256':th(don),'L18_non_R_hashes':{k:v for k,v in branch.items() if k.startswith('non_R_')},'actual_L1_kwargs':bc['L1_kwargs'],'actual_L9_kwargs':bc['L9_kwargs'],'actual_L18_sdpa_kwargs':branch['sdpa_kwargs'],'actual_L18_sdpa_tail':branch['sdpa_tail'],'actual_L18_sdpa_shapes_strides_dtypes':branch['sdpa_shapes_strides_dtypes'],'synchronized_forward_wall_seconds_includes_hooks_head':bc['synchronized_forward_wall_seconds_includes_hooks_head']}
 z,_,bc,_,_=run(E);assert bits(z,K1) and bits(bc['K2_applied'],E)
 assert len(results)==14 and versions(model)==weights
 return {'identity':{'K1_full_logits_sha256':th(K1),'N_full_logits_sha256':th(N),'K1_scores':score(K1,q['targets']),'N_scores':score(N,q['targets']),'unobserved_vs_observed_native_K1_K2_exact':True,'E_all_equals_K1':True,'B_H11_equals_N':True,'E_sha256':th(E),'B_sha256':th(B),'runtime_torch':torch.__version__},'conditions':results},stored
