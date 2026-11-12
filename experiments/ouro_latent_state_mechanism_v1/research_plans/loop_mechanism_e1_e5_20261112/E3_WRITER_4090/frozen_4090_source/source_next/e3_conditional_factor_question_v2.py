"""E3 Q→R interval: 14 treatment forwards, after N/P references and donor freeze."""
import pathlib,sys,hashlib,torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
from e3_qkey_producer_observer import producer_observer,bits
from e3_conditional_factor_hook import intervene_attention_factor,UndefinedNormControl
if not __debug__:raise RuntimeError('optimized Python forbidden')
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def score(z,targets):return [float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=z.device)).sum()) for t in targets]
@torch.inference_mode()
def run_treatments(model,*,ident,ids,R,prefix_len,targets,ref,unrelated,mapping,guard):
 """ref contains hidden1,clean_cut,clean_k,deltas, and background full target/QKV."""
 assert not model.training and ids.ndim==2 and ids.shape[0]==1 and R==ids.shape[1]-1 and 0<prefix_len<R and set(ref['deltas'])=={(i,k) for i in range(8,16) for k in ('a','b')} and set(unrelated)==set(ref['deltas']);assert set(ref['backgrounds'])=={'N','P'}
 for i in range(8,16):
  for branch in ('a','b'):
   v=mapping[f'{i:02d}_{branch}'];assert v['reference_sha256']==thash(ref['deltas'][(i,branch)]);u=unrelated[(i,branch)];assert (u is None and v['status']=='UNDEFINED_NO_NONZERO_DONOR') or (u is not None and v['status'] in ('OK','ZERO_REFERENCE_IDENTITY') and thash(u)==v['unrelated_sha256'])
 saved={};result={};core=model.model;oldsteps=core.total_ut_steps;core.total_ut_steps=2
 try:
  for factor in ('E','L'):
   branch='a';indices=(8,) if factor=='E' else tuple(range(9,16))
   for background,kind in (('P','plus'),('N','minus')):
    for variant in ('main','norm','unrelated'):
     name=f'{background}_{kind}_{factor}_{variant}';isP=background=='P';chosen={i:(ref['deltas'][(i,branch)] if variant!='unrelated' else unrelated[(i,branch)]) for i in indices}
     if any(x is None for x in chosen.values()):result[name]={'status':'UNDEFINED_NO_NONZERO_DONOR','undefined_sites':[i for i,x in chosen.items() if x is None]};continue
     z,rec,patches,extra=one(model,ids,R,prefix_len,targets,ref,chosen,background,branch=branch,factor=factor,sign=1 if isP else -1,norm=variant=='norm',restore_key=False,guard=guard)
     record(name,z,rec,patches,extra,saved,result,targets,mapping)
  assert len(result)==12;return result,saved
 finally:core.total_ut_steps=oldsteps

def one(model,ids,R,P,targets,ref,chosen,background,*,branch,factor,sign,norm,restore_key,guard):
 guard();core=model.model;roots=[];target_events=[];target_active=[];target_a=[];cut=model.model.layers[17]
 def root(m,args,kw):
  x=args[0] if args else kw.get('input_ids');assert torch.is_tensor(x) and bits(x,ids) and kw.get('labels') is None and kw.get('use_cache') is False and kw.get('past_key_values') is None;roots.append(1)
 def target(m,args,kw):
  if kw.get('current_ut')==1:
   x=args[0] if args else kw.get('hidden_states');assert not target_events;target_events.append(x.detach().clone());target_active.append(1)
 def post(m,args,kw,out):
  if kw.get('current_ut')==1:assert target_active.pop()==1
 def anorm(m,args,out):
  if target_active:assert not target_a;target_a.append(out[:,R,:].detach().clone())
 l9post=[]
 def l9out(mod,args,kw,out):
  if kw.get('current_ut')==1:
   y=out[0] if isinstance(out,(tuple,list)) else out;assert torch.is_tensor(y) and y.ndim==3;l9post.append(y[:,R,:].detach().clone())
 lh=core.layers[8].register_forward_hook(l9out,with_kwargs=True);rh=model.register_forward_pre_hook(root,with_kwargs=True);th=cut.register_forward_pre_hook(target,with_kwargs=True);ph=cut.register_forward_hook(post,with_kwargs=True);ah=cut.input_layernorm_2.register_forward_hook(anorm);patches={};rec={};z=None
 try:
  try:
   with producer_observer(model,8,R,P,hidden_donor=ref['hidden1'] if background=='P' else None,key_donor=ref['clean_k'] if restore_key else None,expected_clean_cut=ref['clean_cut']) as rec,intervene_attention_factor(model,R,expected_length=ids.shape[1],factor=factor,reference_deltas=chosen,sign=sign,norm_preserve=norm) as patches:
    y=model(ids,labels=None,use_cache=False,exit_at_step=1);z=y.logits[:,R,:].detach()
  except UndefinedNormControl as e:return None,{},patches,{'status':'UNDEFINED_ZERO_PROPOSAL','detail':str(e)}
 finally:lh.remove();rh.remove();th.remove();ph.remove();ah.remove()
 assert len(roots)==1 and len(target_events)==len(target_a)==len(l9post)==1 and not target_active and len(patches)==(1 if factor=='E' else 7) and z.dtype==torch.bfloat16 and z.shape==(1,49152) and torch.isfinite(z).all();base=ref['backgrounds'][background];mask=torch.arange(ids.shape[1],device=ids.device)!=R;assert bits(target_events[0][:,mask,:],base['target_full'][:,mask,:]);
 for key in ('q','k','v'):assert bits(rec[key][...,mask,:],base[key][...,mask,:])
 assert bits(rec['k'][...,P:R,:],base['k'][...,P:R,:]) and rec['outside_R_unchanged'];
 if background=='P':assert rec['cut_outside_Q_unchanged'] and rec['cut_donor_Q_equal']
 if restore_key:assert rec['donor_K_Q_sha256']==thash(ref['clean_k'][...,P:R,:]) and rec['mixed_K_Q_sha256']==rec['donor_K_Q_sha256']
 for v in patches.values():assert v['outside_R_unchanged'] and bits((v['after'].double()-v['effective_delta']).to(torch.bfloat16),v['before'])
 rec['target_a_R']=target_a[0];rec['l9_post_mlp_R']=l9post[0];return z,rec,patches,{'status':'OK','target_nonR_matches_background':True,'qkv_nonR_matches_background':True,'receiver_K_Q_matches_background':True,'key_edge_restored':restore_key}

def record(name,z,rec,patches,extra,saved,result,targets,mapping):
 if z is None:result[name]=extra;return
 entry={**extra,'scores':score(z,targets),'candidate_logits':[float(z[0,t]) for t in targets],'logits_sha256':thash(z),'target_x_R_sha256':thash(rec['receiver_x_R']),'sdpa_R_sha256':thash(rec['injected_sdpa_R']),'sites':{}}
 saved[name+'_logits']=z.cpu().contiguous();saved[name+'_target_x_R']=rec['receiver_x_R'].cpu().contiguous();saved[name+'_sdpa_R']=rec['injected_sdpa_R'].cpu().contiguous();saved[name+'_target_a_R']=rec['target_a_R'].cpu().contiguous();saved[name+'_l9_post_mlp_R']=rec['l9_post_mlp_R'].cpu().contiguous();entry['target_a_R_sha256']=thash(rec['target_a_R'])
 for i,v in patches.items():
  row={}
  for key in ('before','after','reference_delta','effective_delta','cast_residual','exact'):
   t=v[key].cpu().contiguous();saved[f'{name}_{i:02d}_{key}']=t;row[key+'_sha256']=thash(t)
  for key in ('reference_dose','effective_dose','before_norm','after_norm','outside_R_unchanged'):row[key]=v[key]
  if 'unrelated' in name:
   branch='a';row['frozen_unrelated_source']=mapping[f'{i:02d}_{branch}']
  entry['sites'][str(i)]=row
 result[name]=entry
