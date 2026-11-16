"""Ideal FP64 paired-message delta on native BF16 SDPA base; NOT native KV-recompute."""
import pathlib,json,hashlib,time,contextlib,torch
from record_operator import ROOT,RAW,LAYERS,initialize,bits,digest,clone
from matched_gaussian import apply as gaussian
@contextlib.contextmanager
def edit(m,q,kind='native',arm=2,native_prefix=None,capture_prefix=False,witness=False,target_doses=None):
 assert kind in ['native','noop','self','semantic','V','joint','random'] and arm in [1,2];a,b=q['record_spans'][0];c,d=q['record_spans'][1];h=q['H_start'];T=q['R']+1;assert 0<=a<b==c<d==h<=q['R'] and b-a==d-c;active=[];visits=[];events=[];handles=[];pref={};w={};pending={};norms=[];audit={'kind':kind,'arm':arm,'events':events,'actual_original_SDPA_invocations':0,'explicit_FP64_attention_calls':0,'prefix_identity_sites':0,'unchanged_nonH_checks':0};original=torch.nn.functional.scaled_dot_product_attention
 def check(name,x):
  z=digest(x)
  if capture_prefix:pref[name]=z
  if native_prefix is not None:assert z==native_prefix[name],('prefixchanged',name);audit['prefix_identity_sites']+=1
 def pre(i):
  def hook(mod,args,kw):
   u=kw['current_ut'];assert not kw['use_cache'] and kw.get('past_key_value') is None;active.append((u,i));visits.append([u,i]);x=args[0] if args else kw['hidden_states'];check('input_'+str((u,i)),x[:,:h])
  return hook
 def post(i):
  def hook(mod,args,kw,out):assert active.pop()==(kw['current_ut'],i)
  return hook
 def norm(mod,args,out):u=len(norms);norms.append(1);check('norm_'+str(u),out[:,:h])
 def projection(i):
  def hook(mod,args,out):
   if not active:return
   u,k=active[-1]
   if (u,k) not in pending:return
   oldH=pending.pop((u,k));base=clone(args[0]);base[:,h:]=oldH;reference=torch.nn.functional.linear(base,mod.weight,mod.bias);delta=out[:,h:].double()-reference[:,h:].double();e=events[-1];assert e['loop']==u+1 and e['layer']==i+1;e['projected_actual_delta_norms']=delta[0].norm(dim=-1).cpu().tolist()
   if witness:w[(u,k)]['actual_projected_H']=out[:,h:].cpu();w[(u,k)]['native_base_projected_H']=reference[:,h:].cpu()
  return hook
 def sdpa(*args,**kw):
  out=original(*args,**kw);audit['actual_original_SDPA_invocations']+=1
  assert active;u,i=active[-1];query,key,value=args[:3];assert query.shape[-2]==key.shape[-2]==value.shape[-2]==T;assert kw.get('is_causal') is True and kw.get('attn_mask') is None and kw.get('dropout_p',0)==0;assert query.shape[1]==key.shape[1]==value.shape[1]
  check('K_'+str((u,i)),key[:,:,a:h]);check('V_'+str((u,i)),value[:,:,a:h]);check('attention_'+str((u,i)),out[:,:,:h])
  if kind in ['native','noop'] or u!=arm-1 or i not in LAYERS:return out
  scale=kw.get('scale') or query.shape[-1]**-.5;qh=query[:,:,h:].double();kd=key.double();vd=value.double();score=(qh@kd.transpose(-1,-2))*scale;future=torch.arange(T,device=out.device).unsqueeze(0)>torch.arange(h,T,device=out.device).unsqueeze(1);score.masked_fill_(future,-torch.inf);p=score.softmax(-1);reference=p@vd;delta=(p[:,:,:,a:b]-p[:,:,:,c:d])@(vd[:,:,c:d]-vd[:,:,a:b]);audit['explicit_FP64_attention_calls']+=1
  before=out[:,:,h:];flat=lambda z:z.permute(0,2,1,3).reshape(T-h,-1);dose=None
  if kind in ['self','joint']:requested=torch.zeros_like(delta);after=before.clone()
  elif kind=='random':
   assert target_doses is not None and str(i+1) in target_doses;seed=int.from_bytes(hashlib.sha256(('paired-gaussian-v2|'+q['id']+'|'+str(arm)+'|'+str(i)).encode()).digest()[:8],'big')&((1<<63)-1);rows,dose=gaussian(flat(before),target_doses[str(i+1)],seed);after=rows.reshape(1,T-h,before.shape[1],before.shape[-1]).permute(0,2,1,3);requested=None
  else:requested=delta;after=(before.double()+delta).to(before.dtype)
  z=out.clone();z[:,:,h:]=after;assert bits(z[:,:,:h],out[:,:,:h]);audit['unchanged_nonH_checks']+=1;actual=after.double()-before.double();error=reference-before.double();entry={'loop':u+1,'layer':i+1,'receiver_span':[h,T],'record_spans':[[a,b],[c,d]],'same_layer_current_QKV':True,'all_heads':True,'requested_paired_delta_norms':flat(delta).norm(dim=-1).cpu().tolist(),'actual_delta_norms':flat(actual).norm(dim=-1).cpu().tolist(),'native_norms':flat(before.double()).norm(dim=-1).cpu().tolist(),'FP64_native_reconstruction_max_abs':float(error.abs().max()),'FP64_native_reconstruction_rms':float(error.square().mean().sqrt()),'FP64_native_reconstruction_relative_l2':float(error.norm()/reference.norm().clamp_min(1e-30)),'dose':dose};events.append(entry)
  pending[(u,i)]=out.permute(0,2,1,3).reshape(1,T,-1)[:,h:].detach().clone()
  if witness:
   w[(u,i)]={'Q_H':query[:,:,h:].cpu(),'K_full':key.cpu(),'V_full':value.cpu(),'native_SDPA_H':before.cpu(),'edited_SDPA_H':after.cpu(),'paired_delta_FP64':delta.cpu(),'record_spans':[[a,b],[c,d]],'H_start':h,'scale':scale,'kind':kind}
  return z
 try:
  for i,l in enumerate(m.model.layers):handles.extend([l.register_forward_pre_hook(pre(i),with_kwargs=True),l.register_forward_hook(post(i),with_kwargs=True),l.self_attn.o_proj.register_forward_hook(projection(i))])
  handles.append(m.model.norm.register_forward_hook(norm));torch.nn.functional.scaled_dot_product_attention=sdpa;yield audit,pref,w
  assert visits==[[u,i] for u in [0,1] for i in range(24)] and audit['actual_original_SDPA_invocations']==48 and len(norms)==2 and not active and not pending;expected=[] if kind in ['native','noop'] else [[arm,i+1] for i in LAYERS];assert [[e['loop'],e['layer']] for e in events]==expected;audit['receiver_row_writebacks']=len(expected)*(T-h)
 finally:
  torch.nn.functional.scaled_dot_product_attention=original
  for handle in handles:handle.remove()
@torch.inference_mode()
def run(m,q,kind='native',arm=2,native_prefix=None,capture_prefix=False,witness=False,target_doses=None):
 assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256'];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
 with edit(m,q,kind,arm,native_prefix,capture_prefix,witness,target_doses) as (audit,pref,w):_,states,_=m.model(torch.tensor([q['ids']],device='cuda'),use_cache=False,past_key_values=None)
 heads={d+1:m.lm_head(states[d])[:,q['R']].detach().clone() for d in [0,1]};torch.cuda.synchronize();elapsed=time.monotonic()-t;scores={}
 for d,z in heads.items():
  x=z.flatten()[q['targets']].float();lp=z.flatten().float().log_softmax(-1)[q['targets']];ties=torch.nonzero(x==x.max()).flatten().tolist();scores[str(d)]={'candidate_logits':x.tolist(),'candidate_logprobs':lp.tolist(),'prediction':ties[0],'ties':ties,'full49152_sha256':digest(z)}
 return {'scores':scores,'audit':audit,'seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated()},heads,pref,w
