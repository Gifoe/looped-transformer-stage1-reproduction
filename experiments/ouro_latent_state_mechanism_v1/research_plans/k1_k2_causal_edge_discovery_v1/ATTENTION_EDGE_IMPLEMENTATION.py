"""Source-local post-RoPE K/V counterfactuals of one actual causal SDPA row."""
import pathlib,contextlib,hashlib,time,json,torch
from ab1_loader_draft import load_model
ROOT=pathlib.Path('/root/ouro-causal-edge-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-causal-edge-sol-20261117/raw');LAYERS=(5,9,13)
def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def bits(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def initialize():
 torch.set_num_threads(4);torch.manual_seed(20261117);torch.backends.cuda.matmul.allow_tf32=False;m,v=load_model('/root/autodl-tmp/ouro-e4-e5/model',authorized=True,device='cuda:0');m.model.total_ut_steps=2
 for p in m.parameters():p.requires_grad_(False)
 v.update(total_ut_steps=2);(ROOT/'SOURCE_RUNTIME.json').write_text(json.dumps(v,indent=2));return m
@torch.inference_mode()
def shuffled(before,actual_delta,seed):
 target=actual_delta.double().norm();n=float(target)
 if n==0:return before.clone(),{'target':0.,'actual':0.,'relative_error':0.,'matched':True}
 g=torch.Generator().manual_seed(seed);perm=torch.randperm(before.numel(),generator=g).cuda();d=actual_delta.reshape(-1).double()[perm].reshape_as(before);d/=d.norm();b=before.double();lo=0.;hi=n*2;best=before.clone();error=n;norm=0.;amplitude=0.
 for _ in range(32):
  z=(b+hi*d).to(before.dtype);v=float((z.double()-b).norm())
  if abs(v-n)<error:best=z;error=abs(v-n);norm=v;amplitude=hi
  if v>=n:break
  hi*=2
 for _ in range(32):
  mid=(lo+hi)/2;z=(b+mid*d).to(before.dtype);v=float((z.double()-b).norm())
  if abs(v-n)<error:best=z;error=abs(v-n);norm=v;amplitude=mid
  if v<n:lo=mid
  else:hi=mid
 return best,{'target':n,'actual':norm,'relative_error':error/n,'matched':error<=max(1e-9,.01*n),'amplitude':amplitude}
@contextlib.contextmanager
def context(m,q,operations=(),capture=False,witness=False):
 active=[];events=[];handles=[];cache={};records=[];witnesses={};original=torch.nn.functional.scaled_dot_product_attention;lo,hi=q['source_span']
 def pre(i):
  def h(mod,args,kw):
   assert not kw['use_cache'] and kw.get('past_key_value') is None;active.append((kw['current_ut'],i));events.append([kw['current_ut'],i])
  return h
 def post(i):
  def h(mod,args,kw,out):assert active.pop()==(kw['current_ut'],i)
  return h
 def sdpa(*args,**kw):
  out=original(*args,**kw)
  if not active:return out
  u,l=active[-1]
  if l not in LAYERS:return out
  query,key,value=args[:3];assert kw.get('is_causal') is True and kw.get('attn_mask') is None and kw.get('dropout_p',0)==0;assert query.shape[-2]==key.shape[-2]==value.shape[-2]==q['R']+1;factor=query.shape[1]//key.shape[1];assert query.shape[1]%key.shape[1]==0
  if factor!=1:key=key.repeat_interleave(factor,dim=1);value=value.repeat_interleave(factor,dim=1)
  if capture:
   cache[(u,l,'S')]=(key[:,:,lo:hi].detach().clone(),value[:,:,lo:hi].detach().clone());cache[(u,l,'M')]=(key[:,:,q['M']:q['M']+1].detach().clone(),value[:,:,q['M']:q['M']+1].detach().clone())
  for op in operations:
   if op['pass']!=u or op['layer']!=l:continue
   t=q[op['target']];s0,s1=op.get('span',(lo,hi));assert 0<=s0<s1<=t+1;dk,dv=op['source'];assert dk.shape==key[:,:,s0:s1].shape and dv.shape==value[:,:,s0:s1].shape;kind=op['type'];sameK=bits(dk,key[:,:,s0:s1]);sameV=bits(dv,value[:,:,s0:s1]);before=out[:,:,t].clone()
   if (kind=='V' and sameV) or (kind=='K' and sameK) or (kind=='KV' and sameK and sameV):
    records.append({'pass':u+1,'physical_layer':l+1,'target':t,'type':kind,'null_identity':True,'same_K':sameK,'same_V':sameV,'actual_delta_norm':0.,'requested_delta_norm':0.,'native_output_norm':float(before.double().norm()),'reconstruction_relative_error':0.,'dose':None});continue
   qr=query[:,:,t:t+1].double();k=key[:,:,:t+1].double();v=value[:,:,:t+1].double();scale=kw.get('scale') or query.shape[-1]**-.5;basep=((qr@k.transpose(-1,-2))*scale).softmax(-1);base=(basep@v)[:,:,0];kk=k.clone();vv=v.clone()
   if kind in ['K','KV']:kk[:,:,s0:s1]=dk.double()
   if kind in ['V','KV']:vv[:,:,s0:s1]=dv.double()
   counterp=((qr@kk.transpose(-1,-2))*scale).softmax(-1);counter=(counterp@vv)[:,:,0];err=float((base-before.double()).norm()/base.norm().clamp_min(1e-30));assert err<.01;delta=counter-base;after=(before.double()+delta).to(before.dtype);dose=None
   if op.get('shuffled'):
    seed=int.from_bytes(hashlib.sha256(('edge-dose-v1|'+op['unit_id']+'|'+str(l)+'|'+kind).encode()).digest()[:8],'big')&((1<<63)-1);after,dose=shuffled(before,after.double()-before.double(),seed)
   z=out.clone();z[:,:,t]=after;assert bits(z[:,:,:t],out[:,:,:t]) and bits(z[:,:,t+1:],out[:,:,t+1:]);assert z.data_ptr()!=out.data_ptr();out=z;assert torch.isfinite(out).all();records.append({'pass':u+1,'physical_layer':l+1,'target':t,'source_span':[s0,s1],'type':kind,'same_K':sameK,'same_V':sameV,'null_identity':False,'native_output_norm':float(before.double().norm()),'requested_delta_norm':float(delta.norm()),'actual_delta_norm':float((after.double()-before.double()).norm()),'reconstruction_relative_error':err,'probability_row_L1_change':float((counterp-basep).abs().sum()),'source_probability_mass_before':float(basep[:,:,:,s0:s1].sum()),'source_probability_mass_after':float(counterp[:,:,:,s0:s1].sum()),'dose':dose,'native_receiver_row_sha256':digest(before),'modified_receiver_row_sha256':digest(after),'GQA_expansion':factor})
   if witness:witnesses[(u,l,t)]={'query_t':qr.cpu(),'key_causal_prefix':k.cpu(),'value_causal_prefix':v.cpu(),'donor_key_source':dk.cpu(),'donor_value_source':dv.cpu(),'native_SDPA_row':before.cpu(),'base_reconstructed':base.cpu(),'counter_reconstructed':counter.cpu(),'after_BF16':after.cpu(),'source_span':[s0,s1],'target':t,'type':kind,'scale':scale}
  return out
 try:
  for i,l in enumerate(m.model.layers):handles.extend([l.register_forward_pre_hook(pre(i),with_kwargs=True),l.register_forward_hook(post(i),with_kwargs=True)])
  torch.nn.functional.scaled_dot_product_attention=sdpa;yield cache,records,witnesses
  assert events==[[u,i] for u in range(2) for i in range(24)] and not active and len(records)==len(operations)
 finally:
  torch.nn.functional.scaled_dot_product_attention=original
  for h in handles:h.remove()
@torch.inference_mode()
def forward(m,q,operations=(),capture=False,witness=False):
 assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256'];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();ids=torch.tensor([q['ids']],device='cuda')
 with context(m,q,operations,capture,witness) as (cache,records,witnesses):_,states,_=m.model(ids,use_cache=False,past_key_values=None)
 heads={d+1:m.lm_head(states[d])[:,q['R']].detach().clone() for d in range(2)};torch.cuda.synchronize();seconds=time.monotonic()-t;scores={}
 for d,z in heads.items():
  x=z.reshape(-1)[q['targets']].float();lp=z.float().reshape(-1).log_softmax(-1)[q['targets']];ties=torch.nonzero(x==x.max()).flatten().tolist();scores[str(d)]={'candidate_logits':x.tolist(),'candidate_logprobs':lp.tolist(),'prediction':ties[0],'ties':ties,'full49152_sha256':digest(z)}
 return {'scores':scores,'edits':records,'seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated()},heads,cache,witnesses
