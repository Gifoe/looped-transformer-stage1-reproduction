"""CATC: original SDPA, patch ONLY K2 R at actual o_proj input. All weights frozen."""
import pathlib,sys,json,hashlib,time,contextlib
import torch
ROOT=pathlib.Path('/root/ouro-catc-sol-20261116');MODEL='/root/autodl-tmp/ouro-e4-e5/model'
from ab1_loader_draft import load_model

def sha(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def save(name,obj):
 p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2));t.replace(p)
def score(z,targets):
 v=z.reshape(-1)[targets].float();lp=z.reshape(-1).float().log_softmax(-1);ties=torch.nonzero(v==v.max()).flatten().tolist();return {'candidate_logits':v.tolist(),'scores':lp[targets].tolist(),'prediction':ties[0],'ties':ties,'logits_sha256':sha(z)}
def initialize():
 torch.set_num_threads(4);torch.manual_seed(20261116)
 m,receipt=load_model(MODEL,authorized=True,device='cuda:0');m.model.total_ut_steps=4
 for p in m.parameters():p.requires_grad_(False)
 assert all(not p.requires_grad for p in m.parameters());receipt['total_ut_steps']=4;save('SOURCE_RUNTIME.json',receipt);return m
@contextlib.contextmanager
def control(model,q,av=0.,ap=0.,kind='structured',audit=False):
 sites=tuple(range(1,8)) if kind=='unrelated' else tuple(range(9,16));P=q['prefix_len'];R=q['R'];T=R+1;Qlen=R-P;assert P>=Qlen>0
 lo,hi=(P-Qlen,P) if kind=='prefix' else (P,R)
 rec={'sites':{},'events':[],'audit':[],'norms':[],'witness':{}};active=[];k1={};pending={};handles=[];old=torch.nn.functional.scaled_dot_product_attention
 g=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(('CATC-random-v1|'+q['id']).encode()).digest()[:8],'big')&((1<<63)-1));perm=torch.randperm(2048,generator=g).cuda()
 def pre(i):
  def f(m,args,kw):
   ut=kw['current_ut'];assert kw.get('use_cache') is False and kw.get('past_key_value') is None;active.append((i,ut));rec['events'].append([i,ut])
  return f
 def post(i):
  def f(m,args,kw,out):assert active.pop()==(i,kw['current_ut'])
  return f
 def sdpa(*args,**kw):
  out=old(*args,**kw)
  if not active:return out
  i,ut=active[-1]
  if i not in sites or ut not in (0,1) or (av==ap==0 and not audit):return out
  query,key,val=args[:3];assert kw['is_causal'] is True and kw.get('attn_mask') is None and query.shape[-2]==key.shape[-2]==val.shape[-2]==T
  assert query.shape[1]%key.shape[1]==0
  if key.shape[1]!=query.shape[1]:
   n=query.shape[1]//key.shape[1];key=key.repeat_interleave(n,dim=1);val=val.repeat_interleave(n,dim=1)
  qr=query[:,:,R:R+1,:].double();kk=key.double();vv=val[:,:,lo:hi].double();probs=((qr@kk.transpose(-1,-2))*kw.get('scale',query.shape[-1]**-.5)).softmax(-1)[:,:,0,:]
  pp=probs[:,:,lo:hi];actual=out[:,:,R,:].detach().clone()
  if ut==0:
   k1[i]=(pp,vv)
   if audit:rec.setdefault('k1_full_qkv',{})[i]={'q_R':query[:,:,R:R+1].cpu(),'key':key.cpu(),'value':val.cpu()}
   return out
  p1,v1=k1.pop(i);dv=(pp.unsqueeze(-1)*(vv-v1)).sum(-2);dp=((pp-p1).unsqueeze(-1)*v1).sum(-2);pending[i]=(dv,dp)
  if audit:
   d=(pp.unsqueeze(-1)*vv).sum(-2)-(p1.unsqueeze(-1)*v1).sum(-2);err=(dv+dp-d).norm()/d.norm().clamp_min(1e-30);full=(probs.unsqueeze(-1)*val.double()).sum(-2);sdpaerr=(full-actual.double()).norm()/full.norm().clamp_min(1e-30)
   assert float(err)<1e-10 and float(sdpaerr)<.01
   rec['audit'].append({'layer':i+1,'decomposition_relative_error':float(err),'reconstructed_full_vs_SDPA_relative_error':float(sdpaerr),'heads':query.shape[1],'source_span':[lo,hi],'R':R,'causal':True})
   rec['witness'][i]={'k1_qkv':rec['k1_full_qkv'][i],'k2_qkv':{'q_R':query[:,:,R:R+1].cpu(),'key':key.cpu(),'value':val.cpu()},'p1':p1.cpu(),'v1':v1.cpu(),'p2':pp.cpu(),'v2':vv.cpu(),'dv':dv.cpu(),'dp':dp.cpu(),'actual_sdpa_R':actual.cpu()}
  return out
 def project(i):
  def f(m,args):
   if active[-1]!=(i,1) or i not in sites:return
   if av==ap==0:return # EXACT zero bypass: no reconstructed vector ever written.
   dv,dp=pending.pop(i);x=args[0];before=x[:,R].detach().clone();delta=(av*dv+ap*dp).reshape(1,2048)
   assert delta.numel()==before.numel()
   if kind=='random':delta=delta[:,perm]
   if kind=='scaling':delta=(1 if av+ap>=0 else -1)*before.double()*delta.norm()/before.double().norm().clamp_min(1e-30) # positive same-norm unstructured scaling
   requested=delta.norm();cap=.10*before.double().norm();scale=torch.clamp(cap/requested.clamp_min(1e-30),max=1.);delta=delta*scale;after=(before.double()+delta).to(before.dtype)
   while bool((after.double()-before.double()).norm()>cap):
    delta=delta*.99;scale=scale*.99;after=(before.double()+delta).to(before.dtype)
   assert bool((after.double()-before.double()).norm()<=cap)
   z=x.clone();z[:,R]=after
   assert torch.equal(z[:,:R],x[:,:R]);rec['norms'].append(torch.stack([requested,cap,scale,(after.double()-before.double()).norm(),before.double().norm()]).detach());rec['sites'][i]=sha(after)
   return (z,*args[1:])
  return f
 try:
  for i,l in enumerate(model.model.layers):
   handles.extend([l.register_forward_pre_hook(pre(i),with_kwargs=True),l.register_forward_hook(post(i),with_kwargs=True)])
   if i in sites:handles.append(l.self_attn.o_proj.register_forward_pre_hook(project(i)))
  torch.nn.functional.scaled_dot_product_attention=sdpa
  yield rec
  assert rec['events']==[[i,u] for u in range(4) for i in range(24)] and not active
  assert len(rec['norms'])==(0 if av==ap==0 else 7)
 finally:
  torch.nn.functional.scaled_dot_product_attention=old
  for h in handles:h.remove()
@torch.inference_mode()
def forward(model,q,av=0.,ap=0.,kind='structured',audit=False):
 assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256'];ids=torch.tensor([q['ids']],device='cuda');start=time.monotonic();torch.cuda.reset_peak_memory_stats()
 with control(model,q,av,ap,kind,audit) as rec:_,states,gates=model.model(ids,use_cache=False,past_key_values=None)
 depths=range(4) if av==ap==0 else (1,3);z={i+1:model.lm_head(states[i])[:,q['R']].detach().clone() for i in depths};torch.cuda.synchronize()
 r={'seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'norms':torch.stack(rec['norms']).cpu().tolist() if rec['norms'] else [],'scores':{str(i):score(v,q['targets']) for i,v in z.items()},'directly_changed_K2_R_sites':rec['sites']}
 if av==ap==0:
  zz=[z[i].float().reshape(-1) for i in range(1,5)];methods={'logit_mean':torch.stack(zz).mean(0),'LoopCD_logits_0.25':zz[-1]+.25*(zz[-1]-zz[0]),'LoopCD_logits_1':zz[-1]+(zz[-1]-zz[0])};lp=torch.stack([v.log_softmax(-1) for v in zz]);methods['probability_mean']=lp.exp().mean(0).log();confidence=[float(v[q['targets']].softmax(-1).max()) for v in zz];methods['max_candidate_confidence']=zz[max(range(4),key=lambda i:confidence[i])]
  remaining=torch.ones_like(gates[0].squeeze(-1));weighted=None;pdfs=[]
  for i in range(4):
   lam=torch.sigmoid(gates[i].squeeze(-1));pdf=remaining if i==3 else lam*remaining
   if i<3:remaining=remaining*(1-lam)
   pdfs.append(float(pdf[:,q['R']]));term=z[i+1]*pdf[:,q['R']].unsqueeze(-1).to(z[i+1].dtype);weighted=term if weighted is None else weighted+term
  r['weighted4_pdf_R']=pdfs;methods['weighted4']=weighted.reshape(-1);r['output_baselines']={n:score(v,q['targets']) for n,v in methods.items()}
 return r,{i:v.cpu() for i,v in z.items()},rec
