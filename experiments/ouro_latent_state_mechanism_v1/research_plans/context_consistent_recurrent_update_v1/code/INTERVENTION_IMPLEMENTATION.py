"""Single natural-delta edit at K2 L9 input. Original model operations stay intact."""
import pathlib,json,hashlib,time,contextlib,torch
from ab1_loader_draft import load_model
ROOT=pathlib.Path('/root/ouro-context-update-sol-20261117')
def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def initialize():
 torch.set_num_threads(4);m,v=load_model('/root/autodl-tmp/ouro-e4-e5/model',authorized=True,device='cuda:0');m.model.total_ut_steps=4
 for p in m.parameters():p.requires_grad_(False)
 v['total_ut_steps']=4;(ROOT/'SOURCE_RUNTIME.json').write_text(json.dumps(v,indent=2));return m
@torch.inference_mode()
def bounded(before,d):
 b=before.double();requested=d.double();norm=requested.norm(dim=-1);base=b.norm(dim=-1);cap=.1*base;scale=(cap/norm.clamp_min(1e-30)).clamp(max=1);delta=requested*scale.unsqueeze(-1);after=(b+delta).to(before.dtype)
 for _ in range(1000):
  actual=(after.double()-b).norm(dim=-1);bad=actual>cap
  if not bool(bad.any()):break
  factor=torch.where(bad,.99,1.);scale*=factor;delta*=factor.unsqueeze(-1);after=(b+delta).to(before.dtype)
 assert bool(((after.double()-b).norm(dim=-1)<=cap).all())
 return after,torch.stack([norm,cap,scale,(after.double()-b).norm(dim=-1),base],dim=-1)
@torch.inference_mode()
def randomized(before,d,target,cap,qid):
 gen=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(('CCRU-random-v1|'+qid).encode()).digest()[:8],'big')&((1<<63)-1));perm=torch.randperm(d.shape[-1],generator=gen).cuda();unit=d[:,perm].double();unit/=unit.norm(dim=-1,keepdim=True).clamp_min(1e-30);b=before.double();lo=torch.zeros_like(target);hi=torch.maximum(target*2,torch.full_like(target,1e-10));best=before.clone();bestnorm=torch.zeros_like(target);besterr=target.clone();amp=torch.zeros_like(target)
 for step in range(48):
  mid=hi if step<8 else (lo+hi)/2;z=(b+mid.unsqueeze(-1)*unit).to(before.dtype);actual=(z.double()-b).norm(dim=-1);err=(actual-target).abs();take=(actual<=cap)&(err<besterr);best=torch.where(take.unsqueeze(-1),z,best);bestnorm=torch.where(take,actual,bestnorm);besterr=torch.where(take,err,besterr);amp=torch.where(take,mid,amp)
  if step<8:hi=torch.where(actual<target,hi*2,hi)
  else:lo=torch.where(actual<target,mid,lo);hi=torch.where(actual>=target,mid,hi)
 assert bool((bestnorm<=cap).all());return best,{'target':target.cpu().tolist(),'actual':bestnorm.cpu().tolist(),'relative_error':(besterr/target.clamp_min(1e-30)).cpu().tolist(),'matched':(besterr<=torch.maximum(target*.01,torch.full_like(target,1e-9))).cpu().tolist(),'amplitude':amp.cpu().tolist()}
@contextlib.contextmanager
def intervention(m,q,region,eta):
 active=[];events=[];handles=[];donor={};observations={};record={'patch_calls':0,'norms':[],'region':region,'eta':eta};P=q['prefix_len'];R=q['R'];lo,hi={'N':(P,R),'Q':(P,R),'Random':(P,R),'P':(0,P),'R':(R,R+1)}[region]
 def pre(i):
  def hook(module,args,kw):
   u=kw['current_ut'];active.append((i,u));events.append([i,u]);assert not kw['use_cache'] and kw['past_key_value'] is None
   if i!=8:return
   x=args[0] if args else kw['hidden_states'];assert x.shape[1]==R+1
   if u==0:
    if eta:donor['state']=x[:,lo:hi].detach().clone()
    return
   if u!=1 or not eta:return
   before=x[:,lo:hi].clone();delta=eta*(before.double()-donor.pop('state').double());after,norms=bounded(before,delta);dose=None
   if region=='Random':after,dose=randomized(before[0],delta[0],norms[0,:,3],norms[0,:,1],q['id']);after=after.unsqueeze(0);norms[:,:,3]=(after.double()-before.double()).norm(dim=-1)
   z=x.clone();z[:,lo:hi]=after;assert torch.equal(z[:,:lo],x[:,:lo]) and torch.equal(z[:,hi:],x[:,hi:]);assert z.data_ptr()!=x.data_ptr();record.update({'patch_calls':record['patch_calls']+1,'span':[lo,hi],'tokens':hi-lo,'norms':norms[0].cpu().tolist(),'receiver_sha256':sha(before),'modified_sha256':sha(after),'dose_matching':dose,'total_displacement_L2':float((after.double()-before.double()).norm())})
   if args:return (z,*args[1:]),kw
   return args,{**kw,'hidden_states':z}
  return hook
 def post(i):
  def hook(module,args,kw,out):assert active.pop()==(i,kw['current_ut'])
  return hook
 def observe(i):
  def hook(module,args,out):
   j,u=active[-1];assert j==i
   if u in [1,2,3]:observations[f'{u+1}:{i+1}']=out[:,R].detach().clone()
  return hook
 try:
  for i,l in enumerate(m.model.layers):
   handles.extend([l.register_forward_pre_hook(pre(i),with_kwargs=True),l.register_forward_hook(post(i),with_kwargs=True)])
   if i in range(9,16):handles.append(l.input_layernorm_2.register_forward_hook(observe(i)))
  yield record,observations
  assert events==[[i,u] for u in range(4) for i in range(24)] and record['patch_calls']==int(bool(eta)) and not active
 finally:
  for h in handles:h.remove()
@torch.inference_mode()
def forward(m,q,region='N',eta=0):
 assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256'];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();ids=torch.tensor([q['ids']],device='cuda')
 with intervention(m,q,region,eta) as (rec,obs):_,states,_=m.model(ids,use_cache=False,past_key_values=None)
 heads={d+1:m.lm_head(states[d])[:,q['R']].detach() for d in range(4)};torch.cuda.synchronize();seconds=time.monotonic()-t;scores={}
 for d,z in heads.items():
  v=z.reshape(-1)[q['targets']].float();lp=z.float().reshape(-1).log_softmax(-1)[q['targets']];ties=torch.nonzero(v==v.max()).flatten().tolist();scores[str(d)]={'candidate_logits':v.tolist(),'candidate_logprobs':lp.tolist(),'prediction':ties[0],'ties':ties,'full49152_sha256':sha(z)}
 return {'seconds':seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'scores':scores,'intervention':rec},{d:z.cpu() for d,z in heads.items()},{k:v.cpu() for k,v in obs.items()}
