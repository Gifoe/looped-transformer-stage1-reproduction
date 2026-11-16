"""Semantic prefix-channel replacement through original full-shaped SDPA; no D-state patch."""
import pathlib,json,hashlib,torch,contextlib,time
from ab1_loader_draft import load_model
ROOT=pathlib.Path('/root/ouro-temporal-bridge-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-temporal-bridge-sol-20261117/raw')
def bits(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def clone_layout(x):
 z=x.clone(memory_format=torch.preserve_format)
 if z.stride()!=x.stride():z=torch.empty_strided(x.shape,x.stride(),device=x.device,dtype=x.dtype);z.copy_(x)
 assert z.stride()==x.stride();return z
def initialize():
 torch.set_num_threads(4);torch.manual_seed(2026111711);torch.backends.cuda.matmul.allow_tf32=False;m,r=load_model('/root/autodl-tmp/ouro-e4-e5/model',authorized=True,device='cuda:0');m.model.total_ut_steps=2
 for p in m.parameters():p.requires_grad_(False)
 r['total_ut_steps']=2;(ROOT/'SOURCE_RUNTIME.json').write_text(json.dumps(r,indent=2));return m
@contextlib.contextmanager
def cut(m,q,reference=None,native=None,g1=1,mask2=7,capture=False,technical=False):
 b=q['b'];T=q['R']+1;active=[];events=[];handles=[];cache={'kv':{},'S_input':{},'S_attention':{},'norms':{},'D_input':{},'D_output':{}};rec={'g1':g1,'mask2':mask2,'extra_SDPA_calls':0,'closed_rows_per_pass':[0,0],'source_exact_checks':0,'C00_D_norm_exact':[],'g1closed_K2_input_exact':None};norm_count=[]
 masks=[]
 for u in [0,1]:
  closed=torch.zeros(T,dtype=torch.bool,device='cuda')
  if u==0:
   if not g1:closed[b:]=True
  else:
   for bit,name in [(1,'map'),(2,'query'),(4,'read')]:
    if not mask2&bit:a,e=q['regions'][name];closed[a:e]=True
  masks.append(closed);rec['closed_rows_per_pass'][u]=int(closed.sum())
 def pre(i):
  def h(mod,args,kw):
   u=kw['current_ut'];assert not kw['use_cache'] and kw.get('past_key_value') is None;active.append((u,i));events.append([u,i]);x=args[0] if args else kw['hidden_states'];assert x.shape[1]==T
   if capture:cache['S_input'][(u,i)]=x[:,:b].detach().clone()
   if technical and capture:cache['D_input'][(u,i)]=x[:,b:].detach().clone()
   if native is not None:
    assert bits(x[:,:b],native['S_input'][(u,i)]);rec['source_exact_checks']+=1
    if i==0 and u==1 and not g1:assert bits(x[:,b:],reference['norms'][0][:,b:]);rec['g1closed_K2_input_exact']=True
   if technical and reference is not None and ((u==0 and not g1) or (u==1 and not g1 and mask2==0)):assert bits(x[:,b:],reference['D_input'][(u,i)])
  return h
 def post(i):
  def h(mod,args,kw,out):
   u=kw['current_ut'];assert active.pop()==(u,i)
   if technical and capture:cache['D_output'][(u,i)]=out[:,b:].detach().clone()
   if technical and reference is not None and ((u==0 and not g1) or (u==1 and not g1 and mask2==0)):assert bits(out[:,b:],reference['D_output'][(u,i)])
  return h
 def normhook(mod,args,out):
  u=len(norm_count);norm_count.append(1);assert u<2
  if capture:cache['norms'][u]=out.detach().clone()
  if native is not None:
   assert bits(out[:,:b],native['norms'][u][:,:b]);rec['source_exact_checks']+=1
   if (u==0 and not g1) or (u==1 and not g1 and mask2==0):assert bits(out[:,b:],reference['norms'][u][:,b:]);rec['C00_D_norm_exact'].append(u+1)
 original=torch.nn.functional.scaled_dot_product_attention
 def sdpa(*args,**kw):
  out=original(*args,**kw)
  if not active:return out
  u,i=active[-1];query,key,value=args[:3];assert kw.get('is_causal') is True and kw.get('attn_mask') is None and kw.get('dropout_p',0)==0;assert query.shape[-2]==key.shape[-2]==value.shape[-2]==T
  if capture:cache['kv'][(u,i)]=(key[:,:,:b].detach().clone(),value[:,:,:b].detach().clone());cache['S_attention'][(u,i)]=out[:,:,:b].detach().clone()
  if native is not None:
   kn,vn=native['kv'][(u,i)];assert bits(key[:,:,:b],kn) and bits(value[:,:,:b],vn) and bits(out[:,:,:b],native['S_attention'][(u,i)]);rec['source_exact_checks']+=3
  closed=masks[u]
  if not bool(closed.any()):return out
  assert reference is not None;kr,vr=reference['kv'][(u,i)];assert kr.shape==key[:,:,:b].shape and vr.shape==value[:,:,:b].shape;kk=clone_layout(key);vv=clone_layout(value);kk[:,:,:b]=kr;vv[:,:,:b]=vr;aa=list(args);aa[1]=kk;aa[2]=vv;replacement=original(*aa,**kw);z=out.clone();z[:,:,closed]=replacement[:,:,closed];assert bits(z[:,:,:b],out[:,:,:b]) and bits(z[:,:,~closed],out[:,:,~closed]);rec['extra_SDPA_calls']+=1;return z
 try:
  handles.append(m.model.norm.register_forward_hook(normhook))
  for i,l in enumerate(m.model.layers):handles.extend([l.register_forward_pre_hook(pre(i),with_kwargs=True),l.register_forward_hook(post(i),with_kwargs=True)])
  torch.nn.functional.scaled_dot_product_attention=sdpa;yield cache,rec
  assert events==[[u,i] for u in [0,1] for i in range(24)] and len(norm_count)==2 and not active
 finally:
  torch.nn.functional.scaled_dot_product_attention=original
  for h in handles:h.remove()
@torch.inference_mode()
def run(m,q,reference=None,native=None,g1=1,mask2=7,capture=False,technical=False):
 assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256'];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();ids=torch.tensor([q['ids']],device='cuda')
 with cut(m,q,reference,native,g1,mask2,capture,technical) as (cache,rec):_,states,_=m.model(ids,use_cache=False,past_key_values=None)
 heads={d+1:m.lm_head(states[d])[:,q['R']].detach().clone() for d in [0,1]};torch.cuda.synchronize();elapsed=time.monotonic()-t;scores={}
 for d,z in heads.items():
  x=z.reshape(-1)[q['targets']].float();lp=z.float().reshape(-1).log_softmax(-1)[q['targets']];ties=torch.nonzero(x==x.max()).flatten().tolist();scores[str(d)]={'candidate_logits':x.tolist(),'candidate_logprobs':lp.tolist(),'prediction':ties[0],'ties':ties,'full49152_sha256':digest(z)}
 return {'scores':scores,'cut':rec,'seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated()},heads,cache
