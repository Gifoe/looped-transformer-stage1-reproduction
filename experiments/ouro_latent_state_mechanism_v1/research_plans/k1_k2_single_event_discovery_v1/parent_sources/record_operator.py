"""Natural full-context, actual post-RoPE record K/V permutations. No prefix cut."""
import pathlib,json,hashlib,time,contextlib,torch
from ab1_loader_draft import load_model
ROOT=pathlib.Path('/root/ouro-semantic-read-sol-20261117');RAW=pathlib.Path('/root/autodl-tmp/ouro-semantic-read-sol-20261117/raw');LAYERS=tuple(range(8,16))
def bits(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def clone(x):
 z=x.clone(memory_format=torch.preserve_format)
 if z.stride()!=x.stride():z=torch.empty_strided(x.shape,x.stride(),device=x.device,dtype=x.dtype);z.copy_(x)
 assert z.stride()==x.stride();return z
def initialize():
 torch.set_num_threads(4);torch.manual_seed(2026111731);torch.backends.cuda.matmul.allow_tf32=False;m,r=load_model('/root/autodl-tmp/ouro-e4-e5/model',authorized=True,device='cuda:0');m.model.total_ut_steps=2
 for p in m.parameters():p.requires_grad_(False)
 r['total_ut_steps']=2;(ROOT/'SOURCE_RUNTIME.json').write_text(json.dumps(r,indent=2));return m
@contextlib.contextmanager
def edit(m,q,kind='native',arm=2,native_prefix=None,capture_prefix=False,witness=False):
 assert kind in ['native','noop','self','K','V','joint'] and arm in [1,2];a,b=q['record_spans'][0];c,d=q['record_spans'][1];h=q['H_start'];T=q['R']+1;assert 0<=a<b==c<d==h<=q['R'] and b-a==d-c;active=[];visits=[];edits=[];handles=[];pref={};witnesses={};audit={'kind':kind,'arm':arm,'events':edits,'extra_SDPA_calls':0,'unchanged_nonH_checks':0,'prefix_identity_sites':0,'max_SDPA_delta':0.,'max_SDPA_delta_rms':0.};original=torch.nn.functional.scaled_dot_product_attention;norms=[]
 def check(key,x):
  z=digest(x)
  if capture_prefix:pref[key]=z
  if native_prefix is not None:assert z==native_prefix[key],('prefix changed',key);audit['prefix_identity_sites']+=1
 def pre(i):
  def hook(mod,args,kw):
   u=kw['current_ut'];assert not kw['use_cache'] and kw.get('past_key_value') is None;active.append((u,i));visits.append([u,i]);x=args[0] if args else kw['hidden_states'];check('input_'+str((u,i)),x[:,:h])
  return hook
 def post(i):
  def hook(mod,args,kw,out):assert active.pop()==(kw['current_ut'],i)
  return hook
 def normhook(mod,args,out):
  u=len(norms);norms.append(1);check('norm_'+str(u),out[:,:h])
 def sdpa(*args,**kw):
  out=original(*args,**kw)
  if not active:return out
  u,i=active[-1];query,key,value=args[:3];assert kw.get('is_causal') is True and kw.get('attn_mask') is None and kw.get('dropout_p',0)==0;assert query.shape[-2]==key.shape[-2]==value.shape[-2]==T
  check('K_'+str((u,i)),key[:,:,a:h]);check('V_'+str((u,i)),value[:,:,a:h]);check('attention_'+str((u,i)),out[:,:,:h])
  if kind in ['native','noop'] or u!=arm-1 or i not in LAYERS:return out
  kk=clone(key);vv=clone(value)
  if kind in ['K','joint']:kk[:,:,a:b]=key[:,:,c:d];kk[:,:,c:d]=key[:,:,a:b]
  if kind in ['V','joint']:vv[:,:,a:b]=value[:,:,c:d];vv[:,:,c:d]=value[:,:,a:b]
  if kind=='self':assert bits(kk,key) and bits(vv,value)
  aa=list(args);aa[1]=kk;aa[2]=vv;changed=original(*aa,**kw);z=out.clone();z[:,:,h:]=changed[:,:,h:];assert bits(z[:,:,:h],out[:,:,:h]);audit['unchanged_nonH_checks']+=1;audit['extra_SDPA_calls']+=1;delta=changed[:,:,h:].double()-out[:,:,h:].double();audit['max_SDPA_delta']=max(audit['max_SDPA_delta'],float(delta.abs().max()));audit['max_SDPA_delta_rms']=max(audit['max_SDPA_delta_rms'],float(delta.square().mean().sqrt()));edits.append({'loop':u+1,'layer':i+1,'receiver_span':[h,T],'record_spans':[[a,b],[c,d]],'all_heads':True,'same_layer_input_KV':True,'actual_recomputed':True,'H_delta_l2':delta.norm().item()})
  if kind=='self':assert bits(changed,out)
  if witness and i==8:
   witnesses[(u,i)]={'Q_H':query[:,:,h:].cpu(),'K_full':key.cpu(),'V_full':value.cpu(),'native_SDPA_H':out[:,:,h:].cpu(),'edited_SDPA_H':changed[:,:,h:].cpu(),'record_spans':[[a,b],[c,d]],'H_start':h,'scale':kw.get('scale') or query.shape[-1]**-.5,'kind':kind}
  return z
 try:
  for i,l in enumerate(m.model.layers):handles.extend([l.register_forward_pre_hook(pre(i),with_kwargs=True),l.register_forward_hook(post(i),with_kwargs=True)])
  handles.append(m.model.norm.register_forward_hook(normhook));torch.nn.functional.scaled_dot_product_attention=sdpa;yield audit,pref,witnesses
  assert visits==[[u,i] for u in [0,1] for i in range(24)] and not active and len(norms)==2
  expected=[] if kind in ['native','noop'] else [[arm,i+1] for i in LAYERS];assert [[e['loop'],e['layer']] for e in edits]==expected;assert audit['extra_SDPA_calls']==len(expected);audit['SDPA_native_invocations']=len(visits);audit['receiver_row_writebacks']=len(expected)*(T-h)
 finally:
  torch.nn.functional.scaled_dot_product_attention=original
  for handle in handles:handle.remove()
@torch.inference_mode()
def run(m,q,kind='native',arm=2,native_prefix=None,capture_prefix=False,witness=False):
 assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256'];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
 with edit(m,q,kind,arm,native_prefix,capture_prefix,witness) as (audit,pref,w):_,states,_=m.model(torch.tensor([q['ids']],device='cuda'),use_cache=False,past_key_values=None)
 heads={d+1:m.lm_head(states[d])[:,q['R']].detach().clone() for d in [0,1]};torch.cuda.synchronize();elapsed=time.monotonic()-t;scores={}
 for d,z in heads.items():
  x=z.flatten()[q['targets']].float();lp=z.flatten().float().log_softmax(-1)[q['targets']];ties=torch.nonzero(x==x.max()).flatten().tolist();scores[str(d)]={'candidate_logits':x.tolist(),'candidate_logprobs':lp.tolist(),'prediction':ties[0],'ties':ties,'full49152_sha256':digest(z)}
 return {'scores':scores,'audit':audit,'seconds':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated()},heads,pref,w
