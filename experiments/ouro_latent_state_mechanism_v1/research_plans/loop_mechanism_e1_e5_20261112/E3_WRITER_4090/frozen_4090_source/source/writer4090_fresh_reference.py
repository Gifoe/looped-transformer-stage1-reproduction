"""Fresh same-site BF16 Q-to-R references, without A6000 numerical anchors."""
import hashlib,json,torch
from e3_qkey_producer_observer import producer_observer,bits
from capture_branch_reference import capture_real_branches,reference_deltas
def need(x,s):
 if not x:raise ValueError(s)
def thash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def score(z,targets):return [float(z.float().log_softmax(-1).gather(-1,torch.tensor([[t]],device=z.device)).sum()) for t in targets]
@torch.inference_mode()
def capture_case(model,q,prefix,guard):
 ids=q['ids'];R=q['R'];P=len(prefix);need(q['input_ids_sha256']==hashlib.sha256(json.dumps(ids).encode()).hexdigest() and ids[:P]==prefix and 0<P<R==len(ids)-1,'input boundary');x=torch.tensor([ids],device='cuda');core=model.model;prior=core.total_ut_steps;core.total_ut_steps=2;result={'id':q['id'],'split':q['split'],'R':R,'targets':q['targets'],'input_ids_sha256':q['input_ids_sha256'],'conditions':{},'reference':{}};saved={}
 def run(h=None,k=None,expected_cut=None):
  roots=[]
  def root(m,args,kw):
   y=args[0] if args else kw.get('input_ids');need(torch.is_tensor(y) and bits(y,x) and kw.get('labels') is None and kw.get('use_cache') is False and kw.get('past_key_values') is None,'root');roots.append(1)
  astages=[]
  def capture_a(m,args,out):
   need(torch.is_tensor(out) and out.dtype==torch.bfloat16 and out.shape==(1,len(ids),2048) and torch.isfinite(out).all(),'target a');astages.append(out[:,R,:].detach().clone())
  full=[]
  def target_full(m,args,kw):
   if kw.get('current_ut')==1:
    x0=args[0] if args else kw.get('hidden_states');need(torch.is_tensor(x0) and x0.shape==(1,len(ids),2048),'target full');full.append(x0.detach().clone())
  hook=model.register_forward_pre_hook(root,with_kwargs=True);ah=core.layers[17].input_layernorm_2.register_forward_hook(capture_a);fh=core.layers[17].register_forward_pre_hook(target_full,with_kwargs=True)
  try:
   with producer_observer(model,8,R,P,hidden_donor=h,key_donor=k,expected_clean_cut=expected_cut) as rec,capture_real_branches(model,R,expected_length=len(ids)) as branches:y=model(x,labels=None,use_cache=False,exit_at_step=1)
  finally:hook.remove();ah.remove();fh.remove()
  need(len(astages)==2 and len(full)==1,'target events');rec['target_a_R']=astages[1];rec['target_x_full']=full[0]
  z=y.logits[0,R,:].unsqueeze(0);need(len(roots)==1 and z.shape==(1,49152) and z.dtype==torch.bfloat16 and torch.isfinite(z).all(),'forward');return z,rec,branches
 try:
  guard();N,n,nref=run();guard();Pv,p,pref=run(h=n['loop1_cut_hidden'],expected_cut=n['receiver_cut_hidden']);need(p['cut_outside_Q_unchanged'] and p['cut_donor_Q_equal'] and bits(n['receiver_cut_hidden'][:,R,:],p['receiver_cut_hidden'][:,R,:]),'cut identity');guard();PK,k,kref=run(h=n['loop1_cut_hidden'],k=n['k'],expected_cut=n['receiver_cut_hidden']);need(k['outside_R_unchanged'],'key edge')
  for name,z,rec in [('N',N,n),('P',Pv,p),('PK',PK,k)]:
   for tag,v in [('logits',z),('sdpa_R',rec['injected_sdpa_R']),('x_R',rec['receiver_x_R']),('a_R',rec['target_a_R'])]:saved[name+'_'+tag]=v.cpu().contiguous().clone()
   mask=torch.arange(len(ids),device=z.device)!=R;result['conditions'][name]={'nonR_target_x_sha256':thash(rec['target_x_full'][:,mask,:]),'nonR_qkv_sha256':{v:thash(rec[v][...,mask,:]) for v in ('q','k','v')},'scores':score(z,q['targets']),'logits_sha256':thash(z),'sdpa_R_sha256':thash(rec['injected_sdpa_R']),'x_R_sha256':thash(rec['receiver_x_R']),'a_R_sha256':thash(rec['target_a_R']),'mixed_K_Q_sha256':rec['mixed_K_Q_sha256'],'outside_R_unchanged':rec['outside_R_unchanged']}
  saved['cut8_loop1_hidden_Q']=n['loop1_cut_hidden'][:,P:R,:].cpu().contiguous().clone();saved['l18_clean_K_Q']=n['k'][...,P:R,:].cpu().contiguous().clone();delta=reference_deltas(nref,pref)
  for (i,kind),d in delta.items():
   Nval=nref[(i,kind)];Pval=pref[(i,kind)];need(Nval.shape==Pval.shape==(1,2048) and Nval.dtype==Pval.dtype==torch.bfloat16 and torch.isfinite(d).all(),'reference branch');tag=f'layer{i:02d}_{kind}';saved[tag+'_N']=Nval.cpu().contiguous().clone();saved[tag+'_P']=Pval.cpu().contiguous().clone();saved[tag+'_delta']=d.cpu().contiguous().clone();result['reference'][tag]={'N_sha256':thash(Nval),'P_sha256':thash(Pval),'delta_sha256':thash(d)}
  guard();return result,saved,{'hidden1':n['loop1_cut_hidden'],'clean_cut':n['receiver_cut_hidden'],'clean_k':n['k'],'backgrounds':{'N':{'target_full':n['target_x_full'],**{j:n[j] for j in ('q','k','v')}},'P':{'target_full':p['target_x_full'],**{j:p[j] for j in ('q','k','v')}}},'deltas':delta}
 finally:core.total_ut_steps=prior
