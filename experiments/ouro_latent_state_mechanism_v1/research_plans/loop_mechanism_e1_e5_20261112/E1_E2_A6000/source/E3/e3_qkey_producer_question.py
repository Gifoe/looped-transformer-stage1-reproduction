"""Single-case Q-key producer candidate. Technical validation pending."""
import time,hashlib,torch
import recovery_adapter,eight_observer,eight_runner,natural_l18_hook,technical,e1_question
from e3_qkey_producer_observer import producer_observer,bits
def need(x,s):
 if not x:raise ValueError(s)
def thash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
@torch.inference_mode()
def run_case(model,split,ident,boundary,catalog,prefix,guard):
 src,_,_=recovery_adapter.load(split,ident);e=src['natural'];old,ot,ledger=e1_question.read_old(split,ident);R=e['R'];ids=e['ids'];P=len(prefix)
 need(ids==catalog['input_ids'] and e['targets']==catalog['targets'] and e['metadata_sha256']==catalog['metadata_sha256'] and e['gold']==catalog['gold_index']==old['gold'] and old['fixed_wrong']==max((j for j in range(len(e['targets'])) if j!=e['gold']),key=lambda j:e['scores'][0][j]) and old['input_ids_sha256']==catalog['input_ids_sha256']==boundary['input_sha256'] and R==catalog['R']==boundary['readout_position']==len(ids)-1 and boundary['groups']['prefix']==list(range(P)) and boundary['groups']['current']==list(range(P,len(ids))) and not boundary['crossing_offsets'] and not boundary['groups']['crossing'] and ids[:P]==prefix,'input boundary identity')
 x=torch.tensor([ids],device='cuda');core=model.model;prior=core.total_ut_steps; caps={};saved={};result={'id':ident,'split':split,'R':R,'input_ids_sha256':catalog['input_ids_sha256'],'targets':e['targets'],'gold':e['gold'],'fixed_wrong':old['fixed_wrong'],'old_metadata_sha256':ledger['metadata_sha256'],'old_tensor_sha256':ledger['tensor_sha256'],'natural_metadata_sha256':e['metadata_sha256'],'historical_natural_tensor_sha256':e['tensor_sha256'],'branch_metadata_sha256':e['actual_branch_metadata_sha256'],'branch_tensor_sha256':e['actual_branch_tensor_sha256'],'conditions':{},'captures':{},'cut_scope':'0,8,16 upstream; 17 technical endpoint only'};start=time.monotonic()
 def forward():
  roots=[]
  def root(m,args,kw):
   z=args[0] if args else kw.get('input_ids');need(torch.is_tensor(z) and bits(z,x) and kw.get('labels') is None and kw.get('use_cache') is False and kw.get('past_key_values') is None,'root input');roots.append(1)
  events=[]
  def localnorm(m,args,out):
   need(torch.is_tensor(out) and out.dtype==torch.bfloat16 and out.ndim==3 and out.shape[0]==1 and out.shape[1]==len(ids) and out.shape[-1]==2048 and torch.isfinite(out).all(),'actual localnorm');events.append(out[:,R,:].detach().clone())
  h=model.register_forward_pre_hook(root,with_kwargs=True); ah=core.layers[17].input_layernorm_2.register_forward_hook(localnorm)
  try:y=model(x,labels=None,use_cache=False,exit_at_step=core.total_ut_steps-1)
  finally:h.remove();ah.remove()
  z=y.logits[0,R,:].unsqueeze(0);need(len(roots)==1 and len(events)==core.total_ut_steps and z.dtype==torch.bfloat16 and z.shape==(1,49152) and torch.isfinite(z).all(),'forward logits');return z,events[-1]
 try:
  for depth in (1,2):
   guard();core.total_ut_steps=depth;own=src['states'][depth-1]
   with eight_observer.observe_qkv(model,depth,R) as rec,natural_l18_hook.intervene(model,depth,R,own['x'],own['a']) as ar:z,a=forward()
   snap=eight_runner.snapshot(rec);desc=eight_runner.descriptors(snap);need(desc==old['capture_receipts'][str(depth)]['qkv'] and bits(z,e['logits'][depth-1].cuda()) and technical.score(z,e['targets'])==e['scores'][depth-1] and bits(rec['receiver_x'],own['x'].cuda()) and bits(ar['receiver_a'],own['a'].cuda()),'natural capture')
   caps[depth]=snap;result['captures'][str(depth)]=desc
  for cut in (0,8,16,17):
   guard();core.total_ut_steps=2;
   with producer_observer(model,cut,R,P) as rec:clean,aclean=forward()
   need(bits(clean,e['logits'][1].cuda()) and technical.score(clean,e['targets'])==e['scores'][1] and all(bits(rec[n],caps[2][n]) for n in ('q','k','v')) and bits(rec['receiver_x_R'],src['states'][1]['x'].cuda()) and bits(aclean,src['states'][1]['a'].cuda()),'clean natural K2 identity')
   h1=rec['loop1_cut_hidden'];k0=rec['k']
   with producer_observer(model,cut,R,P,hidden_donor=h1,expected_clean_cut=rec['receiver_cut_hidden']) as rec1:pert,apert=forward()
   need(rec1['cut_outside_Q_unchanged'] and rec1['cut_donor_Q_equal'],'producer hidden substitution')
   k1=rec1['k']
   if cut==17:need(bits(k1[...,P:R,:],caps[1]['k'][...,P:R,:]) and bits(k1[...,:P,:],k0[...,:P,:]) and bits(k1[...,R:,:],k0[...,R:,:]) and bits(rec1['q'][...,R,:],caps[2]['q'][...,R,:]),'L18 endpoint')
   result['conditions'][str(cut)]={}
   for name,h,k,z0,a0 in [('A0K0',None,None,clean,aclean),('A1K1',h1,None,pert,apert),('A0K1',None,k1,None,None),('A1K0',h1,k0,None,None)]:
    guard()
    if z0 is None:
     with producer_observer(model,cut,R,P,hidden_donor=h,key_donor=k,expected_clean_cut=rec['receiver_cut_hidden']) as rr:z,a=forward()
    else:
     rr=rec if name=='A0K0' else rec1;z=z0;a=a0
    if name.startswith('A0'):need(all(bits(rr[n],caps[2][n]) for n in ('q','k','v')) and bits(rr['receiver_x_R'],src['states'][1]['x'].cuda()),'clean receiver drift')
    if name.startswith('A1'):need(all(bits(rr[n],rec1[n]) for n in ('q','k','v')) and bits(rr['receiver_x_R'],rec1['receiver_x_R']),'perturbed receiver drift')
    namekey=f'cut{cut}_{name}';saved[namekey+'_logits']=z.cpu().contiguous().clone();saved[namekey+'_sdpa_R']=rr['injected_sdpa_R'].cpu().contiguous().clone();saved[namekey+'_a_R']=a.cpu().contiguous().clone();saved[namekey+'_receiver_x_R']=rr['receiver_x_R'].cpu().contiguous().clone();result['conditions'][str(cut)][name]={'scores':technical.score(z,e['targets']),'logits_sha256':thash(z),'a_R_sha256':thash(a),'sdpa_R_sha256':thash(rr['injected_sdpa_R']),'receiver_x_R_sha256':thash(rr['receiver_x_R']),'q_sha256':thash(rr['q']),'receiver_k_sha256':thash(rr['k']),'mixed_K_Q_sha256':rr['mixed_K_Q_sha256'],'donor_K_Q_sha256':rr['donor_K_Q_sha256'],'mixed_K_Q_shape':rr['mixed_K_Q_shape'],'mixed_K_Q_dtype':rr['mixed_K_Q_dtype'],'cut_donor_Q_sha256':rr['loop1_cut_donor_Q_sha256'],'v_sha256':thash(rr['v']),'outside_R_unchanged':rr['outside_R_unchanged'],'cut_shape':rr['cut_shape'],'cut_input_sha256':rr['cut_input_sha256'],'cut_input_Q_sha256':rr['cut_input_Q_sha256'],'cut_input_R_sha256':rr['cut_input_R_sha256'],'cut_patched_Q_sha256':rr.get('cut_patched_Q_sha256')}
  guard();torch.cuda.synchronize();result['question_wall_seconds_cuda_synchronized']=time.monotonic()-start;return result,saved
 finally:core.total_ut_steps=prior
