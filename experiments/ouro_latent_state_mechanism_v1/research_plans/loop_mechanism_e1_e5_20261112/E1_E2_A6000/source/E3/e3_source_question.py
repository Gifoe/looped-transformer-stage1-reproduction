"""Natural Q/K/V capture and full-shape region rollback on fixed K1 receiver."""
import json,hashlib,time,torch
import torch.nn.functional as F
from pathlib import Path
import recovery_adapter,eight_observer,eight_runner,qkv_runner,natural_l18_hook,technical,e1_question
from e1_injection_observer import inject_sdpa_R,bits
from e3_source_control_v2 import build
ROOT=Path('/home/luansiyu/loop-latent')
def need(x,s):
 if not x:raise ValueError(s)
def thash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
@torch.inference_mode()
def run_case(model,split,ident,boundary,catalog,prefix,guard):
 start=time.monotonic();src,_,_=recovery_adapter.load(split,ident);e=src['natural'];old,ot,ledger=e1_question.read_old(split,ident);R=e['R'];ids=e['ids'];need(ids==catalog['input_ids'] and e['targets']==catalog['targets'] and e['metadata_sha256']==catalog['metadata_sha256'] and e['gold']==catalog['gold_index']==old['gold'] and old['fixed_wrong']==max((j for j in range(len(e['targets'])) if j!=e['gold']),key=lambda j:e['scores'][0][j]) and old['input_ids_sha256']==catalog['input_ids_sha256']==boundary['input_sha256'] and R==catalog['R']==boundary['readout_position']==len(ids)-1 and boundary['groups']['prefix']==list(range(len(prefix))) and boundary['groups']['current']==list(range(len(prefix),len(ids))) and not boundary['crossing_offsets'] and not boundary['groups']['crossing'] and ids[:len(prefix)]==prefix,'input/boundary');x=torch.tensor([ids],device='cuda');core=model.model;prior=core.total_ut_steps;caps={};result={'id':ident,'split':split,'R':R,'input_ids_sha256':catalog['input_ids_sha256'],'targets':e['targets'],'gold':e['gold'],'fixed_wrong':old['fixed_wrong'],'old_metadata_sha256':ledger['metadata_sha256'],'old_tensor_sha256':ledger['tensor_sha256'],'natural_metadata_sha256':e['metadata_sha256'],'historical_natural_tensor_sha256':e['tensor_sha256'],'branch_metadata_sha256':e['actual_branch_metadata_sha256'],'branch_tensor_sha256':e['actual_branch_tensor_sha256'],'conditions':{},'captures':{}};saved={}
 try:
  for depth in (1,2):
   guard();core.total_ut_steps=depth;own=src['states'][depth-1];roots=[]
   def root(m,args,kw):
    z=args[0] if args else kw.get('input_ids');need(torch.is_tensor(z) and bits(z,x) and kw.get('labels') is None and kw.get('use_cache') is False and kw.get('past_key_values') is None,'root');roots.append(1)
   h=model.register_forward_pre_hook(root,with_kwargs=True)
   try:
    with eight_observer.observe_qkv(model,depth,R) as rec,natural_l18_hook.intervene(model,depth,R,own['x'],own['a']) as ar:y=model(x,labels=None,use_cache=False,exit_at_step=depth-1)
   finally:h.remove()
   z=y.logits[0,R,:].unsqueeze(0);snap=eight_runner.snapshot(rec);d=eight_runner.descriptors(snap);need(len(roots)==1 and d==old['capture_receipts'][str(depth)]['qkv'] and bits(z,e['logits'][depth-1].cuda()) and technical.score(z,e['targets'])==e['scores'][depth-1] and bits(rec['receiver_x'],own['x'].cuda()) and bits(ar['receiver_a'],own['a'].cuda()),'natural capture');need(not snap['tail'] and snap['kwargs']['attn_mask'] is None and snap['kwargs']['is_causal'] is True and snap['kwargs']['dropout_p']==0,'unsupported actual SDPA params');caps[depth]=snap;result['captures'][str(depth)]=d
  for q in (1,2):
   qm=caps[1]['q'].clone()
   if q==2:qm[...,R,:]=caps[2]['q'][...,R,:]
   result['conditions'][str(q)]={}
   for name in ('ID','K_Q','V_Q','KV_Q','KV_P','K_Q_NORM','V_Q_NORM','V_Q_ROLL'):
    guard();k,v,control=build(caps[1]['k'],caps[2]['k'],caps[1]['v'],caps[2]['v'],R,len(prefix),name)
    if k is None:result['conditions'][str(q)][name]={'control':control};continue
    h=F.scaled_dot_product_attention(qm,k,v,*caps[1]['tail'],**caps[1]['kwargs']);need(torch.isfinite(h).all(),'hybrid SDPA');S=h[...,R,:].detach().clone();nameold=f'G{q}22'
    if name=='ID':need(bits(S,ot[nameold+'_sdpa_R'].cuda()),'ID SDPA parity')
    core.total_ut_steps=1;roots=[]
    def root(m,args,kw):
     z=args[0] if args else kw.get('input_ids');need(torch.is_tensor(z) and bits(z,x) and kw.get('labels') is None and kw.get('use_cache') is False and kw.get('past_key_values') is None,'receiver input');roots.append(1)
    handle=model.register_forward_pre_hook(root,with_kwargs=True)
    try:
     with inject_sdpa_R(model,R,S,expected_x=src['states'][0]['x'].cuda()) as rec:y=model(x,labels=None,use_cache=False,exit_at_step=0)
    finally:handle.remove()
    z=y.logits[0,R,:].unsqueeze(0);sc=technical.score(z,e['targets']);need(len(roots)==1 and qkv_runner.descriptors(rec)==old['corner_receipts']['G111']['actual_receiver_qkv'] and rec['outside_R_unchanged'] and bits(rec['injected_sdpa_R'],S) and z.dtype==torch.bfloat16 and z.shape==(1,49152) and torch.isfinite(z).all(),'actual receiver');a=rec['stages']['attention_localnorm']
    if name=='ID':need(bits(z,ot[nameold+'_logits'].cuda()) and bits(a,ot[nameold+'_a_R'].cuda()) and sc==old['scores'][nameold],'ID endpoint')
    tag=f'q{q}_{name}';saved[tag+'_logits']=z.cpu().contiguous().clone();saved[tag+'_sdpa_R']=S.cpu().contiguous().clone();saved[tag+'_a_R']=a.cpu().contiguous().clone();result['conditions'][str(q)][name]={'control':control,'scores':sc,'q_sha256':thash(qm),'k_sha256':thash(k),'v_sha256':thash(v),'sdpa_R_sha256':thash(S),'a_R_sha256':thash(a),'logits_sha256':thash(z),'actual_receiver_qkv':qkv_runner.descriptors(rec),'outside_R_unchanged':True}
  guard();torch.cuda.synchronize();result['question_wall_seconds_cuda_synchronized_excludes_save']=time.monotonic()-start;return result,saved
 finally:core.total_ut_steps=prior
