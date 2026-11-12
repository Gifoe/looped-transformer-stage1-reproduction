"""Three fixed cases: sealed natural QKV, FP64 decomposition, BF16 ID/ADD/norm/unrelated/restore."""
import os,json,hashlib,time,fcntl,resource,torch,numpy as np
from pathlib import Path
from safetensors.torch import load_file,save_file
from transformers import AutoModelForCausalLM
import recovery_adapter,eight_observer,eight_runner,qkv_runner,technical,natural_l18_hook,probe_eight,natural_sources,full_input,source_input,state_capture,replay_eight
from e1_injection_observer import inject_sdpa_R,bits
from e1_control_core import controls
from e1_donor_pool import Pool,OLD,sha
from e1_fp64_attention import decompose
ROOT=Path('/home/luansiyu/loop-latent');B=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112';OUT=B/'E1/technical_probe';SEL=['ACTAAP_2013_7_16','Mercury_405804','MSA_2012_8_30']
if not __debug__:raise RuntimeError('optimized Python forbidden')
def need(x,msg):
 if not x:raise ValueError(msg)
def desc(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def read_old(split,ident):
 key=hashlib.sha256(ident.encode()).hexdigest();mp=OLD/split/(key+'.json');tp=mp.with_suffix('.safetensors');covpath=OLD/(split+'_independent_cuda_coverage.json');expected={'train':'d257487676f31d0d122a913caad2c85369629ce1acca21bdabb5b2de0603c381','validation':'b4bf09b8a1d3c2f2c54740851e6482eb7a761e7091d70bff0d592d5f79df9f7c','test':'6af57d83f0c58c5832eb810c3a22bfbf3086db22f31bb37e41c3303fd5111526'};need(sha(covpath)==expected[split],'old coverage');cov=json.loads(covpath.read_text());match=[x for x in cov['questions'] if x['id']==ident];need(len(match)==1 and sha(mp)==match[0]['metadata_sha256'] and sha(tp)==match[0]['tensor_sha256'],'old eight source');return json.loads(mp.read_text()),load_file(str(tp)),match[0]
@torch.inference_mode()
def run_case(model,ident,pool,guard,split,include_restore=False,save_kv=False):
 t0=time.monotonic();guard();src,old4,_=recovery_adapter.load(split,ident);e=src['natural'];old,ot,ledger=read_old(split,ident);R=e['R'];ids=torch.tensor([e['ids']],device='cuda');targets=e['targets'];core=model.model;orig=core.total_ut_steps;caps={};saved={};meta={'id':ident,'split':split,'R':R,'input_ids_sha256':hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest(),'targets':targets,'gold':e['gold'],'old_eight_metadata_sha256':ledger['metadata_sha256'],'old_eight_tensor_sha256':ledger['tensor_sha256'],'natural_metadata_sha256':e['metadata_sha256'],'historical_natural_tensor_sha256':e['tensor_sha256'],'branch_metadata_sha256':e['actual_branch_metadata_sha256'],'branch_tensor_sha256':e['actual_branch_tensor_sha256'],'conditions':{},'runtime_receiver':'natural K1 ut0; observed runtime QKV remain K1','natural_capture':{},'fp64_attention':{}}
 try:
  _,bases=technical.verify(model,ids,R,targets,e['states'],e['norms'],e['logits'],e['scores']);guard()
  for depth in (1,2):
   core.total_ut_steps=depth;own=src['states'][depth-1]
   with eight_observer.observe_qkv(model,depth,R) as qr,natural_l18_hook.intervene(model,depth,R,own['x'],own['a']) as ar:y=model(ids,labels=None,use_cache=False,exit_at_step=depth-1)
   z=y.logits[0,R,:].unsqueeze(0);need(bits(z,bases[depth-1]) and technical.score(z,targets)==e['scores'][depth-1] and bits(qr['receiver_x'],own['x'].cuda()) and bits(ar['receiver_a'],own['a'].cuda()),'natural bridge')
   snap=eight_runner.snapshot(qr);d=eight_runner.descriptors(snap);need(d==old['capture_receipts'][str(depth)]['qkv'],'capture prior descriptors');need(not snap['tail'] and snap['kwargs'].get('dropout_p')==0 and snap['kwargs'].get('attn_mask') is None and snap['kwargs'].get('is_causal') is True,'actual mask/dropout unsupported; stop');caps[depth]=snap;meta['natural_capture'][str(depth)]={'qkv':d,'logits_sha256':desc(z),'x_sha256':desc(qr['receiver_x']),'a_sha256':desc(ar['receiver_a'])}
   for name in ('q','k','v'):
    v=snap[name][...,R,:] if name=='q' else snap[name]
    if name=='q' or save_kv:saved[f'natural_u{depth-1}_{name}']=v.detach().cpu().contiguous().clone()
  for q in (1,2):
   qs=caps[q]['q'].float().cpu().numpy();K1=caps[1]['k'].float().cpu().numpy();K2=caps[2]['k'].float().cpu().numpy();V1=caps[1]['v'].float().cpu().numpy();V2=caps[2]['v'].float().cpu().numpy();kw=caps[q]['kwargs'];d=decompose(qs,K1,V1,K2,V2,R,is_causal=kw['is_causal'],scale=kw['scale'],enable_gqa=kw.get('enable_gqa',False));need(d['valid_A1'].all() and d['valid_A2'].all() and np.max(np.abs(d['accounting_residual']))<1e-10,'FP64 account');meta['fp64_attention'][str(q)]={'scale':kw['scale'],'accounting_max_abs':float(np.max(np.abs(d['accounting_residual']))),'native_residuals':{}}
   for a in ('A1','A2','key','value','interaction','accounting_residual'):saved[f'q{q}_{a}_fp64']=torch.from_numpy(d[a].copy())
   for j,k in ((1,1),(1,2),(2,1),(2,2)):
    native=ot[f'G{q}{j}{k}_sdpa_R'].float().numpy().astype('float64');ideal=d[f'O{j}{k}'];res=native-ideal;saved[f'q{q}_native_minus_ideal_{j}{k}']=torch.from_numpy(res.copy());meta['fp64_attention'][str(q)]['native_residuals'][f'{j}{k}']={'l2':float(np.linalg.norm(res)),'max_abs':float(np.max(np.abs(res))),'relative_l2':float(np.linalg.norm(res)/np.linalg.norm(ideal)) if np.linalg.norm(ideal) else None}
   nativeI=(ot[f'G{q}22_sdpa_R'].double()-ot[f'G{q}21_sdpa_R'].double()-ot[f'G{q}12_sdpa_R'].double()+ot[f'G{q}11_sdpa_R'].double()).numpy();saved[f'q{q}_native_interaction_fp64']=torch.from_numpy(nativeI.copy());saved[f'q{q}_native_minus_ideal_interaction_fp64']=torch.from_numpy((nativeI-d['interaction']).copy())
   # SDPA internal finite-difference controls: donor selection is independent of labels.
   s11,s12,s21,s22=[ot[f'G{q}{j}{k}_sdpa_R'].cuda() for j,k in ((1,1),(1,2),(2,1),(2,2))];placeholder=torch.ones_like(s22,dtype=torch.float64);first=controls(s11,s12,s21,s22,placeholder);donor=None;donor_meta=None
   if first['delta_norm']!=0:donor,donor_meta=pool.select(split,ident,meta['input_ids_sha256'],q);donor=donor.cuda()
   C=controls(s11,s12,s21,s22,donor);meta['conditions'][str(q)]={'runtime_receiver':'natural K1 ut0; actual QKV remain K1','donor_query_background':q,'donor':donor_meta,'delta_norm':C['delta_norm'],'target_norm':C['target_norm'],'add_norm':C['add_norm'],'actual_norm':C['actual_norm'],'ADD_NORM_status':'OK' if C['ADD_NORM'] is not None else 'UNDEFINED'}
   for k in ('ADD_exact_fp64','cast_residual_fp64','delta_fp64','restore'):saved[f'q{q}_{k}']=C[k].detach().cpu().contiguous().clone()
   for name in (('ID','ADD','ADD_NORM','UNRELATED','restore') if include_restore else ('ID','ADD','ADD_NORM','UNRELATED')):
    v=C[name]
    if v is None:continue
    guard();core.total_ut_steps=1;roots=[]
    def root(m,args,kw):
     x=args[0] if args else kw.get('input_ids');need(torch.is_tensor(x) and bits(x,ids) and kw.get('labels') is None and kw.get('use_cache') is False and kw.get('past_key_values') is None,'root input');roots.append(1)
    rh=model.register_forward_pre_hook(root,with_kwargs=True)
    try:
     with inject_sdpa_R(model,R,v,expected_x=src['states'][0]['x'].cuda()) as rec:y=model(ids,labels=None,use_cache=False,exit_at_step=0)
    finally:rh.remove()
    z=y.logits[0,R,:].unsqueeze(0);need(len(roots)==1 and z.dtype==torch.bfloat16 and torch.isfinite(z).all() and qkv_runner.descriptors(rec)==old['corner_receipts']['G111']['actual_receiver_qkv'],'receiver capture');score=technical.score(z,targets);a=rec['stages']['attention_localnorm'];need(bits(rec['injected_sdpa_R'],v) and rec['outside_R_unchanged'],'patch site')
    if name in ('ID','restore'):need(bits(z,ot[f'G{q}22_logits'].cuda()) and bits(a,ot[f'G{q}22_a_R'].cuda()) and score==old['scores'][f'G{q}22'],'ID/restore endpoint')
    if C['delta_norm']==0 and name in ('ADD','UNRELATED'):need(bits(z,ot[f'G{q}22_logits'].cuda()),'zero delta identity')
    tag=f'q{q}_{name}';saved[tag+'_logits']=z.cpu().contiguous().clone();saved[tag+'_injected_sdpa_R']=v.cpu().contiguous().clone()
    for stage,x in rec['stages'].items():saved[tag+'_'+stage]=x.cpu().contiguous().clone()
    meta['conditions'][str(q)][name]={'scores':score,'logits_sha256':desc(z),'a_sha256':desc(a),'injected_sha256':desc(v),'actual_receiver_qkv':qkv_runner.descriptors(rec),'outside_R_unchanged':True,'root_input_events':len(roots)}
  guard();meta['raw_kv_scope']='all K/V bytes saved for designated discovery sample' if save_kv else 'Q_R saved; full K/V bytes not archived, actual source descriptor/hash and reproducible input retained';meta['restore_scope']='tensor equality verified; no additional restore forward' if not include_restore else 'additional restore forward';meta['question_wall_seconds']=time.monotonic()-t0;return meta,saved
 finally:core.total_ut_steps=orig

