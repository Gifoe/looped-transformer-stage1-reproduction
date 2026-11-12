"""Order004: actual L18 R-state block and attention mediator factorial."""
import argparse,os,json,hashlib,time,fcntl
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
import runner as r
import attested_runner
from state_mediation_components import input_r_block
from observer import bits,thash,hybrid
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=Path(__file__).resolve().parent
NON_R=('non_R_x_sha256','non_R_Q_sha256','non_R_K_sha256','non_R_V_sha256','non_R_sdpa_sha256')
def state_run(model,q,x=None,a=None,capture=False):
 with input_r_block(model,2,q['R'],x) as state:row,rec=r.run(model,q,2,patch=a,capture=capture)
 assert bits(state['applied_x_R'],rec['x_R']) and bits(state['actual_sdpa_R'],rec['O']) and state['non_R_x_unchanged']
 assert 'after_attention_R' in state and 'after_mlp_R' in state
 return row,rec,state
def receipt(state):
 return {k:({'sha256':thash(v),'shape':list(v.shape),'dtype':str(v.dtype)} if torch.is_tensor(v) else v) for k,v in state.items()}
def vectors(prefix,rec,state):
 values={name:rec[name] for name in ('x_R','O','injected','a_R','final_norm_R')}
 values.update({name:state[name] for name in ('Q_R','K_R','V_R','after_attention_R','after_mlp_R')})
 return {prefix+'_'+k:v.cpu().contiguous() for k,v in values.items()}
def save_tensors(path,t):
 tmp=Path(str(path)+'.partial');save_file(t,str(tmp));back=load_file(str(tmp));assert all(bits(v,back[k]) for k,v in t.items());os.replace(tmp,path)
def check_non_r(a,b):assert all(a[k]==b[k] for k in NON_R)
def check_params(a,b):
 assert a['tail']==b['tail'] and a['kwargs']==b['kwargs']
 for k in ('q','k','v'):
  for attr in ('shape','stride','dtype'):assert a[k][attr]==b[k][attr]
def collect(model,rows,out,prior,identity):
 for index,q in enumerate(rows):
  np=out/(r.key(q)+'.native.json');tp=np.with_suffix('.safetensors')
  if np.exists():
   old=json.loads(np.read_text());assert old['identity']==identity and old['tensor_sha256']==r.sha(tp);continue
  started=time.monotonic();pp=prior/(r.key(q)+'.native.json');p=json.loads(pp.read_text());assert p['identity']==json.loads((prior/'receipt.json').read_text())['identity'] and p['tensor_sha256']==r.sha(pp.with_suffix('.safetensors'));pt=load_file(str(pp.with_suffix('.safetensors')));assert q['input_ids_sha256']==p['input_ids_sha256'] and q['id']==p['id'] and q['split']==p['split'] and q['R']==p['R'] and q['targets']==p['targets']
  n,nrec,ns=state_run(model,q);n['condition']='N';assert n['full_logits_sha256']==p['native']['2']['full_logits_sha256'] and n['scores']==p['native']['2']['scores']
  for name in ('O','a_R','x_R','final_norm_R'):assert bits(nrec[name].cpu(),pt['K2_'+name])
  capture=q['id'] in identity['precision_ids'];u,urec,us=state_run(model,q,pt['K1_x_R'],capture=capture);u['condition']='U'
  assert bits(urec['x_R'].cpu(),pt['K1_x_R']);check_non_r(ns,us);check_params(nrec['sdpa_args'],urec['sdpa_args'])
  t={**vectors('N',nrec,ns),**vectors('U',urec,us),'X1':pt['K1_x_R'].clone(),'A1':pt['K1_O'].clone()}
  if capture:
   cap=urec.pop('qkv');t['U_fp32_sdpa_bf16_writeback']=hybrid(cap,cap,q['R'],fp32=True).cpu().contiguous();del cap
  save_tensors(tp,t)
  r.write(np,{'id':q['id'],'split':q['split'],'R':q['R'],'targets':q['targets'],'input_ids_sha256':q['input_ids_sha256'],'metadata_sha256':q['metadata_sha256'],'identity':identity,'native':{'2':n},'reused_K1_native':p['native']['1'],'upstream_block':u,'N_site_receipt':receipt(ns),'U_site_receipt':receipt(us),'N_sdpa_args':nrec['sdpa_args'],'U_sdpa_args':urec['sdpa_args'],'prior_native_sha256':r.sha(pp),'prior_tensor_sha256':r.sha(pp.with_suffix('.safetensors')),'tensor_sha256':r.sha(tp),'A_U_equals_natural_A1':bits(t['U_O'],t['A1']),'total_wall_seconds':time.monotonic()-started})
  print('MEDIATION_NATIVE_COMPLETE',index+1,len(rows),q['id'],round(time.monotonic()-started,3),flush=True)
def replay(model,rows,out,identity,donor_map):
 byid={q['id']:q for q in rows}
 for index,q in enumerate(rows):
  rp=out/(r.key(q)+'.results.json');np=out/(r.key(q)+'.native.json');m=json.loads(np.read_text());t=load_file(str(np.with_suffix('.safetensors')))
  if rp.exists():assert json.loads(rp.read_text())['identity']==identity;continue
  start=time.monotonic();x2=t['N_x_R'];x1=t['X1'];a2=t['N_O'];au=t['U_O'];conditions={'identity':(x2,a2),'U_restore':(x1,a2),'A_only':(None,au)};sources={};fallbacks={};unavailable={}
  for name,base,target,is_state in [('state',x2,x1,True),('mediator',au,a2,False)]:
   normname=name+'_norm_only';unrelname=name+'_unrelated_delta'
   if float(base.float().norm())>0:
    value=(base.float()*(target.float().norm()/base.float().norm())).to(base.dtype);conditions[normname]=(value,None) if is_state else (x1,value)
   else:unavailable[normname]='zero base norm'
   donor=donor_map[q['id']];fallback=[];found=False
   for unused in range(len(rows)-1):
    ot=load_file(str((out/(r.key(byid[donor])+'.native.json')).with_suffix('.safetensors')))
    od=ot['X1'].float()-ot['N_x_R'].float() if is_state else ot['N_O'].float()-ot['U_O'].float()
    if float(od.norm())>0:found=True;break
    fallback.append(donor);donor=donor_map[donor]
   if found:
    assert donor!=q['id'];value=(base.float()+(target.float()-base.float()).norm()*od/od.norm()).to(base.dtype);conditions[unrelname]=(value,None) if is_state else (x1,value);sources[unrelname]=donor;fallbacks[unrelname]=fallback
   else:unavailable[unrelname]='no nonzero unrelated source delta in frozen cohort'
  if 'U_fp32_sdpa_bf16_writeback' in t:conditions['U_fp32_sdpa_bf16_writeback']=(x1,t['U_fp32_sdpa_bf16_writeback'])
  u_receipt=m['U_site_receipt'];result=[{'condition':'U','pair':'1-2','direction':'state_block','receiver_loop':2,'donor_loop':1,'donor_id':q['id'],'native':m['native']['2'],'site_receipt':u_receipt,'outside_R_unchanged':True,'native_sdpa_R_sha256':thash(a2),'injected_sha256':thash(au),'a_R_sha256':thash(t['U_a_R']),'final_norm_R_sha256':thash(t['U_final_norm_R']),**m['upstream_block']}];intermediate={}
  for name,(state_patch,attention_patch) in conditions.items():
   row,rec,s=state_run(model,q,state_patch,attention_patch);check_non_r(m['N_site_receipt'],receipt(s));check_params(m['N_sdpa_args'],rec['sdpa_args'])
   if name=='identity':assert row['full_logits_sha256']==m['native']['2']['full_logits_sha256']
   if name in ('U_restore','mediator_norm_only','mediator_unrelated_delta','U_fp32_sdpa_bf16_writeback'):
    assert bits(s['actual_sdpa_R'].cpu(),au) and bits(rec['x_R'].cpu(),x1)
    for key in ('Q_R','K_R','V_R'):assert thash(s[key])==u_receipt[key]['sha256']
   if name=='A_only':assert bits(s['actual_sdpa_R'].cpu(),a2)
   if attention_patch is not None:assert bits(rec['injected'].cpu(),attention_patch)
   result.append({'condition':name,'pair':'1-2','direction':'state_attention_mediation','receiver_loop':2,'donor_loop':1,'donor_id':sources.get(name,q['id']),'donor_fallbacks':fallbacks.get(name,[]),'native':m['native']['2'],'site_receipt':receipt(s),'outside_R_unchanged':rec['outside_R_unchanged'],'native_sdpa_R_sha256':thash(a2),'injected_sha256':thash(rec['injected']),'a_R_sha256':thash(rec['a_R']),'final_norm_R_sha256':thash(rec['final_norm_R']),'input_delta_norm':float((rec['x_R'].cpu().float()-x2.float()).norm()),'attention_delta_from_A_U_norm':float((rec['injected'].cpu().float()-au.float()).norm()),**row})
   # Save actual residual, increment and normalized vectors for all controls/corners.
   for key in ('x_R','O','injected','a_R','final_norm_R'):intermediate[name+'_'+key]=rec[key].cpu().contiguous()
   for key in ('after_attention_R','after_mlp_R'):intermediate[name+'_'+key]=s[key].cpu().contiguous()
   print('MEDIATION_CONDITION_COMPLETE',q['id'],name,round(row['wall_seconds'],3),flush=True)
  ip=out/(r.key(q)+'.interventions.safetensors');save_tensors(ip,intermediate)
  r.write(rp,{'id':q['id'],'split':q['split'],'identity':identity,'native_file_sha256':r.sha(np),'rows':result,'unavailable':unavailable,'interventions_tensor_sha256':r.sha(ip),'total_wall_seconds':time.monotonic()-start})
  print('MEDIATION_QUESTION_COMPLETE',index+1,len(rows),q['id'],round(time.monotonic()-start,3),flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--inputs',required=True);a.add_argument('--cohort',required=True);a.add_argument('--prior',required=True);a.add_argument('--output',required=True);a.add_argument('--limit',type=int);args=a.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);lock=open(ROOT/'gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cohort=json.loads(Path(args.cohort).read_text());allrows={q['id']:q for q in map(json.loads,Path(args.inputs).read_text().splitlines())};rows=[allrows[i] for i in cohort['ids']]
 if args.limit:rows=rows[:args.limit]
 assert len(rows)>=2 and len(rows)==len({q['id'] for q in rows});ordered=sorted(q['id'] for q in rows);donor_map={ident:ordered[(i+1)%len(ordered)] for i,ident in enumerate(ordered)};db=json.dumps(donor_map,indent=2).encode()
 for name in ('state_donor_map.json','mediator_donor_map.json'):
  p=out/name
  if p.exists():assert p.read_bytes()==db
  else:p.write_bytes(db)
 groups={}
 for q in cohort.get('questions',[]):groups.setdefault(q['historical_group'],[]).append(q['id'])
 precision=sorted({i for v in groups.values() for i in v[:2]}.intersection(ordered));prior=Path(args.prior);pr=json.loads((prior/'receipt.json').read_text());assert pr['status']=='COMPLETE'
 identity={'experiment':'ASTRA_ORDER004_R_STATE_ATTENTION_MEDIATION','model_sha256':r.sha(ROOT/'model/model.safetensors'),'model_source_sha256':r.sha(ROOT/'model/modeling_ouro.py'),'configuration_sha256':r.sha(ROOT/'model/configuration_ouro.py'),'config_sha256':r.sha(ROOT/'model/config.json'),'mediation_runner_sha256':r.sha(__file__),'mediation_components_sha256':r.sha(ROOT/'state_mediation_components.py'),'base_runner_sha256':r.sha(ROOT/'runner.py'),'observer_sha256':r.sha(ROOT/'observer.py'),'attestation_wrapper_sha256':r.sha(ROOT/'attested_runner.py'),'inputs_sha256':r.sha(args.inputs),'cohort_sha256':r.sha(args.cohort),'actual_selected_ids_sha256':hashlib.sha256(json.dumps([q['id'] for q in rows]).encode()).hexdigest(),'state_donor_map_sha256':hashlib.sha256(db).hexdigest(),'mediator_donor_map_sha256':hashlib.sha256(db).hexdigest(),'donor_rule':'LEXICOGRAPHIC_ID_SORT_CYCLIC_BEFORE_OUTCOMES','precision_ids':precision,'torch':torch.__version__,'transformers':r.transformers.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16','attention':'sdpa','batch_size':1,'use_cache':False,'prior_results':str(prior),'prior_receipt_sha256':r.sha(prior/'receipt.json')}
 assert identity['model_sha256']=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
 for k in ('model_sha256','model_source_sha256','config_sha256','inputs_sha256','torch','transformers','cuda','gpu','dtype','attention','batch_size','use_cache'):assert identity[k]==pr['identity'][k]
 p=out/'execution_identity.json'
 if p.exists():assert json.loads(p.read_text())==identity
 r.write(p,identity);start=time.monotonic();model=r.AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;versions=[(n,id(p),p._version) for n,p in model.named_parameters()]
 with torch.inference_mode():collect(model,rows,out,prior,identity);replay(model,rows,out,identity,donor_map)
 assert versions==[(n,id(p),p._version) for n,p in model.named_parameters()]
 r.write(out/'receipt.json',{'status':'COMPLETE','N':len(rows),'identity':identity,'wall_seconds':time.monotonic()-start,**r.measure()})
if __name__=='__main__':main()
