"""Order006 preregistered natural K2 Q x KV confirmation, with reduced-base KV reuse."""
import argparse,os,json,hashlib,time,fcntl
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
import runner as r
import attested_runner  # source-attestation-only factory wrapper
from q_patch_components import derive_q_patch
from observer import bits,hybrid,thash
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=Path(__file__).resolve().parent
def collect(model,rows,out,prior,identity):
 for index,q in enumerate(rows):
  p=out/(r.key(q)+'.native.json');tp=p.with_suffix('.safetensors')
  if p.exists():
   m=json.loads(p.read_text());assert m['identity']==identity and m['tensor_sha256']==r.sha(tp);continue
  started=time.monotonic();oldp=prior/(r.key(q)+'.native.json');oldrp=prior/(r.key(q)+'.results.json');old=json.loads(oldp.read_text());oldres=json.loads(oldrp.read_text());oldt=load_file(str(oldp.with_suffix('.safetensors')))
  native={};captures={};receipts={};t={}
  for depth in (1,2):
   row,rec=r.run(model,q,depth,capture=True);assert row['full_logits_sha256']==old['native'][str(depth)]['full_logits_sha256'] and row['scores']==old['native'][str(depth)]['scores']
   native[str(depth)]=row;captures[depth]=rec.pop('qkv');receipts[str(depth)]={k:v for k,v in rec.items() if not torch.is_tensor(v)}
   for name in ('O','a_R','x_R','final_norm_R'):
    t[f'K{depth}_{name}']=rec[name].cpu().contiguous();assert bits(t[f'K{depth}_{name}'],oldt[f'K{depth}_{name}'])
  kv=hybrid(captures[2],captures[1],q['R']);assert bits(kv.cpu(),oldt['2_1_target'])
  candidates=[x for x in oldres['rows'] if x['receiver_loop']==2 and x['donor_loop']==1 and x['condition']=='target'];assert len(candidates)==1 and candidates[0]['injected_sha256']==thash(kv)
  head_receipts={};unavailable={}
  for name,mode in [('q_only','q_only'),('joint_qkv','joint_qkv'),('q_direction_headnorm_matched','donor_q_to_receiver_headnorm'),('q_temperature_headnorm','receiver_q_to_donor_headnorm')]:
   try:patch,receipt=derive_q_patch(captures[2],captures[1],q['R'],mode=mode,return_receipt=True)
   except ZeroDivisionError as error:
    unavailable[name]=str(error);continue
   t[name+'_target']=patch.cpu().contiguous();head_receipts[name]=receipt
  assert bits(t['joint_qkv_target'],t['K1_O'])
  if q['id'] in identity['precision_ids']:t['q_only_fp32_sdpa_bf16_writeback_target']=derive_q_patch(captures[2],captures[1],q['R'],fp32_sdpa=True).cpu().contiguous()
  del captures
  temp=Path(str(tp)+'.partial');save_file(t,str(temp));back=load_file(str(temp));assert all(bits(v,back[k]) for k,v in t.items());os.replace(temp,tp)
  r.write(p,{'id':q['id'],'split':q['split'],'R':q['R'],'targets':q['targets'],'input_ids_sha256':q['input_ids_sha256'],'metadata_sha256':q['metadata_sha256'],'identity':identity,'native':native,'capture_receipts':receipts,'q_head_receipts':head_receipts,'unavailable':unavailable,'reused_kv_rollback':candidates[0],'prior_native_sha256':r.sha(oldp),'prior_result_sha256':r.sha(oldrp),'prior_tensor_sha256':r.sha(oldp.with_suffix('.safetensors')),'tensor_sha256':r.sha(tp),'total_wall_seconds':time.monotonic()-started})
  print('QUERY_NATIVE_COMPLETE',index+1,len(rows),q['id'],round(time.monotonic()-started,3),flush=True)
def replay(model,rows,out,identity,donor_map,technical):
 byid={q['id']:q for q in rows}
 for index,q in enumerate(rows):
  rp=out/(r.key(q)+'.results.json');np=out/(r.key(q)+'.native.json');m=json.loads(np.read_text());t=load_file(str(np.with_suffix('.safetensors')))
  if rp.exists():assert json.loads(rp.read_text())['identity']==identity;continue
  started=time.monotonic();base=t['K2_O'];controls={};sources={};fallbacks={};native=m['native']['2']
  for kind in ('q_only','joint_qkv'):
   target=t[kind+'_target'];delta=target.float()-base.float();otherid=donor_map[q['id']];fallback=[]
   for unused in range(len(rows)-1):
    ot=load_file(str((out/(r.key(byid[otherid])+'.native.json')).with_suffix('.safetensors')));od=ot[kind+'_target'].float()-ot['K2_O'].float()
    if float(od.norm())>0:break
    fallback.append(otherid);otherid=donor_map[otherid]
   assert otherid!=q['id'] and float(od.norm())>0 and float(base.float().norm())>0
   controls[kind]=target;controls[kind+'_norm_only']=(base.float()*(target.float().norm()/base.float().norm())).to(base.dtype);controls[kind+'_unrelated_delta']=(base.float()+delta.norm()*od/od.norm()).to(base.dtype)
   sources[kind+'_unrelated_delta']=otherid;fallbacks[kind+'_unrelated_delta']=fallback
  for name in ('q_direction_headnorm_matched','q_temperature_headnorm','q_only_fp32_sdpa_bf16_writeback'):
   if name+'_target' in t:controls[name]=t[name+'_target']
  row,rec=r.run(model,q,2,base);assert row['full_logits_sha256']==native['full_logits_sha256'];result=[{'pair':'1-2','direction':'rollback','receiver_loop':2,'donor_loop':2,'condition':'identity','donor_id':q['id'],'outside_R_unchanged':rec['outside_R_unchanged'],**row}]
  if technical:
   # A small endpoint bridge verifies that a reused KV cell has unchanged downstream semantics.
   # Its vector was already verified against the exact prior current-hardware tensor.
   from observer import hybrid as unused_hybrid
   prior_tensor=load_file(str(Path(identity['prior_results'])/(r.key(q)+'.native.safetensors')))
   bridge,unused_rec=r.run(model,q,2,prior_tensor['2_1_target']);assert bridge['full_logits_sha256']==m['reused_kv_rollback']['full_logits_sha256']
   m['technical_reused_kv_full_forward_bridge']=bridge
  for name,patch in controls.items():
   row,rec=r.run(model,q,2,patch);assert bits(rec['O'].cpu(),base)
   if name=='joint_qkv':assert bits(rec['a_R'].cpu(),t['K1_a_R'])
   group='joint_qkv' if name.startswith('joint_qkv') else 'q_only';target=t[group+'_target'];pf=patch.float();bf=base.float()
   result.append({'pair':'1-2','direction':'rollback','receiver_loop':2,'donor_loop':1,'condition':name,'donor_id':sources.get(name,q['id']),'donor_fallbacks':fallbacks.get(name,[]),'native':native,'native_sdpa_R_sha256':thash(base),'injected_sha256':thash(patch),'a_R_sha256':thash(rec['a_R']),'final_norm_R_sha256':thash(rec['final_norm_R']),'target_delta_norm':float((target.float()-bf).norm()),'actual_delta_norm':float((pf-bf).norm()),'native_norm':float(bf.norm()),'injected_norm':float(pf.norm()),'cosine':float(torch.nn.functional.cosine_similarity(pf.flatten(),bf.flatten(),dim=0)),'q_head_receipt':m['q_head_receipts'].get(name),'outside_R_unchanged':rec['outside_R_unchanged'],**row})
   print('QUERY_CONDITION_COMPLETE',q['id'],name,round(row['wall_seconds'],3),flush=True)
  r.write(rp,{'id':q['id'],'split':q['split'],'identity':identity,'native_file_sha256':r.sha(np),'reused_kv_rollback':m['reused_kv_rollback'],'technical_reused_kv_full_forward_bridge':m.get('technical_reused_kv_full_forward_bridge'),'unavailable':m['unavailable'],'rows':result,'total_wall_seconds':time.monotonic()-started})
  print('QUERY_QUESTION_COMPLETE',index+1,len(rows),q['id'],round(time.monotonic()-started,3),flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--inputs',required=True);a.add_argument('--cohort',required=True);a.add_argument('--prior',required=True);a.add_argument('--output',required=True);a.add_argument('--limit',type=int);a.add_argument('--technical',action='store_true');args=a.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 lock=open(ROOT/'gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cohort=json.loads(Path(args.cohort).read_text());ids=cohort['ids'];byid={q['id']:q for q in map(json.loads,Path(args.inputs).read_text().splitlines())};rows=[byid[i] for i in ids]
 if args.limit:rows=rows[:args.limit]
 assert len(rows)>=2 and len({q['id'] for q in rows})==len(rows)
 ordered=sorted(q['id'] for q in rows);donor_map={ident:ordered[(i+1)%len(ordered)] for i,ident in enumerate(ordered)};donor_bytes=json.dumps(donor_map,indent=2).encode();dp=out/'control_donor_map.json'
 if dp.exists():assert dp.read_bytes()==donor_bytes
 else:dp.write_bytes(donor_bytes)
 precision=sorted(set(cohort['precision_ids']).intersection(ordered))
 assert r.sha(ROOT/'model/model.safetensors')=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
 prior=Path(args.prior);pr=json.loads((prior/'receipt.json').read_text());assert pr['status']=='COMPLETE'
 identity={'experiment':'ASTRA_ORDER006_PREREGISTERED_NATURAL_K2_QUERY_FACTORIAL','model_sha256':r.sha(ROOT/'model/model.safetensors'),'model_source_sha256':r.sha(ROOT/'model/modeling_ouro.py'),'configuration_sha256':r.sha(ROOT/'model/configuration_ouro.py'),'config_sha256':r.sha(ROOT/'model/config.json'),'query_runner_sha256':r.sha(__file__),'q_components_sha256':r.sha(ROOT/'q_patch_components.py'),'base_runner_sha256':r.sha(ROOT/'runner.py'),'observer_sha256':r.sha(ROOT/'observer.py'),'attestation_wrapper_sha256':r.sha(ROOT/'attested_runner.py'),'inputs_sha256':r.sha(args.inputs),'cohort_sha256':r.sha(args.cohort),'actual_selected_ids_sha256':hashlib.sha256(json.dumps([q['id'] for q in rows]).encode()).hexdigest(),'donor_map_sha256':hashlib.sha256(donor_bytes).hexdigest(),'donor_rule':'LEXICOGRAPHIC_ID_SORT_CYCLIC_BEFORE_OUTCOMES','precision_ids':precision,'torch':torch.__version__,'transformers':r.transformers.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16','attention':'sdpa','batch_size':1,'use_cache':False,'prior_results':str(prior),'prior_receipt_sha256':r.sha(prior/'receipt.json'),'technical':args.technical}
 oldidentity=pr['identity']
 for k in ('model_sha256','model_source_sha256','config_sha256','torch','transformers','cuda','gpu','dtype','attention','batch_size','use_cache'):assert identity[k]==oldidentity[k]
 ip=out/'execution_identity.json'
 if ip.exists():assert json.loads(ip.read_text())==identity
 r.write(ip,identity);started=time.monotonic();model=r.AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;versions=[(n,id(p),p._version) for n,p in model.named_parameters()]
 with torch.inference_mode():collect(model,rows,out,prior,identity);replay(model,rows,out,identity,donor_map,args.technical)
 assert versions==[(n,id(p),p._version) for n,p in model.named_parameters()]
 r.write(out/'receipt.json',{'status':'COMPLETE','N':len(rows),'identity':identity,'wall_seconds':time.monotonic()-started,**r.measure()})
if __name__=='__main__':main()
