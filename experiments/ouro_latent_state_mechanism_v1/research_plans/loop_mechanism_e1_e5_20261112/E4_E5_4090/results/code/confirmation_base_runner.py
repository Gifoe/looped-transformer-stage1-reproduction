"""Order006 reduced baseline: native4, identity4, natural K2 KV rollback and two controls only."""
import argparse,json,os,hashlib,time,fcntl
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
import runner as r
import attested_runner
from observer import bits,thash,hybrid
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=Path(__file__).resolve().parent
SCOPE='NATIVE_K1_K4_IDENTITY_K1_K4_K2_KV_TARGET_NORM_UNRELATED_ONLY'
def save(path,t):
 tmp=Path(str(path)+'.partial');save_file(t,str(tmp));back=load_file(str(tmp));assert all(bits(v,back[k]) for k,v in t.items());os.replace(tmp,path)
def collect(model,rows,out,identity):
 for i,q in enumerate(rows):
  np=out/(r.key(q)+'.native.json');tp=np.with_suffix('.safetensors')
  if np.exists():
   m=json.loads(np.read_text());assert m['identity']==identity and m['tensor_sha256']==r.sha(tp);continue
  start=time.monotonic();t={};native={};captures={};receipts={}
  for depth in (1,2,3,4):
   row,rec=r.run(model,q,depth,capture=depth in (1,2));native[str(depth)]=row
   if depth in (1,2):captures[depth]=rec.pop('qkv')
   receipts[str(depth)]={k:v for k,v in rec.items() if not torch.is_tensor(v)}
   for name in ('O','a_R','x_R','final_norm_R'):t[f'K{depth}_{name}']=rec[name].cpu().contiguous()
  t['2_1_target']=hybrid(captures[2],captures[1],q['R']).cpu().contiguous();del captures;save(tp,t)
  r.write(np,{'id':q['id'],'split':q['split'],'R':q['R'],'targets':q['targets'],'input_ids_sha256':q['input_ids_sha256'],'metadata_sha256':q['metadata_sha256'],'identity':identity,'native':native,'capture_receipts':receipts,'tensor_sha256':r.sha(tp),'scope':SCOPE,'total_wall_seconds':time.monotonic()-start})
  print('CONFIRM_BASE_NATIVE_COMPLETE',i+1,len(rows),q['id'],flush=True)
def replay(model,rows,out,identity,donors):
 byid={q['id']:q for q in rows}
 for i,q in enumerate(rows):
  rp=out/(r.key(q)+'.results.json');np=out/(r.key(q)+'.native.json');m=json.loads(np.read_text());t=load_file(str(np.with_suffix('.safetensors')))
  if rp.exists():assert json.loads(rp.read_text())['identity']==identity;continue
  start=time.monotonic();result=[];raw={}
  for depth in (1,2,3,4):
   row,rec=r.run(model,q,depth,t[f'K{depth}_O']);assert row['full_logits_sha256']==m['native'][str(depth)]['full_logits_sha256'];result.append({'receiver_loop':depth,'donor_loop':depth,'condition':'identity','donor_id':q['id'],'outside_R_unchanged':rec['outside_R_unchanged'],**row})
  base=t['K2_O'];target=t['2_1_target'];delta=target.float()-base.float();donor=donors[q['id']];fallback=[]
  for unused in range(len(rows)-1):
   ot=load_file(str((out/(r.key(byid[donor])+'.native.json')).with_suffix('.safetensors')));od=ot['2_1_target'].float()-ot['K2_O'].float()
   if float(od.norm())>0:break
   fallback.append(donor);donor=donors[donor]
  assert donor!=q['id'] and float(od.norm())>0 and float(base.float().norm())>0
  conditions={'target':target,'norm_only':(base.float()*(target.float().norm()/base.float().norm())).to(base.dtype),'unrelated_delta':(base.float()+delta.norm()*od/od.norm()).to(base.dtype)}
  for name,value in conditions.items():
   row,rec=r.run(model,q,2,value);assert bits(rec['O'].cpu(),base)
   result.append({'pair':'1-2','direction':'rollback','receiver_loop':2,'donor_loop':1,'condition':name,'donor_id':donor if name=='unrelated_delta' else q['id'],'donor_fallbacks':fallback if name=='unrelated_delta' else [],'native':m['native']['2'],'native_sdpa_R_sha256':thash(base),'injected_sha256':thash(value),'a_R_sha256':thash(rec['a_R']),'final_norm_R_sha256':thash(rec['final_norm_R']),'target_delta_norm':float(delta.norm()),'actual_delta_norm':float((value.float()-base.float()).norm()),'native_norm':float(base.float().norm()),'injected_norm':float(value.float().norm()),'cosine':float(torch.nn.functional.cosine_similarity(value.float().flatten(),base.float().flatten(),dim=0)),'outside_R_unchanged':rec['outside_R_unchanged'],**row})
   for key in ('injected','a_R','final_norm_R'):raw[name+'_'+key]=rec[key].cpu().contiguous()
  ip=out/(r.key(q)+'.interventions.safetensors');save(ip,raw)
  r.write(rp,{'id':q['id'],'split':q['split'],'identity':identity,'native_file_sha256':r.sha(np),'scope':SCOPE,'rows':result,'interventions_tensor_sha256':r.sha(ip),'total_wall_seconds':time.monotonic()-start});print('CONFIRM_BASE_QUESTION_COMPLETE',i+1,len(rows),q['id'],flush=True)
def main():
 a=argparse.ArgumentParser();a.add_argument('--inputs',required=True);a.add_argument('--cohort',required=True);a.add_argument('--output',required=True);a.add_argument('--limit',type=int);args=a.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);lock=open(ROOT/'gpu.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cohort=json.loads(Path(args.cohort).read_text());allrows={q['id']:q for q in map(json.loads,Path(args.inputs).read_text().splitlines())};rows=[allrows[i] for i in cohort['ids']]
 if args.limit:rows=rows[:args.limit]
 assert len(rows)>=2 and len(rows)==len({q['id'] for q in rows});ordered=sorted(q['id'] for q in rows);donors={ident:ordered[(i+1)%len(ordered)] for i,ident in enumerate(ordered)};db=json.dumps(donors,indent=2).encode();dp=out/'control_donor_map.json'
 if dp.exists():assert dp.read_bytes()==db
 else:dp.write_bytes(db)
 identity={'experiment':'ASTRA_ORDER006_REDUCED_NATIVE_AND_K2_KV_CONFIRMATION','scope':SCOPE,'model_sha256':r.sha(ROOT/'model/model.safetensors'),'model_source_sha256':r.sha(ROOT/'model/modeling_ouro.py'),'configuration_sha256':r.sha(ROOT/'model/configuration_ouro.py'),'config_sha256':r.sha(ROOT/'model/config.json'),'base_confirmation_sha256':r.sha(__file__),'base_runner_sha256':r.sha(ROOT/'runner.py'),'observer_sha256':r.sha(ROOT/'observer.py'),'attestation_wrapper_sha256':r.sha(ROOT/'attested_runner.py'),'inputs_sha256':r.sha(args.inputs),'cohort_sha256':r.sha(args.cohort),'actual_selected_ids_sha256':hashlib.sha256(json.dumps([q['id'] for q in rows]).encode()).hexdigest(),'donor_map_sha256':hashlib.sha256(db).hexdigest(),'donor_rule':'LEXICOGRAPHIC_ID_SORT_CYCLIC_BEFORE_OUTCOMES','torch':torch.__version__,'transformers':r.transformers.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'dtype':'bfloat16','attention':'sdpa','batch_size':1,'use_cache':False}
 assert identity['model_sha256']=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af';p=out/'execution_identity.json'
 if p.exists():assert json.loads(p.read_text())==identity
 r.write(p,identity);start=time.monotonic();model=r.AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();model.config.use_cache=False;model.model.config.use_cache=False;versions=[(n,id(p),p._version) for n,p in model.named_parameters()]
 with torch.inference_mode():collect(model,rows,out,identity);replay(model,rows,out,identity,donors)
 assert versions==[(n,id(p),p._version) for n,p in model.named_parameters()];r.write(out/'receipt.json',{'status':'COMPLETE','N':len(rows),'scope':SCOPE,'implemented_conditions':{'native_depths':[1,2,3,4],'identity_depths':[1,2,3,4],'interventions':['K2_receiver_K1_KV_target','K2_receiver_K1_KV_norm_only','K2_receiver_K1_KV_unrelated_delta']},'unexecuted_original_six_direction_design':True,'identity':identity,'wall_seconds':time.monotonic()-start,**r.measure()})
if __name__=='__main__':main()
