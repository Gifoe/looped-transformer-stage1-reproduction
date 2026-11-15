"""AB1 fixed-three technical collector; no scientific queue or effect selection."""
import os,sys,json,pathlib,hashlib,time,traceback,fcntl,subprocess,shutil,resource
from ab1_loader_draft import load_model,sha
from ab1_runner_draft import collect_case,th
from ab1_storage import serialize_exclusive
from ab1_e2_adapter_draft import apply_frozen_heads
import torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
BASE=pathlib.Path(__file__).resolve().parent.parent;MODEL=pathlib.Path('/root/autodl-tmp/ouro-e4-e5/model');OUT=BASE/'technical_three_v2';LOCK=pathlib.Path('/root/autodl-tmp/ouro-e6a/gpu.lock');MAN=BASE/'inputs/CPU_PREPARATION_MANIFEST.json';NPZ=BASE/'inputs/selected_models.npz';ORDER=BASE/'orders/ASTRA_ORDER_AB1_RECURRENT_BOUNDARY_v1.md'
EXPECTED_ORDER='6b06d39d959084e72db0aa05570c15f913e3456301a6b24a97187e8f7fdbe8a8';EXPECTED_MAN='6aed74c334e2b6365f00b47a6424f6eba6070705a15dce0abc94509d421943b5';EXPECTED_NPZ='204a3788301c7aa826a198b9e40ce495ef81284b6cd41a4758dfd942f8650890'
def write_exclusive(path,x):
 with path.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def resources():
 data=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines();assert len(data)==1;name,free=data[0].rsplit(',',1);free=int(free.strip());jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();disk=shutil.disk_usage(BASE).free
 if name.strip()!='NVIDIA GeForce RTX 4090' or free<20480 or disk<8*1024**3:raise RuntimeError('resource guard: expected RTX4090, >=20GiB free, >=8GiB disk for technical stage')
 return {'gpu':name.strip(),'free_mib':free,'compute_processes':jobs.splitlines() if jobs else [],'disk_free_bytes':disk}
def model_state(model):
 h=hashlib.sha256()
 for n,t in sorted(model.state_dict().items()):h.update(json.dumps([n,str(t.dtype),list(t.shape)],separators=(',',':')).encode());h.update(th(t).encode())
 return h.hexdigest()
def main():
 assert sha(ORDER)==EXPECTED_ORDER and sha(MAN)==EXPECTED_MAN and sha(NPZ)==EXPECTED_NPZ;inp=json.loads(MAN.read_text());assert inp['status']=='CPU_INPUT_PREPARATION_BEFORE_AB1_OUTCOMES';p=BASE/'inputs/train_model_inputs.jsonl';assert sha(p)==inp['files_sha256'][p.name];ids=inp['technical_train_ids'];assert len(ids)==len(set(ids))==3;data={}
 for line in p.read_text().splitlines():
  q=json.loads(line)
  if q['id'] in ids:q['prefix_len']=inp['prefix_length'];data[q['id']]=q
 assert set(data)==set(ids)
 sources={str(x):sha(x) for x in sorted((BASE/'code').glob('*.py'))};sources.update({str(MAN):sha(MAN),str(p):sha(p),str(ORDER):sha(ORDER),str(NPZ):sha(NPZ)});r0=resources();lock=os.open(LOCK,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);r1=resources();OUT.mkdir(exist_ok=False);write_exclusive(OUT/'STARTED.json',{'status':'TECHNICAL_STARTED_NOT_SCIENCE','sources':sources,'before_lock':r0,'after_lock':r1,'technical_ids':ids,'utc_epoch':time.time()})
  start=time.monotonic();model,identity=load_model(MODEL,authorized=True,device='cuda:0');initial=model_state(model);print('AB1_TECH_MODEL_LOADED',flush=True);cases=[]
  for ident in ids:
   torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic();q=data[ident];meta,tensors=collect_case(model,q);condition_names=[k for k,v in meta['conditions'].items() if v['status']=='OK'];assert len(meta['conditions'])==14;meta['frozen_e2']=apply_frozen_heads(NPZ,tensors,q['targets'],condition_names,all_models=True);torch.cuda.synchronize();meta['question_synchronized_wall_includes_hook_and_e2_excludes_save']=time.monotonic()-t;meta['peak_allocated_bytes']=torch.cuda.max_memory_allocated();meta['peak_reserved_bytes']=torch.cuda.max_memory_reserved();meta['input_ids_sha256']=q['input_ids_sha256'];meta['R']=q['R'];meta['prefix_len']=q['prefix_len'];meta['targets']=q['targets'];case=serialize_exclusive(OUT,ident,meta,tensors);cases.append(case);print('AB1_TECH_CASE_SAVED',ident,flush=True)
  assert model_state(model)==initial and all(sha(pathlib.Path(k))==v for k,v in sources.items());write_exclusive(OUT/'PENDING_INDEPENDENT_REPLAY.json',{'status':'THREE_TECHNICAL_PENDING_INDEPENDENT_MODEL_REPLAY_AND_CONTROL_AUDIT','technical_ids':ids,'cases':cases,'source_sha256':sources,'model_state_sha256':initial,'model_identity':identity,'elapsed_after_lock_seconds':time.monotonic()-start,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'scope':'Technical only, no outcome selection, unobserved and observed native/every corner on selected same-site model; no independent replay yet.'});print('AB1_TECH_PENDING_REPLAY',flush=True)
 except BaseException as e:
  if OUT.is_dir() and not (OUT/'FAILED.json').exists():write_exclusive(OUT/'FAILED.json',{'status':'FAILED_STOPPED','type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
  raise
 finally:os.close(lock)
if __name__=='__main__':main()
