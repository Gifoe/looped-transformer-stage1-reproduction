"""Single AB1 natural baseline and boundary technical capture, no intervention."""
import os,json,pathlib,time,fcntl,subprocess,shutil,hashlib,traceback,resource
from ab1_loader_draft import load_model,sha
from ab1_boundary_hooks import capture_and_patch,bits
from ab1_storage import serialize_exclusive
from ab1_metrics import score
import torch
if not __debug__:raise RuntimeError('optimized mode forbidden')
BASE=pathlib.Path(__file__).resolve().parent.parent;MODEL=pathlib.Path('/home/luansiyu/loop-latent/model');OUT=BASE/'natural_technical_one';LOCK=pathlib.Path('/home/luansiyu/loop-latent/ouro_watch.lock');MAN=BASE/'inputs/CPU_PREPARATION_MANIFEST.json';ORDER=BASE/'orders/ASTRA_ORDER_AB1_RECURRENT_BOUNDARY_v1.md';NPZ=BASE/'inputs/selected_models.npz'
def write(path,x):
 with path.open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def resource_check():
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines();assert len(lines)==1;name,free=lines[0].rsplit(',',1);jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();disk=shutil.disk_usage(BASE).free
 if name.strip()!='NVIDIA RTX A6000' or int(free)<20480 or jobs or disk<8*1024**3:raise RuntimeError('resource guard: A6000, >=20GiB free, no other compute process and >=8GiB disk required')
 return {'gpu':name.strip(),'free_mib':int(free),'compute_processes':jobs.splitlines() if jobs else [],'disk_free_bytes':disk}
def main():
 assert sha(ORDER)=='6b06d39d959084e72db0aa05570c15f913e3456301a6b24a97187e8f7fdbe8a8' and sha(MAN)=='6aed74c334e2b6365f00b47a6424f6eba6070705a15dce0abc94509d421943b5';manifest=json.loads(MAN.read_text());inputs=BASE/'inputs/train_model_inputs.jsonl';assert sha(inputs)==manifest['files_sha256'][inputs.name];ident=manifest['technical_train_ids'][0];q=next(q for line in inputs.read_text().splitlines() if (q:=json.loads(line))['id']==ident);assert set(q)=={'id','split','ids','R','targets','input_ids_sha256'} and q['split']=='train';assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256'];P=manifest['prefix_length'];R=q['R'];T=len(q['ids']);assert R==T-1 and 0<P<R
 sources={str(p):sha(p) for p in [pathlib.Path(__file__).resolve(),BASE/'code/ab1_boundary_hooks.py',BASE/'code/ab1_loader_draft.py',BASE/'code/ab1_storage.py',BASE/'code/ab1_metrics.py',MAN,ORDER,inputs]};before=resource_check();lock=os.open(LOCK,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);after=resource_check();OUT.mkdir(exist_ok=False);write(OUT/'STARTED.json',{'status':'SINGLE_TECHNICAL_NATURAL_STARTED','sources':sources,'id':ident,'input_ids_sha256':q['input_ids_sha256'],'before_lock':before,'after_lock':after,'utc_epoch':time.time(),'scope':'natural source/observer identity only; no scientific intervention, no outcome analysis'})
  started=time.monotonic();model,identity=load_model(MODEL,authorized=True,device='cuda:0');params=[(p,p._version) for p in model.parameters()];ids=torch.tensor([q['ids']],device='cuda');torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
  with torch.inference_mode():
   _,base,_=model.model(ids,use_cache=False,past_key_values=None);base1=model.lm_head(base[0])[:,R,:].detach().clone();base2=model.lm_head(base[1])[:,R,:].detach().clone()
   with capture_and_patch(model,T,R,P) as rec:
    _,observed,_=model.model(ids,use_cache=False,past_key_values=None);z1=model.lm_head(observed[0])[:,R,:].detach().clone();z2=model.lm_head(observed[1])[:,R,:].detach().clone()
   assert bits(base1,z1) and bits(base2,z2) and bits(rec['B'],rec['B_at_L1']) and bits(rec['E'],model.model.embed_tokens(ids));assert all(p._version==v for p,v in params)
  torch.cuda.synchronize();elapsed=time.monotonic()-t;tensors={'E':rec['E'].cpu(),'B':rec['B'].cpu(),'Q9_N_full':rec['Q9_actual'].cpu(),'K1_logits':z1.cpu(),'K2_logits':z2.cpu()};meta={'id':ident,'input_ids_sha256':q['input_ids_sha256'],'R':R,'prefix_len':P,'targets':q['targets'],'observed_equals_unobserved_full_vocab_K1_K2':True,'B_full_sequence_equals_K2_L1':True,'E_equals_embedding':True,'L1_kwargs':rec['L1_kwargs'],'L9_kwargs':rec['L9_kwargs'],'K1_scores':score(z1,q['targets']),'K2_scores':score(z2,q['targets']),'model_identity':identity,'synchronized_2_forward_wall_seconds_excludes_save':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'source_sha256':sources};case=serialize_exclusive(OUT,ident,meta,tensors);assert all(sha(pathlib.Path(k))==v for k,v in sources.items());write(OUT/'NATURAL_ONE_COMPLETE.json',{'status':'SINGLE_TECHNICAL_NATURAL_CAPTURE_COMPLETE_NOT_FULL_TECH_GATE','case':case,'model_identity':identity,'total_elapsed_includes_model_load_seconds':time.monotonic()-started,'scope':'one natural example; no E-all/corner/control/Q9-cross, no independent replay, no scientific effects'});print('AB1_NATURAL_ONE_COMPLETE',ident,flush=True)
 except BaseException as e:
  if OUT.is_dir() and not (OUT/'FAILED.json').exists():write(OUT/'FAILED.json',{'status':'FAILED_STOPPED','error_type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()})
  raise
 finally:os.close(lock)
if __name__=='__main__':main()
