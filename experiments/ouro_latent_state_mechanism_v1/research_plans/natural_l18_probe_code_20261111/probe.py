"""Single supervised three-case L18 probe. Requires Astra code review before use."""
import os,json,hashlib,fcntl,subprocess,time,resource
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file,load_file
import natural_l18_hook,natural_l18_runner,natural_sources,replay_probe,technical,full_input,state_capture,source_input,full_runner_draft
from source_input import ROOT,SEL,initial_sources,sha
from natural_l18_runner import question
from state_capture import bits
if not __debug__:raise RuntimeError('optimized Python forbidden')
OUT=ROOT/'research_plans/natural_l18_probe_20261111'
TERMS=ROOT/'research_plans/NATURAL_L18_CAUSAL_EXECUTION_TERMS_20261111.md'
DRAFT=ROOT/'research_plans/NATURAL_L18_CAUSAL_EXPERIMENT_DRAFT_20261111.md'
def require(x,msg):
 if not x:raise ValueError(msg)
def tensor_hash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def model_hash(model):
 h=hashlib.sha256()
 for n,t in sorted(model.state_dict().items()):h.update(json.dumps([n,str(t.dtype),list(t.shape)],separators=(',',':')).encode());h.update(tensor_hash(t).encode())
 return h.hexdigest()
def write_json(path,data):
 tmp=path.with_suffix(path.suffix+'.partial')
 try:
  with tmp.open('x') as f:json.dump(data,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.link(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()
def main():
 require(sha(TERMS)=='bec2c0cf590d74ca6c4f9f54f105acbbbc126d252c4ef19dda5913d57a564f3d' and sha(DRAFT)=='335a1bd1cd12d4393a94b78b89a06fd2cb27494510c25f3adaad3bc682805448','plan hash')
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False)
  sources=initial_sources();paths={**sources,'terms':TERMS,'draft':DRAFT,'probe':Path(__file__).resolve()}
  for module in (natural_l18_hook,natural_l18_runner,natural_sources,replay_probe,technical,full_input,state_capture,source_input,full_runner_draft):paths['code_'+module.__name__]=Path(module.__file__).resolve()
  start={k:sha(p) for k,p in paths.items()};stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths.values()}
  selection=json.loads(SEL.read_text())['selection'];require(set(('A','B','C'))<=set(selection),'selection')
  free=torch.cuda.mem_get_info()[0];require(torch.cuda.is_available() and torch.cuda.is_bf16_supported() and free>3850*1024**2,'GPU resources');torch.cuda.reset_peak_memory_stats()
  if free>40*1024**3+512*1024**2:reservation=full_runner_draft.reserve_to_target()
  else:reservation={'status':'UNAVAILABLE','free_bytes':free}
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False
  require(not model.training and model.config._attn_implementation=='sdpa' and next(model.parameters()).dtype==torch.bfloat16,'model mode')
  initial_model_hash=model_hash(model);params=lambda:[(n,id(t),t.data_ptr(),t._version,str(t.dtype),str(t.device),tuple(t.shape)) for n,t in list(model.named_parameters())+list(model.named_buffers())];versions=params()
  def guard():
   require(params()==versions and all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in stats.items()),'model/source drift')
  collected=[];t0=time.monotonic()
  for label in ('A','B','C'):
   s=selection[label];result=question(model,s['split'],s['id'],guard);require(result['input_ids_sha256']==s['input_sha256'],'selection input')
   tensors={k:v.detach().cpu().contiguous().clone() for k,v in result.pop('logits').items()};tmp=OUT/(label+'.safetensors.partial');tp=OUT/(label+'.safetensors')
   save_file(tensors,str(tmp));back=load_file(str(tmp));require(set(back)==set(tensors) and all(bits(back[k],v) for k,v in tensors.items()),'tensor readback');os.link(tmp,tp);tmp.unlink()
   scores=result['scores'];gold=result['gold'];result['fixed_wrong']=max((j for j in range(len(result['targets'])) if j!=gold),key=lambda j:scores['B1'][j]);result['predictions']={k:max(range(len(v)),key=lambda j:v[j]) for k,v in scores.items()};result['top_ties']={k:sum(x==max(v) for x in v)>1 for k,v in scores.items()};result['label']=label;result['tensor_sha256']=sha(tp)
   mp=OUT/(label+'.json');write_json(mp,result);collected.append({'label':label,'id':result['id'],'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)});print('PROBE_CASE_COMPLETE',label,flush=True)
  guard();require({k:sha(p) for k,p in paths.items()}==start and model_hash(model)==initial_model_hash,'final model/source hash');pending={'status':'PENDING_REPLAY','questions':collected,'source_hashes':start,'model_state_sha256':initial_model_hash,'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'sdpa':model.config._attn_implementation},'reservation':reservation,'elapsed_s':time.monotonic()-t0,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};write_json(OUT/'probe_pending.json',pending)
  env=os.environ.copy();cmd=[os.sys.executable,'-u',str(Path(replay_probe.__file__).resolve()),str(OUT)];subprocess.run(cmd,env=env,check=True,timeout=300)
  r=json.loads((OUT/'replay.json').read_text());require(r['status']=='PROBE_CUDA_REPLAY_PASSED' and r['pending_sha256']==sha(OUT/'probe_pending.json'),'replay');guard();require({k:sha(p) for k,p in paths.items()}==start,'source after replay')
  write_json(OUT/'receipt.json',{'status':'THREE_CASE_PROBE_AWAITING_ASTRA_REVIEW','pending_sha256':sha(OUT/'probe_pending.json'),'replay_sha256':sha(OUT/'replay.json'),'questions':collected});print('THREE_CASE_PROBE_COMPLETE',flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
