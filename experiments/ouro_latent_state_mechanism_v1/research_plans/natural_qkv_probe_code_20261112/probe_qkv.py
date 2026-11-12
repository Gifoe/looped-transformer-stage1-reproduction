"""Three-case Q_R x KV endpoint probe; no VRAM preallocation."""
import os,json,hashlib,fcntl,subprocess,time,resource
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file,load_file
import qkv_observer,qkv_runner,replay_qkv,natural_l18_hook,natural_sources,technical,full_input,state_capture,source_input
from source_input import ROOT,SEL,initial_sources,sha
from qkv_runner import question
from state_capture import bits
if not __debug__:raise RuntimeError('optimized Python forbidden')
OUT=ROOT/'research_plans/natural_qkv_probe_20261112';TERMS=ROOT/'research_plans/NATURAL_L18_QR_KV_FACTORIAL_EXECUTION_TERMS_20261112.md';L18COV=ROOT/'research_plans/NATURAL_L18_INDEPENDENT_COVERAGE_20261111.json'
def require(x,msg):
 if not x:raise ValueError(msg)
def thash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def model_hash(model):
 h=hashlib.sha256()
 for n,t in sorted(model.state_dict().items()):h.update(json.dumps([n,str(t.dtype),list(t.shape)],separators=(',',':')).encode());h.update(thash(t).encode())
 return h.hexdigest()
def write_json(path,data):
 tmp=path.with_suffix(path.suffix+'.partial')
 try:
  with tmp.open('x') as f:json.dump(data,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.link(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()
def main():
 require(sha(TERMS)=='c42fa7dd4021e4a77fcf68e9da110024f3d8df44e810aaa4d40ae21fea534b61' and sha(L18COV)=='da4478aef58b320f5aa079f85b06ac70eab524922c340b2326a9487020da2535','terms/coverage')
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=False)
  sources=initial_sources();paths={**sources,'terms':TERMS,'l18_coverage':L18COV,'probe':Path(__file__).resolve()}
  for module in (qkv_observer,qkv_runner,replay_qkv,natural_l18_hook,natural_sources,technical,full_input,state_capture,source_input):paths['code_'+module.__name__]=Path(module.__file__).resolve()
  start={k:sha(p) for k,p in paths.items()};stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in paths.values()};selection=json.loads(SEL.read_text())['selection'];require(set(('A','B','C'))<=set(selection),'selection')
  require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA BF16');free=torch.cuda.mem_get_info()[0];print('GPU_INITIAL_FREE',free,flush=True);torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False;require(not model.training and model.config._attn_implementation=='sdpa' and next(model.parameters()).dtype==torch.bfloat16,'model')
  initial_model_hash=model_hash(model);params=lambda:[(n,id(t),t.data_ptr(),t._version,str(t.dtype),str(t.device),tuple(t.shape)) for n,t in list(model.named_parameters())+list(model.named_buffers())];versions=params()
  def guard():require(params()==versions and all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in stats.items()),'model/source drift')
  collected=[];t0=time.monotonic()
  for label in ('A','B','C'):
   s=selection[label];r=question(model,s['split'],s['id'],guard);require(r['input_ids_sha256']==s['input_sha256'],'selection input');logits=r.pop('logits');inj=r.pop('injections');astates=r.pop('a_states');names=('F11','F21','F12','F22');require(set(logits)==set(inj)==set(astates)==set(names),'conditions')
   tensors={f'{k}_{suffix}':v.detach().cpu().contiguous().clone() for suffix,d in [('logits',logits),('sdpa_R',inj),('a_R',astates)] for k,v in d.items()};tmp=OUT/(label+'.safetensors.partial');tp=OUT/(label+'.safetensors');save_file(tensors,str(tmp));back=load_file(str(tmp));require(set(back)==set(tensors) and all(bits(back[k],v) for k,v in tensors.items()),'tensor readback');os.link(tmp,tp);tmp.unlink()
   scores=r['scores'];r['predictions']={k:max(range(len(v)),key=lambda j:v[j]) for k,v in scores.items()};r['top_ties']={k:sum(x==max(v) for x in v)>1 for k,v in scores.items()};r['label']=label;r['tensor_sha256']=sha(tp);mp=OUT/(label+'.json');write_json(mp,r);collected.append({'label':label,'id':r['id'],'metadata_sha256':sha(mp),'tensor_sha256':sha(tp)});print('QKV_PROBE_CASE_COMPLETE',label,flush=True)
  guard();require({k:sha(p) for k,p in paths.items()}==start and model_hash(model)==initial_model_hash,'model/source')
  pending={'status':'PENDING_REPLAY','questions':collected,'source_hashes':start,'model_state_sha256':initial_model_hash,'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'sdpa':model.config._attn_implementation,'kernel':'UNKNOWN'},'elapsed_s':time.monotonic()-t0,'gpu_initial_free':free,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};write_json(OUT/'probe_pending.json',pending)
  subprocess.run([os.sys.executable,'-u',str(Path(replay_qkv.__file__).resolve()),str(OUT)],env=os.environ.copy(),check=True,timeout=300);r=json.loads((OUT/'replay.json').read_text());require(r['status']=='QKV_PROBE_CUDA_REPLAY_PASSED' and r['pending_sha256']==sha(OUT/'probe_pending.json'),'replay');guard();require({k:sha(p) for k,p in paths.items()}==start,'source after replay');write_json(OUT/'receipt.json',{'status':'THREE_CASE_QKV_PROBE_AWAITING_ASTRA_REVIEW','pending_sha256':sha(OUT/'probe_pending.json'),'replay_sha256':sha(OUT/'replay.json'),'questions':collected});print('THREE_CASE_QKV_PROBE_COMPLETE',flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
