"""Full natural L18 causal swap; no VRAM preallocation; review before execution."""
import os,json,hashlib,fcntl,time,uuid,resource,re,shutil
from pathlib import Path
import torch,psutil
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file,load_file
import natural_l18_hook,natural_l18_runner,natural_sources,technical,full_input,state_capture,source_input
from source_input import ROOT,initial_sources,sha
from natural_l18_runner import question
from natural_sources import load as load_source
from technical import score
from state_capture import bits
if not __debug__:raise RuntimeError('optimized Python forbidden')
OUT=ROOT/'research_plans/natural_l18_full_run_20261111';TERMS=ROOT/'research_plans/NATURAL_L18_CAUSAL_EXECUTION_TERMS_20261111.md';DRAFT=ROOT/'research_plans/NATURAL_L18_CAUSAL_EXPERIMENT_DRAFT_20261111.md';REVIEW=ROOT/'research_plans/NATURAL_L18_ASTRA_PROBE_REVIEW_20261111.md';POLICY=ROOT/'research_plans/NATURAL_L18_RESOURCE_POLICY_UPDATE_20261111.md'
def require(x,msg):
 if not x:raise ValueError(msg)
def tensor_hash(t):return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
def model_hash(model):
 h=hashlib.sha256()
 for n,t in sorted(model.state_dict().items()):h.update(json.dumps([n,str(t.dtype),list(t.shape)],separators=(',',':')).encode());h.update(tensor_hash(t).encode())
 return h.hexdigest()
def write_json(path,data):
 tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
 try:
  with tmp.open('x') as f:json.dump(data,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  os.link(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()
def check_existing(split,ident,source,model,code_hashes):
 key=split+'_'+hashlib.sha256(ident.encode()).hexdigest();mp=OUT/(key+'.json');tp=OUT/(key+'.safetensors')
 if not mp.exists() and not tp.exists():return False
 require(mp.exists() and tp.exists(),'orphan pair');m=json.loads(mp.read_text());src=load_source(split,ident);e=src['natural']
 require(m['id']==ident and m['split']==split and m['tensor_sha256']==sha(tp) and m['source_metadata_sha256']==src['source_metadata_sha256'] and m['source_tensor_sha256']==src['source_tensor_sha256'] and m['natural_metadata_sha256']==e['metadata_sha256'] and m['natural_tensor_sha256']==e['tensor_sha256'] and m['combined_sha256']==src['combined_sha256'] and m['input_ids_sha256']==hashlib.sha256(json.dumps(e['ids']).encode()).hexdigest() and m['R']==e['R'] and m['targets']==e['targets'] and m['gold']==e['gold'] and m['source_hashes']==source and m['code_hashes']==code_hashes,'resume provenance')
 controls=m['controls'];require(set(controls)=={'T21_norm','T12_norm'} and all(v.get('status') in ('OK','UNDEFINED_ZERO_DONOR') for v in controls.values()),'controls')
 expected={'B1','B2','I1','I2','T21','T12'}|{k for k,v in controls.items() if v['status']=='OK'};t=load_file(str(tp));require(set(t)==expected==set(m['scores']),'conditions')
 for k,z in t.items():require(z.dtype==torch.bfloat16 and z.shape==(1,49152) and torch.isfinite(z).all() and score(z.to('cuda'),e['targets'])==m['scores'][k],'replay saved score')
 require(bits(t['B1'],e['logits'][0]) and bits(t['B2'],e['logits'][1]) and bits(t['I1'],t['B1']) and bits(t['I2'],t['B2']) and m['scores']['B1']==e['scores'][0] and m['scores']['B2']==e['scores'][1],'resume baseline/identity')
 if bits(src['states'][0]['a'],src['states'][1]['a']):require(bits(t['T21'],t['B2']) and bits(t['T12'],t['B1']),'equal donor anchor')
 return True
def main():
 require(sha(TERMS)=='bec2c0cf590d74ca6c4f9f54f105acbbbc126d252c4ef19dda5913d57a564f3d' and sha(DRAFT)=='335a1bd1cd12d4393a94b78b89a06fd2cb27494510c25f3adaad3bc682805448' and sha(REVIEW)=='cae0eb7763c880dcdc05f6be4d5097a5e70dd4c5013603bee244a815b201c396' and sha(POLICY)=='ee26a9ed93c0f9f555236756aa7716eb1a52e87813bedfec00f741100113fa72','plan/review')
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);OUT.mkdir(exist_ok=True);require(not (OUT/'receipt.json').exists(),'already complete')
  sources=initial_sources();sources.update({'terms':TERMS,'draft':DRAFT,'combined':natural_sources.COMBINED,'gpu_replay':natural_sources.GPU,'probe_review':REVIEW,'resource_policy':POLICY});start={k:sha(p) for k,p in sources.items()}
  modules=(natural_l18_hook,natural_l18_runner,natural_sources,technical,full_input,state_capture,source_input);paths={'runner':Path(__file__).resolve(),**{m.__name__:Path(m.__file__).resolve() for m in modules}};code_hashes={k:sha(p) for k,p in paths.items()};stats={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in list(sources.values())+list(paths.values())}
  covers={s:full_input.coverage(s) for s in ('train','validation','test')};pattern=re.compile(r'^(train|validation|test)_[0-9a-f]{64}$')
  def quarantine(p):
   require(p.is_file() and not p.is_symlink(),'unsafe orphan');d=OUT/'orphans';d.mkdir(exist_ok=True);os.rename(p,d/(p.name+'.'+uuid.uuid4().hex))
  for p in OUT.glob('*.partial'):quarantine(p)
  for p in OUT.glob('*.safetensors'):
   require(pattern.fullmatch(p.stem),'unexpected tensor');
   if not p.with_suffix('.json').exists():quarantine(p)
  for p in OUT.glob('*.json'):require(pattern.fullmatch(p.stem) and p.with_suffix('.safetensors').exists(),'unexpected/orphan metadata')
  require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'cuda/bf16');print('GPU_INITIAL_FREE',torch.cuda.mem_get_info()[0],flush=True);torch.cuda.reset_peak_memory_stats()
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False;require(not model.training and model.config._attn_implementation=='sdpa' and next(model.parameters()).dtype==torch.bfloat16,'model')
  initial_model_hash=model_hash(model);params=lambda:[(n,id(t),t.data_ptr(),t._version,str(t.dtype),str(t.device),tuple(t.shape)) for n,t in list(model.named_parameters())+list(model.named_buffers())];versions=params()
  def guard():require(params()==versions and all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in stats.items()),'source/model drift')
  done={s:0 for s in covers};started=time.monotonic()
  for split,rows in covers.items():
   for q in rows:
    guard();require(shutil.disk_usage(ROOT).free>20*1024**3 and psutil.virtual_memory().available>4*1024**3,'disk/RAM low');ident=q['id'];key=split+'_'+hashlib.sha256(ident.encode()).hexdigest()
    if check_existing(split,ident,start,model,code_hashes):done[split]+=1;continue
    result=question(model,split,ident,guard);t={k:v.detach().cpu().contiguous().clone() for k,v in result.pop('logits').items()};tmp=OUT/(key+'.'+uuid.uuid4().hex+'.partial');tp=OUT/(key+'.safetensors')
    try:
     save_file(t,str(tmp));back=load_file(str(tmp));require(set(t)==set(back) and all(bits(t[k],back[k]) for k in t),'save/readback');os.link(tmp,tp)
    finally:
     if tmp.exists():tmp.unlink()
    scores=result['scores'];gold=result['gold'];result['fixed_wrong']=max((j for j in range(len(result['targets'])) if j!=gold),key=lambda j:scores['B1'][j]);result['predictions']={k:max(range(len(v)),key=lambda j:v[j]) for k,v in scores.items()};result['top_ties']={k:sum(x==max(v) for x in v)>1 for k,v in scores.items()};result['tensor_sha256']=sha(tp);result['source_hashes']=start;result['code_hashes']=code_hashes
    write_json(OUT/(key+'.json'),result);require(check_existing(split,ident,start,model,code_hashes),'saved verification');done[split]+=1;print('QUESTION_COMPLETE',split,done[split],ident,flush=True)
  guard();require(done=={'train':1119,'validation':299,'test':1172} and {k:sha(v) for k,v in sources.items()}==start and {k:sha(v) for k,v in paths.items()}==code_hashes and model_hash(model)==initial_model_hash,'final')
  write_json(OUT/'receipt.json',{'status':'FULL_L18_COLLECTION_PENDING_INDEPENDENT_COVERAGE','counts':done,'source_hashes':start,'code_hashes':code_hashes,'model_state_sha256':initial_model_hash,'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},'elapsed_s':time.monotonic()-started,'peak_allocated':torch.cuda.max_memory_allocated(),'peak_reserved':torch.cuda.max_memory_reserved(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss});print('FULL_COLLECTION_COMPLETE',flush=True)
 finally:os.close(lock)
if __name__=='__main__':main()
