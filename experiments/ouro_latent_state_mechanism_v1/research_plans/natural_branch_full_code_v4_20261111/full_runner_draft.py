"""Full natural branch observation draft; CLI DISABLED pending Astra review."""
import os,json,fcntl,time,uuid,hashlib,shutil,re,torch,psutil
from pathlib import Path
from safetensors.torch import save_file,load_file
from transformers import AutoModelForCausalLM
from source_input import ROOT,initial_sources,sha
from full_input import coverage,load
from technical import verify,score
from state_capture import bits
from decompose import decompose

def write_once(p,data):
 tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial')
 try:
  with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.link(tmp,p)
 finally:
  if tmp.exists():tmp.unlink()
def source_files():
 result=initial_sources();here=Path(__file__).resolve().parent
 for n in ('state_capture.py','technical.py','decompose.py','source_input.py','full_input.py','full_runner_draft.py'):result['code_'+n]=here/n
 return result

def check_existing(out,key,e,start,device,model,eps):
 mp=out/(key+'.json');tp=out/(key+'.safetensors')
 if not mp.exists() and not tp.exists():return False
 if not mp.exists() or not tp.exists():raise RuntimeError('orphan: preserve and inspect')
 m=json.loads(mp.read_text())
 legacy=dict(start);legacy['code_full_runner_draft.py']='3441948e63da0b1ca02dc59636113715915254a944f32cb46fee0c10ce248d1b'
 legacy3=dict(start);legacy3['code_full_runner_draft.py']='d19f7fd0b6f1b6b03a73ce821851c1dc4a475b3ffa0138db27ddd19baa11be72'
 if m['id']!=e['id'] or m['split']!=e['split'] or m['natural_metadata_sha256']!=e['metadata_sha256'] or m['natural_tensor_sha256']!=e['tensor_sha256'] or m['source_hashes'] not in (start,legacy,legacy3) or m['probe_tensor_sha256']!=sha(tp) or m['prior_scores']!=e['scores'] or m['targets']!=e['targets'] or m['R_absolute']!=e['R'] or m['R_saved_row']!=e['saved_row'] or m['gold']!=e['gold']:raise ValueError('resume provenance')
 t=load_file(str(tp));expected={f'u{u}_layer{i:02d}_{k}' for u in (0,1) for i in range(24) for k in ('x','a','y','b','z')}|{f'loop{u}_logits' for u in (1,2)}|{f'loop{u}_post_norm' for u in (1,2)}
 if set(t)!=expected or len(m['attention'])!=48:raise ValueError('resume incomplete')
 trace={}
 for u in (0,1):
  for i in range(24):
   d={k:t[f'u{u}_layer{i:02d}_{k}'] for k in ('x','a','y','b','z')}
   if any(v.ndim!=2 or v.shape[0]!=1 or not torch.isfinite(v).all() for v in d.values()):raise ValueError('resume trace values')
   if not bits(d['x']+d['a'],d['y']) or not bits(d['y']+d['b'],d['z']):raise ValueError('resume residual arithmetic')
   if not bits(d['z'],e['states'][(u,i)]):raise ValueError('resume layer anchor')
   trace[(u,i)]=d
  if not bits(t[f'loop{u+1}_post_norm'],e['norms'][u]) or not bits(t[f'loop{u+1}_logits'],e['logits'][u]) or score(t[f'loop{u+1}_logits'].to(device),e['targets'])!=e['scores'][u]:raise ValueError('resume score anchor')
 if [a.get('loop_layer') for a in m['attention']]!=[[u,i] for u in (0,1) for i in range(24)]:raise ValueError('resume attention sequence')
 W=model.lm_head.weight[e['targets']].detach().float().cpu().numpy();gamma=model.model.norm.weight.detach().float().cpu().numpy()
 if m['lens']!=decompose(trace,W,gamma,eps,list(range(len(e['targets']))),e['gold']):raise ValueError('resume lens mismatch')
 return True

def run(out):
 out=Path(out).resolve();out.relative_to(ROOT/'research_plans');out.mkdir(exist_ok=True);lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  gpu_free=torch.cuda.mem_get_info()[0];disk_free=shutil.disk_usage(ROOT).free;ram_free=psutil.virtual_memory().available;required=3850*1024**2
  print('RESOURCE_PREFLIGHT',json.dumps({'gpu_free':gpu_free,'required_measured_probe_peak':required,'disk_free':disk_free,'ram_available':ram_free}),flush=True)
  if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported() or gpu_free<=required or disk_free<30*1024**3 or ram_free<8*1024**3:raise RuntimeError('resource preflight')
  source=source_files();start={k:sha(v) for k,v in source.items()};config=json.loads(source['config'].read_text());covers={s:coverage(s) for s in ('train','validation','test')}
  pattern=re.compile(r'^(train|validation|test)_[0-9a-f]{64}$')
  def quarantine(p):
   if p.is_symlink() or not p.is_file():raise RuntimeError('unexpected orphan path')
   orphan=out/'orphans';orphan.mkdir(exist_ok=True);os.rename(p,orphan/(p.name+'.'+uuid.uuid4().hex))
  for tmp in out.glob('*.partial'):quarantine(tmp)
  for tp in out.glob('*.safetensors'):
   if not pattern.fullmatch(tp.stem):raise RuntimeError('unexpected tensor filename')
   if not tp.with_suffix('.json').exists():quarantine(tp)
  for mp in out.glob('*.json'):
   if mp.name=='receipt.json':raise RuntimeError('run already complete')
   if not pattern.fullmatch(mp.stem) or not mp.with_suffix('.safetensors').exists():raise RuntimeError('unexpected or orphan metadata')
  start_stats={str(v):(v.stat().st_size,v.stat().st_mtime_ns) for v in source.values()}
  with torch.no_grad():model=AutoModelForCausalLM.from_pretrained(ROOT/'model',local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
  model.config.use_cache=False;model.model.config.use_cache=False
  runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),'attention':model.config._attn_implementation,'dtype':str(next(model.parameters()).dtype)}
  if model.training or runtime['attention']!='sdpa' or runtime['dtype']!='torch.bfloat16':raise ValueError('model mode')
  versions=[(id(t),t.data_ptr(),t._version) for t in list(model.parameters())+list(model.buffers())];done={s:0 for s in covers}
  for split,rows in covers.items():
   for q in rows:
    gpu_free=torch.cuda.mem_get_info()[0];disk_free=shutil.disk_usage(ROOT).free;ram_free=psutil.virtual_memory().available
    if disk_free<20*1024**3 or ram_free<4*1024**3:raise RuntimeError('resource reserve '+json.dumps({'gpu_free':gpu_free,'disk_free':disk_free,'ram_available':ram_free}))
    if any((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=v for p,v in start_stats.items()) or versions!=[(id(t),t.data_ptr(),t._version) for t in list(model.parameters())+list(model.buffers())]:raise ValueError('source drift')
    e=load(split,q)
    if any(e['runtime'][k]!=runtime[k] for k in ('torch','cuda','gpu')) or e['runtime']['attention']!='sdpa' or e['runtime']['dtype']!='bfloat16' or e['runtime']['use_cache'] is not False:raise ValueError('natural runtime mismatch')
    key=split+'_'+hashlib.sha256(e['id'].encode()).hexdigest()
    if check_existing(out,key,e,start,'cuda',model,config['rms_norm_eps']):done[split]+=1;continue
    ids=torch.tensor([e['ids']],dtype=torch.long,device='cuda');obs,logits=verify(model,ids,e['R'],e['targets'],e['states'],e['norms'],e['logits'],e['scores'])
    if any(v.dtype!=torch.bfloat16 for d in obs['states'].values() for v in d.values()):raise ValueError('state dtype')
    W=model.lm_head.weight[e['targets']].detach().float().cpu().numpy();gamma=model.model.norm.weight.detach().float().cpu().numpy();lens=decompose(obs['states'],W,gamma,config['rms_norm_eps'],list(range(len(e['targets']))),e['gold'])
    t={f'u{u}_layer{i:02d}_{k}':v.detach().cpu().contiguous().clone() for (u,i),d in obs['states'].items() for k,v in d.items()};t.update({f'loop{u+1}_logits':v.detach().cpu().contiguous().clone() for u,v in enumerate(logits)});t.update({f'loop{u+1}_post_norm':v.detach().cpu().contiguous().clone() for u,v in enumerate(obs['norms'])})
    tmp=out/(key+'.'+uuid.uuid4().hex+'.partial');tp=out/(key+'.safetensors')
    try:
     save_file(t,str(tmp));back=load_file(str(tmp))
     if set(t)!=set(back) or any(not bits(t[k],back[k]) for k in t):raise ValueError('tensor readback')
     os.link(tmp,tp)
    finally:
     if tmp.exists():tmp.unlink()
    meta={'id':e['id'],'split':split,'gold':e['gold'],'R_absolute':e['R'],'R_saved_row':e['saved_row'],'targets':e['targets'],'prior_scores':e['scores'],'natural_metadata_sha256':e['metadata_sha256'],'natural_tensor_sha256':e['tensor_sha256'],'probe_tensor_sha256':sha(tp),'attention':obs['attention'],'lens':lens,'kernel':'UNKNOWN','source_hashes':start,'runtime':runtime}
    write_once(out/(key+'.json'),(json.dumps(meta,sort_keys=True,allow_nan=False)+'\n').encode());check_existing(out,key,e,start,'cuda',model,config['rms_norm_eps']);done[split]+=1
    print('QUESTION_COMPLETE',split,done[split],e['id'],flush=True)
  if done!={'train':1119,'validation':299,'test':1172} or {k:sha(v) for k,v in source.items()}!=start:raise ValueError('final coverage/source')
  write_once(out/'receipt.json',(json.dumps({'status':'FULL_OBSERVATION_PENDING_INDEPENDENT_COVERAGE','counts':done,'source_hashes':start,'runtime':runtime},sort_keys=True)+'\n').encode())
 finally:os.close(lock)
if __name__=='__main__':raise SystemExit('DISABLED: code review and probe approval required')
