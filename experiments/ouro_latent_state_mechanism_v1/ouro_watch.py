"""Single-owner, fail-closed GPU watcher. Does not grant exclusive GPU ownership."""
import argparse,contextlib,ctypes,datetime,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;REPO=R.parents[1];ROOT=REPO.parent;PY=ROOT/'conda-env/bin/python';RESULTS=R/'results';GIB=1024**3
sys.path.insert(0,str(R))
from cohort_runner import bound_hashes,split_data_path,sha
from manifest_guard import validate_manifest

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def gpu_snapshot():
 args=['nvidia-smi','--query-gpu=memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']
 rows=subprocess.check_output(args,text=True,timeout=15).strip().splitlines()
 if len(rows)!=1:raise RuntimeError('expected exactly one authorized GPU')
 total,used,free,util=[int(x.strip()) for x in rows[0].split(',')]
 text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
 pids=[] if not text or text.startswith('No running') else [int(x.strip()) for x in text.splitlines()]
 return {'total_mib':total,'used_mib':used,'free_mib':free,'util_percent':util,'compute_pids':pids}
def eligible(x):return x['free_mib']>=40*1024 and x['free_mib']/x['total_mib']>=0.90 and x['util_percent']<10 and not x['compute_pids']
def meminfo():
 data={}
 for line in Path('/proc/meminfo').read_text().splitlines():
  k,_,v=line.partition(':')
  if k in ('MemAvailable','MemTotal'):data[k]=int(v.split()[0])*1024
 return data
def machine():return {'gpu':gpu_snapshot(),'disk_free_bytes':shutil.disk_usage(ROOT).free,'memory':meminfo()}
def atomic_json(path,obj):
 temp=path.with_name(path.name+'.partial');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');os.replace(temp,path)
def require_regular_tree():
 for p in (ROOT,REPO,R,RESULTS,ROOT/'model',ROOT/'data',ROOT/'conda-env'):
  if p.is_symlink() or not p.is_dir():raise RuntimeError('unsafe/missing project directory '+str(p))
def external_tasks():
 output=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True,timeout=15);hits=[]
 for line in output.splitlines():
  stripped=line.strip();pid_text,_,args=stripped.partition(' ')
  if not pid_text.isdigit() or int(pid_text)==os.getpid():continue
  if any(x in args for x in ('gpu_parity_gate.py','resource_probe.py','cohort_runner.py','audit_coverage.py','behavior_summary.py','representation_observation.py','summarize_representations.py')):hits.append(int(pid_text))
 return hits
def fresh_outputs():
 for p in [ROOT/'shards',RESULTS/'GPU_RESOURCE_PROBE.json']+[RESULTS/f'GPU_{kind}_{split}.json' for kind in ('PARITY','GATE') for split in ('train','validation','test')]+[ROOT/f'{prefix}_{split}{ext}' for split in ('train','validation','test') for prefix,ext in (('coverage','.json'),('behavior','.json'),('representations','.jsonl'),('representation_summary','.json'))]:
  if os.path.lexists(p):raise RuntimeError('existing output requires manual review: '+str(p))
def preflight(expected_commit):
 require_regular_tree();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 if commit!=expected_commit:raise RuntimeError('unexpected source commit')
 if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise RuntimeError('repository not clean at startup')
 if external_tasks():raise RuntimeError('existing project task process')
 frozen=sorted((R/'environment_pip_freeze.txt').read_text().splitlines());current=sorted(subprocess.check_output([str(PY),'-m','pip','freeze'],text=True,timeout=60).splitlines())
 if frozen!=current:raise RuntimeError('Python environment drift')
 info={}
 for split in ('train','validation','test'):
  info[split]=validate_manifest(split,split_data_path(split),RESULTS/'INPUT_MANIFEST_SUMMARY.json',RESULTS/f'INPUT_MANIFEST_{split}.jsonl')
 package=json.loads((R/'SOURCE_PACKAGE_MANIFEST.json').read_text());archive=ROOT/'source_7c3aa9cffd10.zip'
 if sha(archive)!=package['archive_sha256']:raise RuntimeError('supplied archive hash mismatch')
 for e in package['entries']:
  path=R/e['path']
  if path.is_symlink() or not path.is_file() or sha(path)!=e['sha256']:raise RuntimeError('supplied package drift: '+e['path'])
 est=json.loads((RESULTS/'STORAGE_ESTIMATE.json').read_text());payload=sum(x['state_bytes']+x['vocab_logit_bytes'] for x in est['splits'].values());required=int(payload*1.20)+20*GIB
 if shutil.disk_usage(ROOT).free<required:raise RuntimeError('insufficient storage reserve before start')
 return {'commit':commit,'input_manifests':info,'bound_hashes':{s:bound_hashes(s) for s in ('train','validation','test')},'package_sha256':package['archive_sha256'],'expected_payload_bytes':payload,'required_initial_free_bytes':required,'source_hashes':{p.name:sha(p) for p in sorted(R.glob('*.py'))},'gpu_exclusivity':'OS flock protects this watcher only; no exclusive GPU ownership'}
def gate_audit(split,span_review,output_dir):
 import pyarrow.parquet as pq
 gate=json.loads((output_dir/f'GPU_GATE_{split}.json').read_text());evidence_path=(ROOT/gate['evidence_path']).resolve();evidence_path.relative_to(ROOT);evidence=json.loads(evidence_path.read_text())
 if gate.get('status')!='passed' or gate.get('sha256')!=bound_hashes(split) or gate.get('producer_sha256')!=sha(R/'gpu_parity_gate.py') or gate.get('evidence_sha256')!=sha(evidence_path):raise RuntimeError('invalid gate provenance '+split)
 required={'historical_scorer','independent_depths','identity_hook','bf16_roundtrip','repeatability','single_sequence','causal_mask'}
 if any(gate.get('checks',{}).get(k)!='passed' for k in required) or evidence.get('max_allowed_error')!=0.0:raise RuntimeError('incomplete gate checks')
 rows=pq.read_table(split_data_path('validation')).to_pylist();by_id={r['id']:r for r in rows};selected={r['id'] for r in span_review['records']};records=evidence['records'];seen=set()
 if len(selected)!=3 or {r['id'] for r in records}!=selected:raise RuntimeError('gate sample does not match span review')
 for r in records:
  ident=r['id'];candidate=r['candidate'];key=(ident,candidate)
  if key in seen or ident not in by_id or r['evaluation_split']!='validation' or r['target_split']!=split:raise RuntimeError('gate duplicate/wrong identity')
  seen.add(key)
  if any(r[k]!=0.0 for k in ('causal_suffix_max_error','repeated_max_error','hook_max_error','identity_max_error')):raise RuntimeError('nonzero gate discrepancy')
  if len(r['loops'])!=4 or any(x['k']!=k or x['historical']!=x['capture'] or x['absolute_error']!=0 or x['depth_max_error']!=0 for k,x in enumerate(r['loops'],1)):raise RuntimeError('invalid loop evidence')
 expected={(ident,j) for ident in selected for j in range(len(by_id[ident]['choices']['text']))}
 if seen!=expected:raise RuntimeError('gate candidate coverage mismatch')
 return {'records':len(records),'ids':sorted(selected),'evidence_sha256':sha(evidence_path)}
class WatchInterrupted(Exception):pass
def install_signal_handlers():
 def stop(signum,frame):raise WatchInterrupted('controller received signal '+str(signum))
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
def _child_death_signal(expected_parent=None,restore_mask=None):
 libc=ctypes.CDLL(None,use_errno=True)
 if libc.prctl(1,signal.SIGTERM,0,0,0)!=0:os._exit(125)
 if (expected_parent is not None and os.getppid()!=expected_parent) or os.getppid()==1:os._exit(125)
 if restore_mask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,restore_mask)
def group_live_members(pgid):
 text=subprocess.check_output(['ps','-eo','pid=,pgid=,uid=,stat='],text=True,timeout=15);members=[]
 for line in text.splitlines():
  fields=line.split()
  if len(fields)>=4 and int(fields[1])==pgid and not fields[3].startswith('Z'):
   if int(fields[2])!=os.getuid():raise RuntimeError('unexpected foreign process in own process group')
   members.append(int(fields[0]))
 return members
def terminate_process_group(proc):
 old={sig:signal.signal(sig,signal.SIG_IGN) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP)}
 try:
  members=group_live_members(proc.pid)
  if members:
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   deadline=time.monotonic()+30
   while group_live_members(proc.pid) and time.monotonic()<deadline:time.sleep(0.2)
   if group_live_members(proc.pid):
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass
   while group_live_members(proc.pid):time.sleep(1) # retain lock until no living member
  proc.wait()
 finally:
  for sig,handler in old.items():signal.signal(sig,handler)
@contextlib.contextmanager
def owned_process(args,**kwargs):
 watched={signal.SIGTERM,signal.SIGINT,signal.SIGHUP};previous=signal.pthread_sigmask(signal.SIG_BLOCK,watched);proc=None;parent=os.getpid()
 try:
  proc=subprocess.Popen(args,start_new_session=True,preexec_fn=lambda:_child_death_signal(parent,previous),**kwargs)
  signal.pthread_sigmask(signal.SIG_SETMASK,previous) # pending signal is delivered only after proc is registered
  yield proc
 finally:
  signal.pthread_sigmask(signal.SIG_BLOCK,watched)
  try:
   if proc is not None:terminate_process_group(proc)
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,previous)
def verify_snapshot(pre,expected_commit):
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=expected_commit:raise RuntimeError('source commit changed')
 if subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=REPO,text=True).strip():raise RuntimeError('source tree changed or untracked files appeared')
 current={p.name:sha(p) for p in sorted(R.glob('*.py'))}
 if current!=pre['source_hashes']:raise RuntimeError('source file set or hash changed')
 bound={}
 for mapping in pre['bound_hashes'].values():
  for path,digest in mapping.items():
   if path in bound and bound[path]!=digest:raise RuntimeError('inconsistent initial dependency hash')
   bound[path]=digest
 for path,digest in bound.items():
  p=ROOT/path
  if not p.is_file() or p.is_symlink() or sha(p)!=digest:raise RuntimeError('frozen dependency changed: '+path)
 frozen=sorted((R/'environment_pip_freeze.txt').read_text().splitlines());current=sorted(subprocess.check_output([str(PY),'-m','pip','freeze'],text=True,timeout=60).splitlines())
 if current!=frozen:raise RuntimeError('environment drift')
def run(run_dir,expected_commit):
 state={'run_id':run_dir.name,'pid':os.getpid(),'status':'INITIALIZING','stage':'preflight','updated_utc':utc(),'expected_commit':expected_commit,'note':'flock is not exclusive GPU ownership'}
 def update(status,stage,**extra):
  state.update(status=status,stage=stage,updated_utc=utc(),**extra);atomic_json(run_dir/'status.json',state)
 env=os.environ.copy();env.update(HF_HOME=str(ROOT/'cache/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'))
 def source_guard():verify_snapshot(pre,expected_commit)
 def stage(name,cmd,uses_gpu=False):
  source_guard();snap=machine()
  if snap['disk_free_bytes']<20*GIB or snap['memory']['MemAvailable']<8*GIB:raise RuntimeError('disk or RAM reserve below minimum')
  if uses_gpu and not eligible(snap['gpu']):raise RuntimeError('GPU changed or occupied immediately before '+name)
  update('RUNNING',name,resources=snap);log=run_dir/(name+'.log')
  with log.open('x',encoding='utf8') as f:
   with owned_process([str(PY),'-u']+[str(x) for x in cmd],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT) as proc:
    while proc.poll() is None:
     time.sleep(5);snap=machine();conflict=[]
     if uses_gpu:
      for pid in snap['gpu']['compute_pids']:
       try:
        if os.getpgid(pid)!=proc.pid:conflict.append(pid)
       except ProcessLookupError:conflict.append(pid)
     if conflict or snap['disk_free_bytes']<20*GIB or snap['memory']['MemAvailable']<4*GIB:raise RuntimeError('external GPU process or storage/memory reserve conflict')
    if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}; see {log}')
  source_guard()
 try:
  pre=preflight(expected_commit);atomic_json(run_dir/'provenance.json',pre);update('WAITING','waiting_for_gpu',resources=machine())
  # No automatic retries once execution begins. Query failures also stop.
  while True:
   first=machine();update('WAITING','waiting_for_gpu',resources=first)
   if eligible(first['gpu']) and first['disk_free_bytes']>=pre['required_initial_free_bytes'] and first['memory']['MemAvailable']>=12*GIB:
    time.sleep(10);second=machine()
    if eligible(second['gpu']) and second['disk_free_bytes']>=pre['required_initial_free_bytes'] and second['memory']['MemAvailable']>=12*GIB:break
   time.sleep(60)
  if external_tasks():raise RuntimeError('project task started while waiting')
  source_guard()
  span_path=run_dir/'span_review.json';stage('span_review',[R/'audit_gate_spans.py','--out',span_path]);span=json.loads(span_path.read_text())
  if span.get('status')!='passed' or len(span.get('records',[]))!=3:raise RuntimeError('decoded span audit incomplete')
  resource_path=run_dir/'GPU_RESOURCE_PROBE.json';stage('resource_probe',[R/'resource_probe.py','--out',resource_path],True);probe=json.loads(resource_path.read_text());cases=probe.get('cases',{})
  if set(cases)!={'longest_input','largest_question_capture'} or any(v['candidate_count']!=len(v['candidates']) or v['candidate_count']<2 or v['serialized_bytes']<=0 for v in cases.values()):raise RuntimeError('resource probe incomplete')
  if any(max(v['peak_allocated_bytes'],v['peak_reserved_bytes'])+(probe['device_total_bytes']-probe['pre_model_free_bytes'])>probe['device_total_bytes']-4*GIB for v in cases.values()):raise RuntimeError('resource probe leaves insufficient GPU headroom')
  if meminfo()['MemAvailable']<12*GIB or shutil.disk_usage(ROOT).free<pre['required_initial_free_bytes']:raise RuntimeError('resources inadequate after probe')
  audited={};gate_dir=run_dir/'gates';gate_dir.mkdir()
  for split in ('train','validation','test'):
   stage('gate_'+split,[R/'gpu_parity_gate.py','--split',split,'--count','3','--output-dir',gate_dir],True);audited[split]=gate_audit(split,span,gate_dir);atomic_json(run_dir/'gate_review.json',audited)
  for split in ('train','validation','test'):
   shard=run_dir/'shards'/split;coverage=run_dir/f'coverage_{split}.json';behavior=run_dir/f'behavior_{split}.json';obs=run_dir/f'representations_{split}.jsonl';summ=run_dir/f'representation_summary_{split}.json'
   stage('collect_'+split,[R/'cohort_runner.py','--split',split,'--gate',gate_dir/f'GPU_GATE_{split}.json','--out',shard],True)
   stage('coverage_'+split,[R/'audit_coverage.py','--split',split,'--shards',shard,'--out',coverage],True)
   cov=json.loads(coverage.read_text());cnt=cov['counts'];expected=pre['input_manifests'][split]
   if cnt['expected_questions']!=expected['questions'] or cnt['complete_questions']!=expected['questions'] or cnt['expected_requests']!=expected['requests'] or cnt['complete_requests']!=expected['requests'] or cnt['failed_questions'] or cnt['missing_questions'] or cov['score_replay']['device']!='cuda' or cov['score_replay']['status']!='passed':raise RuntimeError('coverage evidence incomplete')
   stage('behavior_'+split,[R/'behavior_summary.py','--split',split,'--shards',shard,'--coverage',coverage,'--out',behavior]);beh=json.loads(behavior.read_text())
   if beh['all']['n']!=expected['questions'] or sum(beh['all']['trajectory_counts'].values())!=expected['questions']:raise RuntimeError('behavior evidence incomplete')
   stage('representations_'+split,[R/'representation_observation.py','--split',split,'--shards',shard,'--coverage',coverage,'--out',obs]);stage('representation_summary_'+split,[R/'summarize_representations.py','--split',split,'--input',obs,'--coverage',coverage,'--out',summ]);sm=json.loads(summ.read_text())
   if sm['question_count']!=expected['questions'] or not sm['groups']:raise RuntimeError('representation evidence incomplete')
  update('OBSERVATION_READY','awaiting_astra_review',message='Complete descriptive observations only; no causal mechanism claimed')
 except BaseException as e:
  import traceback
  update('FAILED',state.get('stage','unknown'),error=repr(e),traceback=traceback.format_exc())
  raise
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run-dir',required=True);ap.add_argument('--expected-commit',required=True);a=ap.parse_args();directory=Path(a.run_dir).resolve();directory.relative_to((ROOT/'orchestration_runs').resolve())
 if not directory.is_dir() or any(p.name!='controller.log' for p in directory.iterdir()):raise RuntimeError('run directory must be new and otherwise empty')
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('another watcher owns the lock')
  install_signal_handlers()
  run(directory,a.expected_commit)
 finally:os.close(lock)
if __name__=='__main__':main()
