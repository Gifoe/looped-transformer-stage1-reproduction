import gzip,hashlib,json,os,pathlib,signal,subprocess,time,traceback
R=pathlib.Path(__file__).resolve().parents[1];PY='/root/autodl-tmp/ouro-e6a/venv/bin/python'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(pathlib.Path(p).read_text())
sf=R/'B2_FULL_SOURCE_FREEZE.json';science=read(sf)
assert all(sha(R/k)==v for k,v in science['sources'].items())
assert not (R/'parallel/TRANSITION_RECEIPT.json').exists()
assert not (R/'b2_data/treatments/train1030/PENDING.json').exists()
assert read(R/'B2_QUEUE_STATE.json')['step']=='treatments_train1030_async_v1'
def argv(pid):return [x.decode() for x in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
assert argv(12921)==[PY,str(R/'async_io/run_queue_async_v1.py')]
assert argv(12922)==[PY,str(R/'async_io/b2_collect_async_v1.py'),'--root',str(R),'--kind','treatment','--stage','train1030']
intent=dict(status='BEFORE_STOP',epoch=time.time(),old_supervisor=12921,old_collector=12922,science_freeze_sha256=sha(sf),script_sha256=sha(__file__))
(R/'parallel/TRANSITION_INTENT.json').write_text(json.dumps(intent,indent=2))
os.kill(12921,signal.SIGTERM);os.kill(12922,signal.SIGSTOP)
for _ in range(100):
 if pathlib.Path('/proc/12922/stat').read_text().split()[2] in ('T','t'):break
 time.sleep(.05)
write_paths=[]
for fd in pathlib.Path('/proc/12922/fd').iterdir():
 try:
  target=fd.resolve();info=pathlib.Path('/proc/12922/fdinfo')/fd.name
  flags=int(next(x.split()[1] for x in info.read_text().splitlines() if x.startswith('flags:')),8)
  if target.is_relative_to(R/'b2_data/treatments/train1030') and flags & (os.O_WRONLY|os.O_RDWR):write_paths.append(target)
 except (FileNotFoundError,StopIteration):pass
os.kill(12922,signal.SIGTERM);os.kill(12922,signal.SIGCONT)
for _ in range(100):
 p=pathlib.Path('/proc/12922/stat')
 if not p.exists() or p.read_text().split()[2]=='Z':break
 time.sleep(.2)
else:raise RuntimeError('Original collector did not stop; do not launch new GPU job')
for _ in range(100):
 jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
 if not jobs:break
 time.sleep(.2)
assert not jobs
stage=R/'b2_data/treatments/train1030';retained=[];quarantine=[];used=set()
qdir=R/'parallel/interrupted_files';qdir.mkdir()
for ident in science['cohorts']['train1030']:
 p=stage/(hashlib.sha256(ident.encode()).hexdigest()+'.json')
 if not p.exists():continue
 try:m=read(p)
 except json.JSONDecodeError:
  dest=qdir/p.name;quarantine.append(dict(path=str(p.relative_to(R)),sha256=sha(p),preserved_at=str(dest.relative_to(R))));p.rename(dest);continue
 assert m['id']==ident and m['source_freeze_sha256']==sha(sf)
 retained.append(dict(id=ident,metadata_file=p.name,metadata_sha256=sha(p),tensor_raw_sha256=m['raw_tensor_sha256']))
 for t in m['tensor_manifest'].values():
  for c in t['chunks']:used.add((stage/c['file']).resolve())
for p in write_paths:
 if not p.exists() or p in used or p.suffix!='.gz':continue
 try:assert hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==p.name.split('.')[0]
 except (OSError,EOFError,AssertionError):
  dest=qdir/p.name;quarantine.append(dict(path=str(p.relative_to(R)),sha256=sha(p),preserved_at=str(dest.relative_to(R))));p.rename(dest)
receipt=dict(status='ORIGINAL_STOPPED_RETAINED',epoch=time.time(),retained=retained,n_retained=len(retained),quarantined=quarantine,original_science_freeze_sha256=sha(sf),scope='No completed record or raw chunk deleted. Invalid uncommitted files retained in quarantine; new writers use isolated namespaces.')
(R/'parallel/TRANSITION_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
statepath=R/'B2_QUEUE_STATE.json';statepath.write_text(json.dumps(dict(status='RUNNING',step='empty_gpu_parallel_acceptance',n_retained=len(retained),epoch=time.time()),indent=2))
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (R/'parallel/empty_gpu_acceptance.log').open('x') as log:
 result=subprocess.run([PY,str(R/'parallel/benchmark_empty_gpu.py')],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT)
try:
 with (R/'parallel/freeze_production.log').open('x') as log:subprocess.run([PY,str(R/'parallel/freeze_production.py')],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
except BaseException:
 legacy=(R/'async_io/run_queue_async_v1.py').read_text().replace('_async_v1.log','_after_parallel_reject.log')
 fallback=R/'parallel/fallback_original_async.py';fallback.write_text(legacy)
 with (R/'parallel/fallback.log').open('x') as log:
  child=subprocess.Popen([PY,str(fallback)],env=env,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (R/'parallel/ACCEPTANCE_AND_LAUNCH.json').write_text(json.dumps(dict(status='PARALLEL_REJECTED_ORIGINAL_ASYNC_RESTORED',pid=child.pid,gate_returncode=result.returncode),indent=2));print('ORIGINAL_ASYNC_RESTORED',child.pid,flush=True)
else:
 try:
  if argv(13225)==[PY,str(R/'async_io/verify_execution_lineage.py')]:os.kill(13225,signal.SIGTERM)
 except FileNotFoundError:pass
 for script,label in [('run_scientific_queue.py','production'),('verify_lineage_v2.py','lineage')]:
  e=dict(env)
  if label=='lineage':e['CUDA_VISIBLE_DEVICES']=''
  with (R/'parallel'/(label+'.log')).open('x') as log:
   child=subprocess.Popen([PY,str(R/'parallel'/script)],env=e,cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  print('PARALLEL_'+label.upper()+'_LAUNCHED',child.pid,flush=True)
 a=read(R/'parallel/EXECUTION_AMENDMENT.json')
 (R/'parallel/ACCEPTANCE_AND_LAUNCH.json').write_text(json.dumps(dict(status='VERSIONED_ISOLATED_EXECUTION_LAUNCHED',workers=a['workers'],epoch=time.time(),execution_amendment_sha256=sha(R/'parallel/EXECUTION_AMENDMENT.json'),n_retained=len(retained)),indent=2))
