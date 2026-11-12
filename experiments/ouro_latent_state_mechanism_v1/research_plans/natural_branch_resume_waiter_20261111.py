"""One-shot wait and resume; isolated systemd user service, no GPU until admitted."""
import os,sys,json,time,fcntl,subprocess,shutil,hashlib,traceback
from pathlib import Path
import psutil
ROOT=Path('/home/luansiyu/loop-latent');RUN=ROOT/'research_plans/natural_branch_full_run_20261111';CODE=ROOT/'research_plans/natural_branch_full_code_v2_20261111';AUDIT=ROOT/'research_plans/natural_branch_preflight14_cpu.json';STATE=ROOT/'research_plans/natural_branch_resume_wait_20261111';EXP=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1';sys.path.insert(0,str(EXP))
from ouro_watch import owned_process,install_signal_handlers
G=1024**3
EXPECTED_AUDIT='a7ebc2864b537884be89576c80d40de50ddd6323b63f1423107f56f74c1c81da'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def resources():
 txt=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10).strip();parts=[int(v.strip()) for v in txt.split(',')]
 return {'gpu_free_mib':parts[0],'gpu_used_mib':parts[1],'gpu_util_percent':parts[2],'disk_free_bytes':shutil.disk_usage(ROOT).free,'ram_available_bytes':psutil.virtual_memory().available}
def admissible(s):return s['gpu_free_mib']>20*1024 and s['disk_free_bytes']>=30*G and s['ram_available_bytes']>=8*G
def reserve(s):return s['gpu_free_mib']>=4*1024 and s['disk_free_bytes']>=20*G and s['ram_available_bytes']>=4*G
def update(**v):
 v['time_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());tmp=STATE/'status.tmp';tmp.write_text(json.dumps(v,sort_keys=True));os.replace(tmp,STATE/'status.json');print(json.dumps(v,sort_keys=True),flush=True)
def preflight():
 if sha(AUDIT)!=EXPECTED_AUDIT:raise ValueError('CPU preflight hash')
 j=json.loads(AUDIT.read_text());records=j['records']
 if j['status']!='CPU_STRUCTURE_14_PASS_NOT_CUDA_GATE' or len(records)!=14:raise ValueError('CPU preflight status')
 names={}
 for r in records:
  key='train_'+hashlib.sha256(r['id'].encode()).hexdigest();names[key]=r
 if len(names)!=14:raise ValueError('duplicate ID')
 m={p.stem for p in RUN.glob('train_*.json')};t={p.stem for p in RUN.glob('train_*.safetensors')}
 if m!=set(names) or t!=set(names) or list(RUN.glob('*.partial')):raise ValueError('changed shards')
 for k,r in names.items():
  if sha(RUN/(k+'.json'))!=r['metadata_sha256'] or sha(RUN/(k+'.safetensors'))!=r['tensor_sha256']:raise ValueError('shard drift')
 active=subprocess.check_output(['systemctl','--user','show','ouro-natural-branch-full-20261111.service','-p','ActiveState','--value'],text=True,timeout=8).strip()
 if active not in ('failed','inactive'):raise RuntimeError('original unit active')
def main():
 STATE.mkdir(exist_ok=False);fd=os.open(ROOT/'natural_branch_resume_waiter.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);install_signal_handlers();preflight();update(status='WAITING_RESOURCE',resources=resources(),reason='require GPU free >20GiB, disk >=30GiB, RAM >=8GiB')
  while True:
   first=resources()
   if admissible(first):
    time.sleep(10);second=resources()
    if admissible(second):break
   update(status='WAITING_RESOURCE',resources=first,reason='admission threshold unmet');time.sleep(30)
  preflight();update(status='STARTING_RESUME',resources=second)
  env=os.environ.copy();env.update(PYTHONPATH=str(CODE),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
  cmd=[str(ROOT/'conda-env/bin/python'),'-u','-c','from full_runner_draft import run; run("'+str(RUN)+'")']
  with (STATE/'runner.log').open('x') as log:
   with owned_process(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT) as child:
    update(status='RUNNING',child_pid=child.pid,resources=resources())
    while child.poll() is None:
     time.sleep(5);current=resources()
     if not reserve(current):update(status='STOPPING_RESOURCE',child_pid=child.pid,resources=current);raise RuntimeError('reserve threshold breached')
    if child.returncode:raise RuntimeError('runner exited '+str(child.returncode))
  update(status='COLLECTION_COMPLETE_PENDING_COVERAGE',resources=resources())
 except BaseException as e:
  update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
 finally:os.close(fd)
if __name__=='__main__':main()
