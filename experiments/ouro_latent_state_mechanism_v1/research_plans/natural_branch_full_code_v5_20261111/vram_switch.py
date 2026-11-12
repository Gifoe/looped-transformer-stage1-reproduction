"""One-shot switch only when 40 GiB can physically be reserved."""
import fcntl,hashlib,os,subprocess,time,json
from pathlib import Path
ROOT=Path('/home/luansiyu/loop-latent')
OUT=ROOT/'research_plans/natural_branch_full_run_20261111'
V4='ouro-natural-branch-resume-v4-20261111.service'
V5='ouro-natural-branch-reserve-v5-20261111'
CODE=ROOT/'research_plans/natural_branch_full_code_v5_20261111'
PY=ROOT/'conda-env/bin/python'
READY=CODE/'reservation_ready.json'
FALLBACK='ouro-natural-branch-reserve-fallback-20261111'
THRESHOLD=40*1024+512 # MiB; includes current worker memory plus free
EXPECTED={'full_runner_draft.py': '70669efd01c2cfd981c31879c0b8b2ebcba5bcd7a81f9378273aa46e8315893d', 'state_capture.py': 'f757d947c3eea7966a86f3a17cc5008418488c3e2915ec685cdebb8f30839c1e', 'technical.py': '537231dbd895221d1775daf31ac6a172f19f97a28cbccb2c066b7e99b20c4dcd', 'decompose.py': 'cb730f7b0c09d3e8074a488955ed525d6b50acc29d26b13f23c4405872549243', 'source_input.py': 'd52ef07a08011cf28cb78c2883cb86b047b319317a163ce61145fb630963a050', 'full_input.py': '0da5cc5db704290daba15d8db23cdd6ee31c636f8adb6b01a23ef0a748717986'}

def cmd(*args):return subprocess.check_output(args,text=True,timeout=20).strip()
def unit():
 s=cmd('systemctl','--user','show',V4,'-p','ActiveState','-p','MainPID');d=dict(x.split('=',1) for x in s.splitlines());return d['ActiveState'],int(d['MainPID'])
def snapshot():
 state,pid=unit()
 free=int(cmd('nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits').splitlines()[0].strip())
 q=cmd('nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits')
 matches=[int(x.split(',')[1].strip()) for x in q.splitlines() if int(x.split(',')[0].strip())==pid] if q else []
 return state,pid,free,(matches[0] if len(matches)==1 else 0)
def eligible(s):return s[0]=='active' and s[1]>0 and s[3]>0 and s[2]+s[3]>=THRESHOLD
def log(kind,**d):print(kind,json.dumps(d,sort_keys=True),flush=True)
def launch(name,mode,ready=False):
 args=['systemd-run','--user','--unit='+name,'--property=KillMode=control-group','--property=Restart=no','--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=HF_HUB_OFFLINE=1','--setenv=TRANSFORMERS_OFFLINE=1','--setenv=PYTHONPATH='+str(CODE),'--setenv=BRANCH_RESERVE_MODE='+mode]
 if ready:args.append('--setenv=BRANCH_RESERVE_READY='+str(READY))
 args += [str(PY),'-u','-c','from full_runner_draft import run; run("'+str(OUT)+'")']
 subprocess.run(args,check=True,timeout=30)
def details(name):
 s=cmd('systemctl','--user','show',name+'.service','-p','ActiveState','-p','MainPID');d=dict(x.split('=',1) for x in s.splitlines());return d['ActiveState'],int(d['MainPID'])
def owned_gpu(pid):
 q=cmd('nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits')
 matches=[int(x.split(',')[1].strip()) for x in q.splitlines() if int(x.split(',')[0].strip())==pid] if q else []
 return matches[0] if len(matches)==1 else 0
def require_global_lock_free():
 f=(ROOT/'ouro_watch.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN);f.close()
def main():
 f=(ROOT/'research_plans/natural_branch_vram_switch.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 log('WAITING',target_GiB=40,threshold_MiB=THRESHOLD)
 while True:
  a=snapshot()
  if a[0]!='active' or a[1]<=0 or (OUT/'receipt.json').exists():log('STOP_NO_LIVE_WORKER',sample=a);return
  if not eligible(a):time.sleep(30);continue
  time.sleep(10);b=snapshot()
  if not eligible(b) or b[1]!=a[1]:log('TRIGGER_NOT_STABLE',first=a,second=b);time.sleep(30);continue
  # Fail closed if another v5 unit exists, even if inactive.
  if subprocess.run(['systemctl','--user','show',V5+'.service','-p','LoadState','--value'],capture_output=True,text=True).stdout.strip()!='not-found':raise RuntimeError('v5 unit already exists')
  if (OUT/'receipt.json').exists():return
  if READY.exists():raise RuntimeError('ready file unexpectedly exists')
  for name,digest in EXPECTED.items():
   if hashlib.sha256((CODE/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('v5 source drift '+name)
  log('SWITCH_BEGIN',first=a,second=b)
  subprocess.run(['systemctl','--user','stop',V4],check=True,timeout=40)
  state,pid,free,own=snapshot()
  if state=='active' or pid!=0:raise RuntimeError('old unit did not stop')
  if (OUT/'receipt.json').exists():log('WORK_COMPLETE_DURING_SWITCH');return
  if free<THRESHOLD:
   # A competing allocation raced the transition: restart the original once.
   log('RACE_FALLBACK_TO_V4',free_MiB=free)
   subprocess.run(['systemctl','--user','start',V4],check=True,timeout=30);return
  require_global_lock_free()
  try:launch(V5,'1',True)
  except Exception as ex:
   log('V5_LAUNCH_ERROR',error=repr(ex))
   # systemd-run failure does not prove no process was launched.
   subprocess.run(['systemctl','--user','stop',V5+'.service'],capture_output=True,timeout=60)
   state,pid=details(V5)
   if state=='active' or pid!=0:raise RuntimeError('v5 possibly alive')
   require_global_lock_free()
   if (OUT/'receipt.json').exists():return
   launch(FALLBACK,'0');log('FALLBACK_LAUNCHED_NO_RESERVATION');return
  deadline=time.monotonic()+600
  while time.monotonic()<deadline:
   state,pid=details(V5)
   if state not in ('active','activating') or pid==0:break
   if READY.exists():
    ready=json.loads(READY.read_text());physical=owned_gpu(pid)
    if ready['pid']==pid and ready['reserved']>=ready['target']==40*1024**3 and ready['runner_sha256']==EXPECTED['full_runner_draft.py'] and physical>=40*1024:
     log('RESERVATION_READY',pid=pid,physical_MiB=physical,ready=ready);return
   time.sleep(5)
  log('V5_START_FAILED_OR_TIMEOUT',state=state,pid=pid)
  journal=subprocess.run(['journalctl','--user','-u',V5+'.service','--no-pager','-n','100'],capture_output=True,text=True,timeout=20).stdout
  resource_failure=('out of memory' in journal.lower() or 'resource preflight' in journal or state in ('active','activating'))
  subprocess.run(['systemctl','--user','stop',V5+'.service'],check=True,timeout=60)
  state,pid=details(V5)
  if state=='active' or pid!=0:raise RuntimeError('v5 did not stop: refusing fallback')
  require_global_lock_free()
  if (OUT/'receipt.json').exists():return
  if not resource_failure:raise RuntimeError('non-resource failure: no automatic fallback')
  launch(FALLBACK,'0')
  log('FALLBACK_LAUNCHED_NO_RESERVATION')
  return
if __name__=='__main__':main()
