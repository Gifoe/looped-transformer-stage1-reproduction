"""Single-run supervised ROUND 2 anchor process, not a cohort intervention."""
import os,sys,json,time,fcntl,signal,subprocess,traceback
from pathlib import Path
from ouro_watch import machine,owned_process,atomic_json,sha,GIB,external_tasks
P=Path(__file__).resolve();ROOT=P.parents[2].parent;run=ROOT/'orchestration_runs/round2-anchor-20261111';run.mkdir(exist_ok=False);state={'status':'INITIALIZING','kind':'ANCHORS_ONLY_NOT_CAUSAL','controller_sha256':sha(P)}
def update(**kw):state.update(kw);atomic_json(run/'status.json',state)
def handler(signum,frame):raise InterruptedError('signal '+str(signum))
def eligible(g):return g['free_mib']>20*1024 and g['util_percent']<10 and not g['compute_pids']
def no_other_round2():
 output=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True)
 for line in output.splitlines():
  a=line.strip().split(None,1)
  if len(a)==2 and int(a[0])!=os.getpid() and any(n in a[1] for n in ('round2_anchor.py','round2_anchor_controller.py')):raise RuntimeError('another round2 task '+a[0])
lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR,0o600)
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,handler)
 try:
  if subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=P.parents[2],text=True).strip():raise RuntimeError('dirty worktree')
  if external_tasks():raise RuntimeError('other project job active')
  no_other_round2();before=machine();update(stage='preflight',resources=before)
  if not eligible(before['gpu']) or before['disk_free_bytes']<20*GIB or before['memory']['MemAvailable']<8*GIB:raise RuntimeError('resource preflight failed')
  time.sleep(10);again=machine();update(second_preflight=again)
  if not eligible(again['gpu']) or again['disk_free_bytes']<20*GIB or again['memory']['MemAvailable']<8*GIB or external_tasks():raise RuntimeError('second resource preflight failed')
  no_other_round2();env=os.environ.copy();env.update(HF_HOME=str(ROOT/'cache/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'));env.pop('USE_TORCH',None)
  update(status='RUNNING',stage='anchor');started=time.monotonic();report=run/'anchor.json'
  with (run/'anchor.log').open('x') as log:
   with owned_process([str(ROOT/'conda-env/bin/python'),'-u',str(P.with_name('round2_anchor.py')),'--out',str(report)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT) as proc:
    while proc.poll() is None:
     time.sleep(5);snap=machine();foreign=[]
     for pid in snap['gpu']['compute_pids']:
      try:
       if os.getpgid(pid)!=proc.pid:foreign.append(pid)
      except ProcessLookupError:foreign.append(pid)
     flags={'foreign_gpu_pids':foreign,'disk_under_20GiB':snap['disk_free_bytes']<20*GIB,'ram_under_4GiB':snap['memory']['MemAvailable']<4*GIB,'timeout_1800s':time.monotonic()-started>1800}
     if foreign or any(v for k,v in flags.items() if k!='foreign_gpu_pids'):
      update(interruption_resources=snap,interruption_conditions=flags);raise RuntimeError('resource or timeout interruption')
    if proc.returncode:raise RuntimeError('anchor exited '+str(proc.returncode))
  data=json.loads(report.read_text());assert data['status']=='ANCHORS_PASSED_ONLY' and len(data['cases'])==4
  update(status='ANCHORS_PASSED_ONLY',stage='awaiting_review',report_sha256=sha(report))
 except BaseException as e:
  update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
finally:os.close(lock)
