"""One complete maximum-question resource probe; no full-cohort inference."""
import os,sys,json,time,fcntl,signal,subprocess,traceback
from pathlib import Path
from ouro_watch import machine,owned_process,atomic_json,sha,GIB,external_tasks
from round2_resource_guard import check_probe_resources
P=Path(__file__).resolve();ROOT=P.parents[2].parent;run=ROOT/'orchestration_runs/round2-question-probe-20261111';run.mkdir(exist_ok=False);state={'status':'INITIALIZING','kind':'FULL_QUESTION_PROBE_ONLY','controller_sha256':sha(P)}
def update(**kw):state.update(kw);atomic_json(run/'status.json',state)
def handler(sig,frame):raise InterruptedError('signal '+str(sig))
def eligible(g):return g['free_mib']>20*1024 and g['util_percent']<10 and not g['compute_pids']
def no_other():
 output=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True)
 for line in output.splitlines():
  a=line.strip().split(None,1)
  if len(a)==2 and int(a[0])!=os.getpid() and any(n in a[1] for n in ('round2_anchor.py','round2_intervention_runner.py','round2_probe_controller.py')):raise RuntimeError('other round2 task '+a[0])
lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR,0o600)
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,handler)
 try:
  if subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=P.parents[2],text=True).strip():raise RuntimeError('dirty worktree')
  if external_tasks():raise RuntimeError('other project job active')
  no_other();snap=machine();update(stage='preflight',resources=snap)
  if not eligible(snap['gpu']) or snap['disk_free_bytes']<20*GIB or snap['memory']['MemAvailable']<8*GIB:raise RuntimeError('resource preflight failed')
  time.sleep(10);snap=machine();update(second_preflight=snap)
  if not eligible(snap['gpu']) or snap['disk_free_bytes']<20*GIB or snap['memory']['MemAvailable']<8*GIB or external_tasks():raise RuntimeError('second preflight failed')
  no_other();env=os.environ.copy();env.update(HF_HOME=str(ROOT/'cache/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'));env.pop('USE_TORCH',None)
  update(status='RUNNING',stage='probe');started=time.monotonic();out=run/'question'
  with (run/'probe.log').open('x') as log:
   with owned_process([str(ROOT/'conda-env/bin/python'),'-u',str(P.with_name('round2_intervention_runner.py')),'--split','test','--probe-id','MSA_2012_8_30','--out',str(out)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT) as proc:
    while proc.poll() is None:
     time.sleep(5);snap=machine();foreign=[]
     for pid in snap['gpu']['compute_pids']:
      try:
       if os.getpgid(pid)!=proc.pid:foreign.append(pid)
      except ProcessLookupError:foreign.append(pid)
     flags={'foreign_gpu_pids':foreign,'disk_under_20GiB':snap['disk_free_bytes']<20*GIB,'ram_under_4GiB':snap['memory']['MemAvailable']<4*GIB,'timeout_1800s':time.monotonic()-started>1800}
     if foreign or any(v for k,v in flags.items() if k!='foreign_gpu_pids'):update(interruption_resources=snap,interruption_conditions=flags);raise RuntimeError('resource or timeout interruption')
    if proc.returncode:raise RuntimeError('probe exited '+str(proc.returncode))
  receipt=json.loads((out/'_receipt.json').read_text());assert receipt['status']=='PROBE_COMPLETE_ONLY' and receipt['questions']==receipt['expected']==1
  files=[p for p in out.glob('*.json') if p.name!='_receipt.json'];assert len(files)==1
  data=json.loads(files[0].read_text());assert data['payload']['identity']['id']=='MSA_2012_8_30' and data['payload']['result']['input_length']==2032
  tensor=files[0].with_suffix('.safetensors');assert tensor.is_file() and sha(tensor)==data['payload']['logits_sha256']
  after=machine();update(post_probe_resources=after)
  check_probe_resources(state['second_preflight'],after,receipt['gpu_peak_reserved'])
  estimate={'single_question_bytes':files[0].stat().st_size+tensor.stat().st_size,'question_metadata_bytes':files[0].stat().st_size,'question_logits_bytes':tensor.stat().st_size,'elapsed_seconds':receipt['elapsed_seconds'],'gpu_peak_allocated':receipt['gpu_peak_allocated'],'gpu_peak_reserved':receipt['gpu_peak_reserved'],'rss_bytes':receipt['rss_bytes'],'source_receipt_sha256':sha(out/'_receipt.json'),'source_question_sha256':sha(files[0]),'note':'single maximum-shape question; output is not full cohort'}
  atomic_json(run/'resource_estimate.json',estimate);update(status='PROBE_COMPLETE_ONLY',stage='awaiting_review',resource_estimate=estimate)
 except BaseException as e:
  update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
finally:os.close(lock)
