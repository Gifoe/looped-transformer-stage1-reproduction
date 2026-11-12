import sys
sys.path.insert(0,'/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1')
"""Single-instance supervised full ROUND 2 execution. Fail closed; no retries."""
import os,sys,json,time,fcntl,signal,subprocess,traceback,math,hashlib
from pathlib import Path
from ouro_watch import machine,owned_process,atomic_json,sha,GIB,external_tasks
from cohort_runner import bound_hashes
P=Path('/home/luansiyu/loop-latent/repo/experiments/ouro_latent_state_mechanism_v1/round2_full_controller.py');ROOT=P.parents[2].parent;run=ROOT/'orchestration_runs/round2-resume-20261111'; target=ROOT/'orchestration_runs/round2-full-20261111';state={'status':'INITIALIZING','kind':'ROUND2_FULL_INTERVENTION_NOT_MECHANISM','controller_sha256':sha(Path(__file__).resolve()),'original_controller_sha256':sha(P)}
def update(**kw):state.update(kw);atomic_json(run/'status.json',state)
def handler(sig,frame):raise InterruptedError('signal '+str(sig))
def eligible(g):return g['free_mib']>20*1024 
def no_other():
 output=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True)
 for line in output.splitlines():
  a=line.strip().split(None,1)
  if len(a)==2 and int(a[0])!=os.getpid() and any(n in a[1] for n in ('round2_anchor.py','round2_intervention_runner.py','round2_intervention_coverage.py','round2_intervention_analysis.py','round2_full_controller.py','round2_probe_controller.py')):raise RuntimeError('other round2 task '+a[0])
lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR,0o600)
try:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,handler)
 try:
  if subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=P.parents[2],text=True).strip():raise RuntimeError('dirty worktree')
  if external_tasks():raise RuntimeError('other project job active')
  no_other();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parents[2],text=True).strip();code={str(x):sha(x) for x in sorted(P.parent.glob('*.py'))};frozen=sorted((P.parent/'environment_pip_freeze.txt').read_text().splitlines());python=ROOT/'conda-env/bin/python';assert sorted(subprocess.check_output([str(python),'-m','pip','freeze'],text=True,timeout=60).splitlines())==frozen
  bound={s:bound_hashes(s) for s in ('train','validation','test')}
  resume_sources={str(x):sha(x) for x in (Path(__file__).resolve(),run/'partial_replay.py')};update(resume_sources=resume_sources)
  def guard():
   assert all(sha(Path(k))==v for k,v in resume_sources.items())
   assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parents[2],text=True).strip()==commit and not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=P.parents[2],text=True).strip()
   assert {str(x):sha(x) for x in sorted(P.parent.glob('*.py'))}==code and all(bound_hashes(s)==bound[s] for s in bound)
   assert sorted(subprocess.check_output([str(python),'-m','pip','freeze'],text=True,timeout=60).splitlines())==frozen
  probe=ROOT/'orchestration_runs/round2-question-probe-20261111';ps=json.loads((probe/'status.json').read_text());audit=json.loads((probe/'probe_replay_audit.json').read_text());assert ps['status']=='PROBE_COMPLETE_ONLY' and audit['status']=='PROBE_REPLAY_PASSED_ONLY' and audit['groups']==11 and audit['score_comparisons']==44 and audit['current_validation_sha256']==sha(P.with_name('round2_intervention_runner.py'))
  audit_source=audit['original_probe_commit'];core=P.with_name('round2_intervention_core.py');rel=core.relative_to(P.parents[2]).as_posix();probe_core=subprocess.check_output(['git','show',audit_source+':'+rel],cwd=P.parents[2]);assert hashlib.sha256(probe_core).hexdigest()==sha(core),'probe computational core drift'
  per=ps['resource_estimate']['single_question_bytes'];required=math.ceil(per*2590*1.2)+2*per+20*GIB;before=machine();update(stage='preflight',resources=before,required_bytes=required,probe_sha256=sha(probe/'status.json'),probe_replay_sha256=sha(probe/'probe_replay_audit.json'),commit=commit)
  if not eligible(before['gpu']) or before['disk_free_bytes']<required or before['memory']['MemAvailable']<12*GIB:raise RuntimeError('resource preflight failed')
  time.sleep(10);again=machine();update(second_preflight=again)
  if not eligible(again['gpu']) or again['disk_free_bytes']<required or again['memory']['MemAvailable']<12*GIB or external_tasks():raise RuntimeError('second preflight failed')
  no_other();env=os.environ.copy();env.update(HF_HOME=str(ROOT/'cache/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'));env.pop('USE_TORCH',None)
  def stage(name,args,gpu=True):
   guard();snap=machine();flags={'gpu_ineligible':gpu and not eligible(snap['gpu']),'disk_under_20GiB':snap['disk_free_bytes']<20*GIB,'ram_under_8GiB':snap['memory']['MemAvailable']<8*GIB};update(status='RUNNING',stage=name,stage_preflight=snap)
   if any(flags.values()):raise RuntimeError('preflight '+json.dumps(flags))
   started=time.monotonic()
   with (run/(name+'.log')).open('x') as log:
    with owned_process([str(python),'-u']+list(map(str,args)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT) as proc:
     while proc.poll() is None:
      time.sleep(5);snap=machine();foreign=[]
      if gpu:
       for pid in snap['gpu']['compute_pids']:
        try:
         if os.getpgid(pid)!=proc.pid:foreign.append(pid)
        except ProcessLookupError:foreign.append(pid)
      flags={'foreign_gpu_pids':foreign,'disk_under_20GiB':snap['disk_free_bytes']<20*GIB,'ram_under_4GiB':snap['memory']['MemAvailable']<4*GIB,'timeout_12h':time.monotonic()-started>43200}
      update(last_monitor={'resources':snap,'foreign_gpu_pids':foreign});flags['gpu_free_below_4GiB']=gpu and snap['gpu']['free_mib']<4*1024
      if any(v for k,v in flags.items() if k!='foreign_gpu_pids'):update(interruption_resources=snap,interruption_conditions=flags);raise RuntimeError('resource or timeout interruption')
     if proc.returncode:raise RuntimeError(name+' exited '+str(proc.returncode))
   guard()
  stage('partial_replay',[run/'partial_replay.py','--split','train','--shards',target/'train','--report',run/'partial_replay.json']); pr=json.loads((run/'partial_replay.json').read_text()); assert pr['status']=='PARTIAL_387_REPLAY_PASSED_NOT_COVERAGE' and pr['complete_questions']==387 and pr['replayed_condition_logits']==387*11
  expected={'train':1119,'validation':299,'test':1172}
  for split in ('train','validation','test'):
   directory=target/split;stage('intervention_'+split,[P.with_name('round2_intervention_runner.py'),'--split',split,'--out',directory]);receipt=json.loads((directory/'_receipt.json').read_text());assert receipt['status']=='SPLIT_COMPLETE_PENDING_COVERAGE' and receipt['questions']==receipt['expected']==expected[split]
   report=target/('coverage_'+split+'.json');stage('coverage_'+split,[P.with_name('round2_intervention_coverage.py'),'--split',split,'--shards',directory,'--report',report]);v=json.loads(report.read_text());assert v['status']=='COMPLETE_CUDA_REPLAY_ZERO_TOLERANCE' and v['expected_questions']==v['complete_questions']==expected[split] and v['replayed_condition_logits']==11*expected[split]
  out=target/'analysis.json';stage('analysis',[P.with_name('round2_intervention_analysis.py'),'--run',target,'--out',out],False);a=json.loads(out.read_text());assert a['status']=='COMPLETE_COVERAGE_BOUND_DESCRIPTIVE_INTERVENTION' and all(a['splits'][s]['n']==expected[s] for s in expected)
  update(status='INTERVENTION_COMPLETE_PENDING_ASTRA_REVIEW',stage='awaiting_scientific_review',analysis_sha256=sha(out),note='Full intervention evidence needs scientific review; no automatic mechanism claim.')
 except BaseException as e:
  update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
finally:os.close(lock)
