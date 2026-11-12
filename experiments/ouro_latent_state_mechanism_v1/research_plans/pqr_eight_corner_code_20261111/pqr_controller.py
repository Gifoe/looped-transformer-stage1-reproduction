"""DRAFT only: no start until reviewed, real probe, and prior run complete."""
import os,sys,json,time,fcntl,signal,traceback,uuid,argparse
from pathlib import Path
if not __debug__:raise RuntimeError('optimized Python forbidden')
ROOT=Path('/home/luansiyu/loop-latent');HERE=Path(__file__).resolve().parent;EXP=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1';sys.path.insert(0,str(EXP));sys.path.insert(0,str(HERE))
from ouro_watch import machine,atomic_json,sha,GIB,external_tasks
from pqr_supervision import run_owned,stage_admissible
from pqr_probe_guard import probe_pair,exact_probe_logits
run=ROOT/'orchestration_runs/pqr-round3-20261111';attempt=uuid.uuid4().hex;state={'status':'INITIALIZING','kind':'ROUND3_PQR_DRAFT_NOT_MECHANISM','attempt':attempt,'controller_sha256':sha(Path(__file__))}
def update(**v):state.update(v);atomic_json(run/'status.json',state)
def handler(sig,frame):raise InterruptedError('signal '+str(sig))
def eligible(g):return g['free_mib']>20*1024
def main(review_path=None):
 lock=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR,0o600)
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if run.exists():
   if run.is_symlink() or not run.is_dir():raise RuntimeError('unsafe run path')
   previous=run/'status.json'
   if not previous.is_file():raise RuntimeError('missing prior status')
   prior=json.loads(previous.read_text());assert prior['kind']=='ROUND3_PQR_DRAFT_NOT_MECHANISM' and prior['status']!='PQR_COMPLETE_PENDING_ASTRA_REVIEW';os.link(previous,run/('prior_status_'+attempt+'.json'))
  else:run.mkdir(exist_ok=False)
  for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,handler)
  try:
   if external_tasks():raise RuntimeError('other project task active')
   oldstatus=json.loads((ROOT/'orchestration_runs/round2-resume-20261111/status.json').read_text());assert oldstatus['status']=='INTERVENTION_COMPLETE_PENDING_ASTRA_REVIEW';old_status_sha=sha(ROOT/'orchestration_runs/round2-resume-20261111/status.json')
   first=machine();required=20*GIB+2590*3*1024**2+2*3*1024**2;update(stage='preflight',resources=first,required_bytes=required,old_status_sha256=old_status_sha)
   if not eligible(first['gpu']) or first['disk_free_bytes']<required or first['memory']['MemAvailable']<12*GIB:raise RuntimeError('resource preflight')
   time.sleep(10);again=machine();update(second_preflight=again)
   if not eligible(again['gpu']) or again['disk_free_bytes']<required or again['memory']['MemAvailable']<12*GIB or external_tasks():raise RuntimeError('second resource preflight')
   env=os.environ.copy();env.update(HF_HOME=str(ROOT/'cache/hf'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'));env.pop('USE_TORCH',None);python=ROOT/'conda-env/bin/python'
   required_gpu_bytes=20*GIB
   def stage(name,args):
    snap=machine();update(status='RUNNING',stage=name,stage_preflight=snap,required_gpu_bytes=required_gpu_bytes)
    if not stage_admissible(snap,required_gpu_bytes,GIB):raise RuntimeError('stage resource preflight')
    def monitor(proc,elapsed):
     v=machine();foreign=[]
     for pid in v['gpu']['compute_pids']:
      try:
       if os.getpgid(pid)!=proc.pid:foreign.append(pid)
      except ProcessLookupError:foreign.append(pid)
     flags={'gpu_free_below_4GiB':v['gpu']['free_mib']<4*1024,'disk_under_20GiB':v['disk_free_bytes']<20*GIB,'ram_under_4GiB':v['memory']['MemAvailable']<4*GIB,'timeout_12h':elapsed>43200};update(last_monitor={'resources':v,'foreign_gpu_pids':foreign})
     if any(flags.values()):update(interruption_conditions=flags,interruption_resources=v);raise RuntimeError('resource or timeout interruption')
    run_owned([str(python),'-u']+list(map(str,args)),run/(name+'_'+attempt+'.log'),ROOT,env,monitor)
   bind=run/'binding.json'
   if not bind.exists():stage('binding',[HERE/'pqr_binding.py','--out',bind])
   b=json.loads(bind.read_text());assert b['status']=='PQR_BINDING_COMPLETE_PENDING_PROBE';binding_sha=sha(bind);longest=b['longest_request']
   if review_path is not None:
    rp=Path(review_path).resolve();rp.relative_to(run.resolve());review=json.loads(rp.read_text());assert review['status']=='PQR_PROBE_REVIEW_APPROVED' and review['binding_sha256']==binding_sha and review['reviewer']=='Astra';old_receipt=Path(review['probe_receipt_path']).resolve();old_report=Path(review['probe_coverage_path']).resolve();old_receipt.relative_to(run.resolve());old_report.relative_to(run.resolve());assert sha(old_receipt)==review['probe_receipt_sha256'] and sha(old_report)==review['probe_coverage_sha256'];reviewed_receipt,reviewed,reviewed_tensor=probe_pair(old_receipt,old_report,binding_sha,longest,sha);update(review_path=str(rp),review_sha256=sha(rp))
   probe=run/('probe_'+attempt);probe_receipt=probe/'_receipt.json'
   if not probe_receipt.exists():stage('probe',[HERE/'pqr_runner.py','--split',longest['split'],'--out',probe,'--binding',bind,'--probe-id',longest['id']])
   probe_audit=run/('probe_coverage_'+attempt+'.json');stage('probe_coverage',[HERE/'pqr_coverage.py','--split',longest['split'],'--shards',probe,'--binding',bind,'--report',probe_audit,'--probe-id',longest['id']]);receipt,pr,fresh_tensor=probe_pair(probe_receipt,probe_audit,binding_sha,longest,sha)
   if review_path is not None:exact_probe_logits(reviewed_tensor,fresh_tensor)
   rs=receipt['resources'];assert receipt['questions']==1 and receipt['binding_sha256']==binding_sha and rs['newly_computed_questions']==1 and rs['measurement_scope']=='model load through fresh full-question forward and saved-output validation; RSS whole process' and rs['max_new_question_bytes']>0 and rs['max_new_question_bytes']<=3*1024**2 and rs['peak_reserved_bytes']>=rs['baseline_reserved_bytes'] and rs['peak_allocated_bytes']>=rs['baseline_allocated_bytes'] and rs['peak_rss_bytes']>=rs['rss_bytes'] and rs['peak_reserved_bytes']+(rs['device_total_bytes']-rs['pre_model_free_bytes'])<=rs['device_total_bytes']-4*GIB;required_gpu_bytes=max(20*GIB,rs['peak_reserved_bytes']+5*GIB);update(probe_resources=rs,probe_receipt_path=str(probe_receipt),probe_receipt_sha256=sha(probe_receipt),probe_coverage_path=str(probe_audit),probe_coverage_sha256=sha(probe_audit),post_probe_required_gpu_bytes=required_gpu_bytes)
   if review_path is None:
    update(status='PQR_PROBE_COMPLETE_PENDING_ASTRA_REVIEW',stage='awaiting_probe_review',binding_sha256=binding_sha);return
   for split,n in [('train',1119),('validation',299),('test',1172)]:
    d=run/split;receipt=d/'_receipt.json'
    if not receipt.exists():stage('intervention_'+split,[HERE/'pqr_runner.py','--split',split,'--out',d,'--binding',bind,'--probe-receipt',probe_audit,'--probe-shards',probe])
    v=json.loads(receipt.read_text());assert v['status']=='PQR_SPLIT_PENDING_COVERAGE' and v['questions']==n and v['binding_sha256']==binding_sha
    audit=run/('coverage_'+split+'_'+attempt+'.json');stage('coverage_'+split,[HERE/'pqr_coverage.py','--split',split,'--shards',d,'--binding',bind,'--report',audit]);r=json.loads(audit.read_text());assert r['status']=='PQR_COMPLETE_CUDA_REPLAY_ZERO' and r['expected_questions']==r['complete_questions']==n and r['replayed_condition_logits']==4*n and r['binding_sha256']==binding_sha
    canonical=run/('coverage_'+split+'.json')
    if canonical.exists():
     old=json.loads(canonical.read_text());assert old['status']==r['status'] and old['questions']==r['questions'] and old['binding_sha256']==binding_sha
    else:os.link(audit,canonical)
   analysis=run/('analysis_'+attempt+'.json');stage('analysis',[HERE/'pqr_analysis.py','--run',run,'--binding',bind,'--out',analysis]);result=json.loads(analysis.read_text());assert result['status']=='PQR_FULL_COVERAGE_DESCRIPTIVE_PENDING_REVIEW' and all(result['splits'][s]['all']['n']==n for s,n in [('train',1119),('validation',299),('test',1172)])
   update(status='PQR_COMPLETE_PENDING_ASTRA_REVIEW',stage='awaiting_review',analysis_path=str(analysis),analysis_sha256=sha(analysis),note='No automatic semantic or mechanism claim.')
  except BaseException as e:
   update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
 finally:os.close(lock)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--review');a=p.parse_args();main(a.review)
