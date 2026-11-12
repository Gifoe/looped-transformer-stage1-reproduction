"""Wait for actual train completion before train-only exploratory audit."""
import datetime,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/home/luansiyu/loop-latent');PLAN=ROOT/'research_plans';OUT=PLAN/'natural_branch_full_run_20261111';CODE=PLAN/'natural_branch_train_analysis_code_20261111';PY=ROOT/'conda-env/bin/python';DISC=PLAN/'NATURAL_BRANCH_TRAIN_DISCLOSURE_AMENDMENT_20261111.md'
UNITS=['ouro-natural-branch-resume-v4-20261111.service','ouro-natural-branch-reserve-v5-20261111.service','ouro-natural-branch-reserve-fallback-20261111.service']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def main():
 lock=(PLAN/'natural_branch_train_analysis.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 print('WAITING_TRAIN',flush=True)
 while True:
  log=subprocess.run(['journalctl','--user','--no-pager',*[v for u in UNITS for v in ('-u',u)],'-n','30'],capture_output=True,text=True,timeout=20).stdout
  if 'QUESTION_COMPLETE train 1119 ' in log:break
  active=False
  for u in UNITS:
   q=subprocess.run(['systemctl','--user','is-active',u],capture_output=True,text=True,timeout=10)
   active|=q.stdout.strip()=='active'
  if not active:raise RuntimeError('no active collector and train completion not observed')
  time.sleep(15)
 if len(list(OUT.glob('train_*.json')))!=1119 or len(list(OUT.glob('train_*.safetensors')))!=1119:raise RuntimeError('train shard count')
 sources={'analysis_plan':sha(PLAN/'NATURAL_BRANCH_COVERAGE_ANALYSIS_PLAN.md'),'telescope_addendum':sha(PLAN/'NATURAL_BRANCH_TELESCOPE_ADDENDUM.md'),'natural_train_coverage':'44b6bafcdcf7066859e56e4c83bfb77ec9d7c22f78c4c6952b6315b41e52f16f','audit_code':sha(CODE/'train_audit.py'),'summary_code':sha(CODE/'summarize_train.py')}
 body='# Train-only disclosure amendment\n\nUTC recorded: '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n\nAt user direction, after all 1119 train questions are captured, explore train only before validation/test. The prior plan remains unchanged as a historical document; this is an explicit later change of disclosure order. No validation or test branch effects are read during train analysis. The 25 training demonstrations are listed separately. This is exploratory, not independent or blinded validation. CPU source/shape/state/logit/hash and lens verification must pass before reporting, while an additional independent CUDA score replay remains pending until the active GPU collection releases the lock. All original samples and conditions remain intact.\n\nSources and SHA-256: '+json.dumps(sources,sort_keys=True)+'\n'
 with DISC.open('x') as f:f.write(body);f.flush();os.fsync(f.fileno())
 print('DISCLOSURE_WRITTEN',str(DISC),sha(DISC),flush=True)
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(PLAN/'natural_branch_full_code_v4_20261111')})
 for script in ('train_audit.py','summarize_train.py'):
  subprocess.run(['ionice','-c','3','nice','-n','19',str(PY),'-u',str(CODE/script)],env=env,check=True)
 print('TRAIN_ANALYSIS_DONE',flush=True)
if __name__=='__main__':main()
