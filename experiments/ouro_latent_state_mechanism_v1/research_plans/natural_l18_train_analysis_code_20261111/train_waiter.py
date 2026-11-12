"""Wait for completed L18 train before CPU-only preliminary audit and analysis."""
import os,fcntl,subprocess,time
from pathlib import Path
ROOT=Path('/home/luansiyu/loop-latent');RUN=ROOT/'research_plans/natural_l18_full_run_20261111';CODE=ROOT/'research_plans/natural_l18_train_analysis_code_20261111';PY=ROOT/'conda-env/bin/python';UNIT='ouro-natural-l18-full-20261111.service'
def main():
 f=(ROOT/'research_plans/natural_l18_train_wait.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);print('WAITING_L18_TRAIN',flush=True)
 while True:
  log=subprocess.run(['journalctl','--user','--no-pager','-u',UNIT,'-n','50'],capture_output=True,text=True,timeout=15).stdout
  if 'QUESTION_COMPLETE train 1119 ' in log:break
  status=subprocess.run(['systemctl','--user','is-active',UNIT],capture_output=True,text=True,timeout=15).stdout.strip()
  if status!='active':raise RuntimeError('collector is not active before train completion: '+status)
  time.sleep(10)
 if len(list(RUN.glob('train_*.json')))!=1119 or len(list(RUN.glob('train_*.safetensors')))!=1119:raise RuntimeError('train incomplete')
 env=os.environ.copy();env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','PYTHONPATH':str(ROOT/'research_plans/natural_l18_probe_code_20261111')+':'+str(ROOT/'research_plans/natural_branch_full_code_v5_20261111')})
 for name in ('train_pre_audit.py','train_analysis.py'):subprocess.run(['ionice','-c','3','nice','-n','19',str(PY),'-u',str(CODE/name)],env=env,check=True)
 print('L18_TRAIN_ANALYSIS_DONE',flush=True)
if __name__=='__main__':main()
