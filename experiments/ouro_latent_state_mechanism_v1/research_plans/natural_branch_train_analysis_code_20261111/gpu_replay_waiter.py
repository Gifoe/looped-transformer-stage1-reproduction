"""Launch one independent blind GPU replay after successful collection."""
import os,fcntl,json,subprocess,time,hashlib
from pathlib import Path
ROOT=Path('/home/luansiyu/loop-latent');RUN=ROOT/'research_plans/natural_branch_full_run_20261111';CODE=ROOT/'research_plans/natural_branch_train_analysis_code_20261111/gpu_logit_replay.py';EXPECTED='121102eca436b0f8b0b47d4b2503e8ff6c234484d9117492cf396a5772812292';UNIT='ouro-natural-branch-reserve-v5-20261111.service';REPLAY='ouro-natural-branch-gpu-replay-20261111'
def main():
 f=(ROOT/'research_plans/natural_branch_gpu_replay_wait.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 print('WAIT_COLLECTOR',flush=True)
 while True:
  x=subprocess.check_output(['systemctl','--user','show',UNIT,'-p','ActiveState','-p','Result'],text=True,timeout=15);d=dict(l.split('=',1) for l in x.splitlines())
  if d['ActiveState']=='active':time.sleep(10);continue
  if d['ActiveState']!='inactive' or d['Result']!='success':raise RuntimeError('collector failed '+str(d))
  break
 receipt=RUN/'receipt.json'
 if not receipt.exists():raise RuntimeError('collector receipt missing')
 data=json.loads(receipt.read_text())
 if data.get('status')!='FULL_OBSERVATION_PENDING_INDEPENDENT_COVERAGE' or data.get('counts')!={'train':1119,'validation':299,'test':1172}:raise RuntimeError('collector receipt incomplete')
 if hashlib.sha256(CODE.read_bytes()).hexdigest()!=EXPECTED:raise RuntimeError('replay code drift')
 fd=os.open(ROOT/'ouro_watch.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(fd,fcntl.LOCK_UN)
 finally:os.close(fd)
 cmd=['systemd-run','--user','--unit='+REPLAY,'--property=KillMode=control-group','--property=Restart=no','--setenv=OMP_NUM_THREADS=1','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=PYTHONPATH='+str(ROOT/'research_plans/natural_branch_full_code_v5_20261111'),str(ROOT/'conda-env/bin/python'),'-u',str(CODE)]
 subprocess.run(cmd,check=True,timeout=30);print('GPU_REPLAY_LAUNCHED',flush=True)
if __name__=='__main__':main()
