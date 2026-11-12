"""Disabled ABCA supervisor draft. Requires control-group service and review."""
import sys,os,fcntl,time,subprocess
from pathlib import Path
ROOT=Path('/home/luansiyu/loop-latent');PQR=ROOT/'research_plans/pqr_eight_corner_code_20261111';EXP=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1'
sys.path[:0]=[str(PQR),str(EXP)]
from ouro_watch import machine,owned_process,install_signal_handlers
from pqr_supervision import stage_admissible
GIB=1024**3

def check(snapshot,elapsed,initial=False):
 if elapsed>7200:raise RuntimeError('timeout')
 if initial:
  if not stage_admissible(snapshot,20*GIB,GIB):raise RuntimeError('resource admission')
 else:
  if snapshot['gpu']['free_mib']<4096 or snapshot['disk_free_bytes']<20*GIB or snapshot['memory']['MemAvailable']<4*GIB:raise RuntimeError('resource reserve')

def supervise(driver,log,env,lockpath=ROOT/'ouro_watch.lock'):
 install_signal_handlers()
 fd=os.open(lockpath,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
 try:
  fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);check(machine(),0,True)
  childenv=env.copy();childenv['ABCA_LOCK_FD']=str(fd)
  with Path(log).open('x') as f:
   with owned_process([str(ROOT/'conda-env/bin/python'),'-u',str(driver)],cwd=ROOT,env=childenv,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,pass_fds=(fd,)) as proc:
    started=time.monotonic()
    while proc.poll() is None:
     time.sleep(5);check(machine(),time.monotonic()-started)
    if proc.returncode:raise RuntimeError('child exited '+str(proc.returncode))
 finally:os.close(fd)
if __name__=='__main__':raise SystemExit('DISABLED: driver review and control-group service required')
