"""Single bounded attempt, actual worker-inclusive wall debit, no retries."""
import os,sys,time,signal,subprocess
from core import ROOT,OUT,H,read,save,gate,OUTPUTS

def main():
 s=gate();assert all(not (OUT/p).exists() for p in OUTPUTS);old=sum(read(ROOT/p)['used_seconds'] for p in s['inherited_ledgers']);assert len(s['inherited_ledgers'])==20 and old==20889.326696651056 and old+680<=21600
 import fcntl
 mutex=(ROOT/'logs/r7_gpu.lock').open('a+');fcntl.flock(mutex.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);start=time.monotonic();wall=time.time();p=None;code=None;timed=False
 try:
  with (OUT/'worker.out').open('x') as out,(OUT/'worker.err').open('x') as err:
   env=dict(os.environ,GPU_DEADLINE_UNIX=str(wall+665));p=subprocess.Popen([sys.executable,'-B','-u',str(OUT/'worker.py')],cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True,preexec_fn=lambda:os.nice(15));print(__import__('json').dumps(dict(supervisor_pid=os.getpid(),worker_pid=p.pid,start_unix=wall)),flush=True)
   while p.poll() is None:
    elapsed=time.monotonic()-start
    if elapsed>=679:timed=True;os.killpg(p.pid,signal.SIGKILL);break
    if elapsed>=665 and not timed:timed=True;os.killpg(p.pid,signal.SIGTERM)
    time.sleep(.1)
   code=p.wait(timeout=max(.1,680-(time.monotonic()-start)))
 finally:
  if p is not None and p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
  actual=time.monotonic()-start;unchanged=all(H((ROOT/n).read_bytes())==h for n,h in s['inherited_ledgers'].items());valid=code==0 and (OUT/'COMPLETE.json').exists() and not timed and actual<=680 and unchanged
  save('SUPERVISOR.json',dict(status='complete' if valid else 'incomplete_or_invalid',supervisor_pid=os.getpid(),worker_pid=p.pid if p else None,start_unix=wall,exit_code=code,timed_out=timed,overrun=actual>680,gpu_worker_seconds=actual,old_ledgers_unchanged=unchanged,protocol_sha256=H((OUT/'EXECUTION_LOCK.json').read_bytes())));save('gpu_budget.json',dict(used_seconds=actual,inherited_used_seconds=old,total_used_seconds=old+actual,remaining_seconds=21600-old-actual,cap_seconds=21600,old_ledgers_unchanged=unchanged));fcntl.flock(mutex.fileno(),fcntl.LOCK_UN);mutex.close();print('B_SUPERVISOR_TERMINAL',code,actual,flush=True)
if __name__=='__main__':main()
