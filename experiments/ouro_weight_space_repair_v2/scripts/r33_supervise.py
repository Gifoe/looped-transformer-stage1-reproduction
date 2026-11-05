"""Proposed R33 hard200 lifecycle with sole actual debit and no retry."""
import fcntl,os,signal,subprocess,time,sys
from r33_common import ROOT,read,sha,save,gate,assert_fresh,OUTPUTS
def main():
 s=gate();b=s['budget'];assert b['hard_seconds']==200 and b['soft_seconds']==185 and b['kill_seconds']==199 and b['reserve_seconds']==5 and not b['provisional']
 with (ROOT/'logs/r7_gpu.lock').open('a') as mutex:
  fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB);assert len(b['inherited_ledger_sha256'])==18
  for p,h in b['inherited_ledger_sha256'].items():assert sha(p)==h
  old=sum(read(p)['used_seconds'] for p in b['inherited_ledger_sha256']);assert old==20755.46525352029 and old+200<21600;assert_fresh(OUTPUTS)
  start=time.monotonic();wall=time.time();p=None;state=dict(status='starting',supervisor_pid=os.getpid(),started_unix=wall,hard_limit_seconds=200,soft_stop_seconds=185,kill_seconds=199);env=dict(os.environ,GPU_DEADLINE_UNIX=str(wall+185),R33_SOFT_MONOTONIC=str(start+185),PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'))
  try:
   with (ROOT/'logs/r33_worker.out').open('x') as out,(ROOT/'logs/r33_worker.err').open('x') as err:
    p=subprocess.Popen([sys.executable,'-B','-u','scripts/r33_worker.py'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,preexec_fn=lambda:os.nice(15));state.update(status='running',worker_pid=p.pid);save('logs/r33_SUPERVISOR.json',state)
    try:code=p.wait(timeout=max(0,start+185-time.monotonic()))
    except subprocess.TimeoutExpired:
     state['timed_out']=True;os.killpg(p.pid,signal.SIGTERM)
     try:code=p.wait(timeout=max(0,start+199-time.monotonic()))
     except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
    state.update(status='complete' if code==0 else 'failed',exit_code=code)
  except BaseException as e:
   state.update(status='failed',error=repr(e))
   if p is not None and p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise
  finally:
   elapsed=time.monotonic()-start;state.update(finished_unix=time.time(),gpu_worker_seconds=elapsed,overrun=elapsed>200)
   if elapsed>200:state['status']='overrun_ineligible'
   save('logs/r33_SUPERVISOR.json',state);save('logs/r33_gpu_budget.json',dict(original_limit_seconds=21600,inherited_total_seconds=old,used_seconds=elapsed,total_used_seconds=old+elapsed,runs=[state]))
  if state['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
