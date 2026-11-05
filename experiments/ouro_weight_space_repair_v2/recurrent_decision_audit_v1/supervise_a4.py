"""One bounded native-loop audit attempt; actual wall time is never reset."""
from pathlib import Path
import sys,subprocess,time,os,signal,json,hashlib
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
H=lambda b:hashlib.sha256(b).hexdigest()
OUTPUTS=['a4_PARITY.json','a4_RUNTIME_ADMISSION.json','a4_RAW.jsonl','a4_PROGRESS.json','a4_NATIVE_VECTORS.npz','a4_VECTOR_INDEX.json','a4_TIMINGS.json','a4_COMPLETE.json','a4_SUPERVISOR.json','a4_gpu_budget.json','a4_worker.out','a4_worker.err','a4_ANALYSIS.json']
def main():
 lock=json.loads((OUT/'a4_EXECUTION_LOCK.json').read_text());assert lock['status']=='frozen_native_loop_audit';assert all(not (OUT/p).exists() for p in OUTPUTS)
 inherited=lock['inherited_ledgers'];total=0
 for p,h in inherited.items():
  b=(ROOT/p).read_bytes();assert H(b)==h;total+=json.loads(b)['used_seconds']
 assert total==20834.389865819365 and total+500<=21600
 import fcntl
 guard=(ROOT/'logs/r7_gpu.lock').open('a+');fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 begin=time.monotonic();wall=time.time();env=dict(os.environ,GPU_DEADLINE_UNIX=str(wall+485));code=None;timed=False;proc=None
 try:
  with (OUT/'a4_worker.out').open('x') as out,(OUT/'a4_worker.err').open('x') as err:
   proc=subprocess.Popen(['nice','-n','15',sys.executable,'-B',str(OUT/'collect_a4.py')],cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
   print(json.dumps(dict(supervisor_pid=os.getpid(),worker_pid=proc.pid,start_unix=wall)),flush=True)
   while proc.poll() is None:
    elapsed=time.monotonic()-begin
    if elapsed>=499:timed=True;os.killpg(proc.pid,signal.SIGKILL);break
    if elapsed>=485 and not timed:timed=True;os.killpg(proc.pid,signal.SIGTERM)
    time.sleep(.1)
   code=proc.wait(timeout=max(.1,500-(time.monotonic()-begin)))
 finally:
  if proc is not None and proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  actual=time.monotonic()-begin;old_exact=all(H((ROOT/p).read_bytes())==h for p,h in inherited.items());complete=OUT/'a4_COMPLETE.json';status='complete' if code==0 and complete.exists() and not timed and actual<=500 and old_exact else 'incomplete_or_invalid'
  receipt=dict(status=status,supervisor_pid=os.getpid(),worker_pid=proc.pid if proc else None,start_unix=wall,exit_code=code,timed_out=timed,overrun=actual>500,gpu_worker_seconds=actual,old_ledgers_unchanged=old_exact,protocol_sha256=H((OUT/'a4_EXECUTION_LOCK.json').read_bytes()))
  (OUT/'a4_SUPERVISOR.json').write_text(json.dumps(receipt,indent=2)+'\n');(OUT/'a4_gpu_budget.json').write_text(json.dumps(dict(used_seconds=actual,inherited_used_seconds=total,total_used_seconds=total+actual,cap_seconds=21600,remaining_seconds=21600-total-actual,old_ledgers_unchanged=old_exact),indent=2)+'\n');fcntl.flock(guard.fileno(),fcntl.LOCK_UN);guard.close();print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
