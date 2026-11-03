"""Bounded GPU child and durable charging; never uses old experiments' budgets."""
from pathlib import Path
import argparse,fcntl,json,os,signal,subprocess,time
from runtime import ROOT,save

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--seconds',type=int,required=True);ap.add_argument('worker',nargs=argparse.REMAINDER);a=ap.parse_args()
 assert a.name.replace('_','').isalnum() and 0<a.seconds<=8*3600
 args=a.worker[1:] if a.worker[:1]==['--'] else a.worker
 script=(ROOT/'scripts'/args[0]).resolve();assert script.is_relative_to(ROOT/'scripts') and script.is_file()
 lock=(ROOT/'logs/gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ledger=ROOT/'logs/gpu_budget.json';budget=json.loads(ledger.read_text()) if ledger.exists() else dict(limit_seconds=28800,used_seconds=0,runs=[])
 remaining=budget['limit_seconds']-budget['used_seconds'];assert remaining>30
 limit=min(a.seconds,remaining-30);start=time.time();name=a.name
 state=dict(status='starting',name=name,started_unix=start,limit_seconds=limit,supervisor_pid=os.getpid())
 save('logs/'+name+'_SUPERVISOR.json',state)
 # These interpreter/cache variables must be present BEFORE Python startup.
 env=dict(os.environ,GPU_DEADLINE_UNIX=str(start+limit),
   PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
   TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache/xdg'))
 try:
  with (ROOT/'logs'/f'{name}.out').open('x') as out,(ROOT/'logs'/f'{name}.err').open('x') as err:
   child=subprocess.Popen([str(ROOT/'env/bin/python'),'-u',str(script),*args[1:]],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
   state.update(status='running',worker_pid=child.pid);save('logs/'+name+'_SUPERVISOR.json',state)
   try:code=child.wait(timeout=limit)
   except subprocess.TimeoutExpired:
    os.killpg(child.pid,signal.SIGTERM)
    try:code=child.wait(timeout=15)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
    state['timed_out']=True
  state.update(status='complete' if code==0 else 'failed',exit_code=code)
 finally:
  elapsed=time.time()-start;state.update(finished_unix=time.time(),gpu_worker_seconds=elapsed)
  budget['used_seconds']+=elapsed;budget['runs'].append(state.copy());save('logs/gpu_budget.json',budget);save('logs/'+name+'_SUPERVISOR.json',state)
 print(json.dumps(state),flush=True)
 if state.get('exit_code',1)!=0:raise SystemExit(state.get('exit_code',1))

if __name__=='__main__':main()
