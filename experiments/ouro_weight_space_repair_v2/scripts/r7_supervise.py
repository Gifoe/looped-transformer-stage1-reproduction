"""Independent21600s R7 ledger; reserve5400s for full ARC until final stage."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from r7_core import ROOT,save,read,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--seconds',type=int,required=True)
    parser.add_argument('--phase',choices=['development','arc_test','external'],default='development');parser.add_argument('worker',nargs=argparse.REMAINDER);args=parser.parse_args()
    assert args.name.replace('_','').isalnum() and 0<args.seconds<=21600
    worker=args.worker[1:] if args.worker[:1]==['--'] else args.worker
    script=(ROOT/'scripts'/worker[0]).resolve();assert script.parent==ROOT/'scripts' and script.name.startswith('r7_') and script.is_file()
    with (ROOT/'logs/r7_gpu.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        budget=read('logs/r7_gpu_budget.json') if (ROOT/'logs/r7_gpu_budget.json').exists() else dict(limit_seconds=21600,used_seconds=0.,reserved_arc_seconds=5400,runs=[])
        remaining=budget['limit_seconds']-budget['used_seconds']
        if args.phase=='external':
            arc=read('results/r7_ARC_COMPLETE.json');assert arc['status']=='complete' and arc['n']==1172 and arc['correct']>736
        reserve=5400 if args.phase=='development' else 30
        limit=min(args.seconds,remaining-reserve-15);assert limit>30,'ARC reserve / total budget prevents this worker'
        name='r7_'+args.name;state_path='logs/'+name+'_SUPERVISOR.json'
        assert not (ROOT/state_path).exists(),'Existing attempt must be inspected; never restart on observation timeout'
        start=time.time();state=dict(status='starting',name=name,phase=args.phase,supervisor_pid=os.getpid(),started_unix=start,
            limit_seconds=limit,worker_script_sha256=sha('scripts/'+script.name))
        save(state_path,state)
        env=dict(os.environ,GPU_DEADLINE_UNIX=str(start+limit),PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'),
                 MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache/xdg'),HF_HOME=str(ROOT/'cache/hf'))
        try:
            with (ROOT/f'logs/{name}.out').open('x') as out,(ROOT/f'logs/{name}.err').open('x') as err:
                child=subprocess.Popen([str(ROOT/'env/bin/python'),'-B','-u',str(script),*worker[1:]],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                state.update(status='running',worker_pid=child.pid);save(state_path,state)
                try:code=child.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:code=child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
                    state['timed_out']=True
            state.update(status='complete' if code==0 else 'failed',exit_code=code)
        finally:
            elapsed=time.time()-start;state.update(finished_unix=time.time(),gpu_worker_seconds=elapsed)
            budget['used_seconds']+=elapsed;budget['runs'].append(state.copy());save('logs/r7_gpu_budget.json',budget);save(state_path,state)
        print(json.dumps(state),flush=True)
        if state.get('exit_code',1):raise SystemExit(state.get('exit_code',1))


if __name__=='__main__':main()
