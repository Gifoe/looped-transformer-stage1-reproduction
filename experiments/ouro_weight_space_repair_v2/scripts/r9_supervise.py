"""Carry the settled R7 spend into the same original21600s cap. No externals."""
import argparse,fcntl,os,signal,subprocess,time
from r9_core import ROOT,read,save,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seconds',type=int,required=True)
    p.add_argument('--phase',choices=['development','arc_test'],default='development');p.add_argument('worker',nargs=argparse.REMAINDER);a=p.parse_args()
    assert a.name.replace('_','').isalnum() and 0<a.seconds<=21600
    worker=a.worker[1:] if a.worker[:1]==['--'] else a.worker
    script=(ROOT/'scripts'/worker[0]).resolve();assert script.parent==ROOT/'scripts' and script.name.startswith('r9_') and script.is_file()
    protocol=read('configs/r9_PROTOCOL.json');inherited=protocol['budget']
    with (ROOT/'logs/r7_gpu.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert sha('logs/r7_gpu_budget.json')==inherited['settled_R7_ledger_sha256']
        budget=read('logs/r9_gpu_budget.json') if (ROOT/'logs/r9_gpu_budget.json').exists() else dict(
            original_total_limit_seconds=21600,inherited_R7_used_seconds=inherited['R7_used_seconds'],
            remaining_round_limit_seconds=inherited['remaining_seconds'],used_seconds=0.,runs=[])
        remaining=budget['remaining_round_limit_seconds']-budget['used_seconds']
        reserve=5400 if a.phase=='development' else 30
        limit=min(a.seconds,remaining-reserve-15);assert limit>30,'Original remaining budget / ARC reserve prevents worker'
        name='r9_'+a.name;statepath='logs/'+name+'_SUPERVISOR.json';assert not (ROOT/statepath).exists()
        started=time.time();state=dict(status='starting',name=name,phase=a.phase,supervisor_pid=os.getpid(),started_unix=started,
            limit_seconds=limit,worker_script_sha256=sha('scripts/'+script.name))
        save(statepath,state)
        env=dict(os.environ,GPU_DEADLINE_UNIX=str(started+limit),PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'),
            MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),TMPDIR=str(ROOT/'tmp'),XDG_CACHE_HOME=str(ROOT/'cache/xdg'),HF_HOME=str(ROOT/'cache/hf'))
        try:
            with (ROOT/f'logs/{name}.out').open('x') as out,(ROOT/f'logs/{name}.err').open('x') as err:
                child=subprocess.Popen([str(ROOT/'env/bin/python'),'-B','-u',str(script),*worker[1:]],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                state.update(status='running',worker_pid=child.pid);save(statepath,state)
                try:code=child.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:code=child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait()
                    state['timed_out']=True
            state.update(status='complete' if code==0 else 'failed',exit_code=code)
        finally:
            elapsed=time.time()-started;state.update(finished_unix=time.time(),gpu_worker_seconds=elapsed)
            budget['used_seconds']+=elapsed;budget['runs'].append(state.copy());save('logs/r9_gpu_budget.json',budget);save(statepath,state)
        print(state,flush=True)
        if state.get('exit_code',1):raise SystemExit(state.get('exit_code',1))


if __name__=='__main__':main()
