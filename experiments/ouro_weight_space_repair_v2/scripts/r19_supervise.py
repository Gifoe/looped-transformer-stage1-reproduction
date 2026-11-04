"""Unchanged original six-hour aggregate cap; one 120s+15s worker."""
import fcntl,os,signal,subprocess,time
from r19_common import ROOT,read,sha,save,gate


def main():
    spec=gate();book=spec['budget'];assert book['limit_seconds']==120 and book['termination_grace_seconds']==15
    with (ROOT/'logs/r7_gpu.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for p,h in book['inherited_ledger_sha256'].items():assert sha(p)==h
        inherited=sum(read(p)['used_seconds'] for p in book['inherited_ledger_sha256']);assert abs(inherited-book['inherited_total_seconds'])<1e-6
        assert not any((ROOT/p).exists() for p in ['logs/r19_gpu_budget.json','logs/r19_SUPERVISOR.json']),'No unreviewed retry'
        assert inherited+120+15<21600
        start=time.time();state=dict(status='starting',name='r19_last_loop_protection_pilot',phase='post_evaluation_last_loop_protection',supervisor_pid=os.getpid(),started_unix=start,limit_seconds=120,termination_grace_seconds=15)
        save('logs/r19_SUPERVISOR.json',state);env=dict(os.environ,GPU_DEADLINE_UNIX=str(start+120),PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'))
        try:
            with (ROOT/'logs/r19_worker.out').open('x') as out,(ROOT/'logs/r19_worker.err').open('x') as err:
                p=subprocess.Popen([str(ROOT/'env/bin/python'),'-B','-u','scripts/r19_worker.py'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                state.update(status='running',worker_pid=p.pid);save('logs/r19_SUPERVISOR.json',state)
                try:code=p.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:code=p.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
                    state['timed_out']=True
                state.update(status='complete' if code==0 else 'failed',exit_code=code)
        except BaseException as error:state.update(status='failed',error=repr(error));raise
        finally:
            elapsed=time.time()-start;state.update(finished_unix=time.time(),gpu_worker_seconds=elapsed);save('logs/r19_SUPERVISOR.json',state)
            save('logs/r19_gpu_budget.json',dict(original_limit_seconds=21600,inherited_total_seconds=inherited,used_seconds=elapsed,total_used_seconds=inherited+elapsed,runs=[state]))
        if state.get('exit_code',1):raise SystemExit(state.get('exit_code',1))


if __name__=='__main__':main()
