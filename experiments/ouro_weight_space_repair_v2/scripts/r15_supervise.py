"""One post-evaluation probe under the unchanged aggregate six-hour cap."""
import fcntl,os,signal,subprocess,time
from r15_common import ROOT,read,sha,save,gate


def main():
    spec=gate();book=spec['budget'];assert book['limit_seconds']==240
    with (ROOT/'logs/r7_gpu.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for p,h in book['inherited_ledger_sha256'].items():assert sha(p)==h
        inherited=sum(read(p)['used_seconds'] for p in book['inherited_ledger_sha256']);assert abs(inherited-book['inherited_total_seconds'])<1e-6
        assert not (ROOT/'logs/r15_gpu_budget.json').exists() and not (ROOT/'logs/r15_probe_SUPERVISOR.json').exists(),'Explicit inspection required; no automatic retry'
        assert inherited+240+15<21600
        started=time.time();state=dict(status='starting',name='r15_context_probe',phase='post_evaluation_observation',supervisor_pid=os.getpid(),started_unix=started,limit_seconds=240,termination_grace_seconds=15,worker_source_sha256=sha('scripts/r15_context_worker.py'))
        save('logs/r15_probe_SUPERVISOR.json',state);env=dict(os.environ,GPU_DEADLINE_UNIX=str(started+240),PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'))
        try:
            with (ROOT/'logs/r15_probe.out').open('x') as out,(ROOT/'logs/r15_probe.err').open('x') as err:
                p=subprocess.Popen([str(ROOT/'env/bin/python'),'-B','-u','scripts/r15_context_worker.py'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                state.update(status='running',worker_pid=p.pid);save('logs/r15_probe_SUPERVISOR.json',state)
                try:code=p.wait(timeout=240)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:code=p.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
                    state['timed_out']=True
                state.update(status='complete' if code==0 else 'failed',exit_code=code)
        except BaseException as error:
            state.update(status='failed',error=repr(error));raise
        finally:
            elapsed=time.time()-started;state.update(finished_unix=time.time(),gpu_worker_seconds=elapsed)
            save('logs/r15_probe_SUPERVISOR.json',state);save('logs/r15_gpu_budget.json',dict(original_limit_seconds=21600,inherited_total_seconds=inherited,used_seconds=elapsed,total_used_seconds=inherited+elapsed,runs=[state]))
        if state.get('exit_code',1):raise SystemExit(state.get('exit_code',1))


if __name__=='__main__':main()
