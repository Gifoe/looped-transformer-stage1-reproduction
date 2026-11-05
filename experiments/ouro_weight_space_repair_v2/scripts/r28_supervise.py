"""R28 single-worker absolute monotonic deadlines, no retry, actual accounting."""
import fcntl,os,signal,subprocess,time
from r28_common import ROOT,read,sha,save,gate,assert_fresh,RUNTIME_OUTPUTS

def main():
    spec=gate();book=spec['budget']
    with (ROOT/'logs/r7_gpu.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert len(book['inherited_ledger_sha256'])==16
        for p,h in book['inherited_ledger_sha256'].items():assert sha(p)==h
        inherited=sum(read(p)['used_seconds'] for p in book['inherited_ledger_sha256']);assert inherited==18930.428655151278
        assert_fresh(RUNTIME_OUTPUTS)
        assert inherited+1500<21600
        start=time.monotonic();wall=time.time();p=None;state=dict(status='starting',started_unix=wall,supervisor_pid=os.getpid(),hard_limit_seconds=1500,soft_stop_seconds=1485,kill_seconds=1499)
        env=dict(os.environ,GPU_DEADLINE_UNIX=str(wall+1485),R28_SOFT_MONOTONIC=str(start+1485),PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'))
        try:
            with (ROOT/'logs/r28_worker.out').open('x') as out,(ROOT/'logs/r28_worker.err').open('x') as err:
                p=subprocess.Popen([str(ROOT/'env/bin/python'),'-B','-u','scripts/r28_worker.py'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,preexec_fn=lambda:os.nice(15))
                state.update(status='running',worker_pid=p.pid);save('logs/r28_SUPERVISOR.json',state)
                try:code=p.wait(timeout=max(0,start+1485-time.monotonic()))
                except subprocess.TimeoutExpired:
                    state['timed_out']=True;os.killpg(p.pid,signal.SIGTERM)
                    try:code=p.wait(timeout=max(0,start+1499-time.monotonic()))
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
                state.update(status='complete' if code==0 else 'failed',exit_code=code)
        except BaseException as error:
            state.update(status='failed',error=repr(error))
            if p is not None and p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
        finally:
            elapsed=time.monotonic()-start;state.update(finished_unix=time.time(),gpu_worker_seconds=elapsed,overrun=elapsed>1500)
            if elapsed>1500:state['status']='overrun_ineligible'
            save('logs/r28_SUPERVISOR.json',state);save('logs/r28_gpu_budget.json',dict(original_limit_seconds=21600,inherited_total_seconds=inherited,used_seconds=elapsed,total_used_seconds=inherited+elapsed,runs=[state]))
        if state['status']!='complete':raise SystemExit(1)
if __name__=='__main__':main()
