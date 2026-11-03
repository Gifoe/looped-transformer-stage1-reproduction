"""Sequential frozen final evaluation; no alternate test winner and no retuning."""
import os,subprocess,time
from slcd_core import ROOT,read,save,sha

def run(args):
    subprocess.run([str(ROOT/'env/bin/python'),'-u',*args],cwd=ROOT,check=True)

def gpu(name,seconds,args):
    run(['scripts/slcd_supervise.py','--name',name,'--seconds',str(seconds),'--',*args])

def external(task,seconds,minimum_remaining):
    """Preserve an external failure/timeout and still deliver completed ARC evidence."""
    complete=ROOT/f'results/slcd_{task}_COMPLETE.json'
    if complete.exists():
        receipt=read(str(complete.relative_to(ROOT)))
        assert receipt['status']=='complete' and receipt['n']=={'hellaswag':10042,'openbookqa':500}[task]
        return dict(status='complete',completion_sha256=sha(complete.relative_to(ROOT)))
    budget=read('logs/slcd_gpu_budget.json');remaining=budget['limit_seconds']-budget['used_seconds']
    if remaining<=minimum_remaining:
        result=dict(status='not_run',reason='insufficient remaining GPU-worker budget',remaining_seconds=remaining)
    else:
        # A previous terminal attempt is preserved rather than overwritten or
        # automatically retried under a reused supervisor name.
        supervisor=ROOT/f'logs/slcd_{task}_SUPERVISOR.json'
        if supervisor.exists():
            state=read(str(supervisor.relative_to(ROOT)))
            assert state['status'] not in ['starting','running'],'External worker still active'
            result=dict(status='incomplete',reason='prior attempt ended without complete task receipt',supervisor=state)
        else:
            try:
                gpu(task,min(seconds,int(remaining)-60),['slcd_external_worker.py','--task',task])
                receipt=read(str(complete.relative_to(ROOT)))
                assert receipt['status']=='complete' and receipt['n']=={'hellaswag':10042,'openbookqa':500}[task]
                return dict(status='complete',completion_sha256=sha(complete.relative_to(ROOT)))
            except subprocess.CalledProcessError as exc:
                state=read(str(supervisor.relative_to(ROOT))) if supervisor.exists() else None
                result=dict(status='incomplete',reason='supervised worker failed or exhausted its budget; no partial accuracy reported',
                            returncode=exc.returncode,supervisor=state)
    save(f'results/slcd_{task}_NOT_COMPLETE.json',result)
    return result

def main():
    # Validation was explicitly launched under its frozen lock. Wait only for
    # terminal supervisor state; never interpret partial candidate scores.
    wait_started=time.time()
    while True:
        state=read('logs/slcd_validation_SUPERVISOR.json')
        if state['status']=='complete':
            assert state['exit_code']==0;break
        if state['status']=='failed':raise RuntimeError('Validation failed; preserve failure before diagnosis')
        assert time.time()-wait_started<21600,'Bounded validation wait exceeded'
        time.sleep(15)
    if not (ROOT/'FINAL_METHOD_LOCK.json').exists():run(['scripts/slcd_finalize.py','select'])
    lock=read('FINAL_METHOD_LOCK.json')
    need=any(s['kind']=='token' for s in lock['method_registry'].values())
    if need and not (ROOT/'results/slcd_tokens_test_final_COMPLETE.json').exists():
        gpu('arc_test',3600,['slcd_token_worker_v2.py','--split','test','--job','final','--recipes','configs/slcd_TOKEN_TEST.json'])
    if not (ROOT/'results/slcd_ARC_TEST_COMPLETE.json').exists():run(['scripts/slcd_finalize.py','test'])
    result=read('results/slcd_ARC_TEST_COMPLETE.json')
    assert result['status']=='complete' and result['n']==1172 and result['primary']==lock['primary']
    assert result['final_lock_sha256']==sha('FINAL_METHOD_LOCK.json')
    assert result['passed_absolute_gate']==result['other_benchmarks_authorized']==(result['primary_correct']>717)
    external_status={}
    if result['other_benchmarks_authorized']:
        external_status['hellaswag']=external('hellaswag',14400,180)
        external_status['openbookqa']=external('openbookqa',1800,900)
    else:
        save('results/slcd_OTHER_BENCHMARKS_STOPPED.json',dict(reason='Frozen ARC primary did not exceed717/1172; no test reselection or retuning',primary=result['primary'],correct=result['primary_correct']))
    save('results/slcd_EXTERNAL_EXECUTION.json',dict(authorized=result['other_benchmarks_authorized'],tasks=external_status))
    run(['scripts/slcd_make_deliverables.py'])
    save('results/slcd_FINAL_CHAIN_COMPLETE.json',dict(status='complete',primary=result['primary'],arc_correct=result['primary_correct'],external_tasks=external_status))

if __name__=='__main__':main()
