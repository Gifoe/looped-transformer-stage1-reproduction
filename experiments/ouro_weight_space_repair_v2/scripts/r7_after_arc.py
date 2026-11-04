"""Detached CPU coordinator: acceptance, deployment, conditional transfer, report."""
import fcntl,json,os,subprocess,time
from r7_core import ROOT,read,save,sha


def state(stage,**kwargs):
    value=dict(status='running',stage=stage,pid=os.getpid(),updated_unix=time.time(),**kwargs)
    save('logs/r7_AFTER_ARC_STATUS.json',value);print(json.dumps(value),flush=True)


def cpu(script):
    assert subprocess.call([str(ROOT/'env/bin/python'),'-B','-u',str(ROOT/'scripts'/script)],cwd=ROOT)==0,script+' failed'


def gpu(name,script,phase,*args):
    sp=f'logs/r7_{name}_SUPERVISOR.json'
    if not (ROOT/sp).exists():
        code=subprocess.call([str(ROOT/'env/bin/python'),'-B','-u',str(ROOT/'scripts/r7_supervise.py'),'--name',name,'--seconds','20000',
            '--phase',phase,'--',script,*args],cwd=ROOT)
        assert code==0,(name,'failed',code)
    else:assert read(sp)['status']=='complete',('Existing attempt must be inspected',read(sp))


def work():
    state('waiting_for_complete_ARC')
    while not (ROOT/'results/r7_ARC_COMPLETE.json').exists():
        p=ROOT/'logs/r7_PIPELINE_STATUS.json'
        if p.exists():assert read('logs/r7_PIPELINE_STATUS.json')['status']!='failed','ARC pipeline failed; inspect before resuming'
        time.sleep(10)
    state('exact_R6_acceptance');cpu('r7_accept.py')
    state('deployment_profile');gpu('deployment','r7_deploy.py','arc_test')
    if read('results/r7_ARC_COMPLETE.json')['correct']>736:
        state('external_preparation');cpu('r7_external_prepare.py')
        for task in ['openbookqa','hellaswag']:
            state('external_'+task);gpu(task,'r7_external_worker.py','external','--task',task)
    state('reporting');cpu('r7_report.py')
    save('logs/r7_AFTER_ARC_STATUS.json',dict(status='experiments_complete_publication_pending',stage='packaging',updated_unix=time.time(),
        primary=read('results/r7_ARC_COMPLETE.json')['primary'],arc_correct=read('results/r7_ARC_COMPLETE.json')['correct']))
    cpu('r7_package.py');print('R7_ALL_EXPERIMENTS_AND_PACKAGE_COMPLETE',flush=True)


def main():
    with (ROOT/'logs/r7_after_arc.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.environ.update(MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),TMPDIR=str(ROOT/'tmp'),HF_HOME=str(ROOT/'cache/hf'),XDG_CACHE_HOME=str(ROOT/'cache/xdg'))
        try:work()
        except Exception as error:
            save('logs/r7_AFTER_ARC_STATUS.json',dict(status='failed',error=repr(error),updated_unix=time.time(),pid=os.getpid()));raise


if __name__=='__main__':main()
