"""Detached experiment-local continuation; no account messages or external writes.

Waits for the finite GPU pipeline, runs one fixed training-only replay gate,
then CPU analysis/audit/report/package. Fails closed on worker/script errors.
"""
import json
import os
import subprocess
import time
from runtime import ROOT,save


def main():
    env=dict(os.environ,PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'),
        MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),TMPDIR=str(ROOT/'tmp'),
        HF_HOME=str(ROOT/'cache/hf'),XDG_CACHE_HOME=str(ROOT/'cache/xdg'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
    started=time.time();save('logs/FINISHER_STATUS.json',dict(status='waiting_for_pipeline',started_unix=started))
    while True:
        state=json.loads((ROOT/'logs/edit_pipeline_SUPERVISOR.json').read_text())
        if state['status']=='failed':raise RuntimeError('Main bounded GPU worker failed; inspect its complete logs')
        if state['status']=='complete':
            assert state['exit_code']==0 and (ROOT/'results/PIPELINE_COMPLETE.json').exists();break
        assert time.time()-started<18500,'Finisher wait exceeded bounded pipeline duration'
        time.sleep(20)
    commands=[('selected_checkpoint_gate',[str(ROOT/'scripts/supervise.py'),'--name','selected_checkpoint_gate',
        '--seconds','600','--','verify_selected_checkpoint.py']),
        ('final_analysis',[str(ROOT/'scripts/analyze_v2.py')]),
        ('artifact_audit',[str(ROOT/'scripts/audit_v2.py')]),
        ('report_assembly',[str(ROOT/'scripts/assemble_report.py')]),
        ('package_results',[str(ROOT/'scripts/package_v2.py')])]
    for name,args in commands:
        save('logs/FINISHER_STATUS.json',dict(status='running',stage=name,started_unix=started))
        with (ROOT/'logs'/(name+'_launcher.out')).open('x') as out,(ROOT/'logs'/(name+'_launcher.err')).open('x') as err:
            result=subprocess.run([str(ROOT/'env/bin/python'),'-u',*args],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=1200)
        if result.returncode:raise RuntimeError(f'{name} failed with returncode {result.returncode}; inspect preserved logs')
    save('logs/FINISHER_STATUS.json',dict(status='complete',started_unix=started,finished_unix=time.time()))
    print('V2_FINISHER_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save('logs/FINISHER_FAILURE.json',dict(status='failed',error=repr(exc),time_unix=time.time()))
        raise
