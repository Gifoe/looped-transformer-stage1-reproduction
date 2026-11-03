"""Continue CPU delivery after the documented observer-fingerprint gate retry.

Preserves failed receipts and all frozen scores. No candidate/model selection.
"""
import json,os,subprocess,time
from runtime import ROOT,save


def main():
    env=dict(os.environ,PYTHONPYCACHEPREFIX=str(ROOT/'cache/pycache'),
        MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),TMPDIR=str(ROOT/'tmp'),
        HF_HOME=str(ROOT/'cache/hf'),XDG_CACHE_HOME=str(ROOT/'cache/xdg'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    started=time.time()
    save('logs/FINISHER_STATUS.json',dict(status='waiting_for_gate_retry',started_unix=started))
    while True:
        state=json.loads((ROOT/'logs/selected_checkpoint_gate_retry1_SUPERVISOR.json').read_text())
        if state['status']=='failed':raise RuntimeError('Selected gate retry failed; preserve and inspect its logs')
        if state['status']=='complete':
            assert state['exit_code']==0;break
        assert time.time()-started<650;time.sleep(2)
    assert json.loads((ROOT/'results/SELECTED_CHECKPOINT_AUDIT.json').read_text())['status']=='passed'
    save('logs/FINISHER_RECOVERY.json',dict(status='resolved',failed_stage='selected_checkpoint_gate',
        failure='Observer hook registration changed the scorer fingerprint without invalidation',
        correction='Call scorer.invalidate after registering and after removing observer hooks',
        successful_retry='selected_checkpoint_gate_retry1',frozen_1172_scores_and_weights_untouched=True,
        failed_attempt_kept_and_gpu_time_charged=True))
    for name in ['final_analysis','artifact_audit','report_assembly','package_results']:
        script={'final_analysis':'analyze_v2.py','artifact_audit':'audit_v2.py',
            'report_assembly':'assemble_report.py','package_results':'package_v2.py'}[name]
        save('logs/FINISHER_STATUS.json',dict(status='running',stage=name,started_unix=started,recovery=True))
        with (ROOT/'logs'/(name+'_launcher.out')).open('x') as out,(ROOT/'logs'/(name+'_launcher.err')).open('x') as err:
            result=subprocess.run([str(ROOT/'env/bin/python'),'-u',str(ROOT/'scripts'/script)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=1200)
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    save('logs/FINISHER_STATUS.json',dict(status='complete',started_unix=started,finished_unix=time.time(),recovered=True))
    print('RECOVERED_FINISHER_COMPLETE',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        save('logs/FINISHER_RECOVERY_FAILURE.json',dict(status='failed',error=repr(exc),time_unix=time.time()))
        raise
