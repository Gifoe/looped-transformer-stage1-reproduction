"""Explicitly invoked orchestration AFTER strict original-path readiness/discovery.

Does not run on import. Never evaluates validation/test. GPU subprocesses always
use the existing independent-budget supervisor. Active old workers are untouched.
"""
import argparse
import json
import subprocess
import sys
import time

from runtime import ROOT, save


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--readiness-study',default='original_call_parity')
    ap.add_argument('--discovery-study',default='discovery_original')
    ap.add_argument('--screen-study',default='layer_screen_original')
    ap.add_argument('--shared-study',default='shared_confirmation_original')
    ap.add_argument('--discovery-n',type=int,default=192)
    ap.add_argument('--screen-seconds',type=int,default=2400)
    ap.add_argument('--shared-seconds',type=int,default=900)
    ap.add_argument('--run-tag',default='original_pipeline_v1')
    a=ap.parse_args()
    for value in [a.readiness_study,a.discovery_study,a.screen_study,a.shared_study,a.run_tag]:
        assert value.replace('_','').isalnum()
    readiness=json.loads((ROOT/'results'/a.readiness_study/'COMPLETE.json').read_text())
    assert readiness['exact_original_path_parity'] is True
    assert readiness['original_repeat_exact'] is True
    assert readiness['exact_intermediate_depth_parity'] is True
    discovery=json.loads((ROOT/'results'/a.discovery_study/f'COMPLETE_{a.discovery_n}.json').read_text())
    assert discovery['status']=='complete' and discovery['n']==a.discovery_n
    state=dict(status='running',started_unix=time.time(),arguments=vars(a),steps=[])
    status_path='logs/'+a.run_tag+'_PIPELINE.json'
    def run(label,args):
        step=dict(name=label,started_unix=time.time(),args=args,status='running')
        state['steps'].append(step);save(status_path,state)
        command=[sys.executable,str(ROOT/'scripts'/args[0]),*args[1:]]
        log=ROOT/'logs'/f'{a.run_tag}_{label}.log'
        try:
            with log.open('a',encoding='utf8') as stream:
                result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
            step.update(status='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,finished_unix=time.time())
            save(status_path,state)
            if result.returncode:raise RuntimeError(f'{label} failed; inspect {log}')
        except BaseException:
            state['status']='failed';save(status_path,state);raise
    run('discovery_analysis',['analyze_stage_a.py','--study',a.discovery_study])
    if not (ROOT/'results'/a.screen_study/'COMPLETE.json').exists():
        run('screen',['supervise.py','--name',a.run_tag+'_screen','--seconds',str(a.screen_seconds),'--',
            'run_layer_screen.py','--study',a.screen_study])
    run('screen_analysis',['analyze_layer_screen.py','--study',a.screen_study])
    mechanism=json.loads((ROOT/'results'/a.screen_study/'MECHANISM_ANALYSIS.json').read_text())
    qualified=any(row['decision'] in ('localization_only_requires_shared_invocation_check',
                  'eligible_for_bounded_label_blind_direction_diagnostic')
                  for row in mechanism['top_at_most_two_modules'])
    if qualified and not (ROOT/'results'/a.shared_study/'COMPLETE.json').exists():
        run('shared_confirmation',['supervise.py','--name',a.run_tag+'_shared','--seconds',str(a.shared_seconds),'--',
            'run_shared_confirmation.py','--screen-study',a.screen_study,'--study',a.shared_study])
    state.update(status='complete',finished_unix=time.time(),shared_confirmation_qualified=qualified,
        next_action='Independent scientific review; no automatic PCA, weights, validation, or test execution')
    save(status_path,state);print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
