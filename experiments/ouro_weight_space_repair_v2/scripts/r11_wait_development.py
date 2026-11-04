"""Wait for the already running exact GPU worker, then do bounded CPU replay."""
import json,subprocess,time
from pathlib import Path
from r7_core import ROOT,read,sha


def state(status,**values):
    p=ROOT/'logs/r11_CPU_STATUS.json';tmp=p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(dict(status=status,updated_unix=time.time(),**values),indent=2)+'\n');tmp.replace(p)


def main():
    state('waiting_specific_R10_worker',worker_pid=652879)
    while True:
        s=read('logs/r10_development_SUPERVISOR.json')
        assert s['worker_pid']==652879 and s['supervisor_pid']==652859
        if s['status'] in ['complete','failed']:
            assert s['status']=='complete' and s['exit_code']==0,('R10 collection failed; never auto restart',s)
            break
        for pid,script in [(652859,'r10_supervise.py'),(652879,'r10_worker.py')]:
            p=Path('/proc')/str(pid)/'cmdline'
            if not p.exists() or script not in p.read_bytes().decode():
                time.sleep(1);terminal=read('logs/r10_development_SUPERVISOR.json')
                assert terminal['status']=='complete',('Live process missing with unsettled supervisor; inspect, no restart',pid)
        time.sleep(15)
    state('verifying_complete_joint_development')
    from r10_evaluate import table
    _,t=table('development');by={r['method']:r for r in t}
    correspondence={'joint4_fusion':'joint4_only','joint4_protected':'joint4_only','jointFixed_protected':'jointFixed_only','balanced4_protected':'balanced4_only'}
    contrast=[dict(method=n,corresponding_joint=k,correct=by[n]['correct'],joint_correct=by[k]['correct'],net_vs_joint=by[n]['correct']-by[k]['correct']) for n,k in correspondence.items()]
    report=dict(status='stopped_for_goal_fidelity_after_development',n=576,comparison=contrast,original_preregistered_gate_claimed_failed=False,
        additional_attribution_rule_added_after_partial_feedback=True,scope_change_sha256=sha('logs/r10_SCOPE_REALIGNMENT.json'),
        no_R10_confirmation=True,no_R10_validation=True,no_R10_test=True,external_evaluation_authorized=False,
        reason='Joint-format diagnostic retained. Direct fusion does not establish prefix-conditional prior repair; reuse forwards to test R11 instead.')
    (ROOT/'results/r10_MECHANISM_STOP.json').write_text(json.dumps(report,indent=2)+'\n')
    assert not (ROOT/'configs/r10_CANDIDATE_LOCK.json').exists()
    state('running_frozen_R11_CPU_replay')
    subprocess.run([str(ROOT/'env/bin/python'),'-B','scripts/r11_development.py'],cwd=ROOT,check=True)
    result=read('results/r11_DEVELOPMENT_COMPLETE.json')
    state('needs_candidate_freeze' if result['eligible_for_confirmation'] else 'R11_No_Go',eligible_for_confirmation=result['eligible_for_confirmation'],
        R11_receipt_sha256=sha('results/r11_DEVELOPMENT_COMPLETE.json'),all_worker_costs_settled=read('logs/r10_gpu_budget.json')['used_seconds'])


if __name__=='__main__':
    try:main()
    except Exception as error:
        state('failed_requires_inspection',error=repr(error));raise
