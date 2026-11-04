"""Bounded unattended ARC-only chain, all GPU work through inherited budget.

Never retunes a formula and never selects a post-test alternative. Pauses before
test until the desktop connector has published and verified the final lock.
"""
import json,os,subprocess,time
from r10_core import ROOT,read,sha,save,sources
from r10_scores import ELIGIBLE
from r10_heldout import extra_sources
from r10_evaluate import table,control,final_report

PRIORITY={'balanced4_protected':0,'joint4_protected':1,'jointFixed_protected':2,'joint4_fusion':3}


def status(stage,**kwargs):save('logs/r10_CHAIN_STATUS.json',dict(status=stage,updated_unix=time.time(),coordinator_pid=os.getpid(),external_evaluation_authorized=False,**kwargs))


def wait_development():
    while True:
        v=read('logs/r10_development_SUPERVISOR.json')
        if v['status'] in ['complete','failed']:
            assert v['status']=='complete',('Development failed',v)
            break
        time.sleep(20)


def run(name,phase,seconds,budget_phase='development'):
    path=ROOT/f'logs/r10_{name}_SUPERVISOR.json'
    if path.exists():
        v=read(str(path.relative_to(ROOT)));assert v['status']=='complete',('Inspect prior attempt; no automatic retry',v)
        return
    status('running_'+phase,job=name)
    subprocess.run([str(ROOT/'env/bin/python'),'-B','-u','scripts/r10_supervise.py','--name',name,'--seconds',str(seconds),
                    '--phase',budget_phase,'--','r10_heldout.py','--phase',phase],cwd=ROOT,check=True)
    assert read(str(path.relative_to(ROOT)))['status']=='complete'


def lock(path,**values):
    assert not (ROOT/path).exists(),'Locks are immutable, inspect existing before continuing'
    value=dict(created_unix=time.time(),protocol_sha256=sha('configs/r10_PROTOCOL.json'),source_sha256=sources(),
        extra_source_sha256=extra_sources(),post_pilot_audit_sha256=sha('configs/r10_POST_PILOT_AUDIT.json'),
        depth_attribution_addendum_sha256=sha('configs/r10_DEPTH_ATTRIBUTION_ADDENDUM.json'),**values)
    save(path,value)


def nogo(reason,**values):
    save('results/r10_NO_GO.json',dict(status='No-Go',reason=reason,**values));status('No-Go',reason=reason)


def main():
    assert sources()==read('configs/r10_PROTOCOL.json')['source_sha256'];status('waiting_development')
    wait_development();_,dev=table('development');dt={r['method']:r for r in dev};r6=dt['M0_R6']['correct']
    eligible=[n for n in ELIGIBLE if dt[n]['correct']>r6 and dt[n]['correct']>dt[control(n)]['correct']]
    ranked=sorted(eligible,key=lambda n:(-dt[n]['correct'],dt[n]['damage'],PRIORITY[n]))[:2]
    if not ranked:return nogo('No candidate beats R6 and own matched control on576',development=dev)
    lock('configs/r10_CANDIDATE_LOCK.json',status='frozen_before_current_confirmation',candidates=ranked,
        dataset_sha256=read('configs/r10_PROTOCOL.json')['dataset_sha256'],development_summary=dev,
        development_receipt_sha256=sha('results/r10_development_initial_COMPLETE.json'),
        requirement='Only shortlisted methods produce heldout outcomes; joint-only diagnostics are nonselectable attribution baselines')
    run('heldout_parity','parity',160)
    assert read('results/r10_parity_selected_COMPLETE.json')['max_parity_error']==0.
    run('confirmation','confirmation',2700);_,conf=table('confirmation');ct={r['method']:r for r in conf}
    pooled={n:dict(correct=dt[n]['correct']+ct[n]['correct'],rescue=dt[n]['rescue']+ct[n]['rescue'],damage=dt[n]['damage']+ct[n]['damage'],
        own_control_correct=dt[control(n)]['correct']+ct[control(n)]['correct']) for n in ranked}
    validate=[n for n in ranked if pooled[n]['correct']>706 and pooled[n]['correct']>pooled[n]['own_control_correct']]
    save('results/r10_TRAIN_COMPLETE.json',dict(status='complete',n=1093,R6_correct=705,candidates=ranked,summary=pooled,
        candidate_lock_sha256=sha('configs/r10_CANDIDATE_LOCK.json'),validation_candidates=validate))
    if not validate:return nogo('No shortlisted method beats706 and own control on fulltrain1093',pooled_train=pooled)
    lock('configs/r10_VALIDATION_LOCK.json',status='frozen_before_current_validation',candidates=validate,
        dataset_sha256=read('configs/r10_PROTOCOL.json')['dataset_sha256'],pooled_train=pooled,candidate_lock_sha256=sha('configs/r10_CANDIDATE_LOCK.json'))
    run('validation','validation',1600);_,val=table('validation');vt={r['method']:r for r in val}
    finalists=[n for n in validate if vt[n]['correct']>174]
    if not finalists:return nogo('No shortlisted method beats174 on validation299',pooled_train=pooled,validation=val)
    primary=sorted(finalists,key=lambda n:(-vt[n]['correct'],-pooled[n]['correct'],pooled[n]['damage']+vt[n]['damage'],PRIORITY[n]))[0]
    lock('configs/r10_FINAL_METHOD_LOCK.json',status='frozen_before_new_full_ARC',primary=primary,candidates=[primary],test_access_authorized=True,
        dataset_sha256=read('configs/r10_PROTOCOL.json')['dataset_sha256'],model_identity=read('configs/r10_PROTOCOL.json')['model_identity'],
        candidate_lock_sha256=sha('configs/r10_CANDIDATE_LOCK.json'),validation_lock_sha256=sha('configs/r10_VALIDATION_LOCK.json'),
        train=pooled,validation_summary=val,selection_rule='Predeclared validation first, train second, fewer total damages third, fixed priority last',
        history_exposed=True,external_evaluation_authorized=False,joint_only_baselines_selectable=False)
    status('awaiting_public_final_lock',primary=primary,final_method_lock_sha256=sha('configs/r10_FINAL_METHOD_LOCK.json'))
    # User approval is not needed. Desktop GitHub connector must publish the
    # already concrete lock, verify readback, and record commit/lock SHA here.
    while not (ROOT/'logs/r10_PUBLIC_FINAL_LOCK_RECEIPT.json').exists():time.sleep(20)
    public=read('logs/r10_PUBLIC_FINAL_LOCK_RECEIPT.json')
    assert public['status']=='verified_public' and public['final_method_lock_sha256']==sha('configs/r10_FINAL_METHOD_LOCK.json')
    assert public['repository']=='Gifoe/looped-transformer-stage1-reproduction' and len(public['commit'])==40
    run('arc_test','test',5200,budget_phase='arc_test');result=final_report()
    status('complete_ARC',primary=primary,correct=result['correct'],n=1172,external_evaluation_authorized=False)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed_requires_inspection',error=repr(error));raise
