"""Finite R11 sequence. Stop failed hypotheses, await public final lock before ARC."""
import os,subprocess,time
from r11_core import ROOT,read,sha,save,extra_sources
from r11_evaluate import table,final_report


def status(stage,**values):save('logs/r11_CHAIN_STATUS.json',dict(status=stage,updated_unix=time.time(),coordinator_pid=os.getpid(),external_evaluation_authorized=False,**values))


def stage_lock(path,**values):
    assert not (ROOT/path).exists(),'Existing locks are immutable'
    protocol=read('configs/r11_PROTOCOL.json')
    candidate=read('configs/r11_CANDIDATE_LOCK.json');assert candidate['extra_source_sha256']==extra_sources() and candidate['inference_definition_sha256']==protocol['source_sha256']
    data=dict(created_unix=time.time(),protocol_sha256=sha('configs/r11_PROTOCOL.json'),definition_lock_sha256=sha('configs/r11_CANDIDATE_DEFINITION_LOCK.json'),
        inference_definition_sha256=protocol['source_sha256'],extra_source_sha256=extra_sources(),candidate_lock_sha256=sha('configs/r11_CANDIDATE_LOCK.json'))
    data.update(values);save(path,data)


def run(name,phase,seconds,budget_phase='development'):
    file=f'logs/r11_{name}_SUPERVISOR.json'
    if (ROOT/file).exists():
        assert read(file)['status']=='complete',('Inspect previous attempt, no auto retry',read(file));return
    status('running_'+phase,job=name)
    subprocess.run([str(ROOT/'env/bin/python'),'-B','-u','scripts/r11_supervise.py','--name',name,'--seconds',str(seconds),'--phase',budget_phase,'--','r11_worker.py','--phase',phase],cwd=ROOT,check=True)
    assert read(file)['status']=='complete'


def stop(reason,**values):save('results/r11_NO_GO.json',dict(status='No-Go',reason=reason,**values));status('No-Go',reason=reason)


def main():
    definition=read('configs/r11_CANDIDATE_DEFINITION_LOCK.json');names=read('configs/r11_CANDIDATE_LOCK.json')['candidates'];assert names==definition['candidates']
    dev=read('results/r11_DEVELOPMENT_COMPLETE.json');assert sha(dev['score_file'])==dev['score_file_sha256'];dt={r['method']:r for r in dev['summary']}
    run('parity','parity',180);table('parity');assert read('results/r11_parity_selected_COMPLETE.json')['max_parity_error']==0.
    run('confirmation','confirmation',2200);_,confirmation=table('confirmation');ct={r['method']:r for r in confirmation}
    pooled={n:{k:dt[n][k]+ct[n][k] for k in ['correct','rescue','damage','retained_R6_special','total_R6_special']} for n in names}
    advance=[]
    for n in names:
        pooled[n].update(prior_norm_correct=dt[n+'_prior_norm']['correct']+ct[n+'_prior_norm']['correct'],shuffle_correct=dt[n+'_shuffle_update']['correct']+ct[n+'_shuffle_update']['correct'])
        if pooled[n]['correct']>706 and pooled[n]['correct']>pooled[n]['prior_norm_correct'] and pooled[n]['correct']>pooled[n]['shuffle_correct']:advance.append(n)
    save('results/r11_TRAIN_COMPLETE.json',dict(status='complete',n=1093,R6_correct=705,summary=pooled,validation_candidates=advance,
        development_receipt_sha256=sha('results/r11_DEVELOPMENT_COMPLETE.json'),confirmation_receipt_sha256=sha('results/r11_confirmation_selected_COMPLETE.json')))
    if not advance:return stop('No frozen main beats706 and both own controls on fulltrain1093',pooled_train=pooled)
    stage_lock('configs/r11_VALIDATION_LOCK.json',status='frozen_before_current_validation',candidates=advance,pooled_train=pooled)
    run('validation','validation',1400);_,validation=table('validation');vt={r['method']:r for r in validation};eligible=[n for n in advance if vt[n]['correct']>174]
    if not eligible:return stop('No frozen main beats174 on validation299',pooled_train=pooled,validation=validation)
    ranked=sorted(eligible,key=lambda n:(-vt[n]['correct'],-pooled[n]['correct'],-(pooled[n]['retained_R6_special']+vt[n]['retained_R6_special']),pooled[n]['damage']+vt[n]['damage'],n!='single_loop_axis',n))
    primary=ranked[0]
    stage_lock('configs/r11_FINAL_METHOD_LOCK.json',status='frozen_before_new_full_ARC',primary=primary,candidates=[primary],test_access_authorized=True,
        train=pooled,validation_summary=validation,selection_rule=read('configs/r11_PROTOCOL.json')['validation']['gate'],
        candidate_lock_sha256=sha('configs/r11_CANDIDATE_LOCK.json'),validation_lock_sha256=sha('configs/r11_VALIDATION_LOCK.json'),
        external_evaluation_authorized=False,history_exposed=True)
    run('profile','profile',240);assert read('results/r11_DEPLOYMENT_PROFILE.json')['status']=='complete'
    status('awaiting_public_final_lock',primary=primary,final_method_lock_sha256=sha('configs/r11_FINAL_METHOD_LOCK.json'))
    while not (ROOT/'logs/r11_PUBLIC_FINAL_LOCK_RECEIPT.json').exists():time.sleep(20)
    public=read('logs/r11_PUBLIC_FINAL_LOCK_RECEIPT.json');assert public['status']=='verified_public' and public['final_method_lock_sha256']==sha('configs/r11_FINAL_METHOD_LOCK.json') and public['repository']=='Gifoe/looped-transformer-stage1-reproduction' and len(public['commit'])==40
    run('arc_test','test',5000,budget_phase='arc_test');result=final_report();status('complete_ARC',primary=primary,correct=result['correct'],n=1172)


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed_requires_inspection',error=repr(error));raise
