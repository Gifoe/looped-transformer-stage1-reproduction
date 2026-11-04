"""Full conditional external evaluation, same frozen projection and GPU ledger."""
import importlib.metadata,subprocess
from slcd_core import ROOT,read,sha,save

LOCK='configs/slcd_R6_EXTERNAL_LOCK.json'

def freeze():
    f=read('configs/slcd_R6_FINAL_LOCK.json');data=read('configs/slcd_R6_EXTERNAL_DATA.json')
    arc=read('results/slcd_R6_ARC_COMPLETE.json');a=read('results/slcd_R6_TEST_ACCEPTANCE.json')
    assert arc['n']==a['n']==1172 and arc['correct']==a['correct']>=733 and a['status']=='passed'
    assert arc['final_lock_sha256']==a['final_lock_sha256']==sha('configs/slcd_R6_FINAL_LOCK.json')
    versions={k:importlib.metadata.version(k) for k in list(f['package_versions'])+['numpy','scipy']}
    assert all(versions[k]==v for k,v in f['package_versions'].items())
    lock=dict(status='frozen_before_external_model_inference',expected_final_sha=sha('configs/slcd_R6_FINAL_LOCK.json'),primary=f['primary'],
        arc_result_sha256=sha('results/slcd_R6_ARC_COMPLETE.json'),arc_acceptance_sha256=sha('results/slcd_R6_TEST_ACCEPTANCE.json'),
        ranking_source_sha256=f['ranking_source_sha256'],worker_source_sha256=sha('scripts/slcd_r6_external_worker.py'),package_versions=versions,
        inference_sources=f['inference_source_sha256'],tasks=data['tasks'],model_identity=f['model_identity'],prior_prefix=f['prior_prefix'],
        preparation_source_sha256=sha('scripts/slcd_r6_prepare_external.py'),external_data_sha256=sha('configs/slcd_R6_EXTERNAL_DATA.json'),
        controller_sha256=sha('scripts/slcd_r6_external_chain.py'),metric='acc_norm',num_fewshot=0,seed=2021,
        secondary='A1_g1p0; original fixed R1 definition, no selection on external outcomes',runtime_speed_claim_permitted=False)
    if (ROOT/LOCK).exists():assert read(LOCK)==lock,'Preserve existing external lock'
    else:save(LOCK,lock)

def guard():
    lock=read(LOCK)
    assert sha('scripts/slcd_r6_external_chain.py')==lock['controller_sha256']
    assert sha('configs/slcd_R6_FINAL_LOCK.json')==lock['expected_final_sha']
    assert sha('results/slcd_R6_ARC_COMPLETE.json')==lock['arc_result_sha256'] and sha('results/slcd_R6_TEST_ACCEPTANCE.json')==lock['arc_acceptance_sha256']
    assert sha('scripts/slcd_r6_external_worker.py')==lock['worker_source_sha256'] and sha('scripts/slcd_r6_scores.py')==lock['ranking_source_sha256']
    for name,h in lock['inference_sources'].items():assert sha('scripts/'+name)==h
    return lock

def main():
    freeze();guard();statuses={}
    for task,limit,reserve in [('hellaswag',14400,900),('openbookqa',1800,60)]:
        completion=f'results/slcd_r6_{task}_COMPLETE.json'
        if not (ROOT/completion).exists():
            budget=read('logs/slcd_gpu_budget.json');remaining=budget['limit_seconds']-budget['used_seconds']
            if task=='openbookqa' and remaining<=900:
                statuses[task]=dict(status='not_run',reason='remaining6h budget below conservative900s reserve',remaining_seconds=remaining);continue
            name='r6_'+task+'_joint';supervisor=ROOT/f'logs/slcd_{name}_SUPERVISOR.json'
            assert not supervisor.exists(),'Existing attempt must be inspected, not silently restarted'
            seconds=min(limit,int(remaining)-reserve-30);assert seconds>120
            subprocess.run([str(ROOT/'env/bin/python'),'-u','scripts/slcd_supervise.py','--name',name,'--seconds',str(seconds),'--','slcd_r6_external_worker.py','--task',task],cwd=ROOT,check=True)
        guard();r=read(completion);assert r['status']=='complete' and r['n']=={'hellaswag':10042,'openbookqa':500}[task]
        assert r['external_lock_sha256']==sha(LOCK) and r['score_file_sha256']==sha(r['score_file'])
        statuses[task]=dict(status='complete',n=r['n'],completion_sha256=sha(completion),score_file_sha256=r['score_file_sha256'])
    save('results/slcd_R6_EXTERNAL_CHAIN_COMPLETE.json',dict(status='complete',tasks=statuses,external_lock_sha256=sha(LOCK),method_unchanged=True))
    print('R6_EXTERNAL_CHAIN_COMPLETE',flush=True)

if __name__=='__main__':main()
