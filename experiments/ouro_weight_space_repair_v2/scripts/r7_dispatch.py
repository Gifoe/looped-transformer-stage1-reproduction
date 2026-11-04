"""Detached CPU coordinator. Each GPU worker has its own bounded supervisor.

No auto-retries after failures/timeouts. Locks precede held-out data reads.
This coordinator does not import torch or consume the GPU itself.
"""
import fcntl,json,os,subprocess,time
from r7_core import ROOT,read,save,sha,BASE_COMMIT


def status(stage,**kwargs):
    value=dict(status='running',stage=stage,pid=os.getpid(),updated_unix=time.time(),**kwargs)
    save('logs/r7_PIPELINE_STATUS.json',value);print(json.dumps(value),flush=True)


def run_stage(split,job,name,seconds,methods='all'):
    receipt=f'results/r7_{split}_{job}_COMPLETE.json';sp=f'logs/r7_{name}_SUPERVISOR.json'
    if not (ROOT/sp).exists():
        command=[str(ROOT/'env/bin/python'),'-B','-u',str(ROOT/'scripts/r7_supervise.py'),'--name',name,'--seconds',str(seconds),
            '--phase','arc_test' if split=='test' else 'development','--','r7_worker.py','--split',split,'--job',job,'--methods',methods]
        code=subprocess.call(command,cwd=ROOT);assert code==0,('Supervised stage failed',name,code)
    else:
        while read(sp)['status'] in ['starting','running']:
            state=read(sp)
            try:os.kill(state['supervisor_pid'],0)
            except ProcessLookupError:raise RuntimeError('Supervisor disappeared without completion; inspect before resuming')
            time.sleep(5)
        assert read(sp)['status']=='complete',('Existing supervised attempt failed',read(sp))
    done=read(receipt);assert done['status']=='complete' and sha(done['score_file'])==done['score_file_sha256']
    return done


def main():
    with (ROOT/'logs/r7_pipeline.lock').open('a') as mutex:
        fcntl.flock(mutex,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:work()
        except Exception as error:
            save('logs/r7_PIPELINE_STATUS.json',dict(status='failed',error=repr(error),pid=os.getpid(),updated_unix=time.time()))
            raise


def work():
    from r7_worker import SOURCES
    from r7_evaluate import development,train_rank,validation_rank,rows,paired,arc_report
    from r7_core import csvout
    for split,seconds in [('discovery',4000),('selection',2400)]:
        status(split);run_stage(split,'development',split,seconds)
    status('development_analysis');bysplit,table=development(['discovery','selection'])
    # No formula/threshold search. The required eight configurations, numerical
    # constants and exact selection program are frozen together at this gate.
    candidate_path='configs/r7_CANDIDATE_LOCK.json'
    if not (ROOT/candidate_path).exists():
        lock=dict(status='frozen_before_confirmation',created_unix=time.time(),base_commit=BASE_COMMIT,
            methods_sha256=sha('configs/r7_METHODS.json'),reference_manifest_sha256=sha('configs/r7_reference_manifest.json'),
            inference_source_sha256={n:sha('scripts/'+n) for n in SOURCES},selection_source_sha256=sha('scripts/r7_evaluate.py'),
            dataset_sha256={s:sha('data/slcd_'+s+'.jsonl') for s in ['discovery','selection','confirmation','validation','test']},
            baseline_reference_sha256={s:sha('results/slcd_original_'+s+'.jsonl') for s in ['validation','test']},
            fixed_demonstrations_sha256=sha('data/slcd_fixed_demonstrations.json'),
            model_identity=read('configs/slcd_R6_FINAL_LOCK.json')['model_identity'],
            selection=read('configs/r7_METHODS.json')['selection'],development_results_before_confirmation_sha256=sha('results/r7_development_results.csv'),
            confirmation_previously_exposed_in_history=True,test_previously_exposed_in_history=True,
            train1093_R6_historical_correct=705,formula_revision_rounds=0,substantive_configurations=8)
        save(candidate_path,lock)
    assert read(candidate_path)['selection_source_sha256']==sha('scripts/r7_evaluate.py')
    status('confirmation',candidate_lock_sha256=sha(candidate_path));run_stage('confirmation','development','confirmation',6000)
    bysplit,table=development(['discovery','selection','confirmation']);train=[r for v in bysplit.values() for r in v]
    assert len(train)==1093 and sum(r['correct']['M0_R6'] for r in train)==705
    candidates=train_rank(bysplit)[:2]
    vpath='configs/r7_VALIDATION_LOCK.json'
    if not (ROOT/vpath).exists():save(vpath,dict(status='frozen_before_validation',created_unix=time.time(),
        candidate_lock_sha256=sha(candidate_path),validation_candidates=candidates,train_ranking=train_rank(bysplit),
        train_results_sha256=sha('results/r7_development_results.csv'),selection_source_sha256=sha('scripts/r7_evaluate.py')))
    assert read(vpath)['validation_candidates']==candidates
    status('validation',validation_candidates=candidates);run_stage('validation','final','validation',3500,','.join(candidates))
    val=rows('validation');csvout('results/r7_validation_results.csv',[paired(val,name) for name in val[0]['predictions']])
    primary=validation_rank(val,train,candidates)[0]
    final_path='configs/r7_FINAL_METHOD_LOCK.json'
    if not (ROOT/final_path).exists():
        frozen=read(candidate_path).copy();frozen.update(status='frozen_before_R7_test',created_unix=time.time(),
            primary=primary,test_access_authorized=True,candidate_lock_sha256=sha(candidate_path),
            external_reference_rules_sha256=sha('configs/r7_EXTERNAL_REFERENCE_RULES.json'),
            pretest_github_commit=read('configs/r7_GITHUB_PREREGISTRATION.json')['commit'],
            validation_lock_sha256=sha(vpath),validation_results_sha256=sha('results/r7_validation_results.csv'),
            validation_ranking=validation_rank(val,train,candidates),
            primary_definition=read('configs/r7_METHODS.json')['methods'][primary],
            numerical_constants=dict(projection_statistic_dtype='float64',logit_dtype='float32',
                indistinguishable_epsilon_multiplier=64,alpha_bounds=[0,1],prediction_block=32,shuffle_seed=2021),
            attention='Exact original no-mask no-KV uncached model.model call; failed KV optimization rejected',
            fallback='Any own-reference prefix alignment failure falls back whole question to frozen R6; all other errors stop',
            primary_comparison='R6',minimum_correct_to_run_external=737,aspirational_correct=751,
            no_post_test_reselection=True,blind_test=False)
        save(final_path,frozen)
    assert read(final_path)['primary']==primary
    status('arc_test',primary=primary,final_lock_sha256=sha(final_path));run_stage('test','final','arc',20000,primary)
    metrics=arc_report()
    state=dict(status='arc_complete_external_gate_open' if metrics['correct']>736 else 'arc_complete_external_gate_closed',
        stage='arc_complete',primary=primary,arc_correct=metrics['correct'],arc_n=1172,updated_unix=time.time(),
        final_lock_sha256=sha(final_path),no_post_test_reselection=True)
    save('logs/r7_PIPELINE_STATUS.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':main()
