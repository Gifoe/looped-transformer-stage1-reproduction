"""Execute one frozen R6 test; compensate transitive source-binding checks."""
import csv,importlib.metadata,json,subprocess,time,os
from slcd_core import ROOT,read,sha,save

FINAL='configs/slcd_R6_FINAL_LOCK.json'
EXPECTED_FINAL_SHA='6de605b885b4fb9ff72b273ecc0c0484e9fe72fc5d51d793b22367db879b7da7'

def guard():
    assert sha(FINAL)==EXPECTED_FINAL_SHA
    f=read(FINAL);assert f['primary']=='r6_prior_projection' and f['test_access_authorized']
    assert sha('configs/slcd_R6_DEVELOPMENT_LOCK.json')==f['development_lock_sha256']
    d=read('configs/slcd_R6_DEVELOPMENT_LOCK.json')
    assert sha('scripts/slcd_r5_evaluate.py')==d['imported_prior_evaluator_sha256']
    assert sha('configs/slcd_R5_DEVELOPMENT_LOCK.json')==d['parent_prior_development_lock_sha256']
    assert sha('scripts/slcd_r6_scores.py')==d['ranking_source_sha256']==f['ranking_source_sha256']
    assert sha('configs/slcd_R6_METHODS.json')==d['methods_config_sha256']==f['methods_config_sha256']
    assert sha('scripts/slcd_r6_evaluate.py')==d['evaluation_source_sha256']==f['evaluation_source_sha256']
    assert sha('scripts/slcd_r2_evaluate.py')==d['imported_evaluation_source_sha256']==f['imported_evaluation_source_sha256']
    for name,h in f['inference_source_sha256'].items():assert sha('scripts/'+name)==h
    actual={n:importlib.metadata.version(n) for n in f['package_versions']};assert actual==f['package_versions']
    assert sha('data/slcd_test.jsonl')==f['data_sha256']['test']
    assert sha('results/slcd_original_test.jsonl')==f['original_test_scores_sha256']
    assert sha('results/slcd_BASELINE_REUSE_AUDIT.json')==f['baseline_reuse_audit_sha256']
    return f

def main():
    f=guard();before=sha(FINAL)
    completion=ROOT/'results/slcd_r6_priors_test_COMPLETE.json'
    supervisor=ROOT/'logs/slcd_r6_prior_test_SUPERVISOR.json'
    if not completion.exists():
        if supervisor.exists():
            state=read(str(supervisor.relative_to(ROOT)))
            assert state['status'] in ['starting','running'],'Preserve failed attempt; diagnose before a uniquely named retry'
            started=time.time()
            while state['status'] in ['starting','running']:
                assert time.time()-started<2100
                pid=state.get('worker_pid',state['supervisor_pid'])
                assert os.path.exists('/proc/'+str(pid)),'Stale state requires diagnosis, not restart'
                time.sleep(5);state=read(str(supervisor.relative_to(ROOT)))
            assert state['status']=='complete' and completion.exists()
        else:
            subprocess.run([str(ROOT/'env/bin/python'),'-u','scripts/slcd_supervise.py','--name','r6_prior_test','--seconds','1800','--','slcd_r6_prior_worker.py','--split','test'],cwd=ROOT,check=True)
    guard();receipt=read('results/slcd_r6_priors_test_COMPLETE.json')
    assert receipt['status']=='complete' and receipt['n']==1172 and receipt['provenance']['authorization_lock_sha256']==before
    assert receipt['provenance']['package_versions']==f['package_versions']
    subprocess.run([str(ROOT/'env/bin/python'),'-u','scripts/slcd_r6_evaluate.py','test'],cwd=ROOT,check=True)
    guard();result=read('results/slcd_R6_ARC_COMPLETE.json');assert result['n']==1172 and result['final_lock_sha256']==before and result['primary']==f['primary']
    with (ROOT/'results/slcd_r6_paired_transitions.csv').open(newline='') as h:trans=list(csv.DictReader(h))
    primary=[r for r in trans if r['method']==f['primary']];assert len(primary)==len({r['id'] for r in primary})==1172
    correct=sum(int(r['prediction'])==int(r['gold']) for r in primary);assert correct==result['correct']
    for baseline,column in [('fixed','fixed_prediction'),('incumbent_A1','incumbent_prediction')]:
        rescues=sum(int(r['prediction'])==int(r['gold']) and int(r[column])!=int(r['gold']) for r in primary)
        damage=sum(int(r['prediction'])!=int(r['gold']) and int(r[column])==int(r['gold']) for r in primary)
        summary=next(r for r in result['results'] if r['method']==f['primary'] and r['baseline']==baseline)
        assert (rescues,damage)==(summary['corrections'],summary['regressions'])
    save('results/slcd_R6_TEST_ACCEPTANCE.json',dict(status='passed',n=1172,primary=f['primary'],correct=correct,
        final_lock_sha256=before,transitive_source_guards_before_and_after=True,paired_csv_counts_match=True,
        controller_sha256=sha('scripts/slcd_r6_final_chain.py'),result_sha256=sha('results/slcd_R6_ARC_COMPLETE.json'),
        prior_receipt_sha256=sha('results/slcd_r6_priors_test_COMPLETE.json'),target_reached=result['target_reached']))
    print('R6_FROZEN_TEST_ACCEPTED',correct,'/1172',flush=True)

if __name__=='__main__':main()
