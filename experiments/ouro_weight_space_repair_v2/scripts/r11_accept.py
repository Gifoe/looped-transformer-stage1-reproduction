"""Completion remains unproven until complete records pass independent checks.

This is a data-integrity audit, not a declaration that the research hypothesis
or thread goal succeeded. It never changes the selected method or any scores.
"""
import argparse,csv,importlib.metadata,json,math,platform
import numpy as np
from r11_core import ROOT,read,sha,save,gate,extra_sources
from r11_evaluate import records
from r11_scores import ELIGIBLE


def exact_bootstrap_interval(n,rescue,damage):
    # Independent exact empirical paired-bootstrap distribution. Each sampled
    # row has delta -1,0,+1. No RNG implementation shared with the reported20k.
    p=np.asarray([damage/n,1-(rescue+damage)/n,rescue/n]);mass=np.array([1.])
    for _ in range(n):mass=np.convolve(mass,p)
    assert abs(float(mass.sum())-1)<1e-10
    cdf=mass.cumsum();indices=[int(np.searchsorted(cdf,q)) for q in [.025,.975]]
    return [(i-n)*100/n for i in indices]


def main(public_records=False):
    protocol,final,finalpath,names=gate('test');primary=final['primary'];assert names==[primary]
    arc=read('results/r11_ARC_COMPLETE.json');assert arc['status']=='complete' and arc['n']==1172 and arc['primary']==primary
    checks=[]
    def passed(name,**values):checks.append(dict(check=name,status='passed',**values))
    stages={p:records(p) for p in ['parity','confirmation','validation','test']}
    assert read('results/r11_parity_selected_COMPLETE.json')['max_parity_error']==0.
    passed('fresh_auxiliary_K4_all_rotations_and_method_scores_exact_vs_historical_authority',n=2)
    definition=read('configs/r11_CANDIDATE_DEFINITION_LOCK.json');development=read('results/r11_DEVELOPMENT_COMPLETE.json')
    assert definition['development_receipt_sha256']==sha('results/r11_DEVELOPMENT_COMPLETE.json') and sha(development['score_file'])==definition['development_score_sha256']==development['score_file_sha256']
    train=read('results/r11_TRAIN_COMPLETE.json');assert train['n']==1093 and train['R6_correct']==705
    dt={r['method']:r for r in development['summary']};conf=stages['confirmation'];validation=stages['validation']
    correct=lambda data,n:sum(r['correct'][n] for r in data)
    eligible=[]
    for n in definition['candidates']:
        total=dt[n]['correct']+correct(conf,n);assert total==train['summary'][n]['correct']
        prior=dt[n+'_prior_norm']['correct']+correct(conf,n+'_prior_norm');shuffle=dt[n+'_shuffle_update']['correct']+correct(conf,n+'_shuffle_update')
        if total>706 and total>prior and total>shuffle:eligible.append(n)
    assert eligible==read('configs/r11_VALIDATION_LOCK.json')['candidates']
    assert correct(validation,primary)>174 and correct(validation,primary)==max(correct(validation,n) for n in eligible)
    # For this run the validation winner must be unique; do not silently omit
    # tie-break verification if a future run reaches a tie.
    if sum(correct(validation,n)==correct(validation,primary) for n in eligible)>1:
        def rank(n):
            v=validation;t=train['summary'][n]
            retained=t['retained_R6_special']+sum(r['correct'][n] and r['correct']['M0_R6'] and not r['correct']['fixed'] for r in v)
            damage=t['damage']+sum(not r['correct'][n] and r['correct']['M0_R6'] for r in v)
            return (-correct(v,n),-t['correct'],-retained,damage,n!='single_loop_axis',n)
        assert min(eligible,key=rank)==primary
    passed('frozen_two_candidate_confirmation_training_and_validation_selection',primary=primary)
    data=stages['test'];assert {n for n in ELIGIBLE if n in data[0]['method_scores']}=={primary}
    assert arc['raw_score_file_sha256']==sha('results/r11_test_selected.jsonl') and arc['final_method_lock_sha256']==sha(finalpath)
    assert correct(data,'M0_R6')==736 and correct(data,primary)==arc['correct']
    comparisons={r['reference']:r for r in arc['comparisons']};stat_checks=[]
    assert len(comparisons)==len(arc['comparisons']) and set(comparisons)==set(data[0]['predictions'])-{primary}
    assert arc['metrics']==comparisons['M0_R6']
    for ref,reported in comparisons.items():
        rescue=sum(r['correct'][primary] and not r['correct'][ref] for r in data);damage=sum(not r['correct'][primary] and r['correct'][ref] for r in data);discord=rescue+damage
        assert reported['method']==primary and reported['n']==1172 and reported['correct']==correct(data,primary) and reported['reference_correct']==correct(data,ref)
        assert abs(reported['accuracy']-correct(data,primary)/1172)<1e-15 and abs(reported['delta_pp']-(rescue-damage)*100/1172)<1e-12
        assert (rescue,damage,rescue-damage)==(reported['rescue'],reported['damage'],reported['net'])
        exact_p=1. if discord==0 else min(1.,2*sum(math.comb(discord,k) for k in range(min(rescue,damage)+1))/(2**discord))
        assert abs(exact_p-reported['exact_mcnemar_p'])<=max(1e-14,abs(exact_p)*1e-10)
        assert reported['bootstrap_samples']==20000 and reported['bootstrap_seed']==2021
        exact_interval=exact_bootstrap_interval(1172,rescue,damage)
        empirical=[reported['paired_bootstrap_95_lower_pp'],reported['paired_bootstrap_95_upper_pp']]
        assert max(abs(x-y) for x,y in zip(exact_interval,empirical))<=2*100/1172+1e-12,('Bootstrap disagreement requires inspection',ref,exact_interval,empirical)
        stat_checks.append(dict(reference=ref,independent_exact_mcnemar_p=exact_p,exact_discrete_bootstrap_95_interval_pp=exact_interval,reported_20k_interval_pp=empirical))
    passed('full1172_selected_only_scores_outcomes_pairs_and_independent_statistics',comparisons=len(stat_checks))
    def verify_csv(path,expected):
        with (ROOT/path).open(newline='',encoding='utf8') as stream:
            reader=csv.DictReader(stream);actual=list(reader)
            assert set(reader.fieldnames)==set(expected[0]),('CSV schema mismatch',path)
        assert actual==[{k:str(v) for k,v in row.items()} for row in expected],('CSV differs from complete raw outcomes',path)
    expected_summary=[]
    for name in data[0]['predictions']:
        expected_summary.append(dict(method=name,n=1172,correct=correct(data,name),R6_correct=736,net=correct(data,name)-736,
            rescue=sum(r['correct'][name] and not r['correct']['M0_R6'] for r in data),damage=sum(not r['correct'][name] and r['correct']['M0_R6'] for r in data),
            retained_R6_special=sum(r['correct'][name] and r['correct']['M0_R6'] and not r['correct']['fixed'] for r in data),
            total_R6_special=sum(r['correct']['M0_R6'] and not r['correct']['fixed'] for r in data)))
    assert arc['retention']==next(row for row in expected_summary if row['method']==primary)
    verify_csv('results/r11_arc_full_test_results.csv',expected_summary)
    verify_csv('results/r11_arc_paired_statistics.csv',arc['comparisons'])
    expected_transitions=[dict(id=r['id'],gold=r['gold'],R6_prediction=r['predictions']['M0_R6'],new_prediction=r['predictions'][primary],
        R6_correct=r['correct']['M0_R6'],new_correct=r['correct'][primary],Fixed_correct=r['correct']['fixed'],
        R6_scores=json.dumps(r['baseline_scores']['M0_R6']),new_scores=json.dumps(r['method_scores'][primary])) for r in data]
    verify_csv('results/r11_paired_transitions.csv',expected_transitions)
    passed('primary_metrics_retention_and_three_final_CSVs_rebuilt_from_complete_raw_outcomes',transition_rows=len(data))
    special=[r for r in data if r['correct']['M0_R6'] and not r['correct']['fixed']]
    assert len(special)==35 and sum(r['correct'][primary] for r in special)==arc['retention']['retained_R6_special']
    max_norm_error=0.
    for r in data:
        a=np.asarray(r['baseline_scores']['A1']);delta=np.asarray(r['method_scores'][primary])-a;size=float(np.linalg.norm(delta-delta.mean()))
        for c in [primary+'_prior_norm',primary+'_shuffle_update']:
            d=np.asarray(r['method_scores'][c])-a;err=abs(float(np.linalg.norm(d-d.mean()))-size);max_norm_error=max(max_norm_error,err);assert err<1e-12*max(1.,size)
    passed('R6_unique_rescue_retention_and_actual_centered_correction_norm_controls',R6_unique=35,retained=arc['retention']['retained_R6_special'],max_norm_error=max_norm_error)
    profile=read('results/r11_DEPLOYMENT_PROFILE.json');assert profile['status']=='complete' and profile['n']==12 and profile['primary']==primary and profile['final_method_lock_sha256']==sha(finalpath)
    assert (profile['actual_main_depth'],profile['actual_prior_depth'],profile['actual_auxiliary_depth'])==(5,4,4)
    assert not any(profile[k] for k in ['KV_cache','attention_mask_argument','historical_states','cached_main_scores','diagnostic_hooks'])
    assert all(r['score_max_error']==0 for r in profile['records']+profile['R6_records'])
    mean=float(np.mean([r['seconds'] for r in profile['records'][1:]]));base=float(np.mean([r['seconds'] for r in profile['R6_records'][1:]]))
    assert abs(mean-profile['warm_mean_question_seconds'])<1e-12 and abs(base-profile['R6_warm_mean_question_seconds'])<1e-12 and abs(mean/base-profile['time_ratio'])<1e-12
    passed('actual_uncached_fresh_end_to_end_profile_and_same12_R6_comparator',warm11_seconds=mean,time_ratio=mean/base)
    book=definition['budget'];inherited=0.
    for k,h in book['inherited_ledger_sha256'].items():
        path='logs/'+k+'_gpu_budget.json';assert sha(path)==h;inherited+=read(path)['used_seconds']
    assert inherited==book['inherited_total_seconds']
    ledger=read('logs/r11_gpu_budget.json');assert abs(sum(r['gpu_worker_seconds'] for r in ledger['runs'])-ledger['used_seconds'])<1e-6
    assert inherited+ledger['used_seconds']<=21600 and ledger['remaining_round_limit_seconds']==book['remaining_round_limit_seconds']
    charged=0.;success_test=[]
    for r in ledger['runs']:
        assert r['status'] in ['complete','failed'] and r['gpu_worker_seconds']>=0
        assert r['worker_script_sha256']==sha('scripts/r11_worker.py')
        reserve=5400 if r['phase']=='development' else 30
        assert r['limit_seconds']<=book['remaining_round_limit_seconds']-charged-reserve-15+1e-6
        assert read('logs/'+r['name']+'_SUPERVISOR.json')==r
        if r['phase']=='arc_test' and r['status']=='complete':success_test.append(r)
        charged+=r['gpu_worker_seconds']
    assert success_test and final['created_unix']<min(r['started_unix'] for r in success_test)
    assert read('logs/r11_profile_SUPERVISOR.json')['status']=='complete'
    assert read('logs/r11_profile_SUPERVISOR.json')['finished_unix']<=min(r['started_unix'] for r in success_test)
    public=read('logs/r11_PUBLIC_FINAL_LOCK_RECEIPT.json');assert public['readback_verified_before_test'] and public['final_method_lock_sha256']==sha(finalpath)
    passed('original_6h_all_attempts_inheritance_and_ARC_reserve_no_reset',GPU_worker_seconds=inherited+ledger['used_seconds'],R11_seconds=ledger['used_seconds'],public_final_commit=public['commit'])
    for phase in ['parity','confirmation','validation','test']:
        r=read(f'results/r11_{phase}_selected_COMPLETE.json');assert r['head_norm_before']==r['head_norm_after'] and not r['weights_modified'] and r['actual_auxiliary_depth']==4 and not r['KV_cache'] and not r['attention_mask_argument']
    if not public_records:
        identity=protocol['model_identity'];assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
        for name,h in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+name)==h
        assert sha('data/slcd_fixed_demonstrations.json')==protocol['fixed25_sha256']
    passed('model_freeze_and_tokenizer_demo_identity',live_model_file_check=not public_records)
    inherited_files=read('artifacts/slcd_R6_RECORDS_CONTENTS.json')['files']
    for r in inherited_files:assert sha(r['path'])==r['sha256'],('Inherited R6 evidence changed',r['path'])
    assert read('results/r7_PRESERVATION_AUDIT.json')['status']=='passed'
    passed('inherited_R6_records_preserved_and_R7_preservation_audit',R6_files=len(inherited_files))
    assert not final['external_evaluation_authorized'] and not read('logs/r11_CHAIN_STATUS.json')['external_evaluation_authorized']
    versions={}
    for n in ['torch','transformers','tokenizers','numpy','scipy','safetensors','lm_eval']:
        try:versions[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:versions[n]='not_installed_in_audit_environment'
    result=dict(status='passed_records_only' if public_records else 'passed_original_server_experiment_integrity',goal_success_declared=False,n=1172,primary=primary,checks=checks,independent_statistics=stat_checks,
        extra_source_sha256=extra_sources(),audit_source_sha256=sha('scripts/r11_accept.py'),full_model_file_check=not public_records,
        environment=dict(python=platform.python_version(),platform=platform.platform(),packages=versions),
        scientific_flags=dict(exceeds_R6=arc['correct']>736,reaches_751=arc['correct']>=751,beats_own_norm_point_estimate=arc['correct']>comparisons[primary+'_prior_norm']['reference_correct'],
            semantic_interpretation_proven=False,history_exposed=True,external_tasks_stopped=True),
        remaining_release_checks=['Render/inspect final figures','Verify lossless public archive and GitHub readback'],scope='Complete experimental evidence and accounting; a passing audit is not proof the method is effective or novel')
    save('results/r11_PUBLIC_RECORDS_ACCEPTANCE.json' if public_records else 'results/r11_EXPERIMENT_ACCEPTANCE.json',result);print(json.dumps(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public-records',action='store_true');a=p.parse_args();main(a.public_records)
