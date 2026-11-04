"""Independent CPU completion gate over complete results and archive bytes.

No winner selection, model forward or partial-test reading. Git readback and
visual inspection remain explicit release checks outside this CPU evidence.
"""
import ast,hashlib,json,math,tarfile
import numpy as np
from r7_core import ROOT,read,save,sha


def main():
    checks=[]
    def passed(name,**details):checks.append(dict(check=name,status='passed',**details))
    arc=read('results/r7_ARC_COMPLETE.json');assert arc['status']=='complete' and arc['n']==1172
    final=read('configs/r7_FINAL_METHOD_LOCK.json');candidate=read('configs/r7_CANDIDATE_LOCK.json')
    validation=read('configs/r7_VALIDATION_LOCK.json');primary=final['primary']
    assert arc['primary']==primary and final['test_access_authorized'] and primary in ['M1_multi_candidate','M2_N0_token','M3_multi_token']
    assert sha('configs/r7_FINAL_METHOD_LOCK.json')==arc['final_method_lock_sha256']
    for name,value in final['inference_source_sha256'].items():assert sha('scripts/'+name)==value
    assert final['inference_source_sha256']==candidate['inference_source_sha256']
    assert sha('scripts/r7_evaluate.py')==candidate['selection_source_sha256']
    assert sha('configs/r7_METHODS.json')==candidate['methods_sha256']==final['methods_sha256']
    assert sha('configs/r7_reference_manifest.json')==candidate['reference_manifest_sha256']==final['reference_manifest_sha256']
    assert len(read('configs/r7_METHODS.json')['methods'])==8
    assert len(validation['validation_candidates'])==2 and primary in validation['validation_candidates']
    assert candidate['created_unix']<read('logs/r7_confirmation_SUPERVISOR.json')['started_unix']
    assert validation['created_unix']<read('logs/r7_validation_SUPERVISOR.json')['started_unix']
    assert final['created_unix']<read('logs/r7_arc_SUPERVISOR.json')['started_unix']
    passed('frozen_sources_methods_references_stage_order',primary=primary,substantive_configurations=8)
    from r7_evaluate import rows
    allrows={split:rows(split) for split in ['discovery','selection','confirmation','validation','test']}
    expected={'discovery':384,'selection':192,'confirmation':517,'validation':299,'test':1172}
    for split,data in allrows.items():
        assert len(data)==expected[split]
        assert sha('data/slcd_'+split+'.jsonl')==final['dataset_sha256'][split]
        suffix='_development_COMPLETE.json' if split in ['discovery','selection','confirmation'] else '_final_COMPLETE.json'
        receipt=read('results/r7_'+split+suffix)
        assert receipt['baseline_parity_status']=='exact' and receipt['baseline_parity_questions']==len(data) and receipt['baseline_parity_max_error']==0.
        assert receipt['head_norm_before']==receipt['head_norm_after'] and not receipt['weights_modified']
        assert receipt['logical_main_depth']==5 and receipt['actual_reference_depth']==4 and not receipt['kv_cache_used']
        if split in ['discovery','selection','confirmation']:
            assert receipt['historical_main_state_collection_depth']==6 and receipt['actual_main_depth'] is None
            assert set(data[0]['method_scores'])==set(read('configs/r7_METHODS.json')['methods'])
        else:assert receipt['actual_main_depth']==5 and not receipt['historical_states_reused']
        for row in data:
            assert all(isinstance(n,int) and n>0 for n in row['choice_char_lengths'])
            for name,fallback in row['alignment_fallback_by_method'].items():
                if fallback:assert row['method_scores'][name]==row['baseline_scores']['M0_R6']
        passed('complete_split_'+split,n=len(data),fresh_main=split in ['validation','test'])
    train=[r for split in ['discovery','selection','confirmation'] for r in allrows[split]]
    assert sum(r['correct']['M0_R6'] for r in train)==705
    for name,n in {'native4':702,'fixed':717,'adaptive':716,'A1':726,'M0_R6':736}.items():assert sum(r['correct'][name] for r in allrows['test'])==n
    assert sum(r['correct'][primary] for r in allrows['test'])==arc['correct']
    acceptance=read('results/r7_TEST_ACCEPTANCE.json')
    assert acceptance['status']=='passed' and acceptance['n']==1172 and acceptance['max_R6_candidate_score_abs_error']==0.
    assert acceptance['N0_prior_sums_exact'] and acceptance['new_score_file_sha256']==arc['score_file_sha256']
    assert acceptance['final_method_lock_sha256']==sha('configs/r7_FINAL_METHOD_LOCK.json')
    passed('complete_test_counts_and_frozen_R6_per_question_equivalence',correct=arc['correct'])
    from r7_core import jsonlines
    cases=list(jsonlines('results/r7_failure_cases.jsonl'));expected_changes={r['id']:r for r in allrows['test'] if r['predictions'][primary]!=r['predictions']['M0_R6']}
    assert len(cases)==len(expected_changes) and {c['id'] for c in cases}==set(expected_changes)
    case_summary=read('results/r7_CASE_DECOMPOSITION.json');assert case_summary['n']==len(cases) and case_summary['full_cases_sha256']==sha('results/r7_failure_cases.jsonl')
    for case in cases:
        row=expected_changes[case['id']];assert case['R6_scores']==row['baseline_scores']['M0_R6'] and case['new_scores']==row['method_scores'][primary]
        assert case['gold']==row['gold'] and case['R6_prediction']==row['predictions']['M0_R6'] and case['new_prediction']==row['predictions'][primary]
    passed('all_changed_predictions_and_score_decompositions',cases=len(cases))
    # Recheck saved scalar projection diagnostics independently of target labels.
    positions=0
    for row in train:
        for diagnostics in row['token_diagnostics']:
            for name,d in diagnostics.items():
                a=np.asarray(d['alpha']);num=np.asarray(d['numerator']);den=np.asarray(d['denominator']);threshold=np.asarray(d['indistinguishable_threshold'])
                assert np.isfinite(a).all() and np.isfinite(num).all() and np.isfinite(den).all() and ((a>=0)&(a<=1)).all()
                if name=='C_R6alpha_token':assert np.array_equal(a,np.full_like(a,row['alpha6']))
                else:
                    predicted=np.clip(num/np.where(den<=threshold,1.,den),0.,1.);predicted=np.where(den<=threshold,0.,predicted)
                    assert np.allclose(a,predicted,rtol=1e-12,atol=1e-13)
                if name=='C_shuffled_direction':assert np.allclose(d['applied_direction_norm2'],d['projected_correction_norm2'],rtol=1e-11,atol=1e-12)
                if name=='C_matched_guidance':assert np.allclose(np.square(d['rho'])*d['guidance_norm2'],d['residual_norm2'],rtol=1e-11,atol=1e-12)
                positions+=len(a)
    passed('saved_projection_coefficients_and_matched_control_norms',mode_positions=positions,scope='Scalar diagnostics, not reconstruction of undistributed full-vocabulary logits')
    tree=ast.parse((ROOT/'scripts/r7_inference.py').read_text(encoding='utf8'))
    accesses={node.slice.value for node in ast.walk(tree) if isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name) and node.value.id=='row' and isinstance(node.slice,ast.Constant)}
    assert accesses=={'request_pairs','choice_char_lengths'}
    maths=read('results/r7_MATH_CHECKS.json');assert maths['status']=='passed' and len(maths['checks'])>=20
    for name,value in maths['source_sha256'].items():assert sha('scripts/'+name)==value
    pilot=read('results/r7_PILOT_COMPLETE.json');assert pilot['independent_depth_parity']['passed'] and max(pilot['baseline_max_sum_errors'].values())==0.
    assert not pilot['cache_parity']['passed'] and not pilot['cache_optimization_eligible']
    passed('metadata_exclusion_math_invariants_original_mask_and_depth',math_checks=len(maths['checks']),independent_depth_probe_questions=1,KV_cache_rejected=True)
    identity=final['model_identity'];assert identity['revision']=='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
    assert sha('models/Ouro-1.4B/model.safetensors')==identity['safetensors_sha256']
    for name,value in identity['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+name)==value
    assert sha('data/slcd_fixed_demonstrations.json')==final['fixed_demonstrations_sha256']
    passed('original_model_and_tokenizer_bytes_fixed25_unchanged')
    profile=read('results/r7_DEPLOYMENT_PROFILE.json');assert profile['status']=='complete' and profile['n']==12 and profile['primary']==primary
    assert not profile['diagnostic_hooks'] and not profile['historical_main_states'] and not profile['KV_cache']
    assert profile['actual_main_depth']==5 and profile['actual_reference_depth']==4
    assert len(profile['R6_records'])==12 and all(r['score_max_error']==0. for r in profile['records']+profile['R6_records'])
    assert profile['warm_mean_question_seconds']>0 and profile['R6_warm_mean_question_seconds']>0
    assert profile['real_model_invariants']['status']=='passed' and profile['real_model_invariants']['zero_correction_complete_sum_max_abs_error']==0.
    assert profile['real_model_invariants']['zero_correction_recovers_A1'] and profile['real_model_invariants']['max_continuation_tokens']>32
    passed('fresh_5_4_deployment_and_R6_timing',questions=12,time_ratio_vs_R6=profile['warm_time_ratio_vs_R6'])
    ledger=read('logs/r7_gpu_budget.json');assert ledger['limit_seconds']==21600 and ledger['reserved_arc_seconds']==5400
    assert abs(sum(r['gpu_worker_seconds'] for r in ledger['runs'])-ledger['used_seconds'])<1e-6 and ledger['used_seconds']<=21600
    charged=0.
    for run in ledger['runs']:
        if run['phase']=='development':assert run['limit_seconds']<=21600-charged-5400-15+1e-6
        charged+=run['gpu_worker_seconds']
    passed('independent_6h_budget_all_attempts_and_ARC_reserve',GPU_worker_seconds=ledger['used_seconds'])
    assert read('results/r7_PRESERVATION_AUDIT.json')['status']=='passed'
    if arc['correct']>736:
        rules=read('configs/r7_EXTERNAL_REFERENCE_RULES.json');external=read('configs/r7_EXTERNAL_LOCK.json')
        assert rules['created_unix']<final['created_unix'] and sha('configs/r7_EXTERNAL_REFERENCE_RULES.json')==final['external_reference_rules_sha256']==external['pretest_reference_rules_sha256']
        for task,n in [('hellaswag',10042),('openbookqa',500)]:
            receipt=read(f'results/r7_{task}_COMPLETE.json');assert receipt['status']=='complete' and receipt['n']==n and receipt['primary']==primary
            assert sha(receipt['score_file'])==receipt['score_file_sha256'] and external['tasks'][task]['num_fewshot']==0
            if primary=='M1_multi_candidate':assert len(set(receipt['independent_original_replay_ids']))==3
            passed('gated_complete_external_'+task,n=n,correct=receipt['correct'])
    else:
        assert not any((ROOT/f'logs/r7_{task}_SUPERVISOR.json').exists() for task in ['hellaswag','openbookqa'])
        passed('external_gate_closed_no_external_execution')
    required=['r7_README.md','r7_SELF_ITERATION_LOG.md','results/r7_REPORT_ZH.md','results/r7_baseline_parity.json',
        'results/r7_oracle_alpha_diagnostics.csv','configs/r7_reference_manifest.json','results/r7_development_results.csv',
        'results/r7_ablation_results.csv','configs/r7_FINAL_METHOD_LOCK.json','results/r7_arc_full_test_results.csv',
        'results/r7_paired_transitions.csv','results/r7_compute_costs.csv']
    assert all((ROOT/name).is_file() and (ROOT/name).stat().st_size>0 for name in required)
    index=read('artifacts/r7_RECORDS_INDEX.json');assert index['status']=='verified'
    assert sha(index['archive_path'])==index['archive_sha256'] and sha('artifacts/r7_RECORDS_CONTENTS.json')==index['contents_sha256']
    parts=hashlib.sha256()
    for part in index['parts']:
        raw=(ROOT/part['path']).read_bytes();assert len(raw)==part['bytes'] and hashlib.sha256(raw).hexdigest()==part['sha256'];parts.update(raw)
    assert parts.hexdigest()==index['archive_sha256']
    records=read('artifacts/r7_RECORDS_CONTENTS.json')['files']
    with tarfile.open(ROOT/index['archive_path'],'r:xz') as archive:
        assert archive.getnames()==[r['path'] for r in records]
        for entry in records:assert hashlib.sha256(archive.extractfile(entry['path']).read()).hexdigest()==entry['sha256']
    assert set(required)<={entry['path'] for entry in records}
    passed('all_named_deliverables_and_lossless_archive',files=len(records),archive_bytes=index['archive_bytes'])
    save('results/r7_COMPLETION_AUDIT.json',dict(status='passed_experiments_and_archive',primary=primary,arc_correct=arc['correct'],
        external_required=arc['correct']>736,checks=checks,scope='Completed experimental data, invariants, temporal/data gates, accounting and archive byte integrity',
        remaining_release_checks=['Human visual inspection of generated figures','Connector GitHub final commit/readback verification'],
        final_method_lock_sha256=sha('configs/r7_FINAL_METHOD_LOCK.json'),archive_index_sha256=sha('artifacts/r7_RECORDS_INDEX.json'),
        audit_source_sha256=sha('scripts/r7_completion_audit.py')))
    print('R7_COMPLETION_AUDIT_PASSED',len(checks),flush=True)


if __name__=='__main__':main()
