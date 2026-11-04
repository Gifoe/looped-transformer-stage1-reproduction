"""Fixed prior-corrected scores. Known calibration family; no novelty assertion."""
import argparse,json,time
import numpy as np
from slcd_core import ROOT,read,rows,sha,save,writecsv
from slcd_r2_evaluate import stats,predictions

from slcd_r6_scores import METHOD_IDS, selected_score, all_scores
from slcd_r5_evaluate import load_prior as load_r5_prior
METHODS=read('configs/slcd_R6_METHODS.json')['methods']
assert set(METHODS)==set(METHOD_IDS)

def freeze_development():
    assert not (ROOT/'configs/slcd_R6_DEVELOPMENT_LOCK.json').exists()
    old=read('configs/slcd_R2_DEVELOPMENT_LOCK.json');from slcd_r6_prior_worker import SOURCES,PREFIX
    original_sha={};cohort={}
    for split in ['discovery','selection','confirmation']:
        path=f'results/slcd_r2_{split}_development.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'));assert receipt['status']=='complete' and sha(path)==receipt['score_file_sha256'];original_sha[path]=sha(path)
        for r in rows('data/slcd_'+split+'.jsonl'):cohort[r['id']]=split
    prior_path='results/slcd_r5_priors_train.jsonl';prior_receipt=read(prior_path.replace('.jsonl','_COMPLETE.json'))
    assert prior_receipt['status']=='complete' and prior_receipt['n']==1093 and sha(prior_path)==prior_receipt['score_file_sha256']
    parent=read('configs/slcd_R5_DEVELOPMENT_LOCK.json')
    assert sha('scripts/slcd_r5_evaluate.py')==parent['evaluation_source_sha256'] and prior_receipt['provenance']['authorization_lock_sha256']==sha('configs/slcd_R5_DEVELOPMENT_LOCK.json')
    lock=dict(status='frozen_before_R6_feedback',created_unix=time.time(),methods=METHODS,prior_prefix=PREFIX,model_identity=old['model_identity'],data_sha256=old['train_data_sha256'],fold_by_id=old['fold_by_id'],cohort_by_id=cohort,
        original_score_sha256=original_sha,inference_source_sha256={n:sha('scripts/'+n) for n in SOURCES},evaluation_source_sha256=sha('scripts/slcd_r6_evaluate.py'),imported_evaluation_source_sha256=sha('scripts/slcd_r2_evaluate.py'),
        minimum_train_net_for_test=7,selection_rule=['Eligible only if train1093 net>=7 vs A1 and at least3/5 fixed reused-train folds nonnegative','Rank total correct descending','then worst proportional fold gain descending','then number of nonnegative folds descending','then ID lexicographically'],
        cache_policy='Raw LL only, key exact Answer: prefix+whitespace-preserved continuation+prefix/target token IDs+model/tokenizer/inference source identity. No gold, occurrence-frequency or dataset statistics. No pre-enumeration of test answers.',
        interpretation='Partial correction of early-guidance domain prior. Projection is over centered answer-choice scores and depends on the choice set; matched control matches only this local Euclidean guidance norm, not KL or Fisher. Two fixed coefficients .25/.5 are openly compared on train. No new probability model or originality claim.',
        methods_config_sha256=sha('configs/slcd_R6_METHODS.json'),ranking_source_sha256=sha('scripts/slcd_r6_scores.py'),imported_prior_evaluator_sha256=sha('scripts/slcd_r5_evaluate.py'),
        train_prior_sha256=sha(prior_path),parent_prior_development_lock_sha256=sha('configs/slcd_R5_DEVELOPMENT_LOCK.json'),package_versions=read('results/slcd_SOURCE_AUDIT.json')['package_versions'],
        target_correct=733,previous_test_known='A1=726 and Fixed717. R2,R3,R4,R5 all rejected without new test; no validation reselection in follow-up.',fold_scope='Previously reused train robustness slices, not blind CV')
    save('configs/slcd_R6_DEVELOPMENT_LOCK.json',lock);print('R6_DEVELOPMENT_FROZEN',sha('configs/slcd_R6_DEVELOPMENT_LOCK.json'))

def load_prior(split,lock_path):
    lock=read(lock_path);path=f'results/slcd_r6_priors_{split}.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'));assert receipt['status']=='complete' and sha(path)==receipt['score_file_sha256'] and receipt['provenance']['authorization_lock_sha256']==sha(lock_path)
    assert receipt['provenance']['inference_source_sha256']==lock['inference_source_sha256'] and receipt['provenance']['data_sha256']==lock['data_sha256']
    data=rows(path);n=1093 if split=='train' else 1172;assert len(data)==receipt['n']==n and len({r['id'] for r in data})==n
    for r in data:assert r['signature']==receipt['signature']
    return {r['id']:r for r in data},receipt

def summarize(data,names,training=False):
    summary=[];folds=[];trans=[]
    for name in names:
        for subset in (['all','discovery','selection','confirmation'] if training else ['all']):
            rs=[r for r in data if subset=='all' or r['split']==subset]
            for base in ['fixed','incumbent_A1']:
                summary.append(dict(method=name,subset=subset,baseline=base,**stats([r['predictions'][name]==r['gold'] for r in rs],[r['predictions'][base]==r['gold'] for r in rs])))
        if training:
            for fold in range(5):
                rs=[r for r in data if r['fold']==fold];folds.append(dict(method=name,fold=fold,**stats([r['predictions'][name]==r['gold'] for r in rs],[r['predictions']['incumbent_A1']==r['gold'] for r in rs])))
        for r in data:trans.append(dict(method=name,id=r['id'],gold=r['gold'],prediction=r['predictions'][name],fixed_prediction=r['predictions']['fixed'],incumbent_prediction=r['predictions']['incumbent_A1'],correct=r['predictions'][name]==r['gold']))
    return summary,folds,trans

def train():
    lp='configs/slcd_R6_DEVELOPMENT_LOCK.json';lock=read(lp);assert lock['methods']==METHODS and sha('scripts/slcd_r6_evaluate.py')==lock['evaluation_source_sha256'] and sha('scripts/slcd_r2_evaluate.py')==lock['imported_evaluation_source_sha256']
    assert sha('configs/slcd_R6_METHODS.json')==lock['methods_config_sha256'] and sha('scripts/slcd_r6_scores.py')==lock['ranking_source_sha256'] and sha('scripts/slcd_r5_evaluate.py')==lock['imported_prior_evaluator_sha256']
    assert sha('configs/slcd_R5_DEVELOPMENT_LOCK.json')==lock['parent_prior_development_lock_sha256'] and sha('results/slcd_r5_priors_train.jsonl')==lock['train_prior_sha256']
    priors,_=load_r5_prior('train','configs/slcd_R5_DEVELOPMENT_LOCK.json');originals={}
    for path,h in lock['original_score_sha256'].items():
        assert sha(path)==h
        for row in rows(path):assert row['id'] not in originals;originals[row['id']]=row
    assert set(originals)==set(priors)==set(lock['fold_by_id'])
    data=[]
    for qid,row in originals.items():
        p=priors[qid];assert row['gold']==p['gold'] and row['prompt_sha256']==p['prompt_sha256'] and row['choice_char_lengths']==p['choice_char_lengths'] and row['choice_token_counts']==p['choice_token_counts']
        ps=predictions(row);ps.update({mid:int(x.argmax()) for mid,x in all_scores(row['original_methods'],p['prior_methods']).items()})
        data.append(dict(id=qid,gold=row['gold'],predictions=ps,fold=lock['fold_by_id'][qid],split=lock['cohort_by_id'][qid]))
    res,folds,trans=summarize(data,['native4','fixed','adaptive','incumbent_A1']+list(METHODS),True);by={r['method']:r for r in res if r['subset']=='all' and r['baseline']=='incumbent_A1'}
    def order(mid):
        fs=[r for r in folds if r['method']==mid]
        return (-by[mid]['correct'],-min(r['net_correct']/r['n'] for r in fs),-sum(r['net_correct']>=0 for r in fs),mid)
    eligible=[mid for mid in METHODS if by[mid]['net_correct']>=lock['minimum_train_net_for_test'] and sum(r['net_correct']>=0 for r in folds if r['method']==mid)>=3]
    ranking=sorted(METHODS,key=order);selected=sorted(eligible,key=order)[0] if eligible else None
    save('results/slcd_R6_DEVELOPMENT_COMPLETE.json',dict(status='complete',selected=selected,ranking=ranking,eligible=eligible,summary=res,folds=folds,development_lock_sha256=sha(lp)))
    writecsv('results/slcd_r6_development_results.csv',res);writecsv('results/slcd_r6_fold_results.csv',folds);writecsv('results/slcd_r6_train_predictions.csv',trans)
    print(json.dumps(dict(selected=selected,ranking=[by[mid] for mid in ranking]),indent=2))

def freeze_test():
    lp='configs/slcd_R6_DEVELOPMENT_LOCK.json';old=read(lp);result=read('results/slcd_R6_DEVELOPMENT_COMPLETE.json');mid=result['selected'];assert mid is not None and mid in result['eligible'] and result['development_lock_sha256']==sha(lp)
    assert sha('scripts/slcd_r6_evaluate.py')==old['evaluation_source_sha256'] and sha('scripts/slcd_r2_evaluate.py')==old['imported_evaluation_source_sha256']
    assert sha('configs/slcd_R6_METHODS.json')==old['methods_config_sha256'] and sha('scripts/slcd_r6_scores.py')==old['ranking_source_sha256'] and sha('scripts/slcd_r5_evaluate.py')==old['imported_prior_evaluator_sha256']
    for n,h in old['inference_source_sha256'].items():assert sha('scripts/'+n)==h
    by=next(r for r in result['summary'] if r['method']==mid and r['subset']=='all' and r['baseline']=='incumbent_A1');assert by['net_correct']>=7 and sum(r['net_correct']>=0 for r in result['folds'] if r['method']==mid)>=3
    assert not (ROOT/'configs/slcd_R6_FINAL_LOCK.json').exists();first=read('FINAL_METHOD_LOCK.json')
    lock=dict(status='frozen_before_R6_test',created_unix=time.time(),primary=mid,formula=METHODS[mid],methods={mid:METHODS[mid]},test_access_authorized=True,prior_prefix=old['prior_prefix'],model_identity=old['model_identity'],
        data_sha256=first['dataset_sha256'],inference_source_sha256=old['inference_source_sha256'],evaluation_source_sha256=old['evaluation_source_sha256'],imported_evaluation_source_sha256=old['imported_evaluation_source_sha256'],
        fixed_demonstrations_sha256=first['fixed_demonstrations_sha256'],prompt_protocol=first['prompt_protocol'],seed=2021,target_correct=733,
        development_lock_sha256=sha(lp),development_results_sha256=sha('results/slcd_R6_DEVELOPMENT_COMPLETE.json'),selection_rule=old['selection_rule'],cache_policy=old['cache_policy'],
        ranking_source_sha256=old['ranking_source_sha256'],methods_config_sha256=old['methods_config_sha256'],package_versions=old['package_versions'],
        logical_depths=dict(context=5,prior=4),vocabulary_head_depths=dict(context=[1,4,5],prior=[1,4]),actual_collected_depths=dict(reused_context=6,new_prior=6),
        coefficients=dict(late_score_gamma=1.,fixed_guidance_omega=.5,prior_correction=METHODS[mid]),first_round_lock_sha256=sha('FINAL_METHOD_LOCK.json'),data_source_notice_sha256=sha('data/DATA_SOURCE_NOTICE.txt'),
        original_test_scores_sha256=sha('results/slcd_original_test.jsonl'),baseline_reuse_audit_sha256=sha('results/slcd_BASELINE_REUSE_AUDIT.json'),test_is_blind=False,previous_test_known=old['previous_test_known'],no_post_test_reselection=True)
    save('configs/slcd_R6_FINAL_LOCK.json',lock);print('R6_FROZEN',mid,sha('configs/slcd_R6_FINAL_LOCK.json'))

def test():
    lp='configs/slcd_R6_FINAL_LOCK.json';lock=read(lp);assert lock['test_access_authorized'] and sha('scripts/slcd_r6_evaluate.py')==lock['evaluation_source_sha256'] and sha('scripts/slcd_r2_evaluate.py')==lock['imported_evaluation_source_sha256']
    assert sha('configs/slcd_R6_METHODS.json')==lock['methods_config_sha256'] and sha('scripts/slcd_r6_scores.py')==lock['ranking_source_sha256']
    for n,h in lock['inference_source_sha256'].items():assert sha('scripts/'+n)==h
    priors,receipt=load_prior('test',lp);assert sha('data/slcd_test.jsonl')==lock['data_sha256']['test'] and sha('results/slcd_original_test.jsonl')==lock['original_test_scores_sha256'] and sha('results/slcd_BASELINE_REUSE_AUDIT.json')==lock['baseline_reuse_audit_sha256']
    audit=read('results/slcd_BASELINE_REUSE_AUDIT.json');assert audit['status']=='passed' and audit['n']==1172 and audit['full_prompt_candidate_gold_character_lengths_and_token_counts_equal']
    manifest={r['id']:r for r in rows('data/slcd_test.jsonl')};originals=rows('results/slcd_original_test.jsonl');assert len(originals)==1172 and {r['id'] for r in originals}==set(manifest)==set(priors)
    mid=lock['primary'];data=[]
    for row in originals:
        e=manifest[row['id']];p=priors[row['id']]
        assert row['gold']==p['gold']==e['gold'] and row['prompt_sha256']==p['prompt_sha256']==e['prompt_sha256'] and row['choice_char_lengths']==p['choice_char_lengths']==e['choice_char_lengths'] and row['choice_token_counts']==p['choice_token_counts']
        original=dict(id=row['id'],gold=row['gold'],original_methods=row['methods'],methods={});ps=predictions(original)
        ps[mid]=int(selected_score(row['methods'],p['prior_methods'],mid).argmax());data.append(dict(id=row['id'],gold=row['gold'],predictions=ps))
    res,_,trans=summarize(data,['native4','fixed','adaptive','incumbent_A1',mid]);rng=np.random.default_rng(2025)
    for r in res:
        n=r['n'];a=r['corrections'];b=r['regressions'];s=rng.multinomial(n,[a/n,b/n,1-(a+b)/n],size=50000);d=100*(s[:,0]-s[:,1])/n
        r.update(paired_bootstrap_low_pp=float(np.quantile(d,.025)),paired_bootstrap_high_pp=float(np.quantile(d,.975)))
    counts={r['method']:r['correct'] for r in res};assert [counts[k] for k in ['native4','fixed','adaptive','incumbent_A1']]==[702,717,716,726]
    result=dict(status='complete',n=1172,primary=mid,correct=counts[mid],target_correct=733,target_reached=counts[mid]>=733,results=res,final_lock_sha256=sha(lp),no_test_reselection=True,fresh_auxiliary_prior_rows=1172,original_test_GPU_replay_coverage=0,prior_independent_baseline_parity_coverage=0)
    save('results/slcd_R6_ARC_COMPLETE.json',result);writecsv('results/slcd_r6_arc_full_test_results.csv',res);writecsv('results/slcd_r6_paired_transitions.csv',trans);print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze-development','train','freeze-test','test']);a=ap.parse_args();{'freeze-development':freeze_development,'train':train,'freeze-test':freeze_test,'test':test}[a.stage]()
