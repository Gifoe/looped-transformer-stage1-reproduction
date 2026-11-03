"""Fixed guidance-energy budgets; one training-selected final primary only."""
import argparse,json,time
import numpy as np
from slcd_core import ROOT,read,rows,sha,save,writecsv
from slcd_r2_evaluate import stats,predictions

def definitions():
    result={}
    for recipe in read('configs/slcd_R4_RECIPES.json')['recipes']:
        if recipe['id'].startswith('r4_'):
            mid=recipe['id'];result[mid]=dict(mode='token',base=mid,gamma=0.)
            result[mid+'_scorelate']=dict(mode='token',base=mid,gamma=1.)
    result['r4_adaptive_scorelate']=dict(mode='original',base='loopcd_k4_adaptive',gamma=1.)
    return result

def scores(row,spec):
    source=row['methods'] if spec['mode']=='token' else row['original_methods']
    base=np.asarray(source[spec['base']]['normalized_scores'])
    orig=row['original_methods'];late=np.asarray(orig['native_k4']['normalized_scores'])-np.asarray(orig['native_k5']['normalized_scores'])
    return base.copy() if spec['gamma']==0 else base+late

def freeze_development():
    assert not (ROOT/'configs/slcd_R4_DEVELOPMENT_LOCK.json').exists()
    lock=read('configs/slcd_R2_DEVELOPMENT_LOCK.json').copy()
    from slcd_r4_worker import SOURCES
    lock.update(status='frozen_before_R4_feedback',created_unix=time.time(),methods=definitions(),recipes_sha256=sha('configs/slcd_R4_RECIPES.json'),
        inference_source_sha256={n:sha('scripts/'+n) for n in SOURCES},evaluation_source_sha256=sha('scripts/slcd_r4_evaluate.py'),imported_evaluation_source_sha256=sha('scripts/slcd_r2_evaluate.py'),
        hypothesis='R2/R3 direction edits did not beat matched local-norm damping. Test per-token guidance energy budgets; RMS is local, bisection controls actual categorical KL. No first-invention or calibration guarantee.',
        selection_rule=['Eligible only if total1093 correct at least7 above frozen incumbent A1 and at least3/5 folds nonnegative vsA1','Rank by total correct descending','then minimum proportional fold net vsA1 descending','then number of nonnegative folds descending','then ID lexicographically'],
        minimum_train_net_for_test=7,reason_for_guard='Avoid spending held-out evaluation on R3 best gain of only1 training question; user target needs at least7 additional test corrections, not guaranteed by this guard')
    save('configs/slcd_R4_DEVELOPMENT_LOCK.json',lock);print('R4_DEVELOPMENT_FROZEN',sha('configs/slcd_R4_DEVELOPMENT_LOCK.json'))

def load_training():
    lock=read('configs/slcd_R4_DEVELOPMENT_LOCK.json');assert sha('scripts/slcd_r4_evaluate.py')==lock['evaluation_source_sha256'] and sha('scripts/slcd_r2_evaluate.py')==lock['imported_evaluation_source_sha256']
    allrows=[]
    for split in ['discovery','selection','confirmation']:
        path=f'results/slcd_r4_{split}_development.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'));assert receipt['status']=='complete' and sha(path)==receipt['score_file_sha256'] and receipt['provenance']['authorization_lock_sha256']==sha('configs/slcd_R4_DEVELOPMENT_LOCK.json')
        assert receipt['provenance']['inference_source_sha256']==lock['inference_source_sha256'] and receipt['provenance']['data_sha256']==lock['train_data_sha256'][split]
        data=rows(path);assert len(data)==receipt['n']
        for row in data:
            assert row['signature']==receipt['signature'] and len(row['baseline_parity_errors'])==8 and all(v==0 for v in row['baseline_parity_errors'].values())
            ps=predictions(row)
            for mid,spec in lock['methods'].items():ps[mid]=int(scores(row,spec).argmax())
            allrows.append(dict(id=row['id'],gold=row['gold'],predictions=ps,split=split,fold=lock['fold_by_id'][row['id']]))
    assert len(allrows)==1093 and {r['id'] for r in allrows}==set(lock['fold_by_id'])
    return lock,allrows

def summarize(allrows,names,include_subsets=False):
    summary=[];folds=[];trans=[]
    for mid in names:
        for subset in (['all','discovery','selection','confirmation'] if include_subsets else ['all']):
            rs=[r for r in allrows if subset=='all' or r['split']==subset];good=[r['predictions'][mid]==r['gold'] for r in rs]
            for baseline in ['fixed','incumbent_A1']:
                summary.append(dict(method=mid,subset=subset,baseline=baseline,**stats(good,[r['predictions'][baseline]==r['gold'] for r in rs])))
        if include_subsets:
            for fold in range(5):
                rs=[r for r in allrows if r['fold']==fold];folds.append(dict(method=mid,fold=fold,**stats([r['predictions'][mid]==r['gold'] for r in rs],[r['predictions']['incumbent_A1']==r['gold'] for r in rs])))
        for r in allrows:trans.append(dict(method=mid,id=r['id'],gold=r['gold'],prediction=r['predictions'][mid],incumbent_prediction=r['predictions']['incumbent_A1'],fixed_prediction=r['predictions']['fixed'],correct=r['predictions'][mid]==r['gold']))
    return summary,folds,trans

def train():
    lock,data=load_training();res,folds,trans=summarize(data,['native4','fixed','adaptive','incumbent_A1']+list(lock['methods']),True)
    by={r['method']:r for r in res if r['subset']=='all' and r['baseline']=='incumbent_A1'}
    def order(mid):
        fs=[r for r in folds if r['method']==mid]
        return (-by[mid]['correct'],-min(r['net_correct']/r['n'] for r in fs),-sum(r['net_correct']>=0 for r in fs),mid)
    eligible=[mid for mid in lock['methods'] if by[mid]['net_correct']>=lock['minimum_train_net_for_test'] and sum(r['net_correct']>=0 for r in folds if r['method']==mid)>=3]
    ranking=sorted(lock['methods'],key=order);selected=sorted(eligible,key=order)[0] if eligible else None
    save('results/slcd_R4_DEVELOPMENT_COMPLETE.json',dict(status='complete',selected=selected,ranking=ranking,eligible=eligible,summary=res,folds=folds,development_lock_sha256=sha('configs/slcd_R4_DEVELOPMENT_LOCK.json')))
    writecsv('results/slcd_r4_development_results.csv',res);writecsv('results/slcd_r4_fold_results.csv',folds);writecsv('results/slcd_r4_train_predictions.csv',trans)
    print(json.dumps(dict(selected=selected,ranking=[by[mid] for mid in ranking]),indent=2))

def freeze_test():
    dlock=read('configs/slcd_R4_DEVELOPMENT_LOCK.json');development=read('results/slcd_R4_DEVELOPMENT_COMPLETE.json');mid=development['selected'];assert mid is not None
    assert dlock['evaluation_source_sha256']==sha('scripts/slcd_r4_evaluate.py') and development['development_lock_sha256']==sha('configs/slcd_R4_DEVELOPMENT_LOCK.json')
    assert sha('configs/slcd_R4_RECIPES.json')==dlock['recipes_sha256'] and sha('scripts/slcd_r2_evaluate.py')==dlock['imported_evaluation_source_sha256'] and mid in development['eligible']
    selected_summary=next(r for r in development['summary'] if r['method']==mid and r['subset']=='all' and r['baseline']=='incumbent_A1')
    assert selected_summary['net_correct']>=dlock['minimum_train_net_for_test'] and sum(r['net_correct']>=0 for r in development['folds'] if r['method']==mid)>=3
    spec=dlock['methods'][mid];recipes=[r for r in read('configs/slcd_R4_RECIPES.json')['recipes'] if r['id'] in ['tf_fixed','tf_adaptive']+([spec['base']] if spec['mode']=='token' else [])]
    assert not (ROOT/'configs/slcd_R4_FINAL_LOCK.json').exists();save('configs/slcd_R4_TEST_RECIPES.json',dict(recipes=recipes))
    old=read('FINAL_METHOD_LOCK.json')
    lock=dict(status='frozen_before_R4_test',created_unix=time.time(),primary=mid,method=spec,test_access_authorized=True,target_correct=733,
        model_identity=dlock['model_identity'],dataset_sha256=old['dataset_sha256'],allowed_recipe_sha256=[sha('configs/slcd_R4_TEST_RECIPES.json')],inference_source_sha256=dlock['inference_source_sha256'],
        evaluation_source_sha256=sha('scripts/slcd_r4_evaluate.py'),imported_evaluation_source_sha256=sha('scripts/slcd_r2_evaluate.py'),fixed_demonstrations_sha256=old['fixed_demonstrations_sha256'],prompt_protocol=old['prompt_protocol'],seed=2021,
        development_lock_sha256=sha('configs/slcd_R4_DEVELOPMENT_LOCK.json'),development_results_sha256=sha('results/slcd_R4_DEVELOPMENT_COMPLETE.json'),selection_rule=dlock['selection_rule'],
        test_is_blind=False,test_history='Known R1 aggregate726/1172 and Fixed717. R2 and R3 not tested; R4 selected on reused train only; no official validation reselection.',no_post_test_reselection=True,
        baseline_reuse_audit_sha256=sha('results/slcd_BASELINE_REUSE_AUDIT.json'),original_test_score_sha256=sha('results/slcd_original_test.jsonl'))
    save('configs/slcd_R4_FINAL_LOCK.json',lock);print('R4_FROZEN',mid,'needs_fresh_test_GPU',spec['mode']=='token')

def test():
    lock=read('configs/slcd_R4_FINAL_LOCK.json');assert lock['test_access_authorized'] and lock['evaluation_source_sha256']==sha('scripts/slcd_r4_evaluate.py') and lock['imported_evaluation_source_sha256']==sha('scripts/slcd_r2_evaluate.py')
    for n,h in lock['inference_source_sha256'].items():assert sha('scripts/'+n)==h
    assert sha('data/slcd_test.jsonl')==lock['dataset_sha256']['test']
    expected={r['id']:r for r in rows('data/slcd_test.jsonl')};assert len(expected)==1172
    for name,h in lock['model_identity']['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+name)==h
    spec=lock['method'];mid=lock['primary']
    if spec['mode']=='token':
        path='results/slcd_r4_test_final.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'));assert receipt['status']=='complete' and receipt['n']==1172 and sha(path)==receipt['score_file_sha256'] and receipt['provenance']['final_lock_sha256']==sha('configs/slcd_R4_FINAL_LOCK.json') and receipt['independent_parity_coverage']==1172
        assert receipt['provenance']['inference_source_sha256']==lock['inference_source_sha256'] and receipt['provenance']['recipes_sha256'] in lock['allowed_recipe_sha256'];source=rows(path)
        for row in source:
            e=expected[row['id']]
            assert row['signature']==receipt['signature'] and len(row['baseline_parity_errors'])==8 and all(v==0 for v in row['baseline_parity_errors'].values())
            assert set(row['methods'])=={'tf_fixed','tf_adaptive',spec['base']} and row['gold']==e['gold'] and row['prompt_sha256']==e['prompt_sha256'] and row['choice_char_lengths']==e['choice_char_lengths']
    else:
        assert sha('results/slcd_BASELINE_REUSE_AUDIT.json')==lock['baseline_reuse_audit_sha256'] and sha('results/slcd_original_test.jsonl')==lock['original_test_score_sha256']
        from slcd_token_worker_v2 import file_sha,MODEL_SHA
        assert file_sha(ROOT/'models/Ouro-1.4B/model.safetensors')==MODEL_SHA
        audit=read('results/slcd_BASELINE_REUSE_AUDIT.json');assert audit['status']=='passed' and audit['n']==1172 and audit['full_prompt_candidate_gold_character_lengths_and_token_counts_equal'] and audit['full_baseline_receipt_bound']
        from transformers import AutoTokenizer
        from runtime import encode_pair
        tok=AutoTokenizer.from_pretrained(ROOT/'models/Ouro-1.4B',local_files_only=True)
        raw=rows('results/slcd_original_test.jsonl');assert {r['id'] for r in raw}==set(expected)
        for r in raw:
            e=expected[r['id']];assert r['gold']==e['gold'] and r['prompt_sha256']==e['prompt_sha256'] and r['choice_char_lengths']==e['choice_char_lengths']
            assert r['fewshot']==25 and r['demonstrations']=='fixed_global_25' and r['cached'] is False and r['data_sha256']==lock['dataset_sha256']['test']
            assert [len(encode_pair(tok,*p)[1]) for p in e['request_pairs']]==r['choice_token_counts']
        source=[dict(id=r['id'],gold=r['gold'],original_methods=r['methods'],methods={}) for r in raw]
    assert len(source)==1172 and len({r['id'] for r in source})==1172
    assert {r['id'] for r in source}==set(expected)
    data=[]
    for row in source:
        ps=predictions(row);ps[mid]=int(scores(row,spec).argmax());data.append(dict(id=row['id'],gold=row['gold'],predictions=ps))
    res,_,trans=summarize(data,['native4','fixed','adaptive','incumbent_A1',mid]);rng=np.random.default_rng(2024)
    for r in res:
        n=r['n'];a=r['corrections'];b=r['regressions'];sample=rng.multinomial(n,[a/n,b/n,1-(a+b)/n],size=50000);delta=100*(sample[:,0]-sample[:,1])/n
        r.update(paired_bootstrap_low_pp=float(np.quantile(delta,.025)),paired_bootstrap_high_pp=float(np.quantile(delta,.975)))
    counts={r['method']:r['correct'] for r in res};assert [counts[k] for k in ['native4','fixed','adaptive','incumbent_A1']]==[702,717,716,726]
    result=dict(status='complete',n=1172,primary=mid,correct=counts[mid],target_correct=733,target_reached=counts[mid]>=733,results=res,final_lock_sha256=sha('configs/slcd_R4_FINAL_LOCK.json'),no_test_reselection=True,independent_test_replay_coverage=1172 if spec['mode']=='token' else 0)
    save('results/slcd_R4_ARC_COMPLETE.json',result);writecsv('results/slcd_r4_arc_full_test_results.csv',res);writecsv('results/slcd_r4_paired_transitions.csv',trans);print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze-development','train','freeze-test','test']);a=ap.parse_args();{'freeze-development':freeze_development,'train':train,'freeze-test':freeze_test,'test':test}[a.stage]()
