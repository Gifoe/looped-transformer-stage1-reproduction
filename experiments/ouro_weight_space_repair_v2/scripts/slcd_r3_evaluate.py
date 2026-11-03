"""Four predeclared candidate-score compositions; reused TRAIN only selection."""
import argparse,json,time
import numpy as np
from slcd_core import ROOT,read,rows,sha,save,writecsv
from slcd_r2_evaluate import stats,predictions

METHODS={
 'r3_ref2_scorelate':dict(base='Ref2_w0p5',family='R1',gamma=1.,interpretation='non-novel reference2 composition control'),
 'r3_fisher_scorelate':dict(base='r2_fisher_signed',family='R2',gamma=1.,interpretation='late-residual bounded decorrelation plus candidate-level reversal'),
 'r3_matchednorm_scorelate':dict(base='r2_fisher_matched_norm',family='R2',gamma=1.,interpretation='matched local Fisher-norm magnitude control'),
 'r3_euclidean_scorelate':dict(base='r2_euclidean_signed',family='R2',gamma=1.,interpretation='uniform geometry control')}

def freeze_development():
    assert not (ROOT/'configs/slcd_R3_DEVELOPMENT_LOCK.json').exists()
    source={}
    for s in ['discovery','selection','confirmation']:
        for path in [f'results/slcd_r2_{s}_development.jsonl',f'results/slcd_tokens_{s}_'+('confirmation' if s=='confirmation' else 'neighbors')+'.jsonl']:
            receipt=read(path.replace('.jsonl','_COMPLETE.json'));assert sha(path)==receipt['score_file_sha256'];source[path]=sha(path)
    old=read('configs/slcd_R2_DEVELOPMENT_LOCK.json')
    lock=dict(status='frozen_before_R3_feedback',created_unix=time.time(),methods=METHODS,source_scores_sha256=source,fold_by_id=old['fold_by_id'],selection_rule=old['selection_rule'],target_correct=733,
        evaluation_source_sha256=sha('scripts/slcd_r3_evaluate.py'),imported_evaluation_source_sha256=sha('scripts/slcd_r2_evaluate.py'),
        hypothesis='R2 token interventions failed incumbent. Test a small fixed separation of probability-normalized token bases and complete-candidate late LL reversal; no guaranteed gain, no claim that this isolates causal logZ effects.',
        log_partition_identity='A1-A2 = sum_t[logZ(b-u)-logZ(b)+logZ(z5)-logZ(z4)]/answer_character_length',
        fixed_fold_scope='Previously reused train1093 robustness slices, not blind independent cross-validation',official_validation_selection=False,no_test_feedback=True)
    save('configs/slcd_R3_DEVELOPMENT_LOCK.json',lock);print('R3_DEVELOPMENT_FROZEN',sha('configs/slcd_R3_DEVELOPMENT_LOCK.json'))

def train():
    lock=read('configs/slcd_R3_DEVELOPMENT_LOCK.json');assert lock['methods']==METHODS and lock['evaluation_source_sha256']==sha('scripts/slcd_r3_evaluate.py') and lock['imported_evaluation_source_sha256']==sha('scripts/slcd_r2_evaluate.py')
    sources={}
    for path,h in lock['source_scores_sha256'].items():assert sha(path)==h;sources[path]={r['id']:r for r in rows(path)}
    allrows=[]
    for split in ['discovery','selection','confirmation']:
        r2=sources[f'results/slcd_r2_{split}_development.jsonl'];prior=sources[f'results/slcd_tokens_{split}_'+('confirmation' if split=='confirmation' else 'neighbors')+'.jsonl']
        assert set(r2)==set(prior)
        for qid,row in r2.items():
            assert row['gold']==prior[qid]['gold'] and row['prompt_sha256']==prior[qid]['prompt_sha256'] and row['choice_char_lengths']==prior[qid]['choice_char_lengths']
            ps=predictions(row);orig=row['original_methods'];late=np.asarray(orig['native_k4']['normalized_scores'])-np.asarray(orig['native_k5']['normalized_scores'])
            for mid,spec in METHODS.items():
                source=prior[qid] if spec['family']=='R1' else row
                x=np.asarray(source['methods'][spec['base']]['normalized_scores'])+late
                ps[mid]=int(x.argmax())
            allrows.append(dict(id=qid,gold=row['gold'],split=split,fold=lock['fold_by_id'][qid],predictions=ps))
    assert len(allrows)==1093 and len({r['id'] for r in allrows})==1093
    summaries=[];folds=[];trans=[]
    for mid in ['incumbent_A1','fixed']+list(METHODS):
        for subset in ['all','discovery','selection','confirmation']:
            rs=[r for r in allrows if subset=='all' or r['split']==subset]
            good=[r['predictions'][mid]==r['gold'] for r in rs]
            for baseline in ['fixed','incumbent_A1']:
                summaries.append(dict(method=mid,subset=subset,baseline=baseline,**stats(good,[r['predictions'][baseline]==r['gold'] for r in rs])))
        for fold in range(5):
            rs=[r for r in allrows if r['fold']==fold]
            folds.append(dict(method=mid,fold=fold,**stats([r['predictions'][mid]==r['gold'] for r in rs],[r['predictions']['incumbent_A1']==r['gold'] for r in rs])))
        for r in allrows:trans.append(dict(method=mid,id=r['id'],fold=r['fold'],split=r['split'],gold=r['gold'],prediction=r['predictions'][mid],incumbent_prediction=r['predictions']['incumbent_A1'],correct=r['predictions'][mid]==r['gold']))
    by={r['method']:r for r in summaries if r['subset']=='all' and r['baseline']=='incumbent_A1'}
    def order(mid):
        fs=[r for r in folds if r['method']==mid]
        return (-by[mid]['correct'],-min(r['net_correct']/r['n'] for r in fs),-sum(r['net_correct']>=0 for r in fs),mid)
    eligible=[mid for mid in METHODS if by[mid]['net_correct']>0 and sum(r['net_correct']>=0 for r in folds if r['method']==mid)>=3]
    ranked=sorted(METHODS,key=order);selected=sorted(eligible,key=order)[0] if eligible else None
    save('results/slcd_R3_DEVELOPMENT_COMPLETE.json',dict(status='complete',n=1093,selected=selected,ranking=ranked,eligible=eligible,summary=summaries,folds=folds,development_lock_sha256=sha('configs/slcd_R3_DEVELOPMENT_LOCK.json')))
    writecsv('results/slcd_r3_development_results.csv',summaries);writecsv('results/slcd_r3_fold_results.csv',folds);writecsv('results/slcd_r3_train_predictions.csv',trans)
    print(json.dumps(dict(selected=selected,ranking=[by[mid] for mid in ranked]),indent=2))

def freeze_test():
    development=read('results/slcd_R3_DEVELOPMENT_COMPLETE.json');mid=development['selected'];assert mid is not None
    dlock=read('configs/slcd_R3_DEVELOPMENT_LOCK.json');assert dlock['evaluation_source_sha256']==sha('scripts/slcd_r3_evaluate.py')
    spec=METHODS[mid];r2=read('configs/slcd_R2_DEVELOPMENT_LOCK.json')
    if spec['family']=='R1':base=dict(id='r3_token_base',kind='fixed',omega=.5,reference=2,final=4)
    else:
        base=next(r for r in read('configs/slcd_R2_RECIPES.json')['recipes'] if r['id']==spec['base']).copy();base['id']='r3_token_base'
    recipes=[dict(id='tf_fixed',kind='fixed',omega=.5),dict(id='tf_adaptive',kind='adaptive',omega=1.),base]
    assert not (ROOT/'configs/slcd_R3_FINAL_LOCK.json').exists();save('configs/slcd_R3_TEST_RECIPES.json',dict(recipes=recipes))
    from slcd_r3_worker import SOURCES
    old=read('FINAL_METHOD_LOCK.json')
    lock=dict(status='frozen_before_R3_test',created_unix=time.time(),primary=mid,method=spec,base_recipe=base,score_formula='normalized_token_base + (native_k4-normalized - native_k5-normalized)',test_access_authorized=True,target_correct=733,
        model_identity=r2['model_identity'],dataset_sha256=old['dataset_sha256'],allowed_recipe_sha256=[sha('configs/slcd_R3_TEST_RECIPES.json')],
        inference_source_sha256={n:sha('scripts/'+n) for n in SOURCES},evaluation_source_sha256=sha('scripts/slcd_r3_evaluate.py'),imported_evaluation_source_sha256=sha('scripts/slcd_r2_evaluate.py'),
        fixed_demonstrations_sha256=old['fixed_demonstrations_sha256'],prompt_protocol=old['prompt_protocol'],seed=2021,
        development_lock_sha256=sha('configs/slcd_R3_DEVELOPMENT_LOCK.json'),development_results_sha256=sha('results/slcd_R3_DEVELOPMENT_COMPLETE.json'),selection_rule=dlock['selection_rule'],
        test_is_blind=False,test_history='Previously used regression set, known R1 A1=726 and Fixed717. R2 failed training gate and was not tested. R3 uses reused train only; no official val reselection.',one_tested_R3_primary=True,no_post_test_reselection=True)
    save('configs/slcd_R3_FINAL_LOCK.json',lock);print('R3_FROZEN',mid,sha('configs/slcd_R3_FINAL_LOCK.json'))

def test():
    lock=read('configs/slcd_R3_FINAL_LOCK.json');assert sha('scripts/slcd_r3_evaluate.py')==lock['evaluation_source_sha256'] and sha('scripts/slcd_r2_evaluate.py')==lock['imported_evaluation_source_sha256']
    path='results/slcd_r3_test_final.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'));assert receipt['status']=='complete' and receipt['n']==1172 and sha(path)==receipt['score_file_sha256'] and receipt['provenance']['final_lock_sha256']==sha('configs/slcd_R3_FINAL_LOCK.json') and receipt['independent_parity_coverage']==1172
    for name,h in lock['inference_source_sha256'].items():assert sha('scripts/'+name)==h==receipt['provenance']['inference_source_sha256'][name]
    assert receipt['provenance']['recipes_sha256'] in lock['allowed_recipe_sha256'] and receipt['provenance']['data_sha256']==lock['dataset_sha256']['test']
    data=rows(path);assert len(data)==1172 and len({r['id'] for r in data})==1172
    mid=lock['primary'];allrows=[]
    for row in data:
        assert row['signature']==receipt['signature'] and len(row['baseline_parity_errors'])==8 and all(v==0 for v in row['baseline_parity_errors'].values())
        ps=predictions(row);m=row['original_methods'];x=np.asarray(row['methods']['r3_token_base']['normalized_scores'])+(np.asarray(m['native_k4']['normalized_scores'])-np.asarray(m['native_k5']['normalized_scores']));ps[mid]=int(x.argmax())
        allrows.append(dict(id=row['id'],gold=row['gold'],predictions=ps))
    rng=np.random.default_rng(2023);results=[];trans=[]
    for name in ['native4','fixed','adaptive','incumbent_A1',mid]:
        good=[r['predictions'][name]==r['gold'] for r in allrows]
        for baseline in ['fixed','incumbent_A1']:
            s=stats(good,[r['predictions'][baseline]==r['gold'] for r in allrows]);r=s['corrections'];d=s['regressions'];n=1172
            sample=rng.multinomial(n,[r/n,d/n,1-(r+d)/n],size=50000);delta=100*(sample[:,0]-sample[:,1])/n
            results.append(dict(method=name,baseline=baseline,**s,paired_bootstrap_low_pp=float(np.quantile(delta,.025)),paired_bootstrap_high_pp=float(np.quantile(delta,.975))))
        for row in allrows:trans.append(dict(method=name,id=row['id'],gold=row['gold'],prediction=row['predictions'][name],fixed_prediction=row['predictions']['fixed'],incumbent_prediction=row['predictions']['incumbent_A1'],correct=row['predictions'][name]==row['gold']))
    counts={r['method']:r['correct'] for r in results};assert [counts[k] for k in ['native4','fixed','adaptive','incumbent_A1']]==[702,717,716,726]
    result=dict(status='complete',n=1172,primary=mid,correct=counts[mid],target_correct=733,target_reached=counts[mid]>=733,results=results,final_lock_sha256=sha('configs/slcd_R3_FINAL_LOCK.json'),no_test_reselection=True)
    save('results/slcd_R3_ARC_COMPLETE.json',result);writecsv('results/slcd_r3_arc_full_test_results.csv',results);writecsv('results/slcd_r3_paired_transitions.csv',trans);print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze-development','train','freeze-test','test']);a=ap.parse_args();{'freeze-development':freeze_development,'train':train,'freeze-test':freeze_test,'test':test}[a.stage]()
