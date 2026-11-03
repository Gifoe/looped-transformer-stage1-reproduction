"""R2 train-only fixed-fold comparison, followed by one preregistered test lock."""
import argparse,json,time
import numpy as np
from slcd_core import ROOT,read,rows,sha,digest,save,writecsv

def stats(values,base):
    from scipy.stats import binomtest
    good=np.asarray(values,dtype=bool);old=np.asarray(base,dtype=bool);r=int((good&~old).sum());d=int((old&~good).sum())
    return dict(n=len(good),correct=int(good.sum()),accuracy=float(good.mean()),corrections=r,regressions=d,net_correct=r-d,delta_pp=100*(r-d)/len(good),exact_mcnemar_p=1. if r+d==0 else float(binomtest(r,r+d,.5).pvalue))

def collect(split,job):
    path=f'results/slcd_r2_{split}_{job}.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'));assert receipt['status']=='complete' and sha(path)==receipt['score_file_sha256'] and receipt['baseline_parity_coverage']==receipt['n']
    data=rows(path);assert len(data)==receipt['n']==len({r['id'] for r in data})
    assert all(r['signature']==receipt['signature'] and all(v==0 for v in r['baseline_parity_errors'].values()) for r in data)
    return data,receipt

def predictions(row):
    orig=row['original_methods'];arrays={name:np.asarray(v['normalized_scores'],dtype=float) for name,v in orig.items()}
    a1=arrays['loopcd_k4_fixed']+(arrays['native_k4']-arrays['native_k5'])
    out={'incumbent_A1':int(a1.argmax()),'native4':orig['native_k4']['prediction'],'fixed':orig['loopcd_k4_fixed']['prediction'],'adaptive':orig['loopcd_k4_adaptive']['prediction']}
    out.update({name:v['prediction'] for name,v in row['methods'].items() if name not in ['tf_fixed','tf_adaptive']})
    return out

def train():
    lock=read('configs/slcd_R2_DEVELOPMENT_LOCK.json');assert sha('configs/slcd_R2_RECIPES.json')==lock['recipes_sha256'] and sha('scripts/slcd_r2_evaluate.py')==lock['evaluation_source_sha256']
    allrows=[];receipts={}
    for split in ['discovery','selection','confirmation']:
        data,r=collect(split,'development');receipts[split]=r
        assert r['provenance']['inference_source_sha256']==lock['inference_source_sha256'] and r['provenance']['data_sha256']==lock['train_data_sha256'][split] and r['provenance']['authorization_lock_sha256']==sha('configs/slcd_R2_DEVELOPMENT_LOCK.json')
        for v in data:v['development_split']=split
        allrows+=data
    assert len(allrows)==1093 and {r['id'] for r in allrows}==set(lock['fold_by_id'])
    names=list(predictions(allrows[0]));summaries=[];folds=[];outcomes=[]
    for mid in names:
        for subset in ['all','discovery','selection','confirmation']:
            rs=[r for r in allrows if subset=='all' or r['development_split']==subset]
            good=[predictions(r)[mid]==r['gold'] for r in rs]
            for baseline in ['fixed','incumbent_A1']:
                old=[predictions(r)[baseline]==r['gold'] for r in rs];summaries.append(dict(method=mid,subset=subset,baseline=baseline,**stats(good,old)))
        for fold in range(5):
            rs=[r for r in allrows if lock['fold_by_id'][r['id']]==fold]
            good=[predictions(r)[mid]==r['gold'] for r in rs];old=[predictions(r)['incumbent_A1']==r['gold'] for r in rs]
            folds.append(dict(method=mid,fold=fold,baseline='incumbent_A1',**stats(good,old)))
        for r in allrows:
            ps=predictions(r);outcomes.append(dict(method=mid,id=r['id'],fold=lock['fold_by_id'][r['id']],split=r['development_split'],gold=r['gold'],prediction=ps[mid],incumbent_prediction=ps['incumbent_A1'],fixed_prediction=ps['fixed'],correct=ps[mid]==r['gold']))
    candidates=[r['id'] for r in read('configs/slcd_R2_RECIPES.json')['recipes'] if r['id'].startswith('r2_')]
    by={r['method']:r for r in summaries if r['subset']=='all' and r['baseline']=='incumbent_A1'}
    eligible=[mid for mid in candidates if by[mid]['net_correct']>0 and sum(r['net_correct']>=0 for r in folds if r['method']==mid)>=3]
    def order(mid):
        f=[r for r in folds if r['method']==mid]
        return (-by[mid]['correct'],-min(r['net_correct']/r['n'] for r in f),-sum(r['net_correct']>=0 for r in f),mid)
    ranked=sorted(candidates,key=order);selected=sorted(eligible,key=order)[0] if eligible else None
    save('results/slcd_R2_DEVELOPMENT_COMPLETE.json',dict(status='complete',n=1093,candidates=candidates,ranking=ranked,eligible=eligible,selected=selected,summary=summaries,folds=folds,
        development_lock_sha256=sha('configs/slcd_R2_DEVELOPMENT_LOCK.json'),selection_rule=lock['selection_rule'],no_validation_or_test_feedback=True))
    writecsv('results/slcd_r2_development_results.csv',summaries);writecsv('results/slcd_r2_fold_results.csv',folds);writecsv('results/slcd_r2_train_predictions.csv',outcomes)
    print(json.dumps(dict(selected=selected,ranking=[dict(by[mid]) for mid in ranked]),indent=2))

def freeze():
    dev=read('results/slcd_R2_DEVELOPMENT_COMPLETE.json');mid=dev['selected'];assert mid is not None,'No R2 candidate beat incumbent on training under preregistered rule'
    config=read('configs/slcd_R2_RECIPES.json');selected=[r for r in config['recipes'] if r['id'] in ['tf_fixed','tf_adaptive',mid]];assert len(selected)==3
    assert not (ROOT/'configs/slcd_R2_FINAL_LOCK.json').exists()
    save('configs/slcd_R2_TEST_RECIPES.json',dict(recipes=selected,status='one frozen R2 primary plus controls'))
    old=read('FINAL_METHOD_LOCK.json');dlock=read('configs/slcd_R2_DEVELOPMENT_LOCK.json')
    lock=dict(status='frozen_before_R2_test',created_unix=time.time(),primary=mid,recipe=next(r for r in selected if r['id']==mid),test_access_authorized=True,target_correct=733,
        model_checkpoint=old['model_checkpoint'],model_revision=old['model_revision'],model_safetensors_sha256=old['model_safetensors_sha256'],model_identity=dlock['model_identity'],weights_modified=False,evaluation_source_sha256=dlock['evaluation_source_sha256'],
        fixed_demonstrations_sha256=old['fixed_demonstrations_sha256'],prompt_protocol=old['prompt_protocol'],seed=2021,
        dataset_sha256=old['dataset_sha256'],allowed_recipe_sha256=[sha('configs/slcd_R2_TEST_RECIPES.json')],inference_source_sha256=dlock['inference_source_sha256'],
        selection_rule=dlock['selection_rule'],development_lock_sha256=sha('configs/slcd_R2_DEVELOPMENT_LOCK.json'),development_results_sha256=sha('results/slcd_R2_DEVELOPMENT_COMPLETE.json'),
        primary_comparison='R2 vs incumbent A1=726; original Fixed717 secondary',test_is_blind=False,test_history='Known original Fixed717, exploratory and R1 A1 gamma1=726, and historical edited V1/V2 results. R2 selected only on reused training folds; no official validation reselection.',
        user_scope_extension='User requested newer methods and at least62.5% after R1 test; R1 lock remains unchanged',one_tested_R2_primary=True,no_post_test_reselection=True)
    save('configs/slcd_R2_FINAL_LOCK.json',lock);print('R2_FROZEN',mid,sha('configs/slcd_R2_FINAL_LOCK.json'))

def test():
    lock=read('configs/slcd_R2_FINAL_LOCK.json');assert sha('scripts/slcd_r2_evaluate.py')==lock['evaluation_source_sha256'];data,receipt=collect('test','final');assert len(data)==1172 and receipt['provenance']['final_lock_sha256']==sha('configs/slcd_R2_FINAL_LOCK.json')
    mid=lock['primary'];assert all(set(r['methods'])=={'tf_fixed','tf_adaptive',mid} for r in data)
    rng=np.random.default_rng(2022);summaries=[];trans=[]
    for method in ['native4','fixed','adaptive','incumbent_A1',mid]:
        good=[predictions(r)[method]==r['gold'] for r in data]
        for baseline in ['fixed','incumbent_A1']:
            old=[predictions(r)[baseline]==r['gold'] for r in data];s=stats(good,old);r=s['corrections'];d=s['regressions'];n=len(data)
            draws=rng.multinomial(n,[r/n,d/n,1-(r+d)/n],size=50000);delta=100*(draws[:,0]-draws[:,1])/n
            summaries.append(dict(method=method,baseline=baseline,**s,paired_bootstrap_low_pp=float(np.quantile(delta,.025)),paired_bootstrap_high_pp=float(np.quantile(delta,.975))))
        for row in data:
            ps=predictions(row);trans.append(dict(method=method,id=row['id'],gold=row['gold'],prediction=ps[method],incumbent_prediction=ps['incumbent_A1'],fixed_prediction=ps['fixed'],correct=ps[method]==row['gold']))
    counts={r['method']:r['correct'] for r in summaries};assert [counts[k] for k in ['native4','fixed','adaptive','incumbent_A1']]==[702,717,716,726]
    result=dict(status='complete',n=1172,primary=mid,correct=counts[mid],target_correct=733,target_reached=counts[mid]>=733,results=summaries,final_lock_sha256=sha('configs/slcd_R2_FINAL_LOCK.json'),no_test_reselection=True)
    save('results/slcd_R2_ARC_COMPLETE.json',result);writecsv('results/slcd_r2_arc_full_test_results.csv',summaries);writecsv('results/slcd_r2_paired_transitions.csv',trans);print(json.dumps(result,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','freeze','test']);a=ap.parse_args();{'train':train,'freeze':freeze,'test':test}[a.stage]()
