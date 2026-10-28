"""One-command CPU rebuild: immutable audit -> gold-free features -> nested CV."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import binomtest, mannwhitneyu
from sklearn.metrics import roc_auc_score, average_precision_score
from quality_core import *


def write_csv(path,records):
    pd.DataFrame(records).to_csv(ROOT/path,index=False,encoding='utf8',lineterminator='\n')


def seal_audit():
    manifest=json.loads((BASE/'ARTIFACT_MANIFEST.json').read_text(encoding='utf8'))
    checked={k:digest(BASE/k)==v['sha256'] and (BASE/k).stat().st_size==v['bytes'] for k,v in manifest.items()}
    assert all(checked.values()),[k for k,v in checked.items() if not v]
    return dict(passed=True,files=len(checked),manifest_sha256=digest(BASE/'ARTIFACT_MANIFEST.json'))


def build_data():
    initial=seal_audit()
    rows=read(BASE/'data/qwen_test.jsonl')
    original=read(BASE/'results/qwen_A0_test.jsonl');euler=read(BASE/'results/qwen_A3_test.jsonl')
    index=[]
    for values in [rows,original,euler]:
        d={r['id']:r for r in values};assert len(values)==len(d)==600;index.append(d)
    d,a,b=index;assert set(d)==set(a)==set(b)
    F=[];labels=[];leakage=[]
    for row in rows:
        uid=row['id'];o,e=a[uid],b[uid]
        for r in [o,e]:
            for k in ['question','choices','gold','task','dataset_revision','official_split']:assert r[k]==row[k],(uid,k)
            assert r['prediction']==int(np.argmax(r['scores']))
            assert r['correct']==(r['prediction']==r['gold'])
        assert o['protocol']==e['protocol']=='zero_shot_complementary_audit_v1'
        assert o['configuration']=={'id':'A0','strategy':'original','K':0}
        assert e['configuration']=={'id':'A3','strategy':'euler','K':6,'alpha':1/6}
        f=features(visible(o),visible(e));g=int(e['correct'])-int(o['correct'])
        for r in [o,e]:
            r2=dict(r,gold=-123,correct=not r['correct'],gold_nll_per_token=1e12,gold_choice_logp=1e12)
            assert visible(r2)==visible(r)
        label=dict(id=uid,task=row['task'],group=group(row['task']),gold=row['gold'],
                   original_prediction=o['prediction'],euler_prediction=e['prediction'],
                   original_correct=int(o['correct']),euler_correct=int(e['correct']),gain=g,
                   changed=o['prediction']!=e['prediction'],original_elapsed_s=o['elapsed_s'],euler_elapsed_s=e['elapsed_s'])
        F.append(f);labels.append(label)
    gain=np.array([x['gain'] for x in labels]);changed=np.array([x['changed'] for x in labels])
    assert sum(x['original_correct'] for x in labels)==297
    assert sum(x['euler_correct'] for x in labels)==314
    assert int((gain==1).sum())==32 and int((gain==-1).sum())==15 and changed.sum()==75
    assert np.all(gain[~changed]==0)
    sources={str(p.relative_to(BASE)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in [
        BASE/'data/qwen_test.jsonl',BASE/'results/qwen_A0_test.jsonl',BASE/'results/qwen_A3_test.jsonl',
        BASE/'scripts/qwen_audit.py',BASE/'scripts/qwen_recipe.py',BASE/'configs/protocol.json']}
    audit=dict(passed=True,n=600,original_correct=297,euler_correct=314,beneficial=32,harmful=15,
               prediction_changes=75,both_wrong_changed=28,both_correct=282,both_wrong=271,
               oracle_correct=329,oracle_accuracy_percent=329/6,
               model='Qwen/Qwen3-4B-Base',model_revision=json.loads((BASE/'results/qwen_parity.json').read_text())['model_revision'],
               join='unique exact sample ID; order determined only by pinned data manifest',
               input_sources=sources,original_seal=initial,label_mutation_feature_invariance_all600=True,
               historical_log_limitations=['No full hidden vectors or update-vector directions',
               'Hidden summaries and full-vocabulary statistics refer to last prompt token from candidate0; candidate scoring is causal',
               'Only forward full-vocab KL is stored, not full distributions; reverse KL/JS/TV computed at choice level only',
               'No candidate token lengths saved; character lengths and prompt length available; no GPU rerun required'])
    put('results/DATA_AUDIT.json',audit)
    write_csv('data/discovery_labels.csv',labels)
    write_csv('data/discovery_features.csv',[dict(id=l['id'],**f) for l,f in zip(labels,F)])
    # Feature-only files intentionally contain neither gold nor task ID.
    assert not any(k in F[0] for k in ['gold','correct','task','gain','gold_choice_logp','gold_nll_per_token'])
    return F,labels


def sanity(F):
    frame=pd.DataFrame(F);rows=[]
    for name in frame:
        x=frame[name];sd=x.std(ddof=1);z=np.abs((x-x.mean())/sd) if sd>0 else np.zeros(len(x))
        rows.append(dict(feature=name,n=int(x.notna().sum()),missing_fraction=float(x.isna().mean()),
                         mean=float(x.mean()),variance=float(x.var(ddof=1)),min=float(x.min()),max=float(x.max()),
                         constant=bool(sd==0),outliers_absolute_z_over4=int((z>4).sum()),
                         linear_model_included=name in PLAN['linear_features']))
    write_csv('results/feature_sanity.csv',rows)
    # Constant-feature correlations are undefined and deliberately exported blank.
    frame.corr().to_csv(ROOT/'results/feature_correlations.csv',index=True,lineterminator='\n')
    pairs=[]
    c=frame.corr()
    for i,a in enumerate(c):
        for b in c.columns[i+1:]:
            if np.isfinite(c.loc[a,b]) and abs(c.loc[a,b])>=.95:pairs.append(dict(feature1=a,feature2=b,pearson_r=float(c.loc[a,b])))
    write_csv('results/high_feature_correlations.csv',pairs)
    metadata={}
    for name in frame:
        category=('choice_level' if name.startswith(('choice_','score_','original_top','euler_top')) else
                  'full_vocabulary_summary' if name.startswith('vocab_') else
                  'window_last_prompt_token' if name.startswith(('hidden_','step_')) else 'context_or_control')
        metadata[name]=dict(source=category,normalization='None before splitting; linear StandardScaler is train-fold only',
                            gold_free=True,model_feature=name in PLAN['linear_features'])
    put('configs/feature_schema.json',metadata)


def auc_ci(score,y,seed=2021):
    score=np.asarray(score,float);y=np.asarray(y,int)
    p=score[y==1];n=score[y==0];np_,nn=len(p),len(n)
    if not np_ or not nn:return {}
    rng=np.random.default_rng(seed);B=PLAN['bootstrap_resamples']
    wp=rng.multinomial(np_,np.full(np_,1/np_),size=B)
    wn=rng.multinomial(nn,np.full(nn,1/nn),size=B)
    pair=(p[:,None]>n[None,:]).astype(float)+.5*(p[:,None]==n[None,:])
    auc=np.sum((wp@pair)*wn,axis=1)/(np_*nn)
    allscore=np.r_[p,n];order=np.argsort(-allscore,kind='stable')
    weights=np.c_[wp,wn][:,order];ispos=np.r_[np.ones(np_,bool),np.zeros(nn,bool)][order]
    tp=np.cumsum(weights*ispos,axis=1);fp=np.cumsum(weights*~ispos,axis=1)
    endpoints=np.r_[np.where(np.diff(allscore[order])!=0)[0],np_+nn-1]
    tp=tp[:,endpoints];fp=fp[:,endpoints]
    recall=tp/np_;precision=np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=tp+fp>0)
    ap=np.sum(np.diff(np.c_[np.zeros(B),recall],axis=1)*precision,axis=1)
    mp=wp@p/np_;mn=wn@n/nn
    vp=np.maximum(0,(wp@(p*p)-np_*mp*mp)/(np_-1));vn=np.maximum(0,(wn@(n*n)-nn*mn*mn)/(nn-1))
    pooled=np.sqrt(((np_-1)*vp+(nn-1)*vn)/(np_+nn-2));difference=mp-mn
    hg=np.divide(difference,pooled,out=np.zeros(B),where=pooled>0)*(1-3/(4*(np_+nn)-9))
    return dict(auroc=float(roc_auc_score(y,score)),auroc_low=float(np.quantile(auc,.025)),auroc_high=float(np.quantile(auc,.975)),
                auprc=float(average_precision_score(y,score)),auprc_low=float(np.quantile(ap,.025)),auprc_high=float(np.quantile(ap,.975)),
                rank_biserial_low=float(np.quantile(2*auc-1,.025)),rank_biserial_high=float(np.quantile(2*auc-1,.975)),
                mean_difference_low=float(np.quantile(difference,.025)),mean_difference_high=float(np.quantile(difference,.975)),
                hedges_g_low=float(np.quantile(hg,.025)),hedges_g_high=float(np.quantile(hg,.975)),
                positive_prevalence=np_/(np_+nn),n_beneficial=np_,n_harmful=nn)


def univariate(F,labels):
    gain=np.array([x['gain'] for x in labels]);sel=gain!=0;y=(gain[sel]>0).astype(int);rows=[]
    for name in F[0]:
        raw=np.array([f[name] for f in F],float)[sel];valid=np.isfinite(raw)
        x=raw[valid];yy=y[valid]
        stats=auc_ci(x,yy);p=x[yy==1];n=x[yy==0]
        u=mannwhitneyu(p,n,alternative='two-sided',method='asymptotic')
        sd=np.sqrt(((len(p)-1)*p.var(ddof=1)+(len(n)-1)*n.var(ddof=1))/(len(p)+len(n)-2))
        delta=float(p.mean()-n.mean());g=delta/sd*(1-3/(4*(len(p)+len(n))-9)) if sd>0 else 0.
        # Signed raw AUC is primary; direction picked after seeing this set is exploratory.
        rows.append(dict(feature=name,**stats,mean_beneficial=float(p.mean()),mean_harmful=float(n.mean()),
                         mean_difference=delta,hedges_g=float(g),rank_biserial=2*stats['auroc']-1,
                         mann_whitney_p=float(u.pvalue),exploratory_best_direction='higher' if stats['auroc']>=.5 else 'lower',
                         exploratory_direction_optimized_auc=max(stats['auroc'],1-stats['auroc'])))
    order=np.argsort([x['mann_whitney_p'] for x in rows]);m=len(rows);last=0
    for rank,i in enumerate(order):last=max(last,min(1,(m-rank)*rows[i]['mann_whitney_p']));rows[i]['holm_p']=last
    write_csv('results/univariate_predictability.csv',rows)
    # Null permutation association for each signal is evaluated without choosing orientation.
    rng=np.random.default_rng(2021)
    null=[]
    from scipy.stats import rankdata
    for name in F[0]:
        raw=np.array([f[name] for f in F],float)[sel];valid=np.isfinite(raw);x=raw[valid];yy=y[valid]
        ys=np.array([rng.permutation(yy) for _ in range(1000)]);ranks=rankdata(x)
        positive=int(yy.sum());negative=len(yy)-positive
        auc=(ys@ranks-positive*(positive+1)/2)/(positive*negative)
        observed=next(z['auroc'] for z in rows if z['feature']==name)
        null.append(dict(feature=name,permutations=1000,null_auc_mean=float(auc.mean()),null_auc_low=float(np.quantile(auc,.025)),
                         null_auc_high=float(np.quantile(auc,.975)),two_sided_permutation_p=float((1+np.sum(np.abs(auc-.5)>=abs(observed-.5)))/1001)))
    write_csv('results/shuffled_feature_label_null.csv',null)


def policies(F,labels):
    N=len(labels);gain=np.array([x['gain'] for x in labels]);changed=np.array([x['changed'] for x in labels]);groups=np.array([x['group'] for x in labels])
    names=PLAN['linear_features'];X=np.array([[f[k] for k in names] for f in F]);orig_names=PLAN['original_only_features'];XO=np.array([[f[k] for k in orig_names] for f in F])
    allindices=np.arange(N);trainable=np.flatnonzero(changed)
    split=cv_splits(allindices,gain,PLAN['outer_folds'],2021)
    put('data/outer_folds.json',[dict(fold=i,train_ids=[labels[j]['id'] for j in tr],test_ids=[labels[j]['id'] for j in va]) for i,(tr,va) in enumerate(split)])
    decisions=[];models=[];scores={}
    fixed={'confidence_gate':['choice_confidence_delta'],'margin_entropy_gate':['choice_margin_delta','choice_entropy_delta'],
           'kl_gate':['choice_kl_euler_original'],'original_confidence_only':['choice_original_confidence'],
           'euler_confidence_only':['choice_euler_confidence'],'random_feature_gate':['random_feature']}
    def run(scope,fold,tr,te):
        fitting=tr[changed[tr]];assert len(set(tr)&set(te))==0 or scope=='discovery_fit'
        acts={'always_original':np.zeros(N,bool),'always_euler':np.ones(N,bool),'oracle':gain>0}
        quality={}
        for name,fields in fixed.items():
            m=fit_gate(F,gain,fitting,fields);acts[name]=gate_decisions(m,F)
            quality[name]=(m.get('sign') or 1)*np.array([f[m['feature']] if m.get('feature') else 0 for f in F])
            models.append(dict(scope=scope,fold=fold,policy=name,training_ids=[labels[i]['id'] for i in fitting],model=m))
        for name,x,fields in [('linear',X,names),('original_only_linear',XO,orig_names)]:
            m=fit_linear(x,gain,fitting,2021+fold);s=predict_ridge(m,x);acts[name]=s>m['threshold'];quality[name]=s
            models.append(dict(scope=scope,fold=fold,policy=name,features=fields,training_ids=[labels[i]['id'] for i in fitting],model=m))
        shuffled=gain.copy();shuffled[fitting]=np.random.default_rng(2021+fold).permutation(shuffled[fitting])
        m=fit_linear(X,shuffled,fitting,4021+fold);s=predict_ridge(m,X);acts['shuffled_label_linear']=s>m['threshold'];quality['shuffled_label_linear']=s
        models.append(dict(scope=scope,fold=fold,policy='shuffled_label_linear',features=names,training_ids=[labels[i]['id'] for i in fitting],
                           permuted_training_targets=shuffled[fitting].tolist(),model=m))
        priors={k:float(gain[tr[groups[tr]==k]].mean()) for k in np.unique(groups[tr])}
        fallback=float(gain[tr].mean());acts['dataset_prior']=np.array([priors.get(k,fallback)>0 for k in groups])
        models.append(dict(scope=scope,fold=fold,policy='dataset_prior',training_ids=[labels[i]['id'] for i in tr],model=dict(priors=priors,fallback=fallback)))
        for name in ['confidence_gate','margin_entropy_gate','kl_gate','linear']:
            # Match both total and changed-answer acceptance counts within each task.
            # Using another policy's unlabeled acceptance count is transductive, not label leakage.
            accept=np.zeros(N,bool);rng=np.random.default_rng(6021+fold)
            for k in np.unique(groups[te]):
                for status in [False,True]:
                    subset=te[(groups[te]==k)&(changed[te]==status)];count=int(acts[name][subset].sum())
                    if count:accept[rng.choice(subset,count,replace=False)]=True
            acts['matched_random_'+name]=accept
        for policy,accept in acts.items():
            for i in te:
                # Preserve actual policy actions, including accuracy-neutral same answers.
                # Always Euler must have acceptance rate1; this is not a cost-saving claim.
                action=bool(accept[i]);l=labels[i]
                decisions.append(dict(scope=scope,fold=fold,id=l['id'],group=l['group'],policy=policy,
                    accept_euler=action,changed=bool(changed[i]),gain=int(gain[i]),
                    selected_prediction=l['euler_prediction'] if action else l['original_prediction'],
                    correct=l['original_correct']+int(gain[i])*action,
                    quality_score=float(quality[policy][i]) if policy in quality else None))
    run('discovery_fit',0,allindices,allindices)
    for fold,(tr,te) in enumerate(split,1):run('nested_cv',fold,tr,te)
    for fold,k in enumerate(sorted(set(groups)),11):
        run('cross_dataset',fold,allindices[groups!=k],allindices[groups==k])
    put('results/fitted_models_and_selection.json',models)
    with (ROOT/'results/policy_predictions.jsonl').open('w',encoding='utf8',newline='\n') as f:
        for row in decisions:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    # Frozen discovery-trained rules, explicitly not yet validated independently.
    selected=[x for x in models if x['scope']=='discovery_fit' and x['policy'] in ['confidence_gate','margin_entropy_gate','kl_gate','linear','original_only_linear']]
    put('configs/discovery_fitted_rules.json',dict(status='frozen_for_future_independent_test_not_yet_run',rules=selected,
           plan_sha256=digest(ROOT/'configs/analysis_plan.json'),features_sha256=digest(ROOT/'data/discovery_features.csv')))
    return decisions,models


def metrics(decisions,labels):
    frame=pd.DataFrame(decisions);lookup={l['id']:l for l in labels};summary=[];auc=[];randomization=[]
    for scope in ['discovery_fit','nested_cv','cross_dataset']:
        df=frame[frame.scope==scope]
        for subset in ['all','arc_challenge','openbookqa','mmlu']:
            sub=df if subset=='all' else df[df.group==subset]
            ids=[l['id'] for l in labels if subset=='all' or l['group']==subset];n=len(ids)
            ori=np.array([lookup[x]['original_correct'] for x in ids]);euler=np.array([lookup[x]['euler_correct'] for x in ids]);gain=euler-ori
            rng=np.random.default_rng(2021);boot=rng.integers(0,n,(10000,n))
            for policy in sorted(sub.policy.unique()):
                rows=sub[sub.policy==policy].set_index('id').loc[ids];assert len(rows)==n
                correct=rows.correct.to_numpy(int);accept=rows.accept_euler.to_numpy(bool);changed=rows.changed.to_numpy(bool)
                ben=gain==1;harm=gain==-1;retained=int(np.sum(ben&accept));avoided=int(np.sum(harm&~accept));rejected=int(np.sum(ben&~accept));accepted=int(np.sum(harm&accept))
                accboot=correct[boot].mean(1);row=dict(scope=scope,subset=subset,policy=policy,n=n,correct=int(correct.sum()),accuracy_percent=float(correct.mean()*100),
                    accuracy_ci_low=float(np.quantile(accboot,.025)*100),accuracy_ci_high=float(np.quantile(accboot,.975)*100),
                    wrong_to_correct=retained,correct_to_wrong=accepted,beneficial_retained=retained,harmful_avoided=avoided,
                    beneficial_rejected=rejected,harmful_accepted=accepted,beneficial_total=int(ben.sum()),harmful_total=int(harm.sum()),
                    euler_acceptance_rate_all=float(accept.mean()),euler_acceptance_rate_changed=float(accept[changed].mean()) if changed.any() else None,
                    prediction_disagreement_rate=float(changed.mean()),net_gain_over_euler=avoided-rejected)
                assert int(correct.sum()-euler.sum())==avoided-rejected
                for name,base in [('original',ori),('euler',euler)]:
                    d=correct-base;b=d[boot].mean(1)*100;w=int((d>0).sum());l=int((d<0).sum())
                    row.update({f'delta_{name}_pp':float(d.mean()*100),f'delta_{name}_ci_low':float(np.quantile(b,.025)),
                                f'delta_{name}_ci_high':float(np.quantile(b,.975)),f'mcnemar_{name}_p':float(binomtest(w,w+l,.5).pvalue) if w+l else 1.})
                summary.append(row)
                signal=rows.quality_score.to_numpy(float);selection=(gain!=0)&np.isfinite(signal)
                if selection.sum()==(gain!=0).sum() and np.unique(gain[selection]).size==2:
                    auc.append(dict(scope=scope,subset=subset,policy=policy,**auc_ci(signal[selection],(gain[selection]>0).astype(int))))
            for target in ['confidence_gate','margin_entropy_gate','kl_gate','linear']:
                t=sub[sub.policy==target].set_index('id').loc[ids];targetcorrect=int(t.correct.sum())
                # Exact sampling of beneficial/harmful/neutral categories at fixed unlabeled per-task accept counts.
                simulation=np.full(10000,int(ori.sum()),int);rr=np.random.default_rng(2021)
                for g in sorted(t.group.unique()):
                    positions=np.flatnonzero((t.group.to_numpy()==g)&t.changed.to_numpy(bool));k=int(t.accept_euler.to_numpy(bool)[positions].sum())
                    if not len(positions):continue
                    sizes=np.array([(gain[positions]==1).sum(),(gain[positions]==-1).sum(),(gain[positions]==0).sum()])
                    draws=rr.multivariate_hypergeometric(sizes,k,size=10000);simulation+=draws[:,0]-draws[:,1]
                randomization.append(dict(scope=scope,subset=subset,policy=target,randomizations=10000,
                    policy_correct=targetcorrect,random_mean_correct=float(simulation.mean()),random_accuracy_mean_percent=float(simulation.mean()/n*100),
                    random_accuracy_low_percent=float(np.quantile(simulation,.025)/n*100),random_accuracy_high_percent=float(np.quantile(simulation,.975)/n*100),
                    one_sided_randomization_p=float((1+np.sum(simulation>=targetcorrect))/10001)))
    write_csv('results/policy_summary.csv',summary);write_csv('results/predictor_predictability.csv',auc);write_csv('results/acceptance_matched_random.csv',randomization)
    # Fold scores come from distinct fitted functions. Report within-fold AUC as
    # a diagnostic rather than silently treating pooled rankings as calibrated.
    foldrows=[]
    for (scope,fold,policy),d in frame.groupby(['scope','fold','policy']):
        if scope=='discovery_fit':continue
        d=d[(d.gain!=0)&d.quality_score.notna()]
        if d.gain.nunique()!=2:continue
        y=(d.gain.to_numpy()>0).astype(int);s=d.quality_score.to_numpy(float)
        foldrows.append(dict(scope=scope,fold=int(fold),policy=policy,n=len(d),beneficial=int(y.sum()),harmful=int((1-y).sum()),
                             auroc=float(roc_auc_score(y,s)),auprc=float(average_precision_score(y,s))))
    write_csv('results/foldwise_predictability.csv',foldrows)
    return summary


def costs(F,labels,models):
    model=next(x['model'] for x in models if x['scope']=='discovery_fit' and x['policy']=='linear')
    X=np.array([[f[k] for k in PLAN['linear_features']] for f in F])
    for _ in range(50):predict_ridge(model,X)
    start=time.perf_counter()
    for _ in range(2000):predict_ridge(model,X)
    elapsed=time.perf_counter()-start
    t0=np.array([x['original_elapsed_s'] for x in labels]);te=np.array([x['euler_elapsed_s'] for x in labels])
    start=time.perf_counter()
    o=read(BASE/'results/qwen_A0_test.jsonl');e=read(BASE/'results/qwen_A3_test.jsonl');e={x['id']:x for x in e}
    for x in o:features(visible(x),visible(e[x['id']]))
    extract=time.perf_counter()-start
    report=dict(new_gpu_hours=0.,offline_selection_gpu_cost=0.,original_historical_total_s=float(t0.sum()),euler_historical_total_s=float(te.sum()),
                serial_two_output_total_s=float(t0.sum()+te.sum()),historical_serial_pair_vs_euler_ratio=float((t0.sum()+te.sum())/te.sum()),
                historical_original_mean_s=float(t0.mean()),historical_euler_mean_s=float(te.mean()),
                cpu_predictor_microseconds_per_question=elapsed/(2000*len(labels))*1e6,cpu_feature_extraction_microseconds_per_question=extract/len(labels)*1e6,
                cpu_platform=platform.platform(),cpu_python=sys.version,scope='CPU overhead measured live; model timings reused historical per-question records, not fresh online/shared-compute measurements',
                reuse='First natural window is mathematically reusable, but the pinned recipe recomputes each complete A0/A3 candidate separately; no shared-compute implementation benchmarked.',
                layer_evaluations_per_candidate_original=36,layer_evaluations_per_candidate_euler=56,layer_evaluations_per_candidate_serial_pair=92,
                caveat='Layer counts ignore sequence-length-dependent FLOPs, head, attention and feature logging. No acceleration claim.')
    put('results/COMPUTE_COST.json',report)


def main():
    start=time.time();print('Audit sealed originals',flush=True)
    F,labels=build_data();sanity(F);print('Features and univariate analysis',flush=True)
    univariate(F,labels);print('Nested and cross-task validation',flush=True)
    decisions,models=policies(F,labels);print('Paired statistics',flush=True)
    summary=metrics(decisions,labels);costs(F,labels,models)
    put('results/COMPLETION_AUDIT.json',dict(passed=True,discovery_questions=600,prediction_rows=len(decisions),
        scopes=['discovery_fit','nested_cv','cross_dataset'],independent_test_run=False,new_gpu_hours=0.,
        source_seal_after=seal_audit(),plan_sha256=digest(ROOT/'configs/analysis_plan.json'),
        elapsed_cpu_wall_s=time.time()-start,packages={k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','pandas','matplotlib']},
        scientific_status='Exploratory discovery and internal/cross-task validation complete. Independent confirmation remains deferred by user.'))
    for row in summary:
        if row['subset']=='all' and row['policy'] in ['always_original','always_euler','linear','original_only_linear','oracle']:
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
