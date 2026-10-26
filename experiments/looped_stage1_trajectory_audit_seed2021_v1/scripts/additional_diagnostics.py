import json
import numpy as np
import pandas as pd
from analyze import ROOT,read,wilson,trajectory_pattern_masks

def auc(y,score):
    from scipy.stats import rankdata
    y=np.asarray(y,dtype=bool);score=np.asarray(score)
    npos=y.sum();nneg=len(y)-npos
    if npos==0 or nneg==0:return None
    ranks=rankdata(score)
    return float((ranks[y].sum()-npos*(npos+1)/2)/(npos*nneg))

def main():
    val=read('ouro_validation.jsonl');test=read('ouro_test.jsonl');summary=[]
    # Descriptive strata only: they do not select deployment thresholds.
    strata=[]
    for variable,values in [('initial_vocab_confidence',[r['trajectory'][0]['confidence'] for r in test]),('prompt_tokens',[r['prompt_tokens'] for r in test])]:
        bins=pd.qcut(values,5,duplicates='drop')
        for interval in bins.categories:
            subset=[r for r,b in zip(test,bins) if b==interval]
            masks=trajectory_pattern_masks([[s['correct'] for s in r['trajectory']] for r in subset])
            for pattern,mask in masks.items():
                lo,hi=wilson(int(mask.sum()),len(subset))
                strata.append(dict(variable=variable,interval=str(interval),pattern=pattern,n=len(subset),count=int(mask.sum()),rate=float(mask.mean()),ci_low=lo,ci_high=hi,scope='post-hoc descriptive strata; no deployment selection'))
    pd.DataFrame(strata).to_csv(ROOT/'results/trajectory_strata.csv',index=False)
    for depth in [1,2,3]:
        for metric in ['full_vocab_kl','choice_kl','hidden_relative_l2','choice_confidence','choice_margin','choice_logit_margin']:
            vv=[r['trajectory'][depth-1][metric] for r in val if r['trajectory'][depth-1][metric] is not None]
            if not vv:continue
            low=float(np.quantile(vv,.2));high=float(np.quantile(vv,.8))
            for group in ['all','arc_challenge','openbookqa','mmlu']:
                rows=[r for r in test if group=='all' or r['task']==group or group=='mmlu' and r['task'].startswith('mmlu_')]
                current=[r['trajectory'][depth-1] for r in rows]
                correction=np.array([not t['correct'] and r['trajectory'][-1]['correct'] for t,r in zip(current,rows)])
                degradation=np.array([t['correct'] and not r['trajectory'][-1]['correct'] for t,r in zip(current,rows)])
                score=np.array([t[metric] for t in current]);wrong=np.array([not t['correct'] for t in current]);correct=~wrong
                stable=score<=low if metric in ['full_vocab_kl','choice_kl','hidden_relative_l2'] else score>=high
                lo,hi=wilson(int((stable&correction).sum()),int(stable.sum()))
                signed_score=-score if metric in ['choice_confidence','choice_margin','choice_logit_margin'] else score
                summary.append(dict(depth=depth,task=group,metric=metric,validation_low_threshold=low,validation_high_threshold=high,
                                    n=len(rows),validation_quantile_group_count=int(stable.sum()),quantile_group_final_corrections=int((stable&correction).sum()),quantile_group_final_degradations=int((stable&degradation).sum()),
                                    quantile_group_correction_rate_ci_low=lo,quantile_group_correction_rate_ci_high=hi,
                                    auc_future_correction_among_current_wrong=auc(correction[wrong],signed_score[wrong]),
                                    auc_future_degradation_among_current_correct=auc(degradation[correct],signed_score[correct])))
    pd.DataFrame(summary).to_csv(ROOT/'results/continuation_predictability.csv',index=False)
    absolute=[]
    for depth in [2,3]:
        for metric,cuts in [('full_vocab_kl',[.001,.01,.05,.1]),('choice_kl',[.0001,.001,.01]),('hidden_relative_l2',[.01,.05,.1])]:
            for cut in cuts:
                subset=[r for r in test if r['trajectory'][depth-1][metric]<=cut]
                corrections=sum(not r['trajectory'][depth-1]['correct'] and r['trajectory'][-1]['correct'] for r in subset)
                degradations=sum(r['trajectory'][depth-1]['correct'] and not r['trajectory'][-1]['correct'] for r in subset)
                lo,hi=wilson(corrections,len(subset))
                absolute.append(dict(depth=depth,metric=metric,absolute_cut=cut,n=len(subset),final_corrections=corrections,final_degradations=degradations,correction_rate_ci_low=lo,correction_rate_ci_high=hi,scope='post-hoc descriptive audit; never used to change a deployable policy'))
    pd.DataFrame(absolute).to_csv(ROOT/'results/absolute_stability_audit.csv',index=False)
    # Fair compute control: random assignment with the exact same chosen exit-depth
    # histogram. It is a descriptive comparator, never a label-based deployment rule.
    policies=pd.read_csv(ROOT/'results/halting_selected.csv');from analyze import policy
    matched=[];rng=np.random.default_rng(2021)
    for _,row in policies.iterrows():
        result,indices=policy(test,row['method'],float(row['threshold']))
        values=[]
        for _ in range(10000):
            shuffled=rng.permutation(indices)
            values.append(np.mean([r['trajectory'][i]['correct'] for r,i in zip(test,shuffled)]))
        matched.append(dict(method=row['method'],target_validation_depth=row['target_validation_depth'],average_depth=result['average_depth'],accuracy=result['accuracy'],
                            matched_random_mean=float(np.mean(values)),matched_random_95_low=float(np.quantile(values,.025)),matched_random_95_high=float(np.quantile(values,.975))))
    pd.DataFrame(matched).to_csv(ROOT/'results/halting_matched_random.csv',index=False)

if __name__=='__main__':main()
