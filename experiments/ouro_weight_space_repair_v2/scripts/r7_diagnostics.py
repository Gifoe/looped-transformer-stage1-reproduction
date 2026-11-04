"""Development-only baseline, path oracle and fixed-prompt choice perturbations."""
import itertools
import json
import numpy as np
from r7_core import ROOT,BASE_COMMIT,BASE_METHODS,development_baselines,save,sha,csvout
from slcd_r6_scores import projection_parameters


def path_oracle(a,e,gold):
    """Finite breakpoints plus open intervals cover all top1 choices on [0,1]."""
    points={0.,1.}
    for i,j in itertools.combinations(range(len(a)),2):
        den=e[i]-e[j]
        if den:
            x=float((a[i]-a[j])/den)
            if 0<=x<=1:points.add(x)
    points=sorted(points);probes=points+[(x+y)/2 for x,y in zip(points,points[1:]) if x<y]
    hits=[x for x in probes if int((a-x*e).argmax())==gold]
    return bool(hits),min(hits) if hits else None,len(points)


def main():
    data=development_baselines();summaries=[];oracle=[];mutations=[];predictions=[]
    rng=np.random.default_rng(2021)
    for row in data:
        item=row['item'];gold=item['gold'];scores=row['baseline'];a=scores['A1'];g=scores['fixed']-scores['native4']
        e=np.asarray(row['prior']['loopcd_k4_fixed']['normalized_scores'])-np.asarray(row['prior']['native_k4']['normalized_scores'])
        alpha,_,_,ec=projection_parameters(g,e);den=float(np.dot(ec,ec))
        possible,first,breaks=path_oracle(a,e,gold)
        old=int(a.argmax());new=int(scores['M0_R6'].argmax())
        oracle.append(dict(split=row['split'],id=item['id'],gold=gold,alpha=alpha,denominator=den,
                           alpha_region='zero' if alpha==0 else 'one' if alpha==1 else 'interior',
                           A1_correct=old==gold,R6_correct=new==gold,oracle_path_reachable=possible,
                           first_oracle_probe=first,number_breakpoints=breaks,
                           limitation='coefficient' if new!=gold and possible else 'direction' if new!=gold else 'already_correct'))
        for name,value in scores.items():predictions.append(dict(split=row['split'],id=item['id'],method=name,gold=gold,prediction=int(value.argmax())))
        def altered(indices,label):
            index=np.asarray(indices,dtype=int);ga=g[index];ea=e[index];aa=a[index]
            al,_,_,_=projection_parameters(ga,ea);out=aa-al*ea
            originals=list(dict.fromkeys(int(i) for i in index));positions=[list(index).index(i) for i in originals]
            delta=out[positions]-scores['M0_R6'][originals];centered=delta-delta.mean()
            restricted=originals[int(scores['M0_R6'][originals].argmax())]
            mutations.append(dict(split=row['split'],id=item['id'],perturbation=label,indices=json.dumps(index.tolist()),
                                  alpha_original=alpha,alpha_perturbed=al,max_shared_score_difference=float(np.max(np.abs(delta))),
                                  max_shared_centered_difference=float(np.max(np.abs(centered))),
                                  original_choice_prediction=new,perturbed_choice_prediction=int(index[int(out.argmax())]),
                                  original_winner_present=new in originals,baseline_best_among_shared=restricted,
                                  shared_ranking_winner_changed=restricted!=int(index[int(out.argmax())]),
                                  unchanged_prompt_sha256=item['prompt_sha256']))
        n=len(a);altered(rng.permutation(n),'permute')
        for i in range(n):
            altered(list(range(n))+[i],'duplicate_'+str(i))
            if n>2:altered([j for j in range(n) if j!=i],'delete_'+str(i))
    for split in ['discovery','selection','all_development']:
        rs=[r for r in data if split=='all_development' or r['split']==split]
        for method in BASE_METHODS:
            correct=[int(r['baseline'][method].argmax())==r['item']['gold'] for r in rs]
            base=[int(r['baseline']['M0_R6'].argmax())==r['item']['gold'] for r in rs]
            summaries.append(dict(split=split,method=method,n=len(rs),correct=sum(correct),accuracy=sum(correct)/len(rs),
                                  rescue_vs_R6=sum(a and not b for a,b in zip(correct,base)),damage_vs_R6=sum(b and not a for a,b in zip(correct,base))))
    csvout('results/r7_baseline_reconstruction.csv',summaries)
    csvout('results/r7_oracle_alpha_diagnostics.csv',oracle);csvout('results/r7_choice_perturbations.csv',mutations)
    csvout('results/r7_baseline_dev_predictions.csv',predictions)
    summary=dict(status='passed_cpu_reconstruction',base_commit=BASE_COMMIT,n=576,
        baselines=summaries,confirmation_rows_parsed=0,test_rows_parsed=0,
        oracle_summary={s:{key:sum(x['split']==s and x['limitation']==key for x in oracle) for key in ['already_correct','coefficient','direction']} for s in ['discovery','selection']},
        choice_perturbation_summary={kind:dict(n=sum(x['perturbation'].startswith(kind) for x in mutations),
                    changed_shared_centered_scores=sum(x['perturbation'].startswith(kind) and x['max_shared_centered_difference']>1e-12 for x in mutations),
                    changed_prediction=sum(x['perturbation'].startswith(kind) and x['original_choice_prediction']!=x['perturbed_choice_prediction'] for x in mutations),
                    changed_shared_ranking_winner=sum(x['perturbation'].startswith(kind) and x['shared_ranking_winner_changed'] for x in mutations),
                    changed_prediction_with_original_winner_present=sum(x['perturbation'].startswith(kind) and x['original_winner_present'] and x['original_choice_prediction']!=x['perturbed_choice_prediction'] for x in mutations)) for kind in ['permute','duplicate','delete']},
        historical_train_1093_R6_correct=705,historical_is_new_independent_evidence=False,
        gpu_parity_status='pending',source_sha256={p:sha('scripts/'+p) for p in ['r7_core.py','r7_diagnostics.py','slcd_r6_scores.py']})
    save('results/r7_baseline_parity.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='baselines'},indent=2));print(json.dumps([x for x in summaries if x['split']=='all_development']))


if __name__=='__main__':main()
