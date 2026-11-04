"""One frozen association analysis, never a fitted gate or heldout method."""
import csv,itertools,math
from collections import Counter
import numpy as np
from r15_common import ROOT,LOCK,read,sha,digest,jsonlines,save,gate


def auc(score,positive):
    p=np.asarray(score)[np.asarray(positive,dtype=bool)];n=np.asarray(score)[~np.asarray(positive,dtype=bool)]
    assert len(p) and len(n)
    return float(((p[:,None]>n[None]).astype(float)+.5*(p[:,None]==n[None])).mean())


def exact_auc_tail(score,positive):
    v=np.asarray(score);labels=np.asarray(positive,dtype=bool);order=np.argsort(v,kind='stable');rank2=np.empty(len(v),dtype=int);i=0
    while i<len(v):
        j=i+1
        while j<len(v) and v[order[j]]==v[order[i]]:j+=1
        rank2[order[i:j]]=i+1+j;i=j
    observed=int(rank2[labels].sum());hit=0;total=0
    for assignment in itertools.combinations(range(len(v)),int(labels.sum())):
        hit+=sum(int(rank2[i]) for i in assignment)>=observed;total+=1
    assert total==math.comb(len(v),int(labels.sum()))
    return dict(AUC=auc(v,labels),one_sided_exact_label_permutation_p=hit/total,assignments=total,positive_means_rescue=True,ties_count_half=True)


def coverage_null(deltas,selected):
    counts=Counter(deltas);k=sum(selected);n=len(deltas);den=math.comb(n,k);mass=Counter()
    for r in range(counts[1]+1):
        for d in range(counts[-1]+1):
            z=k-r-d
            if 0<=z<=counts[0]:mass[r-d]+=math.comb(counts[1],r)*math.comb(counts[-1],d)*math.comb(counts[0],z)
    assert sum(mass.values())==den;observed=sum(d for d,s in zip(deltas,selected) if s)
    return dict(approved=k,disagreements=n,observed_net=observed,exact_same_coverage_random_tail_p=sum(v for value,v in mass.items() if value>=observed)/den,
        distribution=[dict(net=value,probability=value_count/den) for value,value_count in sorted(mass.items())],scope='Diagnostic zero-sign partition, not a released routing method')


def main():
    spec=gate();receipt=read('results/r15_context_COMPLETE.json');supervisor=read('logs/r15_probe_SUPERVISOR.json')
    assert supervisor['status']=='complete' and receipt['status']=='complete' and receipt['n']==58 and receipt['unsupported_questions']==0,'No partial/alignment-filtered feature analysis'
    assert receipt['protocol_sha256']==sha(LOCK) and sha(receipt['score_file'])==receipt['score_file_sha256'] and receipt['head_norm_before']==receipt['head_norm_after']
    assert not receipt['weights_modified'] and not receipt['heldout_accessed']
    records=list(jsonlines(receipt['score_file']));assert len(records)==len({r['id'] for r in records})==58
    expected={r['id']:r for r in spec['cohort']};assert set(expected)=={r['id'] for r in records}
    authority={}
    for split in ['discovery','selection']:
        for row in jsonlines('results/r7_'+split+'_development.jsonl'):authority[row['id']]=row
    signature=digest(dict(protocol_sha256=sha(LOCK),cohort=spec['cohort']));items=[]
    for row in records:
        chosen=expected[row['id']];old=authority[row['id']];assert row['signature']==signature and row['cohort_entry_sha256']==digest(chosen)
        assert row['old_candidate']==chosen['old_candidate'] and row['probe_candidate']==chosen['probe_candidate']
        by={r['choice']:r for r in row['candidates']};a,b=row['old_candidate'],row['probe_candidate'];feature={}
        for index in [a,b]:
            c=by[index];assert c['alignment_supported'] and c['max_parity_error']==c['max_alpha_error']==0.
            m=c['metrics'];assert all(len(v)==c['tokens'] for v in m.values())
            feature[index]=dict(primary=math.fsum(min(x,y) for x,y in zip(m['change_N1'],m['change_N2']))/c['characters'],
                N1=math.fsum(m['change_N1'])/c['characters'],N2=math.fsum(m['change_N2'])/c['characters'],
                static=math.fsum(min(x,y) for x,y in zip(m['static_N1'],m['static_N2']))/c['characters'],
                energy=math.fsum(m['correction_energy'])/c['characters'])
        values={name:feature[b][name]-feature[a][name] for name in ['primary','N1','N2','static','energy']}
        values['old_margin']=old['baseline_scores']['M0_R6'][b]-old['baseline_scores']['M0_R6'][a]
        assert all(math.isfinite(v) for v in values.values())
        if row['group']=='disagreement':
            delta=int(old['correct']['M2_N0_token'])-int(old['correct']['M0_R6']);outcome='rescue' if delta==1 else 'damage' if delta==-1 else 'both_wrong'
        else:delta=None;outcome='concordant_background'
        items.append(dict(id=row['id'],group=row['group'],outcome=outcome,correctness_delta=delta,old_candidate=a,probe_candidate=b,**values))
    D=[r for r in items if r['group']=='disagreement'];assert len(D)==29 and Counter(r['outcome'] for r in D)==dict(rescue=7,damage=15,both_wrong=7)
    informative=[r for r in D if r['outcome']!='both_wrong'];labels=[r['outcome']=='rescue' for r in informative]
    metrics={name:exact_auc_tail([r[name] for r in informative],labels) for name in ['primary','static','energy','old_margin']}
    assert metrics['primary']['assignments']==170544
    # Stratified paired question resampling; identical samples for every feature.
    rng=np.random.default_rng(2021);p=np.flatnonzero(labels);n=np.flatnonzero(~np.asarray(labels));replicates={k:[] for k in metrics}
    for _ in range(5000):
        ids=np.r_[rng.choice(p,len(p),replace=True),rng.choice(n,len(n),replace=True)];label=np.r_[np.ones(len(p),dtype=bool),np.zeros(len(n),dtype=bool)]
        for name in metrics:replicates[name].append(auc([informative[i][name] for i in ids],label))
    differences={name:dict(point=metrics['primary']['AUC']-metrics[name]['AUC'],paired_bootstrap_95_CI=np.quantile(np.asarray(replicates['primary'])-replicates[name],[.025,.975]).tolist()) for name in ['static','energy','old_margin']}
    primary_CI=np.quantile(replicates['primary'],[.025,.975]).tolist();selected=[r['primary']>0 for r in D];null=coverage_null([r['correctness_delta'] for r in D],selected)
    sign_groups={}
    for label,predicate in [('positive',lambda r:r['primary']>0),('nonpositive',lambda r:r['primary']<=0)]:
        subset=[r for r in D if predicate(r)];sign_groups[label]=dict(n=len(subset),outcomes=dict(Counter(r['outcome'] for r in subset)),net=sum(r['correctness_delta'] for r in subset))
    with (ROOT/'results/r15_question_features.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    association=metrics['primary']['AUC']>=.75 and metrics['primary']['one_sided_exact_label_permutation_p']<.05 and null['observed_net']>0 and null['exact_same_coverage_random_tail_p']<.05
    incremental=association and all(differences[k]['paired_bootstrap_95_CI'][0]>0 for k in ['static','energy'])
    budget=read('logs/r15_gpu_budget.json');assert budget['total_used_seconds']<=21600
    save('results/r15_ASSOCIATION_COMPLETE.json',dict(status='complete_frozen_development_association',n=58,disagreements=29,informative_questions=22,protocol_sha256=sha(LOCK),
        question_features_sha256=sha('results/r15_question_features.csv'),metrics=metrics,primary_stratified_bootstrap_95_CI=primary_CI,
        paired_AUC_differences=differences,bootstrap_samples=5000,bootstrap_seed=2021,diagnostic_sign_groups=sign_groups,same_coverage_null=null,
        predefined_association_screen_passed=association,incremental_over_static_and_energy_screen_passed=incremental,
        reference_sign_agreement=sum((r['N1']>0)==(r['N2']>0) for r in D),
        heldout_accessed=False,method_selected=False,controller_trained=False,GPU_worker_seconds=budget['used_seconds'],all_original_GPU_worker_seconds=budget['total_used_seconds'],
        interpretation='Posthoc association in22 reused development questions. Forward target-free but candidate-prefix-conditioned. Measures M2-vs-Fixed context preference, not truth or mutual information; R6 undo confound remains. Controls are not winners. No new gate fitting or heldout execution authorized.'))
    print('R15_COMPLETE_ASSOCIATION',metrics['primary'],association,incremental,flush=True)


if __name__=='__main__':main()
