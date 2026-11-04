"""Read-only complete-record verification with independent DP permutation tails."""
import csv,math
from collections import defaultdict
import numpy as np
from r15_common import ROOT,LOCK,read,sha,digest,jsonlines,gate


def subset_counts(values,k):
    ways=[defaultdict(int) for _ in range(k+1)];ways[0][0]=1
    for seen,value in enumerate(values):
        for size in range(min(k,seen+1),0,-1):
            for previous,count in list(ways[size-1].items()):ways[size][previous+int(value)]+=count
    assert sum(ways[k].values())==math.comb(len(values),k)
    return ways[k]


def independent_stat(score,positive):
    s=np.asarray(score);p=np.asarray(positive,dtype=bool);rank2=[1+2*int(np.sum(s<x))+int(np.sum(s==x)) for x in s]
    observed=sum(v for v,yes in zip(rank2,p) if yes);ways=subset_counts(rank2,int(p.sum()))
    tail=sum(c for value,c in ways.items() if value>=observed)/sum(ways.values())
    pairs=[1. if x>y else .5 if x==y else 0. for x in s[p] for y in s[~p]]
    return math.fsum(pairs)/len(pairs),tail


def main():
    spec=gate();receipt=read('results/r15_context_COMPLETE.json');analysis=read('results/r15_ASSOCIATION_COMPLETE.json');budget=read('logs/r15_gpu_budget.json');state=read('logs/r15_probe_SUPERVISOR.json')
    assert receipt['status']=='complete' and receipt['n']==58 and receipt['unsupported_questions']==0 and state['status']=='complete' and state['exit_code']==0
    assert receipt['protocol_sha256']==analysis['protocol_sha256']==sha(LOCK) and sha(receipt['score_file'])==receipt['score_file_sha256']
    assert sha('results/r15_question_features.csv')==analysis['question_features_sha256'] and receipt['head_norm_before']==receipt['head_norm_after']
    assert not receipt['weights_modified'] and receipt['main_transformer_forwards']==0 and not receipt['KV_cache'] and not receipt['attention_mask_argument']
    raw=list(jsonlines(receipt['score_file']));assert [r['id'] for r in raw]==[r['id'] for r in spec['cohort']] and len(raw)==58
    authority={};ordered=[]
    for split in ['discovery','selection']:
        for row in jsonlines('results/r7_'+split+'_development.jsonl'):authority[row['id']]=row;ordered.append((split,row))
    D=[(s,r) for s,r in ordered if r['predictions']['M0_R6']!=r['predictions']['M2_N0_token']];C=[(s,r) for s,r in ordered if r['predictions']['M0_R6']==r['predictions']['M2_N0_token']]
    rng=np.random.default_rng(2021);selected=sorted(rng.choice(len(C),len(D),replace=False).tolist());cohort=[]
    for group,rows in [('disagreement',D),('concordant',[C[i] for i in selected])]:
        for split,row in rows:
            a=row['predictions']['M0_R6'];b=row['predictions']['M2_N0_token'] if group=='disagreement' else max((j for j in range(len(row['baseline_scores']['M0_R6'])) if j!=a),key=lambda j:row['baseline_scores']['M0_R6'][j])
            cohort.append(dict(id=row['id'],split=split,group=group,old_candidate=a,probe_candidate=b))
    cohort=[cohort[i] for i in rng.permutation(len(cohort))];assert cohort==spec['cohort']
    with (ROOT/'results/r15_question_features.csv').open(newline='',encoding='utf8') as f:csvrows=list(csv.DictReader(f))
    assert len(csvrows)==58;values=[];positions=0;zero_positions=0
    signature=digest(dict(protocol_sha256=sha(LOCK),cohort=cohort))
    for row,chosen,stored in zip(raw,cohort,csvrows):
        assert row['signature']==signature and row['cohort_entry_sha256']==digest(chosen) and row['id']==stored['id']
        old=authority[row['id']];candidates={v['choice']:v for v in row['candidates']};feature={}
        for c in [chosen['old_candidate'],chosen['probe_candidate']]:
            rec=candidates[c];m=rec['metrics'];assert rec['alignment_supported'] and rec['max_parity_error']==rec['max_alpha_error']==0
            assert rec['characters']==old['choice_char_lengths'][c] and rec['tokens']==old['token_counts'][c]
            assert all(len(v)==rec['tokens'] and all(math.isfinite(x) for x in v) for v in m.values())
            assert m['alpha']==old['token_diagnostics'][c]['M2_N0_token']['alpha'] and m['correction_energy']==old['token_diagnostics'][c]['M2_N0_token']['projected_correction_norm2']
            assert all(x>=-1e-12 for x in m['repair_KL'])
            for a,x,y in zip(m['alpha'],m['change_N1'],m['change_N2']):
                if a==0:assert x==y==0.;zero_positions+=1
            positions+=rec['tokens'];length=rec['characters']
            feature[c]=dict(primary=math.fsum(min(x,y) for x,y in zip(m['change_N1'],m['change_N2']))/length,
                N1=math.fsum(m['change_N1'])/length,N2=math.fsum(m['change_N2'])/length,
                static=math.fsum(min(x,y) for x,y in zip(m['static_N1'],m['static_N2']))/length,energy=math.fsum(m['correction_energy'])/length)
        a,b=chosen['old_candidate'],chosen['probe_candidate'];v={k:feature[b][k]-feature[a][k] for k in feature[a]};v['old_margin']=old['baseline_scores']['M0_R6'][b]-old['baseline_scores']['M0_R6'][a]
        for k,x in v.items():assert float(stored[k])==x
        v['delta']=int(old['correct']['M2_N0_token'])-int(old['correct']['M0_R6']) if chosen['group']=='disagreement' else None;v['group']=chosen['group'];values.append(v)
    disagreements=[r for r in values if r['group']=='disagreement'];informative=[r for r in disagreements if r['delta']!=0];labels=np.asarray([r['delta']==1 for r in informative]);assert len(informative)==22 and labels.sum()==7
    names=['primary','static','energy','old_margin'];scores=np.asarray([[r[name] for r in informative] for name in names]);verified={}
    for name,s in zip(names,scores):
        AUC,p=independent_stat(s,labels);assert abs(AUC-analysis['metrics'][name]['AUC'])<1e-15 and p==analysis['metrics'][name]['one_sided_exact_label_permutation_p'];verified[name]=dict(AUC=AUC,independent_DP_exact_p=p)
    rng=np.random.default_rng(2021);positives=np.flatnonzero(labels);negatives=np.flatnonzero(~labels);bootstrap=[]
    for _ in range(5000):
        p=rng.choice(positives,len(positives),replace=True);n=rng.choice(negatives,len(negatives),replace=True)
        left=scores[:,p,None];right=scores[:,None,n];bootstrap.append(((left>right).astype(float)+.5*(left==right)).mean(axis=(1,2)))
    bootstrap=np.asarray(bootstrap);assert np.array_equal(np.quantile(bootstrap[:,0],[.025,.975]),analysis['primary_stratified_bootstrap_95_CI'])
    for i,name in enumerate(names[1:],1):assert np.array_equal(np.quantile(bootstrap[:,0]-bootstrap[:,i],[.025,.975]),analysis['paired_AUC_differences'][name]['paired_bootstrap_95_CI'])
    selected=[r['primary']>0 for r in disagreements];observed=sum(r['delta'] for r,s in zip(disagreements,selected) if s);ways=subset_counts([r['delta'] for r in disagreements],sum(selected))
    coverage_p=sum(v for score,v in ways.items() if score>=observed)/sum(ways.values());assert coverage_p==analysis['same_coverage_null']['exact_same_coverage_random_tail_p'] and observed==analysis['same_coverage_null']['observed_net']
    inherited=0.
    for p,h in spec['budget']['inherited_ledger_sha256'].items():assert sha(p)==h;inherited+=read(p)['used_seconds']
    assert len(budget['runs'])==1 and budget['runs'][0]==state and state['worker_source_sha256']==spec['source_sha256']['r15_context_worker.py']
    assert budget['used_seconds']==state['gpu_worker_seconds'] and abs(budget['total_used_seconds']-inherited-budget['used_seconds'])<1e-6 and budget['total_used_seconds']<=21600
    assert not analysis['predefined_association_screen_passed'] and not analysis['incremental_over_static_and_energy_screen_passed']
    assert not analysis['heldout_accessed'] and not analysis['method_selected'] and not analysis['controller_trained']
    return dict(status='verified_complete_observational_probe',n=58,informative_questions=22,candidate_prefix_positions=positions,zero_alpha_positions_exact_zero_context_change=zero_positions,
        independent_statistics=verified,all_bootstrap_intervals_exact_replay=True,independent_same_coverage_DP_p=coverage_p,
        GPU_worker_seconds=budget['used_seconds'],total_original_GPU_worker_seconds=budget['total_used_seconds'],remaining_GPU_worker_seconds=21600-budget['total_used_seconds'],
        heldout_accessed=False,new_method_selected=False,goal_success_declared=False)


if __name__=='__main__':
    import json
    print(json.dumps(main(),indent=2))
