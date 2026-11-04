"""CPU paired analysis and predeclared selection; no inference modifications."""
import argparse
import numpy as np
from r7_core import ROOT,read,sha,save,csvout,jsonlines

PRIMARY_CANDIDATES=['M1_multi_candidate','M2_N0_token','M3_multi_token']


def rows(split):
    path=f'results/r7_{split}_development.jsonl' if split in ['discovery','selection','confirmation'] else f'results/r7_{split}_final.jsonl'
    receipt=read(path.replace('.jsonl','_COMPLETE.json'))
    assert receipt['status']=='complete' and sha(path)==receipt['score_file_sha256']
    data=list(jsonlines(path));assert len(data)==receipt['n']==len({r['id'] for r in data})
    for row in data:
        assert row['signature']==receipt['signature']
        for name,scores in {**row['baseline_scores'],**row['method_scores']}.items():
            pred=int(np.argmax(scores));assert pred==row['predictions'][name] and row['correct'][name]==(pred==row['gold'])
    return data


def paired(data,name,reference='M0_R6',bootstrap=False):
    x=np.asarray([r['correct'][name] for r in data],dtype=np.int8)
    b=np.asarray([r['correct'][reference] for r in data],dtype=np.int8)
    fixed=np.asarray([r['correct']['fixed'] for r in data],dtype=np.int8)
    old_rescues=(b==1)&(fixed==0)
    rescue=int(np.sum((x==1)&(b==0)));damage=int(np.sum((x==0)&(b==1)))
    from scipy.stats import binomtest
    result=dict(method=name,n=len(data),correct=int(x.sum()),accuracy=float(x.mean()),
        reference=reference,reference_correct=int(b.sum()),rescue=rescue,damage=damage,net=rescue-damage,
        accuracy_delta_pp=float(100*(x-b).mean()),
        retained_R6_vs_Fixed_rescues=int(np.sum(old_rescues&(x==1))),
        total_R6_vs_Fixed_rescues=int(old_rescues.sum()),
        retained_R6_rescue_fraction=float(np.sum(old_rescues&(x==1))/old_rescues.sum()) if old_rescues.any() else None,
        exact_mcnemar_p=float(binomtest(rescue,rescue+damage,.5).pvalue) if rescue+damage else 1.,
        alignment_fallbacks=sum(r.get('alignment_fallback_by_method',{}).get(name,False) for r in data))
    if bootstrap:
        rng=np.random.default_rng(2021);delta=x-b;boot=[]
        for _ in range(20):boot.extend((100*delta[rng.integers(0,len(data),size=(1000,len(data)))].mean(axis=1)).tolist())
        lo,hi=np.quantile(boot,[.025,.975]);result.update(paired_bootstrap_95_lower_pp=float(lo),paired_bootstrap_95_upper_pp=float(hi),bootstrap_replicates=20000,bootstrap_seed=2021)
    return result


def development(splits):
    bysplit={s:rows(s) for s in splits};allrows=[r for v in bysplit.values() for r in v];table=[]
    for label,data in list(bysplit.items())+[('train1093' if len(splits)==3 else 'development576',allrows)]:
        for name in data[0]['predictions']:
            table.append(dict(split=label,**paired(data,name)))
    csvout('results/r7_development_results.csv',table)
    csvout('results/r7_ablation_results.csv',[r for r in table if r['method'].startswith('C_') or r['method'] in PRIMARY_CANDIDATES+['M0_R6']])
    return bysplit,table


def train_rank(bysplit):
    allrows=[r for values in bysplit.values() for r in values];reg=read('configs/r7_METHODS.json')['methods']
    return sorted(PRIMARY_CANDIDATES,key=lambda name:(
        -sum(r['correct'][name] for r in allrows),
        -min(np.mean([int(r['correct'][name])-int(r['correct']['M0_R6']) for r in values]) for values in bysplit.values()),
        sum(r['correct']['M0_R6'] and not r['correct'][name] for r in allrows),len(reg[name]['references']),name))


def validation_rank(data,train,candidates):
    reg=read('configs/r7_METHODS.json')['methods']
    return sorted(candidates,key=lambda name:(-sum(r['correct'][name] for r in data),
        -sum(r['correct'][name] for r in train),sum(r['correct']['M0_R6'] and not r['correct'][name] for r in data),
        len(reg[name]['references']),name))


def arc_report():
    data=rows('test');lock=read('configs/r7_FINAL_METHOD_LOCK.json');primary=lock['primary']
    expected={'native4':702,'fixed':717,'adaptive':716,'A1':726,'M0_R6':736}
    for name,n in expected.items():assert sum(r['correct'][name] for r in data)==n,('Historical baseline mismatch',name)
    assert len(data)==1172 and set(data[0]['method_scores'])=={primary}
    table=[paired(data,name,bootstrap=True) for name in data[0]['predictions']]
    csvout('results/r7_arc_full_test_results.csv',table)
    transitions=[]
    for row in data:
        for baseline in ['native4','fixed','adaptive','A1','M0_R6']:
            before=row['correct'][baseline];after=row['correct'][primary]
            transitions.append(dict(id=row['id'],gold=row['gold'],primary=primary,reference=baseline,
                before_prediction=row['predictions'][baseline],after_prediction=row['predictions'][primary],
                transition=('Correct' if before else 'Wrong')+'->'+('Correct' if after else 'Wrong'),
                R6_rescued_vs_Fixed=row['correct']['M0_R6'] and not row['correct']['fixed'],
                alignment_fallback=row['alignment_fallback_by_method'][primary]))
    csvout('results/r7_paired_transitions.csv',transitions)
    main=next(r for r in table if r['method']==primary)
    save('results/r7_ARC_COMPLETE.json',dict(status='complete',n=1172,correct=main['correct'],primary=primary,
        final_method_lock_sha256=sha('configs/r7_FINAL_METHOD_LOCK.json'),score_file_sha256=sha('results/r7_test_final.jsonl'),
        metrics=main,external_authorized=main['correct']>736,aspirational_target_reached=main['correct']>=751,
        no_post_test_reselection=True))
    return main


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['development','train','arc'],required=True);a=parser.parse_args()
    print(arc_report() if a.stage=='arc' else development(['discovery','selection']+(['confirmation'] if a.stage=='train' else []))[1])
