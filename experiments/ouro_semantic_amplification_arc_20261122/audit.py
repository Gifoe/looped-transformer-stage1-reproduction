"""Join published reference scores to the frozen run and audit complete ARC outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from arc_features import archive, sha
from evaluate import from_jsonl


SEED=261122722
METHODS=('magnitude','direction','completion','shuffled')
DEPTHS=(1,2,3,4)
METRICS=('acc','acc_norm')


def winners(rows,arm,metric):
    out=[]
    for row in rows:
        scores=np.asarray(row['scores'][arm],dtype=np.float64)
        assert len(scores)==len(row['labels']) and np.isfinite(scores).all()
        if metric=='acc_norm':
            scores=scores/row['characters']
        out.append(int(scores.argmax()))
    return np.asarray(out)


def paired(a,b):
    n=len(a)
    fixes=int((~a&b).sum())
    harms=int((a&~b).sum())
    draws=np.random.default_rng(SEED).multinomial(n,[harms/n,1-(fixes+harms)/n,fixes/n],size=20000)
    ci=np.quantile((draws[:,2]-draws[:,0])/n*100,[.025,.975])
    return dict(rescued=fixes,damaged=harms,delta_pp=100*(fixes-harms)/n,ci95_pp=ci.tolist(),
                mcnemar_p=float(binomtest(fixes,fixes+harms).pvalue) if fixes+harms else 1.)


def holm(stats):
    arr=sorted(stats,key=lambda x:x['mcnemar_p'])
    previous=0.
    for i,row in enumerate(arr):
        previous=max(previous,min(1.,(len(arr)-i)*row['mcnemar_p']))
        row['holm_four_depth_p']=previous


def reference(rows,old_rows,files,selected,n):
    combined=[]
    for a,b in zip(rows,old_rows,strict=True):
        assert all(a[k]==b[k] for k in ('id','labels','gold','answer_tokens','characters','scored_input_tokens'))
        assert not set(a['scores']).intersection(b['scores'])
        combined.append(dict(a,scores={**b['scores'],**a['scores']}))
    assert len(combined)==len(set(x['id'] for x in combined))==n
    gold=np.asarray([r['gold'] for r in combined])
    comparison, by_metric, choices={}, {}, {}
    for metric in METRICS:
        arms=tuple(combined[0]['scores'])
        choices[metric]={a:winners(combined,a,metric) for a in arms}
        correct={a:v==gold for a,v in choices[metric].items()}
        counts={a:int(v.sum()) for a,v in correct.items()}
        past=json.loads((files/f'{"test" if n==1172 else "validation"}_metrics.json').read_text())
        assert all(counts[a]==past[metric][a]['correct'] for a in old_rows[0]['scores'])
        refs={f'k{k}':max((a for a in old_rows[0]['scores'] if a.startswith('loopcd_') and a.endswith(f'k{k}')),
                        key=lambda a:(counts[a],a)) for k in DEPTHS if k>1}
        output={}
        for a in arms:
            depth=a.rsplit('_k',1)[-1]
            native=f'native_k{depth}'
            different={'vs_native':paired(correct[native],correct[a])}
            if depth!='1':
                different['vs_loopcd_historical_best']=paired(correct[refs[f'k{depth}']],correct[a])
                different['vs_loopcd_validation_locked']=paired(correct[selected[f'k{depth}']],correct[a])
            if a not in old_rows[0]['scores'] and int(depth)<4:
                higher=[m for m in old_rows[0]['scores'] if m.endswith(f'k{int(depth)+1}')]
                prior=max(higher,key=lambda m:(counts[m],m))
                different['vs_historical_next_depth_best']=dict(reference=prior,**paired(correct[prior],correct[a]))
            output[a]=dict(correct=counts[a],pct=100*counts[a]/n,**different)
        experiments={'direction_vs_magnitude':('magnitude','direction'),
                     'completion_vs_direction':('direction','completion'),
                     'completion_vs_shuffled':('shuffled','completion'),
                     'direction_vs_same_magnitude':('direction_magnitude_only','direction'),
                     'completion_vs_same_magnitude':('completion_magnitude_only','completion'),
                     'direction_vs_no_magnitude':('direction_direction_only','direction'),
                     'completion_vs_no_magnitude':('completion_direction_only','completion'),
                     'direction_vs_rotated_same_geometry':('direction_rotated','direction'),
                     'completion_vs_rotated_same_geometry':('completion_rotated','completion')}
        comparisons={key:{f'k{k}':paired(correct[f'{a}_k{k}'],correct[f'{b}_k{k}']) for k in DEPTHS}
                     for key,(a,b) in experiments.items()}
        for grouping in comparisons.values():
            holm(list(grouping.values()))
        for a in METHODS:
            holm([output[f'{a}_k{k}']['vs_native'] for k in DEPTHS])
        comparison[metric]=comparisons
        by_metric[metric]=dict(methods=output,loopcd_historical_best=refs,loopcd_validation_locked=selected)
    per_question=[]
    for i,r in enumerate(combined):
        per_question.append(dict(id=r['id'],gold=r['gold'],labels=r['labels'],
                                 **{m:{a:int(v[i]) for a,v in choices[m].items()} for m in METRICS}))
    return dict(n=n,metrics=by_metric,ablation=comparison),per_question


def validation_pairs(out,old):
    gold=np.asarray([r['gold'] for r in old])
    table={}
    for name in METHODS:
        rows=from_jsonl(out/f'VALIDATION_{name}.jsonl')
        assert len(rows)==299
        assert all(all(a[k]==b[k] for k in ('id','labels','gold','characters')) for a,b in zip(rows,old,strict=True))
        table[name]={m:{f'k{k}':paired(winners(old,f'native_k{k}',m)==gold,winners(rows,f'k{k}',m)==gold)
                        for k in DEPTHS} for m in METRICS}
    return table


def failure_summary(rows,paired_rows,official):
    out={}
    for arm in METHODS:
        for k in DEPTHS:
            name=f'{arm}_k{k}'
            native=f'native_k{k}'
            strata={a:[] for a in ('rescued','damaged','both_correct','both_wrong')}
            examples={a:[] for a in ('rescued','damaged')}
            for x,p,q in zip(rows,paired_rows,official,strict=True):
                assert x['id']==p['id']==q['id']
                c,n=p['acc_norm'][name],p['acc_norm'][native]
                g=p['gold']
                category='rescued' if c==g and n!=g else ('damaged' if n==g and c!=g else
                         ('both_correct' if c==g else 'both_wrong'))
                u=np.asarray(x['update'][name])
                props=dict(gain_on_gold=float(u[g,0]),abs_gain_on_pred=float(abs(u[c,0])),
                           direction_on_pred=float(u[c,1]),write_to_raw_rms=float(u[c,2]),
                           picked_chars=x['characters'][c],delta_length_vs_native=x['characters'][c]-x['characters'][n])
                strata[category].append(props)
                if category in examples and len(examples[category])<4:
                    examples[category].append(dict(id=x['id'],question=q['question'],choices=q['choices']['text'],
                                                   native=n,prediction=c,gold=g,**props))
            out[name]=dict(cohorts={a:dict(n=len(r),medians={field:float(np.median([x[field] for x in r]))
                                                            for field in r[0]} if r else {}) for a,r in strata.items()},
                           examples=examples)
    return out


def write_jsonl(path,rows):
    with path.open('w',encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n')


def run(out,archived,corpus):
    old=archive(archived)
    results=archived/'results'
    freeze=json.loads((out/'FREEZE.json').read_text())
    inputs=json.loads((out/'TEST_INPUT.json').read_text())
    assert inputs['freeze_sha256']==sha(out/'FREEZE.json')
    raw=from_jsonl(out/'TEST_PREDICTIONS.jsonl')
    geometry=from_jsonl(out/'GEOMETRY_PREDICTIONS.jsonl')
    for a,b in zip(raw,geometry,strict=True):
        assert all(a[k]==b[k] for k in ('id','labels','gold','answer_tokens','characters','scored_input_tokens'))
        a['scores'].update(b['scores'])
    historical=results/'test_predictions.jsonl'
    verified,paired_rows=reference(raw,from_jsonl(historical),results,freeze['loopcd_validation'],1172)
    assert all(len(x['scores'])==40 for x in raw)
    write_jsonl(out/'TEST_PAIRED.jsonl',paired_rows)
    old.dump(out/'METRICS.json',verified)
    old.dump(out/'VALIDATION_PAIRED.json',validation_pairs(out,from_jsonl(results/'validation_predictions.jsonl')))
    old.dump(out/'FAILURES.json',failure_summary(raw,paired_rows,old.read_data(corpus,'test')))
    old.dump(out/'HISTORICAL_COST.json',json.loads((results/'standalone_cost.json').read_text()))
    artifact=archived/'ARTIFACTS.sha256'
    reference_digest=sha(historical)
    assert any(reference_digest in line and 'test_predictions.jsonl' in line for line in artifact.read_text().splitlines())
    ledger=(out/'GPU_LEDGER.tsv').read_text().splitlines()
    assert ledger[-1].startswith('END\t')
    used=sum(int(x.split('\t')[4]) for x in ledger if x.startswith('END\t'))
    assert used<=18000
    old.dump(out/'VERIFICATION.json',dict(n=len(raw),methods=len(raw[0]['scores']),metric_names=list(METRICS),
                                         total_new_scores=sum(sum(len(a) for a in x['scores'].values()) for x in raw),
                                         historic_prediction_sha256=reference_digest,gpu_seconds=used,
                                         predictions_sha256=sha(out/'TEST_PREDICTIONS.jsonl'),
                                         geometry_sha256=sha(out/'GEOMETRY_PREDICTIONS.jsonl'),paired_sha256=sha(out/'TEST_PAIRED.jsonl')))
    print('Verified complete inference, pairing and historic score parity',len(raw),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for opt in ('out','archive','corpus'):
        p.add_argument(opt,type=Path)
    a=p.parse_args()
    run(a.out,a.archive,a.corpus)
