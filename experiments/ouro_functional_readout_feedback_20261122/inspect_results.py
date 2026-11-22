"""Verify case alignment, report paired outcomes and audit the score-calibration explanation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest

from bridge import dump, rows

METRICS=('acc','acc_norm')
DEPTHS=range(1,5)
SEED=261123007


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def merge(*groups):
    assert groups and groups[0]
    result=[dict(r,scores={}) for r in groups[0]]
    for group in groups:
        for a,b in zip(result,group,strict=True):
            assert all(a[k]==b[k] for k in ('id','labels','gold','characters','answer_tokens','scored_input_tokens'))
            assert not set(a['scores']).intersection(b['scores'])
            a['scores'].update(b['scores'])
    assert len(result)==len({r['id'] for r in result})
    expected=set(result[0]['scores'])
    for row in result:
        assert set(row['scores'])==expected
        assert all(len(x)==len(row['labels']) and np.isfinite(x).all() for x in row['scores'].values())
    return result


def rename(records,include):
    return [dict(r,scores={new:r['scores'][key] for key,new in include.items()}) for r in records]


def paired(base,new):
    n=len(base)
    f=int((~base&new).sum()); d=int((base&~new).sum())
    if f+d==0:
        return dict(rescued=0,damaged=0,delta_pp=0.,ci95_pp=[0.,0.],p=1.)
    j=np.random.default_rng(SEED).multinomial(n,[d/n,1-(f+d)/n,f/n],size=20000)
    return dict(rescued=f,damaged=d,delta_pp=100*(f-d)/n,
                ci95_pp=np.quantile(100*(j[:,2]-j[:,0])/n,[.025,.975]).tolist(),p=float(binomtest(f,f+d).pvalue))


def holm(entries,key='holm4_p'):
    prev=0.
    for i,r in enumerate(sorted(entries,key=lambda a:a['p'])):
        prev=max(prev,min(1.,(len(entries)-i)*r['p']))
        r[key]=prev


def calculate(records,prior_families):
    n=len(records)
    arms=tuple(records[0]['scores'])
    gold=np.asarray([r['gold'] for r in records])
    stats={}; predictions={}
    contrasts=(('gate','dual'),('gate','confidence'),('teacher','shuffled'),('functional','dual'),
               ('functional','shuffled'),('teacher','dual'),('feedback_only','dual'),
               ('readout_plus_feedback','dual'),('readout_plus_feedback','feedback_only'),
               ('feedback_only','feedback_rotation'),('feedback_only','feedback_state_distillation'),
               ('feedback_only','feedback_previous_direction'),('feedback_only','feedback_shuffled_teacher'),('dual','scoreonly'),
               ('readout_plus_feedback','scoreonly'),('feedback_only','scoreonly'))
    for m in METRICS:
        predictions[m]={a:np.array([np.argmax(np.asarray(r['scores'][a])/(r['characters'] if m=='acc_norm' else 1)) for r in records])
                        for a in arms}
        correct={a:(v==gold) for a,v in predictions[m].items()}
        baselines={f'k{k}':max((s for s in prior_families if s.endswith(f'_k{k}')),key=lambda s:(int(correct[s].sum()),s)) for k in DEPTHS}
        methods={}
        for a,v in correct.items():
            k=a.rsplit('_k',1)[1]
            methods[a]=dict(correct=int(v.sum()),pct=100*float(v.mean()),
                            vs_native=paired(correct[f'native_k{k}'],v),
                            vs_previous_best=paired(correct[baselines[f'k{k}']],v))
        families={a.rsplit('_k',1)[0] for a in arms}
        for f in families:
            for key in ('vs_native','vs_previous_best'):
                holm([methods[f'{f}_k{k}'][key] for k in DEPTHS if f'{f}_k{k}' in methods])
        contrast={}
        for a,b in contrasts:
            if a in families and b in families:
                contrast[f'{a}_vs_{b}']={f'k{k}':paired(correct[f'{b}_k{k}'],correct[f'{a}_k{k}']) for k in DEPTHS}
                holm(list(contrast[f'{a}_vs_{b}'].values()))
        cross={}
        for short,deep in ((1,2),(2,3),(2,4),(3,4)):
            a=f'readout_plus_feedback_k{short}'
            b=baselines[f'k{deep}']
            if a in correct:
                cross[f'k{short}_vs_k{deep}']=dict(reference=b,**paired(correct[b],correct[a]))
        stats[m]=dict(methods=methods,previous_best=baselines,contrasts=contrast,cross_budget=cross)
    for family in {a.rsplit('_k',1)[0] for a in arms}:
        for key in ('vs_native','vs_previous_best'):
            holm([stats[m]['methods'][f'{family}_k{k}'][key] for m in METRICS for k in DEPTHS
                  if f'{family}_k{k}' in stats[m]['methods']],'holm8_p')
    for name in stats['acc']['contrasts']:
        holm([c for m in METRICS for c in stats[m]['contrasts'][name].values()],'holm8_p')
    records=[dict(id=r['id'],labels=r['labels'],gold=r['gold'],
                  **{m:{a:int(pred[i]) for a,pred in predictions[m].items()} for m in METRICS}) for i,r in enumerate(records)]
    return dict(n=n,metrics=stats),records


def write_jsonl(path,records):
    with path.open('w',encoding='utf-8') as w:
        for x in records:
            w.write(json.dumps(x,ensure_ascii=False)+'\n')


WORDS=re.compile(r'\b[a-z0-9]+\b')


def center(a):
    return a-a.mean(0,keepdims=True)


def calibration_shift(arc,prior,official,archive_test,arc_calibrated):
    base=rows(arc/'results/validation_predictions.jsonl'); old=rows(prior/'results/VALIDATION_completion.jsonl')
    counts=Counter(w for r in pq.read_table(official/'train.parquet').to_pylist() for option in r['choices']['text'] for w in WORDS.findall(option.casefold()))
    raw_val=pq.read_table(official/'validation.parquet').to_pylist(); raw_test=pq.read_table(official/'test.parquet').to_pylist()
    done={}
    test_prior=rows(prior/'results/TEST_PREDICTIONS.jsonl')
    for k in DEPTHS:
        def table(native,after,raw,after_key):
            X,Y=[],[]
            for a,b,c in zip(native,after,raw,strict=True):
                assert a['id']==b['id']==c['id']
                s=np.asarray(a['scores'][f'native_k{k}'])
                ch=np.asarray(a['characters']); tk=np.asarray(a['answer_tokens'])
                freq=np.array([np.mean([np.log1p(counts[w]) for w in WORDS.findall(q.casefold())] or [0.]) for q in c['choices']['text']])
                X.append(center(np.stack((s,ch,tk,s/ch,s/tk,s*np.log(ch),freq),axis=1)))
                Y.append(center(np.asarray(b['scores'][after_key])-s))
            return np.concatenate(X),np.concatenate(Y)
        a,b=table(base,old,raw_val,f'k{k}')
        c,d=table(archive_test,test_prior,raw_test,f'completion_k{k}')
        without=np.linalg.lstsq(a[:,:-1],b,rcond=None)[0]
        withfreq=np.linalg.lstsq(a,b,rcond=None)[0]
        denom=np.sum(d**2)
        cal=np.concatenate([center(np.asarray(p['scores'][f'scoreonly_k{k}'])-np.asarray(q['scores'][f'native_k{k}'])) for p,q in zip(arc_calibrated,archive_test,strict=True)])
        done[f'k{k}']=dict(n=len(c),score_length_r2=float(1-np.sum((d-c[:,:-1]@without)**2)/denom),
                           plus_train_word_frequency_r2=float(1-np.sum((d-c@withfreq)**2)/denom),
                           score_only_shift_correlation=float(np.corrcoef(cal,d)[0,1]))
    return done


def failures(phase,decisions,official):
    data=[]
    for m in METRICS:
        for arm in ('dual','gate','teacher','feedback_only','readout_plus_feedback'):
            for k in DEPTHS:
                key=f'{arm}_k{k}'; base=f'native_k{k}'
                take=[]
                for p,d,r in zip(phase,decisions,official,strict=True):
                    a,b=d[m][base],d[m][key]
                    g=d['gold']
                    if a==g and b!=g:
                        ar=np.asarray(p['scores'][key])/(p['characters'] if m=='acc_norm' else 1)
                        take.append(dict(id=r['id'],native=a,new=b,gold=g,labels=p['labels'],
                                         length_change=p['characters'][b]-p['characters'][a],margin=float(ar[g]-ar[b])))
                data.append(dict(method=key,metric=m,n=len(take),median_length_change=float(np.median([r['length_change'] for r in take])) if take else None,
                                 sample=take[:5]))
    return data


def run(root,arc,prior,official,external):
    hist=rows(arc/'results/test_predictions.jsonl'); p=rows(prior/'results/TEST_PREDICTIONS.jsonl')
    map_previous={f'{name}_k{k}':f'previous_{name}_k{k}' for name in ('completion','shuffled','direction','magnitude') for k in DEPTHS}
    p=rename(p,map_previous)
    actual=rows(root/'ARC_READOUT.jsonl')
    fb=root/'feedback'
    combined=merge(hist,p,actual,rows(fb/'TEST_PREDICTIONS.jsonl'),rows(fb/'TEST_CONTROL_PREDICTIONS.jsonl'))
    assert len(combined)==1172
    baseline_keys=list(hist[0]['scores'])+list(p[0]['scores'])
    agg,choices=calculate(combined,baseline_keys)
    reference=json.loads((arc/'results/test_metrics.json').read_text())
    for m in METRICS:
        assert all(agg['metrics'][m]['methods'][x]['correct']==reference[m][x]['correct'] for x in hist[0]['scores'])
    external_readout=rows(root/'OPENBOOKQA_READOUT.jsonl')
    external_records=merge(external_readout,rows(fb/'EXTERNAL_PREDICTIONS.jsonl'))
    assert len(external_records)==500
    ext,ext_choices=calculate(external_records,[f'native_k{k}' for k in DEPTHS])
    ordinary=calibration_shift(arc,prior,official,hist,actual)
    write_jsonl(root/'ARC_PAIRED.jsonl',choices)
    write_jsonl(root/'OPENBOOKQA_PAIRED.jsonl',ext_choices)
    dump(root/'METRICS.json',dict(arc=agg,openbookqa=ext,calibration_explained=ordinary))
    dump(root/'FAILURES.json',failures(combined,choices,pq.read_table(official/'test.parquet').to_pylist()))
    budget=(root/'GPU_LEDGER.tsv').read_text().splitlines()
    assert budget[-1].startswith('END\t')
    used=sum(int(x.split('\t')[4]) for x in budget if x.startswith('END\t')); assert used<=18000
    records=('ARC_READOUT.jsonl','feedback/TEST_PREDICTIONS.jsonl','feedback/TEST_CONTROL_PREDICTIONS.jsonl',
             'OPENBOOKQA_READOUT.jsonl','feedback/EXTERNAL_PREDICTIONS.jsonl')
    amount=sum(sum(sum(len(score) for score in r['scores'].values()) for r in rows(root/f)) for f in records)
    heldout=pq.read_table(external).to_pylist()
    def stem(r):
        return ' '.join(WORDS.findall(r['question'].casefold()))
    seen={stem(r) for split in ('train','validation') for r in pq.read_table(official/f'{split}.parquet').to_pylist()}
    overlap=sum(stem(r) in seen for r in heldout)
    dump(root/'VERIFICATION.json',dict(arc=1172,openbookqa=500,matched_candidate_scores_in_new_artifacts=amount,
                                      gpu_seconds=used,data_sha256=sha(external),openbookqa_exact_normalized_stem_overlap_with_arc_fit=overlap,
                                      records={f:sha(root/f) for f in records},
                                      historical_reference_sha256=sha(arc/'results/test_predictions.jsonl')))
    print('Verified ARC 1172 and OpenBookQA 500',amount,'candidate scores',used,'GPU seconds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','archive','prior','corpus','external'):
        p.add_argument(name,type=Path)
    a=p.parse_args();run(a.root,a.archive,a.prior,a.corpus,a.external)
