"""Evaluate locked minimal modules on the two complete candidate benchmarks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import binomtest

from minimal import DEPTHS, dump, load_jsonl, load_net, runtime, sha
from train import evaluate

METRICS=('acc','acc_norm')
REFERENCE=('native','feedback_only','readout_plus_feedback')


def merge(groups):
    combined=[dict(x,scores={}) for x in groups[0]]
    for group in groups:
        for a,b in zip(combined,group,strict=True):
            assert all(a[k]==b[k] for k in ('id','labels','gold','characters','answer_tokens','scored_input_tokens'))
            assert set(a['scores']).isdisjoint(b['scores'])
            a['scores'].update(b['scores'])
    assert len(combined)==len({r['id'] for r in combined})
    arms=set(combined[0]['scores'])
    for x in combined:
        assert set(x['scores'])==arms and all(len(s)==len(x['labels']) and np.isfinite(s).all() for s in x['scores'].values())
    return combined


def paired(v,w):
    n=len(v)
    a=int((~v&w).sum()); b=int((v&~w).sum())
    probs=np.array([b,n-a-b,a])/n
    x=np.random.default_rng(261123157).multinomial(n,probs,size=20000)
    return dict(rescued=a,damaged=b,delta_pp=100*(a-b)/n,
                ci95_pp=np.quantile(100*(x[:,2]-x[:,0])/n,[.025,.975]).tolist(),
                mcnemar=float(binomtest(a,a+b).pvalue) if a+b else 1.)


def holm(x):
    running=0.
    for i,r in enumerate(sorted(x,key=lambda a:a['mcnemar'])):
        running=max(running,min(1.,(len(x)-i)*r['mcnemar']))
        r['p_holm']=running


def statistic(data,new):
    truth=np.array([r['gold'] for r in data])
    names=tuple(data[0]['scores'])
    stats={}; winners={}
    for metric in METRICS:
        normal=metric=='acc_norm'
        predictions={a:np.array([np.argmax(np.array(r['scores'][a])/(r['characters'] if normal else 1)) for r in data]) for a in names}
        mask={a:x==truth for a,x in predictions.items()}
        table={a:dict(correct=int(q.sum()),pct=100*float(q.mean())) for a,q in mask.items()}
        refs={}
        for k in DEPTHS:
            keys=[a for a in names if a.startswith('loopcd_') and a.endswith(f'k{k}')]
            refs[f'k{k}']=max(keys,key=lambda a:(table[a]['correct'],a)) if keys else None
        for name in new:
            for k in DEPTHS:
                a=f'{name}_k{k}'; end=table[a]
                for base in REFERENCE:
                    end[f'vs_{base}']=paired(mask[f'{base}_k{k}'],mask[a])
                key=refs[f'k{k}']
                if key:
                    end['vs_top_loopcd']=paired(mask[key],mask[a])
                b=f'adapter8_t1_k{k}'
                end['vs_adapter']=paired(mask[b],mask[a])
        stats[metric]=dict(methods=table,top_loopcd=refs)
        winners[metric]=predictions
    for name in new:
        for comparator in (*REFERENCE,'top_loopcd','adapter'):
            collection=[stats[m]['methods'][f'{name}_k{k}'][f'vs_{comparator}'] for m in METRICS for k in DEPTHS
                        if f'vs_{comparator}' in stats[m]['methods'][f'{name}_k{k}']]
            holm(collection)
    detailed=[dict(id=r['id'],gold=r['gold'],**{m:{a:int(w[i]) for a,w in winners[m].items()} for m in METRICS})
              for i,r in enumerate(data)]
    return dict(n=len(data),metrics=stats),detailed


def existing(archive,legacy,split):
    fp=legacy/'results/feedback'
    if split=='ARC':
        native=load_jsonl(archive/'results/test_predictions.jsonl')
        feedback=load_jsonl(fp/'TEST_PREDICTIONS.jsonl')
        for r in native:
            r['scores']={a:s for a,s in r['scores'].items() if a.startswith(('native_','loopcd_','secant_','laststep_'))}
    else:
        native=load_jsonl(legacy/'results/OPENBOOKQA_READOUT.jsonl')
        feedback=load_jsonl(fp/'EXTERNAL_PREDICTIONS.jsonl')
        for r in native:
            r['scores']={a:s for a,s in r['scores'].items() if a.startswith('native_')}
    return merge((native,feedback))


def write(path,rows):
    p=path.with_suffix('.partial')
    with p.open('w',encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    p.replace(path)


def run(a):
    freeze=a.out/'FREEZE.json'
    cfg=json.loads(freeze.read_text())
    for source,s in cfg['source'].items():
        assert sha(Path(__file__).parent/source)==s
    for item in cfg['arms'].values():
        assert sha(a.out/item['file'])==item['sha256']
    old,model,tok,info=runtime(a.archive,a.previous,a.model)
    phase={}; costs={}; records={}
    for label in ('ARC','OPENBOOKQA'):
        if label=='ARC':
            source=a.corpus/'test.parquet'
            assert sha(source)==json.loads((a.archive/'results/test_input.json').read_text())['sha256']
            q=old.read_data(a.corpus,'test'); assert len(q)==1172
        else:
            source=a.external
            q=pq.read_table(source).to_pylist(); assert len(q)==500
            assert sha(source)==json.loads((a.legacy/'external/openbookqa.json').read_text())['canonical_sha256']
        records[label]=dict(n=len(q),sha=sha(source))
        ref=existing(a.archive,a.legacy,label)
        progress=a.out/'progress'; progress.mkdir(exist_ok=True)
        output=[ref]; costs[label]={}
        for arm,x in cfg['arms'].items():
            name=progress/f'{label}_{arm}'
            if not (name.with_suffix('.jsonl').exists() and name.with_suffix('.json').exists()):
                net=load_net(a.out/x['file'])
                val,elapsed=evaluate(old,model,tok,q,net,x['dose'])
                write(name.with_suffix('.jsonl'),val); dump(name.with_suffix('.json'),elapsed)
            else:
                val=load_jsonl(name.with_suffix('.jsonl')); elapsed=json.loads(name.with_suffix('.json').read_text())
            for r in val:
                r['scores']={f'{arm}_{tag}':s for tag,s in r['scores'].items()}
            output.append(val)
            costs[label][arm]=dict(parameters=x['params'],dose=x['dose'],depths=elapsed)
            counts={m:[sum(int(np.argmax(np.asarray(r['scores'][f'{arm}_k{k}'])/(r['characters'] if m=='acc_norm' else 1)))==r['gold'] for r in val) for k in DEPTHS] for m in METRICS}
            print(label,arm,counts,flush=True)
        joined=merge(output)
        st,winners=statistic(joined,cfg['arms'])
        for row,pred in zip(joined,winners,strict=True):
            row['predictions']={m:pred[m] for m in METRICS}
        write(a.out/f'RESULTS_{label}.jsonl',joined)
        phase[label]=st
    dump(a.out/'COST.json',costs)
    dump(a.out/'BENCHMARK.json',phase)
    dump(a.out/'BENCHMARK_INPUTS.json',dict(checkpoint_info=info,freeze_sha256=sha(freeze),data=records))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('model','archive','previous','legacy','corpus','external','out'):
        p.add_argument(key,type=Path)
    run(p.parse_args())
