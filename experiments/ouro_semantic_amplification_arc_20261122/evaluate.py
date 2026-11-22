"""Evaluate only new frozen ARC methods on the entire official Test split."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from amplify import DEPTHS, METHODS, digest_bundle, normalized, operator, rms
from arc_features import archive, forward, sha


ABLATIONS = {'direction':('magnitude_only','direction_only'),
             'completion':('magnitude_only','direction_only')}
BATCH = 16


def key(name,k):
    return f'{name}_k{k}'


def item_scores(old,model,net,state,group,target,n,k,dose,component):
    x, meta = net(state,k,dose,component)
    logits = model.lm_head(normalized(model,x)).float()
    scores = old.scores(logits,group.cpu().numpy(),target,n)
    return scores,meta


def atomic_predictions(old,path,rows):
    temp = path.with_suffix('.tmp')
    old.write_predictions(temp,rows)
    temp.replace(path)


def from_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def record_update(info,items,group,meta,k,name,scale):
    gain, shift, wr, _, hr = meta
    x = torch.cat((gain,rms(shift)/wr,wr/hr),1).cpu().numpy()
    groups = group.cpu().numpy()
    t = np.bincount(groups,minlength=len(items))
    means = [np.bincount(groups,weights=v,minlength=len(items))/t for v in x.T]
    for i,(_,_,q,a) in enumerate(items):
        row = info[q]
        table = row['update'].setdefault(key(name,k),[None]*len(row['labels']))
        table[a] = [scale*float(means[0][i]),scale*float(means[1][i]),float(means[2][i])]


@torch.no_grad()
def timed_eval(old,model,net,items,info,depth,name,dose):
    def batch(it):
        state, group, targets = forward(model,old,it,depth)
        value, meta = item_scores(old,model,net,state,group,targets,len(it),depth,dose,'full')
        return value,group,meta

    batch(items[:BATCH])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    elapsed = []
    obs = {}
    started = time.perf_counter()
    for i in range(0,len(items),BATCH):
        it = items[i:i+BATCH]
        begin = time.perf_counter()
        scores,group,meta = batch(it)
        elapsed.append(time.perf_counter()-begin)
        for (_,_,q,a),value in zip(it,scores,strict=True):
            slots = obs.setdefault(q,[None]*len(info[q]['labels']))
            slots[a] = float(value)
        # Transfer only the compact diagnostics after the primary scored section.
        record_update(info,it,group,meta,depth,name,dose)
    torch.cuda.synchronize()
    end = time.perf_counter()
    peak = torch.cuda.max_memory_allocated()
    tokens = sum(len(x[0])-1 for x in items)
    padded = sum(len(chunk)*max(len(x[0])-1 for x in chunk) for i in range(0,len(items),BATCH)
                 for chunk in [items[i:i+BATCH]])
    tr = dict(seconds_full_split=float(sum(elapsed)),wall_including_audit_s=end-started,
              median_batch_ms=float(1000*np.median(elapsed)),p95_batch_ms=float(1000*np.quantile(elapsed,.95)),
              batch_size=BATCH,peak_allocated_bytes=peak,operator_parameters=sum(p.numel() for p in net.parameters()),
              candidates=len(items),executed_loop_passes=depth,token_uses=tokens*depth,
              padded_token_uses=padded*depth,batches=len(elapsed),scope='model to candidate sums; warm model, no cache')
    return obs,tr


@torch.no_grad()
def ablate(old,model,net,items,info,name,k,dose):
    contexts = ABLATIONS[name]
    for i in range(0,len(items),BATCH):
        it = items[i:i+BATCH]
        state,group,target=forward(model,old,it,k)
        for component in contexts:
            scores,_ = item_scores(old,model,net,state,group,target,len(it),k,dose,component)
            old.record(info,it,{key(name+'_'+component,k):scores})


def merge(info,partial):
    assert [x['id'] for x in partial]==[x['id'] for x in info]
    for row,p in zip(info,partial,strict=True):
        row['scores'].update(p['scores'])
        row['update'].update(p['update'])


def run(model_path,corpus,out,archived):
    old = archive(archived)
    freeze_path = out/'FREEZE.json'
    freeze = json.loads(freeze_path.read_text())
    assert freeze['hashes']==digest_bundle(out,freeze['chosen'])
    historical = json.loads((archived/'results/test_input.json').read_text())
    test_hash=sha(corpus/'test.parquet')
    assert historical['sha256']==test_hash and historical['dataset_revision']==old.REVISION
    rows = old.read_data(corpus,'test')
    assert len(rows)==1172
    for split in ('train','validation'):
        previous = old.read_data(corpus,split)
        assert {x['id'] for x in previous}.isdisjoint(x['id'] for x in rows)
    model,tok,runtime = old.load_runtime(model_path)
    model.requires_grad_(False)
    items, info = old.encode(tok,rows)
    items.sort(key=lambda x:len(x[0]))
    assert len(items)==4687
    for row in info:
        row['update']={}
    manifest = dict(freeze_sha256=sha(freeze_path),dataset_sha256=test_hash,revision=old.REVISION,count=len(rows),
                    candidates=len(items),runtime=runtime,test_count_previously_exposed=len(rows))
    old.dump(out/'TEST_INPUT.json',manifest)
    progress = out/'progress'
    progress.mkdir(exist_ok=True)
    cost = {}
    for name in METHODS:
        selection = freeze['chosen'][name]
        net = operator(out/selection['file'])
        dose = selection['dose']
        for k in DEPTHS:
            method = key(name,k)
            primary = progress/f'{method}.jsonl'
            timing = progress/f'{method}.json'
            if not (primary.exists() and timing.exists()):
                blank = [dict(r,scores={},update={}) for r in info]
                pred,c = timed_eval(old,model,net,items,blank,k,name,dose)
                for q,x in pred.items():
                    blank[q]['scores'][method]=x
                atomic_predictions(old,primary,blank)
                old.dump(timing,c)
            merge(info,from_jsonl(primary))
            cost[method]=json.loads(timing.read_text())
            print('test',method,cost[method]['seconds_full_split'],flush=True)
            if name in ABLATIONS:
                path=progress/f'{method}_ablation.jsonl'
                if not path.exists():
                    blank = [dict(r,scores={},update={}) for r in info]
                    ablate(old,model,net,items,blank,name,k,dose)
                    atomic_predictions(old,path,blank)
                merge(info,from_jsonl(path))
    atomic_predictions(old,out/'TEST_PREDICTIONS.jsonl',info)
    old.dump(out/'COST.json',cost)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for opt in ('model','corpus','out','archive'):
        p.add_argument(opt,type=Path)
    a=p.parse_args()
    run(a.model,a.corpus,a.out,a.archive)
