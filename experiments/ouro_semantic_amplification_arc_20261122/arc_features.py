"""Collect output-position states using the archived ARC scorer and frozen Ouro."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys
import time

import torch


def archive(path):
    sys.path.insert(0, str(path.resolve()))
    import run_arc
    return run_arc


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_save(obj, path):
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


@contextmanager
def record_writes(model, group, pos, architecture='late'):
    layers = model.model.layers
    state = defaultdict(list)
    branches = ('attention', 'mlp')
    selected = range(18,24) if architecture=='late' else range(12,24)
    mid = selected.start+3
    pending = {}

    def write(name, index):
        def record(_, args, x):
            band = 'last' if index>=mid else 'first'
            key = name+'_'+band
            value = x[group,pos].float()
            pending[key] = pending[key]+value if key in pending else value
        return record

    def finish(_, args):
        state['raw'].append(args[0][group,pos].contiguous())
        for name in branches:
            for band in ('first','last'):
                key = name+'_'+band
                state[key].append(pending.pop(key).to(torch.bfloat16))
        assert not pending

    hooks = [model.model.norm.register_forward_pre_hook(finish)]
    for i in selected:
        hooks.append(layers[i].input_layernorm_2.register_forward_hook(write('attention',i)))
        hooks.append(layers[i].post_attention_layernorm_2.register_forward_hook(write('mlp',i)))
    try:
        yield state
    finally:
        for hook in hooks:
            hook.remove()


@torch.no_grad()
def forward(model, old, items, depth, architecture='late'):
    inp, group, pos, target = old.pack(items)
    model.model.total_ut_steps = depth
    with record_writes(model,group,pos,architecture) as taps:
        _, normed, _ = model.model(input_ids=inp,use_cache=False)
    assert len(normed)==len(taps['raw'])==depth
    state = {k:torch.stack(v,1) for k,v in taps.items()}
    state['hidden'] = torch.stack([x[group,pos] for x in normed],1)
    return state, group, target


def collect(model_path, corpus, out, archived, batch=24):
    old = archive(archived)
    model, tok, runtime = old.load_runtime(model_path)
    model.requires_grad_(False)
    out.mkdir(parents=True,exist_ok=True)
    old.dump(out/'RUNTIME.json',runtime)
    for split, depth in (('train',5),('validation',4)):
        begin = time.perf_counter()
        rows = old.read_data(corpus,split)
        candidates, info = old.encode(tok,rows)
        part = out / split
        part.mkdir(exist_ok=True)
        old.dump(part/'index.json',dict(rows=info, depth=depth, dataset_sha256=sha(corpus/f'{split}.parquet'),
                                      revision=old.REVISION,candidates=len(candidates),chunk_questions=32))
        # Group complete questions in each on-disk shard. Length-sort only within shards
        # for collection. Restore exact question/choice/token order for training.
        offsets = [0]
        for row in info:
            offsets.append(offsets[-1]+len(row['labels']))
        for lo in range(0,len(rows),32):
            path = part/f'{lo:04}.pt'
            if path.exists():
                continue
            hi = min(lo+32,len(rows))
            items = candidates[offsets[lo]:offsets[hi]]
            order = sorted(range(len(items)),key=lambda i:len(items[i][0]))
            records = [None]*len(items)
            for j in range(0,len(order),batch):
                chosen = order[j:j+batch]
                states, _, targets = forward(model,old,[items[i] for i in chosen],depth)
                states = {k:v.cpu() for k,v in states.items()}
                targets = targets.cpu()
                cursor = 0
                for i in chosen:
                    _, n, q, a = items[i]
                    sl = slice(cursor,cursor+n)
                    records[i] = dict(q=q,a=a,target=targets[sl],state={k:v[sl] for k,v in states.items()})
                    cursor += n
            atomic_save(records,path)
            print('features',split,hi,len(rows),round(time.perf_counter()-begin,1),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('model','corpus','out','archive'):
        p.add_argument(arg,type=Path)
    a = p.parse_args()
    collect(a.model,a.corpus,a.out,a.archive)
