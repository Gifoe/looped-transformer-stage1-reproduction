"""Verify that A/B actually shared their per-update training exposure."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def signature(path):
    with path.open('rb') as file:
        return file_digest(file,'sha256').hexdigest()


def run(root, seed):
    dirs={arm:root/arm/str(seed) for arm in 'AB'}
    locks={arm:read(path/'TRAIN_LOCK.json') for arm,path in dirs.items()}
    subject=('seed','initial_factors_sha256','dataset_sha256','depths','per_step','model_sha256','lr','code_sha256')
    if any(locks['A'][key]!=locks['B'][key] for key in subject):
        raise ValueError('A/B initial states, data, implementation or schedule differ')
    if tuple(locks[arm]['rank_weight'] for arm in 'AB')!=(0,.25):
        raise ValueError('A/B ranking coefficients changed')
    blocks=[]
    for epoch in (1,2,3):
        summaries={}; traces={}
        for arm, path in dirs.items():
            e=path/f'epoch_{epoch:02d}'; w=e/'controller.pt'; summary=read(e/'TRAIN.json')
            net=torch.load(w,map_location='cpu',weights_only=True)['state']
            if signature(w)!=summary['checkpoint_sha256'] or sum(v.numel() for v in net.values())!=4096 or any(v.dtype!=torch.float32 for v in net.values()):
                raise ValueError(f'{arm} epoch {epoch} trained checkpoint failed weight audit')
            subset=[t for t in summary['logs'] if t['epoch']==epoch]
            exposed={event[1] for trace in subset for event in trace['views']}
            if len(subset)!=500 or len(exposed)!=4000 or summary['optimizer_updates']!=500*epoch:
                raise ValueError('Invalid per-epoch update or original question count')
            traces[arm]=subset
            summaries[arm]=dict(checkpoint_sha256=summary['checkpoint_sha256'], updates=summary['optimizer_updates'],
                                target_tokens=summary['target_tokens'], padded_token_loops=summary['padded_token_loops'],
                                cumulative_seconds=summary['seconds'], epoch_mean_nll=sum(t['nll'] for t in subset)/len(subset))
        matched=('epoch','step','depth','views','target_tokens')
        if any([a[key] for key in matched]!=[b[key] for key in matched] for a,b in zip(traces['A'],traces['B'],strict=True)):
            raise ValueError(f'A/B training exposures differ in epoch {epoch}')
        blocks.append(dict(epoch=epoch, fully_matched=True, arms=summaries,
                           rank_auxiliary_work_ratio=summaries['B']['padded_token_loops']/summaries['A']['padded_token_loops']))
    return dict(seed=seed,matched=True,shared={key:locks['A'][key] for key in subject},epochs=blocks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fits',type=Path,default=Path('results/fits'))
    p.add_argument('--seed',type=int,default=261124137)
    p.add_argument('--out',type=Path,default=Path('results/A_B_TRAINING_AUDIT.json'))
    a=p.parse_args()
    out=run(a.fits,a.seed)
    a.out.write_text(json.dumps(out,indent=2)+'\n')
    print([(row['epoch'],row['arms']['A']['updates'],row['rank_auxiliary_work_ratio']) for row in out['epochs']],flush=True)
