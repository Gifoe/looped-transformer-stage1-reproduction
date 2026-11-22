"""Replace only the learned tangent orientation, preserving dose and raw-state norm."""

import argparse
import json
from pathlib import Path
import time

import torch

from amplify import DEPTHS, digest_bundle, normalized, operator, rms
from arc_features import archive, forward, sha
from evaluate import BATCH, atomic_predictions


def unit(x):
    return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-10)


def project(x, axis):
    return (x*axis).sum(-1,keepdim=True)*axis


def rotated_tangent(h,w,shift):
    w_axis = unit(w)
    h_axis = unit(h-project(h,w_axis))
    axis = project(shift,h_axis)
    tangent = shift-axis
    signs = 1-2*(torch.arange(h.shape[-1],device=h.device)%2)
    rotated = tangent.roll(h.shape[-1]//3,-1)*signs
    rotated = rotated-project(rotated,w_axis)
    rotated = rotated-project(rotated,h_axis)
    return axis+unit(rotated)*tangent.norm(dim=-1,keepdim=True)


@torch.no_grad()
def run(model_path,corpus,out,archived):
    old=archive(archived)
    freeze=json.loads((out/'FREEZE.json').read_text())
    assert freeze['hashes']==digest_bundle(out,freeze['chosen'])
    lock=out/'GEOMETRY_CODE_LOCK.sha256'
    assert lock.read_text().split()[0]==sha(Path(__file__))
    data=old.read_data(corpus,'test')
    model,tok,_=old.load_runtime(model_path)
    model.requires_grad_(False)
    items,info=old.encode(tok,data)
    items.sort(key=lambda a:len(a[0]))
    nets={name:(operator(out/freeze['chosen'][name]['file']),freeze['chosen'][name]['dose'])
          for name in ('direction','completion')}
    begin=time.perf_counter()
    maximum={}
    for k in DEPTHS:
        max_diff=torch.zeros(3,device='cuda')
        for offset in range(0,len(items),BATCH):
            batch=items[offset:offset+BATCH]
            state,group,target=forward(model,old,batch,k)
            h=state['raw'][:,k-1].float()
            for name,(net,dose) in nets.items():
                _,(gain,shift,wr,w,_)=net(state,k,dose)
                rotated=rotated_tangent(h,w,shift)
                true=dose*(gain*w+shift)
                geom=dose*(gain*w+rotated)
                residual=torch.stack(((rms(true)-rms(geom)).abs().max(),
                                      (rms(h+true)-rms(h+geom)).abs().max(),
                                      (((true-geom)*w).mean(-1).abs()/(wr[:,0]+1e-8)).max()))
                max_diff=torch.maximum(max_diff,residual)
                logits=model.lm_head(normalized(model,(h+geom).to(torch.bfloat16))).float()
                values=old.scores(logits,group.cpu().numpy(),target,len(batch))
                old.record(info,batch,{f'{name}_rotated_k{k}':values})
        maximum[f'k{k}']=max_diff.tolist()
        assert max(maximum[f'k{k}']) < .002, maximum
        print('geometry',k,maximum[f'k{k}'],flush=True)
    torch.cuda.synchronize()
    atomic_predictions(old,out/'GEOMETRY_PREDICTIONS.jsonl',info)
    old.dump(out/'GEOMETRY_AUDIT.json',dict(max_residuals=maximum,seconds=time.perf_counter()-begin,
                                            constraints=['update RMS','edited raw RMS','normalized projection on write']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for opt in ('model','corpus','out','archive'):
        p.add_argument(opt,type=Path)
    a=p.parse_args()
    run(a.model,a.corpus,a.out,a.archive)
