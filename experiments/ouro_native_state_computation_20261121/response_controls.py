"""Separate hidden-direction effects from global state scale and perturbation dose."""
import argparse
from pathlib import Path

import torch

from causal import direction_like, unit, vector_stats
from native import Forward, SEED, load, save, setup, sha, tensors, timer


def response(f,ids,h,d):
    a=f.run(ids,1,start=h)[0]
    b=f.run(ids,1,start=h+d)[0]
    return b-a


def one(f,tok,item):
    ids,_=tensors(tok,[item["prompt"]])
    h0=f.core.embed_tokens(ids)
    h1=f.run(ids,1)[0]
    pos=(ids.shape[-1]-1)//2
    d=torch.zeros_like(h0)
    d[:,pos]=direction_like(h0[:,pos],SEED)
    f0,f1=(float(h[:,pos].norm()) for h in (h0,h1))
    dose=unit(d,h0.new_tensor(.12*f0))
    equal_amp=[response(f,ids,h,dose) for h in (h0,h1)]
    halfway=[response(f,ids,h,dose*.5) for h in (h0,h1)]
    relative=[equal_amp[0],response(f,ids,h1,dose*f1/f0)]
    # Whole-token scaling preserves H0's token directions while matching H1 token lengths.
    adjusted=h0*(h1.norm(dim=-1,keepdim=True)/h0.norm(dim=-1,keepdim=True).clamp_min(1e-9))
    normmatched=response(f,ids,adjusted,dose)
    return dict(id=item["id"],position=pos,norm_original=[f0,f1],
       same_delta=vector_stats(equal_amp[1],equal_amp[0]),
       half_delta=vector_stats(halfway[1],halfway[0]),
       same_relative_delta=vector_stats(relative[1],relative[0]),
       norm_matched_H0_vs_H1=vector_stats(equal_amp[1],normmatched),
       H0_response_scaling=vector_stats(normmatched,equal_amp[0]))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("blind",type=Path)
    a=p.parse_args()
    data=load(a.blind)
    cohorts={k:data[k][:24] for k in ("confirm_arc","confirm_external")}
    pred=a.out/"RESPONSE_PREDICTIONS.json"
    if not pred.exists():
        save(pred,dict(candidates_sha256=sha(a.out/"CANDIDATES.json"),manifest_sha256=sha(a.blind),
             cohort={k:[q["id"] for q in v] for k,v in cohorts.items()},
             hypothesis="amplitude alone should preserve response direction even after tokenwise state-scale matching"))
    model,tok=setup(a.model)
    f=Forward(model)
    t=timer()
    result={}
    for k,items in cohorts.items():
        result[k]=[one(f,tok,q) for q in items]
    save(a.out/"RESPONSE_CONTROLS.json",dict(rows=result,prediction_sha256=sha(pred),seconds=timer()-t,
          peak_cuda_bytes=torch.cuda.max_memory_allocated()))


if __name__=="__main__":
    main()

