"""Determine whether Up affects the answer through earlier prompt tokens or the readout slot."""
import argparse
from pathlib import Path

import torch

from native import Forward,load,save,setup,sha,timer
from route_ops import lane_plan
from up_mediation import PRODUCT_ROLL,capture,patch_down


def run(f,tok,q):
    ids,lens,read=lane_plan(tok,q["prompt"],q["choices"])
    n=lens[0]
    t=timer()
    hc,c,_=capture(f,ids,n)
    hd,d,_=capture(f,ids,n,1.)
    u={k:x["u"][:,:n] for k,x in (("clean",c),("drop",d))}
    g={k:f.layers[0].mlp.act_fn(x["g_pre"][:,:n]) for k,x in (("clean",c),("drop",d))}
    states={"clean":hc,"drop":hd}
    scores={k:[read(f.model,a) for a in h] for k,h in states.items()}
    branches={}
    for rec,don in (("drop","clean"),("clean","drop")):
        base=u[rec]*g[rec]
        prod=(u[don]-u[rec])*g[rec]
        suite={}
        for key,start,end in (("earlier",0,n-1),("readout",n-1,n),("all",0,n)):
            z=base.clone()
            p=prod[:,start:end]
            z[:,start:end]+=p
            patched=patch_down(f,ids,states[rec][0],n,z)
            z=base.clone()
            z[:,start:end]+=p.roll(PRODUCT_ROLL,-1)
            matched=patch_down(f,ids,states[rec][0],n,z)
            suite[key]=dict(actual=read(f.model,patched),control=read(f.model,matched),
                           product_norm=float(p.norm()))
        branches[rec]=suite
    return dict(id=q["id"],native=scores,arms=branches,seconds=timer()-t)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("dataset")
    a=p.parse_args()
    items=load(a.manifest)["cohorts"][a.dataset]["rows"][:128]
    model,tok=setup(a.model)
    f=Forward(model)
    out=load(a.out)["rows"] if a.out.exists() else []
    assert [x["id"] for x in out]==[x["id"] for x in items[:len(out)]]
    for q in items[len(out):]:
        out.append(run(f,tok,{k:v for k,v in q.items() if k!="gold"}))
        if len(out)%4==0:
            save(a.out,dict(rows=out,expected=len(items),manifest_sha256=sha(a.manifest),
                            peak_bytes=torch.cuda.max_memory_allocated()))
            print(a.dataset,len(out),len(items),flush=True)


if __name__=="__main__":
    main()

