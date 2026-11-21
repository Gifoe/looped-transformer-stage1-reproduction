"""Patch only earlier native states while preserving every K1 candidate readout position."""
import argparse
from pathlib import Path

import torch

from native import Forward,load,save,setup,sha,timer
from route_ops import lane_plan,norms
from up_mediation import capture


def intervention(receiver,donor,n,kind):
    output=receiver.clone()
    last=n-1
    d=donor[:,:last]
    b=receiver[:,:last]
    if kind=="body":
        output[:,:last]=d
    elif kind=="body_scaled":
        output[:,:last]=b*(d.norm(dim=-1,keepdim=True)/b.norm(dim=-1,keepdim=True).clamp_min(1e-10))
    elif kind=="body_direction":
        output[:,:last]=d*(b.norm(dim=-1,keepdim=True)/d.norm(dim=-1,keepdim=True).clamp_min(1e-10))
    elif kind=="body_rolled":
        output[:,:last]=b+(d-b).roll(277,-1)
    elif kind=="boundary":
        output[:,last:n]=donor[:,last:n]
    elif kind=="whole_prompt":
        output[:,:n]=donor[:,:n]
    else:
        raise ValueError(kind)
    return output


def row(f,tok,item):
    ids,lens,read=lane_plan(tok,item["prompt"],item["choices"])
    n=lens[0]
    assert all(p==n for p in lens)
    t=timer()
    clean,_,_=capture(f,ids,n)
    drop,_,_=capture(f,ids,n,1.)
    worlds={"clean":clean,"drop":drop}
    a={}
    for rec,don in (("drop","clean"),("clean","drop")):
        u,v=worlds[rec][0],worlds[don][0]
        arms={}
        for k in ("body","body_rolled","body_scaled","body_direction","boundary","whole_prompt"):
            inp=intervention(u,v,n,k)
            unchanged=torch.equal(inp[:,n-1:],u[:,n-1:])
            if k.startswith("body"):
                assert unchanged
            h=f.run(ids,1,start=inp)[0]
            arms[k]=dict(ll=read(f.model,h),dose=float((inp-u).norm()),
                         k1_readout_bitwise_held=unchanged,
                         k1_margin_input=read(f.model,inp),
                         input_norm_max_ratio=norms(inp,u,lens),
                         output_norm_max_ratio=norms(h,worlds["clean"][1],lens))
        contrast=abs(arms["body"]["dose"]-arms["body_rolled"]["dose"])/max(arms["body"]["dose"],1e-12)
        assert contrast<1e-5
        a[rec]=arms
    return dict(id=item["id"],native={s:[read(f.model,h) for h in y] for s,y in worlds.items()},
                arms=a,seconds=timer()-t)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("family")
    args=p.parse_args()
    data=load(args.manifest)["cohorts"][args.family]["rows"]
    model,tok=setup(args.model)
    f=Forward(model)
    records=load(args.out)["rows"] if args.out.exists() else []
    assert [r["id"] for r in records]==[r["id"] for r in data[:len(records)]]
    for q in data[len(records):]:
        records.append(row(f,tok,{k:v for k,v in q.items() if k!="gold"}))
        if len(records)%4==0 or len(records)==len(data):
            save(args.out,dict(rows=records,expected=len(data),manifest_sha256=sha(args.manifest),
                               peak_bytes=torch.cuda.max_memory_allocated()))
            print(args.family,len(records),len(data),flush=True)


if __name__=="__main__":
    main()

