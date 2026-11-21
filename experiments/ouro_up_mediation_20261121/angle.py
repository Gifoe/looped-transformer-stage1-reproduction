"""Readout-preserving state rotations at matched token norms and displacements."""
import argparse
import hashlib
from pathlib import Path

import torch

from native import Forward,load,save,setup,sha,timer
from route_ops import lane_plan,norms
from up_mediation import capture


def variants(r,d,seed):
    # Double precision for the geometry, real model remains FP32.
    x,y=r.double(),d.double()
    scale=x.norm(dim=-1,keepdim=True).clamp_min(1e-18)
    u=x/scale
    v=y/y.norm(dim=-1,keepdim=True).clamp_min(1e-18)
    c=(u*v).sum(dim=-1,keepdim=True).clamp(-1,1)
    tangent=v-c*u
    length=tangent.norm(dim=-1,keepdim=True)
    assert not ((length<1e-14)&(c<0)).any(), "Antipodal geometry has no unique native tangent"
    t=tangent/length.clamp_min(1e-18)
    unchanged=(x==y).all(dim=-1,keepdim=True) | (length<1e-14)
    def safe(q):
        return torch.where(unchanged,x,q*scale).to(r.dtype)
    sides={"direction":safe(v)}
    sides["half"]=safe((u+v)/(u+v).norm(dim=-1,keepdim=True).clamp_min(1e-18))
    for j in (0,1):
        g=torch.Generator(device=r.device).manual_seed(seed+173*j)
        n=torch.randn(u[:1].shape,device=u.device,dtype=u.dtype,generator=g).expand_as(u)
        n=n-(n*u).sum(dim=-1,keepdim=True)*u
        n=n-(n*t).sum(dim=-1,keepdim=True)*t
        n=n/n.norm(dim=-1,keepdim=True).clamp_min(1e-18)
        sides[f"angle_control_{j}"]=safe(c*u+length*n)
    return sides


def one(f,tok,q):
    ids,lens,read=lane_plan(tok,q["prompt"],q["choices"])
    n=lens[0]; last=n-1
    assert all(x==n for x in lens)
    begin=timer()
    clean,_,_=capture(f,ids,n)
    drop,_,_=capture(f,ids,n,1.)
    world={"clean":clean,"drop":drop}
    arms={}
    seed=int.from_bytes(hashlib.sha256(q["id"].encode()).digest()[:7],"little")
    for side,other in (("drop","clean"),("clean","drop")):
        h=world[side][0]
        ref=h[:,:last]
        donor=world[other][0][:,:last]
        s=variants(ref,donor,seed)
        out={}
        goal=(s["direction"]-ref).norm(dim=-1)
        for name,x in s.items():
            inp=h.clone(); inp[:,:last]=x
            assert torch.equal(inp[:,last:],h[:,last:])
            ratio=(x.norm(dim=-1)-ref.norm(dim=-1)).abs()/ref.norm(dim=-1).clamp_min(1e-12)
            assert float(ratio.max())<2e-5
            displacement=(x-ref).norm(dim=-1)
            err=float((displacement-goal).abs().max()/goal.max().clamp_min(1e-10))
            if name!="half":
                assert err<3e-5, (q["id"],err)
            h2=f.run(ids,1,start=inp)[0]
            out[name]=dict(ll=read(f.model,h2),k1_ll=read(f.model,inp),dose=float(displacement.norm()),
                           norm_error=float(ratio.max()),per_token_dose_error=err,
                           output_norm_ratio=norms(h2,world["clean"][1],lens))
        arms[side]=out
    return dict(id=q["id"],native={k:[read(f.model,h) for h in z] for k,z in world.items()},
                arms=arms,gpu_seconds=timer()-begin)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("dataset")
    a=p.parse_args()
    rows=load(a.manifest)["cohorts"][a.dataset]["rows"]
    model,tok=setup(a.model)
    f=Forward(model)
    out=load(a.out)["rows"] if a.out.exists() else []
    assert [x["id"] for x in out]==[x["id"] for x in rows[:len(out)]]
    for q in rows[len(out):]:
        out.append(one(f,tok,{k:v for k,v in q.items() if k!="gold"}))
        if len(out)%4==0 or len(out)==len(rows):
            save(a.out,dict(expected=len(rows),manifest_sha256=sha(a.manifest),rows=out,
                            peak_bytes=torch.cuda.max_memory_allocated()))
            print(a.dataset,len(out),len(rows),flush=True)


if __name__=="__main__":
    main()

