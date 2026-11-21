"""Attribute a native factorial score interaction to the operator product or its suffix."""
import argparse
from pathlib import Path

import torch

from native import Forward, Recorder, load, save, setup, sha, timer
from route_ops import HEADS, LAYER, lane_plan, norms


def solve(f,tok,item):
    t=timer()
    ids,lens,read=lane_plan(tok,item["prompt"],item["choices"])
    n=lens[0]
    tap=Recorder(keys={(k,l) for k in (0,1) for l in (0,LAYER)})
    h0,h1=f.run(ids,2,record=tap)
    heads=list(HEADS)

    def attention(r):
        return r["a"][:,heads,:n,:n],r["v"][:,heads,:n]
    a,v=attention(tap.data[1,LAYER]); a0,v0=attention(tap.data[0,LAYER])
    z00=a@v
    z10=a0@v
    z01=a@v0
    z11=a0@v0
    joint=(a0-a)@(v0-v)

    def mlp(r):
        return f.layers[0].mlp.act_fn(r["g_pre"][:,:n]),r["u"][:,:n]
    g,u=mlp(tap.data[1,0]); g0,u0=mlp(tap.data[0,0])
    gates=(g*u,g0*u,g*u0,g0*u0,(g0-g)*(u0-u))
    shared={"L6_attention":(f.layers[LAYER].self_attn.o_proj,z00,z10,z01,z11,joint),
            "L1_mlp":(f.layers[0].mlp.down_proj,*gates)}

    def replay(site,value,is_attention):
        def inject(_,args):
            old=args[0]
            if is_attention:
                r=old.reshape(old.shape[0],old.shape[1],16,-1).clone()
                r[:,:n,heads]=value.transpose(1,2)
                return (r.reshape_as(old),)
            r=old.clone()
            r[:,:n]=value
            return (r,)
        handle=site.register_forward_pre_hook(inject)
        try:
            return f.run(ids,1,start=h0)[0]
        finally:
            handle.remove()

    report={}
    for key,(site,p,x,y,b,z) in shared.items():
        a=key=="L6_attention"
        add=x+y-p
        resid=float(((b-add-z).norm()/b.norm().clamp_min(1e-10)))
        assert resid<2e-5,(key,resid)
        obs={}
        base=replay(site,p,a)
        parity=float((base-h1).abs().max())
        assert parity<5e-5,(key,parity,item["id"])
        for condition,value in (("early_first",x),("early_second",y),("early_both",b),("additive",add),
                                ("permuted",add+z.roll(17,dims=-1))):
            out=replay(site,value,a)
            obs[condition]=dict(ll=read(f.model,out),norm_ratio=norms(out,h1,lens))
        report[key]=dict(native_ll=read(f.model,h1),branches=obs,algebra_residual=resid,noop_error=parity,
                         relative_operator_product=float(z.norm()/p.norm().clamp_min(1e-12)))
    return dict(id=item["id"],K1=read(f.model,h0),cases=report,seconds=timer()-t)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("family")
    a=p.parse_args()
    rows=load(a.manifest)["cohorts"][a.family]["rows"][:128]
    model,tok=setup(a.model)
    f=Forward(model)
    result=load(a.out)["rows"] if a.out.exists() else []
    assert [q["id"] for q in result]==[q["id"] for q in rows[:len(result)]]
    for q in rows[len(result):]:
        result.append(solve(f,tok,{k:v for k,v in q.items() if k!="gold"}))
        if len(result)%4==0 or len(result)==len(rows):
            save(a.out,dict(expected=len(rows),manifest_sha256=sha(a.manifest),rows=result,
                            peak_bytes=torch.cuda.max_memory_allocated()))
            print(a.family,len(result),len(rows),flush=True)


if __name__=="__main__":
    main()

