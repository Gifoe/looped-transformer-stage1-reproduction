"""Numeric, parity, and causal-prefix gates before composition testing."""
import argparse
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from pair_fields import EmbeddingOperators, PairOps, pair, rel, timed, weight_hash, write_json, load_runtime
from causal_worlds import masked
from norm_flow import radial
from dynamics_embedding import read_json


LAYERS=(0,5,11,17,23)


def parity(model,members):
    op=PairOps(model,[masked(w) for w in members])
    truth=EmbeddingOperators(model,members,2)
    assert torch.equal(op.start,truth.k1)
    x=op.start
    checks=[]
    for i in range(48):
        x=x+op.branch(i,x)
        checks.append(torch.equal(truth.actual[i+1],x))
    assert all(checks)
    assert torch.equal(op.norm(x),truth.k2)
    dense=truth.parity_receipt()
    return dict(dense,all_k1_capture_branch_outputs=all(checks))


def prefix(model,group):
    a,_,b,_=group["worlds"]
    cut=next(i for i,(x,y) in enumerate(zip(a["ids"],b["ids"],strict=True)) if x!=y)
    op=PairOps(model,[masked(a),masked(b)])
    assert torch.equal(op.start[0,:cut],op.start[1,:cut])
    data={}
    for method in ("native","picard","anderson","prefix_anderson","hybrid","full"):
        h=op.start
        y=[]
        for l in range(24):
            h=pair(op,l,h,method,steps=2)
            y.append(float((h[0,:cut]-h[1,:cut]).abs().max()))
        n=op.norm_flow(h)
        data[method]=dict(per_pair=y,norm_prefix_max=float((n[0,:cut]-n[1,:cut]).abs().max()))
    return dict(group=group["group"],shared_tokens=cut,divergence=data)


def norm_audit(op,h):
    n=op.norm_flow
    x=h.double()
    exact=x*(n.gamma.double()*torch.rsqrt(x.square().mean(-1,keepdim=True)+n.epsilon))
    eps1=radial(x,.6,1.)
    near=radial(x,.6,1+1e-10)
    mixed=n(h)
    return dict(epsilon=n.epsilon,minimum_gamma=float(n.gamma.min()),maximum_gamma=float(n.gamma.max()),
        true_fp64_max_error=float((n.at(x,2.)-exact).abs().max()),
        epsilon_limit_max_error=float((eps1-near).abs().max()),
        float_backend=rel(op,mixed,op.norm(h)).tolist())


def local(model,groups,arc):
    rows=[]
    timing=[]
    for f in ("tableau","twohop","arc"):
        members=[g["worlds"][0] for g in groups[f][:8]] if f!="arc" else arc
        op=PairOps(model,[masked(w) for w in members])
        true=EmbeddingOperators(model,members,2)
        with timed(op,"local/"+f,timing):
            for layer in LAYERS:
                x=true.actual[2*layer]
                for alpha in (.0625,.125,.25,.5,1.):
                    target=pair(op,layer,x,"native",alpha=alpha)
                    for method in ("base","self","order","full","reverse","taylor","base_fine","branch_bea",
                                   "branch_flow","picard","prefix_anderson"):
                        grids=[(2,"rk4")]
                        if method in ("full","base"):
                            grids += [(4,"rk4"),(4,"midpoint")]
                            if layer in (0,23):
                                grids += [(8,"rk4"),(16,"rk4")]
                        estimates={}
                        for steps,scheme in grids:
                            v=pair(op,layer,x,method,alpha,steps,scheme)
                            estimates[f"{steps}/{scheme}"]=v
                        fine=estimates[("16/rk4" if layer in (0,23) else "4/rk4") if len(grids)>1 else "2/rk4"]
                        for (steps,scheme),v in zip(grids,estimates.values(),strict=True):
                            rows.append(dict(cohort=f,items=[w["item"] for w in members],layer=layer+1,
                                             alpha=alpha,method=method,m=steps,scheme=scheme,
                                             model_error=rel(op,v,target).tolist(),
                                             solver_error=rel(op,v,fine).tolist()))
    return rows,timing


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,_,runtime=load_runtime(a.model)
    model.requires_grad_(False)
    bf=weight_hash(model,torch.bfloat16)
    m=read_json(a.out/"MANIFEST.json")
    nat=m["groups"]["pilot"]
    batches=([g["worlds"][0] for g in nat["tableau"][:2]],
             [g["worlds"][0] for g in nat["twohop"][:2]],m["arc_pilot"][:2])
    audits={}
    with sdpa_kernel(SDPBackend.MATH):
        for dtype in (torch.bfloat16,torch.float32):
            model.to(dtype=dtype)
            checks=[parity(model,g) for g in batches]
            norms=[]
            for g in batches:
                op=EmbeddingOperators(model,g,2)
                norms.append(norm_audit(PairOps(model,[masked(w) for w in g]),op.actual[-1]))
            audits[str(dtype)]=dict(parity=checks,norm=norms)
        assert not torch.is_autocast_enabled("cuda") and not torch.backends.cuda.matmul.allow_tf32
        hashes=weight_hash(model,torch.float32)
        audit=[prefix(model,nat[f][0]) for f in ("tableau","twohop")]
        write_json(a.out/"AUDIT.json",dict(results=audits,prefix=audit,runtime=runtime,
             backend="torch math SDPA with no KV cache",fp32_hash=hashes,weight_values_bf16_sha256=bf))
        rows,logs=local(model,nat,m["arc_pilot"])
    assert weight_hash(model,torch.float32)==hashes and weight_hash(model,torch.bfloat16)==bf
    write_json(a.out/"LOCAL.json",dict(rows=rows,logs=logs,weights_frozen=True))


if __name__=="__main__":
    main()
