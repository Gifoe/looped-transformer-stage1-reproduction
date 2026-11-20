"""Artifact-first free trajectories. Candidate operators never store native futures."""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from causal_worlds import masked, sha
from pair_fields import PairOps, integrate, pair, timed, weight_hash, write_json, load_runtime
from dynamics_embedding import read_json, norm

LAYERS=(0,5,11,17,23)
DEVM=("base","self","order","full","reverse","base_fine","taylor","picard","prefix_anderson",
      "whole_loop","branch_flow","full_half","full_quarter")
ENDPOINTS=("1/norm","2/norm","3/norm")


def batches(rows,size):
    for k in range(0,len(rows),size):
        yield rows[k:k+size]


def edit(op,records,beta_s=1.,beta_r=1.,control=None):
    h=op.start.to(op.dtype)
    out=h[::4].repeat_interleave(4,dim=0)
    for i,g in enumerate(records):
        j=i*4
        for donor,beta,role,lanes in ((1,beta_s,g["source"],(1,3)),(2,beta_r,g["rule"],(2,3))):
            current=h[j,role]
            other=h[j+donor,role]
            if control=="random":
                gen=torch.Generator(device=h.device).manual_seed(7341+i*31+donor)
                v=torch.randn(current.shape,device=h.device,generator=gen)
                d=other.float()-current.float()
                change=beta*v*d.norm()/v.norm().clamp_min(1e-12)
                value=(current.float()+change).to(h.dtype)
            elif control=="unrelated":
                direction=(other.float()-current.float()).sum(0,keepdim=True)
                role=g["worlds"][0]["roles"]["noise"]
                current=h[j,role]
                change=beta*direction*(other.float()-h[j, g["source"] if donor==1 else g["rule"]].float()).norm()/direction.norm().clamp_min(1e-12)
                value=(current.float()+change).to(h.dtype)
            else:
                value=other if beta==1 else (current.float()+beta*(other.float()-current.float())).to(h.dtype)
            for k in lanes:
                if control=="unrelated" and donor==2 and k==3:
                    out[j+k,role]=(out[j+1,role].float()+value.float()-current.float()).to(h.dtype)
                else:
                    out[j+k,role]=value
    return out


def probed(op,h,groups):
    real=h.float().reshape(len(groups),4,*h.shape[1:])
    mixed=(real[:,3]-real[:,1])-(real[:,2]-real[:,0])
    generator=torch.Generator(device=h.device).manual_seed(6483)
    feature=torch.randn((h.shape[-1],24),device=h.device,generator=generator)/math.sqrt(24)
    projected=mixed@feature
    return projected,mixed


def sample(op,h,groups,show_scores):
    sketch,mixed=probed(op,h,groups)
    per_token=mixed.norm(dim=-1)
    norms=(per_token.square()*op.valid[::4]).sum(-1).sqrt()
    scoring=op.scores(h).tolist() if show_scores else None
    out=[]
    for i,g in enumerate(groups):
        valid=len(g["worlds"][0]["ids"])
        arr=per_token[i,:valid]
        d={"mixed_l2":float(norms[i]),"token_mixed_norm":arr.tolist(),
           "mixed_sketch":sketch[i,:valid].sum(0).tolist(),
           "readout_mixed_sketch":sketch[i,g["worlds"][0]["read"]].tolist()}
        if scoring is not None:
            d["scores"]=scoring[4*i:4*i+4]
        out.append(d)
    return out


def trajectory(op,h,method,loops,steps,clock="linear",start_layer=0,stop_layer=24,record=None,witness=None):
    for loop in range(1,loops+1):
        if method=="persistence":
            pass
        elif method=="whole_loop":
            h=integrate(lambda y,t:op.field(y).float()-y,h,1,8)
        else:
            for l in range(start_layer,stop_layer):
                if method=="full_half" or method=="full_quarter":
                    h=pair(op,l,h,"full",steps=steps,clock=clock,scale=.5 if method=="full_half" else .25)
                else:
                    h=pair(op,l,h,method,steps=steps,clock=clock)
                if record is not None and l in LAYERS:
                    record(f"{loop}/L{l+1}",h,False)
                if witness is not None and loop==1 and l in LAYERS:
                    witness.append(h[:4].detach().cpu().numpy())
            if record is not None:
                record(f"{loop}/pre_norm",h,False)
            h=op.norm(h) if method=="native" else op.norm_flow(h)
        if record is not None:
            record(f"{loop}/norm",h,True)
        if witness is not None and loop==1:
            witness.append(h[:4].detach().cpu().numpy())
    return h


def methods_for(phase):
    return DEVM if phase=="development" else ("persistence",)


def paths(out,phase):
    return out/f"PREDICTIONS_{phase}.json",out/f"PREDICTION_SHA_{phase}.json"


def predict_records(model,manifest,out,phase,methods,loops,steps,limit,keep_witness=False,truth=False):
    rows=[]
    timing=[]
    finite=[]
    with sdpa_kernel(SDPBackend.MATH):
        for family,groups in manifest["groups"][phase].items():
            groups=groups[:limit] if limit else groups
            for j,pack in enumerate(batches(groups,4)):
                enc=[masked(w) for g in pack for w in g["worlds"]]
                op=PairOps(model,enc)
                h=edit(op,pack)
                zero=sample(op,h,pack,False)
                data=[dict(group=g["group"],family=family,initial=zero[k],methods={})
                      for k,g in enumerate(pack)]
                for method in methods:
                    results=[{} for g in pack]
                    def keep(key,z,readout):
                        if not torch.isfinite(z).all():
                            raise FloatingPointError(key)
                        measured=sample(op,z,pack,readout)
                        for q,a in zip(results,measured,strict=True):
                            q[key]=a
                    witness=[] if keep_witness and j==0 and method in ("full","native") else None
                    if witness is not None:
                        witness.append(h[:4].float().cpu().numpy())
                    try:
                        with timed(op,phase+"/"+family+"/"+method,timing):
                            trajectory(op,h,method,loops,steps,record=keep,witness=witness)
                    except FloatingPointError as error:
                        finite.append(dict(family=family,method=method,failed_at=str(error),groups=[g["group"] for g in pack]))
                    for q,v in zip(data,results,strict=True):
                        q["methods"][method]=v
                    if witness is not None:
                        np.savez_compressed(out/f"WITNESS_{phase}_{family}_{method}.npz",
                            states=np.stack(witness).astype("float16"),ids=op.ids[:4].cpu().numpy())
                rows.extend(data)
                print("PHASE",phase,family,j+1,len(groups),flush=True)
    return dict(phase=phase,loops=loops,steps=steps,rows=rows,timing=timing,failures=finite,
                weights_sha256=weight_hash(model,torch.bfloat16))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("phase",choices=("pilot","development","validation","confirmation"))
    p.add_argument("--methods",nargs="+")
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--steps",type=int,default=2)
    p.add_argument("--loops",type=int,default=3)
    a=p.parse_args()
    model,_,runtime=load_runtime(a.model)
    model.requires_grad_(False).float()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=read_json(a.out/"MANIFEST.json")
    methods=a.methods or methods_for(a.phase)
    assert "native" not in methods
    result=predict_records(model,data,a.out,a.phase,methods,a.loops,a.steps,a.limit,a.phase=="development")
    target,receipt=paths(a.out,a.phase)
    write_json(target,result)
    write_json(receipt,dict(sha256=sha(target),methods=methods,hashes={
        k:sha(Path(__file__).parent/k) for k in ("predictor.py","pair_fields.py","norm_flow.py","causal_worlds.py")}))


if __name__=="__main__":
    main()
