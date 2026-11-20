"""Short native-conditioned segments and matched native four-world controls."""
import argparse
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from causal_worlds import masked, sha
from pair_fields import PairOps, pair, rel, timed, load_runtime
from dynamics_embedding import cosine, norm, read_json, write_json
from predictor import batches, edit, sample, trajectory


def short(model,m,out):
    timing=[]
    rows=[]
    for f,items in m["groups"]["development"].items():
        for pack in batches(items[:8],4):
            op=PairOps(model,[masked(w) for g in pack for w in g["worlds"]])
            initial=edit(op,pack)
            x=initial
            for l in range(5):
                x=pair(op,l,x,"native")
            for depth in (2,4):
                data={}
                for name in ("base","self","order","full","reverse","taylor","prefix_anderson","native"):
                    with timed(op,"short/"+f+"/"+name,timing):
                        h=x
                        for l in range(5,5+depth):
                            h=pair(op,l,h,name,steps=2)
                    data[name]=h
                for name in data:
                    y=data[name]
                    ref=data["native"]
                    rows.append(dict(family=f,groups=[g["group"] for g in pack],layer=6,span=depth,name=name,
                        pre_norm_error=rel(op,y,ref).tolist(),
                        norm_probe_error=rel(op,op.norm_flow(y),op.norm(ref)).tolist(),
                        direction=cosine(op,y-x,ref-x).tolist(),
                        update_l2=norm(op,y-x).tolist(),
                        state=sample(op,y,pack,False)))
    write_json(out/"SHORT.json",dict(rows=rows,timing=timing,description="same true native K2 prefix, no resets inside span"))


def controls(model,m,out,phase):
    # All corresponding intervention predictions are already signed.
    pred=out/f"PREDICTIONS_{phase}.json"
    assert sha(pred)==read_json(out/f"PREDICTION_SHA_{phase}.json")["sha256"]
    rows=[]
    timing=[]
    for f,groups in m["groups"][phase].items():
        for pack in batches(groups,4):
            op=PairOps(model,[masked(w) for g in pack for w in g["worlds"]])
            data=[dict(group=g["group"],family=f,conditions={}) for g in pack]
            for k in ("intact","random","unrelated"):
                h=op.start if k=="intact" else edit(op,pack,control=k)
                doses=norm(op,h-h[::4].repeat_interleave(4,dim=0)).tolist()
                for j,d in enumerate(data):
                    d["conditions"][k]={"input_dose":doses[4*j:4*j+4]}
                def note(key,z,readout):
                    if not readout:
                        return
                    a=sample(op,z,pack,True)
                    for d,r in zip(data,a,strict=True):
                        d["conditions"][k][key]=r
                with timed(op,"control/"+f+"/"+k,timing):
                    trajectory(op,h,"native",3,1,record=note)
            rows.extend(data)
    write_json(out/"CONTROLS.json",dict(rows=rows,timing=timing,phase=phase,prediction_sha256=sha(pred)))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("stage",choices=("short","controls"))
    p.add_argument("phase",nargs="?",default="confirmation")
    a=p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,_,_=load_runtime(a.model)
    model.requires_grad_(False).float()
    m=read_json(a.out/"MANIFEST.json")
    with sdpa_kernel(SDPBackend.MATH):
        if a.stage=="short":
            short(model,m,a.out)
        else:
            controls(model,m,a.out,a.phase)


if __name__=="__main__":
    main()
