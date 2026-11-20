"""Fit and test a causal cross-token endpoint baseline on the same split."""
import argparse
from pathlib import Path

import numpy as np
import torch

from dynamics_embedding import EmbeddingOperators, compare, read_json, timed, write_json
from embedding_experiments import causal_compare, patch_state
from embedding_run import load_runtime


def features(x):
    h=x.float()
    length=torch.arange(1,h.shape[1]+1,device=h.device)[None,:,None]
    return torch.cat((h,h.cumsum(1)/length),dim=-1)


def fit(model,groups,out):
    xs=[]
    ys=[]
    for g in groups.values():
        op=EmbeddingOperators(model,g["dev"],2)
        xs.append(features(op.k1)[op.valid])
        ys.append((op.k2-op.k1)[op.valid])
    x,y=torch.cat(xs),torch.cat(ys)
    centre=x.mean(0)
    centre_y=y.mean(0)
    scale=x.std(0).clamp_min(.05)
    a=(x-centre)/scale
    gram=a@a.T
    penalty=.08*float(gram.diag().mean())
    gram.diagonal().add_(penalty)
    w=a.T@torch.linalg.solve(gram,y-centre_y)
    np.savez_compressed(out/"prefix_ridge.npz",weight=w.cpu().numpy(),centre=centre.cpu().numpy(),
                        scale=scale.cpu().numpy(),offset=centre_y.cpu().numpy(),penalty=penalty,
                        train_tokens=len(x))

    def predict(h):
        return h.float()+(features(h)-centre)/scale@w+centre_y
    return predict


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    manifest=read_json(a.out/"MANIFEST.json")
    arcs=read_json(a.out/"ARC_CAUSAL_MANIFEST.json")
    manifest["arc"]["causal"]=arcs
    dst=a.out/"prefix_control"
    dst.mkdir(exist_ok=True,parents=True)
    model,_,_=load_runtime(a.model)
    model.requires_grad_(False)
    model.float()
    predictor=fit(model,manifest,dst)
    for dtype in (torch.float32,torch.bfloat16):
        model.to(dtype=dtype)
        dt=str(dtype).split(".")[-1]
        for f,g in manifest.items():
            natural=[]
            cf=[]
            log=[]
            raws=read_json(a.out/f"causal_{dt}_{f}.json")
            for depth in (2,3):
                op=EmbeddingOperators(model,g["test"],depth)
                with timed(op,f"K{depth}/prefix_ridge",log):
                    state=predictor(op.k1)
                natural.append(dict(loop=depth,data=compare(op,state,op.k2,op.k1)))
                for (rec,src,rule),receipt in zip(g["causal"],
                      (q for q in raws["runs"] if q["loop"]==depth),strict=True):
                    assert receipt["items"]==[w["item"] for w in (rec,src,rule)]
                    batch=EmbeddingOperators(model,[rec,src,rule]+[rec]*4,depth)
                    start,_=patch_state(batch,rec,src,rule)
                    actual=batch.native_suffix(start)
                    with timed(batch,f"causal/K{depth}/prefix_ridge",log):
                        output=predictor(start)
                    q=causal_compare(batch,output,actual,receipt["targets"],receipt["slot_to_content"])
                    assert q["true_scores"]==receipt["methods"]["chart_anderson"]["true_scores"]
                    cf.append(dict(loop=depth,items=receipt["items"],data=q))
            write_json(dst/f"{dt}_{f}.json",dict(natural=natural,causal=cf,timing=log))


if __name__=="__main__":
    main()
