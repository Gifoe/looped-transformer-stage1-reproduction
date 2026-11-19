"""Frozen-config refits without specified entity names or intermediate values."""
import argparse
from pathlib import Path

import numpy as np
import torch

from fit_study import (Data, Scale, TASKS, SPLITS, export, fit_system,
                       grouped_rows, load_lines, cosine, key, mse)
from manifest_org import NAMES
from summarize_transition import summary


def held_out(data):
    entities=set(NAMES[-4:])
    if data.task=="arithmetic":
        remove=np.array([any(int(v)%4==0 for v in g["intermediate_options"]) for g in data.groups])
        test=np.array([int(g["intermediate_options"][m])%4==0 for g,m in zip(data.groups,data.mid)])
        return remove,test,"intermediate multiples of four"
    if data.task=="logic":
        names=np.array([g["meta"]["signature"][0] for g in data.groups])
        both=np.isin(names,list(entities))
        return both,both,"unseen queried names "+", ".join(sorted(entities))
    context=[g["meta"]["signature"][0] for g in data.groups]
    remove=np.array([bool(entities.intersection(names)) for names in context])
    test=np.array([names[g["meta"]["queried"]] in entities for names,g in zip(context,data.groups)])
    return remove,test,"unseen queried names "+", ".join(sorted(entities))


def run(past,runs,results):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    output=[]
    for task in TASKS:
        data=Data(past,runs,task,2048)
        remove,test,rule=held_out(data)
        fit=data.data_mask("fit") & ~remove
        assert fit.sum()%4==0 and fit.sum()//4>=48
        scale=Scale(data.x[fit])
        train=scale.forward(data.x[fit])
        causal=load_lines(runs/task/"causal.jsonl")
        y=scale.forward(np.load(runs/task/"causal_full.npy",mmap_mode="r"))
        true=scale.downstream(y[:,1,1]-y[:,0,1])
        edits=np.array([data.ids[r["id"],r["donor"]] for r in causal])
        ctx=data.context[[data.ids[r["id"],r["world"]] for r in causal]]
        configs=(key("identity_raw",None,None,None),key("ridge_raw",None,None,3.),
                 key("rank24","pca",128,3.),key("mlp","auto",128,3.),
                 key("dictionary","auto",128,3.))
        for config in configs:
            m=config["method"]
            system=fit_system(config,train,scale,data.context[fit])
            response=scale.downstream(system(y[:,1,0],ctx)-system(y[:,0,0],ctx))
            for split in SPLITS:
                mask=data.data_mask(split)&test
                a,b=scale.forward(data.x[mask]).transpose(1,0,2)
                pred=system(a,data.context[mask])
                natural=np.array(data.rows,dtype=object)[mask]
                meta={"task":task,"split":split,"method":m,"rule":rule,"fit_groups":int(fit.sum()//4)}
                output.extend(grouped_rows(natural,{"mse":mse(scale.downstream(pred-b)),
                    "cosine":cosine(scale.downstream(pred-a),scale.downstream(b-a))},
                    {**meta,"arm":"natural"}))
                eligible=test[edits]&np.array([r["arm"]=="semantic" and r["split"]==split for r in causal])
                assert eligible.any()
                output.extend(grouped_rows(np.array(causal,dtype=object)[eligible],
                    {"mse":mse(response[eligible]-true[eligible]),"cosine":cosine(response[eligible],true[eligible])},
                    {**meta,"arm":"semantic"}))
        print(task,rule,"fit groups",fit.sum()//4,flush=True)
    export(results/"novel_inputs_by_group.csv",output)
    export(results/"novel_inputs_summary.csv",summary(output,
           ("task","split","method","rule","fit_groups","arm"),("mse","cosine")))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("past","runs","results"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    run(a.past,a.runs,a.results)
