"""Read-opportunity scope check on reserved group IDs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from data_loop import TASKS, encoding, rows, dump, jsonfile, hash_key
from trace_loop import load_runtime, projection, digest
from route_loop import event
from readouts_loop import partition,export,ci


def run(checkpoint,root,task):
    path=root/"data"/"groups.jsonl"
    groups=[g for g in rows(path) if g["task"]==task]
    windows=json.loads((root/"freeze.json").read_text())["windows"][task]["open"]["2"]
    out=root/"scope"/task
    out.mkdir(parents=True)
    selected=[]
    for split in ("confirm","transfer"):
        matching=sorted((g for g in groups if g["split"]==split),key=lambda g:hash_key("route-independent-v1/"+g["id"]))
        selected.extend(matching[48:72])
    model,tok,env=load_runtime(checkpoint)
    frame=torch.from_numpy(projection()[:,:48].copy()).cuda()
    tiers={"early":windows,"complement":[i for i in range(24) if i not in windows],
           "whole":list(range(24))}
    vals=[]
    started=time.perf_counter()
    with torch.inference_mode(),(out/"scores.jsonl").open("w",encoding="utf-8") as f:
        for i,g in enumerate(selected):
            r=encoding(tok,g,"r","open")
            x=encoding(tok,g,"x","open")
            edited=[j for j,(u,v) in enumerate(zip(r["ids"],x["ids"])) if u!=v]
            nuisance=r["roles"]["noise"][-len(edited):]
            dst=r["roles"]["mapping"]
            assert max(edited+nuisance)<min(dst) and len(nuisance)==len(edited)
            y0,y1=(g["worlds"][w]["gold"] for w in ("r","x"))
            def gap(res,h):
                return res[0][h-1][y0]-res[0][h-1][y1]
            for tier,layer in tiers.items():
                opt=(2,layer)
                clean=event(model,r,frame,window=opt,tap=(23,) if tier=="whole" else ())
                edited_trace=event(model,r,frame,window=opt,dest=dst,source=edited)
                variants={"sham":clean,"cut":edited_trace}
                if tier=="whole":
                    variants["noise"]=event(model,r,frame,window=opt,dest=dst,source=nuisance)
                    variants["restore"]=event(model,r,frame,window=opt,dest=dst,source=edited,
                                  edit={(2,23):(dst,clean[2][2,23][dst])})
                dump(f,{"id":g["id"],"split":g["split"],"task":task,"tier":tier,"layers":layer,
                       "scores":{k:q[0] for k,q in variants.items()},"source_keys":edited,"nuisance_keys":nuisance})
                for k in (2,4):
                    effect={"cost":gap(clean,k)-gap(edited_trace,k)}
                    if tier=="whole":
                        effect["over_noise"]=gap(variants["noise"],k)-gap(edited_trace,k)
                        effect["rescue"]=gap(variants["restore"],k)-gap(edited_trace,k)
                    vals.extend(dict(task=task,split=g["split"],id=g["id"],tier=tier,output_loop=k,
                                     metric=m,value=v) for m,v in effect.items())
            if (i+1)%12==0:
                f.flush()
                print(task,i+1,round(time.perf_counter()-started,2),flush=True)
    export(out/"by_group.csv",vals)
    summary=[]
    columns=("task","split","tier","output_loop","metric")
    for key,block in partition(vals,columns).items():
        m,lo,hi=ci([r["value"] for r in block])
        summary.append({**dict(zip(columns,key)),"groups":len(block),"mean":m,"lower":lo,"upper":hi})
    export(out/"summary.csv",summary)
    jsonfile(out/"complete.json",{"groups":len(selected),"seconds":time.perf_counter()-started,
        "manifest":digest(path),"protocol":digest(Path(__file__).parent/"FOLLOWUP_PROTOCOL.md"),
        "script":digest(Path(__file__)),"runtime":env,"files":{p.name:digest(p) for p in out.iterdir()}})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    p.add_argument("root",type=Path)
    p.add_argument("task",choices=TASKS)
    a=p.parse_args()
    run(a.checkpoint,a.root,a.task)
