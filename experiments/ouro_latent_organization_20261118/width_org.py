"""Check whether fixed random projection masks full-width intermediate information."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from analyze_org import export, group_ci, normalized
from capture_org import capture, positions, projection
from manifest_org import TASKS,dump_line,get_encoding,load_lines,write_json,digest
from collect import load_runtime

REGIONS=("source","mapping","readout")
SPLITS={"fit":120,"confirm":80,"transfer":80}


def decode(data,y,x_test):
    center=data.mean(0)
    ybar=y.mean(0)
    x=data-center
    ker=x@x.T
    ker.flat[::len(ker)+1]+=.75
    beta=np.linalg.solve(ker,y-ybar)
    return ((x_test-center)@x.T)@beta+ybar


def measure(raw,rows,task):
    records=[]
    splits=np.array([r["split"] for r in rows])
    learn=splits=="fit"
    truth=np.array([r["truth"] for r in rows])
    p=projection()
    if task=="arithmetic":
        y=truth.astype(np.float32)[:,None]
        opts=np.array([r["candidates"] for r in rows],np.float32)
    else:
        names=sorted({v for r in rows for v in r["candidates"]})
        ix={v:i for i,v in enumerate(names)}
        y=np.eye(len(names),dtype=np.float32)[[ix[v] for v in truth]]
        opts=np.array([[ix[v] for v in r["candidates"]] for r in rows])
    gold=np.array([r["gold_idx"] for r in rows])
    for k in range(4):
        for length in (0,1):
            for j,reg in enumerate(REGIONS):
                x=raw[:,k,length,j].astype(np.float32)
                for width in (96,2048):
                    xx=normalized(x@p if width==96 else x)
                    for s in ("confirm","transfer"):
                        test=splits==s
                        prediction=decode(xx[learn],y[learn],xx[test])
                        v=(-abs(prediction-opts[test]) if task=="arithmetic"
                           else np.take_along_axis(prediction,opts[test],axis=1))
                        top=np.argmax(v,axis=1)
                        correct=(top==gold[test])&((v==v.max(axis=1,keepdims=True)).sum(axis=1)==1)
                        avg,lo,hi=group_ci(correct.reshape(-1,2).mean(1))
                        records.append(dict(task=task,split=s,K=k+1,layer=(23,24)[length],role=reg,
                                            width=width,groups=int(test.sum()/2),
                                            accuracy=avg,lower=lo,upper=hi))
    return records


def run(root,modeldir):
    groups=load_lines(root/"groups.jsonl")
    assert (root/"analysis_freeze.json").exists()
    model,tok,runtime=load_runtime(modeldir)
    out=root/"width_control"
    out.mkdir()
    write_json(out/"runtime.json",runtime)
    all_rows=[]
    start=time.perf_counter()
    with torch.inference_mode():
        for task in TASKS:
            cohort=[]
            for split,n in SPLITS.items():
                matched=[g for g in groups if (g["task"],g["split"])==(task,split)]
                cohort+=sorted(matched,key=lambda g:hashlib.sha256(("width/"+g["id"]).encode()).digest())[:n]
            n_rows=2*len(cohort)
            storage=np.lib.format.open_memmap(out/f"{task}_bf16_bits.npy",mode="w+",dtype=np.int16,
                                             shape=(n_rows,4,2,len(REGIONS),2048))
            table=[]
            for g in cohort:
                for name in ("r","d"):
                    q=g["worlds"][name]
                    e=get_encoding(tok,g,q)
                    roles=positions(g,e)
                    seq=sorted({i for role in REGIONS for i in roles[role]})
                    scores,_,bank,_,coords=capture(model,e,seq,tap_layers={23})
                    grid=[]
                    for v in bank.values():
                        grid.append(torch.stack([v[[coords.index(p) for p in roles[role]]].float().mean(0)
                                                for role in REGIONS]))
                    exact=torch.stack(grid).reshape(4,2,len(REGIONS),2048).to(torch.bfloat16)
                    storage[len(table)]=exact.contiguous().view(torch.int16).cpu().numpy()
                    alt=g["worlds"]["d" if name=="r" else "r"]["semantic"]["intermediate"]
                    true=q["semantic"]["intermediate"]
                    opts=sorted([alt,true])
                    table.append(dict(id=g["id"],split=g["split"],world=name,truth=true,
                                      candidates=opts,gold_idx=opts.index(true),native_scores=scores))
            storage.flush()
            with (out/f"{task}_rows.jsonl").open("w",encoding="utf-8") as f:
                for r in table:
                    dump_line(f,r)
            exact=torch.from_numpy(np.array(storage)).view(torch.bfloat16).float().numpy()
            all_rows.extend(measure(exact,table,task))
            print(task,n_rows,round(time.perf_counter()-start,2),flush=True)
    torch.cuda.synchronize()
    export(out/"effects.csv",all_rows)
    write_json(out/"complete.json",{"groups_per_task":SPLITS,
       "phase_seconds_including_cpu_fit":time.perf_counter()-start,
       "peak_allocated_bytes":torch.cuda.max_memory_allocated()})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    p.add_argument("--model",type=Path,required=True)
    a=p.parse_args()
    run(a.study,a.model)
