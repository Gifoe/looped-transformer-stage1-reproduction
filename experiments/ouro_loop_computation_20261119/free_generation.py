"""Unrestricted short greedy answers on a predeclared hashed subset."""
import argparse
from collections import defaultdict
from pathlib import Path
import re
import time

import torch

from data_loop import TASKS,encoding,hash_key,rows,dump,jsonfile
from trace_loop import load_runtime,digest
from readouts_loop import export,partition,ci


def answer(text,task,options):
    value=text.lstrip().split("\n",1)[0].strip()
    value=re.sub(r"(?i)^(?:answer\s*:\s*|the answer is\s*:?[ \t]*|it is\s+)","",value)
    if task=="arithmetic":
        m=re.match(r"^-?\d[\d,]*(?:\.\d+)?",value)
        return m[0].replace(",","") if m else None
    value=re.sub(r"(?i)^(?:a|an|the)\s+","",value)
    for w in options:
        if re.match(r"^"+re.escape(w)+r"\b",value,flags=re.IGNORECASE):
            return w
    return None


def run(checkpoint,root):
    data=rows(root/"data"/"groups.jsonl")
    selected=[]
    for t in TASKS:
        for split in ("confirm","transfer"):
            same=[g for g in data if g["task"]==t and g["split"]==split]
            selected.extend(sorted(same,key=lambda g:hash_key("free-generation-v1/"+g["id"]))[:24])
    model,tok,env=load_runtime(checkpoint)
    folder=root/"free_generation"
    folder.mkdir()
    lengths=defaultdict(list)
    for g in selected:
        e=encoding(tok,g,"r","open")
        lengths[len(e["ids"])].append((g,e["ids"]))
    start=time.perf_counter()
    examples=[]
    base=model.model.total_ut_steps
    try:
        with torch.inference_mode():
            for k in (1,2,3,4):
                model.model.total_ut_steps=k
                for count,qs in lengths.items():
                    for j in range(0,len(qs),8):
                        batch=qs[j:j+8]
                        x=torch.tensor([r[1] for r in batch],device="cuda")
                        for step in range(6):
                            o=model(x,use_cache=False,exit_at_step=k-1,logits_to_keep=1).logits
                            y=o[:,0].argmax(1)
                            x=torch.cat((x,y[:,None]),dim=1)
                        for (g,_),suffix in zip(batch,x[:,count:].tolist()):
                            text=tok.decode(suffix,skip_special_tokens=True)
                            choice=answer(text,g["task"],g["options"])
                            gold=g["worlds"]["r"]["Y"]
                            examples.append(dict(task=g["task"],split=g["split"],id=g["id"],
                                loop=k,generated_ids=suffix,generated=text,parsed=choice,gold=gold,
                                correct=int(choice==gold)))
                print("greedy",k,len(selected),round(time.perf_counter()-start,2),flush=True)
    finally:
        model.model.total_ut_steps=base
    with (folder/"predictions.jsonl").open("w",encoding="utf-8") as f:
        for r in examples:
            dump(f,r)
    out=[]
    for key,items in partition(examples,("task","split","loop")).items():
        m,lo,hi=ci([r["correct"] for r in items])
        out.append(dict(task=key[0],split=key[1],loop=key[2],groups=len(items),
                        mean=m,lower=lo,upper=hi))
    export(folder/"summary.csv",out)
    jsonfile(folder/"complete.json",{"groups":len(selected),"tokens_per_answer":6,"seconds":time.perf_counter()-start,
          "runtime":env,"inputs_sha256":digest(root/"data"/"groups.jsonl"),
          "script_sha256":digest(Path(__file__)),"files":{p.name:digest(p) for p in folder.iterdir()}})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.checkpoint,a.root)
