"""Use the paper's pinned public attention metrics and GSM8K sample rule."""
import argparse
import importlib.util
from pathlib import Path
import random
import time

import numpy as np
import torch

from data_loop import rows,dump,jsonfile
from trace_loop import capture,load_runtime,digest
from readouts_loop import macro_profile,export

UPSTREAM="3429a8d61fdae95d441cb18eb3bd695268d086f4"


def run(checkpoint,gsm,author_attention,out):
    spec=importlib.util.spec_from_file_location("paper_attention_metrics",author_attention)
    metrics=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    model,tok,env=load_runtime(checkpoint)
    data=rows(gsm)
    indices=list(range(len(data)))
    random.Random(42).shuffle(indices)
    chosen=sorted(indices[:256])
    out.mkdir(parents=True)
    arrays=np.empty((2,len(chosen),4,24,3),dtype=np.float32)
    registry=model.model.layers[0].self_attn.forward.__func__.__globals__["ALL_ATTENTION_FUNCTIONS"]
    original=registry["sdpa"]
    captured=[]
    masks={}
    def interface(module,q,k,v,mask,**kwargs):
        n=q.shape[-2]
        raw=(q.float()@k.repeat_interleave(module.num_key_value_groups,dim=1).float().transpose(-2,-1))*kwargs["scaling"]
        if mask is None:
            if n not in masks:
                ix=torch.arange(n,device=q.device)
                masks[n]=ix[:,None]>=ix[None,:]
            raw=raw.masked_fill(~masks[n],float("-inf"))
        elif mask.dtype==torch.bool:
            raw=raw.masked_fill(~mask,float("-inf"))
        else:
            raw=raw+mask
        a=raw.softmax(-1)
        x=torch.stack((metrics.get_mixing_scores(a).mean(),metrics.get_colsum_concentrations(a).mean(),
                       metrics.get_mean_sink_rate(a).mean()))
        captured.append(x)
        return original(module,q,k,v,mask,**kwargs)
    start=time.perf_counter()
    registry["sdpa"]=interface
    try:
        with torch.inference_mode(),(out/"items.jsonl").open("w",encoding="utf-8") as f:
            for j,i in enumerate(chosen):
                row=data[i]
                p="Question: "+row["question"]+"\nAnswer:"
                for d,(tag,text) in enumerate((("teacher_forced",p+row["answer"]),("question_only",p))):
                    ids=tok.encode(text,add_special_tokens=False)
                    e={"ids":ids,"read":len(ids)-1,"target":[0]}
                    captured.clear()
                    result=capture(model,e,[],tap_layers=())
                    arrays[d,j]=torch.stack(captured).reshape(4,24,3).cpu().numpy()
                    if j==0:
                        registry["sdpa"]=original
                        ref=capture(model,e,[],tap_layers=())
                        assert torch.equal(result[1],ref[1])
                        registry["sdpa"]=interface
                    dump(f,{"id":i,"rank":j,"protocol":tag,"ids":ids,"tokens":len(ids)})
                if (j+1)%32==0:
                    f.flush()
                    print("paper macro",j+1,round(time.perf_counter()-start,2),flush=True)
    finally:
        registry["sdpa"]=original
    np.save(out/"paper_macro.npy",arrays)
    table=[]
    for d,p in enumerate(("teacher_forced","question_only")):
        table+=macro_profile(arrays[d],np.arange(len(chosen)),"gsm8k",p,"upstream_seed42")
    export(out/"profile.csv",table)
    jsonfile(out/"complete.json",{"paper":"https://arxiv.org/abs/2604.11791v2","upstream_sha":UPSTREAM,
             "upstream_metric_sha256":digest(author_attention),"source_sha256":digest(gsm),
             "selected":chosen,"seconds":time.perf_counter()-start,"runtime":env,
             "outputs_sha256":{p.name:digest(p) for p in out.iterdir()}})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("checkpoint","gsm","author_attention","out"):
        p.add_argument(name,type=Path)
    a=p.parse_args()
    run(a.checkpoint,a.gsm,a.author_attention,a.out)
