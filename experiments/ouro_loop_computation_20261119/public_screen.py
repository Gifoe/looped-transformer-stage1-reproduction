"""Public-data development screen and Eq. 3 attention reproduction."""
import argparse
from decimal import Decimal
from pathlib import Path
import re
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from data_loop import rows, dump, jsonfile, hash_key
from trace_loop import capture, digest, load_runtime, sdpa_tap


def choose(items,n,namespace):
    return sorted(items,key=lambda r:hash_key(f"loop-public-v1/{namespace}/{r['id']}"))[:n]


def public_tasks(data,gsm,org):
    arc=[]
    for r in pq.read_table(data/"arc.parquet").to_pylist():
        arc.append({"id":r["id"],"prompt":"Question: "+r["question"]+"\nAnswer:",
                    "choices":r["choices"]["text"],"gold":r["choices"]["label"].index(r["answerKey"])})
    hella=[]
    for r in pq.read_table(data/"hellaswag.parquet").to_pylist():
        hella.append({"id":str(r["ind"]),"prompt":r["ctx"],"choices":r["endings"],"gold":int(r["label"])})
    proof=[]
    for g in rows(org):
        if g["task"]!="prontoqa":
            continue
        w=g["worlds"]["r"]
        start=w["roles"]["option"][0][0]
        proof.append({"id":g["id"],"prompt":w["prompt"][:start].rstrip()+"\nAnswer:",
                      "choices":g["options"],"gold":w["gold"]})
    math=[]
    for i,r in enumerate(rows(gsm)):
        gold=Decimal(r["answer"].rsplit("####",1)[-1].strip().replace(",",""))
        math.append({"id":f"gsm8k/test/{i:04}","prompt":"Question: "+r["question"]+"\nAnswer:",
                     "gold_number":str(gold),"choices":list(map(str,(gold-3,gold,gold+1,gold+5))),"gold":1})
    return {"arc":arc,"hellaswag":hella,"prontoqa":proof,"gsm8k":math}


def likelihood(model,tok,prompt,answers):
    output=[]
    for option in answers:
        a=tok.encode(prompt,add_special_tokens=False)
        b=tok.encode(prompt+" "+option,add_special_tokens=False)
        common=0
        for t,j in zip(a,b):
            if t!=j:
                break
            common+=1
        assert common>0 and common<len(b)
        end=b[:-1]
        suffix=b[common:]
        selected=list(range(common-1,len(end)))
        e={"ids":end,"read":len(end)-1,"target":[suffix[-1]]}
        _,_,bank,_,idx=capture(model,e,selected,tap_layers=())
        query=torch.stack([bank[k,24][[idx.index(i) for i in selected]] for k in range(1,5)])
        all_logprobs=model.lm_head(query).float().log_softmax(dim=-1)
        score=all_logprobs.gather(-1,torch.tensor(suffix,device=query.device).view(1,-1,1).expand(4,-1,-1))
        output.append(score.sum(1)[:,0].cpu().tolist())
    return np.array(output).T.tolist()


def greedy_number(model,tok,item,k):
    prefix=tok.encode(item["prompt"],add_special_tokens=False)
    complete=[]
    for _ in range(8):
        full=prefix+complete
        e={"ids":full,"read":len(full)-1,"target":[0]}
        out=capture(model,e,[],depth=k,tap_layers=())
        complete.append(int(out[1][-1].argmax()))
        text=tok.decode(complete,skip_special_tokens=False)
        if re.match(r"\s*(-?[\d,]+)([^\d,]|$)",text) and not re.match(r"\s*-?[\d,]+$",text):
            break
        if "\n" in text.strip() or (len(text.strip())>=10 and not re.match(r"\s*[-\d]",text)):
            break
    match=re.match(r"\s*(-?\d[\d,]*(?:\.\d+)?)",text)
    ident=str(Decimal(match[1].replace(",",""))) if match else None
    return text,ident


def run(checkpoint,public,gsm,org,out):
    out.mkdir(parents=True)
    data=public_tasks(public,gsm,org)
    model,tok,env=load_runtime(checkpoint)
    start=time.perf_counter()
    with torch.inference_mode(),(out/"screen.jsonl").open("w",encoding="utf-8") as f:
        for name,items in data.items():
            for i,g in enumerate(choose(items,64,name)):
                scores=likelihood(model,tok,g["prompt"],g["choices"])
                result={**g,"task":name,"scores":scores}
                if name=="gsm8k" and i<32:
                    result["greedy"]=[greedy_number(model,tok,g,k) for k in (1,2,3,4)]
                dump(f,result)
            print(name,"screen",flush=True)
    g=choose(data["gsm8k"],256,"gsm8k-macro")
    macro=np.empty((len(g),4,24,3),np.float32)
    with torch.inference_mode(),(out/"macro_rows.jsonl").open("w",encoding="utf-8") as f:
        for i,item in enumerate(g):
            ids=tok.encode(item["prompt"],add_special_tokens=False)
            e={"ids":ids,"read":len(ids)-1,"target":[0]}
            q=[]
            with sdpa_tap(model,q):
                score,head,_,_,_=capture(model,e,[],tap_layers=())
            macro[i]=torch.stack(q).reshape(4,24,3).cpu().numpy()
            if i==0:
                native=capture(model,e,[],tap_layers=())
                assert torch.equal(head,native[1])
            dump(f,{"i":i,"id":item["id"],"prompt":item["prompt"],"tokens":len(ids)})
    np.save(out/"macro.npy",macro)
    jsonfile(out/"runtime.json",{"runtime":env,"macro_examples":len(g),"benchmark_screen_examples":64,
             "greedy_gsm_examples":32,"seconds":time.perf_counter()-start,
             "source_sha256":{p.name:digest(p) for p in (public/"arc.parquet",public/"hellaswag.parquet",gsm,org)}})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("checkpoint","public","gsm","org","out"):
        p.add_argument(key,type=Path)
    a=p.parse_args()
    run(a.checkpoint,a.public,a.gsm,a.org,a.out)
