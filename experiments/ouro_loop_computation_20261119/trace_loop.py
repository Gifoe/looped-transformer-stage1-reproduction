"""Observe native SDPA and save matched Loop × Layer semantic traces."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import sys
import time

import numpy as np
import torch

from data_loop import TASKS, WORLDS, SITES, encoding, rows, dump, jsonfile, hash_key

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"ouro_latent_organization_20261118"))
from capture_org import projection, capture
from manifest_org import digest
from collect import load_runtime

WIDTH=48
LAYERS=24


def norm_at(x,norm):
    variance=x.float().square().mean(-1,keepdim=True)
    return (x.float()*(variance+norm.variance_epsilon).rsqrt()).to(x.dtype)*norm.weight


@contextmanager
def sdpa_tap(model, stats):
    """Reconstruct probabilities from the already rotary-transformed Q and K."""
    first=model.model.layers[0].self_attn
    registry=first.forward.__func__.__globals__["ALL_ATTENTION_FUNCTIONS"]
    original=registry["sdpa"]
    cache={}
    def interface(module,q,k,v,mask,**kw):
        n=q.shape[-2]
        assert n==k.shape[-2] and q.shape[0]==1
        if n not in cache:
            steps=torch.arange(n,device=q.device)
            cache[n]=(steps[:,None]>=steps[None,:],torch.arange(2,n+1,device=q.device).float().log())
        causal,den=cache[n]
        predicted=(q.float()@k.repeat_interleave(module.num_key_value_groups,dim=1).float().transpose(-2,-1))*kw["scaling"]
        if mask is None:
            predicted=predicted.masked_fill(~causal,float("-inf"))
        elif mask.dtype==torch.bool:
            predicted=predicted.masked_fill(~mask,float("-inf"))
        else:
            predicted=predicted+mask
        probabilities=torch.softmax(predicted,dim=-1)
        log=probabilities.clamp_min(1e-30).log()
        entropy=-torch.sum(probabilities*log,dim=-1)
        mix=(entropy[...,1:]/den).mean()
        column=probabilities.mean(-2)
        concentration=1+(column*column.clamp_min(1e-30).log()).sum(-1).mean()/np.log(n)
        sink=(probabilities[...,1:,0]>.3).float().mean()
        stats.append(torch.stack((mix,concentration,sink)))
        return original(module,q,k,v,mask,**kw)
    registry["sdpa"]=interface
    try:
        yield
    finally:
        registry["sdpa"]=original


def summarize_trace(model,e,frame,mode):
    tracked=sorted({i for key in SITES for i in e["roles"][key]})
    event_stats=[]
    if mode:
        with sdpa_tap(model,event_stats):
            result=capture(model,e,tracked)
        assert len(event_stats)==96
    else:
        result=capture(model,e,tracked)
    scores,heads,bank,_,indices=result
    state=torch.stack(list(bank.values()))
    assert state.shape[0]==100
    mask=torch.zeros((5,len(indices)),device=state.device)
    for j,key in enumerate(SITES):
        role=e["roles"][key]
        mask[j,[indices.index(i) for i in role]]=1/len(role)
    means=(mask@state.float()).reshape(4,25,5,2048)
    projected=(means@frame).half().cpu().numpy()
    boundary=means[:,24].cpu().numpy()
    x=state[:,indices.index(e["read"])]
    reshaped=x.reshape(4,25,2048)
    fixed=torch.cat((norm_at(reshaped[:,:24],model.model.norm),reshaped[:,24:]),dim=1)
    logits=(fixed@model.lm_head.weight[e["target"]].T).float()
    # Retain the exact native full-vocabulary head at loop boundaries.
    # Candidate-only BF16 GEMMs can fall on different rounding boundaries.
    logits[:,24]=heads[:,e["target"]].float()
    overlap=torch.nn.functional.cosine_similarity(means[1:],means[:-1],dim=-1)
    geometry=torch.stack((means.square().mean(-1).sqrt(),
                          torch.cat((torch.full_like(overlap[:1],float("nan")),overlap))),dim=-1)
    attention=torch.stack(event_stats).reshape(4,24,3).cpu().numpy() if mode else None
    return scores,heads.argmax(-1).cpu().tolist(),projected,boundary,logits.cpu().numpy(),geometry.cpu().numpy(),attention


def complete_scores(model,e,scores,top,gold):
    if len(e["answer_tokens"][0])==1:
        return scores,[v==e["target"][gold] for v in top]
    total=np.array(scores)
    hit=[False]*4
    for j,(a,b) in enumerate(e["answer_tokens"]):
        follow={"ids":[*e["ids"],a],"read":len(e["ids"]),"target":[b]}
        sc,logits,_,_,_=capture(model,follow,[],tap_layers=())
        total[:,j]+=np.asarray(sc)[:,0]
        if j==gold:
            hit=[v==a and q==b for v,q in zip(top,logits.argmax(-1).tolist())]
    return total.tolist(),hit


def run(model_path,inputs,out,task,phase):
    all_groups=[r for r in rows(inputs) if r["task"]==task]
    split=("fit","discover") if phase=="development" else ("confirm","transfer")
    groups=[r for r in all_groups if r["split"] in split]
    if phase=="pilot":
        groups=all_groups[:2]
    out.mkdir(parents=True)
    model,tok,runtime=load_runtime(model_path)
    frame=torch.from_numpy(projection()[:,:WIDTH].copy()).cuda()
    n=8*len(groups)
    v=np.lib.format.open_memmap(out/"trajectory.npy",mode="w+",dtype=np.float16,shape=(n,4,25,5,WIDTH))
    b=np.lib.format.open_memmap(out/"boundary.npy",mode="w+",dtype=np.float32,shape=(n,4,5,2048))
    l=np.lib.format.open_memmap(out/"lens.npy",mode="w+",dtype=np.float32,shape=(n,4,25,4))
    a=np.lib.format.open_memmap(out/"attention.npy",mode="w+",dtype=np.float32,shape=(n,4,24,3))
    z=np.lib.format.open_memmap(out/"geometry.npy",mode="w+",dtype=np.float32,shape=(n,4,25,5,2))
    a.fill(np.nan)
    start=time.perf_counter()
    j=0
    with torch.inference_mode(),(out/"rows.jsonl").open("w",encoding="utf-8") as f:
        for i,g in enumerate(groups):
            for p in ("choice","open"):
                for w in WORLDS[:4]:
                    e=encoding(tok,g,w,p)
                    exact=w=="r"
                    scores,top,v[j],b[j],l[j],z[j],metrics=summarize_trace(model,e,frame,exact)
                    if exact:
                        a[j]=metrics
                    if i==0 and w=="r":
                        ref=capture(model,e,[],tap_layers=())
                        assert scores==ref[0]
                        for k in (1,2,3,4):
                            separate=capture(model,e,[],tap_layers=(),depth=k)
                            assert scores[k-1]==separate[0][-1]
                    fact=g["worlds"][w]
                    full,exact_answer=complete_scores(model,e,scores,top,fact["gold"])
                    dump(f,{"j":j,"id":g["id"],"split":g["split"],"task":task,"protocol":p,
                           "world":w,"ids":e["ids"],"roles":e["roles"],"target_ids":e["target"],
                           "S":fact["S"],"M":fact["M"],"Y":fact["Y"],"gold":fact["gold"],
                           "scores":scores,"complete_scores":full,"unconstrained_top_token":top,
                           "first_answer_matches":exact_answer,"has_attention":exact})
                    j+=1
            if (i+1)%24==0:
                f.flush()
                print(task,phase,i+1,round(time.perf_counter()-start,2),flush=True)
    assert j==n
    for x in (v,b,l,a,z):
        x.flush()
    jsonfile(out/"complete.json",{"task":task,"phase":phase,"groups":len(groups),"rows":n,
           "manifest_sha256":digest(inputs),"implementation_sha256":digest(Path(__file__)),
           "seconds":time.perf_counter()-start,"runtime":runtime,
           "files":{file.name:{"sha256":digest(file),"bytes":file.stat().st_size} for file in out.iterdir()}})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("checkpoint","inputs","out"):
        p.add_argument(key,type=Path)
    p.add_argument("task",choices=TASKS)
    p.add_argument("phase",choices=("pilot","development","independent"))
    x=p.parse_args()
    run(x.checkpoint,x.inputs,x.out,x.task,x.phase)
