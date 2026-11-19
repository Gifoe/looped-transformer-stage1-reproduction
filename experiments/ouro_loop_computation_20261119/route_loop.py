"""Window-local SDPA cuts, same-receiver rescue and inter-loop map interchange."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import time

import numpy as np
import torch

from data_loop import TASKS,SITES,encoding,rows,dump,jsonfile,hash_key
from trace_loop import capture,projection,digest,load_runtime


@contextmanager
def mask_window(model,k,layer_set,dest,source):
    hooks=[]
    def hook(module,args,kw):
        if kw["current_ut"]+1!=k:
            return
        h=kw["hidden_states"]
        n=h.shape[-2]
        mask=kw["attention_mask"]
        if mask is None:
            assert not module.sliding_window
            causal=torch.arange(n,device=h.device)
            valid=causal[:,None]>=causal[None,:]
            mask=torch.zeros((1,1,n,n),device=h.device,dtype=h.dtype)
            mask.masked_fill_(~valid,torch.finfo(h.dtype).min)
        else:
            mask=mask.clone()
        if source:
            a=torch.tensor(dest,device=h.device)[:,None]
            b=torch.tensor(source,device=h.device)[None,:]
            mask[...,a,b]=False if mask.dtype==torch.bool else torch.finfo(mask.dtype).min
        return args,{**kw,"attention_mask":mask}
    try:
        for i in layer_set:
            hooks.append(model.model.layers[i].self_attn.register_forward_pre_hook(hook,with_kwargs=True))
        yield
    finally:
        for h in hooks:
            h.remove()


def event(model,e,frame,*,window=None,dest=(),source=(),edit=None,tap=()):
    args={"edits":edit,"full":True,"tap_layers":set(tap)}
    if window is None:
        held=capture(model,e,[],**args)
    else:
        with mask_window(model,*window,dest,source):
            held=capture(model,e,[],**args)
    score,head,bank,_,_=held
    role=torch.stack([torch.stack([bank[k,24][e["roles"][r]].float().mean(0) for r in SITES])
                      for k in (1,2,3,4)])
    top=np.argmax(score,axis=1).tolist()
    trace=(role@frame).cpu().numpy()
    return score,head,bank,top,trace


def norm_delta(x,y):
    return (x.float()-y.float()).norm()


def matched(base,delta,n):
    return (base.float()+delta*n/delta.norm().clamp_min(1e-10)).to(base.dtype)


def cohort(groups,phase):
    if phase=="pilot":
        return [(groups[0],"open")]
    out=[]
    for s in ("confirm","transfer"):
        ordered=sorted((g for g in groups if g["split"]==s),
                       key=lambda g:hash_key("route-independent-v1/"+g["id"]))
        out.extend((g,"open") for g in ordered[:48])
        out.extend((g,"choice") for g in ordered[:16])
    return out


def run_group(model,tok,g,protocol,lock,frame,checks):
    es={w:encoding(tok,g,w,protocol) for w in ("r","x","u","d","n")}
    mapping=es["r"]["roles"]["mapping"]
    q=sorted(set(es["r"]["roles"]["query"]+es["r"]["roles"]["readout"]))
    changed=[i for i,(a,b) in enumerate(zip(es["r"]["ids"],es["x"]["ids"])) if a!=b]
    assert set(changed)<=set(es["r"]["roles"]["source"]) and max(changed)<min(mapping)
    distract=sorted(es["r"]["roles"]["noise"])
    noise=distract[-len(changed):]
    assert len(noise)==len(changed) and max(noise)<min(mapping)
    assert g["worlds"]["r"]["Y"]==g["worlds"]["u"]["Y"]
    assert g["worlds"]["r"]["Y"]!=g["worlds"]["x"]["Y"]
    for key in ("r","x"):
        assert es[key]["roles"]==es["r"]["roles"]
    windows=lock["windows"][g["task"]][protocol]
    bank={w:event(model,es[w],frame) for w in es}
    if checks:
        w="x"
        ids=list(range(len(es["r"]["ids"])))
        swapped=event(model,es["r"],frame,edit={(1,24):(ids,bank[w][2][1,24])})
        assert torch.equal(swapped[1],bank[w][1])
        self_swap=event(model,es["r"],frame,edit={(1,24):(mapping,bank["r"][2][1,24][mapping])})
        assert torch.equal(self_swap[1],bank["r"][1])
    result=[]
    def add(kind,loop,receiver,arm,capture_result,site=None,dose=None):
        score,head,b,top,feature=capture_result
        result.append(({"kind":kind,"loop":loop,"receiver":receiver,"arm":arm,
                    "scores":score,"choice":top,"window":site,"dose":dose},feature))
    for w in bank:
        add("native",0,w,w,bank[w])
    for k in (2,3,4):
        layers=windows[str(k)]
        finish=layers[-1]
        site=(k,layers)
        for w in ("r","x"):
            recipient=es[w]
            sham=event(model,recipient,frame,window=site,tap=(finish,))
            cut=event(model,recipient,frame,window=site,dest=mapping,source=changed)
            assert torch.equal(cut[1][:k-1],sham[1][:k-1])
            add("route",k,w,"sham",sham,layers)
            add("route",k,w,"cut",cut,layers)
            if w=="r":
                alt=event(model,recipient,frame,window=site,dest=mapping,source=noise)
                recovery=event(model,recipient,frame,window=site,dest=mapping,source=changed,
                    edit={(k,finish):(mapping,sham[2][k,finish][mapping])})
                direct=event(model,recipient,frame,window=site,dest=q,source=changed)
                for a,h in (("noise_cut",alt),("rescue",recovery),("direct_cut",direct)):
                    add("route",k,w,a,h,layers)
        source=bank["r"][2][k-1,24][mapping]
        true=bank["x"][2][k-1,24][mapping]
        dose=float(norm_delta(true,source))
        rng=torch.Generator(device="cuda").manual_seed(int.from_bytes(
            hash_key(f"random-map/{g['id']}/{protocol}/{k}")[:8],"big"))
        variants={"matched":true,"same_Y":bank["u"][2][k-1,24][mapping],
                  "mismatched":bank["d"][2][k-1,24][mapping],
                  "irrelevant":bank["n"][2][k-1,24][mapping]}
        variants["random"]=torch.randn(source.shape,generator=rng,device=source.device)
        for name,v in variants.items():
            delta=v if name=="random" else v.float()-source.float()
            value=v if name=="matched" else matched(source,delta,dose)
            patch=event(model,es["r"],frame,edit={(k-1,24):(mapping,value)})
            assert torch.equal(patch[1][:k-1],bank["r"][1][:k-1])
            add("interchange",k,"r",name,patch,dose=float(norm_delta(value,source)))
    return result


def run(checkpoint,manifest,frozen,out,task,phase):
    data=[g for g in rows(manifest) if g["task"]==task]
    lock=json.loads(frozen.read_text())
    groups=cohort(data,phase)
    out.mkdir(parents=True)
    model,tok,env=load_runtime(checkpoint)
    frame=torch.from_numpy(projection()[:,:48].copy()).cuda()
    start=time.perf_counter()
    it=0
    feat=[]
    with torch.inference_mode(),(out/"arms.jsonl").open("w",encoding="utf-8") as f:
        for i,(g,p) in enumerate(groups):
            r=run_group(model,tok,g,p,lock,frame,checks=(i==0))
            for a,value in r:
                dump(f,{"i":it,"id":g["id"],"task":task,"protocol":p,"split":g["split"],
                    "options":g["options"],"gold":{k:v["gold"] for k,v in g["worlds"].items()},
                    "world_semantics":{k:{m:v[m] for m in ("S","M","Y")} for k,v in g["worlds"].items()},**a})
                feat.append(value)
                it+=1
            if (i+1)%8==0:
                f.flush()
                print(task,phase,i+1,round(time.perf_counter()-start,2),flush=True)
    np.save(out/"projected_boundary.npy",np.stack(feat))
    jsonfile(out/"complete.json",{"task":task,"phase":phase,"groups_protocol_pairs":len(groups),"arms":it,
               "manifest_sha256":digest(manifest),"freeze_sha256":digest(frozen),
               "code":digest(Path(__file__)),"runtime":env,"seconds":time.perf_counter()-start,
               "files":{p.name:digest(p) for p in out.iterdir()}})


if __name__=="__main__":
    c=argparse.ArgumentParser()
    for k in ("checkpoint","manifest","frozen","out"):
        c.add_argument(k,type=Path)
    c.add_argument("task",choices=TASKS)
    c.add_argument("phase",choices=("pilot","independent"))
    p=c.parse_args()
    run(p.checkpoint,p.manifest,p.frozen,p.out,p.task,p.phase)
