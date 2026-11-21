"""Prospective ablation of native predecessor writes and mediator restoration."""
import argparse
from pathlib import Path

import numpy as np
import torch

from causal import centered, copy_heads, direction_like, projection, source_tokens, unit
from native import Forward, Recorder, SEED, load, output, save, setup, sha, tensors, timer


class Writer:
    def __init__(self,f,branch,later,kind="remove",mediator=None,heads=(),ref=None,loop=0):
        self.f,self.branch,self.later,self.kind=f,branch,later,kind
        self.mediator,self.heads,self.ref,self.loop=mediator,list(heads),ref,loop

    def write(self,key,name,x):
        if (key[0],key[1],name)!=(self.loop,*self.branch):
            return x
        if self.kind=="normal":
            return x
        if self.kind=="random":
            return x-unit(direction_like(x,SEED),x)
        if self.kind=="roll":
            return x-unit(x.roll(1,dims=1),x)
        return torch.zeros_like(x)

    def resid(self,k,l,x):
        return x

    def qk(self,key,q,k):
        if self.ref is None or key!=(self.loop+1,self.later):
            return q,k
        if self.mediator=="query":
            return copy_heads(q,self.ref["q"],self.heads),k
        if self.mediator=="key":
            return q,copy_heads(k,self.ref["k"],self.heads)
        return q,k

    def attn(self,key,a,v):
        if self.ref is None or key!=(self.loop+1,self.later):
            return a,v
        if self.mediator=="route":
            return copy_heads(a,self.ref["a"],self.heads),v
        if self.mediator=="value":
            return a,copy_heads(v,self.ref["v"],self.heads)
        return a,v

    def mlp(self,key,z,r):
        if self.ref is None or key!=(self.loop+1,self.later):
            return z
        if self.mediator not in ("gate","up"):
            return z
        a=r.data[key]
        act=self.f.layers[self.later].mlp.act_fn
        if self.mediator=="gate":
            r.put(key,"g_pre",self.ref["g_pre"])
            return act(self.ref["g_pre"])*a["u"]
        r.put(key,"u",self.ref["u"])
        return act(a["g_pre"])*self.ref["u"]


def receptor(f,r,mode,layer,heads):
    return r["a"][:,heads] if mode=="attention" else f.layers[layer].mlp.act_fn(r["g_pre"])


def context(f,ids,c,k):
    layers={m:c[f"candidate_{m}_layer"]-1 for m in ("attention","mlp")}
    r=Recorder(keys={(x,l) for x in (k,k+1) for l in layers.values()}|{(k,l) for l in range(24)})
    clean=f.run(ids,k+2,record=r)
    return r,clean,layers


def score_path(f,r,patched,k,l,mode,heads):
    prior=receptor(f,r.data[k,l],mode,l,heads)
    late=receptor(f,r.data[k+1,l],mode,l,heads)
    partial=receptor(f,patched.data[k+1,l],mode,l,heads)
    return 1.-projection(partial,prior,late)


def screen(f,tok,rows,config,allowed):
    rows_out=[]
    heads=config["heads_zero_based"]
    for row in rows:
        start=timer()
        ids,pos=tensors(tok,[row["prompt"]])
        r,states,layers=context(f,ids,config,0)
        obs={}
        for layer in range(24):
            for name in ("a_write","m_write"):
                t=Recorder(keys={(1,l) for l in layers.values()})
                z=f.run(ids,2,record=t,edit=Writer(f,(layer,name),layers["attention"]))
                p={m:dict(factor_loss=score_path(f,r,t,0,i,m,heads)) for m,i in layers.items() if m in allowed}
                h0,h1=states[0],z[0]
                dist=float((h1-h0).norm()/h0.norm().clamp_min(1e-9))
                obs[f"{layer+1}/{name}"]=dict(metrics=p,relative_boundary_change=dist)
        rows_out.append(dict(id=row["id"],values=obs,gpu_seconds=timer()-start))
        print("writers screen",len(rows_out),len(rows),flush=True)
    return rows_out


def lock(out,rows):
    keys=list(rows[0]["values"])
    results={}
    for mode in rows[0]["values"][keys[0]]["metrics"]:
        scores={}
        for key in keys:
            v=[q["values"][key] for q in rows]
            scores[key]=float(np.median([x["metrics"][mode]["factor_loss"] for x in v]))
        ranked=sorted(keys,key=lambda k:scores[k],reverse=True)
        selected=ranked[0]
        layer,sub=selected.split("/")
        control=f"{(int(layer)-1+12)%24+1}/{sub}"
        results[mode]=dict(selected=selected,control=control,median_factor_loss=scores)
    save(out/"ORIGIN_LOCK.json",dict(criterion="largest median loss of native early-to-late factor direction after one real write removal; labels withheld",
                                     screen_sha256=sha(out/"ORIGIN_SCREEN.json"),mode=results))


def replay(f,tok,row,c,lock,loop=0):
    ids,pos=tensors(tok,[row["prompt"]])
    r,clean,layers=context(f,ids,c,loop)
    t=timer()
    native=output(f.model,clean[-1],pos)
    early=output(f.model,clean[-2],pos)
    toks=source_tokens(tok,row)
    heads=c["heads_zero_based"]
    result={}
    for mode,l in layers.items():
        if mode not in lock["mode"]:
            continue
        selected=lock["mode"][mode]["selected"]
        controls=[("drop",selected,"remove",None),("matched_random",selected,"random",None),
                  ("shifted_tokens",selected,"roll",None),("other_write",lock["mode"][mode]["control"],"remove",None)]
        if mode=="attention":
            controls.extend((x,selected,"remove",x) for x in ("route","value","query","key"))
        else:
            controls.extend((x,selected,"remove",x) for x in ("gate","up"))
        cond={}
        a=r.data[loop+1,l]
        controls.append(("mediator_only",selected,"normal","route" if mode=="attention" else "gate"))
        deleted=None
        for key,where,kind,mediator in controls:
            pl,name=where.split("/")
            branch=(int(pl)-1,name)
            trace=Recorder(keys={(loop+1,ll) for ll in layers.values()})
            corrected=f.run(ids,loop+2,record=trace,edit=Writer(f,branch,l,kind,mediator,heads,deleted if key=="mediator_only" else a,loop))
            lag=trace.data[loop+1,l]
            if key=="drop":
                deleted=lag
            logits=output(f.model,corrected[-1],pos)
            h0=clean[-2]
            cond[key]=dict(write=where,mediator=mediator,readout_loss=1.-centered(logits,early,native),
                factor_loss=score_path(f,r,trace,loop,l,mode,heads),
                hidden_distance=float((corrected[-2]-h0).norm()/h0.norm().clamp_min(1e-9)),
                rank_tokens=logits[0,toks].tolist(),top_token=int(logits.argmax()),
                state_norm_max_ratio=float((corrected[-1].norm(dim=-1)/clean[-1].norm(dim=-1).clamp_min(1e-9)).max()),
                perturb_components={x:float((lag[x]-a[x]).norm()/a[x].norm().clamp_min(1e-9))
                 for x in ("q","k","v","a","g_pre","u")})
        result[mode]=cond
    return dict(id=row["id"],family=row["family"],loop_transition=f"K{loop+1}toK{loop+2}",
                native_top=[int(early.argmax()),int(native.argmax())],candidate_first_ids=toks,
                native_first=[early[0,toks].tolist(),native[0,toks].tolist()],mechanisms=result,gpu_seconds=timer()-t)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("phase")
    p.add_argument("--limit",type=int)
    p.add_argument("--later",action="store_true")
    a=p.parse_args()
    data=load(a.manifest)
    c=load(a.out/"CANDIDATES.json")
    if a.phase=="freeze":
        lock(a.out,load(a.out/"ORIGIN_SCREEN.json")["rows"])
        return
    model,tok=setup(a.model)
    f=Forward(model)
    if a.phase=="screen":
        allow=[k for k,v in load(a.out/"GATE.json")["candidates"].items() if v["search_writes"]]
        assert allow
        out=screen(f,tok,data["dev"][:16],c,allow)
        save(a.out/"ORIGIN_SCREEN.json",dict(rows=out,source_candidate_sha256=sha(a.out/"CANDIDATES.json")))
        return
    lockf=a.out/"ORIGIN_LOCK.json"
    config=load(lockf)
    receipt=a.out/"ORIGIN_PREDICTIONS.json"
    if not receipt.exists():
        save(receipt,dict(lock_sha256=sha(lockf),suite={z:[q["id"] for q in data[z]]
              for z in ("confirm_arc","confirm_external","controlled")},expected_sign="positive drop; positive rescue relative to drop"))
    pred=load(receipt)
    assert pred["lock_sha256"]==sha(lockf) and pred["suite"][a.phase]==[q["id"] for q in data[a.phase]]
    rows=data[a.phase][:a.limit]
    dest=a.out/f"ORIGIN_{a.phase}{'_later' if a.later else ''}.json"
    output_rows=load(dest)["rows"] if dest.exists() else []
    for i,row in enumerate(rows[len(output_rows):],start=len(output_rows)):
        output_rows.append(replay(f,tok,row,c,config,loop=1 if a.later else 0))
        if (i+1)%4==0 or i+1==len(rows):
            save(dest,dict(rows=output_rows,expected=len(rows),prediction_sha256=sha(receipt),
                 peak_cuda_bytes=torch.cuda.max_memory_allocated()))
            print("writers",a.phase,i+1,len(rows),flush=True)


if __name__=="__main__":
    main()

