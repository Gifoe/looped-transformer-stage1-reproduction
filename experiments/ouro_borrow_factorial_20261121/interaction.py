"""A controlled mixed finite difference at the L17 readout."""
import argparse
from collections import Counter
from pathlib import Path
import random

import torch

from borrow import ci
from graft import capture, condition
from interchange import fit_margin, independent_like, terminated
from mechanism import Edit, LR, matched_plane
from native import Forward, load, output, pad, save, setup, sha
from rule import MAX_NEW, text_results

MODES=("joint","remove","sham","restore")


class Readout:
    def __init__(self,p):
        self.p=p
        self.ind=torch.arange(len(p),device=p.device)
        self.value=None

    def put(self,key,name,x):
        if key[1]==LR and name=="input":
            self.value=x[self.ind,self.p].detach()


def selection(root,tok,groups):
    source=load(root/"GRAFT_RAW.json")["rows"]
    all_by={x["id"]:x for x in groups}
    bank={}
    for r in sorted(source,key=lambda x:x["id"]):
        b=r["forward"]
        if r["split"]!="confirmation" or not b["aligned"]:
            continue
        expected=b["case"]["expected"]
        predicted=[b["pred"][a]["number"] for a in ("native","a","b","ab")]
        if predicted==expected:
            bank.setdefault(r["group"],r)
    eligible=list(bank.values())
    sample=random.Random(261122023).sample(eligible,min(48,len(eligible)))
    cases=[]
    for r in sample:
        called=condition(tok,groups,all_by[r["id"]],False)
        b=r["forward"]["pred"]
        one,two=[b[a]["tokens"] for a in ("native","ab")]
        i=next(j for j,(v,w) in enumerate(zip(one,two)) if v!=w)
        assert one[:i]==two[:i]
        called.update(kind="factorial_interaction",prefix=one[:i],pivot=[two[i],one[i]],joint_answer=b["ab"],
                      base_answer=b["native"],group=r["group"])
        cases.append(called)
    return cases,len(eligible)


def state(f,ids,h,last,cases,source,use_a,use_b):
    new=h.clone()
    for j,(c,(a,b,_)) in enumerate(zip(cases,source,strict=True)):
        if use_a:
            new[j,c["a"]]=a
        if use_b:
            new[j,c["b"]]=b
    i=torch.arange(len(last),device=last.device)
    assert torch.equal(new[i,last],h[i,last])
    return new


def patched_suffix(f,tok,cases,sources,tokens):
    tokens=[list(i) for i in tokens]
    active=[j for j,i in enumerate(tokens) if len(i)<MAX_NEW and not terminated(tok,i)]
    while active:
        l=[cases[i]["ids"][0]+tokens[i] for i in active]
        ids,last=pad(l)
        old=f.run(ids,1)[0]
        mod=state(f,ids,old,last,[cases[j] for j in active],[sources[j] for j in active],True,True)
        pred=output(f.model,f.run(ids,1,start=mod)[0],last).argmax(-1).tolist()
        rest=[]
        for j,k in zip(active,pred,strict=True):
            tokens[j].append(k)
            if len(tokens[j])<MAX_NEW and not terminated(tok,tokens[j]):
                rest.append(j)
        active=rest
    return text_results(tok,tokens)


def complete(f,tok,cases,src,pivot):
    res=[c["joint_answer"] for c in cases]
    used=[j for j,(c,v) in enumerate(zip(cases,pivot,strict=True)) if v!=c["pivot"][0]]
    if used:
        z=patched_suffix(f,tok,[cases[j] for j in used],[src[j] for j in used],
                         [cases[j]["prefix"]+[pivot[j]] for j in used])
        for k,v in zip(used,z,strict=True):
            res[k]=v
    return res


def activate(f,tok,cases):
    src=capture(f,cases)
    ids,at=pad([c["ids"][0]+c["prefix"] for c in cases])
    one=f.run(ids,1)[0]
    rows={}
    base={}
    for a,b in ((False,False),(True,False),(False,True),(True,True)):
        y=state(f,ids,one,at,cases,src,a,b)
        extract=Readout(at)
        h=f.run(ids,1,start=y,record=extract)[0]
        rows[a,b]=extract.value
        base[a,b]=(y,h)
    last,observe=base[True,True]
    r0,r1=rows[False,False],rows[True,True]
    cross=r1-rows[True,False]-rows[False,True]+r0
    random=matched_plane(r0,r1-r0,cross,independent_like(cross,cases,LR,381))
    labels=torch.tensor([c["pivot"] for c in cases],device=at.device)
    pk=output(f.model,observe,at).argmax(-1).tolist()
    assert pk==[c["pivot"][0] for c in cases]
    past=output(f.model,base[False,False][1],at).argmax(-1).tolist()
    assert past==[c["pivot"][1] for c in cases]
    out={}
    score={}
    for name,d in (("joint",None),("remove",r1-cross),("sham",r1-random),("restore",r1)):
        h=observe if d is None else f.run(ids,1,start=last,edit=Edit(at,readout=d))[0]
        if name=="restore":
            t=torch.arange(len(at),device=at.device)
            assert torch.equal(h[t,at],observe[t,at])
        tok_ids=output(f.model,h,at).argmax(-1).tolist()
        out[name]=complete(f,tok,cases,src,tok_ids)
        score[name]=fit_margin(f,h,at,labels).tolist()
    fig=dict(cross=cross.norm(dim=-1).tolist(),full=(r1-r0).norm(dim=-1).tolist())
    return out,score,fig


def report(data,eligible):
    n=len(data)
    m={}
    for arm in MODES:
        x=Counter()
        for r in data:
            z=r["pred"][arm]["number"]
            x["own_joint"]+=z==r["y"]
            x["base_foil"]+=z==r["foil"]
            x["unparsed"]+=z is None
            x["other"]+=z is not None and z not in (r["y"],r["foil"])
        m[arm]=dict(x)
    pairs=ci([int(r["pred"]["remove"]["number"]==r["foil"])-int(r["pred"]["sham"]["number"]==r["foil"]) for r in data])
    return dict(n=n,eligible=eligible,arms=m,exact_foil_excess=pairs,
                pass_gate=bool(m["remove"].get("base_foil",0)>=8 and pairs["interval"] and pairs["interval"][0]>0),
                mean_margins={a:sum(r["margins"][a] for r in data)/n for a in MODES})


@torch.inference_mode()
def run(model,root):
    assert load(root/"GRAFT_SUMMARY.json")["confirmation"]["forward"]["gate"]
    mod,tok=setup(model)
    f=Forward(mod)
    groups=load(root/"NATIVE.json")["rows"]
    chosen,eligible=selection(root,tok,groups)
    raw=root/"INTERACTION_RAW.json"
    seen=load(raw)["rows"] if raw.exists() else []
    assert [r["id"] for r in seen]==[r["id"] for r in chosen[:len(seen)]]
    for i in range(len(seen),len(chosen),8):
        pack=chosen[i:i+8]
        d,e,stats=activate(f,tok,pack)
        for j,r in enumerate(pack):
            outcomes={m:v[j] for m,v in d.items()}
            assert outcomes["joint"]==outcomes["restore"]==r["joint_answer"]
            seen.append(dict(id=r["id"],y=r["group"],foil=r["group"]+100,
                             pred=outcomes,margins={m:v[j] for m,v in e.items()},
                             norm={m:v[j] for m,v in stats.items()}))
        save(raw,dict(rows=seen,graft_sha256=sha(root/"GRAFT_RAW.json")))
        print("interaction",len(seen),len(chosen),flush=True)
    r=report(seen,eligible)
    save(root/"INTERACTION_SUMMARY.json",r)
    print(r,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

