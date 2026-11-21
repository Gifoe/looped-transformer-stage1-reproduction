"""Decompose the selected Q effect by actual attention value contributions."""
import argparse
from collections import Counter
from pathlib import Path

import torch

from borrow import ci
from interchange import Patch, batch, complete, fit_margin, independent_like, load_cases, text_case
from native import Forward, load, output, save, setup, sha
from rule import PREFIX, prompt

FIELDS=("demonstrations","first_operand","second_operand","rule_and_answer")
MODES=("replay", "only", "only_sham", "without", "between", "within")


class Weights:
    def __init__(self,at,layer):
        self.at=at
        self.layer=layer
        self.row=torch.arange(len(at),device=at.device)
        self.vector=None

    def put(self,key,label,value):
        if key[1]==self.layer and label=="a":
            self.vector=value[self.row,:,self.at,:].detach()


class Route(Patch):
    def __init__(self,at,layer,value):
        super().__init__(at)
        self.layer=layer
        self.value=value

    def attn(self,key,a,v):
        if key[1]!=self.layer:
            return a,v
        x=a.clone()
        x[self.rows,:,self.pos,:]=self.value
        return x,v


def set_fields(tok,row,case,t):
    a,b=row["pairs"][t]
    text=prompt(a,b,1)
    offsets=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    assert offsets["input_ids"]==case["ids"]
    cuts=[len(PREFIX),len(PREFIX+f"A = {a}, "),len(PREFIX+f"A = {a}, B = {b}, ")]
    case["fields"]=[sum(s>=c for c in cuts) for s,e in offsets["offset_mapping"]]+[3]*len(case["prefix"])
    return case


def collect(root,tok,split,with_safe):
    records,repair=load_cases(root,tok,split)
    result=[]
    for r,c in zip(records,repair,strict=True):
        c=set_fields(tok,r,c,1)
        if not with_safe and not all(c["reach"]):
            continue
        result.append(c)
        if with_safe:
            result.append(set_fields(tok,r,text_case(tok,r,0),0))
    return result


def source(f,cases,l):
    ids,at,tr,h1,h2=batch(f,cases)
    base=[]
    changed=[]
    predicted=[]
    for k in (0,1):
        obs=Weights(at,l)
        f.run(ids,1,start=h1 if k else None,record=obs)
        base.append(obs.vector)
        obs=Weights(at,l)
        done=f.run(ids,1,start=h1 if k else None,record=obs,
                   edit=Patch(at,query={l:(None,tr.at(1-k,l,"q"))}))[0]
        changed.append(obs.vector)
        predicted.append(output(f.model,done,at).argmax(-1).tolist())
    span=max(len(x["fields"]) for x in cases)
    tags=torch.tensor([c["fields"]+[-1]*(span-len(c["fields"])) for c in cases],device=ids.device)
    valid=tags[:,None,:]>=0
    groups=[(tags[:,None,:]==i) for i in range(len(FIELDS))]
    assert torch.equal(sum(mask.int() for mask in groups),valid.int())
    return ids,at,h1,h2,base,changed,predicted,groups


def matched(delta,mask,draw):
    n=mask.sum(dim=-1,keepdim=True).clamp_min(1)
    mean=(delta*mask).sum(dim=-1,keepdim=True)/n
    shape=(delta-mean)*mask
    r=(draw-(draw*mask).sum(dim=-1,keepdim=True)/n)*mask
    z=mean*mask+r*(shape.norm(dim=-1,keepdim=True)/r.norm(dim=-1,keepdim=True).clamp_min(1e-12))
    torch.testing.assert_close(z.sum(dim=-1),(delta*mask).sum(dim=-1),rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(z.norm(dim=-1),(delta*mask).norm(dim=-1),rtol=2e-5,atol=2e-6)
    return z


def stationary(p,q,groups,mode):
    dest=torch.zeros_like(p)
    for a in groups:
        u,v=(p*a).sum(dim=-1,keepdim=True),(q*a).sum(dim=-1,keepdim=True)
        if mode=="between":
            dest+=torch.where(u>1e-28,p*a*(v/u.clamp_min(1e-28)),q*a)
        else:
            dest+=torch.where(v>1e-28,q*a*(u/v.clamp_min(1e-28)),p*a)
    torch.testing.assert_close(dest.sum(dim=-1),p.sum(dim=-1),rtol=2e-5,atol=2e-6)
    return dest


def discover(f,tok,root,l):
    items=collect(root,tok,"discovery",False)
    raw=[]
    for i in range(0,len(items),12):
        cases=items[i:i+12]
        ids,at,h1,h2,p,q,pivot,groups=source(f,cases,l)
        labels=torch.tensor([x["labels"] for x in cases],device=ids.device)
        native=[fit_margin(f,h,at,labels) for h in (h1,h2)]
        observations=[]
        for mask in groups:
            gain=[]
            for k in (0,1):
                values=p[k]+(q[k]-p[k])*mask
                h=f.run(ids,1,start=h1 if k else None,edit=Route(at,l,values))[0]
                gain.append((fit_margin(f,h,at,labels)-native[k])*(1 if k==0 else -1))
            observations.append(torch.stack(gain,dim=-1))
        output=torch.stack(observations,dim=1).tolist()
        for j,c in enumerate(cases):
            raw.append(dict(id=c["id"],gain=output[j]))
    scores=[]
    for i,label in enumerate(FIELDS):
        q=[sum(r["gain"][i][k] for r in raw)/len(raw) for k in (0,1)]
        scores.append(dict(field=label,means=q,rank=min(q)))
    k=max(range(len(scores)),key=lambda n:scores[n]["rank"])
    result=dict(field=k,scores=scores,items=raw,layer=l,query_selection_sha256=sha(root/"SELECTION.json"))
    save(root/"ROUTING_SELECTION.json",result)
    print("routing field",FIELDS[k],scores,flush=True)
    return result


def summarize(rows):
    result={}
    for kind in ("repair","safe"):
        block=[r for r in rows if r["kind"]==kind]
        arms={}
        for arm in MODES:
            x=Counter()
            for r in block:
                p,q=[v["number"] for v in r["pred"][arm]]
                x["K1_correct"]+=p==r["target"]
                x["K2_foil"]+=q==r["foil"]
                x["joint"]+=p==r["target"] and q==r["foil"]
                x["K1_unparsed"]+=p is None
                x["K2_unparsed"]+=q is None
                u,v=[j["number"] for j in r["native"]]
                x["K1_harm"]+=u==r["target"] and p!=r["target"]
                x["K2_harm"]+=v==r["target"] and q!=r["target"]
            arms[arm]=dict(x)
        result[kind]=dict(n=len(block),arms=arms)
    for a,b in (("only","only_sham"),("replay","without")):
        clusters={}
        for r in rows:
            if r["kind"]!="repair":
                continue
            v=[int([q["number"] for q in r["pred"][m]]==[r["target"],r["foil"]]) for m in (a,b)]
            y=r["target"]
            clusters[y]=clusters.get(y,0)+v[0]-v[1]
        result[a+"_vs_"+b]=ci(list(clusters.values()),(.0125,.9875))
    return result


@torch.inference_mode()
def run(model,root):
    query=load(root/"QUERY_SUMMARY.json")
    assert query["repair"]["modes"]["layer"]["joint"]>=8
    assert query["comparisons"]["layer_vs_layer_sham"]["joint"]["interval"][0]>0
    l=load(root/"SELECTION.json")["layer"]
    m,tok=setup(model)
    f=Forward(m)
    sel=root/"ROUTING_SELECTION.json"
    s=load(sel) if sel.exists() else discover(f,tok,root,l)
    cases=collect(root,tok,"confirmation",True)
    exact=load(root/"QUERY_RAW.json")["rows"]
    assert [(c["id"],c["kind"]) for c in cases]==[(r["id"],r["kind"]) for r in exact]
    path=root/"ROUTING_RAW.json"
    old=load(path)["rows"] if path.exists() else []
    assert [(c["id"],c["kind"]) for c in old]==[(r["id"],r["kind"]) for r in cases[:len(old)]]
    for i in range(len(old),len(cases),12):
        pack=cases[i:i+12]
        ids,at,h1,h2,p,q,pivot,groups=source(f,pack,l)
        mask=groups[s["field"]]
        per=[dict(id=c["id"],kind=c["kind"],target=c["target"],foil=c["foil"],native=c["native"],pred={}) for c in pack]
        for mode in MODES:
            values=[]
            for k in (0,1):
                delta=q[k]-p[k]
                if mode=="only":
                    z=p[k]+delta*mask
                elif mode=="only_sham":
                    z=p[k]+matched(delta,mask,independent_like(delta,pack,l,70))
                elif mode=="without":
                    z=q[k]-delta*mask
                elif mode=="replay":
                    z=q[k]
                else:
                    z=stationary(p[k],q[k],groups,mode)
                full=f.run(ids,1,start=h1 if k else None,edit=Route(at,l,z))[0]
                native=output(f.model,full,at).argmax(-1).tolist()
                if mode=="replay":
                    assert native==pivot[k]
                values.append(complete(f,tok,pack,native,k))
            for j,x in enumerate(zip(*values,strict=True)):
                per[j]["pred"][mode]=x
                if mode=="replay":
                    assert [y["tokens"] for y in x]==[y["tokens"] for y in exact[i+j]["pred"]["layer"]]
        old.extend(per)
        save(path,dict(rows=old,selection_sha256=sha(sel),query_sha256=sha(root/"QUERY_RAW.json")))
        print("source routing",len(old),len(cases),flush=True)
    report=summarize(old)
    save(root/"ROUTING_SUMMARY.json",report)
    print(report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)

