"""Test which natural cross-loop layer-input subset carries a borrow correction."""
import argparse
from collections import Counter
from pathlib import Path

import torch

from borrow import ci
from interchange import complete, fit_margin, independent_like, load_cases, text_case
from native import Forward, load, output, pad, save, setup, sha
from state import rotate_about

SITES=(0,4,8,12,13,16)
PARTS=("readout","context")


class ReadoutTrace:
    def __init__(self):
        self.data={}

    def put(self,key,name,x):
        if name=="input" and key[1] in SITES:
            self.data[key]=x.detach()


class Edit:
    def __init__(self,at,layer,contents,mode):
        self.layer=layer
        self.contents=contents
        path=torch.arange(contents.shape[1],device=at.device)[None,:]
        if mode.startswith("readout"):
            self.mask=(path==at[:,None]).unsqueeze(-1)
        elif mode.startswith("context"):
            self.mask=(path<at[:,None]).unsqueeze(-1)
        else:
            self.mask=torch.ones_like(contents[:,:,:1],dtype=torch.bool)

    def resid(self,k,l,x):
        if l!=self.layer:
            return x
        updated=torch.where(self.mask,self.contents,x)
        assert torch.equal(torch.where(self.mask,x,updated),x)
        return updated

    def attn(self,key,a,v):
        return a,v

    def mlp(self,key,z,record):
        return z


def pre(f,cases):
    ids,p=pad([c["ids"]+c["prefix"] for c in cases])
    r=ReadoutTrace()
    h1,h2=f.run(ids,2,record=r)
    return ids,p,r,h1,h2


def patch(f,ids,p,tr,h1,k,mode,layer,cases):
    val=tr.data[1-k,layer]
    own=tr.data[k,layer]
    if mode.endswith("sham"):
        val=rotate_about(own,val,independent_like(val,cases,layer,44))
    return f.run(ids,1,start=h1 if k else None,edit=Edit(p,layer,val,mode))[0]


def diagnostic(f,tok,root):
    raw,lst=load_cases(root,tok,"discovery")
    cases=[c for c in lst if all(c["reach"])]
    results=[]
    for i in range(0,len(cases),8):
        a=cases[i:i+8]
        ids,p,tr,h1,h2=pre(f,a)
        labels=torch.tensor([x["labels"] for x in a],device=ids.device)
        native=[fit_margin(f,z,p,labels) for z in (h1,h2)]
        tests=[]
        for layer in SITES:
            by=[]
            for k in (0,1):
                estimates=[]
                for mode in ("readout","context","full"):
                    h=patch(f,ids,p,tr,h1,k,mode,layer,a)
                    if mode=="full":
                        assert torch.equal(h,(h1,h2)[1-k])
                    diff=(fit_margin(f,h,p,labels)-native[k])*(1 if k==0 else -1)
                    estimates.append(diff)
                joined=estimates[2]-estimates[1]-estimates[0]
                by.append(torch.stack(estimates+[joined],dim=-1))
            tests.append(torch.stack(by,dim=1))
        m=torch.stack(tests,dim=1).tolist()
        for j,c in enumerate(a):
            results.append(dict(id=c["id"],layers=m[j]))
    scores={}
    pick={}
    for axis,mode in enumerate(PARTS):
        table=[]
        for i,layer in enumerate(SITES):
            means=[sum(row["layers"][i][k][axis] for row in results)/len(results) if results else 0 for k in (0,1)]
            table.append(dict(layer=layer,means=means,score=min(means)))
        scores[mode]=table
        pick[mode]=max(table,key=lambda x:x["score"])["layer"]
    s=dict(sites=SITES,pick=pick,scores=scores,n=len(results),raw=results,source_sha256=sha(root/"NATIVE.json"))
    save(root/"CONTEXT_SELECTION.json",s)
    print("context selection",pick,"n",len(results),flush=True)
    return s


def summarize(rows):
    out={}
    names=("clean","full","readout","readout_sham","context","context_sham")
    for kind in ("repair","safe"):
        current=[x for x in rows if x["kind"]==kind]
        detail={}
        for a in names:
            p=Counter()
            for row in current:
                x,y=[v["number"] for v in row["pred"][a]]
                base1,base2=[v["number"] for v in row["pred"]["clean"]]
                p["K1_correct"]+=x==row["target"]
                p["K2_foil"]+=y==row["foil"]
                p["K2_correct"]+=y==row["target"]
                p["joint"]+=x==row["target"] and y==row["foil"]
                p["K1_harm"]+=base1==row["target"] and x!=row["target"]
                p["K2_harm"]+=base2==row["target"] and y!=row["target"]
            detail[a]=dict(p)
        out[kind]=dict(n=len(current),arms=detail)
    repaired=[r for r in rows if r["kind"]=="repair"]
    for mode in PARTS:
        group={}
        for row in repaired:
            target=row["target"]
            foil=row["foil"]
            left,right=[int([v["number"] for v in row["pred"][arm]]==[target,foil])
                        for arm in (mode,mode+"_sham")]
            group[target]=group.get(target,0)+left-right
        v=ci(list(group.values()),(.0125,.9875))
        out[mode+"_excess"]=v
        out[mode+"_gate"]=bool(out["repair"]["arms"][mode]["joint"]>=8 and v["interval"] and v["interval"][0]>0)
    return out


@torch.inference_mode()
def run(args):
    root=args.out
    assert load(root/"NATIVE_SUMMARY.json")["gate"]
    m,tok=setup(args.model)
    f=Forward(m)
    s=root/"CONTEXT_SELECTION.json"
    design=load(s) if s.exists() else diagnostic(f,tok,root)
    q=load(root/"QUERY_RAW.json")
    raw,repairs=load_cases(root,tok,"confirmation")
    cases=[r for one in zip(repairs,[text_case(tok,x,0) for x in raw],strict=True) for r in one]
    assert [(r["id"],r["kind"]) for r in cases]==[(r["id"],r["kind"]) for r in q["rows"]]
    path=root/"CONTEXT_RAW.json"
    rows=load(path)["rows"] if path.exists() else []
    assert [(x["id"],x["kind"]) for x in rows]==[(x["id"],x["kind"]) for x in cases[:len(rows)]]
    modes={"clean":None,"full":design["pick"]["context"]}
    for part in PARTS:
        for mode in (part,part+"_sham"):
            modes[mode]=design["pick"][part]
    for i in range(len(rows),len(cases),8):
        pack=cases[i:i+8]
        ids,p,tr,h1,h2=pre(f,pack)
        expected=(h1,h2)
        output_by={}
        for mode,l in modes.items():
            nums=[]
            for k in (0,1):
                h=expected[k] if l is None else patch(f,ids,p,tr,h1,k,mode,l,pack)
                if mode=="full":
                    assert torch.equal(h,expected[1-k])
                v=output(f.model,h,p).argmax(-1).tolist()
                nums.append(complete(f,tok,pack,v,k))
            output_by[mode]=list(zip(*nums,strict=True))
        for j,c in enumerate(pack):
            clean=q["rows"][i+j]["pred"]["clean"]
            detail={k:x[j] for k,x in output_by.items()}
            assert [z["tokens"] for z in detail["clean"]]==[z["tokens"] for z in clean]
            rows.append(dict(id=c["id"],kind=c["kind"],target=c["target"],foil=c["foil"],pred=detail))
        save(path,dict(rows=rows,query_sha256=sha(root/"QUERY_RAW.json"),selection_sha256=sha(s)))
        print("confirm context",len(rows),len(cases),flush=True)
    result=summarize(rows)
    save(root/"CONTEXT_SUMMARY.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    run(p.parse_args())

