"""Use the same unscaled borrow difference on the matched no-borrow recipient."""
import argparse
from collections import Counter
from pathlib import Path

import torch

from borrow import ci
from interchange import complete, independent_like
from mechanism import Edit, LR, matched_plane
from native import Forward, load, output, save, setup, sha
from readout import core, mixed, plan

ARMS=("native","same_vector","geometry")


def effects(f,tok,group):
    items=[c for x in group for c in x["pair"]]
    ids,pos,r,h=core(f,items)
    act=torch.arange(3,len(ids),4,device=ids.device)
    r1,r2=[r.r[k][act] for k in (0,1)]
    d=mixed(r.r[1])-mixed(r.r[0])
    z=torch.repeat_interleave(d[::2],2,dim=0)
    noise=matched_plane(r1,r2-r1,z,independent_like(z,items,LR,497))
    outcomes={}
    for arm in ARMS:
        acc=[]
        for k,old in enumerate((r1,r2)):
            if arm=="native":
                state=h[k][act]
            else:
                use=z if arm=="same_vector" else noise
                patch=old+use*(1 if k==0 else -1)
                state=f.run(ids[act],1,start=h[0][act] if k else None,
                            edit=Edit(pos[act],readout=patch))[0]
            value=output(f.model,state,pos[act]).argmax(-1).tolist()
            acc.append(complete(f,tok,items,value,k))
        outcomes[arm]=list(zip(*acc,strict=True))
    return outcomes,z.norm(dim=-1).tolist()


def summary(rows):
    notes={}
    for label in ("repair","safe"):
        z=[r for r in rows if r["kind"]==label]
        s={}
        for arm in ARMS:
            a=Counter()
            for r in z:
                for k,p in enumerate(r["pred"][arm],1):
                    a[f"K{k}_correct"]+=p["number"]==r["target"]
                    a[f"K{k}_foil"]+=p["number"]==r["foil"]
                    a[f"K{k}_harm"]+=r["pred"]["native"][k-1]["number"]==r["target"] and p["number"]!=r["target"]
            s[arm]=dict(a)
        notes[label]=dict(n=len(z),outcomes=s)
    by={}
    for p,s in zip(rows[::2],rows[1::2],strict=True):
        k=p["target"]
        def changed(row):
            return int(row["pred"]["same_vector"][1]["number"]==row["foil"])-int(row["pred"]["geometry"][1]["number"]==row["foil"])
        assert p["id"]==s["id"] and p["dose"]==s["dose"]
        by[k]=by.get(k,0)+changed(p)-changed(s)
    notes["borrow_excess_K2_foil_vs_safe"]=ci(list(by.values()),(.0125,.9875))
    return notes


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    base=root/"NATIVE.json"
    _,z,_=plan(tok,load(base)["rows"])
    kept=[x for x in z if x["aligned"]]
    raw=root/"SPECIFICITY_RAW.json"
    out=load(raw)["rows"] if raw.exists() else []
    assert [(r["id"],r["kind"]) for r in out]==[(q["id"],q["kind"]) for a in kept for q in a["pair"]][:len(out)]
    for i in range(len(out)//2,len(kept),6):
        pack=kept[i:i+6]
        pred,dose=effects(f,tok,pack)
        for j,item in enumerate(c for z in pack for c in z["pair"]):
            arms={m:p[j] for m,p in pred.items()}
            assert list(arms["native"])==item["native"]
            out.append(dict(id=item["id"],kind=item["kind"],target=item["target"],foil=item["foil"],pred=arms,dose=dose[j]))
        save(raw,dict(rows=out,native_sha256=sha(base),readout_sha256=sha(root/"READOUT_RAW.json")))
        print("same dose exposure",len(out),2*len(kept),flush=True)
    r=summary(out)
    save(root/"SPECIFICITY_SUMMARY.json",r)
    print(r,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

