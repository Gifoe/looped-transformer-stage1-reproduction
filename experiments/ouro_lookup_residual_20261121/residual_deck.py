"""Collect untouched lookup generations and outcome-defined table swaps."""
import argparse
from collections import Counter
from pathlib import Path
from random import Random

import torch

from native import Forward, load, save, setup, sha
from query_deck import KEYS, PREFIX, history as earlier, text
from rule import greedy

PARENT=Path("/root/ouro-lookup-query-20261121/results/DECK.json")
SEED=261124381
VARIANTS=("original","swapped","unrelated_original","unrelated_swapped")


def history():
    return earlier() | {text(r["key"],**e) for r in load(PARENT)["rows"] for e in r["example"].values()}


def build():
    seen=history()
    rng=Random(SEED)
    rows=[]
    for j,y in enumerate(rng.sample(range(100,1000),384)):
        k=KEYS[j%4]
        values=rng.sample([x for x in range(100,1000) if x!=y],3)
        values.insert(k-1,y)
        order=rng.sample(KEYS,4)
        e=dict(order=order,values=values)
        q=text(k,**e)
        assert q not in seen
        seen.add(q)
        rows.append(dict(id=f"lookup-residual-{j:03}",key=k,target=y,example={"original":e}))
    return dict(rows=rows,prior_sha256=sha(PARENT),seed=SEED,prefix=PREFIX)


def natural_repair(r):
    a,b=[p["number"] for p in r["native"]["original"]]
    return a in r["example"]["original"]["values"] and a!=r["target"] and b==r["target"]


def augment(r):
    assert natural_repair(r)
    p=r["example"]["original"]
    y=r["target"]
    w=r["native"]["original"][0]["number"]
    before=p["values"]
    a,b=before.index(y),before.index(w)
    x,z=[j for j in range(4) if j not in (a,b)]
    def change(values,pair):
        out=list(values)
        i,j=pair
        out[i],out[j]=out[j],out[i]
        return out
    after=change(before,(a,b))
    values=dict(swapped=after,unrelated_original=change(before,(x,z)),
                unrelated_swapped=change(after,(x,z)))
    return dict(r,old=w,example=dict(r["example"],**{k:dict(order=p["order"],values=t) for k,t in values.items()}))


def summary(base,donors):
    z=Counter(n=len(base),k1=0,k2=0,corrections=0,wrong_table=0,regressions=0)
    for r in base:
        y=r["target"]
        a,b=[q["number"]==y for q in r["native"]["original"]]
        z["k1"]+=a
        z["k2"]+=b
        z["corrections"]+=not a and b
        z["wrong_table"]+=natural_repair(r)
        z["regressions"]+=a and not b
    q=Counter(n=len(donors),swapped_k1=0,swapped_k2=0,unrelated_original_k2=0,unrelated_swapped_k2=0,eligible=0)
    for r in donors:
        q["swapped_k1"]+=r["native"]["swapped"][0]["number"]==r["old"]
        q["swapped_k2"]+=r["native"]["swapped"][1]["number"]==r["old"]
        q["unrelated_original_k2"]+=r["native"]["unrelated_original"][1]["number"]==r["target"]
        q["unrelated_swapped_k2"]+=r["native"]["unrelated_swapped"][1]["number"]==r["old"]
        q["eligible"]+=is_pair(r)
    return dict(original=dict(z),paired=dict(q),gate=q["eligible"]>=32 and z["wrong_table"]>=40)


def is_pair(r):
    return all(r["native"][k][1]["number"]==r["old" if k.endswith("swapped") else "target"] for k in VARIANTS)


@torch.inference_mode()
def run(model,root):
    deck=root/"DECK.json"
    if not deck.exists():
        save(deck,build())
    d=load(deck)
    model,tok=setup(model)
    f=Forward(model)
    base=root/"NATIVE.json"
    recent=load(base)["rows"] if base.exists() else []
    assert [r["id"] for r in recent]==[s["id"] for s in d["rows"][:len(recent)]]
    for j in range(len(recent),len(d["rows"]),32):
        rows=d["rows"][j:j+32]
        ids=[tok.encode(text(r["key"],**r["example"]["original"]),add_special_tokens=False) for r in rows]
        a,b=[greedy(f,tok,ids,i) for i in (1,2)]
        recent.extend(dict(r,native={"original":[x,y]}) for r,x,y in zip(rows,a,b,strict=True))
        save(base,dict(rows=recent,deck_sha256=sha(deck)))
        print("original lookup",len(recent),len(d["rows"]),flush=True)
    contingent=[augment(r) for r in recent if natural_repair(r)]
    path=root/"PAIR_NATIVE.json"
    check=load(path)["rows"] if path.exists() else []
    assert [r["id"] for r in check]==[p["id"] for p in contingent[:len(check)]]
    for j in range(len(check),len(contingent),12):
        group=contingent[j:j+12]
        ids=[tok.encode(text(r["key"],**r["example"][k]),add_special_tokens=False)
             for r in group for k in VARIANTS[1:]]
        a,b=[greedy(f,tok,ids,k) for k in (1,2)]
        for i,r in enumerate(group):
            extra={k:[a[3*i+s],b[3*i+s]] for s,k in enumerate(VARIANTS[1:])}
            check.append(dict(r,native=dict(r["native"],**extra)))
        save(path,dict(rows=check,native_sha256=sha(base)))
        print("counterfactual lookup tables",len(check),len(contingent),flush=True)
    if not path.exists():
        save(path,dict(rows=check,native_sha256=sha(base)))
    z=summary(recent,check)
    save(root/"NATIVE_SUMMARY.json",z)
    print("residual lookup baselines",z,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)
