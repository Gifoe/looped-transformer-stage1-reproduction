"""Fixed same-answer native borrow and no-borrow deck."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import random

import numpy as np
import torch

from bidi import prior_pairs, unordered
from native import Forward, load, save, setup, sha
from rule import DEMONSTRATIONS, PREFIX, greedy, prompt

ROOT=Path("/root")
INPUTS=(
    ROOT/"ouro-open-carry-20261121/MANIFEST.json",
    ROOT/"ouro-carry-boundary-20261121/results/MANIFEST.json",
    ROOT/"ouro-bidirectional-20261121/results/DECK.json",
    ROOT/"ouro-rule-switch-20261121/results/DECK.json",
    ROOT/"ouro-rule-switch-20261121/results/CONTROL_DECK.json",
)


def history(paths=INPUTS):
    pilot,boundary,bidi,rule,extra=paths
    seen=prior_pairs(pilot,boundary)
    seen.update(unordered(p) for r in load(bidi)["rows"] for p in r["pairs"])
    seen.update(unordered(p) for p in DEMONSTRATIONS)
    seen.update(unordered((a,z)) for r in load(rule)["rows"] for a in r["a"] for z in (r["b"],r["source_b"]))
    seen.update(unordered(p) for r in load(extra)["rows"] for p in r["pairs"])
    return seen


def needs(a,b):
    return b%100<a%100


def build(out):
    seen=history()
    original=len(seen)
    rng=random.Random(261121540)
    choices=[t for t in range(224,576) if 24<=t%100<=75]
    rng.shuffle(choices)
    splits={}
    pool=[]
    for split,n,d in (("discovery",54,2),("confirmation",96,3)):
        y_values=choices[:n]
        choices=choices[n:]
        for y in y_values:
            splits[str(y)]=split
            for j in range(d):
                eligible=[[],[]]
                a_range=range(290,390) if split=="discovery" else range(105,289)
                for a in a_range:
                    b=a+y
                    if 400<=b<=895 and unordered((a,b)) not in seen:
                        eligible[needs(a,b)].append((a,b))
                assert all(eligible),(split,y,j)
                pairs=[rng.choice(x) for x in eligible]
                seen.update(unordered(p) for p in pairs)
                pool.append(dict(id=f"{split}-{y}-{j+1}",split=split,target=y,foil=y+100,pairs=pairs))
    rng.shuffle(pool)
    save(out,dict(seed=261121540,prompt=PREFIX,rows=pool,targets=splits,historical_unordered=original,
                  source_sha256=[sha(x) for x in INPUTS]))


def native_event(row,t):
    k1,k2=[x[t]["number"] for x in row["pred"]]
    return k1==row["foil"] and k2==row["target"]


def ci(x,levels=(.025,.975)):
    a=np.array(x,dtype=float)
    if not len(a):
        return dict(n=0,total=0,mean=None,interval=[])
    rng=np.random.default_rng(261121550)
    idx=rng.integers(0,len(a),size=(20000,len(a)))
    return dict(n=len(a),total=int(a.sum()),mean=float(a.mean()),
                interval=np.quantile(a[idx].mean(axis=1),levels).tolist())


def report(rows,targets):
    output={}
    for s in ("discovery","confirmation"):
        n=[r for r in rows if r["split"]==s]
        stats=[Counter(),Counter()]
        cluster={int(t):0 for t in sorted(targets) if targets[t]==s}
        eligible=set()
        for r in n:
            ev=[native_event(r,k) for k in (0,1)]
            cluster[r["target"]]+=int(ev[1])-int(ev[0])
            if ev[1]:
                eligible.add(r["target"])
            for t,is_repair in enumerate(ev):
                a=stats[t]
                a["n"]+=1
                m,q=[p[t]["number"] for p in r["pred"]]
                for k,x in enumerate((m,q),1):
                    a[f"K{k}_correct"]+=x==r["target"]
                    a[f"K{k}_foil"]+=x==r["foil"]
                    a[f"K{k}_missing"]+=x is None
                a["repairs"]+=m!=r["target"] and q==r["target"]
                a["regressions"]+=m==r["target"] and q!=r["target"]
                a["exact_repair"]+=is_repair
        diff=ci(list(cluster.values()))
        output[s]=dict(safe=dict(stats[0]),borrow=dict(stats[1]),unique_borrow_repairs=len(eligible),
                       paired_excess_per_target=diff)
    output["gate"]=bool(output["discovery"]["unique_borrow_repairs"]>=12 and
                   output["confirmation"]["unique_borrow_repairs"]>=20 and
                   output["confirmation"]["borrow"]["exact_repair"]>=24 and
                   output["confirmation"]["paired_excess_per_target"]["interval"][0]>0)
    return output


@torch.inference_mode()
def run(model,root):
    deck=root/"DECK.json"
    if not deck.exists():
        build(deck)
    m=load(deck)
    raw=root/"NATIVE.json"
    old=load(raw)["rows"] if raw.exists() else []
    assert [r["id"] for r in old]==[g["id"] for g in m["rows"][:len(old)]]
    m_model,tok=setup(model)
    f=Forward(m_model)
    for i in range(len(old),len(m["rows"]),24):
        cases=m["rows"][i:i+24]
        flat=[tok.encode(prompt(*p,1),add_special_tokens=False) for r in cases for p in r["pairs"]]
        by=[greedy(f,tok,flat,k) for k in (1,2)]
        old.extend(dict(r,pred=[p[2*j:2*j+2] for p in by]) for j,r in enumerate(cases))
        save(raw,dict(rows=old,deck_sha256=sha(deck)))
        print("native",len(old),len(m["rows"]),flush=True)
    v=report(old,m["targets"])
    save(root/"NATIVE_SUMMARY.json",v)
    print(v,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)

