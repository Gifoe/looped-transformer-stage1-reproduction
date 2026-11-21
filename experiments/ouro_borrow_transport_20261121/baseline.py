"""Freeze and score disjoint-digit donor/recipient subtraction cohorts."""
import argparse
from collections import Counter
from pathlib import Path
import random

import torch

import borrow
from bidi import unordered
from native import Forward, load, save, setup, sha
from rule import PREFIX, greedy, prompt

EXTRA=Path("/root/ouro-borrow-query-20261121/results/DECK.json")


def history():
    pairs=borrow.history()
    pairs.update(unordered(p) for row in load(EXTRA)["rows"] for p in row["pairs"])
    return pairs


def build(out):
    seen=history()
    origin=len(seen)
    rng=random.Random(261121801)
    groups={}
    cases=[]
    for h in range(2,7):
        pool=list(range(100*h+24,100*h+76))
        rng.shuffle(pool)
        selected={"donors":pool[:13],"receivers":pool[13:29]}
        for label,ys in selected.items():
            bounds=(205,300) if label=="donors" else (105,205)
            for y in ys:
                groups[str(y)]=label
                for j in range(3):
                    support=[[],[]]
                    for a in range(*bounds):
                        b=a+y
                        if 400<=b<998 and unordered((a,b)) not in seen:
                            support[borrow.needs(a,b)].append((a,b))
                    assert all(support),(label,y,j)
                    pairs=[rng.choice(x) for x in support]
                    seen.update(unordered(x) for x in pairs)
                    cases.append(dict(id=f"{label}-{y}-{j+1}",split=label,target=y,foil=y+100,pairs=pairs))
    rng.shuffle(cases)
    save(out,dict(rows=cases,targets=groups,historical_unique=origin,seed=261121801,prompt=PREFIX,
                  prior_sha256=[sha(x) for x in (*borrow.INPUTS,EXTRA)]))


def exact_safe(row):
    return all(x[0]["number"]==row["target"] for x in row["pred"])


def summary(rows,groups):
    tables={}
    for split in ("donors","receivers"):
        counts=[Counter(),Counter()]
        unit={int(t):0 for t,s in groups.items() if s==split}
        eligible=set()
        donor_ready=0
        for r in rows:
            if r["split"]!=split:
                continue
            own=[borrow.native_event(r,k) for k in (0,1)]
            unit[r["target"]]+=int(own[1])-int(own[0])
            if own[1]:
                eligible.add(r["target"])
            donor_ready+=own[1] and exact_safe(r)
            for k in (0,1):
                c=counts[k]
                c["n"]+=1
                p,q=[x[k]["number"] for x in r["pred"]]
                c["K1_correct"]+=p==r["target"]
                c["K2_correct"]+=q==r["target"]
                c["exact_repair"]+=own[k]
                c["K1_unparsed"]+=p is None
                c["K2_unparsed"]+=q is None
        tables[split]=dict(safe=dict(counts[0]),borrow=dict(counts[1]),distinct_repairs=len(eligible),
                           exact_repair_and_safe_double_correct=donor_ready,
                           paired=borrow.ci(list(unit.values())))
    r=tables["receivers"]
    tables["gate"]=(r["borrow"]["exact_repair"]>=28 and r["distinct_repairs"]>=20 and r["paired"]["interval"][0]>0)
    return tables


@torch.inference_mode()
def run(model,root):
    deck=root/"DECK.json"
    if not deck.exists():
        build(deck)
    m=load(deck)
    raw=root/"NATIVE.json"
    rows=load(raw)["rows"] if raw.exists() else []
    assert [r["id"] for r in rows]==[r["id"] for r in m["rows"][:len(rows)]]
    mod,tok=setup(model)
    f=Forward(mod)
    for i in range(len(rows),len(m["rows"]),24):
        block=m["rows"][i:i+24]
        ids=[tok.encode(prompt(*p,1),add_special_tokens=False) for r in block for p in r["pairs"]]
        vals=[greedy(f,tok,ids,k) for k in (1,2)]
        rows.extend(dict(r,pred=[v[j*2:j*2+2] for v in vals]) for j,r in enumerate(block))
        save(raw,dict(rows=rows,deck_sha256=sha(deck)))
        print("native",len(rows),len(m["rows"]),flush=True)
    data=summary(rows,m["targets"])
    save(root/"NATIVE_SUMMARY.json",data)
    print(data,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)

