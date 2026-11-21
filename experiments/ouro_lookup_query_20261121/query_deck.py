"""Freeze a key, table order and value-remapping lookup test."""
import argparse
from collections import Counter
from pathlib import Path
from random import Random

import torch

from lookup import PREFIX, template as old_template
from native import Forward, load, save, setup, sha
from rule import greedy

SEED=261124107
KEYS=(1,2,3,4)
VARIANTS=("original","values","crossed")
PARENT=Path("/root/ouro-value-compose-20261121/results/LOOKUP_DECK.json")


def text(key,values,order):
    table=", ".join(f"{k}: {values[k-1]}" for k in order)
    return PREFIX+f"Query = {key}; Table = {table}; Answer:"


def history():
    r=load(PARENT)["rows"]
    return {PREFIX+old_template(x[a+"_key"],x[a]) for x in r for a in ("receiver","donor")}


def build():
    rng=Random(SEED)
    used=history()
    targets=rng.sample(range(110,990),384)
    records=[]
    for j,y in enumerate(targets):
        key=KEYS[j%len(KEYS)]
        values=rng.sample([n for n in range(110,990) if n!=y],3)
        values.insert(key-1,y)
        order=rng.sample(KEYS,4)
        new_index=rng.choice([k for k in KEYS if k!=key])
        swap=values.copy()
        swap[key-1],swap[new_index-1]=swap[new_index-1],swap[key-1]
        moved=rng.choice([k for k in range(4) if k!=order.index(key)])
        crossed=rng.sample([k for k in KEYS if k!=key],3)
        crossed.insert(moved,key)
        example=dict(original=dict(values=values,order=order),values=dict(values=swap,order=order),
                     crossed=dict(values=swap,order=crossed))
        p={text(key,**e) for e in example.values()}
        assert len(p)==3 and p.isdisjoint(used)
        used.update(p)
        row=dict(id=f"lookup-query-{j:03}",split="development" if j<64 else "confirmation",key=key,
                 target=y,new=swap[key-1],example=example)
        records.append(row)
    return dict(seed=SEED,rows=records,prefix=PREFIX,prior_sha256=sha(PARENT))


def eligible(row,kind):
    a,b=row["native"][kind]
    target=row["target"] if kind=="original" else row["new"]
    return a["number"] in row["example"][kind]["values"] and a["number"]!=target and b["number"]==target


def summary(rows):
    out={}
    for fold in ("development","confirmation"):
        chosen=[r for r in rows if r["split"]==fold]
        stats={}
        for name in VARIANTS:
            counts=Counter(n=len(chosen),k1=0,k2=0,corrected=0,wrong_table_to_correct=0,regressed=0)
            for r in chosen:
                target=r["target"] if name=="original" else r["new"]
                n,m=[p["number"] for p in r["native"][name]]
                counts["k1"]+=n==target
                counts["k2"]+=m==target
                counts["corrected"]+=n!=target and m==target
                counts["regressed"]+=n==target and m!=target
                counts["wrong_table_to_correct"]+=eligible(r,name)
            stats[name]=dict(counts)
        out[fold]=stats
    out["gate"]=out["development"]["original"]["wrong_table_to_correct"]>=8 and out["confirmation"]["original"]["wrong_table_to_correct"]>=32
    return out


@torch.inference_mode()
def run(model,root):
    deck=root/"DECK.json"
    if not deck.exists():
        save(deck,build())
    plan=load(deck)
    m,tok=setup(model)
    f=Forward(m)
    path=root/"NATIVE.json"
    current=load(path)["rows"] if path.exists() else []
    assert [r["id"] for r in current]==[x["id"] for x in plan["rows"][:len(current)]]
    for j in range(len(current),len(plan["rows"]),12):
        group=plan["rows"][j:j+12]
        ids=[tok.encode(text(r["key"],**r["example"][k]),add_special_tokens=False) for r in group for k in VARIANTS]
        a,b=[greedy(f,tok,ids,k) for k in (1,2)]
        for i,r in enumerate(group):
            current.append(dict(r,native={tag:[a[3*i+k],b[3*i+k]] for k,tag in enumerate(VARIANTS)}))
        save(path,dict(rows=current,deck_sha256=sha(deck)))
        print("lookup natural baselines",len(current),len(plan["rows"]),flush=True)
    result=summary(current)
    save(root/"NATIVE_SUMMARY.json",result)
    print("native table counts",result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)
