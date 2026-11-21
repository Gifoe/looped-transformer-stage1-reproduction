"""Freeze fresh natural lookup cases and their outcome-defined value swap."""
import argparse
from pathlib import Path
from random import Random

import torch

from native import Forward, load, save, setup, sha
from query_deck import KEYS, PREFIX, text
from residual_deck import augment, history as earlier, natural_repair
from rule import greedy

PARENT=Path("/root/ouro-lookup-residual-20261121/results")
SEED=261124791


def history():
    out=earlier()
    for filename in ("NATIVE.json","PAIR_NATIVE.json"):
        out.update(text(r["key"],**q) for r in load(PARENT/filename)["rows"] for q in r["example"].values())
    return out


def build():
    rng=Random(SEED)
    seen=history()
    rows=[]
    for j,y in enumerate(rng.sample(range(100,1000),448)):
        key=KEYS[j%4]
        values=rng.sample([v for v in range(100,1000) if v!=y],3)
        values.insert(key-1,y)
        p=dict(order=rng.sample(KEYS,4),values=values)
        prompt=text(key,**p)
        assert prompt not in seen
        seen.add(prompt)
        rows.append(dict(id=f"lookup-branches-{j:03}",target=y,key=key,example={"original":p}))
    return dict(seed=SEED,prefix=PREFIX,parent_sha256=sha(PARENT/"PAIR_NATIVE.json"),rows=rows)


def paired_variant(r):
    q=augment(r)
    return dict(q,example={k:q["example"][k] for k in ("original","swapped")})


def swapped_repair(r):
    a,b=[x["number"] for x in r["native"]["swapped"]]
    return b==r["old"] and a!=b and a in r["example"]["swapped"]["values"]


def summary(base,pairs):
    a=[sum(r["native"]["original"][k]["number"]==r["target"] for r in base) for k in (0,1)]
    b=[sum(r["native"]["swapped"][k]["number"]==r["old"] for r in pairs) for k in (0,1)]
    n=sum(natural_repair(r) for r in base)
    return dict(n=len(base),original_k1=a[0],original_k2=a[1],original_exact=n,swapped_k1=b[0],
                swapped_k2=b[1],swapped_correct_both=sum(all(x["number"]==r["old"] for x in r["native"]["swapped"]) for r in pairs),
                swapped_exact=sum(swapped_repair(r) for r in pairs),gate=n>=40)


@torch.inference_mode()
def run(model,root):
    deck=root/"DECK.json"
    if not deck.exists():
        save(deck,build())
    d=load(deck)
    m,tok=setup(model)
    f=Forward(m)
    files=[root/name for name in ("NATIVE.json","PAIR_NATIVE.json")]
    old=load(files[0])["rows"] if files[0].exists() else []
    assert [r["id"] for r in old]==[r["id"] for r in d["rows"][:len(old)]]
    def bench(rows,side):
        inp=[tok.encode(text(r["key"],**r["example"][side]),add_special_tokens=False) for r in rows]
        a,b=[greedy(f,tok,inp,k) for k in (1,2)]
        return [dict(r,native=dict(r.get("native",{}),**{side:[x,y]})) for r,x,y in zip(rows,a,b,strict=True)]
    for j in range(len(old),len(d["rows"]),32):
        old+=bench(d["rows"][j:j+32],"original")
        save(files[0],dict(rows=old,deck_sha256=sha(deck)))
        print("original baselines",len(old),len(d["rows"]),flush=True)
    nh=sha(files[0])
    want=[paired_variant(r) for r in old if natural_repair(r)]
    pairs=load(files[1])["rows"] if files[1].exists() else []
    assert [r["id"] for r in pairs]==[r["id"] for r in want[:len(pairs)]]
    for j in range(len(pairs),len(want),32):
        pairs+=bench(want[j:j+32],"swapped")
        save(files[1],dict(rows=pairs,native_sha256=nh))
        print("paired baselines",len(pairs),len(want),flush=True)
    if not files[1].exists():
        save(files[1],dict(rows=pairs,native_sha256=nh))
    z=summary(old,pairs)
    save(root/"NATIVE_SUMMARY.json",z)
    print("lookup branch baselines",z,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)
