"""Collect independent standard and shifted-format native lookup cohorts."""
import argparse
from pathlib import Path
from random import Random

import torch

from branches_deck import history as inherited
from native import Forward, load, save, setup, sha
from query_deck import PREFIX, text as standard
from rule import greedy

PARENT=Path("/root/ouro-lookup-branches-20261121/results/PAIR_NATIVE.json")
SEED=261125091
DOMAINS=("standard","shifted")
VARIANTS=("original","swapped","unrelated_original","unrelated_swapped")


def prompt(row,side):
    e=row["example"][side]
    key=row["key"]
    if row["domain"]=="standard":
        return standard(key,**e)
    links=" | ".join(f"{k}: {e['values'][k-5]}" for k in e["order"])
    return PREFIX+f"Lookup key = {key}; Entries = [{links}]; Answer:"


def history():
    seen=inherited()
    seen.update(standard(r["key"],**p) for r in load(PARENT)["rows"] for p in r["example"].values())
    return seen


def build():
    rng=Random(SEED)
    seen=history()
    labels=rng.sample(range(100,1000),768)
    rows=[]
    for j,y in enumerate(labels):
        domain=DOMAINS[j%2]
        keys=list(range(1,5)) if domain=="standard" else list(range(5,9))
        k=(j//2)%4
        other=rng.sample([v for v in range(100,1000) if v!=y],3)
        other.insert(k,y)
        r=dict(id=f"lookup-onset-{j:03}",domain=domain,target=y,key=keys[k],
               example=dict(original=dict(order=rng.sample(keys,4),values=other)))
        t=prompt(r,"original")
        assert t not in seen
        seen.add(t)
        rows.append(r)
    return dict(prefix=PREFIX,seed=SEED,parent_sha256=sha(PARENT),rows=rows)


def repair(r):
    a,b=[p["number"] for p in r["native"]["original"]]
    return a in r["example"]["original"]["values"] and a!=r["target"] and b==r["target"]


def supplement(r):
    values=r["example"]["original"]["values"]
    actual=r["target"]
    wrong=r["native"]["original"][0]["number"]
    a,b=values.index(actual),values.index(wrong)
    x,y=[k for k in range(4) if k not in (a,b)]
    def flip(seq,i,j):
        q=list(seq)
        q[i],q[j]=q[j],q[i]
        return q
    changed=flip(values,a,b)
    states=dict(swapped=changed,unrelated_original=flip(values,x,y),unrelated_swapped=flip(changed,x,y))
    variants={k:dict(order=r["example"]["original"]["order"],values=v) for k,v in states.items()}
    return dict(r,old=wrong,example=dict(r["example"],**variants))


def pair(r):
    return all(r["native"][k][1]["number"]==r["old" if k.endswith("swapped") else "target"] for k in VARIANTS)


def summary(native,paired):
    z={}
    for tag in DOMAINS:
        a=[r for r in native if r["domain"]==tag]
        b=[r for r in paired if r["domain"]==tag]
        counts=[sum(r["native"]["original"][k]["number"]==r["target"] for r in a) for k in (0,1)]
        z[tag]=dict(n=len(a),native_k1=counts[0],native_k2=counts[1],exact=sum(map(repair,a)),
                    pairs=sum(map(pair,b)))
    z["gate"]=all(z[k]["exact"]>=32 and z[k]["pairs"]>=24 for k in DOMAINS)
    return z


@torch.inference_mode()
def run(model,root):
    deck=root/"DECK.json"
    if not deck.exists():
        save(deck,build())
    manifest=load(deck)
    m,tok=setup(model)
    f=Forward(m)
    src=root/"NATIVE.json"
    all_rows=load(src)["rows"] if src.exists() else []
    assert [r["id"] for r in all_rows]==[r["id"] for r in manifest["rows"][:len(all_rows)]]
    for j in range(len(all_rows),len(manifest["rows"]),32):
        grp=manifest["rows"][j:j+32]
        ids=[tok.encode(prompt(r,"original"),add_special_tokens=False) for r in grp]
        k,l=[greedy(f,tok,ids,h) for h in (1,2)]
        all_rows.extend(dict(r,native={"original":[a,b]}) for r,a,b in zip(grp,k,l,strict=True))
        save(src,dict(rows=all_rows,deck_sha256=sha(deck)))
        print("native input tables",len(all_rows),len(manifest["rows"]),flush=True)
    protocol=[supplement(r) for r in all_rows if repair(r)]
    dest=root/"PAIRED.json"
    passed=load(dest)["rows"] if dest.exists() else []
    assert [r["id"] for r in passed]==[r["id"] for r in protocol[:len(passed)]]
    for j in range(len(passed),len(protocol),12):
        grp=protocol[j:j+12]
        ids=[tok.encode(prompt(r,k),add_special_tokens=False) for r in grp for k in VARIANTS[1:]]
        a,b=[greedy(f,tok,ids,p) for p in (1,2)]
        for i,r in enumerate(grp):
            passed.append(dict(r,native=dict(r["native"],**{s:[a[3*i+k],b[3*i+k]] for k,s in enumerate(VARIANTS[1:])})))
        save(dest,dict(rows=passed,native_sha256=sha(src)))
        print("native paired tables",len(passed),len(protocol),flush=True)
    if not dest.exists():
        save(dest,dict(rows=passed,native_sha256=sha(src)))
    z=summary(all_rows,passed)
    save(root/"NATIVE_SUMMARY.json",z)
    print("native summary",z,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)
