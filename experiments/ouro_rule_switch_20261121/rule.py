"""Freeze a matched arithmetic rule switch deck and execute native inference."""
import argparse
from collections import Counter
from pathlib import Path
import random
import re

import numpy as np
import torch

from bidi import prior_pairs, unordered
from native import Forward, load, output, pad, save, setup, sha

OPERATORS=("A + B","B - A")
DEMONSTRATIONS=((147,538),(231,527),(312,484))
PREFIX="Read A and B and apply the rule. Give the integer answer.\n"+"".join(
    f"A = {a}, B = {b}, Rule: {rule}, Answer: {answer}\n"
    for a,b in DEMONSTRATIONS for rule,answer in zip(OPERATORS,(a+b,b-a),strict=True))
MAX_NEW=6


def value(a,b,op):
    return a+b if op==0 else b-a


def prompt(a,b,op):
    return PREFIX+f"A = {a}, B = {b}, Rule: {OPERATORS[op]}, Answer:"


def parse(text):
    first=next((s.strip() for s in text.splitlines() if s.strip()),"")
    return int(first) if re.fullmatch(r"[0-9]{1,4}",first) else None


def freeze(pilot,boundary,previous,out):
    seen=prior_pairs(pilot,boundary)
    for row in load(previous)["rows"]:
        seen.update(unordered(p) for p in row["pairs"])
    seen.update(unordered(p) for p in DEMONSTRATIONS)
    origin=len(seen)
    rng=random.Random(26112101)
    rows=[]
    for split,n in (("discovery",72),("confirmation",144)):
        low,high=(300,449) if split=="discovery" else (115,289)
        for j in range(n):
            band=j%3
            min_d,max_d=((5,9),(17,39),(73,139))[band]
            for _ in range(20000):
                delta=rng.randint(min_d,max_d)
                a0=rng.randint(low,high-delta)
                a1=a0+delta
                b=rng.randint(450,750)
                diff=rng.randint(20,59)*(1 if rng.random()<.5 else -1)
                source_b=b+diff
                keys={unordered((a,z)) for a in (a0,a1) for z in (b,source_b)}
                if len(keys)!=4 or seen.intersection(keys):
                    continue
                if not all(100<=value(a,z,o)<=999
                           for a in (a0,a1) for z in (b,source_b) for o in (0,1)):
                    continue
                if abs(diff)==delta:
                    continue
                break
            else:
                raise ValueError((split,j))
            seen.update(keys)
            rows.append(dict(id=f"{split}-{j+1:03}",split=split,a=[a0,a1],b=b,source_b=source_b,delta=delta,band=band,
                carry_flip=((a0%100+b%100)>=100)!=((a1%100+b%100)>=100),
                borrow_flip=(b%100<a0%100)!=(b%100<a1%100),
                target=[[value(a,b,o) for o in (0,1)] for a in (a0,a1)]))
    rng.shuffle(rows)
    save(out,dict(prefix=PREFIX,seed=26112101,historical_unique_unordered=origin,
                  source_hashes=[sha(p) for p in (pilot,boundary,previous)],rows=rows))


def greedy(f,tok,ids,k):
    tokens=[[] for _ in ids]
    active=list(range(len(ids)))
    for _ in range(MAX_NEW):
        batch,p=pad([ids[j]+tokens[j] for j in active])
        nxt=output(f.model,f.run(batch,k)[-1],p).argmax(-1).tolist()
        remain=[]
        for j,z in zip(active,nxt,strict=True):
            tokens[j].append(z)
            if z!=tok.eos_token_id and "\n" not in tok.decode(tokens[j],skip_special_tokens=True).strip():
                remain.append(j)
        active=remain
        if not active:
            break
    return text_results(tok,tokens)


def text_results(tok,tokens):
    text=[tok.decode(t,skip_special_tokens=True) for t in tokens]
    return [dict(text=s,number=parse(s),tokens=t) for s,t in zip(text,tokens,strict=True)]


def counts(native):
    table={}
    for s in ("discovery","confirmation"):
        r=[x for x in native if x["split"]==s]
        detail=Counter()
        for g in r:
            for k in (0,1):
                hit=[]
                for i in (0,1):
                    for o in (0,1):
                        pred=g["native"][k][2*i+o]["number"]
                        cur=pred==g["target"][i][o]
                        detail[f"K{k+1}_rule{o}_correct"]+=cur
                        detail[f"K{k+1}_rule{o}_unparsed"]+=pred is None
                        hit.append(cur)
                detail[f"K{k+1}_all_four"]+=all(hit)
            for j,y in enumerate(sum(g["target"],[])):
                a,b=[p[j]["number"] for p in g["native"]]
                detail["K1_to_K2_repair"]+=a!=y and b==y
                detail["K1_to_K2_damage"]+=a==y and b!=y
        table[s]=dict(detail,n_stems=len(r),n_rule=2*len(r),
            gate=detail["K2_all_four"]>=90 and all(detail[f"K2_rule{o}_correct"]>=1.6*len(r) for o in (0,1)))
    return table


@torch.inference_mode()
def run(args):
    root=args.out
    deck=root/"DECK.json"
    if not deck.exists():
        freeze(args.pilot,args.boundary,args.previous,deck)
    m=load(deck)
    raw=root/"NATIVE.json"
    old=load(raw)["rows"] if raw.exists() else []
    assert [g["id"] for g in old]==[g["id"] for g in m["rows"][:len(old)]]
    model,tok=setup(args.model)
    f=Forward(model)
    for i in range(len(old),len(m["rows"]),12):
        pack=m["rows"][i:i+12]
        tokens=[tok.encode(prompt(a,g["b"],o),add_special_tokens=False) for g in pack for a in g["a"] for o in (0,1)]
        rows=[greedy(f,tok,tokens,k) for k in (1,2)]
        old.extend(dict(g,native=[y[4*j:4*j+4] for y in rows]) for j,g in enumerate(pack))
        save(raw,dict(deck_sha256=sha(deck),rows=old))
        print("native",len(old),len(m["rows"]),flush=True)
    r=counts(old)
    save(root/"NATIVE_SUMMARY.json",r)
    print(r,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for key in ("model","pilot","boundary","previous","out"):
        p.add_argument(key,type=Path)
    run(p.parse_args())

