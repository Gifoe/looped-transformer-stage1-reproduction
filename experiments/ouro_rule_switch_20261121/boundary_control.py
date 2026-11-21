"""Same-answer prospective replication for a transfer into the hundreds."""
import argparse
from collections import Counter
from pathlib import Path
import random

import torch

from bidi import prior_pairs, unordered
from causal import interval
from native import Forward, load, save, setup, sha
from rule import DEMONSTRATIONS, greedy, prompt, value


def needs(a,b,o):
    return (a%100+b%100>=100) if not o else (b%100<a%100)


def build(pilot,boundary,previous,root):
    rule=root/"DECK.json"
    source=[pilot,boundary,previous,rule]
    seen=prior_pairs(pilot,boundary)
    seen.update(unordered(p) for r in load(previous)["rows"] for p in r["pairs"])
    seen.update(unordered(p) for p in DEMONSTRATIONS)
    seen.update(unordered((a,b)) for g in load(rule)["rows"] for a in g["a"] for b in (g["b"],g["source_b"]))
    rng=random.Random(261121900)
    deck=[]
    for op,span in enumerate(((624,899),(235,538))):
        labels=[x for x in range(*span) if 24<=x%100<=75]
        rng.shuffle(labels)
        for y in labels:
            support=[[],[]]
            for a in range(110,340):
                b=y-a if not op else y+a
                if b<400 or b>795 or unordered((a,b)) in seen:
                    continue
                assert value(a,b,op)==y
                support[needs(a,b,op)].append((a,b))
            if not all(support):
                continue
            pairs=[rng.choice(x) for x in support]
            seen.update(unordered(p) for p in pairs)
            deck.append(dict(id=f"r{op}-{y}",target=y,foil=y+(-100 if op==0 else 100),op=op,pairs=pairs))
            if sum(r["op"]==op for r in deck)==128:
                break
        assert sum(r["op"]==op for r in deck)==128
    rng.shuffle(deck)
    save(root/"CONTROL_DECK.json",dict(source_sha256=[sha(x) for x in source],rows=deck,seed=261121900))


def summary(data):
    results={}
    for o in (0,1):
        rows=[r for r in data if r["op"]==o]
        count=[Counter(),Counter()]
        d=[]
        for g in rows:
            events=[]
            for t in (0,1):
                target=g["target"]
                foil=g["foil"]
                p,q=[x[t]["number"] for x in g["pred"]]
                m=count[t]
                for k,z in enumerate((p,q),1):
                    m[f"K{k}_correct"]+=z==target
                    m[f"K{k}_foil"]+=z==foil
                    m[f"K{k}_unparsed"]+=z is None
                m["WC"]+=p!=target and q==target
                m["CW"]+=p==target and q!=target
                exact=p==foil and q==target
                m["foil_repair"]+=exact
                events.append(exact)
            d.append(int(events[1])-int(events[0]))
        e=interval(d,(.0125,.9875))
        results[str(o)]=dict(n=len(rows),no_need=count[0],need=count[1],paired=e,
                             gate=count[1]["foil_repair"]>=12 and e["interval"][0]>0)
    return results


@torch.inference_mode()
def run(a):
    root=a.root
    manifest=root/"CONTROL_DECK.json"
    if not manifest.exists():
        build(a.pilot,a.boundary,a.previous,root)
    m=load(manifest)["rows"]
    raw=root/"CONTROL_RAW.json"
    rows=load(raw)["rows"] if raw.exists() else []
    assert [r["id"] for r in rows]==[r["id"] for r in m[:len(rows)]]
    model,tok=setup(a.model)
    f=Forward(model)
    for i in range(len(rows),len(m),24):
        block=m[i:i+24]
        ids=[tok.encode(prompt(*p,r["op"]),add_special_tokens=False)
             for r in block for p in r["pairs"]]
        result=[greedy(f,tok,ids,k) for k in (1,2)]
        rows.extend(dict(r,pred=[d[j*2:j*2+2] for d in result]) for j,r in enumerate(block))
        save(raw,dict(rows=rows,manifest_sha256=sha(manifest)))
        print("same answer holdout",len(rows),len(m),flush=True)
    s=summary(rows)
    save(root/"CONTROL_SUMMARY.json",s)
    print(s,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("model","pilot","boundary","previous","root"):
        p.add_argument(n,type=Path)
    run(p.parse_args())

