"""Construct independently composable low-digit changes and measure the frozen model."""
import argparse
from collections import Counter
from pathlib import Path
import random

import torch

import baseline as prior
from bidi import unordered
from borrow import ci, needs
from native import Forward, load, save, setup, sha
from rule import PREFIX, greedy, prompt

EXTRA=Path("/root/ouro-borrow-transport-20261121/results/DECK.json")
LABELS=("base","a_only","b_only","joint","safe")


def history():
    seen=prior.history()
    seen.update(unordered(p) for r in load(EXTRA)["rows"] for p in r["pairs"])
    return seen


def make(out):
    rng=random.Random(261121905)
    seen=history()
    old=len(seen)
    ys=[100*h+x for h in range(2,6) for x in range(53,88)]
    rng.shuffle(ys)
    rows=[]
    targets={}
    for label,take in (("discovery",30),("confirmation",70)):
        pool=ys[:take]
        ys=ys[take:]
        for y in pool:
            y_hi,x=divmod(y,100)
            for j in range(2):
                slot=len(rows)%3
                # Fixed equal, A-smaller and B-smaller distributions.
                if slot==0:
                    ds=(50,)
                elif slot==1:
                    ds=tuple(range(max(29,100-x),50))
                else:
                    ds=tuple(range(51,min(71,x)+1))
                ab=None
                for _ in range(3000):
                    a_hi=rng.choice((3,4) if label=="discovery" else (1,2))
                    b_hi=a_hi+y_hi+1
                    if b_hi>9:
                        continue
                    d=rng.choice(ds)
                    a0=rng.randint(1,98-x)
                    b0=a0+x
                    a1=a0+d
                    b1=b0-(100-d)
                    if min(a0,b0,a1,b1)<0 or max(a0,b0,a1,b1)>99:
                        continue
                    corners=[(100*a_hi+a,100*b_hi+b) for a,b in ((a0,b0),(a1,b0),(a0,b1),(a1,b1))]
                    keys={unordered(p) for p in corners}
                    if len(keys)==4 and not keys.intersection(seen):
                        ab=(corners,d,keys)
                        break
                if ab is None:
                    raise RuntimeError((label,y,j,slot))
                corners,d,keys=ab
                match=[(a,a+y) for a in range(120,390) if 400<=a+y<=996 and not needs(a,a+y)
                       and unordered((a,a+y)) not in seen|keys]
                assert match,(label,y,j)
                control=rng.choice(match)
                seen.update(keys|{unordered(control)})
                pairs=corners+[control]
                assert [needs(*q) for q in pairs]==[False,False,False,True,False]
                values=[b-a for a,b in pairs]
                assert values==[y+100,y+100-d,y+d,y,y]
                stem=dict(id=f"{label}-{y}-{j+1}",split=label,group=y,pairs=pairs,answers=values,
                           foil=y+100,a_delta=d,b_delta=100-d,amplitude_arm=slot)
                rows.append(stem)
                targets[str(y)]=label
    rng.shuffle(rows)
    save(out,dict(rows=rows,targets=targets,seed=261121905,prompt=PREFIX,
                  previous_sha256=sha(EXTRA),excluded_unordered=old))


def number(row,depth,corner):
    return row["pred"][depth][corner]["number"]


def corrected(row,t):
    return number(row,0,t)==row["foil"] and number(row,1,t)==row["answers"][t]


def analyze(rows,targets):
    out={}
    for part in ("discovery","confirmation"):
        subset=[g for g in rows if g["split"]==part]
        summary=[]
        bins={int(y):0 for y,s in targets.items() if s==part}
        natural=set()
        for i,n in enumerate(LABELS):
            all_=Counter()
            for g in subset:
                for k in (0,1):
                    x=number(g,k,i)
                    all_[f"K{k+1}_exact"]+=x==g["answers"][i]
                    all_[f"K{k+1}_unparsed"]+=x is None
                if i>=3:
                    all_["exact_plus100_recovery"]+=corrected(g,i)
            summary.append(dict(name=n,**all_))
        for g in subset:
            move=int(corrected(g,3))-int(corrected(g,4))
            bins[g["group"]]+=move
            if corrected(g,3):
                natural.add(g["group"])
        out[part]=dict(n=len(subset),by_corner=summary,natural_groups=len(natural),
                       K2_all_four=sum(all(number(g,1,k)==g["answers"][k] for k in range(4)) for g in subset),
                       paired_omission=ci(list(bins.values())))
    a=out["confirmation"]
    out["graft_gate"]=a["K2_all_four"]>=100 and all(100*g["K2_exact"]>=85*a["n"] for g in a["by_corner"][:4])
    out["natural_gate"]=(a["by_corner"][3]["exact_plus100_recovery"]>=14 and a["natural_groups"]>=10
                         and a["paired_omission"]["interval"][0]>0)
    return out


@torch.inference_mode()
def run(model,root):
    file=root/"DECK.json"
    if not file.exists():
        make(file)
    deck=load(file)
    raw=root/"NATIVE.json"
    rows=load(raw)["rows"] if raw.exists() else []
    assert [g["id"] for g in rows]==[g["id"] for g in deck["rows"][:len(rows)]]
    m,tok=setup(model)
    f=Forward(m)
    for offset in range(len(rows),len(deck["rows"]),10):
        group=deck["rows"][offset:offset+10]
        inputs=[tok.encode(prompt(*p,1),add_special_tokens=False) for z in group for p in z["pairs"]]
        pred=[greedy(f,tok,inputs,k) for k in (1,2)]
        rows.extend(dict(g,pred=[m[5*j:5*j+5] for m in pred]) for j,g in enumerate(group))
        save(raw,dict(rows=rows,deck_sha256=sha(file)))
        print("native factorial",len(rows),len(deck["rows"]),flush=True)
    summary=analyze(rows,deck["targets"])
    save(root/"NATIVE_SUMMARY.json",summary)
    print(summary,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

