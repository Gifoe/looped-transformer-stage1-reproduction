"""Fresh native borrowing and matched never-borrowing quartets."""
import argparse
from collections import Counter
from pathlib import Path
import random

import torch

from bidi import unordered
from borrow import ci, needs
from factorial import history as inherited_history
from native import Forward, load, save, setup, sha
from rule import PREFIX, greedy, prompt

EXTRA=Path("/root/ouro-borrow-factorial-20261121/results/DECK.json")
SCORED=("00","10","01","11","safe11")


def history():
    seen=inherited_history()
    seen.update(unordered(p) for item in load(EXTRA)["rows"] for p in item["pairs"])
    return seen


def four(a0,a1,b0,b1):
    return [(a0,b0),(a1,b0),(a0,b1),(a1,b1)]


def rand_borrow(y,rng,used,split,band):
    hi,x=divmod(y,100)
    for _ in range(4000):
        ah=rng.choice((2,3) if split=="discovery" else (1,2))
        bh=ah+hi+1
        if bh>9:
            continue
        a=rng.randrange(1,100-x)
        b=a+x
        opts=[n for n in range(29,72) if x>=max(n,100-n) and
              (n==50 if band==0 else (n<50 if band==1 else n>50))]
        d=rng.choice(opts)
        quad=four(100*ah+a,100*ah+a+d,100*bh+b,100*bh+b-(100-d))
        keys={unordered(p) for p in quad}
        if len(keys)==4 and not keys.intersection(used):
            assert [needs(*p) for p in quad]==[False,False,False,True]
            assert [q-p for p,q in quad]==[y+100,y+100-d,y+d,y]
            return quad,d,keys
    raise ValueError((y,split,band))


def rand_safe(y,rng,used,split):
    hi,x=divmod(y,100)
    for _ in range(3000):
        ah=rng.choice((2,3) if split=="discovery" else (1,2))
        a_lo=rng.randrange(1,99-x)
        room=99-x-a_lo
        if not room:
            continue
        da=rng.randint(1,min(18,a_lo))
        db=rng.randint(1,min(18,room))
        a1=100*ah+a_lo
        b1=100*(ah+hi)+a_lo+x
        quad=four(a1-da,a1,b1+db,b1)
        keys={unordered(p) for p in quad}
        if len(keys)==4 and not keys.intersection(used):
            assert not any(needs(*p) for p in quad)
            assert quad[-1][1]-quad[-1][0]==y
            return quad,[da,db],keys
    raise ValueError((y,split))


def build(root):
    expect=history()
    amount=len(expect)
    rng=random.Random(261122100)
    out=[]
    groups={}
    for h in range(2,7):
        pool=[100*h+x for x in range(53,87)]
        rng.shuffle(pool)
        for split,seq in (("discovery",pool[:12]),("confirmation",pool[12:28])):
            for y in seq:
                groups[str(y)]=split
                for j in range(2):
                    arm=len(out)%3
                    suite,d,q=rand_borrow(y,rng,expect,split,arm)
                    control,steps,p=rand_safe(y,rng,expect|q,split)
                    expect.update(q|p)
                    out.append(dict(id=f"{split}-{y}-{j+1}",split=split,target=y,foil=y+100,borrow=suite,safe=control,
                                    borrow_steps=[d,100-d],safe_steps=steps,band=arm))
    rng.shuffle(out)
    save(root,dict(rows=out,groups=groups,earlier_unordered=amount,previous_sha256=sha(EXTRA),
                   prompt=PREFIX,seed=261122100))


def is_repair(g):
    return [p[3]["number"] for p in g["pred"]]==[g["foil"],g["target"]]


def all_other_K1_correct(g):
    return all(q["number"]==y-x for q,(x,y) in zip(g["pred"][0][:3],g["borrow"][:3],strict=True))


def safe_event(g):
    return [p[4]["number"] for p in g["pred"]]==[g["foil"],g["target"]]


def report(rows,groups):
    out={}
    for split in ("discovery","confirmation"):
        sub=[g for g in rows if g["split"]==split]
        totals=Counter()
        per=[Counter() for _ in SCORED]
        diff={int(y):0 for y,s in groups.items() if s==split}
        appeared=set()
        for g in sub:
            targets=[y-x for x,y in [*g["borrow"],g["safe"][3]]]
            for i,y in enumerate(targets):
                for k in (0,1):
                    p=g["pred"][k][i]["number"]
                    per[i][f"K{k+1}_correct"]+=p==y
                    per[i][f"K{k+1}_unparsed"]+=p is None
            r=is_repair(g)
            s=safe_event(g)
            totals["joint_repairs"]+=r
            totals["matched_safe_repairs"]+=s
            totals["other_three_K1_correct_among_joint_repairs"]+=r and all_other_K1_correct(g)
            totals["all_K2"]+=all(q["number"]==y for q,y in zip(g["pred"][1],targets,strict=True))
            if r:
                appeared.add(g["target"])
            diff[g["target"]]+=int(r)-int(s)
        out[split]=dict(n=len(sub),total=totals,per={t:c for t,c in zip(SCORED,per,strict=True)},
                        distinct_repairs=len(appeared),paired=ci(list(diff.values())))
    used=out["confirmation"]
    out["gate"]=bool(used["total"]["joint_repairs"]>=24 and used["distinct_repairs"]>=18 and used["paired"]["interval"][0]>0)
    return out


@torch.inference_mode()
def run(model,root):
    deck=root/"DECK.json"
    if not deck.exists():
        build(deck)
    meta=load(deck)
    raw=root/"NATIVE.json"
    old=load(raw)["rows"] if raw.exists() else []
    assert [x["id"] for x in old]==[r["id"] for r in meta["rows"][:len(old)]]
    m,tok=setup(model)
    f=Forward(m)
    for i in range(len(old),len(meta["rows"]),10):
        block=meta["rows"][i:i+10]
        ids=[tok.encode(prompt(*p,1),add_special_tokens=False) for r in block for p in [*r["borrow"],r["safe"][3]]]
        exact=[greedy(f,tok,ids,k) for k in (1,2)]
        old.extend(dict(r,pred=[p[5*j:5*j+5] for p in exact]) for j,r in enumerate(block))
        save(raw,dict(rows=old,deck_sha256=sha(deck)))
        print("native readout quartet",len(old),len(meta["rows"]),flush=True)
    d=report(old,meta["groups"])
    save(root/"NATIVE_SUMMARY.json",d)
    print(d,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

