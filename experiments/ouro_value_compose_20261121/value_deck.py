"""Construct independent operand mixtures and same-A opposite-borrow contexts."""
import argparse
from pathlib import Path
from random import Random

from bidi import unordered
from borrow import needs
import boundary_deck
from edge_deck import quartet
from native import load, save, sha
import native_deck
from rule import PREFIX

PARENT=Path("/root/ouro-boundary-factor-20261121/results/DECK.json")
SEED=261123745


def history():
    seen=boundary_deck.history()
    for r in load(PARENT)["rows"]:
        seen.update(unordered(p) for p in [*r["borrow"],*r["safe"],*r["donors"].values()])
    return seen


def draw(y,d,e,label,rng,used,band):
    for _ in range(800):
        q,steps,keys=quartet(y,label,3,rng,used,band)
        a,b=q[3]
        n=a-d if label=="borrow" else a+d
        if a//100!=n//100 or b%100-e<0 or b%100+e>=100:
            continue
        pairs={"plus":(n,b+e),"minus":(n,b-e)}
        ids={unordered(x) for x in pairs.values()}
        if not ids.isdisjoint(used|keys):
            continue
        assert len(ids)==2 and not needs(n,b) and not needs(*pairs["plus"])
        assert needs(*pairs["minus"])==(label=="borrow")
        assert len({y,y+100,b-n,b+e-n,b-e-n})==5
        return q,steps,pairs,keys|ids
    raise ValueError((y,label,band))


def build(dst):
    seen=history()
    n=len(seen)
    rng=Random(SEED)
    rows=[]
    groups={}
    for h in (2,3,4,5,6,1,7):
        quota=min(24,120-len(groups))
        if not quota:
            break
        pool=list(range(59,85))
        rng.shuffle(pool)
        allocated=0
        for tail in pool:
            y=h*100+tail
            split,count=("discovery",2) if 2<=h<=6 and allocated<4 else ("confirmation",3)
            cross=rng.randint(5,10)
            d=100-tail+cross
            e=cross+rng.randint(3,7)
            safe_e=min(e,(tail-d)//2,97-tail)
            new=[]
            local=set()
            try:
                for j in range(count):
                    qb,sb,db,b=draw(y,d,e,"borrow",rng,seen|local,j%3)
                    qs,ss,ds,s=draw(y,d,safe_e,"safe",rng,seen|local|b,j%3)
                    local.update(b|s)
                    new.append(dict(id=f"{split}-{y}-{j+1}",split=split,target=y,foil=y+100,
                                    width=3,borrow=qb,safe=qs,borrow_steps=sb,safe_steps=ss,
                                    borrow_donors=db,safe_donors=ds,delta=d,borrow_e=e,safe_e=safe_e,
                                    mix=y+d,safe_mix=y-d))
            except ValueError:
                print("exhausted disjoint operand inventory",y,flush=True)
                continue
            groups[str(y)]=split
            rows+=new
            seen.update(local)
            allocated+=1
            if allocated==quota:
                break
        assert not 2<=h<=6 or allocated>=4
    assert len(rows)==340 and len(groups)==120
    rng.shuffle(rows)
    save(dst,dict(rows=rows,groups=groups,seed=SEED,earlier_unordered=n,
                  parent_sha256=sha(PARENT),prompt=PREFIX))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    if not (a.out/"DECK.json").exists():
        build(a.out/"DECK.json")
    native_deck.run(a.model,a.out)
