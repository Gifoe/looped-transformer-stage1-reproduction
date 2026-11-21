"""Create independent borrow stems with symmetric and opposite-borrow donors."""
import argparse
from pathlib import Path
from random import Random

from bidi import unordered
from borrow import needs
from edge_deck import quartet
from native import load, save, sha
import native_deck as shared
import path_deck
from rule import PREFIX

PARENT=Path("/root/ouro-borrow-paths-20261121/results/DECK.json")
SEED=261123421


def history():
    used=path_deck.history()
    for r in load(PARENT)["rows"]:
        used.update(unordered(q) for c in ("borrow","safe") for q in r[c])
    return used


def change(pair,role,delta):
    return tuple(x+(delta if i==role else 0) for i,x in enumerate(pair))


def balanced(y,d,rng,used,band):
    for _ in range(1000):
        q,steps,new=quartet(y,"borrow",3,rng,used,band)
        receiver=q[3]
        for role in rng.sample((0,1),2):
            x=receiver[role]%100
            if not d<=x<=99-d:
                continue
            step=d if role else -d
            opposite=change(receiver,role,step)
            same=change(receiver,role,-step)
            keys={unordered(opposite),unordered(same)}
            if not keys.isdisjoint(used|new):
                continue
            assert not needs(*opposite) and needs(*same)
            assert [b-a for a,b in (opposite,same)]==[y+d,y-d]
            return q,steps,dict(opposite=opposite,same=same,role=role),new|keys
    raise ValueError((y,d,"borrow"))


def safe_context(y,d,rng,used,band,role):
    for _ in range(800):
        q,steps,new=quartet(y,"safe",3,rng,used,band)
        donor=change(q[3],role,-d if role else d)
        key=unordered(donor)
        if key in used|new:
            continue
        assert not needs(*donor) and donor[1]-donor[0]==y-d
        assert all(x//100==z//100 for x,z in zip(q[3],donor,strict=True))
        return q,steps,donor,new|{key}
    raise ValueError((y,d,"safe"))


def build(dst):
    used=history()
    old=len(used)
    rng=Random(SEED)
    groups={}
    rows=[]
    for h in (2,3,4,5,6,1):
        quota=min(24,120-len(groups))
        if quota==0:
            break
        tails=list(range(59,86))
        rng.shuffle(tails)
        placed=0
        for x in tails:
            y=100*h+x
            split,n=("discovery",2) if h!=1 and placed<4 else ("confirmation",3)
            d=100-x+rng.randint(5,min(16,x-53))
            candidate=[]
            fresh=set()
            try:
                for j in range(n):
                    borrower,steps,don,xs=balanced(y,d,rng,used|fresh,j%3)
                    safe,steps2,ds,ys=safe_context(y,d,rng,used|fresh|xs,j%3,don["role"])
                    fresh.update(xs|ys)
                    candidate.append(dict(id=f"{split}-{y}-{j+1}",split=split,target=y,foil=y+100,
                                          borrow=borrower,safe=safe,borrow_steps=steps,safe_steps=steps2,
                                          donors=dict(opposite=don["opposite"],same=don["same"],safe=ds),
                                          role=don["role"],delta=d))
            except ValueError:
                print("exhausted disjoint operand inventory",y,flush=True)
                continue
            groups[str(y)]=split
            rows.extend(candidate)
            used.update(fresh)
            placed+=1
            if placed==quota:
                break
        assert h==1 or placed>=4
    assert len(groups)==120 and len(rows)==340
    rng.shuffle(rows)
    save(dst,dict(rows=rows,groups=groups,seed=SEED,earlier_unordered=old,
                  parent_sha256=sha(PARENT),prompt=PREFIX))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    if not (a.out/"DECK.json").exists():
        build(a.out/"DECK.json")
    shared.run(a.model,a.out)
