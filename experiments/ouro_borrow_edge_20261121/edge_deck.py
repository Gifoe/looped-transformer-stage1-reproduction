"""Freeze operand-role layouts and independent numeric prompts."""
import argparse
from pathlib import Path
from random import Random

from bidi import unordered
from borrow import needs
from native import load, save, sha
import native_deck as native
import origin_deck
from rule import PREFIX

PARENT=Path("/root/ouro-borrow-origin-20261121/results/DECK.json")
SEED=261122901


def history():
    known,_=origin_deck.history()
    for r in load(PARENT)["rows"]:
        known.update(unordered(x) for k in ("borrow","safe") for x in r[k])
    return known


def quartet(y,kind,width,rng,seen,band):
    h,r=divmod(y,100)
    for _ in range(8000):
        high=rng.randint(1,9-h-(kind=="borrow"))
        shift=0 if width==3 else 1000*rng.randint(2,8)
        if kind=="borrow":
            a=rng.randrange(1,100-r)
            b=a+r
            opts=[d for d in range(30,71) if r>=max(d,100-d) and
                  (d==50 if band==0 else d<50 if band==1 else d>50)]
            d=rng.choice(opts)
            da,db=d,100-d
            a1=shift+100*high+a+d
            b1=shift+100*(high+h+1)+b-db
        else:
            a=rng.randint(1,98-r)
            da=rng.randint(1,min(18,a))
            db=rng.randint(1,min(18,99-r-a))
            a1=shift+100*high+a
            b1=shift+100*(high+h)+a+r
        q=native.four(a1-da,a1,b1+db,b1)
        keys={unordered(x) for x in q}
        if len(keys)==4 and keys.isdisjoint(seen):
            assert [needs(*x) for x in q]==[False,False,False,kind=="borrow"]
            assert b1-a1==y and all(len(str(x))==width for p in q for x in p)
            return q,[da,db],keys
    raise ValueError((y,kind,width))


def build(path):
    seen=history()
    older=len(seen)
    rng=Random(SEED)
    groups={}
    rows=[]
    for h in range(2,7):
        nums=list(range(54,87))
        rng.shuffle(nums)
        for g,values in (("discovery",nums[:4]),("confirmation",nums[4:28])):
            for x in values:
                y=h*100+x
                groups[str(y)]=g
                for w in (3,4):
                    band=len(rows)%3
                    q,step,a=quartet(y,"borrow",w,rng,seen,band)
                    s,p,b=quartet(y,"safe",w,rng,seen|a,band)
                    seen.update(a|b)
                    rows.append(dict(id=f"{g}-{y}-{w}",split=g,target=y,foil=y+100,width=w,
                                     borrow=q,safe=s,borrow_steps=step,safe_steps=p))
    rng.shuffle(rows)
    save(path,dict(rows=rows,groups=groups,seed=SEED,prompt=PREFIX,
                   earlier_unordered=older,parent_sha256=sha(PARENT)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    if not (a.out/"DECK.json").exists():
        build(a.out/"DECK.json")
    native.run(a.model,a.out)
