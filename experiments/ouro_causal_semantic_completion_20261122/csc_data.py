"""Independent full-integer arithmetic, crossing a carry boundary in one of four worlds."""

import argparse
from collections import Counter
from pathlib import Path
from random import Random

import torch

import native_deck
from deck import history as block_history
from native import Forward,load,save,setup,sha
from rule import DEMONSTRATIONS,greedy,prompt as original

SEED=261134293
SPLITS=("train","dev","test","shift")
SIZES={"borrow":(64,16,32,24),"carry":(24,10,16,16)}
ARITHMETIC_HISTORY=("borrow-edge","borrow-origin","borrow-paths","borrow-transport","borrow-query","borrow-block",
                    "value-compose","rule-switch","native-borrow-interaction")


def operation(form,a,b):
    return b-a if form=="borrow" else a+b


def need(form,a,b):
    return a%100>b%100 if form=="borrow" else a%100+b%100>=100


def foil(form,target):
    return target+(100 if form=="borrow" else -100)


def shifted(form,a,b):
    src="Compute the requested integer.\n"+"".join(
        f"Operation: {kind}; Left: {x}; Right: {y}; Result: {operation(k,x,y)}\n"
        for x,y in DEMONSTRATIONS for k,kind in (("carry","left plus right"),("borrow","right minus left")))
    kind="right minus left" if form=="borrow" else "left plus right"
    return src+f"Operation: {kind}; Left: {a}; Right: {b}; Result:"


def prompt(row,pair):
    a,b=pair
    return shifted(row["form"],a,b) if row["split"]=="shift" else original(a,b,int(row["form"]=="borrow"))


def pair_keys(quad):
    return {tuple(sorted(p)) for p in quad}


def historic_pairs(x):
    if isinstance(x,dict):
        for p in x.values():
            yield from historic_pairs(p)
    elif isinstance(x,list):
        if len(x)==2 and all(type(v) is int and 100<=v<=999 for v in x):
            yield tuple(sorted(x))
        else:
            for p in x:
                yield from historic_pairs(p)


def historical(root):
    seen=block_history()
    sources={}
    for name in ARITHMETIC_HISTORY:
        p=root/f"ouro-{name}-20261121"/"results/DECK.json"
        if p.exists():
            seen.update(historic_pairs(load(p)["rows"]))
            sources[name]=sha(p)
    return seen,sources


def carry(y,rng,seen,safe):
    h,r=divmod(y,100)
    for _ in range(6000):
        high=h if safe else h-1
        ha=rng.randint(1,high-1)
        hb=high-ha
        if safe:
            a=rng.randint(0,r)
            b=r-a
            a0=rng.randrange(55)
            b0=rng.randrange(100-max(a0,a))
            if a0==a or b0==b:
                continue
            quad=native_deck.four(100*ha+a0,100*ha+a,100*hb+b0,100*hb+b)
        else:
            a=rng.randint(0,r)
            b=r-a
            da=rng.randint(31,69)
            db=100-da
            if max(a+da,b+db)>=100:
                continue
            quad=native_deck.four(100*ha+a,100*ha+a+da,100*hb+b,100*hb+b+db)
        keys=pair_keys(quad)
        if len(keys)==4 and not seen.intersection(keys):
            return quad,keys
    raise RuntimeError((y,safe))


def safe_subtract(y,rng,seen):
    h,x=divmod(y,100)
    for _ in range(8000):
        ah=rng.randint(1,9-h)
        a1=rng.randint(0,99-x)
        b1=a1+x
        aa=[a for a in range(max(0,a1-25),min(b1,a1+25)+1) if a!=a1]
        if not aa:
            continue
        a0=rng.choice(aa)
        bb=[b for b in range(max(a1,a0,b1-25),min(99,b1+25)+1) if b!=b1]
        if not bb:
            continue
        b0=rng.choice(bb)
        quad=native_deck.four(100*ah+a0,100*ah+a1,100*(ah+h)+b0,100*(ah+h)+b1)
        keys=pair_keys(quad)
        if len(keys)==4 and not seen.intersection(keys):
            return quad,keys
    raise RuntimeError((y,"safe subtraction"))


def build(history_root):
    rng=Random(SEED)
    seen,sources=historical(history_root)
    baseline=len(seen)
    rows=[]
    for form in SIZES:
        pool=[100*h+z for h in range(2,7) for z in range(53,87)] if form=="borrow" else [
            100*h+z for h in range(3,9) for z in range(9,28)]
        rng.shuffle(pool)
        off=0
        for split,n in zip(SPLITS,SIZES[form],strict=True):
            for target in pool[off:off+n]:
                for j in range(2):
                    if form=="borrow":
                        gen="discovery" if split=="train" else "confirmation"
                        quad,_,a=native_deck.rand_borrow(target,rng,seen,gen,len(rows)%3)
                        safe,b=safe_subtract(target,rng,seen|a)
                    else:
                        quad,a=carry(target,rng,seen,False)
                        safe,b=carry(target,rng,seen|a,True)
                    assert not seen.intersection(a|b)
                    assert [need(form,*x) for x in quad]==[False,False,False,True]
                    assert not any(need(form,*x) for x in safe)
                    assert operation(form,*quad[3])==operation(form,*safe[3])==target
                    assert operation(form,*quad[0])==foil(form,target)
                    seen.update(a|b)
                    rows.append(dict(id=f"{form}-{split}-{target}-{j}",form=form,split=split,
                        target=target,foil=foil(form,target),corners=quad,safe=safe))
            off+=n
    rng.shuffle(rows)
    rows.sort(key=lambda r:SPLITS.index(r["split"]))
    return dict(seed=SEED,history_count=baseline,history_hashes=sources,rows=rows)


def repair(row):
    return [x[3]["number"] for x in row["native"]]==[row["foil"],row["target"]]


def scores(rows):
    out={}
    for form in SIZES:
        for split in SPLITS:
            q=[r for r in rows if (r["form"],r["split"])==(form,split)]
            group={"n":len(q),"repair":sum(map(repair,q))}
            for idx in (3,4):
                group["joint" if idx==3 else "safe"]={str(k+1):sum(a["native"][k][idx]["number"]==a["target"] for a in q) for k in (0,1)}
            out[f"{form}/{split}"]=group
    return out


@torch.inference_mode()
def run(model,root):
    deck=root/"DECK.json"
    if not deck.exists():
        save(deck,build(Path("/root")))
    source=load(deck)["rows"]
    target=root/"NATIVE.json"
    raw=load(target) if target.exists() else dict(deck_sha256=sha(deck),rows=[])
    rows=raw["rows"]
    assert raw["deck_sha256"]==sha(deck) and [r["id"] for r in rows]==[r["id"] for r in source[:len(rows)]]
    m,tok=setup(model)
    f=Forward(m)
    for i in range(len(rows),len(source),8):
        batch=source[i:i+8]
        ids=[tok.encode(prompt(r,p),add_special_tokens=False) for r in batch for p in (*r["corners"],r["safe"][3])]
        g=[greedy(f,tok,ids,k) for k in (1,2)]
        rows.extend(dict(r,native=[x[5*j:5*j+5] for x in g]) for j,r in enumerate(batch))
        save(target,raw)
        print("CSC native",len(rows),len(source),flush=True)
    result=scores(rows)
    save(root/"NATIVE_SUMMARY.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("result",type=Path)
    args=p.parse_args()
    run(args.model,args.result)
