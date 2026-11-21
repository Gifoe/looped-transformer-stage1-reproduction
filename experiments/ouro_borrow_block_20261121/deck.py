"""Build a new target-disjoint quartet and reuse the pinned native decoder."""
import argparse
from pathlib import Path
import random

import native_deck as shared
from bidi import unordered
from native import load,save,sha
from rule import PREFIX

PREVIOUS=Path("/root/ouro-native-borrow-interaction-20261121/results/DECK.json")


def history():
    seen=shared.history()
    for g in load(PREVIOUS)["rows"]:
        seen.update(unordered(p) for p in [*g["borrow"],*g["safe"]])
    return seen


def build(path):
    seen=history()
    orig=len(seen)
    rng=random.Random(261122256)
    out=[]
    groups={}
    for h in range(2,7):
        pool=[100*h+r for r in range(53,87)]
        rng.shuffle(pool)
        for split,seq in (("discovery",pool[:10]),("confirmation",pool[10:26])):
            for y in seq:
                groups[str(y)]=split
                for j in range(2):
                    band=len(out)%3
                    borrow,d,a=shared.rand_borrow(y,rng,seen,split,band)
                    safe,step,b=shared.rand_safe(y,rng,seen|a,split)
                    seen.update(a|b)
                    out.append(dict(id=f"{split}-{y}-{j+1}",split=split,target=y,foil=y+100,
                                    borrow=borrow,safe=safe,borrow_steps=[d,100-d],safe_steps=step,band=band))
    rng.shuffle(out)
    save(path,dict(rows=out,groups=groups,earlier_unordered=orig,previous_sha256=sha(PREVIOUS),
                   prompt=PREFIX,seed=261122256))


def main(model,root):
    dest=root/"DECK.json"
    if not dest.exists():
        build(dest)
    shared.run(model,root)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    main(a.model,a.out)

