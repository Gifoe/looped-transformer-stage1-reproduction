"""Draw new target-stratified three-digit borrowing and no-borrowing prompts."""
import argparse
from pathlib import Path
from random import Random

from bidi import unordered
import edge_deck
from native import load, save, sha
import native_deck as shared
from rule import PREFIX

PARENT=Path("/root/ouro-borrow-edge-20261121/results")
SEED=261122971


def history():
    used=edge_deck.history()
    for p in (PARENT,PARENT/"extension"):
        for r in load(p/"DECK.json")["rows"]:
            used.update(unordered(x) for k in ("borrow","safe") for x in r[k])
    return used


def build(path):
    used=history()
    n=len(used)
    rng=Random(SEED)
    rows=[]
    groups={}
    for h in range(2,7):
        x=list(range(55,86))
        rng.shuffle(x)
        for split,part,rep in (("discovery",x[:4],2),("confirmation",x[4:24],3)):
            for u in part:
                y=100*h+u
                groups[str(y)]=split
                for j in range(rep):
                    q,a,k=edge_deck.quartet(y,"borrow",3,rng,used,j%3)
                    s,b,p=edge_deck.quartet(y,"safe",3,rng,used|k,j%3)
                    used.update(k|p)
                    rows.append(dict(id=f"{split}-{y}-{j+1}",split=split,target=y,foil=y+100,width=3,
                                     borrow=q,safe=s,borrow_steps=a,safe_steps=b))
    assert len(rows)==340
    rng.shuffle(rows)
    save(path,dict(rows=rows,groups=groups,seed=SEED,prompt=PREFIX,earlier_unordered=n,
                   prior_main_sha256=sha(PARENT/"DECK.json"),
                   prior_extension_sha256=sha(PARENT/"extension"/"DECK.json")))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    if not (a.out/"DECK.json").exists():
        build(a.out/"DECK.json")
    shared.run(a.model,a.out)
