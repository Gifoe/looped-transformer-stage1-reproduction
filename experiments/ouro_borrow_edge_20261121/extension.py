"""Prepare disjoint target clusters using the primary frozen head."""
from pathlib import Path
from random import Random

from bidi import unordered
from edge_deck import history, quartet
from native import load, save, sha
from rule import PREFIX


def prepare(root):
    output=root/"extension"
    output.mkdir(exist_ok=True)
    base=load(root/"DECK.json")
    if (output/"DECK.json").exists():
        return
    older=history()
    for r in base["rows"]:
        older.update(unordered(x) for k in ("borrow","safe") for x in r[k])
    previous=len(older)
    rng=Random(261122902)
    dev=[r for r in base["rows"] if r["split"]=="discovery"]
    groups={str(r["target"]):"discovery" for r in dev}
    fresh=[]
    for h in range(2,7):
        remaining=[100*h+x for x in range(52,90) if str(100*h+x) not in base["groups"]]
        assert len(remaining)==10
        for y in remaining:
            groups[str(y)]="confirmation"
            for j in range(2):
                band=(len(fresh)+1)%3
                q,a,k=quartet(y,"borrow",3,rng,older,band)
                s,b,u=quartet(y,"safe",3,rng,older|k,band)
                older.update(k|u)
                fresh.append(dict(id=f"extension-{y}-{j+1}",split="confirmation",target=y,foil=y+100,
                                  width=3,borrow=q,safe=s,borrow_steps=a,safe_steps=b))
    assert len(fresh)==100
    rng.shuffle(fresh)
    new=output/"DECK.json"
    save(new,dict(rows=dev+fresh,groups=groups,seed=261122902,prompt=PREFIX,
                  earlier_unordered=previous,primary_deck_sha256=sha(root/"DECK.json")))
    pred=load(root/"NATIVE.json")["rows"]
    assert [r["id"] for r in pred if r["split"]=="discovery"]==[r["id"] for r in dev]
    save(output/"NATIVE.json",dict(rows=[r for r in pred if r["split"]=="discovery"],deck_sha256=sha(new)))
    lock=load(root/"HEAD_LOCK.json")
    assert lock["deck_sha256"]==sha(root/"DECK.json")
    lock.update(deck_sha256=sha(new),primary_lock_sha256=sha(root/"HEAD_LOCK.json"))
    save(output/"HEAD_LOCK.json",lock)


if __name__=="__main__":
    prepare(Path("results"))
