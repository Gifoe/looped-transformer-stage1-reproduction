"""Generate fresh arithmetic stems for a frozen cross-loop source test."""
import argparse
from pathlib import Path
from random import Random

import deck as previous
import native_deck as shared
from bidi import unordered
from borrow import needs
from native import load, save, sha
from rule import PREFIX

PARENT = Path("/root/ouro-borrow-block-20261121/results/DECK.json")
SEED = 261122492


def history():
    seen = previous.history()
    earlier_targets = set()
    for path in (PARENT, previous.PREVIOUS):
        for r in load(path)["rows"]:
            seen.update(unordered(x) for key in ("borrow", "safe") for x in r[key])
            earlier_targets.add(r["target"])
    return seen, earlier_targets


def safe_quartet(target, rng, seen):
    hi, low = divmod(target, 100)
    for offset in (0,2000,4000):
        for _ in range(2000):
            a_high = rng.randint(1,9-hi)
            a_low = rng.randint(1,98-low)
            a_step = rng.randint(1,min(18,a_low))
            b_step = rng.randint(1,min(18,99-low-a_low))
            a1,b1 = offset+100*a_high+a_low,offset+100*(a_high+hi)+a_low+low
            corners = shared.four(a1-a_step,a1,b1+b_step,b1)
            keys = {unordered(x) for x in corners}
            if keys.isdisjoint(seen):
                assert len(keys)==4 and not any(needs(*p) for p in corners) and b1-a1==target
                return corners,[a_step,b_step],keys
    raise ValueError(target)


def build(path):
    seen, earlier_targets = history()
    earlier_count = len(seen)
    rng = Random(SEED)
    pools = {}
    for h in range(1, 8):
        pool = [100*h+d for d in range(52, 93) if 100*h+d not in earlier_targets]
        rng.shuffle(pool)
        pools[h] = pool
    pilot = [y for h in pools for y in pools[h][:3]]
    main = [y for h in pools for y in pools[h][3:(31 if h in (1,7) else 8)]]
    assert len(pilot) == 21 and len(main) >= 78
    rows, groups = [], {}
    for split, targets in (("discovery", pilot), ("confirmation", main)):
        for y in targets:
            groups[str(y)] = split
            for j in range(3):
                band = len(rows) % 3
                borrow,d,a = shared.rand_borrow(y,rng,seen,"confirmation",band)
                safe,steps,b = safe_quartet(y,rng,seen|a)
                seen.update(a|b)
                rows.append(dict(id=f"{split}-{y}-{j+1}",split=split,target=y,foil=y+100,
                                 borrow=borrow,safe=safe,borrow_steps=[d,100-d],safe_steps=steps,band=band))
    rng.shuffle(rows)
    save(path,dict(rows=rows,groups=groups,earlier_unordered=earlier_count,
                   parent_sha256=sha(PARENT),previous_sha256=sha(previous.PREVIOUS),prompt=PREFIX,seed=SEED))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    dest = a.out/"DECK.json"
    if not dest.exists():
        build(dest)
    shared.run(a.model,a.out)
