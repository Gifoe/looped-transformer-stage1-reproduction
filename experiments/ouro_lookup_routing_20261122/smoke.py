"""Exercise all live edits on archived examples from both layouts."""

import sys
from pathlib import Path

import torch

from native import Forward,load,setup
from route_deck import DOMAINS,PARENT
from route_sites import prepare
from route_stats import ARMS,DIRECTIONS
from route_test import decode,step


@torch.inference_mode()
def main():
    m,tok=setup(Path(sys.argv[1]))
    f=Forward(m)
    old=load(PARENT/"PAIRS.json")["rows"]
    rows=[prepare(tok,dict(next(q for q in old if q["domain"]==d and q["kind"]=="repair"),
                           foil=next(q for q in old if q["domain"]==d and q["kind"]=="repair")["old"])) for d in DOMAINS]
    a,dose,_=decode(f,tok,rows)
    assert len(a)==len(DOMAINS) and all(sum(l[0] for l in rows)>0 for rows in dose.values())
    for r in rows:
        _,_,_,audit=step(f,[r],[(0,d,k,()) for d in DIRECTIONS for k in ARMS],audit=True)
        assert audit["checks"]==432 and all(audit[k]<.001 for k in ("mass","write","attention"))
    print("PASS real two-row attention conservation, matched controls, V isolation, live AV, clean answers",flush=True)


if __name__=="__main__":
    main()
