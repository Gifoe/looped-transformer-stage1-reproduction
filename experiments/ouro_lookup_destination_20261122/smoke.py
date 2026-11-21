"""Validate exact debit matching and reproduce the archived fully-live route."""

import sys
from pathlib import Path

import torch

from dest_deck import DOMAINS,PARENT
from dest_stats import ARMS,DIRECTIONS
from dest_test import calibrate,decode,prepare,step
from native import Forward,load,setup
from route_sites import prepare as sites


@torch.inference_mode()
def main():
    m,tok=setup(Path(sys.argv[1]))
    f=Forward(m)
    raw=load(PARENT/"ROUTE_RAW.json")["rows"]
    olds=[next(r for r in raw if r["domain"]==d and r["kind"]=="repair" and
               r["pred"]["rescue"]["route_full"]["number"]==r["target"] and
               r["pred"]["damage"]["route_full"]["number"]==r["foil"]) for d in DOMAINS]
    rows=[sites(tok,r) for r in olds]
    rows=[prepare(r,k) for r,k in zip(rows,calibrate(f,rows),strict=True)]
    winners=decode(f,tok,rows)
    for x,y in zip(olds,winners,strict=True):
        assert all(x["pred"][d]["route_full"]==y[d]["live_full"] for d in DIRECTIONS)
    for r in rows:
        _,a=step(f,[r],[(0,d,k,()) for d in DIRECTIONS for k in ARMS],audit=True)
        assert a["checks"]==192 and all(v<.001 for k,v in a.items() if k!="checks")
    print("PASS archived full routing, equal debit and norm, probability conservation, live AV, identity",flush=True)


if __name__=="__main__":
    main()
