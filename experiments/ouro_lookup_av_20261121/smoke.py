"""Match factored joint replay to the previously archived attention replay."""

import sys
from pathlib import Path

import torch

from av_deck import DOMAINS, PARENT
from av_stats import DIRECTIONS
from av_test import audit_case, decode, prepare
from native import Forward, load, setup


@torch.inference_mode()
def main():
    model,tok = setup(Path(sys.argv[1]))
    f = Forward(model)
    raw = load(PARENT/"MID_RAW.json")["rows"]
    archive = [next(r for r in raw if r["domain"]==d and r["pred"]["attention"]["rescue"]["authentic"]["number"]==r["target"]
                    and r["pred"]["attention"]["damage"]["authentic"]["number"]==r["old"]) for d in DOMAINS]
    rows = [prepare(tok,r) for r in archive]
    values = decode(f,tok,rows)
    for old,new in zip(archive,values,strict=True):
        assert all(new[d]["joint"] == old["pred"]["attention"][d]["authentic"] for d in DIRECTIONS)
    for row in rows:
        a = audit_case(f,tok,row)
        assert a["checks"]==72 and all(x<.001 for k,x in a.items() if k!="checks")
    print("PASS archived joint replay, native projection, exact factorization, and direct suffix",flush=True)


if __name__ == "__main__":
    main()
