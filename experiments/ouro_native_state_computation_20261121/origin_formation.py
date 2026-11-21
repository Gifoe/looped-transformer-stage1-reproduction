"""Follow-up on formation of natural factors after the bidirectional behavior gate failed."""
import argparse
from pathlib import Path

import torch

from native import Forward, load, save, setup, sha
from origin import lock, replay, screen


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("result",type=Path)
    p.add_argument("blind",type=Path)
    p.add_argument("protocol",type=Path)
    a=p.parse_args()
    where=a.result/"formation"
    where.mkdir(parents=True,exist_ok=True)
    gate=a.result/"GATE.json"
    assert not any(x["search_writes"] for x in load(gate)["candidates"].values())
    candidates=a.result/"CANDIDATES.json"
    c=load(candidates)
    d=load(a.blind)
    model,tok=setup(a.model)
    f=Forward(model)
    screenf=where/"ORIGIN_SCREEN.json"
    if not screenf.exists():
        rows=screen(f,tok,d["dev"][:16],c,("attention","mlp"))
        save(screenf,dict(rows=rows,candidates_sha256=sha(candidates),gate_sha256=sha(gate)))
    locked=where/"ORIGIN_LOCK.json"
    if not locked.exists():
        lock(where,load(screenf)["rows"])
    lc=load(locked)
    cohort={"confirm_arc":d["confirm_arc"][:48],"confirm_external":d["confirm_external"][:48],
            "controlled":[r for lo in (0,24) for r in d["controlled"][lo:lo+12]]}
    pred=where/"ORIGIN_PREDICTIONS.json"
    if not pred.exists():
        save(pred,dict(code={k:sha(Path(__file__).resolve().parent/k) for k in ("origin.py","origin_formation.py","native.py")},
                  selected_sha256=sha(locked),protocol_sha256=sha(a.protocol),gate_sha256=sha(gate),
                  candidate_sha256=sha(candidates), expected="drop reduces natural factor progression, restoration of corresponding later factor attenuates a component of readout loss",
                  populations={k:[r["id"] for r in v] for k,v in cohort.items()}))
    assert load(pred)["selected_sha256"]==sha(locked)
    for phase,loop in (("confirm_arc",0),("confirm_external",0),("controlled",0),("confirm_external",1)):
        dest=where/f"ORIGIN_{phase}{'_later' if loop else ''}.json"
        done=load(dest)["rows"] if dest.exists() else []
        total=cohort[phase]
        for row in total[len(done):]:
            done.append(replay(f,tok,row,c,lc,loop))
            if len(done)%4==0 or len(done)==len(total):
                save(dest,dict(rows=done,expected=len(total),prediction_sha256=sha(pred),
                         peak_cuda_bytes=torch.cuda.max_memory_allocated()))
                print(phase,f"K{loop+1}toK{loop+2}",len(done),len(total),flush=True)


if __name__=="__main__":
    main()

