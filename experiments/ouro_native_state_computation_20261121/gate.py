"""Freeze the blinded invitation to upstream search from the 48-item development swaps."""
import argparse
from pathlib import Path

import numpy as np

from native import load, save, sha


def interval(vals, alpha=.05):
    a=np.asarray(vals,dtype=float)
    if not len(a):
        return None
    rng=np.random.default_rng(2111)
    boot=a[rng.integers(len(a),size=(8000,len(a)))].mean(axis=1)
    return dict(n=len(a),mean=float(a.mean()),ci=np.quantile(boot,[alpha/2,1-alpha/2]).tolist(),
                lower_simultaneous=float(np.quantile(boot,.05/14)))


def decide(root):
    fname=root/"CAUSAL_dev.json"
    rows=load(fname)["rows"]
    assert len(rows)==48
    result={}
    for m in ("attention","mlp"):
        contrasts={}
        for tr in ("1from2","2from1"):
            data=[r["tests"][tr][m] for r in rows]
            ctr=["zero","random","other_tokens"]+([] if m=="mlp" else ["other_heads"])
            contrasts[tr]={q:interval([x["factor"]["centered_progress"]-
                      (x[q]["centered_progress"] if q!="zero" else 0.) for x in data]) for q in ctr}
        go=all(q["lower_simultaneous"]>0 for row in contrasts.values() for q in row.values())
        result[m]=dict(search_writes=go,stats=contrasts)
    save(root/"GATE.json",dict(dev_swap_sha256=sha(fname),n=len(rows),
          criterion="Both donor directions positive and above all same-dose controls. Bonferroni .05/14 lower tail.",
          metric="centered vocabulary forward donor-alignment",candidates=result))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    decide(p.parse_args().out)

