"""Predict work-dose defect with a midpoint adjoint on two frozen paths."""

import argparse
from pathlib import Path

import numpy as np
import torch

import minimal
from archive_audit import dump,read,sha
from control_transport import rows_write
from e2_signed import load_control,partition,run,score,write
from functional_operator import batches
from native_to_rank1 import gradient_utility
from state_horizon import score_arrays


def adjoint(model,pack,items,info,receive,end,edit=None):
    edits=None if edit is None else {receive:lambda _:edit}
    with torch.enable_grad():
        terminal,_,leaf,_=run(model,pack,end,leaf_anchor=receive,edits=edits)
        utility,_=gradient_utility(model,terminal,pack,items,info)
        g,=torch.autograd.grad(utility.sum(),leaf)
    return g.detach().float()


def collect(model,archive,net,info,work):
    rows={i:dict(r,scores={},projections={}) for i,r in enumerate(info)}
    start,receive,end=2,3,4
    for items in work:
        pack=archive.pack(items)
        with torch.no_grad():
            ordinary,native,_,_=run(model,pack,end,capture=(start,receive))
            early,sent,_,_=run(model,pack,end,capture=(receive,),
                               edits={start:lambda _:write(net,native[start],pack)})
            x=native[receive].float()
            d=sent[receive].float()-x
            e=write(net,native[receive],pack)
        g0=adjoint(model,pack,items,info,receive,end)
        g1=adjoint(model,pack,items,info,receive,end,d/2)
        g2=adjoint(model,pack,items,info,receive,end,e/2)
        for q,index in partition(items).items():
            vals={"base_adjoint":g0*(d-e),"midpoint_adjoint":g1*d-g2*e,
                  "early_curvature":(g1-g0)*d,"late_curvature":(g2-g0)*e}
            rows[q]["projections"]={k:float(v[index].sum()) for k,v in vals.items()}
        with torch.no_grad():
            late,_,_,_=run(model,pack,end,edits={receive:lambda _:e})
            for name,hidden in (("native",ordinary),("early",early),("late",late)):
                score(archive,model,pack,items,info,hidden,name,rows)
    return list(rows.values())


def evaluate(rows):
    key=lambda row,arm:score_arrays(row["scores"][arm],row["gold"],row["characters"])["utility"]
    finite=np.array([key(r,"early")-key(r,"late") for r in rows])
    naive,midpoint=(np.array([row["projections"][name] for row in rows])
                    for name in ("base_adjoint","midpoint_adjoint"))
    rng=np.random.default_rng(991118)
    draws=rng.integers(len(rows),size=(5000,len(rows)))
    diff=abs(midpoint-finite)-abs(naive-finite)
    return dict(n=len(rows),mean_finite=float(finite.mean()),base_mae=float(abs(naive-finite).mean()),
                midpoint_mae=float(abs(midpoint-finite).mean()),mae_difference=float(diff.mean()),
                mae_difference_ci95=np.quantile(diff[draws].mean(1),[.025,.975]).tolist(),
                base_sign=float((np.sign(naive)==np.sign(finite)).mean()),
                midpoint_sign=float((np.sign(midpoint)==np.sign(finite)).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("archive","previous","model","rank1","data","lock","out"):
        p.add_argument(f"--{key}",required=True,type=Path)
    p.add_argument("--tag",choices=("arc","csqa","qasc"),required=True)
    args=p.parse_args()
    lock=read(args.lock)
    limit=lock["phases"][args.tag]
    assert limit["data_sha256"]==sha(args.data) and limit["weights_sha256"]==sha(args.rank1)
    archive,model,tok,rt=minimal.runtime(args.archive,args.previous,args.model)
    net=load_control(model,args.rank1)
    data=minimal.load_jsonl(args.data)
    info,work=batches(archive,tok,data,3)
    rows=collect(model,archive,net,info,work)
    args.out.mkdir(parents=True,exist_ok=True)
    raw=args.out/"PREDICTIONS.jsonl"
    rows_write(raw,rows)
    report=dict(**evaluate(rows),lock_sha256=sha(args.lock),runtime=rt,rows_sha256=sha(raw))
    dump(args.out/"SUMMARY.json",report)
    print(args.tag,report,flush=True)


if __name__=="__main__":
    main()
