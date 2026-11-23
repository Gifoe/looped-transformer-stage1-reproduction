"""Factor fixed-state horizon changes into task-gradient norm and alignment."""

import argparse
from pathlib import Path

import numpy as np
import torch

import minimal
from archive_audit import dump, read, sha
from control_transport import rows_write
from e2_signed import load_control, partition, run,score,write
from forward_control import scope_indices
from functional_operator import batches,selected_rows
from native_to_rank1 import gradient_utility
from state_horizon import score_arrays


def local_vector(reference, other):
    cosine=torch.nn.functional.cosine_similarity(reference.flatten(),other.flatten(),dim=0)
    return dict(norm=float(reference.norm()),direction=float(cosine),dot=float((reference*other).sum()))


def experiment(archive,model,net,info,work,k):
    rows={i:dict(r,scores={},adjoint={},contrasts={}) for i,r in enumerate(info)}
    for items in work:
        packed=archive.pack(items)
        valid=scope_indices(packed,"all_valid")
        writes=scope_indices(packed,"predictors")
        raw=None
        early_delta=None
        stored={}
        for m in range(4):
            end=k+m
            capture=(k,k+1) if m else (k,)
            with torch.enable_grad():
                term,states,leaf,_=run(model,packed,end,capture=capture,leaf_anchor=k)
                q,_=gradient_utility(model,term,packed,items,info)
                if m:
                    grad,current_receiver=torch.autograd.grad(q.sum(),(leaf,states[k+1]))
                else:
                    grad,=torch.autograd.grad(q.sum(),leaf)
                    current_receiver=None
            if raw is None:
                raw=states[k].detach()
                early_delta=write(net,raw,packed)
            elif not torch.equal(raw,states[k]):
                raise RuntimeError("Receiving state changed under a different future horizon")
            stored[m]=grad.detach().float()[writes]
            with torch.no_grad():
                score(archive,model,packed,items,info,term.detach(),f"native_{m}",rows)
                for d in (.10,1.):
                    edited,_,_,_=run(model,packed,end,edits={k:lambda _:d*early_delta})
                    score(archive,model,packed,items,info,edited,f"dose{d}_{m}",rows)
            if m==1:
                first_receiver_state=states[k+1].detach()
                first_receiver_g=current_receiver.detach().float()
            if m and not torch.equal(first_receiver_state,states[k+1]):
                raise RuntimeError("Common next state changed as the frozen tail was extended")
            for qid,ids in partition(items).items():
                idx=packed[1].new_tensor(ids)
                keep=torch.isin(writes[0],idx)
                d=early_delta[writes][keep]
                value=local_vector(stored[m][keep],d)
                if m:
                    keep_receiver=torch.isin(valid[0],idx)
                    rec=valid[0][keep_receiver],valid[1][keep_receiver]
                    value["common_receiving_grad_cosine"]=local_vector(current_receiver.detach().float()[rec],
                                                                      first_receiver_g[rec])["direction"]
                rows[qid]["adjoint"][str(m)]=value
            del term,states,leaf,q
        for qid,ids in partition(items).items():
            keep=torch.isin(writes[0],packed[1].new_tensor(ids))
            norm=float(early_delta[writes][keep].norm())
            base=rows[qid]["adjoint"]["1"]
            for m in (0,2,3):
                comp=rows[qid]["adjoint"][str(m)]
                scale=.5*norm*(base["norm"]-comp["norm"])*(base["direction"]+comp["direction"])
                orientation=.5*norm*(base["norm"]+comp["norm"])*(base["direction"]-comp["direction"])
                rows[qid]["contrasts"][str(m)]=dict(norm_term=scale,orientation_term=orientation,
                    adjoint_cosine=local_vector(stored[1][keep],stored[m][keep])["direction"],
                    reconstruction=base["dot"]-comp["dot"]-scale-orientation)
    return list(rows.values())


def summarize(rows):
    rng=np.random.default_rng(180118)
    draws=rng.integers(len(rows),size=(5000,len(rows)))
    def stats(a):
        a=np.asarray(a)
        return dict(mean=float(a.mean()),ci95=np.quantile(a[draws].mean(1),[.025,.975]).tolist())
    yields={}
    for m in range(4):
        results={}
        for dose in (.1,1.):
            deltas=[]
            for r in rows:
                native,acted=(score_arrays(r["scores"][arm],r["gold"],r["characters"])
                             for arm in (f"native_{m}",f"dose{dose}_{m}"))
                deltas.append(acted["utility"]-native["utility"])
            results[str(dose)]=stats(deltas)
        keys=rows[0]["adjoint"][str(m)]
        yields[str(m)]=dict(finite=results,adjoint={key:stats([r["adjoint"][str(m)][key] for r in rows]) for key in keys})
    contrasts={}
    for m in (0,2,3):
        keys=rows[0]["contrasts"][str(m)]
        contrasts[str(m)]={key:stats([r["contrasts"][str(m)][key] for r in rows]) for key in keys}
    return dict(horizons=yields,fixed_state_contrasts_vs_one=contrasts)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","previous","model","rank1","data","out","forecast-lock"):
        p.add_argument(f"--{name}",required=True,type=Path)
    p.add_argument("--domain",choices=("arc","csqa"),required=True)
    p.add_argument("--n",type=int,default=64)
    a=p.parse_args()
    lock=read(a.forecast_lock)
    if lock["data_sha256"][a.domain]!=sha(a.data) or sha(a.rank1) not in lock["weights_sha256"]:
        raise ValueError("The frozen cohort or controller differs")
    a.out.mkdir(parents=True,exist_ok=True)
    archive,model,tokenizer,rt=minimal.runtime(a.archive,a.previous,a.model)
    net=load_control(model,a.rank1)
    rows=selected_rows(a.data,a.n,261123381)
    info,work=batches(archive,tokenizer,rows,3)
    for k in (1,2):
        raw=a.out/f"h{k}.jsonl"
        results=experiment(archive,model,net,info,work,k)
        rows_write(raw,results)
        summary=summarize(results)
        dump(raw.with_suffix(".json"),dict(**summary,forecast_sha256=sha(a.forecast_lock),rows_sha256=sha(raw),runtime=rt))
        print(a.domain,k,{m:x["finite"]["1.0"]["mean"] for m,x in summary["horizons"].items()},flush=True)


if __name__=="__main__":
    main()
