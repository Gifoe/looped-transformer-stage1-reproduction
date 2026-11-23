"""Test whether content-independent output shifts account for shared control gains."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

import numpy as np
import torch

import minimal
from archive_audit import dump, read, sha
from control_transport import calibrate, grouped_sum, rows_write, seq_rms
from e2_signed import load_control, run, score, write
from state_horizon import score_arrays


def pair_contexts(rows, tokenizer):
    bins=defaultdict(list)
    for row in rows:
        size=len(tokenizer.encode(f"Question: {row['question']}\nAnswer:",add_special_tokens=False))
        bins[size].append(row)
    original, alternate, ids=[],[],[]
    for size, candidates in sorted(bins.items()):
        if len(candidates)<2:
            continue
        order=sorted(candidates,key=lambda row:hashlib.sha256(f"donor-swap/{row['id']}".encode()).digest())
        for i,target in enumerate(order):
            donor=order[(i+1)%len(order)]
            original.append(target)
            alternate.append(dict(target,question=donor["question"]))
            ids.append(dict(source_id=target["id"],donor_id=donor["id"],exact_prefix_tokens=size))
    return original,alternate,ids


@torch.no_grad()
def collect(archive,model,tok,net,rows):
    original,alternate,ids=pair_contexts(rows,tok)
    if len(ids)<48:
        raise ValueError("Too few disjoint questions with identical prefix token counts")
    output=[]
    for start in range(0,len(ids),6):
        first,info=archive.encode(tok,original[start:start+6])
        other,_=archive.encode(tok,alternate[start:start+6])
        p,b=archive.pack(first),archive.pack(other)
        if not all(torch.equal(x,y) for x,y in zip(p[1:],b[1:],strict=True)):
            raise RuntimeError("Context exchange changed exact candidate target tokens or positional support")
        groups,at=p[1:3]
        counts=torch.bincount(groups,minlength=len(first))
        states={}
        distilled={i:dict(r,scores={},context=ids[start+i],geometry={}) for i,r in enumerate(info)}
        for m in (1,3):
            end=1+m
            native_true,true,_,_=run(model,p,end,capture=(1,))
            native_fake,fake,_,_=run(model,b,end,capture=(1,))
            bo=true[1]
            cf=fake[1]
            if not states:
                states["true"],states["donor"]=bo,cf
            if not (torch.equal(states["true"],bo) and torch.equal(states["donor"],cf)):
                raise RuntimeError("The common context writes changed as the horizon changed")
            same=write(net,bo,p)
            d=same[groups,at]
            target=seq_rms(d,groups,counts)
            stamp=cf[groups,at].float()
            swapped,_,supported=calibrate(stamp,d,groups,counts,cf.dtype,target)
            fake_delta=torch.zeros_like(cf,dtype=swapped.dtype).index_put((groups,at),swapped)
            hidden_true,_,_,_=run(model,p,end,edits={1:lambda _:same})
            hidden_fake,fake_intervened,_,_=run(model,b,end,capture=(1,),edits={1:lambda _:fake_delta})
            measured=fake_intervened[1][groups,at].float()-stamp
            error=(seq_rms(measured,groups,counts)-target).abs()/target.clamp_min(1e-10)
            for key,hidden in (("true_native",native_true),("true_rank1",hidden_true),
                               ("swapped_native",native_fake),("swapped_rank1",hidden_fake)):
                score(archive,model,p,first,info,hidden,f"{key}_{m}",distilled)
            for i,(_,_,q,choice) in enumerate(first):
                d_row=distilled[q]["geometry"].setdefault(str(m),[None]*len(info[q]["labels"]))
                d_row[choice]=dict(dose_error=float(error[i]),supported=bool(supported[i]))
        output.extend(distilled.values())
    return output


def summary(rows):
    rng=np.random.default_rng(111899)
    draws=rng.integers(len(rows),size=(5000,len(rows)))
    def stats(y):
        y=np.asarray(y)
        return dict(mean=float(y.mean()),ci95=np.quantile(y[draws].mean(1),[.025,.975]).tolist())
    out={}
    for m in (1,3):
        true=[]
        donated=[]
        direct_null=[]
        for row in rows:
            a,b,c,d=(np.array(row["scores"][f"{key}_{m}"]) for key in
                     ("true_native","true_rank1","swapped_native","swapped_rank1"))
            measure=lambda x:score_arrays(x,row["gold"],row["characters"])["utility"]
            true.append(measure(b)-measure(a))
            donated.append(measure(a+d-c)-measure(a))
            direct_null.append(measure(d)-measure(c))
        a=np.array(true); b=np.array(donated)
        out[str(m)]=dict(true=stats(a),donor_shift_replayed_on_true_native=stats(b),
                         swapped_label_artificial=stats(direct_null),
                         task_context_residual=stats(a-b),half_gain_contrast=stats(.5*a-b),
                         maximum_relative_matched_error=max(x["dose_error"] for r in rows for x in r["geometry"][str(m)]),
                         n_all_choices_matched=sum(all(x["supported"] and x["dose_error"]<.01 for x in r["geometry"][str(m)]) for r in rows))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","previous","model","rank1","data","forecast-lock","out"):
        p.add_argument(f"--{name}",required=True,type=Path)
    p.add_argument("--domain",choices=("arc","csqa"),required=True)
    a=p.parse_args()
    lock=read(a.forecast_lock)
    if lock["data_sha256"][a.domain]!=sha(a.data) or sha(a.rank1) not in lock["weights_sha256"]:
        raise ValueError("Cohort or checkpoint not sealed")
    a.out.mkdir(parents=True,exist_ok=True)
    archive,model,tok,rt=minimal.runtime(a.archive,a.previous,a.model)
    net=load_control(model,a.rank1)
    rows=minimal.load_jsonl(a.data)
    predictions=collect(archive,model,tok,net,rows)
    raw=a.out/"PREDICTIONS.jsonl"
    rows_write(raw,predictions)
    report=dict(n_donor_pool=len(rows),n_matched_prefix=len(predictions),results=summary(predictions),
                runtime=rt,forecast_sha256=sha(a.forecast_lock),rows_sha256=sha(raw))
    dump(a.out/"SUMMARY.json",report)
    print(a.domain,report["n_matched_prefix"],report["results"],flush=True)


if __name__=="__main__":
    main()
