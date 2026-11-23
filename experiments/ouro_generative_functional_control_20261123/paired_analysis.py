"""Build paired intervals and failure counts from complete frozen answer shards."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import minimal
from task_data import load


METRICS={"correct","strict","loose_correct","valid","completed","constraint","process",
         "f1","content_f1","rouge_l","reference_number_recall","acc","acc_norm"}


def compare(ref,other,seed=261123475):
    if [x["id"] for x in ref]!=[x["id"] for x in other]:
        raise ValueError("Different IDs or order for paired evaluation")
    keys=set(ref[0]["metrics"])&set(other[0]["metrics"])&METRICS
    rng=np.random.default_rng(seed)
    all_out={}
    for key in sorted(keys):
        pairs=[(a["metrics"][key],b["metrics"][key]) for a,b in zip(ref,other,strict=True)
               if a["metrics"].get(key) is not None and b["metrics"].get(key) is not None]
        if not pairs:
            continue
        before,after=np.asarray(pairs,dtype=np.float64).T
        diff=after-before
        n=len(pairs)
        beneficial=int((diff>0).sum())
        harmed=int((diff<0).sum())
        binary=all(isinstance(v,bool) for pair in pairs for v in pair)
        if binary:
            multinomial=rng.multinomial(n,[harmed/n,1.-(beneficial+harmed)/n,beneficial/n],size=20000)
            interval=np.quantile((multinomial[:,2]-multinomial[:,0])/n,[.025,.975]).tolist()
        else:
            selection=rng.integers(n,size=(4000,n))
            interval=np.quantile(diff[selection].mean(-1),[.025,.975]).tolist()
        exact=binomtest(beneficial,beneficial+harmed,.5).pvalue if binary and beneficial+harmed else None
        all_out[key]=dict(n=n,reference=float(before.mean()),controller=float(after.mean()),delta=float(diff.mean()),
                          rescued=beneficial,harmed=harmed,paired_ci95=interval,exact_mcnemar=exact)
    return all_out


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=Path("results"))
    p.add_argument("--phase",choices=("development_rescored","confirmation"),required=True)
    a=p.parse_args()
    folder=a.results/a.phase
    index={}
    for path in sorted(folder.glob("*.json")):
        if path.name.count("__")!=3:
            continue
        dataset,arm,depth,mode=path.stem.split("__")
        raw=path.with_suffix(".jsonl")
        receipt=read(path)
        if minimal.sha(raw)!=receipt["rows_sha256"]:
            raise RuntimeError(f"Prediction receipts differ {path}")
        index.setdefault((dataset,depth,mode),{})[arm]=(raw,receipt)
    data={}
    for (dataset,depth,mode),group in index.items():
        if "native" not in group:
            continue
        base=load(group["native"][0])
        if len(base)!=len({x["id"] for x in base}):
            raise RuntimeError("Duplicate predictions")
        baseline=load(group["rank_only"][0]) if "rank_only" in group else None
        report={}
        for arm,(file,receipt) in group.items():
            own=load(file)
            errors=Counter(row["metrics"].get("failed") for row in own if row["metrics"].get("failed"))
            report[arm]=dict(summary=receipt["metrics"],native_paired=compare(base,own),
                             ranking_baseline_paired=compare(baseline,own) if baseline else None,
                             cost=receipt["cost"],stop_counts=dict(Counter(r.get("stop","candidate") for r in own)),
                             error_counts=dict(errors),rows_sha256=receipt["rows_sha256"])
        data[f"{dataset}__{depth}__{mode}"]=report
    minimal.dump(a.results/f"PAIRED_{a.phase.upper()}.json",data)
    print(a.phase,len(data),"paired conditions",flush=True)


if __name__=="__main__":
    main()
