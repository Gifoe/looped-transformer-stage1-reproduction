"""Publish behavior counts, causal status controls and concurrent phase accounting."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_org import export, group_ci
from manifest_org import TASKS,GOALS,load_lines,write_json


def change(s,r,k=3):
    return s[k][r["counterfactual"]]-s[k][r["original"]]


def pick(x):
    return x.index(max(x)) if x.count(max(x))==1 else -1


def label(r):
    truth=r["original"]
    s=[pick(z)==truth for z in r["native"]]
    if all(s):
        return "always_correct"
    if not any(s):
        return "always_wrong"
    if not s[0] and s[-1]:
        return "corrected"
    if s[0] and not s[-1]:
        return "regressed"
    return "transient"


def causal(study):
    effects=[]
    doses=[]
    for task in TASKS:
        raw=load_lines(study/"causal"/task/"rows.jsonl")
        cohorts=defaultdict(list)
        cohorts["all"]=raw
        for r in raw:
            cohorts[label(r)].append(r)
        for name,current in cohorts.items():
            for arm in ("aligned","full","source_control","map_only"):
                groups=defaultdict(list)
                flips=0
                for r in current:
                    g,o=r["counterfactual"],r["original"]
                    q=r["arms"][arm]["scores"]
                    baseline=r["native"]
                    comp=r["arms"]["random"]["scores"] if arm=="aligned" else baseline
                    groups[r["id"]].append(change(q,r)-change(comp,r))
                    flips+=int(pick(baseline[3])==o and pick(q[3])==g)
                v,lo,hi=group_ci([np.mean(k) for k in groups.values()])
                effects.append(dict(task=task,trajectory=name,arm=arm,groups=len(groups),
                                    directions=len(current),effect=v,lower=lo,upper=hi,strict_flip=flips))
        vals=np.array([r["arms"]["aligned"]["dose"] for r in raw])
        changes=np.array([r["arms"]["aligned"]["mean_axis_delta"] for r in raw])
        for arm in ("random","orthogonal","irrelevant","reverse_aligned"):
            amp=np.array([r["arms"][arm]["dose"] for r in raw])
            good=vals>0
            ratio=np.abs(amp[good]-vals[good])/vals[good]
            doses.append(dict(task=task,arm=arm,n=len(vals),aligned_zero_count=int((~good).sum()),
                    aligned_median_l2=float(np.median(vals)),control_median_l2=float(np.median(amp)),
                    median_abs_relative_error=float(np.median(ratio)) if len(ratio) else None,
                    mean_axis_change_positive=int((changes>0).sum())))
    export(study/"causal_by_outcome.csv",effects)
    export(study/"causal_dose.csv",doses)


def counts(study):
    overview=[]
    for t in GOALS:
        rows=load_lines(study/"natural"/t/"rows.jsonl")
        for split in ["all",*sorted({r["split"] for r in rows})]:
            x=[r for r in rows if split=="all" or r["split"]==split]
            for scope in ("r","factorial"):
                if scope=="factorial" and t not in TASKS:
                    continue
                sample=[r for r in x if (scope=="r" and r["world"]=="r") or
                                         (scope=="factorial" and r["world"]!="n")]
                for k in range(4):
                    overview.append(dict(task=t,split=split,scope=scope,K=k+1,n=len(sample),
                         correct=sum(r["pred"][k]==r["gold"] for r in sample),
                         ties=sum(r["pred"][k]==-1 for r in sample)))
    export(study/"native_accuracy.csv",overview)


def timers(study):
    phases=[]
    for p in study.rglob("complete.json"):
        row=json.loads(p.read_text())
        n=row.get("gpu_phase_seconds",row.get("phase_seconds_including_cpu_fit"))
        if n is None:
            continue
        finish=p.stat().st_mtime
        phases.append(dict(name=str(p.parent.relative_to(study)),phase_wall_seconds=n,
                      start=finish-n,end=finish))
    windows=[]
    for r in sorted(phases,key=lambda p:p["start"]):
        a,b=r["start"],r["end"]
        if not windows or a>windows[-1][1]:
            windows.append([a,b])
        else:
            windows[-1][1]=max(b,windows[-1][1])
    write_json(study/"runtime_accounting.json",{"phase_time_sum":sum(r["phase_wall_seconds"] for r in phases),
             "overlap_adjusted_wall_seconds":sum(b-a for a,b in windows),
             "phases":phases,"clock":"phase receipts mtime minus measured operation duration; width includes CPU fitting"})


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    a=p.parse_args()
    counts(a.study)
    causal(a.study)
    timers(a.study)
