"""Frozen-design supplementary controls for surface, structure, and stable choices."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_org import TaskData, export, group_ci, ridge, performance, ROLES
from manifest_org import TASKS,GOALS,load_lines,NAMES

NAMED_HOLDOUT=set(NAMES[-4:])


def new_values(data):
    ids=np.array([r["id"] for r in data.rows])
    if data.task=="arithmetic":
        y=data.labels("intermediate")
        scene=np.array([any(int(v)%4==0 for v in g["intermediate_options"]) for g in data.groups])
        tested=np.array([int(v)%4==0 for v in y.truth])
        rule="holdout intermediate multiples of four; remove entire fitting groups that contain one"
    else:
        q=[g["meta"]["signature"][0] for g in data.groups]
        named=[v[0] if data.task=="relation" else v for v in q]
        scene=tested=np.array([v in NAMED_HOLDOUT for v in named])
        rule="holdout queried names: " + ",".join(sorted(NAMED_HOLDOUT))
    fit=data.data_mask("fit")&~scene
    y=data.labels("intermediate")
    table=[]
    for j in (0,2,4):
        for k in range(4):
            for l in (11,23,24):
                x=data.feature(k,l,j)
                w=ridge(x[fit],y.y[fit])
                for split in ("confirm","transfer"):
                    test=data.data_mask(split)&tested
                    correct=y.evaluate(w,x,test)
                    values=defaultdict(list)
                    for o,v in zip(ids[test],correct):
                        values[o].append(float(v))
                    mean,lo,hi=group_ci([np.mean(x) for x in values.values()])
                    table.append(dict(task=data.task,rule=rule,role=ROLES[j],K=k+1,layer=l,split=split,
                                 fit_worlds=int(fit.sum()),test_worlds=int(test.sum()),groups=len(values),
                                 accuracy=mean,lower=lo,upper=hi))
    return table


def factorial(data):
    table=[]
    for split in ("confirm","transfer"):
        i=np.array([[v[n] for n in ("r","x","y","d")] for v in data.by_group.values()
                    if data.rows[v["r"]]["split"]==split])
        for k in range(4):
            for l in range(25):
                for j,role in enumerate(ROLES):
                    r,x,y,d=data.feature(k,l,j)[i].transpose(1,0,2)
                    a=(x+d-r-y)/2
                    b=(y+d-r-x)/2
                    nonlinear=(r+d-x-y)/2
                    var=np.stack([(v*v).sum(axis=1) for v in (a,b,nonlinear)],axis=1)
                    totals=var.sum(axis=0)
                    q=totals/max(float(totals.sum()),1e-12)
                    table.append(dict(task=data.task,split=split,role=role,K=k+1,layer=l,groups=len(i),
                                      first_factor_share=float(q[0]),map_factor_share=float(q[1]),
                                      interaction_share=float(q[2]),
                                      total_mean_energy=float(var.sum(1).mean())))
    return table


def stable(data):
    meta=performance(data)
    rows=[]
    for split in sorted(set(x["split"] for x in meta)):
        for stable_choice in (True,False):
            selected=[r["index"] for r in meta if r["split"]==split and r["stable_choice"]==stable_choice]
            if not selected:
                continue
            end=data.raw[selected,3,24,4,3]
            rows.append(dict(task=data.task,split=split,stable_choice=stable_choice,n=len(end),
                         K3_K4_readout_relative_l2_median=float(np.median(end)),
                         significant_state_change_share=float(np.mean(end>.1))))
    return rows


def main(root):
    groups={g["id"]:g for g in load_lines(root/"groups.jsonl")}
    holdout=[]; stability=[]; var=[]
    for task in GOALS:
        v=TaskData(root,task,groups)
        stability.extend(stable(v))
        if task in TASKS:
            holdout.extend(new_values(v))
            var.extend(factorial(v))
    export(root/"lexical_numeric_holdout.csv",holdout)
    export(root/"prediction_stability.csv",stability)
    export(root/"factorial_variance.csv",var)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    args=p.parse_args()
    main(args.study)
