"""Check where a diagnostic decoder distinguishes the designated actual result."""
import argparse
from pathlib import Path

import numpy as np

from capture_composition import ARMS, TASKS, WORLDS
from model_composition import Data, collection
from evaluate_composition import Probe, export, prediction
from summarize_composition import boot, effect, losses, read, summary, table


def accuracy(score,truth):
    return (np.argmax(score,axis=1)==truth)&((score==score.max(1,keepdims=True)).sum(1)==1)


def unique_top(values):
    top=max(values)
    return values.index(top) if values.count(top)==1 else -1


def check(root):
    results=[]
    controls=[]
    comparisons=[]
    behavior=[]
    for task in TASKS:
        old=Data(root/"development"/task)
        independent=Data(root/"independent"/task)
        locked=root/"frozen"/task
        scale=np.load(locked/"role_scaler.npy")
        models=read(locked/"freeze.json")
        name=models["candidate"]
        with np.load(locked/f"{name}.npz") as f:
            params=dict(f)
        keep=old.select("fit")
        rows=[old.meta[i] for i in keep]
        _,train=collection(old,keep,())
        causal=table(root/"evaluation"/task/"causal.csv")
        for split in ("fresh_same","fresh_structure"):
            scored=[r for r in causal if r["split"]==split]
            pairs=[("representation",f"token-{m}-bilinear",f"mean-{m}-bilinear")
                   for m in ("natural","mixed")]
            pairs += [("supervision",f"{r}-mixed-bilinear",f"{r}-natural-bilinear")
                      for r in ("mean","token")]
            for arm in ("pooled","SM","I"):
                for factor,cur,base in pairs:
                    x=losses(scored,arm,"response_mse",cur)
                    y=losses(scored,arm,"response_mse",base)
                    m,lo,hi=boot(x,y)
                    comparisons.append(dict(task=task,split=split,arm=arm,factor=factor,method=cur,
                        baseline=base,groups=len(x),error=x.mean(),baseline_error=y.mean(),
                        relative_reduction=m,lower=lo,upper=hi))
            eligible=[r for r in independent.meta if r["split"]==split and "randSM" in r["scores"]]
            for row in (r for r in independent.meta if r["split"]==split):
                for k in range(4):
                    clean=unique_top(row["scores"]["r"][k])
                    for state,w in (("S","x"),("M","y"),("SM","d")):
                        answer=unique_top(row["scores"][state][k])
                        target=row["gold"][w]
                        valid=clean>=0 and answer>=0
                        behavior.append(dict(task=task,split=split,arm=state,loop=k+1,id=row["id"],
                            flipped_into_target=int(valid and clean!=target and answer==target),
                            lost_target=int(valid and clean==target and answer!=target),
                            clean_tie=int(clean<0),edited_tie=int(answer<0),unique_pair=int(valid)))
            for k in (1,2,3):
                def contrast(arms):
                    return np.array([sum(sign*effect(r["scores"][arm][k],[r["gold"][w] for w in WORLDS])
                                         for sign,arm in zip((1,-1,-1,1),arms)) for r in eligible])
                true=contrast(("r","S","M","SM"))
                patterns={"random":("r","randS","randM","randSM"),
                          "irrelevant":("r","noiseS","M","noiseSM"),
                          "mismatch":("r","S","wrongM","wrongSM")}
                for control,arms in patterns.items():
                    alt=contrast(arms)
                    m,lo,hi=boot(true-alt)
                    controls.append({"task":task,"split":split,"loop":k+1,"reference":control,
                        "groups":len(alt),"mean_patch":true.mean(),"mean_control":alt.mean(),
                        "difference":m,"lower":lo,"upper":hi})
        for kind,slot,key in (("intermediate",2,"intermediate_options"),("final",4,"options")):
            label="mid" if kind=="intermediate" else "gold"
            probe=Probe(task,key,rows,train["y"][:,slot]/scale[slot])
            h1=Probe(task,key,rows,train["full"].reshape(-1,5,2048)[:,slot]/scale[slot])
            for split in ("fresh_same","fresh_structure"):
                ids=independent.select(split)
                m=[independent.meta[i] for i in ids]
                features,dev=collection(independent,ids,("physical",))
                base=independent.raw[ids,ARMS.index("r")]
                candidate=base[:,None]+prediction(params,models["configurations"][name],features,"physical")*scale
                for a,w in enumerate(WORLDS):
                    correct=np.array([r[label][w] for r in m])
                    nat0=h1.scores(dev["full"].reshape(len(m),4,5,2048)[:,a,slot]/scale[slot],m)
                    nat1=probe.scores(independent.raw[ids,ARMS.index(w),slot]/scale[slot],m)
                    i_arm=("r","S","M","SM")[a]
                    state=independent.raw[ids,ARMS.index(i_arm),slot]
                    actual=probe.scores(state/scale[slot],m)
                    pred=base[:,slot] if a==0 else candidate[:,a-1,slot]
                    predicted=probe.scores(pred/scale[slot],m)
                    states={"chance":np.array([1/len(r[key]) for r in m]),
                            "natural_K1":accuracy(nat0,correct),
                            "natural_K2":accuracy(nat1,correct),
                            "physical_K2":accuracy(actual,correct),
                            "predicted_anchored_K2":accuracy(predicted,correct)}
                    for state,v in states.items():
                        results.extend(dict(task=task,split=split,id=r["id"],difficulty=r["difficulty"],
                            arm=i_arm,semantic=kind,state=state,designated_accuracy=float(z))
                            for r,z in zip(m,v))
    out=root/"results"
    export(out/"semantic_accuracy_by_group.csv",results)
    export(out/"semantic_accuracy_summary.csv",summary(results,("task","split","arm","semantic","state"),
                                                      ("designated_accuracy",)))
    export(out/"signed_margin_controls.csv",controls)
    export(out/"factorial_2x2_comparisons.csv",comparisons)
    export(out/"tie_aware_behavior_by_group.csv",behavior)
    export(out/"tie_aware_behavior_summary.csv",summary(behavior,("task","split","arm","loop"),
           ("flipped_into_target","lost_target","clean_tie","edited_tie","unique_pair")))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    check(p.parse_args().root)
