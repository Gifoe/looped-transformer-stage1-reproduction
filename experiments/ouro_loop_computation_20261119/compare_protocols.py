"""Paired option-protocol contrasts and traced single-problem examples."""
import argparse
from pathlib import Path
import json

import numpy as np

from data_loop import TASKS,SITES,rows
from readouts_loop import Data, Probe, SEMANTICS,ci,export,ngram,partition,top
from summary_loop import table


def run(root):
    g={r["id"]:r for r in rows(root/"data"/"groups.jsonl")}
    base=root/"results"
    lens,protocol,case,y_support,scope_outcomes=[],[],[],[],[]
    routes=table(base/"causal_by_group.csv")
    for task in TASKS:
        dev=Data(root/"development"/task,g)
        new=Data(root/"independent"/task,g)
        for p in ("open","choice"):
            tr=dev.choose(p)
            surface=Probe(dev,tr,ngram(dev,tr),"Y")
            for split in ("confirm","transfer"):
                a=new.choose(p,split)
                guessed,_=surface.evaluate(new,a,ngram(new,a))
                for k in range(4):
                    d=Probe(dev,tr,dev.at["boundary"][tr,k,4],"Y")
                    seen,_=d.evaluate(new,a,new.at["boundary"][a,k,4])
                    gain=seen.astype(float)-guessed.astype(float)
                    m,lo,hi=ci(gain.reshape(-1,4).mean(1))
                    y_support.append(dict(task=task,protocol=p,split=split,loop=k+1,groups=len(a)//4,
                        full_width=float(seen.mean()),surface=float(guessed.mean()),
                        mean=m,lower=lo,upper=hi))
        for r in rows(root/"scope"/task/"scores.jsonl"):
            if r["tier"]!="whole":
                continue
            y=g[r["id"]]["worlds"]["r"]["gold"]
            pre,k4=(int(top(r["scores"]["sham"][k])) for k in (1,3))
            status="tie" if min(pre,k4)<0 else (
                {(False,True):"correction",(True,False):"regression",
                 (True,True):"stable_correct",(False,False):"stable_wrong"}[pre==y,k4==y])
            scope_outcomes.append(dict(task=task,split=r["split"],id=r["id"],category=status,
                correct_clean=int(k4==y),**{f"correct_{w}":int(top(r["scores"][w][3])==y)
                                             for w in ("cut","noise","restore")}))
        for split in ("confirm","transfer"):
            a=new.choose("open",split)
            b=new.choose("choice",split)
            assert [new.metadata[i]["id"] for i in a]==[new.metadata[i]["id"] for i in b]
            v0=new.at["boundary"][a]
            v1=new.at["boundary"][b]
            distance=np.sqrt(np.mean((v0-v1)**2,axis=-1))/np.maximum(np.sqrt(np.mean(v0*v0,axis=-1)),1e-7)
            pwin=np.stack([top([new.metadata[i]["complete_scores"] for i in xx])==
                           np.array([new.metadata[i]["gold"] for i in xx])[:,None] for xx in (a,b)])
            for k in range(4):
                outcomes={"choice_minus_open_accuracy":pwin[1,:,k].astype(float)-pwin[0,:,k].astype(float)}
                for r,reg in enumerate(SITES):
                    outcomes[f"paired_protocol_relative_L2_{reg}"]=distance[:,k,r]
                for metric,x in outcomes.items():
                    m,lo,hi=ci(x.reshape(-1,4).mean(1))
                    protocol.append(dict(task=task,split=split,loop=k+1,metric=metric,groups=len(x)//4,
                                         mean=m,lower=lo,upper=hi))
                for phase,idx in (("open",a),("choice",b)):
                    t=np.array([new.metadata[i]["gold"] for i in idx])
                    values=top(new.at["lens"][idx,k])==t[:,None]
                    for l in range(25):
                        m,lo,hi=ci(values[:,l].reshape(-1,4).mean(1))
                        lens.append(dict(task=task,protocol=phase,split=split,loop=k+1,layer=l,
                                         mean=m,lower=lo,upper=hi,groups=len(t)//4))
            for k in (2,3,4):
                for metric in ("source_over_nuisance","same_receiver_rescue","matched_over_same_Y"):
                    selected=[r for r in routes if r["task"]==task and r["split"]==split and r["loop"]==str(k)
                              and r["metric"]==metric and r["horizon"]==str(k)]
                    by=partition(selected,("id",))
                    pair=[[float(r["value"]) for r in sorted(q,key=lambda r:r["protocol"])] for _,q in by.items() if len(q)==2]
                    m,lo,hi=ci([x[0]-x[1] for x in pair])
                    protocol.append(dict(task=task,split=split,loop=k,metric="choice_minus_open_"+metric,
                                         groups=len(pair),mean=m,lower=lo,upper=hi))
        picked=new.choose("open","transfer",True)[0]
        meta=new.metadata[picked]
        tr=dev.choose("open")
        for k in range(4):
            for l in range(25):
                features=[]
                for key,j in zip(SEMANTICS,(0,2,4)):
                    probe=Probe(dev,tr,dev.at["trajectory"][tr,k,l,j].astype(np.float32),key)
                    f=new.at["trajectory"][picked:picked+1,k,l,j].astype(np.float32)
                    valid,margin=probe.evaluate(new,[picked],f)
                    features.extend((int(valid[0]),float(margin[0])))
                case.append(dict(task=task,id=meta["id"],loop=k+1,layer=l,S_hit=features[0],
                    S_margin=features[1],M_hit=features[2],M_margin=features[3],Y_hit=features[4],Y_margin=features[5],
                    native_top=int(top(meta["complete_scores"][k])),gold=meta["gold"],
                    S=meta["S"],M=meta["M"],Y=meta["Y"]))
    export(base/"protocol_comparison.csv",protocol)
    export(base/"candidate_lens.csv",lens)
    export(base/"single_item_trajectories.csv",case)
    export(base/"Y_over_surface.csv",y_support)
    export(base/"scope_outcomes.csv",scope_outcomes)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    run(p.parse_args().root)
