"""Train fixed diagnostic readers on native development groups only."""
from collections import defaultdict
import json

import numpy as np
from scipy.linalg import solve
from scipy.special import softmax

from study import FACTORS, dump, write_row


def label(w, role):
    return w["y_index" if role=="y" else role]


def fit_and_measure(bank, out):
    readers = {}
    packed={}
    for family in ("tableau","twohop"):
        roles=[r for r in FACTORS if family=="twohop" or r!="b"]
        chosen=[g for g in bank if g[0][0]["family"]==family and g[0][0]["cohort"]=="development"]
        x=np.concatenate([g[1]["native"][1][::2].reshape(-1,2048) for g in chosen]).astype("float32")
        # Native t = 0, .25, .5, .75, 1. All groups keep together.
        y=np.concatenate([np.tile(np.array([[label(w,r) for r in roles] for w in g[0]]),(5,1))
                          for g in chosen])
        y=np.eye(3,dtype="float32")[y].reshape(len(y),-1)
        mu=x.mean(0)
        sigma=x.std(0).clip(.1)
        x=(x-mu)/sigma
        offset=y.mean(0)
        gram=x.T@x
        gram.flat[::gram.shape[0]+1]+=250
        beta=solve(gram,x.T@(y-offset),assume_a="pos")
        readers[family]=(roles,mu,sigma,beta,offset)
        packed.update({f"{family}_{k}":v for k,v in zip(("roles","mu","scale","weight","offset"),readers[family])})
    np.savez_compressed(out/"native_readers.npz",**packed)
    with (out/"semantic_predictions.jsonl").open("w",encoding="utf-8") as f:
        for group, tracks in bank:
            family=group[0]["family"]
            roles,mu,sigma,beta,offset=readers[family]
            for name,(times,states) in tracks.items():
                vals=((states.astype("float32")-mu)/sigma)@beta+offset
                vals=vals.reshape(len(times),len(group),len(roles),3)
                conf=softmax(vals,axis=-1)
                for i,w in enumerate(group):
                    true=[label(w,r) for r in roles]
                    labels=vals[:,i].argmax(-1)
                    correct=labels==true
                    # Sustained means the current and next available checkpoint.
                    onset={r:next((t for j,t in enumerate(times[:-1])
                                   if correct[j,ri] and correct[j+1,ri]),None)
                           for ri,r in enumerate(roles)}
                    write_row(f,dict(item=w["item"], group=w["group"],cohort=w["cohort"],family=family,
                       track=name,time=times,roles=roles,true=true,labels=labels.tolist(),
                       true_confidence=conf[:,i,np.arange(len(roles)),true].tolist(),onset=onset))
    dump(out/"semantic_design.json",dict(training="independent development groups, native t = 0,.25,.5,.75,1",
           checkpoint_scope="Common 1/8 grid, or shared subset for coarse Euler. Native hidden width 2048.",
           vector="Final Answer: token state at the same actual model position",
           objective="Ridge (lambda=250), native-trained frozen readouts, three-way factorial labels",
           limitations=["A single role/token is tested", "Source and rule are factors; in old 3x3 the bridge is not separable from source",
                        "The endpoint of artificial trajectories is not the native K2 except at m=1"]))
