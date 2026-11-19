"""Fit and freeze low-capacity natural and factorial-response models."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor,cho_solve
from sklearn.utils.extmath import randomized_svd

from capture_composition import ARMS, MAIN, WORLDS, TASKS, digest, load_lines, write_json
from encoders import decode_bf16,summarize

SEED=20261119
PENALTIES=(.01,.1,1.)
REPRESENTATIONS=("mean","token")
SUPERVISION=("natural","mixed")
RESPONSE=("lowrank","additive","bilinear")
ROLE_BLOCK=(2,3,4)


class Data:
    def __init__(self,folder):
        self.folder=folder
        self.meta=load_lines(folder/"groups.jsonl")
        self.raw=np.load(folder/"role_k2.npy",mmap_mode="r")
        self.complete=json.loads((folder/"completion.json").read_text())
        size=self.complete["token_total"]
        self.tokens=np.memmap(folder/"token_k1.f16",dtype=np.float16,mode="r",shape=(size,64))
        self.full=np.memmap(folder/"token_k1.bf16",dtype=np.uint16,mode="r",shape=(size,2048))
        self.atlas=np.load(folder/"atlas.npy",mmap_mode="r")
        assert len(self.meta)==len(self.raw)

    def select(self,split):
        return np.array([r["i"] for r in self.meta if r["split"]==split],dtype=int)

    def row(self,i):
        m=self.meta[i]
        start,n=m["token_offset"],m["tokens"]
        x=self.tokens[start:start+4*n].astype(np.float32).reshape(4,n,64)
        full=decode_bf16(self.full[start:start+4*n]).reshape(4,n,2048)
        return m,x,full

    def y(self,i,key):
        return self.raw[i,ARMS.index(key)]


def collection(data,indices,phase):
    f=defaultdict(list)
    states=defaultdict(list)
    names={"forward":(("r","x","y","x",0),("r","x","y","y",1),("r","x","y","d",2)),
           "reverse":(("x","r","d","r",0),("y","d","r","r",1),("d","y","x","r",2)),
           "physical":(("r","x","y","S",0),("r","x","y","M",1),("r","x","y","SM",2))}
    for i in indices:
        row,z,x=data.row(i)
        src,mp=row["source_patch"],row["map_patch"]
        grid=list(row["roles"].values())
        for j,w in enumerate(WORLDS):
            states["full"].append(np.stack([x[j,sp].mean(0) for sp in grid]).ravel())
            states["y"].append(data.y(i,w))
            for rep in REPRESENTATIONS:
                _,_,abs_state=summarize(z[j],z[j],z[j],np.array(row["ids"][w]),grid,src,mp,rep=="token")
                states[rep].append(abs_state.ravel())
        for pattern in phase:
            for a,b,c,target,k in names[pattern]:
                j,s,m=map(WORLDS.index,(a,b,c))
                assert np.isfinite(data.y(i,target)).all()
                gate=np.array(((1,0),(0,1),(1,1))[k],np.float32)[:,None]
                full=np.stack(((x[s,src]-x[j,src]).mean(0),(x[m,mp]-x[j,mp]).mean(0)))*gate
                f[pattern,"full"].append(full.ravel())
                f[pattern,"y"].append(data.y(i,target)-data.y(i,a))
                for rep in REPRESENTATIONS:
                    d,base,_=summarize(z[j],z[s],z[m],np.array(row["ids"][a]),grid,src,mp,rep=="token")
                    f[pattern,rep].append(np.stack((d[0]*gate[0],d[1]*gate[1],*base)))
    return ({k:np.stack(v).astype(np.float32,copy=False) for k,v in f.items()},
            {k:np.stack(v).astype(np.float32,copy=False) for k,v in states.items()})


def lift(f,kind):
    if kind in ("full","natural"):
        return f
    s,m,src,mp,read=np.moveaxis(f,1,0)
    if kind=="lowrank":
        return np.concatenate((s,m),axis=1)
    s_context=s*(mp+.25*src+.25*read)
    m_context=m*(src+.25*mp+.25*read)
    pieces=(s,m,s_context,m_context)
    if kind=="bilinear":
        xy=(s[:,:16,None]*m[:,None,:16]).reshape(len(s),256)
        pieces=(*pieces,xy)
    return np.concatenate(pieces,axis=1)


def gain_fit(states):
    x=states["y"]
    scale=np.sqrt(np.mean((x-x.mean(0))**2,axis=(0,2))).reshape(5,1)
    assert min(scale)>0
    return scale


def rms(x):
    return np.sqrt(np.mean(x*x,axis=0)).clip(1e-6)


def fit(x,y,penalty,*,centered=False,basis=None):
    x=x.astype(np.float32,copy=False)
    mx=x.mean(0) if centered else np.zeros(x.shape[1],np.float32)
    my=y.mean(0) if centered else np.zeros(y.shape[1],np.float32)
    a=x-mx
    norm=rms(a)
    a/=norm
    b=y-my
    if basis is not None:
        b=b@basis
    dual=a.shape[0]<a.shape[1]
    gram=a@a.T if dual else a.T@a
    gram.flat[::len(gram)+1]+=penalty*len(x)
    beta=cho_solve(cho_factor(gram,lower=True), b if dual else a.T@b)
    return {"center":mx,"output_center":my,"norm":norm,"beta":beta,
            "dual":a if dual else np.empty((0,0),np.float32),
            "basis":basis if basis is not None else np.empty((0,0),np.float32)}


def apply(params,x):
    a=(x-params["center"])/params["norm"]
    if params["dual"].size:
        a=a @ params["dual"].T
    a=a @ params["beta"]
    if params["basis"].size:
        a=a @ params["basis"].T
    return a+params["output_center"]


def error(a,b):
    r=(a-b).reshape(-1,5,2048)[:,ROLE_BLOCK]
    return float(np.mean(r*r))


def response_set(features,rep,domain):
    keys=("forward","physical" if domain=="mixed" else "reverse")
    return (np.concatenate([features[p,rep] for p in keys]),
            np.concatenate([features[p,"y"] for p in keys]))


def train(data,out):
    out.mkdir(parents=True)
    fit_rows=data.select("fit")
    discovery=data.select("discover")
    learn,absolute=collection(data,fit_rows,("forward","reverse","physical"))
    explore,abs_dev=collection(data,discovery,("forward","physical"))
    scale=gain_fit(absolute)
    np.save(out/"role_scaler.npy",scale)
    configurations={}
    candidates=[]
    bases={}
    for domain in SUPERVISION:
        _, target=response_set(learn,"full",domain)
        Y=(target/scale).reshape(len(target),-1)
        bases[domain]=randomized_svd(Y,n_components=32,n_iter=4,random_state=SEED)[2].T.copy()
    for domain in SUPERVISION:
        split="physical" if domain=="mixed" else "forward"
        for rep in (*REPRESENTATIONS,"full"):
            for kind in RESPONSE if rep!="full" else ("full",):
                X,Y=response_set(learn,rep,domain)
                X=lift(X,kind)
                Y=(Y/scale).reshape(len(Y),-1)
                A=lift(explore[split,rep],kind)
                B=(explore[split,"y"]/scale).reshape(len(A),-1)
                name=f"{rep}-{domain}-{kind}"
                best=float("inf")
                for p in PENALTIES:
                    model=fit(X,Y,p,basis=bases[domain] if kind=="lowrank" else None)
                    cost=error(apply(model,A),B)
                    candidates.append({"method":name,"penalty":p,"discovery_mse":cost,
                                       "discovery_targets":split})
                    if cost<best:
                        best=cost
                        configurations[name]={"rep":rep,"training":domain,"kind":kind,
                                              "penalty":p,"loss":cost}
                        np.savez(out/f"{name}.npz",**model)
    for rep in (*REPRESENTATIONS,"full"):
        X=absolute[rep]
        Y=(absolute["y"]/scale).reshape(len(X),-1)
        A=abs_dev[rep]
        B=(abs_dev["y"]/scale).reshape(len(A),-1)
        name=f"natural-{rep}"
        best=float("inf")
        for p in PENALTIES:
            model=fit(X,Y,p,centered=True)
            cost=error(apply(model,A),B)
            candidates.append({"method":name,"penalty":p,"discovery_mse":cost,
                               "discovery_targets":"absolute natural"})
            if cost<best:
                best=cost
                configurations[name]={"rep":rep,"training":"natural","kind":"natural",
                                      "penalty":p,"loss":cost}
                np.savez(out/f"{name}.npz",**model)
    candidate=min((k for k in configurations if k.endswith("mixed-bilinear")),
                  key=lambda k:configurations[k]["loss"])
    config_path=out/"freeze.json"
    write_json(config_path,{"fit_groups":len(fit_rows),"discovery_groups":len(discovery),
               "candidate":candidate,"seed":SEED,"configurations":configurations,
               "all_discovery_models":candidates,"source":data.complete["source_manifest_sha256"],
               "implementation":{n:digest(Path(__file__).parent/n) for n in
                                 ("PROTOCOL.md","capture_composition.py","encoders.py","model_composition.py")},
               "weights":{n:digest(out/f"{n}.npz") for n in configurations},
               "scaler":digest(out/"role_scaler.npy")})
    print("FROZEN",config_path,candidate,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("capture",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    train(Data(a.capture),a.out)
