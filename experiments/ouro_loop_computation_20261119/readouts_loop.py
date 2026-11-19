"""Freeze early windows and examine structured semantic readouts on held-out worlds."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor,cho_solve
from sklearn.linear_model import LogisticRegression

from data_loop import TASKS,SITES,rows,jsonfile

PROTOCOLS=("open","choice")
SEMANTICS=("S","M","Y")
TEST=("confirm","transfer")
MACROS=("mixing","column_concentration","sink_rate")


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def export(p,data):
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,list(data[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(data)


def ci(x):
    v=np.asarray(x,dtype=float)
    if len(v)==0:
        return (None,None,None)
    bag=np.random.default_rng(20261119).integers(len(v),size=(4000,len(v)),dtype=np.int32)
    lo,hi=np.quantile(v[bag].mean(1),[.025,.975])
    return float(v.mean()),float(lo),float(hi)


def partition(records,keys):
    out=defaultdict(list)
    for r in records:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def top(x):
    v=np.asarray(x)
    z=v.argmax(-1)
    return np.where((v==v.max(-1,keepdims=True)).sum(-1)==1,z,-1)


def model(x,y,penalty=.06):
    x=np.array(x,dtype=np.float32)
    cen=x.mean(0)
    ycen=y.mean(0)
    x-=cen
    scale=max(float(np.sqrt(np.mean(x*x))),1e-6)
    x/=scale
    dual=len(x)<x.shape[1]
    gram=x@x.T if dual else x.T@x
    gram.flat[::len(gram)+1]+=len(x)*penalty
    coef=cho_solve(cho_factor(gram,lower=True),y-ycen if dual else x.T@(y-ycen))
    return cen,scale,ycen,coef,x if dual else None


def apply(mod,v):
    cen,scale,intercept,w,x=mod
    a=(v-cen)/scale
    if x is not None:
        a=a@x.T
    return a@w+intercept


class Data:
    def __init__(self,path,groups):
        self.metadata=rows(path/"rows.jsonl")
        self.at={k:np.load(path/f"{k}.npy",mmap_mode="r") for k in
                 ("trajectory","boundary","geometry","attention","lens")}
        self.groups=groups
        self.protocol=np.array([r["protocol"] for r in self.metadata])
        self.split=np.array([r["split"] for r in self.metadata])
        self.world=np.array([r["world"] for r in self.metadata])

    def choose(self,p,split="fit",r=False):
        return np.flatnonzero((self.protocol==p)&(self.split==split)&(self.world=="r" if r else True))

    def labels(self,ids,key):
        return [self.metadata[i][key] for i in ids]

    def choices(self,i,key):
        g=self.groups[self.metadata[i]["id"]]
        if key=="Y":
            return g["options"]
        if key=="S":
            return g["S_options"]
        return list(dict.fromkeys(g["worlds"][w]["M"] for w in ("r","y")))


class Probe:
    def __init__(self,fit,idx,x,key):
        self.key=key
        self.numeric=fit.metadata[0]["task"]=="arithmetic"
        lab=fit.labels(idx,key)
        self.vocab={w:i for i,w in enumerate(sorted(set(lab)))}
        y=np.asarray(lab,dtype=np.float32)[:,None] if self.numeric else np.eye(len(self.vocab),
                     dtype=np.float32)[[self.vocab[w] for w in lab]]
        self.mod=model(x,y)

    def contrast(self,x,choices,labels):
        raw=apply(self.mod,x)
        if self.numeric:
            answer=-np.abs(raw-np.array(choices,dtype=np.float32))
        else:
            voc=np.pad(raw,((0,0),(0,1)),constant_values=-1000.)
            keys=np.array([[self.vocab.get(y,len(self.vocab)) for y in v] for v in choices])
            answer=np.take_along_axis(voc,keys,axis=1)
        truth=np.array([v.index(y) for v,y in zip(choices,labels)])
        gold=answer[np.arange(len(x)),truth].copy()
        correct=top(answer)==truth
        answer[np.arange(len(x)),truth]=-float("inf")
        return correct,gold-answer.max(1)

    def evaluate(self,data,idx,x):
        return self.contrast(x,[data.choices(i,self.key) for i in idx],data.labels(idx,self.key))


def ngram(data,idx):
    width=512
    out=np.zeros((len(idx),width),dtype=np.float32)
    for j,i in enumerate(idx):
        ids=data.metadata[i]["ids"]
        for order in (1,2,3):
            acc=np.zeros(len(ids)-order+1,np.int64)
            for k in range(order):
                acc=(acc*100003+ids[k:k+len(acc)]+order)&0xffffffff
            np.add.at(out[j],acc%width,1-(acc//width%2)*2)
        out[j]/=max(np.linalg.norm(out[j]),1e-6)
    return out


def freeze(root):
    windows={}
    for task in TASKS:
        data=Data(root/"development"/task,{})
        windows[task]={}
        for p in PROTOCOLS:
            idx=data.choose(p,"discover",True)
            assert len(idx)==48
            prof=data.at["attention"][idx,:,:,0].mean(0)
            windows[task][p]={}
            for k in (2,3,4):
                place=int(np.argmax([prof[k-1,j:j+4].mean() for j in range(9)]))
                windows[task][p][str(k)]=list(range(place,place+4))
    jsonfile(root/"freeze.json",{"windows":windows,"selection":"discovery mean entropy, first half",
             "manifest_sha256":digest(root/"data"/"groups.jsonl"),
             "implementation":{n:digest(Path(__file__).parent/n) for n in
                ("data_loop.py","trace_loop.py","route_loop.py","readouts_loop.py","PROTOCOL.md")}})
    print(windows,flush=True)


def aggregate_accuracy(hit,selection):
    values=np.asarray(hit,dtype=float)
    assert len(values)==len(selection) and len(values)%4==0
    return ci(values.reshape(-1,4).mean(1))


def semantic(task,dev,held,p):
    train=dev.choose(p)
    idx=[held.choose(p,s) for s in TEST]
    probes=[]
    for sem in SEMANTICS:
        ctrl=Probe(dev,train,ngram(dev,train),sem)
        for j,s in enumerate(TEST):
            t=idx[j]
            hit,_=ctrl.evaluate(held,t,ngram(held,t))
            m,lo,hi=aggregate_accuracy(hit,t)
            probes.append(dict(task=task,protocol=p,split=s,semantic=sem,method="surface_ngrams",
                   loop=0,layer=0,role="prompt",accuracy=m,lower=lo,upper=hi,groups=len(t)//4))
        for k in range(4):
            for layer in range(25):
                for r,role in enumerate(SITES):
                    fit=dev.at["trajectory"][train,k,layer,r].astype(np.float32)
                    probe=Probe(dev,train,fit,sem)
                    for j,s in enumerate(TEST):
                        t=idx[j]
                        score=held.at["trajectory"][t,k,layer,r].astype(np.float32)
                        val,_=probe.evaluate(held,t,score)
                        m=float(val.mean())
                        lo=hi=""
                        if layer==24:
                            m,lo,hi=aggregate_accuracy(val,t)
                        probes.append(dict(task=task,protocol=p,split=s,semantic=sem,method="projected",
                              loop=k+1,layer=layer,role=role,accuracy=m,lower=lo,upper=hi,groups=len(t)//4))
            for r,role in enumerate(SITES):
                for method,port in (("full_width",p),("full_width_K1_decoder",p),
                                    ("full_width_other_protocol","choice" if p=="open" else "open")):
                    orig=dev.choose(port)
                    u=0 if method=="full_width_K1_decoder" else k
                    d=Probe(dev,orig,dev.at["boundary"][orig,u,r],sem)
                    for j,s in enumerate(TEST):
                        t=idx[j]
                        score,_=d.evaluate(held,t,held.at["boundary"][t,k,r])
                        m,lo,hi=aggregate_accuracy(score,t)
                        probes.append(dict(task=task,protocol=p,split=s,semantic=sem,method=method,
                              loop=k+1,layer=24,role=role,accuracy=m,lower=lo,upper=hi,groups=len(t)//4))
    return probes


def score_margin(score,gold):
    a=np.asarray(score)
    return a[...,gold]-np.max(np.delete(a,gold,axis=-1),axis=-1)


def behavior(task,data,p,split):
    idx=data.choose(p,split)
    out=[]
    for i in idx:
        r=data.metadata[i]
        scores=np.asarray(r["complete_scores"])
        answer=top(scores)
        gold=r["gold"]
        correct=answer==gold
        first=np.asarray(r["first_answer_matches"])
        margin=score_margin(scores,gold)
        for k in range(4):
            out.append(dict(task=task,protocol=p,split=split,id=r["id"],world=r["world"],
                 loop=k+1,correct=int(correct[k]),tie=int(answer[k]<0),
                 corrected=int(k>0 and correct[k] and not correct[k-1]),
                 regressed=int(k>0 and not correct[k] and correct[k-1]),
                 margin=float(margin[k]),unconstrained_answer_prefix=int(first[k])))
    return out


def explain(task,dev,held,p,lock):
    train=dev.choose(p)
    dest=[("discovery",dev,dev.choose(p,"discover",True))]
    dest.extend((s,held,held.choose(p,s,True)) for s in TEST)
    fits=dev.choose(p,"fit",True)
    output=[]
    def feature(data,ids,k,probes):
        batch=[data.metadata[i] for i in ids]
        layer=lock["windows"][task][p].get(str(k+1),list(range(4)))
        geo=data.at["geometry"][ids,k,24,4]
        mixing=data.at["attention"][ids,k][:,layer,0].mean(1)
        known=np.array([score_margin(r["complete_scores"][k],r["gold"]) for r in batch])
        spread=np.array([np.diff(np.sort(r["complete_scores"][k])[-2:])[0] for r in batch])
        baseline=np.stack((known,spread,np.nan_to_num(geo[:,1]),geo[:,0],mixing),axis=1)
        sem=[d.evaluate(data,ids,data.at["trajectory"][ids,k,24,j].astype(np.float32))[1]
             for d,j in zip(probes,(0,2,4))]
        extra=np.column_stack((baseline,np.stack(sem,axis=1)))
        current=np.array([int(top(r["complete_scores"][k])==r["gold"]) for r in batch])
        y=np.array([int(top(r["complete_scores"][k+1])==r["gold"]) for r in batch])
        return baseline,extra,current,y
    for k in range(3):
        probes=[Probe(dev,train,dev.at["trajectory"][train,k,24,j].astype(np.float32),sem)
                for j,sem in zip((0,2,4),SEMANTICS)]
        x,z,prev,y=feature(dev,fits,k,probes)
        if len(np.unique(y))<2:
            continue
        estimators=[]
        for inp in (x,z):
            m=inp.mean(0)
            sd=inp.std(0).clip(1e-6)
            est=LogisticRegression(C=.3,max_iter=200).fit((inp-m)/sd,y)
            estimators.append((m,sd,est))
        for split,data,ids in dest:
            a,b,c,truth=feature(data,ids,k,probes)
            ps=[est.predict_proba((inp-m)/sd)[:,1] for inp,(m,sd,est) in zip((a,b),estimators)]
            for j,i in enumerate(ids):
                output.append(dict(task=task,protocol=p,split=split,id=data.metadata[i]["id"],loop=k+1,
                           current_correct=int(c[j]),next_correct=int(truth[j]),baseline=float(ps[0][j]),
                           semantic_augmented=float(ps[1][j]),confidence=float(a[j,0]),
                           state_similarity=float(a[j,2]),mixing=float(a[j,4]),
                           intermediate_evidence=float(b[j,-3]),Y_evidence=float(b[j,-1])))
    return output


def macro_profile(data,indices,task,proto,split):
    a=data[indices]
    out=[]
    for k in range(4):
        for l in range(24):
            for j,metric in enumerate(MACROS):
                m,lo,hi=ci(a[:,k,l,j])
                out.append(dict(task=task,protocol=proto,split=split,loop=k+1,layer=l,
                                metric=metric,mean=m,lower=lo,upper=hi,groups=len(a)))
    return out


def run(root):
    groups={g["id"]:g for g in rows(root/"data"/"groups.jsonl")}
    lock=json.loads((root/"freeze.json").read_text())
    result=root/"results"
    result.mkdir(exist_ok=True)
    sem,beh,explain_rows,macro=[],[],[],[]
    public=np.load(root/"public"/"macro.npy",mmap_mode="r")
    macro+=macro_profile(public,np.arange(len(public)),"gsm8k","open","public_macro")
    for task in TASKS:
        d=Data(root/"development"/task,groups)
        h=Data(root/"independent"/task,groups)
        for p in PROTOCOLS:
            sem+=semantic(task,d,h,p)
            for s in ("fit","discover",*TEST):
                data=h if s in TEST else d
                beh+=behavior(task,data,p,s)
                macro+=macro_profile(data.at["attention"],data.choose(p,s,True),task,p,s)
            explain_rows+=explain(task,d,h,p,lock)
        print(task,"semantic and macro complete",flush=True)
    export(result/"semantic.csv",sem)
    export(result/"behavior_worlds.csv",beh)
    export(result/"decision_predictions.csv",explain_rows)
    export(result/"macro.csv",macro)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("stage",choices=("freeze","report"))
    a=p.parse_args()
    (freeze if a.stage=="freeze" else run)(a.root)
