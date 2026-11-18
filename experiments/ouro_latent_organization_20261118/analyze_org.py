"""Group-blocked semantic contrasts and diagnostic readouts on complete prompts."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from manifest_org import TASKS, GOALS, digest, load_lines, write_json
from capture_org import ROLES

ENTRY=("fit","discover","confirm","transfer")
TAPS=(5,11,17,23,24)
LATE_ROLES=("mapping","query","readout")
ALPHA=.75


def export(path,records):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n")
        w.writeheader()
        w.writerows(records)


def normalized(x):
    v=x.astype(np.float32,copy=False)
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-8)


def ridge(x,y,alpha=ALPHA):
    center=x.mean(axis=0)
    bias=y.mean(axis=0)
    xc=x-center
    q=xc.T@xc
    q.flat[::q.shape[0]+1]+=alpha
    return np.linalg.solve(q,xc.T@(y-bias)),center,bias


def apply(w,x):
    h,center,bias=w
    return (x-center)@h+bias


def group_ci(values,seed=741):
    v=np.asarray(values,dtype=float)
    rnd=np.random.default_rng(seed)
    means=v[rnd.integers(0,len(v),(6000,len(v)))].mean(axis=1)
    return float(v.mean()),*map(float,np.quantile(means,(.025,.975)))


class TaskData:
    def __init__(self,study,task,groups):
        location=study/"natural"/task
        assert (location/"complete.json").is_file()
        self.task=task
        self.rows=load_lines(location/"rows.jsonl")
        self.vectors=np.load(location/"vectors.npy",mmap_mode="r")
        self.raw=np.load(location/"stats.npy",mmap_mode="r")
        self.groups=[groups[r["id"]] for r in self.rows]
        self.by_group=defaultdict(dict)
        for i,r in enumerate(self.rows):
            assert i==r["i"]
            self.by_group[r["id"]][r["world"]]=i
        self.splits=np.array([r["split"] for r in self.rows])
        self.valid=np.array([r["world"]!="n" for r in self.rows])
        self.natural=np.array([r["pred"] for r in self.rows])
        self.gold=np.array([r["gold"] for r in self.rows])
        self.length=np.array([len(r["ids"]) for r in self.rows])
        self.mid=np.array([r["mid"] for r in self.rows]) if task in TASKS else None

    def feature(self,k,l,j):
        return normalized(self.vectors[:,k,l,j])

    def data_mask(self,split):
        return (self.splits==split)&self.valid

    def labels(self,target):
        return Targets(self,target)


class Targets:
    def __init__(self,data,target):
        self.task=data.task
        self.target=target
        if target=="intermediate":
            self.options=[g["intermediate_options"] for g in data.groups]
            self.gold=data.mid
        else:
            self.options=[g["options"] for g in data.groups]
            self.gold=data.gold
        self.truth=np.array([options[i] for options,i in zip(self.options,self.gold)])
        self.size=max(map(len,self.options))
        if data.task=="arithmetic":
            self.values=np.full((len(self.truth),self.size),np.inf,np.float32)
            for i,z in enumerate(self.options):
                self.values[i,:len(z)]=list(map(int,z))
            self.y=self.truth.astype(np.float32)[:,None]
        else:
            vocab=sorted({v for i in np.flatnonzero(data.data_mask("fit")) for v in self.options[i]})
            self.vocab={v:i for i,v in enumerate(vocab)}
            self.choice=np.full((len(self.truth),self.size),len(vocab),dtype=int)
            self.y=np.zeros((len(self.truth),len(vocab)),np.float32)
            for i,z in enumerate(self.options):
                self.choice[i,:len(z)]=[self.vocab.get(v,len(vocab)) for v in z]
                if self.truth[i] in self.vocab:
                    self.y[i,self.vocab[self.truth[i]]]=1

    def scores(self,w,x,mask=None):
        y=apply(w,x)
        if self.task=="arithmetic":
            values=self.values if mask is None else self.values[mask]
            return -abs(y-values)
        candidate=self.choice if mask is None else self.choice[mask]
        with_other=np.pad(y,((0,0),(0,1)),constant_values=-1e6)
        return np.take_along_axis(with_other,candidate,axis=1)

    def evaluate(self,w,x,mask):
        scores=self.scores(w,x[mask],mask)
        truth=self.gold[mask]
        best=np.argmax(scores,axis=1)
        wins=(best==truth)&(np.sum(scores==scores.max(axis=1,keepdims=True),axis=1)==1)
        return wins

    def signed_margin(self,w,x):
        scores=self.scores(w,x)
        gold=scores[np.arange(len(scores)),self.gold]
        scores[np.arange(len(scores)),self.gold]=-1e6
        return gold-scores.max(axis=1)


def probe_rows(data,phase):
    fit=data.data_mask("fit")
    splits=("discover",) if phase=="discover" else ("confirm","transfer")
    rows=[]
    indices=[(k,l,j) for j in range(len(ROLES)) for k in range(4)
             for l in (TAPS if phase=="discover" else range(25))]
    for target in ("intermediate","final"):
        y=data.labels(target)
        for k,l,j in indices:
            x=data.feature(k,l,j)
            model=ridge(x[fit],y.y[fit])
            for part in splits:
                mask=data.data_mask(part)
                correct=y.evaluate(model,x,mask)
                rows.append({"task":data.task,"target":target,"split":part,"role":ROLES[j],
                             "k":k+1,"layer":l,"correct":int(correct.sum()),"n":int(mask.sum()),
                             "accuracy":float(correct.mean())})
    return rows


def transfer_rows(data):
    result=[]
    fit=data.data_mask("fit")
    y=data.labels("intermediate")
    for l in (11,23,24):
        for src in (0,2,4):
            for k0 in range(4):
                v=data.feature(k0,l,src)
                w=ridge(v[fit],y.y[fit])
                for dest in (src,2) if src!=2 else (2,):
                    for k1 in range(4):
                        x=data.feature(k1,l,dest)
                        for split in ("confirm","transfer"):
                            mask=data.data_mask(split)
                            correct=y.evaluate(w,x,mask)
                            result.append(dict(task=data.task,split=split,layer=l,
                                               train_role=ROLES[src],test_role=ROLES[dest],
                                               train_K=k0+1,test_K=k1+1,n=int(mask.sum()),
                                               accuracy=float(correct.mean())))
    return result


def surface_features(data,role):
    x=np.zeros((len(data.rows),256),np.float32)
    for i,r in enumerate(data.rows):
        for j,p in enumerate(r["semantic_roles"][role]):
            tok=r["ids"][p]
            h=(tok*65521+j*113) % 256
            x[i,h]+=1 if (tok*126271+j)%11<6 else -1
    return normalized(x)


def lexical_control(data):
    rows=[]
    fit=data.data_mask("fit")
    for label in ("intermediate","final"):
        y=data.labels(label)
        for role in ROLES:
            x=surface_features(data,role)
            w=ridge(x[fit],y.y[fit])
            for split in ("confirm","transfer"):
                mask=data.data_mask(split)
                score=y.evaluate(w,x,mask)
                rows.append(dict(task=data.task,target=label,role=role,split=split,
                                 n=len(score),accuracy=float(score.mean())))
    return rows


def pair_geometry(data):
    group_data=[]
    for g,ix in data.by_group.items():
        base=ix["r"]
        split=data.rows[base]["split"]
        if split not in ("confirm","transfer"):
            continue
        for pair in (("r","x"),("r","y"),("r","d"),("x","y"),("x","d"),("y","d"),("r","n")):
            if pair[1] not in ix:
                continue
            a,b=[ix[i] for i in pair]
            r,s=data.rows[a],data.rows[b]
            surface=np.mean(np.array(r["ids"])!=np.array(s["ids"]))
            inter=int(r["mid"]!=s["mid"])
            final=int(r["gold"]!=s["gold"])
            map_flip=int((pair[0] in "yd")!=(pair[1] in "yd"))
            group_data.append((g,split,"".join(pair),a,b,surface,inter,final,map_flip,int("n" in pair)))
    g=np.array(group_data,dtype=object)
    stats=[]
    contrast=[]
    xa,xb=g[:,3].astype(int),g[:,4].astype(int)
    for k in range(4):
        for l in range(25):
            for j,role in enumerate(ROLES):
                x=data.feature(k,l,j)
                distance=1-(x[xa]*x[xb]).sum(axis=1)
                for part in ("confirm","transfer"):
                    ix=g[:,1]==part
                    structure=g[ix,5:].astype(np.float32)
                    design=np.column_stack([np.ones(np.sum(ix)),structure])
                    y=distance[ix]
                    adj=np.linalg.lstsq(design,y,rcond=None)[0]
                    rank=np.linalg.matrix_rank(design)
                    stats.append(dict(task=data.task,split=part,k=k+1,layer=l,role=role,pairs=len(y),
                         matrix_rank=int(rank),surface=float(adj[1]),intermediate=float(adj[2]),
                         final=float(adj[3]),map_factor=float(adj[4]),nuisance=float(adj[5])))
                    for kind in ("rx","ry","rd","rn"):
                        mask=ix&(g[:,2]==kind)
                        v=distance[mask]
                        limits=group_ci(v) if l in (11,23,24) and j in (0,2,4) else (v.mean(),None,None)
                        contrast.append(dict(task=data.task,split=part,k=k+1,layer=l,role=role,pair=kind,
                                     groups=len(v),mean=float(limits[0]),lower=limits[1],upper=limits[2]))
    return stats,contrast


def performance(data):
    summary=[]
    for part in sorted(set(data.splits)):
        base=np.array([r["world"]=="r" for r in data.rows])&(data.splits==part)
        good=data.natural[base]==data.gold[base,None]
        clean=data.natural[base]
        labels=np.where(good.all(axis=1),"always_correct",
              np.where((~good).all(axis=1),"always_wrong",
              np.where((~good[:,0])&good[:,-1],"corrected",
              np.where(good[:,0]&(~good[:,-1]),"regressed","transient"))))
        a=np.flatnonzero(base)
        for idx,l in enumerate(a):
            summary.append(dict(task=data.task,id=data.rows[l]["id"],split=part,index=int(l),
                                trajectory=labels[idx],stable_choice=bool(np.all(clean[idx]==clean[idx,0])),
                                K1=int(good[idx,0]),K2=int(good[idx,1]),K3=int(good[idx,2]),K4=int(good[idx,3])))
    return summary


def failures(data,raw,recommend):
    y=data.labels("intermediate")
    selection=np.array([r["world"]=="r" and r["split"] in ("confirm","transfer") for r in data.rows])
    k,l,j=recommend
    fit=data.data_mask("fit")
    observed=[]
    for depth in range(4):
        x=data.feature(depth,l,j)
        w=ridge(x[fit],y.y[fit])
        observed.append(y.signed_margin(w,x))
    chunks=defaultdict(list)
    for r in raw:
        i=r["index"]
        if not selection[i]:
            continue
        chunks[r["split"],r["trajectory"]].append(i)
    trajectories=[]
    for (part,label),ix in sorted(chunks.items()):
        for kk in range(4):
            v=np.array(observed[kk])[ix]
            p,lo,hi=group_ci(v)
            ac=float(np.mean(data.natural[ix,kk]==data.gold[ix]))
            trajectories.append(dict(task=data.task,split=part,trajectory=label,k=kk+1,layer=l,role=ROLES[j],
                          n=len(ix),mid_margin_mean=p,mid_margin_lower=lo,mid_margin_upper=hi,
                          final_accuracy=ac))
    association=[]
    # Pre-outcome diagnostic reads K2/L11 and conditions on the K1 decision margin.
    j=ROLES.index("mapping"); l=11
    inc=[]
    for depth in (0,1):
        xx=data.feature(depth,l,j)
        w=ridge(xx[fit],y.y[fit])
        inc.append(y.signed_margin(w,xx))
    predictor=inc[1]-inc[0]
    for part in ("confirm","transfer"):
        ix=np.flatnonzero((data.splits==part)&selection&(data.natural[:,0]!=data.gold))
        nuisance=[]
        for i in ix:
            r=data.rows[i]
            q=r["scores"][0]
            truth=data.gold[i]
            nuisance.append([q[truth]-max(z for u,z in enumerate(q) if u!=truth),
                          data.length[i],data.raw[i,0,l,j,0],r["family"]==1,r["mode"]=="context"])
        v=predictor[ix]
        v=(v-v.mean())/max(v.std(),1e-8)
        Z=np.array(nuisance,dtype=float)
        Z=(Z-Z.mean(0))/np.maximum(Z.std(0),1e-8)
        X=np.column_stack([np.ones(len(v)),v,Z])
        Y=(data.natural[ix,3]==data.gold[ix]).astype(float)
        obs=np.linalg.lstsq(X,Y,rcond=None)[0][1]
        rng=np.random.default_rng(806)
        boots=[]
        for _ in range(1600):
            s=rng.integers(0,len(ix),len(ix))
            boots.append(np.linalg.lstsq(X[s],Y[s],rcond=None)[0][1])
        lo,hi=np.quantile(boots,[.025,.975])
        association.append(dict(task=data.task,split=part,n=len(ix),corrected=int(Y.sum()),
                    change_per_sd=float(obs),lower=float(lo),upper=float(hi)))
    return trajectories,association


def freeze(tasks,study):
    out=[]
    for t in TASKS:
        out.extend(probe_rows(tasks[t],"discover"))
    export(study/"discovery_readouts.csv",out)
    mid={(r["task"],r["role"],r["k"],r["layer"]):r["accuracy"]
         for r in out if r["target"]=="intermediate"}
    selected={}
    for task in TASKS:
        def key(item):
            role,k,l=item
            return mid[task,role,k,l]-mid[task,role,1,l]
        site=max(((role,k,l) for role in ("mapping","query") for k in (2,3,4) for l in (11,23)
                  if (k,l)!=(4,23)),key=key)
        selected[task]={"role":site[0],"k":site[1],"layer":site[2],"discover_gain":key(site)}
    # All cross-task choices below use relation and logic discovery outcomes only.
    def cross(item):
        role,l=item
        ds=[mid[t,role,4,l]-mid[t,role,1,l] for t in ("relation","logic")]
        return min(ds)
    shared=max(((r,l) for r in LATE_ROLES for l in (11,23,24)),key=cross)
    result={"selected_for_causality":selected,"heldout_prediction":{"role":shared[0],"layer":shared[1],
             "fit_tasks":["relation","logic"],"holdout_task":"arithmetic","expect":"K4 > K1",
             "discovery_minimum_gain":cross(shared)},
             "inputs_sha256":digest(study/"groups.jsonl"),"analysis_sha256":digest(Path(__file__)),
             "readout_alpha":ALPHA,"discovery_sha256":digest(study/"discovery_readouts.csv")}
    path=study/"analysis_freeze.json"
    with path.open("x",encoding="utf-8") as f:
        json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


def main(study,stage):
    groups={g["id"]:g for g in load_lines(study/"groups.jsonl")}
    tasks={t:TaskData(study,t,groups) for t in TASKS}
    if stage=="freeze":
        freeze(tasks,study)
        return
    frozen=json.loads((study/"analysis_freeze.json").read_text())
    assert frozen["inputs_sha256"]==digest(study/"groups.jsonl")
    started=time.perf_counter()
    names=["readouts","cross_loop","surface_baselines","rdm_adjusted","paired_distances",
           "behavior","success_failure","correction_association"]
    out={n:[] for n in names}
    evidence=[]
    for task,d in tasks.items():
        out["readouts"].extend(probe_rows(d,"confirm"))
        out["cross_loop"].extend(transfer_rows(d))
        out["surface_baselines"].extend(lexical_control(d))
        raw,contrasts=pair_geometry(d)
        out["rdm_adjusted"].extend(raw)
        out["paired_distances"].extend(contrasts)
        actual=performance(d)
        out["behavior"].extend(actual)
        shared=frozen["heldout_prediction"]
        site=frozen["selected_for_causality"][task]
        p,a=failures(d,actual,(site["k"]-1,site["layer"],ROLES.index(site["role"])))
        out["success_failure"].extend(p); out["correction_association"].extend(a)
        y=d.labels("intermediate")
        train=d.data_mask("fit")
        for split in ("confirm","transfer"):
            mask=d.data_mask(split)
            j=ROLES.index(shared["role"]); l=shared["layer"]
            between=[]
            for k in (0,3):
                x=d.feature(k,l,j)
                w=ridge(x[train],y.y[train])
                between.append(y.evaluate(w,x,mask).reshape(-1,4))
            diff=between[1].mean(1)-between[0].mean(1)
            v,lo,hi=group_ci(diff)
            evidence.append(dict(task=task,split=split,role=shared["role"],layer=l,n=len(diff),
                         K1_correct=float(between[0].mean()),K4_correct=float(between[1].mean()),
                         gain=v,lower=lo,upper=hi))
        print(task,round(time.perf_counter()-started,1),flush=True)
    for t in GOALS:
        if t not in TASKS:
            out["behavior"].extend(performance(TaskData(study,t,groups)))
    for name,rows in out.items():
        export(study/f"{name}.csv",rows)
    export(study/"shared_dynamics.csv",evidence)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=("freeze","confirm"))
    p.add_argument("study",type=Path)
    a=p.parse_args()
    main(a.study,a.stage)
