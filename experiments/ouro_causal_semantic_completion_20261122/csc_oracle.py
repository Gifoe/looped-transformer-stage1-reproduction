"""Prospective complete-answer Oracle with grouped paired tests and low-rank controls."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

from csc_bank import KINDS,LAYER,UPDATES,load_bank,metadata
from csc_data import repair
from interchange import terminated
from native import Forward,load,output,pad,save,setup,sha
from rule import MAX_NEW,text_results
from value_test import shuffle

RANKS=(16,32,64)


class Write:
    def __init__(self,f,l,pos,changes):
        self.module=f.layers[l]
        self.pos=pos
        self.change=changes

    def __enter__(self):
        def add(_,args,x):
            row=torch.arange(len(self.pos),device=x.device)
            y=x.clone()
            y[row,self.pos]+=self.change
            return y
        self.hook=self.module.register_forward_hook(add)
        return self

    def __exit__(self,*exc):
        self.hook.remove()


def last_prefix(case,p):
    return max((q for q in case["prefixes"] if p[:len(q)]==q),key=len)


def prepared(bank,case,p,arm,slot,depth,ranks,other=None):
    prefix=last_prefix(case,p)
    z=bank[case["id"],case["kind"],prefix]
    site=len(case["quad"][3])+len(prefix)-1
    if arm=="identity":
        return site,torch.zeros_like(z["base"][0])
    key=arm.removesuffix("_geom")
    if key in UPDATES:
        u=z["updates"][UPDATES.index(key)].float()
    elif key=="wrong_position":
        u=z["updates"][slot].float()
        site=case["roles"][0][-1]
    elif key=="same_answer_donor":
        ds=bank[case["id"],"joint",()]
        u=ds["updates"][slot].float()
    elif key=="other_operand":
        u=z["updates"][slot].float()
        site=case["roles"][1][-1]
    else:
        u=z["updates"][slot].float()
        if key.startswith("rank"):
            n=int(key[4:])
            basis=ranks[:n]
            u=basis.T@(basis@u)
    if depth==2:
        u=-u
    if arm.endswith("_geom"):
        u=shuffle(z["base"][depth-1].float().unsqueeze(0),u.unsqueeze(0))[0]-z["base"][depth-1].float()
    return site,u


def decode(f,tok,cases,bank,arms,depth=1,slot=0,basis=None):
    active={(i,a):[] for i,r in enumerate(cases) for a in arms if a!="same_answer_donor" or r["kind"]=="safe"}
    done={}
    for _ in range(MAX_NEW):
        if not active:
            break
        items=list(active)
        for j in range(0,len(items),32):
            take=items[j:j+32]
            seq=[cases[i]["quad"][3]+active[i,a] for i,a in take]
            ids,at=pad(seq)
            plan=[prepared(bank,cases[i],tuple(active[i,a]),a,slot,depth,basis) for i,a in take]
            positions=torch.tensor([p for p,_ in plan],device=ids.device)
            changes=torch.stack([u for _,u in plan]).to(ids.device)
            if depth==2:
                start=f.run(ids,1)[0]
            else:
                start=None
            with Write(f,LAYER,positions,changes):
                net=f.run(ids,1,start=start)[0]
            z=output(f.model,net,at).argmax(-1).tolist()
            for key,t in zip(take,z,strict=True):
                active[key].append(t)
                if terminated(tok,active[key]):
                    done[key]=active.pop(key)
    done.update(active)
    answers=dict(zip(done,text_results(tok,list(done.values())),strict=True))
    rows=[{a:answers[j,a] for a in arms if (j,a) in answers} for j in range(len(cases))]
    for r,v in zip(cases,rows,strict=True):
        if "identity" in v:
            assert v["identity"]==r["native"][depth-1]
    return rows


def stat(rows,x,y,depth=1,key=None):
    grouped=defaultdict(list)
    p=key or "target"
    field="pred" if depth==1 else "rollback"
    for r in rows:
        a,b=[r[field][m]["number"]==r[p] for m in (x,y)]
        grouped[r["target"]].append(int(a)-int(b))
    if not grouped:
        return None
    xs=np.array([np.mean(a) for a in grouped.values()])
    rng=np.random.default_rng(261134349)
    idx=rng.integers(len(xs),size=(10000,len(xs)))
    q=np.quantile(xs[idx].mean(-1),[.025,.975]).tolist()
    return dict(mean=float(xs.mean()),interval=q,groups=len(xs),effect_count=int(sum(map(sum,grouped.values()))))


def summary(rows):
    result={}
    for form in ("borrow","carry"):
        for split in sorted({x["split"] for x in rows}):
            for kind in KINDS:
                a=[r for r in rows if (r["form"],r["split"],r["kind"])==(form,split,kind)]
                if not a:
                    continue
                rep=[r for r in a if r["repair"]]
                old=[r["native"][0]["number"]==r["target"] for r in a]
                keys=sorted(set().union(*(r["pred"].keys() for r in a)))
                scores={}
                for k in keys:
                    group=[r for r in a if k in r["pred"]]
                    hit=[r["pred"][k]["number"]==r["target"] for r in group]
                    eff=[r["native"][0]["number"]==r["target"] for r in group]
                    stats=dict(n=len(group),correct=sum(hit),rescued=sum(x and not y for x,y in zip(hit,eff,strict=True)),
                        damaged=sum(y and not x for x,y in zip(hit,eff,strict=True)),
                        exact_natural_rescues=sum(r["pred"][k]["number"]==r["target"] for r in rep if k in r["pred"]))
                    if k+"_geom" in keys and rep:
                        a0=[r for r in rep if k in r["pred"] and k+"_geom" in r["pred"]]
                        stats["repair_vs_geometry"]=stat(a0,k,k+"_geom")
                    scores[k]=stats
                result[f"{form}/{split}/{kind}"]=dict(n=len(a),native=sum(old),repairs=len(rep),arms=scores)
    return result


def replay(f,tok,cases,bank,arms,path,slot=0,basis=None,rollback=()):
    start=time.monotonic()
    state=load(path) if path.exists() else {"rows":[],"arms":list(arms),"rollback":list(rollback)}
    assert state["arms"]==list(arms) and state["rollback"]==list(rollback)
    out=state["rows"]
    load_keys=[(r["id"],r["kind"]) for r in out]
    assert load_keys==[(r["id"],r["kind"]) for r in cases[:len(out)]]
    for off in range(len(out),len(cases),12):
        batch=cases[off:off+12]
        got=decode(f,tok,batch,bank,arms,slot=slot,basis=basis)
        needed=[(i,r) for i,r in enumerate(batch) if rollback and r["repair"]]
        reverse=decode(f,tok,[r for _,r in needed],bank,rollback,depth=2,slot=slot,basis=basis) if needed else []
        redo={i:p for (i,_),p in zip(needed,reverse,strict=True)}
        for j,(r,x) in enumerate(zip(batch,got,strict=True)):
            out.append(dict({k:v for k,v in r.items() if k not in ("quad","roles","prefixes")},pred=x,rollback=redo.get(j,{})))
        save(path,state)
        print("CSC oracle",path.stem,len(out),len(cases),round(time.monotonic()-start,1),flush=True)
    return out


def train_basis(bank,items,slot):
    rows=[bank[r["id"],r["kind"],()]["updates"][slot].float() for r in items if
          r["split"]=="train" and r["form"]=="borrow" and r["kind"]=="joint"]
    _,s,v=torch.linalg.svd(torch.stack(rows),full_matrices=False)
    return dict(basis=v[:max(RANKS)].contiguous(),singular=s,train_entries=len(rows))


def pick(rows,broad,base):
    main=broad["borrow/dev/joint"]["arms"]
    safe=broad["borrow/dev/safe"]["arms"]
    def key(a):
        r=main[a]
        t=safe[a]
        c=r["exact_natural_rescues"]-main[a+"_geom"]["exact_natural_rescues"]
        return (c,r["rescued"]-r["damaged"]+t["rescued"]-t["damaged"],-UPDATES.index(a))
    full=max(UPDATES,key=key)
    return full,key(full)


def verify_gate(test,chosen):
    def row(form,split,kind):
        return test[f"{form}/{split}/{kind}"]
    subj=row("borrow","test","joint")
    safe=row("borrow","test","safe")
    full=subj["arms"][chosen]
    r=full["repair_vs_geometry"]
    s=full["rescued"]-full["damaged"]+safe["arms"][chosen]["rescued"]-safe["arms"][chosen]["damaged"]
    return dict(full=bool(full["exact_natural_rescues"]>=6 and r and r["groups"]>=10 and r["interval"][0]>0 and s>0),
                full_gain=s,full_borrow_test=full)


@torch.inference_mode()
def run(model,root):
    bank=load_bank(root)
    rows=load(root/"NATIVE.json")["rows"]
    m,tok=setup(model)
    f=Forward(m)
    cases=[dict(metadata(tok,r,k),repair=k=="joint" and repair(r)) for r in rows for k in KINDS]
    dev=[r for r in cases if r["split"]=="dev"]
    ways=("identity",*[x for a in UPDATES for x in (a,a+"_geom")])
    scan=replay(f,tok,dev,bank,ways,root/"ORACLE_DISCOVERY.json")
    base=summary(scan)
    chosen,score=pick(scan,base,bank)
    slot=UPDATES.index(chosen)
    model_basis=train_basis(bank,cases,slot)
    basis=model_basis["basis"]
    torch.save(model_basis,root/"ORACLE_BASIS.pt")
    compact=tuple(a for n in RANKS for a in (f"rank{n}",f"rank{n}_geom"))
    rank=replay(f,tok,dev,bank,compact,root/"ORACLE_RANK_DEV.json",slot=slot,basis=basis)
    rr=summary(rank)["borrow/dev/joint"]["arms"]
    benchmark=base["borrow/dev/joint"]["arms"][chosen]["exact_natural_rescues"]
    active=[]
    for n in RANKS:
        arm=f"rank{n}"
        x=rr[arm]
        if x["exact_natural_rescues"]>=max(2,.65*benchmark) and x["exact_natural_rescues"]>rr[arm+"_geom"]["exact_natural_rescues"]:
            active.append(n)
    rank_choice=min(active) if active else 64
    decision=dict(update=chosen,slot=slot,dev_score=score,chosen_rank=rank_choice,
                  rank_dev_gate=bool(active),train_basis="borrow train only",
                  discovery_sha256=sha(root/"ORACLE_DISCOVERY.json"),rank_development_sha256=sha(root/"ORACLE_RANK_DEV.json"))
    lock=root/"ORACLE_LOCK.json"
    if lock.exists():
        import json
        assert load(lock)==json.loads(json.dumps(decision))
    else:
        save(lock,decision)
    print("CSC Oracle frozen",decision,flush=True)
    verify=[r for r in cases if r["split"] in ("test","shift")]
    arms=("identity",chosen,chosen+"_geom",*compact,"same_answer_donor","wrong_position","other_operand")
    rank=f"rank{rank_choice}"
    checked=replay(f,tok,verify,bank,arms,root/"ORACLE_CONFIRM.json",slot=slot,basis=basis,
                   rollback=("identity",chosen,chosen+"_geom",rank,rank+"_geom"))
    test=summary(checked)
    gate=verify_gate(test,chosen)
    reversed_={}
    for form in ("borrow","carry"):
        for split in ("test","shift"):
            eligible=[r for r in checked if (r["form"],r["split"],r["kind"])==(form,split,"joint") and r["repair"]]
            reversed_[f"{form}/{split}"]={a:sum(r["rollback"][a]["number"]==r["foil"] for r in eligible)
                                        for a in ("identity",chosen,chosen+"_geom",rank,rank+"_geom")}
    save(root/"ORACLE_SUMMARY.json",dict(discovery=base,rank_development=summary(load(root/"ORACLE_RANK_DEV.json")["rows"]),
          confirmation=test,gate=gate,lock=decision,rollback_exact=reversed_))
    print("CSC Oracle gate",gate,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("result",type=Path)
    a=p.parse_args()
    run(a.model,a.result)
