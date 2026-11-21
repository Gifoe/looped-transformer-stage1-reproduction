"""Separate one residual stream injection from the later block updates."""
import argparse
from pathlib import Path

import torch

from edge import short_hash
from interchange import Patch, terminated
from native import Forward, load, output, pad, save, setup, sha, timer
from query_test import encode, whitespace
from residual_deck import VARIANTS, is_pair
from residual_stats import ARMS, CONFIG, SIDES, summarize
from rule import MAX_NEW, text_results
from value_test import shuffle

SITE=19
FINAL=23
LATE=tuple(range(SITE+1,FINAL+1))
WRITES=("a_write","m_write")
SWITCH={"original":"swapped","swapped":"original"}


def make_case(tok,r):
    return dict(r,entry={k:encode(tok,r,k)["ids"] for k in VARIANTS},
                prefix={k:whitespace(tok,r["native"][k]) for k in SIDES})


class Native:
    def __init__(self,at):
        self.at=at
        self.rows=torch.arange(len(at),device=at.device)
        self.q={}
        self.writes={}
        self.x=None

    def put(self,key,name,x):
        p,l=key
        if l==SITE and name=="q":
            self.q[p]=x[self.rows,:,self.at,:].detach()
        if p!=1:
            return
        if l==SITE and name=="input":
            self.x=x[self.rows,self.at].detach()
        if l in LATE and name in WRITES:
            self.writes[l,name]=x[self.rows,self.at].detach()


class Replay(Patch):
    def __init__(self,at,q,source,audit):
        super().__init__(at,query={SITE:(None,q)})
        self.source=source
        self.audit=audit

    def write(self,key,name,x):
        k=(key[1],name)
        if k not in self.source:
            return x
        slots,active,values=self.source[k]
        changed=x.clone()
        changed[slots,active]=values
        if self.audit:
            assert torch.equal(changed[slots,active],values)
        return changed


class Injection:
    def __init__(self,block,at,d):
        self.block=block
        self.at=at
        self.delta=d
        self.rows=torch.arange(len(at),device=at.device)

    def __enter__(self):
        def add(_,args,result):
            out=result.clone()
            out[self.rows,self.at]+=self.delta
            return out
        self.hook=self.block.register_forward_hook(add)
        return self

    def __exit__(self,*_):
        self.hook.remove()


class Audit:
    def __init__(self,at):
        self.at=at
        self.idx=torch.arange(len(at),device=at.device)
        self.fields={}

    def put(self,key,name,x):
        l=key[1]
        if l==SITE and name in ("k","v"):
            self.fields[name]=x[self.idx,:,self.at,:].detach()
        if l==SITE and name in WRITES:
            self.fields[name]=x[self.idx,self.at].detach()
        if l==FINAL and name=="output":
            self.fields["last"]=x[self.idx,self.at].detach()


def snapshots(f,rows,keys):
    out={}
    for j in range(0,len(keys),18):
        part=keys[j:j+18]
        seq=[rows[i]["entry"][s]+list(p) for i,s,p in part]
        x,at=pad(seq)
        tr=Native(at)
        h1,h2=f.run(x,2,record=tr)
        nxt=output(f.model,h2,at).argmax(-1).tolist()
        assert tr.x is not None and len(tr.writes)==8 and len(tr.q)==2
        for u,key in enumerate(part):
            i,side,p=key
            if not p:
                assert nxt[u]==rows[i]["native"][side][1]["tokens"][0]
            out[key]=dict(ids=seq[u],start=h1[u,:len(seq[u])],x=tr.x[u],q1=tr.q[0][u],q2=tr.q[1][u],
                          clean=nxt[u],writes={k:v[u] for k,v in tr.writes.items()})
    return out


def dose(direction,displacement):
    z=direction.norm()
    assert z>1e-8
    v=direction*(displacement.norm()/z)
    torch.testing.assert_close(v.norm(),displacement.norm(),rtol=5e-5,atol=5e-6)
    return v


def calc(f,rows,requests,audit=False):
    need={}
    for i,side,a,p in requests:
        need[i,side,p]=None
        if a in ("native","identity"):
            continue
        kind,late=CONFIG[a]
        if kind!="own" or late=="fixed_other":
            need[i,SWITCH[side],p]=None
        if kind=="irrelevant":
            need[i,"unrelated_"+side,p]=None
    bank=snapshots(f,rows,list(need))
    result={}
    edits=[]
    stats={}
    for i,side,a,p in requests:
        recv=bank[i,side,p]
        if a=="native":
            result[i,side,a]=recv["clean"]
            continue
        x=recv["x"]
        conf=CONFIG[a] if a!="identity" else ("own","locked")
        kind,late=conf
        delta=torch.zeros_like(x)
        if kind!="own":
            d=bank[i,SWITCH[side],p]["x"]-x
            if kind=="swap":
                delta=d
            elif kind=="geometry":
                delta=shuffle(x,d)-x
            else:
                delta=dose(bank[i,"unrelated_"+side,p]["x"]-x,d)
        writer=None
        if late=="locked":
            writer=recv
        if late=="fixed_other":
            writer=bank[i,SWITCH[side],p]
        edits.append((i,side,a,recv,delta,writer))
    for j in range(0,len(edits),22):
        group=edits[j:j+22]
        ids,at=pad([e["ids"] for _,_,_,e,_,_ in group])
        q=torch.stack([e["q2" if a=="identity" else "q1"] for _,_,a,e,_,_ in group])
        hidden=torch.nn.utils.rnn.pad_sequence([r["start"] for _,_,_,r,_,_ in group],batch_first=True)
        shift=torch.stack([v for _,_,_,_,v,_ in group])
        keep=[(k,w["writes"]) for k,(_,_,_,_,_,w) in enumerate(group) if w is not None]
        source={}
        if keep:
            alloc=torch.tensor([k for k,_ in keep],device=ids.device)
            source={k:(alloc,at[alloc],torch.stack([x[k] for _,x in keep])) for k in keep[0][1]}
        watch=Audit(at) if audit else None
        with Injection(f.layers[SITE],at,shift):
            end=f.run(ids,1,start=hidden,edit=Replay(at,q,source,audit),record=watch)[0]
        prediction=output(f.model,end,at).argmax(-1).tolist()
        for k,(i,side,a,r,d,_) in enumerate(group):
            result[i,side,a]=prediction[k]
            if a=="identity":
                assert prediction[k]==r["clean"]
            if audit:
                stats[i,side,a]=dict({name:v[k] for name,v in watch.fields.items()},delta=d)
    return result,stats


def decode(f,tok,rows):
    pending={(i,s,a):[] for i in range(len(rows)) for s in SIDES for a in ARMS}
    end={}
    for _ in range(MAX_NEW):
        if not pending:
            break
        q=[(i,s,a,tuple(p)) for (i,s,a),p in pending.items()]
        pred,_=calc(f,rows,q)
        for x,t in list(pending.items()):
            t.append(pred[x])
            if terminated(tok,t):
                end[x]=pending.pop(x)
    end.update(pending)
    out=dict(zip(end,text_results(tok,list(end.values())),strict=True))
    data=[{s:{a:out[i,s,a] for a in ARMS} for s in SIDES} for i in range(len(rows))]
    for r,case in zip(rows,data,strict=True):
        assert all(case[s]["native"]==r["native"][s][1] for s in SIDES)
    return data


def diagnose(f,rows,take):
    arms=("native","identity","own_locked","swap_locked","geometry_locked","irrelevant_locked")
    q=[(i,s,a,tuple(rows[i]["prefix"][s])) for i in take for s in SIDES for a in arms]
    _,p=calc(f,rows,q,audit=True)
    stats={}
    for i in take:
        a={}
        for side in SIDES:
            c=p[i,side,"own_locked"]
            e={}
            for tag in arms[3:]:
                t=p[i,side,tag]
                for k in ("k","v",*WRITES):
                    torch.testing.assert_close(c[k],t[k],rtol=1e-4,atol=4e-4)
                delta=t["last"]-c["last"]
                torch.testing.assert_close(delta,t["delta"],rtol=8e-4,atol=9e-4)
                e[tag]=dict(delta_norm=t["delta"].norm().item(),error=(delta-t["delta"]).abs().max().item(),
                            k_v_max=max((c[k]-t[k]).abs().max().item() for k in ("k","v")),
                            layer20_writes_max=max((c[k]-t[k]).abs().max().item() for k in WRITES))
            a[side]=e
        stats[i]=a
    return stats


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    started=timer()
    p=root/"PAIR_NATIVE.json"
    h=sha(p)
    data=load(p)["rows"]
    chosen=[make_case(tok,r) for r in data if is_pair(r)]
    ids=[r["id"] for r in chosen]
    selected=set(sorted(ids,key=lambda x:short_hash("residual frozen diagnostic",x))[:24])
    doc=root/"COHORT.json"
    proof=dict(pair_native_sha256=h,eligible=ids,diagnostic=sorted(selected),physical_layer=20)
    if doc.exists():
        assert load(doc)==proof
    else:
        save(doc,proof)
    raw=root/"RESIDUAL_RAW.json"
    past=load(raw) if raw.exists() else dict(rows=[],elapsed=0.)
    out=past["rows"]
    if out:
        assert past["pair_native_sha256"]==h and past["cohort_sha256"]==sha(doc)
        assert [r["id"] for r in out]==ids[:len(out)]
    for j in range(len(out),len(chosen),4):
        a=chosen[j:j+4]
        preds=decode(f,tok,a)
        tocheck=[i for i,r in enumerate(a) if r["id"] in selected]
        detail=diagnose(f,a,tocheck) if tocheck else {}
        for i,(r,c) in enumerate(zip(a,preds,strict=True)):
            z=dict({k:r[k] for k in ("id","target","old","key","example","native")},pred=c)
            if i in detail:
                z["diagnostics"]=detail[i]
            out.append(z)
        save(raw,dict(rows=out,pair_native_sha256=h,cohort_sha256=sha(doc),elapsed=past["elapsed"]+timer()-started,
                      peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1)))
        print("residual factorial",len(out),len(chosen),flush=True)
    s=summarize(out)
    save(root/"RESIDUAL_SUMMARY.json",s)
    print("residual gates",s["gates"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)
