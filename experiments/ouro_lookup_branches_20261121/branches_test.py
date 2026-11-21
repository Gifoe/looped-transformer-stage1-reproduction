"""Replay late branch writes while preserving the natural K2 layer-20 state."""
import argparse
from pathlib import Path

import torch

from branches_deck import swapped_repair
from branches_stats import ARMS, CONFIG, summarize
from edge import short_hash
from interchange import Patch, terminated
from native import Forward, load, output, pad, save, setup, sha, timer
from query_deck import text
from query_test import whitespace
from rule import MAX_NEW, text_results
from value_test import shuffle

READOUT=19
LAST=23
SITES=tuple(range(READOUT+1,LAST+1))
NAMES=("a_write","m_write")


def case(tok,r,side):
    baseline=r["native"][side]
    target=r["target"] if side=="original" else r["old"]
    wrong=baseline[0]["number"]
    label="original" if side=="original" else ("paired_correct_k1" if wrong==target else "paired_natural_repair")
    if label=="paired_natural_repair" and not swapped_repair(r):
        label="paired_other_k1_error"
    return dict(id=r["id"],source_target=r["target"],target=target,foil=wrong,side=side,group=label,
                native=baseline,ids=tok.encode(text(r["key"],**r["example"][side]),add_special_tokens=False),
                whitespace=whitespace(tok,baseline))


class Trace:
    def __init__(self,at,audit):
        self.at=at
        self.i=torch.arange(len(at),device=at.device)
        self.values={}
        self.states={}
        self.audit=audit

    def put(self,key,name,x):
        p,l=key
        if l in SITES and name in NAMES:
            self.values[p,l,name]=x.detach()
        if self.audit and l in (READOUT,LAST) and name=="output":
            self.states[p,l]=x[self.i,self.at].detach()


class Replay(Patch):
    def __init__(self,at,inject,audit):
        super().__init__(at)
        self.inject=inject
        self.audit=audit

    def write(self,key,name,x):
        override=self.inject.get((key[1],name))
        if override is None:
            return x
        changed=x.clone()
        for scope,idx,z in override:
            if scope=="local":
                changed[idx,self.pos[idx]]=z
                if self.audit:
                    assert torch.equal(changed[idx,self.pos[idx]],z)
            else:
                u=torch.arange(x.size(1),device=x.device)[None,:]<=self.pos[idx,None]
                changed[idx]=torch.where(u[...,None],z,changed[idx])
                if self.audit:
                    assert torch.equal(changed[idx][u],z[u])
        return changed


class LastState:
    def __init__(self,at):
        self.at=at
        self.rows=torch.arange(len(at),device=at.device)
        self.x=None

    def put(self,key,name,value):
        if key[1]==LAST and name=="output":
            self.x=value[self.rows,self.at].detach()


def snapshots(f,rows,keys,audit):
    native={}
    for j in range(0,len(keys),12):
        group=keys[j:j+12]
        seq=[rows[i]["ids"]+list(p) for i,p in group]
        ids,at=pad(seq)
        tr=Trace(at,audit)
        h1,h2=f.run(ids,2,record=tr)
        clean=output(f.model,h2,at).argmax(-1).tolist()
        normal={(l,k):tr.values[1,l,k] for l in SITES for k in NAMES}
        old={(l,k):tr.values[0,l,k] for l in SITES for k in NAMES}
        control={k:shuffle(x,old[k]-x) for k,x in normal.items()}
        for r,key in enumerate(group):
            length=len(seq[r])
            data=dict(ids=seq[r],start=h1[r,:length],clean=clean[r],
                      banks={k:{l:v[r,:length] for l,v in bank.items()} for k,bank in
                             (("authentic",old),("geometry",control),("identity",normal))})
            if audit:
                data["states"]={k:v[r] for k,v in tr.states.items()}
            if not key[1]:
                assert clean[r]==rows[key[0]]["native"][1]["tokens"][0]
            native[key]=data
    return native


def step(f,rows,requests,audit=False):
    bank=snapshots(f,rows,list(dict.fromkeys((i,p) for i,a,p in requests)),audit)
    out={}
    diag={}
    queued=[]
    for i,a,p in requests:
        b=bank[i,p]
        if a=="native":
            out[i,a]=b["clean"]
            continue
        scope,kind,branches=CONFIG[a] if a!="identity" else ("full","identity",NAMES)
        queued.append((i,a,b,scope,b["banks"][kind],branches))
    for j in range(0,len(queued),16):
        group=queued[j:j+16]
        ids,at=pad([r["ids"] for _,_,r,_,_,_ in group])
        h=torch.nn.utils.rnn.pad_sequence([r["start"] for _,_,r,_,_,_ in group],batch_first=True)
        controls={}
        for k in NAMES:
            for scope in ("local","full"):
                chosen=[(n,s) for n,(_,_,_,sc,s,w) in enumerate(group) if sc==scope and k in w]
                if not chosen:
                    continue
                ix=torch.tensor([n for n,_ in chosen],device=ids.device)
                for l in SITES:
                    if scope=="local":
                        val=torch.stack([d[l,k][-1] for _,d in chosen])
                    else:
                        val=torch.nn.utils.rnn.pad_sequence([d[l,k] for _,d in chosen],batch_first=True)
                        if val.size(1)<ids.size(1):
                            val=torch.nn.functional.pad(val,(0,0,0,ids.size(1)-val.size(1)))
                    controls.setdefault((l,k),[]).append((scope,ix,val))
        watch=LastState(at) if audit else None
        r=f.run(ids,1,start=h,edit=Replay(at,controls,audit),record=watch)[0]
        after=output(f.model,r,at).argmax(-1).tolist()
        for n,(i,a,z,_,_,_) in enumerate(group):
            out[i,a]=after[n]
            if audit:
                if a=="identity":
                    assert after[n]==z["clean"]
                    delta=z["states"][1,LAST]-watch.x[n]
                    torch.testing.assert_close(delta,torch.zeros_like(delta),rtol=0,atol=4e-4)
                else:
                    base=z["states"][1,READOUT]-z["states"][0,READOUT]
                    terminal=watch.x[n]-z["states"][0,LAST]
                    torch.testing.assert_close(terminal,base,rtol=1e-4,atol=4e-4)
                    diag[i,a]=(terminal-base).abs().max().item()
    return out,diag


def decode(f,tok,rows):
    active={(i,a):[] for i in range(len(rows)) for a in ARMS}
    results={}
    for _ in range(MAX_NEW):
        if not active:
            break
        x,_=step(f,rows,[(i,a,tuple(p)) for (i,a),p in active.items()])
        for k,p in list(active.items()):
            p.append(x[k])
            if terminated(tok,p):
                results[k]=active.pop(k)
    results.update(active)
    p=dict(zip(results,text_results(tok,list(results.values())),strict=True))
    out=[{a:p[i,a] for a in ARMS} for i in range(len(rows))]
    assert all(z["native"]==r["native"][1] for z,r in zip(out,rows,strict=True))
    return out


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    clock=timer()
    p=root/"PAIR_NATIVE.json"
    h=sha(p)
    original=load(p)["rows"]
    rows=[]
    for r in original:
        rows.append(case(tok,r,"original"))
        if r["native"]["swapped"][1]["number"]==r["old"]:
            rows.append(case(tok,r,"swapped"))
    use=sorted((i for i,r in enumerate(rows) if r["side"]=="original"),
               key=lambda k:short_hash("branch replay",rows[k]["id"]))[:24]
    record={r["id"] for i,r in enumerate(rows) if i in use}
    cohort=root/"COHORT.json"
    fixed=dict(pair_native_sha256=h,outputs=[(r["id"],r["group"]) for r in rows],diagnostics=sorted(record))
    if cohort.exists():
        import json
        assert load(cohort)==json.loads(json.dumps(fixed))
    else:
        save(cohort,fixed)
    raw=root/"BRANCH_RAW.json"
    prev=load(raw) if raw.exists() else dict(rows=[],elapsed=0.)
    out=prev["rows"]
    if out:
        assert prev["pair_native_sha256"]==h and prev["cohort_sha256"]==sha(cohort)
        assert [(r["id"],r["group"]) for r in out]==fixed["outputs"][:len(out)]
    for j in range(len(out),len(rows),7):
        q=rows[j:j+7]
        values=decode(f,tok,q)
        diag=[(i,a,tuple(r["whitespace"])) for i,r in enumerate(q) if r["side"]=="original" and r["id"] in record
              for a in ("identity","local_authentic_both","full_authentic_both")]
        _,checks=step(f,q,diag,audit=True) if diag else ({},{})
        for i,(r,v) in enumerate(zip(q,values,strict=True)):
            item=dict({k:a for k,a in r.items() if k not in ("ids","whitespace")},pred=v)
            if (i,"local_authentic_both") in checks:
                item["additive_error"]={key:checks[i,key+"_authentic_both"] for key in ("local","full")}
            out.append(item)
        save(raw,dict(rows=out,pair_native_sha256=h,cohort_sha256=sha(cohort),
                      elapsed=prev["elapsed"]+timer()-clock,
                      peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1)))
        print("late branch replay",len(out),len(rows),flush=True)
    z=summarize(out)
    save(root/"BRANCH_SUMMARY.json",z)
    print("late branch gates",z["gates"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)
