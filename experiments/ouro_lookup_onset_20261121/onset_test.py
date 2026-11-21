"""Exchange natural boundary states in a fixed native suffix, with physical parity checks."""
import argparse
from pathlib import Path

import torch

from edge import short_hash
from interchange import Patch, terminated
from native import Forward, load, output, pad, save, setup, sha, timer
from onset_deck import DOMAINS, VARIANTS, pair, prompt
from onset_stats import SITES, MODES, summarize
from query_test import whitespace
from rule import MAX_NEW, text_results
from value_test import shuffle

LAST=23
BRANCH=("a_write","m_write")
DIRECTIONS=("forward","reverse")


def prepare(tok,row):
    return dict(row,ids={t:tok.encode(prompt(row,t),add_special_tokens=False) for t in VARIANTS},
                permitted=pair(row))


def counterpart(kind,direction):
    if kind=="phase":
        return ("original",0,"original",1) if direction=="forward" else ("original",1,"original",0)
    return ("original",1,"swapped",1) if direction=="forward" else ("swapped",1,"original",1)


class Trace:
    def __init__(self,at,physical):
        self.at=at
        self.ix=torch.arange(len(at),device=at.device)
        self.x={}
        self.z={}
        self.w={}
        self.physical=physical

    def put(self,key,name,x):
        k,l=key
        if name=="input" and l+1 in SITES:
            self.x[k,l+1]=x[self.ix,self.at].detach()
        if l==LAST and name=="output":
            self.z[k]=x[self.ix,self.at].detach()
        if self.physical and name in BRANCH:
            self.w[k,l,name]=x[self.ix,self.at].detach()


def readout(f,r):
    return f.model.lm_head(f.core.norm(r)).float()


def snapshots(f,rows,want,physical):
    out={}
    for j in range(0,len(want),20):
        group=want[j:j+20]
        ids0=[rows[i]["ids"][v]+list(p) for i,v,p in group]
        ids,at=pad(ids0)
        tr=Trace(at,physical)
        h1,h2=f.run(ids,2,record=tr)
        h=[h1,h2]
        original=[output(f.model,z,at).argmax(-1).tolist() for z in h] if any(not p for _,_,p in group) else None
        for k,z in enumerate(h):
            raw=tr.z[k]
            idx=torch.arange(len(at),device=at.device)
            torch.testing.assert_close(f.core.norm(raw),z[idx,at],rtol=3e-6,atol=3e-6)
            if not physical:
                continue
            assert readout(f,raw).argmax(-1).tolist()==output(f.model,z,at).argmax(-1).tolist()
        for i,key in enumerate(group):
            a,v,p=key
            pick=dict(x={k:z[i] for k,z in tr.x.items()},z={k:z[i] for k,z in tr.z.items()})
            if physical:
                pick.update(ids=ids0[i],h1=h1[i,:len(ids0[i])],w={k:z[i] for k,z in tr.w.items()})
            if not p:
                # Score the same native fixed positions before any additive transfer.
                b=[z["tokens"][0] for z in rows[a]["native"][v]]
                q=[x[i] for x in original]
                assert b==q
            out[key]=pick
    return out


class Physical(Patch):
    def __init__(self,at,layers,delta,writes):
        super().__init__(at)
        self.layers=layers
        self.delta=delta
        self.writes=writes

    def resid(self,k,l,x):
        group=self.layers.get(l)
        if group is None:
            return x
        y=x.clone()
        y[group,self.pos[group]]+=self.delta[group]
        return y

    def write(self,key,name,value):
        given=self.writes.get((key[1],name))
        if given is None:
            return value
        idx,z=given
        out=value.clone()
        out[idx,self.pos[idx]]=z
        assert torch.equal(out[idx,self.pos[idx]],z)
        return out


def physical_check(f,entries,keys,delta,computed):
    errors=[]
    for p in (0,1):
        seq=[(i,entries[i],keys[i]) for i in range(len(entries)) if entries[i][1]==p]
        for off in range(0,len(seq),16):
            use=seq[off:off+16]
            ids,at=pad([e[0]["ids"] for _,e,_ in use])
            site=torch.tensor([k[1]-1 for _,_,k in use],device=ids.device)
            start=torch.nn.utils.rnn.pad_sequence([e[0]["h1"] for _,e,_ in use],batch_first=True) if p else None
            targets={k:torch.where(site==k)[0] for k in site.unique().tolist()}
            writes={}
            for l in range(min(targets),LAST+1):
                mask=torch.where(site<=l)[0]
                for name in BRANCH:
                    writes[l,name]=(mask,torch.stack([use[j][1][0]["w"][p,l,name] for j in mask.tolist()]))
            ix=[i for i,_,_ in use]
            tr=Trace(at,False)
            h=f.run(ids,1,start=start,edit=Physical(at,targets,delta[ix],writes),record=tr)[0]
            torch.testing.assert_close(tr.z[0],computed[ix],rtol=2e-4,atol=5e-4)
            same=output(f.model,h,at).argmax(-1)==readout(f,computed[ix]).argmax(-1)
            assert same.all()
            errors.extend((tr.z[0]-computed[ix]).abs().max(dim=-1).values.tolist())
    return errors


def step(f,rows,requests,audit=False):
    want={}
    for i,l,k,d,a,p in requests:
        u,_,v,_=counterpart(k,d)
        want[i,u,p]=None
        want[i,v,p]=None
        if a=="unrelated":
            want[i,"unrelated_"+u,p]=None
    snap=snapshots(f,rows,list(want),audit)
    x,y,z,aux=[],[],[],[]
    modes={}
    entries=[]
    for n,(i,l,k,d,a,p) in enumerate(requests):
        u,j,v,t=counterpart(k,d)
        e=snap[i,u,p]
        inside=e["x"][j,l]
        x.append(inside)
        y.append(snap[i,v,p]["x"][t,l]-inside)
        z.append(e["z"][j])
        entries.append((e,j))
        modes.setdefault(a,[]).append(n)
        if a=="unrelated":
            aux.append(snap[i,"unrelated_"+u,p]["x"][j,l]-inside)
    xs,delta,zs=map(torch.stack,(x,y,z))
    geo=modes.get("geometry",[])
    if geo:
        delta[geo]=shuffle(xs[geo],delta[geo])-xs[geo]
    ctrl=modes.get("unrelated",[])
    if ctrl:
        d=torch.stack(aux)
        size=d.norm(dim=-1,keepdim=True)
        assert (size>1e-8).all()
        q=d*(delta[ctrl].norm(dim=-1,keepdim=True)/size)
        torch.testing.assert_close(q.norm(dim=-1),delta[ctrl].norm(dim=-1),rtol=5e-5,atol=5e-6)
        delta[ctrl]=q
    after=zs+delta
    tokens=[]
    for x in after.split(96):
        tokens.extend(readout(f,x).argmax(-1).tolist())
    result={r[:-1]:p for r,p in zip(requests,tokens,strict=True)}
    err=physical_check(f,entries,requests,delta,after) if audit else []
    return result,err


def decode(f,tok,examples):
    current={(i,l,k,d,a):[] for i,e in enumerate(examples) for l in SITES for k in MODES
             if k=="phase" or e["permitted"] for d in DIRECTIONS for a in MODES[k]}
    done={}
    for _ in range(MAX_NEW):
        if not current:
            break
        x,_=step(f,examples,[(*k,tuple(s)) for k,s in current.items()])
        for k,s in list(current.items()):
            s.append(x[k])
            if terminated(tok,s):
                done[k]=current.pop(k)
    done.update(current)
    out=dict(zip(done,text_results(tok,list(done.values())),strict=True))
    return [{str(l):{k:{d:{a:out[i,l,k,d,a] for a in MODES[k]} for d in DIRECTIONS}
                     for k in MODES if k=="phase" or e["permitted"]} for l in SITES} for i,e in enumerate(examples)]


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    clock=timer()
    src=root/"PAIRED.json"
    digest=sha(src)
    examples=[prepare(tok,r) for r in load(src)["rows"]]
    ordered=[r["id"] for r in examples]
    by={k:sorted((r["id"] for r in examples if r["domain"]==k and r["permitted"]),
                 key=lambda x:short_hash("onset replay",x))[:8] for k in DOMAINS}
    audit_ids={i for v in by.values() for i in v}
    lock=root/"COHORT.json"
    plan=dict(paired_sha256=digest,phase=ordered,content=[r["id"] for r in examples if r["permitted"]],audit=by)
    if lock.exists():
        assert load(lock)==plan
    else:
        save(lock,plan)
    dst=root/"ONSET_RAW.json"
    old=load(dst) if dst.exists() else dict(rows=[],elapsed=0.)
    rows=old["rows"]
    if rows:
        assert old["paired_sha256"]==digest and old["cohort_sha256"]==sha(lock)
        assert [r["id"] for r in rows]==ordered[:len(rows)]
    for j in range(len(rows),len(examples),7):
        sample=examples[j:j+7]
        z=decode(f,tok,sample)
        for i,(q,p) in enumerate(zip(sample,z,strict=True)):
            item={k:q[k] for k in ("id","domain","target","old","example","native","permitted")}
            item["pred"]=p
            if item["id"] in audit_ids:
                pref={v:tuple(whitespace(tok,q["native"][v])) for v in ("original","swapped")}
                req=[(i,l,k,d,"authentic",pref[counterpart(k,d)[0]]) for l in SITES for k in MODES for d in DIRECTIONS]
                _,err=step(f,sample,req,audit=True)
                item["parity"]=dict(n=len(err),max_abs=max(err))
            rows.append(item)
        save(dst,dict(rows=rows,paired_sha256=digest,cohort_sha256=sha(lock),
                      elapsed=old["elapsed"]+timer()-clock,peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1)))
        print("lookup onset",len(rows),len(examples),flush=True)
    s=summarize(rows)
    save(root/"ONSET_SUMMARY.json",s)
    print("onset decision",s["decision"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)
