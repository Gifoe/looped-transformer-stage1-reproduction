"""Isolate K1 contextual and pre-digit attention routes while replaying native K2."""
import argparse
from pathlib import Path

import torch

from block import matched
from edge import locations, short_hash
from interchange import Patch, complete, fit_margin
from native import Forward, load, output, pad, save, setup, sha, timer
from origin import check_early_prefix, decompose
from path_stats import ARMS, PATHS, summarize
from rule import greedy


class Layout:
    def __init__(self,tok,pairs,prefix):
        assets=[locations(tok,p,w) for p,w in zip(pairs,prefix,strict=True)]
        self.inputs=[a[0] for a in assets]
        self.ids,self.at=pad(self.inputs)
        self.rows=torch.arange(len(assets),device=self.ids.device)
        begin=[]
        for tokens,q,_,_ in assets:
            assert len(tokens)>q and all(tok.decode([x]).isdigit() for x in tokens[q-2:q+1])
            begin.append(q-2)
        self.begin=torch.tensor(begin,device=self.ids.device)
        self.low=torch.tensor([a[2] for a in assets],device=self.ids.device)
        previous=assets[0][3]
        assert all(z[3]==previous for z in assets)
        self.unrelated=torch.zeros(self.ids.shape[1],dtype=torch.bool,device=self.ids.device)
        self.unrelated[previous]=True
        p=torch.arange(self.ids.shape[1],device=self.ids.device)[None,:]
        self.context=(p>=self.begin[:,None]) & (p<self.at[:,None])
        self.pivot=p==self.at[:,None]
        self.before=p<self.begin[:,None]
        assert (self.context.sum(1)>0).all()
        self.sites={"context":self.context,"pivot":self.pivot,"joint":self.context|self.pivot}


class Route(Patch):
    def __init__(self,layout,which,coupled=False):
        super().__init__(layout.at)
        self.layout=layout
        self.coupled=coupled
        self.rows,self.queries=torch.where(layout.sites[which])
        self.history=[]
        self.full_mass=[]

    def attn(self,key,a,v):
        assert key[0]==0
        p=self.layout
        n=len(p.at)
        assert a.shape[0]==n*(2 if self.coupled else 1)
        i,q=self.rows,self.queries
        r=a[i,:,q,:]
        keys=p.low[i,None,:].expand(-1,r.shape[1],-1)
        selected=r.gather(-1,keys)
        before=selected.sum(-1,keepdim=True)
        if self.coupled:
            c=a[i+n,:,q,:]
            bag=(c*p.unrelated).sum(-1,keepdim=True)
            drop=torch.minimum(before,bag)
            selected=selected*(1-drop/before.clamp_min(1e-20))
            original=r.gather(-1,keys)
            s=r.scatter(-1,keys,selected)
            t=c*(1-p.unrelated*(drop/bag.clamp_min(1e-20)))
            a0=(original-selected).sum(-1)
            a1=((c-t)*p.unrelated).sum(-1)
            torch.testing.assert_close(a0,a1,rtol=4e-6,atol=4e-7)
        else:
            drop=before
            s=r.scatter(-1,keys,0.)
        mass=drop.squeeze(-1).sum(-1)
        candidates=before.squeeze(-1).sum(-1)
        zeros=r.new_zeros(n)
        self.history.append(zeros.index_add(0,i,mass))
        self.full_mass.append(zeros.index_add(0,i,candidates))
        selected_left=s.sum(-1,keepdim=True)
        assert selected_left.min().item()>1e-12
        z=a.clone()
        z[i,:,q,:]=s/selected_left
        if self.coupled:
            intact=t.sum(-1,keepdim=True)
            assert intact.min().item()>1e-12
            z[i+n,:,q,:]=t/intact
        return z,v

    def receipt(self):
        assert len(self.history)==24
        return dict(removed=torch.stack(self.history).sum(0).tolist(),
                    available=torch.stack(self.full_mass).sum(0).tolist())


def edited(f,layout,which,coupled=False):
    e=Route(layout,which,coupled)
    ids=layout.ids.repeat(2,1) if coupled else layout.ids
    h=f.run(ids,1,edit=e)[0]
    return h,e.receipt()


def resolve(f,tok,cases,states,at,depth):
    chosen={a:output(f.model,z,at).argmax(-1).tolist() for a,z in states.items()}
    mixed=list(dict.fromkeys((i,v) for vs in chosen.values() for i,v in enumerate(vs)))
    finished=complete(f,tok,[cases[i] for i,_ in mixed],[v for _,v in mixed],depth-1)
    by=dict(zip(mixed,finished,strict=True))
    return {a:[by[i,v] for i,v in enumerate(vs)] for a,vs in chosen.items()}


@torch.inference_mode()
def evaluate(f,tok,cases,operands):
    layout=Layout(tok,operands,[c["prefix"] for c in cases])
    ids,at=layout.ids,layout.at
    assert layout.inputs==[c["ids"]+c["prefix"] for c in cases]
    native1,native2=f.run(ids,2)
    assert torch.equal(f.run(ids,1,start=native1)[0],native2)
    states={"native":native2}
    receipts={}
    cut=None
    for p in PATHS:
        h,r=edited(f,layout,p)
        states[p]=f.run(ids,1,start=h)[0]
        if p=="joint":
            cut=h
        pair,m=edited(f,layout,p,coupled=True)
        tx,ctl=f.run(ids.repeat(2,1),1,start=pair)[0].chunk(2,dim=0)
        states[p+"_capped"],states[p+"_matched"]=tx,ctl
        receipts[p]={"full":r["removed"],"matched":m["removed"],"available":m["available"]}
    assert cut is not None
    assert torch.equal(cut[layout.before],native1[layout.before])
    restored={}
    for which in PATHS:
        patched=torch.where(layout.sites[which][...,None],native1,cut)
        restored[which]=f.run(ids,1,start=patched)[0]
        if which=="joint":
            valid=layout.before|layout.sites["joint"]
            assert torch.equal(patched[valid],native1[valid])
            assert torch.equal(output(f.model,restored[which],at),output(f.model,native2,at))
        states["restore_"+which]=restored[which]
    assert set(states)==set(ARMS)
    p=resolve(f,tok,cases,states,at,2)
    assert p["native"]==p["restore_joint"]==[c["native"][1] for c in cases]
    w=torch.tensor([c["labels"] for c in cases],device=at.device)
    margins={a:fit_margin(f,z,at,w).tolist() for a,z in states.items()}
    z=[]
    for i,c in enumerate(cases):
        z.append(dict(pred={a:v[i] for a,v in p.items()},margin={a:v[i] for a,v in margins.items()},
                      masses={a:{k:v[i] for k,v in r.items()} for a,r in receipts.items()},
                      query_counts={a:int(layout.sites[a][i].sum().item()) for a in PATHS},
                      pivot_shift=(cut[i,at[i]]-native1[i,at[i]]).norm().item()))
    return z


@torch.inference_mode()
def internal(f,tok,cases,pairs):
    flat=[a for q in pairs for a in q]
    pref=[c["prefix"] for c in cases for _ in range(4)]
    lay=Layout(tok,flat,pref)
    native1=f.run(lay.ids,1)[0]
    full,full_meta=edited(f,lay,"joint")
    pair,pair_meta=edited(f,lay,"joint",coupled=True)
    a,b=pair.chunk(2)
    vals={}
    for named,incoming in (("native",native1),("joint",full),("joint_capped",a),("joint_matched",b)):
        z,_,s,info,_=decompose(f,lay.ids,lay.at,start=incoming)
        vals[named]={"parts":s["full"],"out_of_envelope":info["outside"]}
    tags={1:"propagated",2:"generated_attention",3:"generated_mlp"}
    fields={}
    for named,d in vals.items():
        fields[named]={}
        for k,a in tags.items():
            v=d["parts"][k]
            fields[named][a+"_norm"]=v.norm(dim=-1).tolist()
            fields[named][a+"_shift"]=(v-vals["native"]["parts"][k]).norm(dim=-1).tolist()
        fields[named]["out_of_envelope"]=d["out_of_envelope"]
    return [dict(components={a:{k:v[i] for k,v in d.items()} for a,d in fields.items()})
            for i in range(len(cases))]


@torch.inference_mode()
def deeper(f,tok,cases,operands):
    lay=Layout(tok,operands,[c["prefix"] for c in cases])
    ids,at=lay.ids,lay.at
    natural=f.run(ids,4)
    continue_clean=f.run(ids,3,start=natural[0])
    assert all(torch.equal(a,b) for a,b in zip(natural[1:],continue_clean,strict=True))
    outcomes={"native":natural}
    for p in PATHS:
        h,_=edited(f,lay,p)
        outcomes[p]=(h,*f.run(ids,3,start=h))
    out=[{"depth":{}} for _ in cases]
    for k in (3,4):
        native=greedy(f,tok,[c["ids"] for c in cases],k)
        use=[i for i,(q,c) in enumerate(zip(native,cases,strict=True))
             if len(q["tokens"])>len(c["prefix"]) and q["tokens"][:len(c["prefix"])]==c["prefix"]]
        for i,n in enumerate(native):
            out[i]["depth"][str(k)]={"aligned":False,"pred":{a:n for a in outcomes}}
        if use:
            several=[dict(cases[i],native=cases[i]["native"]+[None]*(k-3)+[native[i]],
                          reach=cases[i]["reach"]+[False]*(k-3)+[True]) for i in use]
            subset={a:z[k-1][use] for a,z in outcomes.items()}
            assert output(f.model,subset["native"],at[use]).argmax(-1).tolist()==[
                native[i]["tokens"][len(cases[i]["prefix"])] for i in use]
            complete_arms=resolve(f,tok,several,subset,at[use],k)
            for j,i in enumerate(use):
                out[i]["depth"][str(k)]={"aligned":True,"pred":{a:z[j] for a,z in complete_arms.items()}}
    return out


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    started=timer()
    nh=sha(root/"NATIVE.json")
    lookup={r["id"]:r for r in load(root/"NATIVE.json")["rows"]}
    raw=root/"PATH_RAW.json"
    existing=load(raw) if raw.exists() else {"rows":[],"elapsed":0.}
    rows=existing["rows"]
    if rows:
        assert existing["native_sha256"]==nh
    plan=matched(f,tok,root,"confirmation")
    assert [v["id"] for v in rows[::2]]==[r["g"]["id"] for r,_,_,_ in plan[:len(rows)//2]]
    pick=sorted((r["g"]["id"] for r,_,_,_ in plan),key=lambda i:short_hash("routes",i))
    watch=set(pick[:24])
    see=set(pick[:48])
    rules=root/"COHORT.json"
    lock=dict(natural_sha256=nh,component_ids=sorted(watch),deeper_ids=sorted(see),
              pairs=[dict(id=r["g"]["id"],safe=s["id"],margin=ma,difference=di) for r,s,ma,di in plan])
    if rules.exists():
        assert load(rules)==lock
    else:
        save(rules,lock)
    for i in range(len(rows)//2,len(plan),4):
        group=plan[i:i+4]
        candidates=[r["aligned"] and all(s["reach"]) for r,s,_,_ in group]
        supp=[c for (r,s,_,_),ok in zip(group,candidates,strict=True) if ok for c in (r["pair"][0],s)]
        early=iter(check_early_prefix(f,supp) if supp else [])
        flags=[all((next(early),next(early))) if q else False for q in candidates]
        cases=[c for (r,s,_,_),ok in zip(group,flags,strict=True) if ok for c in (r["pair"][0],s)]
        pts=[]
        for c in cases:
            r=lookup[c["id"]]
            pts.append(r["borrow" if c["kind"]=="repair" else "safe"][3])
        primary=iter(evaluate(f,tok,cases,pts) if cases else [])
        a=[j for j,c in enumerate(cases) if c["kind"]=="repair" and c["id"] in watch]
        comp=internal(f,tok,[cases[j] for j in a],[lookup[cases[j]["id"]]["borrow"] for j in a]) if a else []
        layer={cases[j]["id"]:v for j,v in zip(a,comp,strict=True)}
        b=[j for j,c in enumerate(cases) if c["kind"]=="repair" and c["id"] in see]
        depths=deeper(f,tok,[cases[j] for j in b],[pts[j] for j in b]) if b else []
        compare={cases[j]["id"]:v for j,v in zip(b,depths,strict=True)}
        for (r,s,_,distance),ok in zip(group,flags,strict=True):
            for c in (r["pair"][0],s):
                value=next(primary) if ok else dict(pred={k:c["native"][1] for k in ARMS})
                extra=dict(layer.get(c["id"],{}),**compare.get(c["id"],{})) if c["kind"]=="repair" else {}
                rows.append(dict(id=r["g"]["id"],kind=c["kind"],source_id=c["id"],target=c["target"],foil=c["foil"],
                                 aligned=ok,distance=distance,native=c["native"],**value,**extra))
        save(raw,dict(rows=rows,native_sha256=nh,cohort_sha256=sha(rules),
                      elapsed=existing["elapsed"]+timer()-started,
                      peak_mib=round(torch.cuda.max_memory_allocated()/2**20,1)))
        print("all-layer routes",len(rows)//2,len(plan),flush=True)
    z=summarize(rows)
    save(root/"PATH_SUMMARY.json",z)
    print("route gates",z["gates"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)
