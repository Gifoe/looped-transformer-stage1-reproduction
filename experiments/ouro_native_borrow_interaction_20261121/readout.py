"""Native four-corner readout states and paired exact behavioral effects."""
import argparse
import hashlib
from pathlib import Path

import torch

from borrow import ci
from graft import encode
from interchange import complete, independent_like, text_case
from mechanism import Edit, LR, matched_plane, same_norm
from native import Forward, load, output, pad, save, setup, sha
from native_deck import all_other_K1_correct, is_repair
from rule import greedy

MODES=("native","whole","delta","geometry","safe_delta","I2","distant","distant_safe","aligned")


class State:
    def __init__(self,at):
        self.at=at
        self.index=torch.arange(len(at),device=at.device)
        self.r={}

    def put(self,key,label,x):
        if key[1]==LR and label=="input":
            self.r[key[0]]=x[self.index,self.at].detach()


def to_old(g):
    out=dict(g,pairs=[g["safe"][3],g["borrow"][3]],
             pred=[[p[4],p[3]] for p in g["pred"]])
    return out


def forms(tok,g):
    b=[]
    for i in (1,0):
        z=text_case(tok,to_old(g),i)
        cols=g["borrow"] if i else g["safe"]
        packed=[encode(tok,p) for p in cols]
        assert packed[-1][0]==z["ids"]
        z.update(quad=[p[0] for p in packed],quad_fields=[p[1] for p in packed])
        b.append(z)
    q,w=b
    white=not tok.decode(q["prefix"],skip_special_tokens=True).strip()
    cur=all(q["reach"]) and all(w["reach"]) and white and q["prefix"]==w["prefix"]
    for p in g["pred"]:
        cur=cur and all(o["tokens"][:len(q["prefix"])]==q["prefix"] for o in p[:4])
    return dict(g=g,pair=b,aligned=bool(cur))


def core(f,pack,record=State):
    for j in range(max(len(c["prefix"]) for c in pack)):
        active=[(z[:]+c["prefix"][:j],c["prefix"][j]) for c in pack if len(c["prefix"])>j for z in c["quad"]]
        seq,last=pad([x for x,_ in active])
        expected=[v for _,v in active]
        for depth in (1,2):
            result=output(f.model,f.run(seq,depth)[-1],last).argmax(-1).tolist()
            assert result==expected
    ids,p=pad([x+c["prefix"] for c in pack for x in c["quad"]])
    z=record(p)
    h=f.run(ids,2,record=z)
    assert torch.equal(f.run(ids,1,start=h[0])[0],h[1])
    return ids,p,z,h


def mixed(v):
    y=v.view(-1,4,v.shape[-1])
    return y[:,3]-y[:,1]-y[:,2]+y[:,0]


def sealed_match(bank,rec):
    g=rec["g"]
    key=lambda x:hashlib.sha256((g["id"]+"|"+x["g"]["id"]+"|261122110").encode()).hexdigest()
    ring=sorted((b for b in bank if [len(x["ids"])+len(x["prefix"]) for x in b["pair"]]
                 ==[len(x["ids"])+len(x["prefix"]) for x in rec["pair"]]),key=key)
    t=g["target"]//100
    afar=next((z for z in ring if abs(z["g"]["target"]//100-t)>=2),None)
    close=next((z for z in ring if z["g"]["target"]//100==t and z["g"]["target"]!=g["target"]),None)
    return [b["g"]["id"] if b else None for b in (afar,close)]


def plan(tok,rows):
    d=[]
    q=[]
    for g in sorted(rows,key=lambda v:v["id"]):
        if is_repair(g):
            z=forms(tok,g)
            if g["split"]=="discovery":
                if z["aligned"] and all(v[4]["number"]==g["target"] for v in g["pred"]) and all(z["pair"][1]["reach"]):
                    d.append(z)
            else:
                q.append(z)
    return d,q,[sealed_match(d,z) if z["aligned"] else [None,None] for z in q]


def donor_trace(f,donors):
    table={}
    for i in range(0,len(donors),5):
        pack=[c for q in donors[i:i+5] for c in q["pair"]]
        _,_,frozen,_=core(f,pack)
        a,b=[mixed(frozen.r[k]) for k in (0,1)]
        for j,q in enumerate(pack):
            table[q["id"],q["kind"]]=b[j]-a[j]
    return table


def raw_case(c):
    return {k:c[k] for k in ("id","target","foil","kind","reach","prefix")}


def verify_decodes(f,tok,pack):
    for k in (0,1):
        predicted=greedy(f,tok,[c["ids"] for c in pack],k+1)
        assert [z["tokens"] for z in predicted]==[c["native"][k]["tokens"] for c in pack]


def effects(f,tok,ready,ident,bank):
    packed=[c for row in ready for c in row["pair"]]
    seq,p,frozen,h=core(f,packed)
    selected=torch.arange(3,len(seq),4,device=seq.device)
    ids,at=seq[selected],p[selected]
    basic=[frozen.r[k][selected] for k in (0,1)]
    parts=[mixed(frozen.r[k]) for k in (0,1)]
    delta=parts[1]-parts[0]
    rnd=independent_like(delta,packed,LR,400)
    geom=matched_plane(basic[0],basic[1]-basic[0],delta,rnd)
    norms={}
    values={}
    safe_sources=torch.repeat_interleave(delta[1::2],2,dim=0)
    values["safe_delta"],_=same_norm(safe_sources,delta)
    for method in ("distant","distant_safe","aligned"):
        query=0 if method.startswith("distant") else 1
        kind="safe" if method.endswith("safe") else "repair"
        handle=[m[query] for m in ident for _ in (0,1)]
        direction=torch.stack([bank[q,kind] if q else torch.zeros_like(delta[k]) for k,q in enumerate(handle)])
        values[method],_=same_norm(direction,delta)
    values.update(delta=delta,geometry=geom,I2=parts[1])
    for k,arr in values.items():
        norms[k]=arr.norm(dim=-1).tolist()
    verify_decodes(f,tok,packed)
    fate={}
    for name in MODES:
        data=[]
        for k in (0,1):
            if name=="native":
                result=h[k][selected]
            else:
                vec=basic[1-k] if name=="whole" else basic[k]+values[name]*(1 if k==0 else -1)
                result=f.run(ids,1,start=h[0][selected] if k else None,edit=Edit(at,readout=vec))[0]
            z=output(f.model,result,at).argmax(-1).tolist()
            if name=="native":
                assert all(not c["reach"][k] or c["native"][k]["tokens"][len(c["prefix"])]==p
                           for c,p in zip(packed,z,strict=True))
            data.append(complete(f,tok,packed,z,k))
        fate[name]=list(zip(*data,strict=True))
    return fate,norms


def result_cell(c,pairs,norm,active,tight):
    val=raw_case(c)
    if not active:
        pairs={m:c["native"] for m in MODES}
    assert list(pairs["native"])==c["native"]
    val.update(pred=pairs,norm=norm,aligned=active,tight=tight)
    return val


def summary(rows,lookup):
    out={}
    for kind in ("repair","safe"):
        pool=[r for r in rows if r["kind"]==kind]
        detail={}
        for mode in MODES:
            cells=[]
            for r in pool:
                a,b=[z["number"] for z in r["pred"][mode]]
                y,x=r["target"],r["foil"]
                v=r["pred"]["native"]
                donor=lookup[r["id"]][0]
                dy=int(donor.split("-")[1]) if donor else None
                cells.append((a==y,b==x,(a,b)==(y,x),r["tight"] and (a,b)==(y,x),
                              v[0]["number"]==y and a!=y,v[1]["number"]==y and b!=y,
                              dy is not None and a in (dy,dy+100),dy is not None and b in (dy,dy+100)))
            k=["K1_exact","K2_foil","joint","tight_joint","K1_harm","K2_harm","K1_donor_collision","K2_donor_collision"]
            detail[mode]=dict(zip(k,map(sum,zip(*cells,strict=True)),strict=True))
            detail[mode]["distinct_joint"]=len({r["target"] for r,hit in zip(pool,cells,strict=True) if hit[2]})
        out[kind]=dict(n=len(pool),aligned=sum(r["aligned"] for r in pool),
                       tight=sum(r["tight"] for r in pool),modes=detail)
    todo=[x for x in rows if x["kind"]=="repair"]
    paired={}
    for a,b in (("delta","geometry"),("delta","safe_delta"),("distant","geometry"),("distant","distant_safe")):
        table={q:{} for q in ("K1","K2","joint")}
        for r in todo:
            t=r["target"]
            f=r["foil"]
            flags=[]
            for m in (a,b):
                one,two=[z["number"] for z in r["pred"][m]]
                flags.append([one==t,two==f,one==t and two==f])
            for j,k in enumerate(table):
                pool=table[k]
                pool[t]=pool.get(t,0)+int(flags[0][j])-int(flags[1][j])
        paired[a+"_vs_"+b]={k:ci(list(v.values()),(.0125,.9875)) for k,v in table.items()}
    same=out["repair"]["modes"]["delta"]
    other=out["repair"]["modes"]["distant"]
    own=all(x["interval"][0]>0 for k,v in paired.items() if k.startswith("delta") for x in v.values())
    foreign=all(v["joint"]["interval"][0]>0 for k,v in paired.items() if k.startswith("distant"))
    out.update(comparisons=paired,same_gate=same["joint"]>=10 and same["distinct_joint"]>=8 and own,
               disjoint_gate=other["joint"]>=8 and other["distinct_joint"]>=6 and foreign and
               5*sum(v[0] is not None for v in lookup.values())>=4*len(lookup))
    return out


@torch.inference_mode()
def run(model,root):
    assert load(root/"NATIVE_SUMMARY.json")["gate"]
    m,tok=setup(model)
    f=Forward(m)
    raw=root/"NATIVE.json"
    before=load(raw)["rows"]
    bank,sampled,which=plan(tok,before)
    matches={r["g"]["id"]:p for r,p in zip(sampled,which,strict=True)}
    save(root/"MATCHING.json",dict(bank=[r["g"]["id"] for r in bank],targets=list(matches),matches=matches,native_sha256=sha(raw)))
    d=donor_trace(f,bank)
    i=root/"READOUT_RAW.json"
    old=load(i)["rows"] if i.exists() else []
    assert [(x["id"],x["kind"]) for x in old]==[(z["id"],z["kind"]) for s in sampled for z in s["pair"]][:len(old)]
    for k in range(len(old)//2,len(sampled),5):
        block=sampled[k:k+5]
        idx=[j for j,r in enumerate(block) if r["aligned"]]
        out={}
        if idx:
            choose=[block[j] for j in idx]
            zz,ll=effects(f,tok,choose,[matches[h["g"]["id"]] for h in choose],d)
            for loc,j in enumerate(idx):
                out[j]=[(dict((m,z[2*loc+t]) for m,z in zz.items()),
                         dict((m,z[2*loc+t]) for m,z in ll.items())) for t in (0,1)]
        for j,x in enumerate(block):
            for t,z in enumerate(x["pair"]):
                pred,norm=out[j][t] if j in out else ({}, {})
                old.append(result_cell(z,pred,norm,j in out,all_other_K1_correct(x["g"])))
        save(i,dict(rows=old,matching_sha256=sha(root/"MATCHING.json"),native_sha256=sha(raw)))
        print("readout interaction",len(old)//2,len(sampled),flush=True)
    ans=summary(old,matches)
    save(root/"READOUT_SUMMARY.json",ans)
    print(ans,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

