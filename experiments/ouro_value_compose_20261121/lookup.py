"""Independent query-key composition using the arithmetic attention-replay rule."""
import argparse
from pathlib import Path
from random import Random

import torch

from edge_stats import upper_binomial
from interchange import Patch, terminated
from native import Forward, load, output, pad, save, setup, sha
from origin_stats import paired, positive
from rule import MAX_NEW, greedy, text_results
from value_test import Trace, indices, shuffle

KEYS=(1,2,3,4)
ARMS=("native","donor","shuffle")
SEED=261123889


def template(key,table):
    entries=", ".join(f"{n}: {v}" for n,v in zip(KEYS,table,strict=True))
    return f"Query = {key}; Table = {entries}; Answer:"


PREFIX="Read the table and give the integer paired with the query key.\n"
for k,table in ((4,(853,371,742,168)),(2,(582,493,664,781)),(1,(347,916,825,519)),(3,(729,958,216,671))):
    PREFIX+=template(k,table)+f" {table[k-1]}\n"


def dataset():
    rng=Random(SEED)
    rows=[]
    expected=rng.sample(range(110,990),128)
    for i,y in enumerate(expected):
        a,b=rng.sample(KEYS,2)
        other=rng.sample([x for x in range(110,990) if x!=y],7)
        recipient=other[:3]
        recipient.insert(a-1,y)
        reference=other[3:]
        assert len({y,recipient[b-1],reference[b-1]})==3
        rows.append(dict(id=f"lookup-{i:03d}",receiver_key=a,donor_key=b,receiver=recipient,donor=reference,
                         target=y,mixed=recipient[b-1],copied=reference[b-1]))
    return dict(seed=SEED,prefix=PREFIX,rows=rows)


def encode(tok,key,values):
    sentence=PREFIX+template(key,values)
    z=tok(sentence,add_special_tokens=False,return_offsets_mapping=True)
    q=len(PREFIX)+len("Query = ")
    start=len(PREFIX)+len(f"Query = {key}; Table = ")
    fields=[tuple(x) for x in z["offset_mapping"]]
    positions=[fields.index((q,q+1))]
    for i in range(start,len(sentence)):
        if sentence[i].isdigit():
            positions.append(fields.index((i,i+1)))
    assert len(positions)==17
    return z["input_ids"],positions


class Replay(Patch):
    def __init__(self,at,trace,pick,mode,values):
        super().__init__(at)
        self.trace=trace
        self.pick=pick
        self.mode=mode
        self.donor=values

    def attn(self,key,a,v):
        l=key[1]
        probability=a.clone()
        probability[:,:,self.trace.first:]=self.trace.attention[l][self.pick]
        base=self.trace.values[l][self.pick]
        inserted=base.clone()
        for label in ("donor","shuffle"):
            j=[i for i,x in enumerate(self.mode) if x==label]
            if j:
                real=self.donor[l][j,:,:1]
                own=base[j,:,:1]
                inserted[j,:,:1]=real if label=="donor" else shuffle(own,real-own)
        where=indices(v,self.trace.positions[self.pick])
        return probability,v.scatter(2,where,inserted)


def step(f,entries,requests,audit):
    keys=list(requests)
    predictions={}
    for p in range(0,len(keys),12):
        pack=keys[p:p+12]
        sequences,at=pad([entries[i]["r"]+list(s) for i,s in pack])
        donor,atd=pad([entries[i]["d"]+list(s) for i,s in pack])
        position=torch.tensor([entries[i]["pos"] for i,s in pack],device=at.device)
        recv=Trace(position,at)
        recv.stage=1
        h,answer=f.run(sequences,2,record=recv)
        dst=Trace(position,atd,weights=False)
        dst.stage=1
        f.run(donor,2,record=dst)
        base=output(f.model,answer,at).argmax(-1).tolist()
        edit=[]
        for j,x in enumerate(pack):
            for arm in requests[x]:
                if arm=="native":
                    predictions[arm,x[0]]=base[j]
                else:
                    edit.append((j,arm,x[0]))
            if audit:
                edit.append((j,"parity",x[0]))
        if not edit:
            continue
        ids=[j for j,_,_ in edit]
        modes=[arm for _,arm,_ in edit]
        loc=at[ids]
        patch=Replay(loc,recv,ids,modes,{l:v[ids] for l,v in dst.values.items()})
        z=f.run(sequences[ids],1,start=h[ids],edit=patch)[0]
        nexts=output(f.model,z,loc).argmax(-1).tolist()
        for v,(j,arm,i) in enumerate(edit):
            if arm=="parity":
                assert base[j]==nexts[v]
                torch.testing.assert_close(z[v],answer[j],rtol=5e-5,atol=3e-4)
            else:
                predictions[arm,i]=nexts[v]
    return predictions


def decode(f,tok,entries,audit):
    pending={(a,i):[] for a in ARMS for i in range(len(entries))}
    out={}
    for n in range(MAX_NEW):
        if not pending:
            break
        work={}
        for (a,i),tokens in pending.items():
            work.setdefault((i,tuple(tokens)),[]).append(a)
        x=step(f,entries,work,audit and n==0)
        for key,t in list(pending.items()):
            t.append(x[key])
            if terminated(tok,t):
                out[key]=pending.pop(key)
    out.update(pending)
    values=dict(zip(out,text_results(tok,list(out.values())),strict=True))
    return [{a:values[a,i] for a in ARMS} for i in range(len(entries))]


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    deck=root/"LOOKUP_DECK.json"
    frozen=dataset()
    if deck.exists():
        assert load(deck)==frozen
    else:
        save(deck,frozen)
    entries=[]
    for row in frozen["rows"]:
        r,p=encode(tok,row["receiver_key"],row["receiver"])
        d,q=encode(tok,row["donor_key"],row["donor"])
        assert p==q and len(r)==len(d)
        entries.append(dict(row,r=r,d=d,pos=p))
    for j in range(0,len(entries),32):
        e=entries[j:j+32]
        for kind in ("r","d"):
            bank=[greedy(f,tok,[x[kind] for x in e],k) for k in (1,2)]
            for q,b in zip(e,zip(*bank,strict=True),strict=True):
                q[kind+"_native"]=list(b)
    eligible=[r for r in entries if r["r_native"][1]["number"]==r["target"]]
    out=dict(n=len(entries),receiver_native=[sum(x["r_native"][k]["number"]==x["target"] for x in entries) for k in (0,1)],
             natural_donor=[sum(x["d_native"][k]["number"]==x["copied"] for x in entries) for k in (0,1)],
             deck_sha256=sha(deck),gate=len(eligible)>=64)
    records=[]
    if out["gate"]:
        for j in range(0,len(eligible),6):
            grp=eligible[j:j+6]
            pred=decode(f,tok,grp,j==0)
            for t,p in zip(grp,pred,strict=True):
                assert p["native"]==t["r_native"][1]
                records.append(dict(id=t["id"],target=t["target"],copied=t["copied"],mixed=t["mixed"],pred=p))
            print("lookup exact composition",len(records),len(eligible),flush=True)
        counts={a:{k:sum(r["pred"][a]["number"]==r[k] for r in records) for k in ("target","mixed","copied")}
                for a in ARMS}
        flags=[int(x["pred"]["donor"]["number"]==x["mixed"])-int(x["pred"]["shuffle"]["number"]==x["mixed"]) for x in records]
        z=paired(records,flags)
        out.update(n_intervened=len(records),counts=counts,paired=z,
                   upper_97_5=upper_binomial(sum(x>0 for x in flags),len(flags)),
                   supported=counts["donor"]["mixed"]>=8 and positive(z))
    save(root/"LOOKUP_RAW.json",dict(native=[{k:v for k,v in x.items() if k not in ("r","d","pos")} for x in entries],
                                    interventions=records))
    out["raw_sha256"]=sha(root/"LOOKUP_RAW.json")
    save(root/"LOOKUP_SUMMARY.json",out)
    print("lookup summary",out,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)
