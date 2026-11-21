"""Resolve the full generated integer under recurrent boundary and read interventions."""
import argparse
from pathlib import Path

import torch

from block import matched
from boundary_stats import ARMS, MODES, summarize
from edge import locations, short_hash
from interchange import terminated
from native import Forward, load, output, pad, save, setup, sha, timer
from path_test import Route
from rule import MAX_NEW, greedy, text_results


def layout_case(tok, receiving, item, row, kind, diag):
    family="borrow" if kind=="repair" else "safe"
    if kind=="repair":
        source={k:row["donors"][k] for k in ("opposite","same")}
    else:
        source={k:row["donors"]["safe"] for k in ("opposite","same")}
    pairs=dict(receiver=row[family][3],**source)
    entries={k:locations(tok,v,[]) for k,v in pairs.items()}
    ids,last,low,prior=entries["receiver"]
    assert ids==item["ids"]
    operands=[low[0]-1,*low,last-2,last-1,last]
    for k,(ids2,q,keys,available) in entries.items():
        assert (q,keys,available)==(last,low,prior) and len(ids2)==len(ids)
        assert all(tok.decode([ids2[p]]).isdigit() for p in operands)
        assert all(a==b for j,(a,b) in enumerate(zip(ids,ids2,strict=True)) if j not in operands)
    return dict(id=receiving["id"],source_id=row["id"],kind=kind,target=receiving["target"],
                foil=receiving["foil"],delta=row["delta"],role=row["role"],native=item["native"],
                prefix=item["prefix"],diagnostic=diag, operand=operands,prior=prior,
                ids={k:x[0] for k,x in entries.items()},
                donor_values={k:b-a for k,(a,b) in source.items()})


class Layout:
    def __init__(self,cases,seq):
        self.ids,self.at=pad(seq)
        self.low=torch.tensor([c["operand"] for c in cases],device=self.ids.device)
        n,t=self.ids.shape
        cols=torch.arange(t,device=self.ids.device)[None,:]
        is_operand=torch.zeros((n,t),device=self.ids.device,dtype=torch.bool)
        is_operand.scatter_(1,self.low,True)
        from_first=cols>=self.low[:,:1]
        self.sites={"joint":from_first & (cols<=self.at[:,None]) & ~is_operand}
        self.unrelated=torch.zeros(t,device=self.ids.device,dtype=torch.bool)
        assert all(c["prior"]==cases[0]["prior"] for c in cases)
        self.unrelated[cases[0]["prior"]]=True


class Observe:
    def __init__(self,at):
        self.at=at
        self.rows=torch.arange(len(at),device=at.device)
        self.values={}

    def put(self,key,name,data):
        if key==(0,14) and name in ("a_write","m_write","output"):
            self.values[name]=data[self.rows,self.at].detach()


def pack_k1(f,cases,histories):
    histories=list(histories)
    out={}
    for start in range(0,len(histories),8):
        group=histories[start:start+8]
        ids={((i,p),tag):tuple(seq)+p for i,p in group for tag,seq in cases[i]["ids"].items()}
        distinct=list(dict.fromkeys(ids.values()))
        built,_=pad([list(p) for p in distinct])
        result=f.run(built,1)[0]
        hidden={p:result[i,:len(p)] for i,p in enumerate(distinct)}
        for i,p in group:
            c=cases[i]
            row=hidden[ids[(i,p),"receiver"]]
            locate=c["operand"]
            out[i,p,"receiver"]=row
            for tag in ("opposite","same"):
                donor=hidden[ids[(i,p),tag]]
                torch.testing.assert_close(row[:locate[0]],donor[:locate[0]],rtol=3e-5,atol=2e-4)
                mixed=donor.clone()
                mixed[:locate[0]]=row[:locate[0]]
                mixed[locate]=row[locate]
                assert torch.equal(mixed[locate],row[locate]) and torch.equal(mixed[:locate[0]],row[:locate[0]])
                out[i,p,tag]=mixed
                if tag=="opposite":
                    out[i,p,"whole_donor"]=donor
    return out


def calc(f,cases,queries,record=False):
    tasks={key:{} for key in ("live","full","equal")}
    contexts={}
    for arm,i,suffix in queries:
        ctx,mode,lane=MODES[arm]
        if cases[i]["kind"]=="safe" and ctx=="same":
            ctx="opposite"
        key=(i,suffix,ctx)
        tasks[mode].setdefault(key,[]).append((arm,i,lane))
        contexts[i,suffix]=None
    bank=pack_k1(f,cases,contexts)
    answer={}
    doses={}
    probes={}
    for mode,which in tasks.items():
        keys=list(which)
        step=10 if mode=="equal" else 20
        for start in range(0,len(keys),step):
            active=keys[start:start+step]
            entries=[cases[i] for i,_,_ in active]
            seq=[c["ids"]["receiver"]+list(p) for c,(_,p,_) in zip(entries,active,strict=True)]
            shape=Layout(entries,seq)
            state=torch.nn.utils.rnn.pad_sequence([bank[key] for key in active],batch_first=True)
            ids,at=shape.ids,shape.at
            intervention=None
            if mode!="live":
                coupled=mode=="equal"
                intervention=Route(shape,"joint",coupled=coupled)
                if coupled:
                    ids,at,state=ids.repeat(2,1),at.repeat(2),state.repeat(2,1,1)
            watch=Observe(at) if record else None
            h=f.run(ids,1,start=state,edit=intervention,record=watch)[0]
            decide=output(f.model,h,at).argmax(-1).tolist()
            amounts=intervention.receipt() if intervention is not None else None
            for j,key in enumerate(active):
                for arm,i,lane in which[key]:
                    ix=j+lane*len(active)
                    answer[arm,i]=decide[ix]
                    if amounts is not None:
                        doses[arm,i]=[amounts["removed"][j],amounts["available"][j]]
                    if watch is not None:
                        probes[arm,i]={a:z[ix] for a,z in watch.values.items()}
    return answer,doses,probes


def branches(f,tok,cases):
    active={(a,i):[] for a in ARMS for i in range(len(cases))}
    result={}
    track={}
    for _ in range(MAX_NEW):
        if not active:
            break
        requests=[(a,i,tuple(seq)) for (a,i),seq in active.items()]
        nexts,amounts,_=calc(f,cases,requests)
        for key,tokens in list(active.items()):
            tokens.append(nexts[key])
            if key in amounts:
                x,y=amounts[key]
                u,v=track.get(key,(0.,0.))
                track[key]=(u+x,v+y)
            if terminated(tok,tokens):
                result[key]=active.pop(key)
    result.update(active)
    ids=list(result)
    out=dict(zip(ids,text_results(tok,[result[k] for k in ids]),strict=True))
    return [{a:out[a,i] for a in ARMS} for i in range(len(cases))],track


def assess(f,tok,cases):
    ids,at=pad([c["ids"]["receiver"] for c in cases])
    h1,h2=f.run(ids,2)
    assert torch.equal(f.run(ids,1,start=h1)[0],h2)
    pred,amounts=branches(f,tok,cases)
    donors={}
    for name in ("opposite","same"):
        x=[c["ids"][name] for c in cases]
        donors[name]=[greedy(f,tok,x,k) for k in (1,2)]
    diagnostic=[j for j,c in enumerate(cases) if c["diagnostic"]]
    chosen=("native","original_full","opposite_intact","opposite_full","opposite_capped",
            "opposite_matched","same_intact","same_full")
    q=[(a,j,tuple(cases[j]["prefix"])) for j in diagnostic for a in chosen]
    _,_,values=calc(f,cases,q,record=True) if q else ({},{},{})
    rows=[]
    for j,c in enumerate(cases):
        d={k:[a[j] for a in v] for k,v in donors.items()}
        assert pred[j]["native"]==c["native"][1]
        assert pred[j]["donor_replay"]==d["opposite"][1]
        info=dict(pred=pred[j],donor_native=d,
                  mass={a:list(amounts[a,j]) for a in ARMS if (a,j) in amounts})
        if j in diagnostic:
            baseline=values["native",j]
            info["component_diagnostics"]={a:{k:{"shift":(z-baseline[k]).norm().item(),"norm":z.norm().item()}
                                             for k,z in values[a,j].items()} for a in chosen}
        rows.append(info)
    return rows


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    clock=timer()
    native=root/"NATIVE.json"
    nh=sha(native)
    by={r["id"]:r for r in load(native)["rows"]}
    stops=matched(f,tok,root,"confirmation")
    wanted=set(sorted((r["g"]["id"] for r,_,_,_ in stops),key=lambda s:short_hash("k2 boundary",s))[:24])
    file=root/"COHORT.json"
    plan=dict(native_sha256=nh,diagnostic_ids=sorted(wanted),
              pairs=[dict(id=r["g"]["id"],safe=s["id"],margin=list(z),difference=d) for r,s,z,d in stops])
    if file.exists():
        assert load(file)==plan
    else:
        save(file,plan)
    raw=root/"BOUNDARY_RAW.json"
    old=load(raw) if raw.exists() else {"rows":[],"elapsed":0.}
    result=old["rows"]
    if result:
        assert old["native_sha256"]==nh and old["cohort_sha256"]==sha(file)
        assert [p["id"] for p in result[::2]]==[r["g"]["id"] for r,_,_,_ in stops[:len(result)//2]]
    for start in range(len(result)//2,len(stops),4):
        opts=stops[start:start+4]
        batch=[]
        for z,s,_,di in opts:
            g=z["g"]
            case=layout_case(tok,g,z["pair"][0],g,"repair",g["id"] in wanted and z["aligned"])
            safe=layout_case(tok,g,s,by[s["id"]],"safe",False)
            for c in (case,safe):
                c["distance"]=di
                batch.append(c)
        data=assess(f,tok,batch)
        for a,b in zip(batch,data,strict=True):
            result.append(dict({k:v for k,v in a.items() if k not in ("ids","prefix","prior")},**b))
        save(raw,dict(rows=result,native_sha256=nh,cohort_sha256=sha(file),
                      elapsed=old["elapsed"]+timer()-clock,
                      peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1)))
        print("boundary generated",len(result)//2,len(stops),flush=True)
    z=summarize(result)
    save(root/"BOUNDARY_SUMMARY.json",z)
    print("boundary gates",z["gates"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)
