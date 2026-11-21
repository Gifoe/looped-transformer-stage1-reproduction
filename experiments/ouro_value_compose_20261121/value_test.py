"""Replay context-specific K2 attention while crossing only the first operand V."""
import argparse
from pathlib import Path

import torch

from block import matched
from boundary_test import Observe, pack_k1
from edge import locations, short_hash
from interchange import Patch, terminated
from native import Forward, load, output, pad, save, setup, sha, timer
from rule import MAX_NEW, greedy, text_results
from value_stats import ARMS, MODES, summarize


def case(tok,g,s,row,kind,diagnostic):
    f="borrow" if kind=="repair" else "safe"
    receiver=row[f][3]
    db=row[f+"_donors"]
    tokens={k:locations(tok,p,[]) for k,p in
            dict(receiver=receiver,opposite=db["plus"],same=db["minus"]).items()}
    ids,last,low,prior=tokens["receiver"]
    indexes=[low[0]-1,*low,last-2,last-1,last]
    for u,q,a,b in tokens.values():
        assert (q,a,b)==(last,low,prior) and len(u)==len(ids)
        assert all(tok.decode([u[p]]).isdigit() for p in indexes)
        assert all(v==w for j,(v,w) in enumerate(zip(ids,u,strict=True)) if j not in indexes)
    assert ids==s["ids"]
    if kind=="repair":
        assert receiver[1]-db["plus"][0]==row["mix"]
    else:
        assert receiver[1]-db["plus"][0]==row["safe_mix"]
    return dict(id=g["id"],source_id=row["id"],kind=kind,target=g["target"],foil=g["foil"],
                mixed=receiver[1]-db["plus"][0],plus=db["plus"][1]-db["plus"][0],
                minus=db["minus"][1]-db["minus"][0],diagnostic=diagnostic,operand=indexes,
                prefix=s["prefix"],native=s["native"],ids={k:p[0] for k,p in tokens.items()})


def indices(x,positions):
    return positions[:,None,:,None].expand(-1,x.shape[1],-1,x.shape[-1])


class Trace:
    def __init__(self,positions,at,weights=True,audit=False):
        self.positions=positions
        self.first=int(positions[0,0])
        assert (positions[:,0]==self.first).all()
        self.stage=0
        self.attention={}
        self.values={}
        self.weights=weights
        self.l15=Observe(at) if audit else None

    def put(self,key,name,x):
        phase,l=key
        if phase!=self.stage:
            return
        if name=="a" and self.weights:
            self.attention[l]=x[:,:,self.first:,:].detach().contiguous()
        if name=="v":
            self.values[l]=x.gather(2,indices(x,self.positions)).detach()
        if self.l15 is not None:
            self.l15.put((0,l),name,x)


def shuffle(base,delta):
    unit=base/base.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    radial=(unit*delta).sum(-1,keepdim=True)*unit
    tangent=delta-radial
    n=delta.shape[-1]
    signs=1-2*(torch.arange(n,device=delta.device)%2)
    z=delta.roll(n//3,dims=-1)*signs
    z=z-unit*(z*unit).sum(-1,keepdim=True)
    z=radial+z*(tangent.norm(dim=-1,keepdim=True)/z.norm(dim=-1,keepdim=True).clamp_min(1e-12))
    for p,q in ((z,delta),(base+z,base+delta)):
        torch.testing.assert_close(p.norm(dim=-1),q.norm(dim=-1),rtol=3e-5,atol=3e-6)
    return base+z


class FixedValues(Patch):
    def __init__(self,trace,at,ids,modes,bank,audit):
        super().__init__(at)
        self.trace=trace
        self.bank=bank
        self.pick=ids
        self.pos=trace.positions[ids]
        self.modes=modes
        self.audit=audit
        assert len(ids)==len(modes)

    def attn(self,key,a,v):
        l=key[1]
        base=self.trace.values[l][self.pick]
        a0,b0=base.split(3,dim=2)
        donor=self.bank[l][:,:,:3]
        values=a0.clone()
        for name in ("donor","shuffle"):
            j=[i for i,x in enumerate(self.modes) if x==name]
            if j:
                source=donor[j]
                values[j]=source if name=="donor" else shuffle(a0[j],source-a0[j])
        selected=torch.cat([values,b0],dim=2)
        vi=indices(v,self.pos)
        fixed=v.scatter(2,vi,selected)
        if self.audit:
            assert torch.equal(fixed.gather(2,vi)[:,:,3:],b0)
        weights=a.clone()
        weights[:,:,self.trace.first:]=self.trace.attention[l][self.pick]
        return weights,fixed


def donor_k2(f,cases,histories,bank,alternate=False):
    values={}
    for start in range(0,len(histories),16):
        group=histories[start:start+16]
        tag="same" if alternate else "opposite"
        tok,at=pad([cases[i]["ids"][tag]+list(s) for i,s in group])
        pos=torch.tensor([cases[i]["operand"] for i,_ in group],device=tok.device)
        tr=Trace(pos,at,weights=False)
        if alternate:
            tr.stage=1
            f.run(tok,2,record=tr)
        else:
            z=torch.nn.utils.rnn.pad_sequence([bank[i,s,"whole_donor"] for i,s in group],batch_first=True)
            f.run(tok,1,start=z,record=tr)
        assert len(tr.values)==24
        for j,entry in enumerate(group):
            values[entry]={l:x[j] for l,x in tr.values.items()}
    return values


def calc(f,cases,queries,audit=False):
    todo={}
    for a,i,p in queries:
        bg,mode=MODES[a]
        todo.setdefault((i,p,bg),[]).append((a,i,mode))
    sources=list(dict.fromkeys((i,p) for i,p,_ in todo))
    initial=pack_k1(f,cases,sources)
    required=list(dict.fromkeys((i,p) for (i,p,bg),q in todo.items() if any(m!="native" for a,j,m in q)))
    donors=donor_k2(f,cases,required,initial)
    if audit:
        alternate=donor_k2(f,cases,required,initial,alternate=True)
        for key in required:
            for l in donors[key]:
                torch.testing.assert_close(donors[key][l][:,:3],alternate[key][l][:,:3],rtol=5e-5,atol=3e-4)
    result={}
    measures={}
    baseline_a={}
    group=list(todo)
    for start in range(0,len(group),10):
        current=group[start:start+10]
        ids,at=pad([cases[i]["ids"]["receiver"]+list(p) for i,p,_ in current])
        pos=torch.tensor([cases[i]["operand"] for i,p,_ in current],device=ids.device)
        z=torch.nn.utils.rnn.pad_sequence([initial[x] for x in current],batch_first=True)
        tr=Trace(pos,at,audit=audit)
        out=f.run(ids,1,start=z,record=tr)[0]
        assert len(tr.attention)==len(tr.values)==24
        logits=output(f.model,out,at).argmax(-1).tolist()
        edits=[]
        for j,x in enumerate(current):
            i,p,bg=x
            if audit:
                here={l:w[j,:,:3] for l,w in tr.values.items()}
                old=baseline_a.get((i,p))
                if old is not None:
                    for l,y in here.items():
                        torch.testing.assert_close(old[l],y,rtol=5e-5,atol=3e-4)
                else:
                    baseline_a[i,p]=here
            for a,i,m in todo[x]:
                if m=="native":
                    result[a,i]=logits[j]
                    if audit:
                        measures[a,i]={k:x[j] for k,x in tr.l15.values.items()}
                else:
                    edits.append((j,a,i,p,m))
            if audit:
                edits.append((j,None,i,p,"native"))
        if not edits:
            continue
        j=[x[0] for x in edits]
        modes=[x[4] for x in edits]
        # A clean replay uses its own V bank. Donor V is needed only for edited arms.
        v_bank={l:torch.stack([donors[i,p][l] if m!="native" else tr.values[l][k]
                               for k,a,i,p,m in edits]) for l in tr.values}
        watch=Observe(at[j]) if audit else None
        changed=f.run(ids[j],1,start=z[j],edit=FixedValues(tr,at[j],j,modes,v_bank,audit),record=watch)[0]
        nexts=output(f.model,changed,at[j]).argmax(-1).tolist()
        for v,(k,a,i,p,m) in enumerate(edits):
            if a is None:
                torch.testing.assert_close(changed[v],out[k],rtol=5e-5,atol=3e-4)
                assert logits[k]==nexts[v]
            else:
                result[a,i]=nexts[v]
                if audit:
                    measures[a,i]={k:x[v] for k,x in watch.values.items()}
    assert set(result)=={(a,i) for a,i,_ in queries}
    return result,measures


def decode(f,tok,cases):
    active={(a,i):[] for a in ARMS for i in range(len(cases))}
    output={}
    for _ in range(MAX_NEW):
        if not active:
            break
        nxt,_=calc(f,cases,[(a,i,tuple(t)) for (a,i),t in active.items()])
        for k,t in list(active.items()):
            t.append(nxt[k])
            if terminated(tok,t):
                output[k]=active.pop(k)
    output.update(active)
    data=dict(zip(output,text_results(tok,list(output.values())),strict=True))
    return [{a:data[a,i] for a in ARMS} for i in range(len(cases))]


def evaluate(f,tok,cases):
    predicted=decode(f,tok,cases)
    donor={k:[greedy(f,tok,[c["ids"][k] for c in cases],d) for d in (1,2)] for k in ("opposite","same")}
    use=[i for i,c in enumerate(cases) if c["diagnostic"]]
    q=[(a,i,tuple(cases[i]["prefix"])) for a in ARMS for i in use]
    _,data=calc(f,cases,q,audit=True) if q else ({},{})
    out=[]
    for i,c in enumerate(cases):
        assert predicted[i]["receiver_native"]==c["native"][1]
        row=dict(pred=predicted[i],donor_native={k:[x[i] for x in v] for k,v in donor.items()})
        if i in use:
            base=data["receiver_native",i]
            row["diagnostics"]={a:{k:(v-base[k]).norm().item() for k,v in data[a,i].items()} for a in ARMS}
        out.append(row)
    return out


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    start=timer()
    native=root/"NATIVE.json"
    h=sha(native)
    by={r["id"]:r for r in load(native)["rows"]}
    pairs=matched(f,tok,root,"confirmation")
    keys=[x["g"]["id"] for x,_,_,_ in pairs]
    diagnosis=set(sorted(keys,key=lambda i:short_hash("value-read",i))[:24])
    cohort=root/"COHORT.json"
    proof=dict(native_sha256=h,diagnostic_ids=sorted(diagnosis),
               pairs=[dict(id=x["g"]["id"],safe=s["id"],margin=list(m),difference=d) for x,s,m,d in pairs])
    if cohort.exists():
        assert load(cohort)==proof
    else:
        save(cohort,proof)
    raw=root/"VALUE_RAW.json"
    old=load(raw) if raw.exists() else dict(rows=[],elapsed=0.)
    rows=old["rows"]
    if rows:
        assert old["native_sha256"]==h and old["cohort_sha256"]==sha(cohort)
        assert [r["id"] for r in rows[::2]]==keys[:len(rows)//2]
    for j in range(len(rows)//2,len(pairs),3):
        entries=[]
        for r,s,_,di in pairs[j:j+3]:
            x=r["g"]
            a=case(tok,x,r["pair"][0],x,"repair",x["id"] in diagnosis and r["aligned"])
            b=case(tok,x,s,by[s["id"]],"safe",False)
            entries.extend(dict(z,distance=di) for z in (a,b))
        results=evaluate(f,tok,entries)
        rows.extend(dict({k:v for k,v in a.items() if k not in ("ids","prefix")},**b) for a,b in zip(entries,results,strict=True))
        save(raw,dict(rows=rows,native_sha256=h,cohort_sha256=sha(cohort),
                      elapsed=old["elapsed"]+timer()-start,
                      peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1)))
        print("value composition",len(rows)//2,len(pairs),flush=True)
    z=summarize(rows)
    save(root/"VALUE_SUMMARY.json",z)
    print("value gates",z["gates"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)
