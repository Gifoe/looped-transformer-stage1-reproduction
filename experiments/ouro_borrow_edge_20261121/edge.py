"""Test one frozen causal K1 head and trace its effects into native K2."""
import argparse
import hashlib
from pathlib import Path

import torch

from block import Intervene, geom
from edge_stats import ARMS, summarize
from interchange import Patch, complete, fit_margin
from native import Forward, load, output, pad, save, setup, sha, timer
from native_deck import is_repair
from origin import LAYER, check_early_prefix, decompose, dose
from readout import forms
from rule import PREFIX, prompt


def short_hash(*data):
    return hashlib.sha256("|".join(map(str,(*data,261122901))).encode()).hexdigest()


def locations(tok, pair, continuation):
    a,b=pair
    text=prompt(a,b,1)
    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    sa=len(PREFIX)+len("A = ")
    sb=sa+len(str(a))+len(", B = ")
    cells=[tuple(z) for z in encoded["offset_mapping"]]
    q=cells.index((sb+len(str(b))-1,sb+len(str(b))))
    keys=[cells.index((p,p+1)) for p in (sa+len(str(a))-2,sa+len(str(a))-1)]
    other=[i for i,(s,e) in enumerate(cells) if 0<i<q and e<=sa and e>s]
    assert len(other)>=2 and all(k<q for k in keys) and not set(keys)&set(other)
    return encoded["input_ids"]+continuation,q,keys,other


class Layout:
    def __init__(self, rows):
        self.ids,self.at=pad([x[0] for x in rows])
        self.rows=torch.arange(len(rows),device=self.ids.device)
        self.q=torch.tensor([x[1] for x in rows],device=self.ids.device)
        self.keys=torch.tensor([x[2] for x in rows],device=self.ids.device)
        assert all(r[3]==rows[0][3] for r in rows)
        self.other=torch.tensor(rows[0][3],device=self.ids.device)


class Routes(Patch):
    def __init__(self, layout, layer, head, control=False, fraction=1.):
        super().__init__(layout.q)
        self.layout=layout
        self.layer=layer
        self.head=head
        self.control=control
        self.fraction=fraction
        self.receipt=None

    def attn(self,key,a,v):
        if key[1]!=self.layer:
            return a,v
        p=self.layout
        row=a[p.rows,self.head,p.q]
        target=row.gather(1,p.keys).sum(1,keepdim=True)*self.fraction
        if self.control:
            pairs=torch.combinations(p.other,r=2)
            both=row[:,pairs].sum(-1)
            surplus=both-target
            ix=torch.where(surplus>=0,surplus,2-surplus).argmin(1)
            keys=pairs[ix]
            mass=row.gather(1,keys).sum(1,keepdim=True)
            factor=(target/mass.clamp_min(1e-15)).clamp(max=1.)
        else:
            keys=p.keys
            mass=row.gather(1,keys).sum(1,keepdim=True)
            factor=self.fraction
        reduced=row.scatter(1,keys,row.gather(1,keys)*(1-factor))
        assert reduced.sum(1).min()>1e-7
        norm=reduced/reduced.sum(1,keepdim=True)
        torch.testing.assert_close(norm.sum(1),torch.ones_like(target[:,0]),rtol=3e-6,atol=3e-6)
        self.receipt=(mass*factor).flatten().tolist()
        b=a.clone()
        b[p.rows,self.head,p.q]=norm
        return b,v


class Writes:
    def __init__(self, layout, layer):
        self.layout=layout
        self.layer=layer
        self.value=None

    def put(self,key,name,x):
        if key[1]==self.layer and name=="a_write":
            p=self.layout
            self.value=x[p.rows,p.q].detach()


def pass1(f,layout,layer,edit=None,start=None):
    book=Writes(layout,layer)
    end=f.run(layout.ids,1,edit=edit,start=start,record=book)[0]
    assert book.value is not None
    return end,book.value


class Screen:
    def __init__(self,f,layout):
        self.f=f
        self.layout=layout
        self.rows=[]
        self.v=None

    def put(self,key,name,x):
        l=key[1]
        mod=self.f.layers[l].self_attn
        p=self.layout
        if name=="v":
            self.v=self.f.module.repeat_kv(x,mod.num_key_value_groups)
        if name!="a":
            return
        current=x[p.rows,:,p.q]
        kept=current.scatter(2,p.keys[:,None,:].expand(-1,current.shape[1],-1),0.)
        kept=kept/kept.sum(-1,keepdim=True).clamp_min(1e-12)
        response=torch.einsum("bht,bhtd->bhd",current-kept,self.v)
        coef=mod.o_proj.weight.view(mod.o_proj.out_features,*response.shape[1:])
        projected=torch.einsum("bhd,ohd->bho",response,coef)
        self.rows.append(projected.norm(dim=-1))


def sets(tok,row,kind,prefix):
    return [locations(tok,p,prefix) for p in row[kind]]


def offsets(rows):
    return [r for x in rows for r in x]


@torch.inference_mode()
def select(f,tok,root,deck):
    dev=sorted((r for r in deck["rows"] if r["split"]=="discovery"),key=lambda r:short_hash("dev",r["id"]))
    assert len(dev)==40
    ws=tok.encode(" ",add_special_tokens=False)
    assert ws and not tok.decode(ws).strip()
    candidate=[]
    for i in range(0,len(dev),5):
        batch=Layout([locations(tok,r["borrow"][3],ws) for r in dev[i:i+5]])
        read=Screen(f,batch)
        f.run(batch.ids,1,record=read)
        candidate.append(torch.stack(read.rows).cpu())
    med=torch.cat(candidate,dim=1).median(dim=1).values
    ranked=sorted(((float(med[l,k]),l,k) for l in range(med.shape[0]) for k in range(med.shape[1])),reverse=True)[:6]
    focus=dev[:12]
    layout=Layout(offsets([sets(tok,r,"borrow",ws) for r in focus]))
    ids,at=layout.ids,layout.at
    h1=f.run(ids,1)[0]
    _,_,ref,_,_=decompose(f,ids,at,start=h1)
    scored=[]
    for local,l,k in ranked:
        vals=[]
        norms=[]
        for control in (False,True):
            op=Routes(layout,l,k,control)
            h=f.run(ids,1,edit=op)[0]
            _,_,part,_,_=decompose(f,ids,at,start=h)
            z=[(part["full"][j]-ref["full"][j]).norm(dim=-1)/ref["full"][j].norm(dim=-1).clamp_min(1e-6)
               for j in (1,3)]
            vals.append(torch.stack(z,dim=-1))
            norms.append(op.receipt)
        delta=(vals[0]-vals[1]).mean(dim=-1)
        scored.append(dict(layer=l,head=k,screen=local,score=delta.median().item(),
                           target=vals[0].tolist(),control=vals[1].tolist(),mass=norms))
    winner=max(scored,key=lambda a:a["score"])
    result=dict(layer=winner["layer"],head=winner["head"],screen_shape=list(med.shape),evaluated=scored,
                focus=[r["id"] for r in focus],dev=[r["id"] for r in dev],deck_sha256=sha(root/"DECK.json"))
    save(root/"HEAD_LOCK.json",result)
    print("selected physical layer, head",result["layer"]+1,result["head"]+1,flush=True)
    return result


def prepare_donors(f,tok,deck,layer,head):
    bank={}
    for row in deck["rows"]:
        if row["split"]!="discovery":
            continue
        batch=Layout([locations(tok,row[k][3],[]) for k in ("borrow","safe")])
        _,a=pass1(f,batch,layer)
        _,b=pass1(f,batch,layer,Routes(batch,layer,head))
        bank[row["id"]]=(row,a-b)
    return bank


def donorpick(r,bank):
    options=[z for z in bank.values() if z[0]["width"]==r["width"] and
             z[0]["target"]//100!=r["target"]//100 and z[0]["foil"]//100!=r["foil"]//100]
    return min(options,key=lambda z:short_hash("donor",r["id"],z[0]["id"]))


@torch.inference_mode()
def evaluate(f,tok,records,cases,bank,layer,head):
    packed=offsets([sets(tok,r,"borrow" if c["kind"]=="repair" else "safe",c["prefix"])
                    for r,c in zip(records,cases,strict=True)])
    b=Layout(packed)
    ids,at=b.ids,b.at
    j=b.rows[3::4]
    wanted=torch.tensor([c["labels"] for c in cases],device=at.device)
    k1,a1=pass1(f,b,layer)
    _,a2=pass1(f,b,layer,start=k1)
    k2,ref,base,info,_=decompose(f,ids,at,start=k1)
    n1,n2=f.run(ids,2)
    assert torch.equal(k1,n1) and torch.equal(k2,n2)
    act=Routes(b,layer,head)
    cut,ca=pass1(f,b,layer,act)
    other=Routes(b,layer,head,control=True)
    sham,ma=pass1(f,b,layer,other)
    old,site,cmp,cinfo,_=decompose(f,ids,at,start=cut)
    full=f.run(ids,1,start=sham)[0]
    donor=[]
    for r in records:
        src,v=donorpick(r,bank)
        donor.append((src,v))
    d=a1-ca
    qc=[dict(c,id=c["id"]+f"#{i%4}") for c in cases for i in range(4)]
    rand=geom(a1,a2,d,qc,layer,992)
    edit=Patch(b.q,writes={(layer,"a_write"):a1-rand})
    e1,_=pass1(f,b,layer,edit)
    endpoints={"native":k2,"readcut":old,"matched_read":full,
               "geometry":f.run(ids,1,start=e1)[0]}

    def late(vector):
        with Intervene(f,LAYER,at[j],site+vector):
            return f.run(ids[j],1,start=cut[j])[0]

    logits={}
    for name,z in endpoints.items():
        logits[name]=(output(f.model,z[j],at[j]),fit_margin(f,z[j],at[j],wanted))
    p=base["full"][1]-cmp["full"][1]
    m=base["full"][3]-cmp["full"][3]
    for name,v in (("restore_p",p),("restore_m",m),("restore_both",p+m),
                   ("restore_geometry",geom(site,ref,p+m,cases,LAYER,994))):
        z=late(v)
        logits[name]=(output(f.model,z,at[j]),fit_margin(f,z,at[j],wanted))
    a_d=torch.stack([v[0] for _,v in donor])
    s_d=torch.stack([v[1] for _,v in donor])
    norm=d[j].norm(dim=-1)
    b_vec=dose(a_d,norm)
    s_vec=dose(s_d,norm)
    for name,v in (("donor_borrow",b_vec),("donor_safe",s_vec),
                   ("donor_geometry",geom(ca[j],a1[j],b_vec,cases,layer,995)),("self_restore",d[j])):
        value=ca.clone()
        value[j]=ca[j]+v
        op=Routes(b,layer,head)
        op.writes[layer,"a_write"]=value
        h,_=pass1(f,b,layer,op)
        z=f.run(ids,1,start=h)[0]
        logits[name]=(output(f.model,z[j],at[j]),fit_margin(f,z[j],at[j],wanted))
    assert set(logits)==set(ARMS)
    decided={k:z.argmax(-1).tolist() for k,(z,_) in logits.items()}
    assert decided["self_restore"]==decided["native"]
    unique=list(dict.fromkeys((i,v) for vs in decided.values() for i,v in enumerate(vs)))
    completions=complete(f,tok,[cases[i] for i,_ in unique],[v for _,v in unique],1)
    table=dict(zip(unique,completions,strict=True))
    assert [table[i,v] for i,v in enumerate(decided["native"])]==[c["native"][1] for c in cases]
    rec=[]
    for i,c in enumerate(cases):
        q=slice(4*i,4*i+4)
        pred={a:table[i,v[i]] for a,v in decided.items()}
        rec.append(dict(pred=pred, margin={a:v[i].item() for a,(_,v) in logits.items()},
                        mass=act.receipt[q],matched_mass=other.receipt[q],
                        edge_write_norm=norm[i].item(),control_write_norm=(a1-ma)[j[i]].norm().item(),
                        propagation_shift=p[i].norm().item(),new_mlp_shift=m[i].norm().item(),
                        new_attention_shift=(base["full"][2]-cmp["full"][2])[i].norm().item(),
                        K2_native_P=info["propagated"][i],K2_native_M=info["m"][i],
                        K2_norm_outside=cinfo["outside"][i],
                        donor={k:donor[i][0][k] for k in ("id","target","foil","width")}))
    return rec


@torch.inference_mode()
def run(model,root,selection_only=False):
    m,tok=setup(model)
    f=Forward(m)
    deck=load(root/"DECK.json")
    lock=load(root/"HEAD_LOCK.json") if (root/"HEAD_LOCK.json").exists() else select(f,tok,root,deck)
    assert lock["deck_sha256"]==sha(root/"DECK.json")
    if selection_only:
        return
    start=timer()
    l,k=lock["layer"],lock["head"]
    bank=prepare_donors(f,tok,deck,l,k)
    native=root/"NATIVE.json"
    full=load(native)["rows"]
    chosen=sorted((r for r in full if r["split"]=="confirmation" and is_repair(r)),key=lambda r:r["id"])
    raw=root/"EDGE_RAW.json"
    data=load(raw) if raw.exists() else {"rows":[],"elapsed":0.}
    rows=data["rows"]
    assert len(rows)%2==0 and [x["id"] for x in rows[::2]]==[r["id"] for r in chosen[:len(rows)//2]]
    if rows:
        assert data["native_sha256"]==sha(native) and data["lock_sha256"]==sha(root/"HEAD_LOCK.json")
    for i in range(len(rows)//2,len(chosen),4):
        group=chosen[i:i+4]
        converted=[forms(tok,r) for r in group]
        flags=[p["aligned"] for p in converted]
        recipients=[c for p,a in zip(converted,flags,strict=True) if a for c in p["pair"]]
        early=iter(check_early_prefix(f,recipients) if recipients else [])
        flags=[all((next(early),next(early))) if ok else False for ok in flags]
        cases=[c for p,ok in zip(converted,flags,strict=True) if ok for c in p["pair"]]
        active=[r for r,ok in zip(group,flags,strict=True) if ok for _ in (0,1)]
        measured=iter(evaluate(f,tok,active,cases,bank,l,k) if cases else [])
        for p,yes in zip(converted,flags,strict=True):
            r=p["g"]
            for c in p["pair"]:
                v=next(measured) if yes else dict(pred={a:c["native"][1] for a in ARMS})
                rows.append(dict(id=r["id"],kind=c["kind"],target=r["target"],foil=r["foil"],
                                 width=r["width"],aligned=yes,native=c["native"],**v))
        save(raw,dict(rows=rows,native_sha256=sha(native),lock_sha256=sha(root/"HEAD_LOCK.json"),
                      elapsed=data["elapsed"]+timer()-start,
                      peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1)))
        print("edge confirmation",len(rows)//2,len(chosen),flush=True)
    z=summarize(rows)
    save(root/"EDGE_SUMMARY.json",z)
    print("causal gates",z["gates"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    p.add_argument("--selection-only",action="store_true")
    a=p.parse_args()
    run(a.model,a.root,a.selection_only)
