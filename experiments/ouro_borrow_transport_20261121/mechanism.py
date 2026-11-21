"""Recipient-projection Q transport and controlled L17 readout subspaces."""
import argparse
import contextlib
import hashlib
from pathlib import Path

import torch

import baseline
from borrow import ci, native_event
from interchange import Patch, complete, independent_like, text_case
from native import Forward, load, output, pad, save, setup, sha
from routing import set_fields
from rule import greedy
from state import rotate_about

LQ=13
LR=16
BANK_RANK=8
TRANSFER=("distant_borrow","distant_safe","aligned_borrow","aligned_safe","geom","other_field","distant_restore")
MEDIATION=("drop","drop_geom","keep","keep_geom")
BASICS=("native","self_query","self_query_restore","self_A","self_A_restore")
MODES=BASICS+TRANSFER+MEDIATION


class Capture:
    def __init__(self,last):
        self.last=last
        self.rows=torch.arange(len(last),device=last.device)
        self.data={}

    def put(self,key,label,z):
        j,l=key
        if l==LR and label=="input":
            self.data[j,"r"]=z[self.rows,self.last].detach()
        if l!=LQ:
            return
        if label in ("k","v"):
            self.data[j,label]=z.detach()
        elif label in ("q","a"):
            self.data[j,label]=z[self.rows,:,self.last].detach()
        elif label=="av":
            self.data[j,label]=z[self.rows,self.last].flatten(1).detach()

    def get(self,k,label):
        return self.data[k,label]


class Edit(Patch):
    def __init__(self,last,query=None,readout=None):
        super().__init__(last,query=query)
        self.readout=readout

    def resid(self,k,l,x):
        if l!=LR or self.readout is None:
            return x
        q=x.clone()
        q[self.rows,self.pos]=self.readout
        return q


@contextlib.contextmanager
def graft_av(f,pos,value):
    if value is None:
        yield
        return
    rows=torch.arange(len(pos),device=pos.device)

    def at_v(_,args):
        before=args[0]
        assert value.shape==before[rows,pos].shape
        x=before.clone()
        x[rows,pos]=value
        return (x,*args[1:])

    hook=f.layers[LQ].self_attn.o_proj.register_forward_pre_hook(at_v)
    try:
        yield
    finally:
        hook.remove()


def cohort(tok,r,k):
    return set_fields(tok,r,text_case(tok,r,k),k)


def prepare(f,cases):
    ids,pos=pad([c["ids"]+c["prefix"] for c in cases])
    rec=Capture(pos)
    h1,h2=f.run(ids,2,record=rec)
    assert torch.equal(f.run(ids,1,start=h1)[0],h2)
    fields=torch.tensor([c["fields"]+[-1]*(ids.shape[1]-len(c["fields"])) for c in cases],device=ids.device)
    return ids,pos,rec,(h1,h2),fields


def labels(c):
    return c["target"]//100,c["foil"]//100


def sort_donor(x,d):
    return hashlib.sha256((x+"|"+d+"|261121818").encode()).hexdigest()


def build_plan(tok,rows):
    db={}
    pairs=[]
    for r in rows:
        if r["split"]!="donors" or not native_event(r,1) or not baseline.exact_safe(r):
            continue
        a,b=cohort(tok,r,1),cohort(tok,r,0)
        if not (all(a["reach"]) and all(b["reach"]) and len(a["fields"])==len(b["fields"])):
            continue
        db[r["id"],"repair"]=a
        db[r["id"],"safe"]=b
        pairs.append(a)
    receipts=[]
    for r in rows:
        if r["split"]!="receivers" or not native_event(r,1):
            continue
        case=cohort(tok,r,1)
        safe=cohort(tok,r,0)
        ok=[p for p in pairs if len(p["fields"])==len(case["fields"]) and all(case["reach"])]
        ordered=sorted(ok,key=lambda p:sort_donor(case["id"],p["id"]))
        different=next((p for p in ordered if set(labels(p)).isdisjoint(labels(case))),None)
        same=next((p for p in ordered if labels(p)==labels(case) and p["target"]!=case["target"]),None)
        receipts.append(dict(id=r["id"],repair=case,safe=safe,
                             disjoint=different["id"] if different else None,
                             aligned=same["id"] if same else None))
    receipts.sort(key=lambda r:r["id"])
    return db,pairs,receipts


def weights(f,q,k,last):
    div=f.layers[LQ].self_attn.scaling
    s=torch.matmul(q.unsqueeze(-2),k.transpose(-1,-2)).squeeze(-2)*div
    pad_mask=torch.arange(k.shape[-2],device=k.device)[None,:] > last[:,None]
    s=torch.masked_fill(s,pad_mask[:,None,:],torch.finfo(s.dtype).min)
    return torch.softmax(s,dim=-1,dtype=torch.float32)


def delta_av(w,v,field):
    return torch.matmul((w*field[:,None,:]).unsqueeze(-2),v).squeeze(-2).flatten(1)


def same_norm(v,d):
    dose=d.norm(dim=-1,keepdim=True)
    n=v.norm(dim=-1,keepdim=True)
    result=v*(dose/n.clamp_min(1e-10))
    active=(n.squeeze(-1)>1e-10) | (dose.squeeze(-1)<=1e-10)
    torch.testing.assert_close(result[active].norm(dim=-1),dose[active,0],rtol=2e-5,atol=2e-5)
    return result,active


def learn(f,cases,root):
    qdb={}
    vector=[]
    for i in range(0,len(cases),12):
        pack=cases[i:i+12]
        ids,pos,r,h,tags=prepare(f,pack)
        for j,c in enumerate(pack):
            qdb[c["id"],c["kind"]]=tuple(r.get(k,"q")[j] for k in (0,1))
        use=[j for j,c in enumerate(pack) if c["kind"]=="repair"]
        for k in (0,1):
            got=Capture(pos)
            f.run(ids,1,start=h[0] if k else None,record=got,edit=Edit(pos,query={LQ:(None,r.get(1-k,"q"))}))
            vector.append((got.get(0,"r")-r.get(k,"r"))[use])
    d=torch.cat(vector)
    assert torch.isfinite(d).all() and len(d)>=BANK_RANK
    u,s,vh=torch.linalg.svd(d,full_matrices=False)
    mat=vh[:BANK_RANK].contiguous()
    torch.testing.assert_close(mat@mat.T,torch.eye(BANK_RANK,device=mat.device),rtol=1e-5,atol=1e-5)
    info=dict(rank=BANK_RANK,n=len(d),donor_ids=sorted({c["id"] for c in cases}),
              singular_values=s.tolist(),learn_energy=float(s[:BANK_RANK].square().sum()/s.square().sum()),basis_vectors=mat.tolist())
    save(root/"SUBSPACE.json",info)
    return qdb,mat


def project(v,u):
    return (v@u.T)@u


def matched_plane(base,change,target,random):
    unit=lambda x:x/x.norm(dim=-1,keepdim=True).clamp_min(1e-10)
    u=unit(base)
    v=unit(change-(change*u).sum(-1,keepdim=True)*u)
    def remove_span(x):
        return x-(x*u).sum(-1,keepdim=True)*u-(x*v).sum(-1,keepdim=True)*v
    perp=remove_span(target)
    matched=target-perp+perp.norm(dim=-1,keepdim=True)*unit(remove_span(random))
    for a in (base,base+change):
        torch.testing.assert_close((a-matched).norm(dim=-1),(a-target).norm(dim=-1),rtol=2e-5,atol=2e-5)
        torch.testing.assert_close((a+matched).norm(dim=-1),(a+target).norm(dim=-1),rtol=2e-5,atol=2e-5)
    torch.testing.assert_close(matched.norm(dim=-1),target.norm(dim=-1),rtol=2e-5,atol=2e-5)
    return matched


def interact(f,tok,pack,qdb,mat):
    cases=[q for _,q in pack]
    ids,p,tr,h,fields=prepare(f,cases)
    n=len(cases)
    modes={}
    for k in (0,1):
        native_rows=greedy(f,tok,[c["ids"] for c in cases],k+1)
        assert [x["tokens"] for x in native_rows]==[c["native"][k]["tokens"] for c in cases]
    records=[{"active":{},"dose":[],"subspace_energy":[]} for _ in cases]
    for k in (0,1):
        own=tr.get(k,"q")
        other=tr.get(1-k,"q")
        keys=tr.get(k,"k")
        values=tr.get(k,"v")
        wav=tr.get(k,"av")
        measured=tr.get(k,"a")
        read=tr.get(k,"r")
        natural=weights(f,own,keys,p)
        torch.testing.assert_close(natural,measured,rtol=4e-5,atol=5e-6)
        all_pos=(fields>=0)
        reproduced=delta_av(measured,values,all_pos)
        torch.testing.assert_close(wav,reproduced,rtol=3e-5,atol=2e-5)
        alt=weights(f,other,keys,p)
        dummy=Capture(p)
        qedit=Edit(p,query={LQ:(None,other)})
        q_state=f.run(ids,1,start=h[0] if k else None,edit=qedit,record=dummy)[0]
        delta=delta_av(alt-natural,values,fields==1)
        torch.testing.assert_close(alt,dummy.get(0,"a"),rtol=4e-5,atol=5e-6)
        only_r=dummy.get(0,"r")-read
        split=project(only_r,mat)
        geo=matched_plane(read,only_r,split,independent_like(split,cases,LR,141))
        for j,r in enumerate(records):
            r["dose"].append(delta[j].norm().item())
            r["subspace_energy"].append((split[j].square().sum()/only_r[j].square().sum().clamp_min(1e-15)).item())
        lookup={}
        for name in TRANSFER:
            if name in ("geom","distant_restore"):
                continue
            pick_key="aligned" if name.startswith("aligned") else "disjoint"
            kind="safe" if name.endswith("safe") else "repair"
            present=[]
            donor=[]
            for j,(meta,c) in enumerate(pack):
                match=meta[pick_key]
                good=bool(match and len(c["fields"])==len(meta["repair"]["fields"]))
                present.append(good)
                fallback=(own[j],own[j])
                donor.append(qdb[match,kind] if good else fallback)
            q1,q2=(torch.stack([w[i] for w in donor]) for i in (0,1))
            group=2 if name=="other_field" else 1
            d=delta_av(weights(f,q2,keys,p)-weights(f,q1,keys,p),values,fields==group)
            delta_scaled,nonzero=same_norm(d,delta)
            eligible=torch.tensor(present,device=p.device)&nonzero
            lookup[name]=(wav+delta_scaled*(1 if not k else -1),eligible)
        dv,eligible=lookup["distant_borrow"]
        lookup["geom"]=(rotate_about(wav,dv,independent_like(dv,cases,LQ,169)),eligible)
        lookup["distant_restore"]=(dv,eligible)
        for name in MODES:
            eligible=torch.ones(n,device=p.device,dtype=torch.bool)
            if name=="native":
                y=h[k]
            elif name=="self_query":
                y=q_state
            else:
                prevent=read if name.endswith("_restore") else None
                inject=None
                custom_q={LQ:(None,other)} if name=="self_query_restore" or name in MEDIATION else None
                if name.startswith("self_A"):
                    inject=wav+delta
                elif name in TRANSFER:
                    inject,eligible=lookup[name]
                if name in MEDIATION:
                    if name=="drop":
                        prevent=read+only_r-split
                    elif name=="drop_geom":
                        prevent=read+only_r-geo
                    elif name=="keep":
                        prevent=read+split
                    else:
                        prevent=read+geo
                with graft_av(f,p,inject):
                    y=f.run(ids,1,start=h[0] if k else None,edit=Edit(p,query=custom_q,readout=prevent))[0]
                if name.endswith("_restore"):
                    torch.testing.assert_close(y[torch.arange(n,device=p.device),p],
                                               h[k][torch.arange(n,device=p.device),p],rtol=2e-5,atol=2e-5)
            nxt=output(f.model,y,p).argmax(-1).tolist()
            if name.endswith("_restore"):
                assert nxt==output(f.model,h[k],p).argmax(-1).tolist()
            vv=[j if ok else c["native"][k]["tokens"][len(c["prefix"])] if c["reach"][k] else j
                for j,ok,c in zip(nxt,eligible.tolist(),cases,strict=True)]
            out=complete(f,tok,cases,vv,k)
            for j,(r,a) in enumerate(zip(records,eligible.tolist(),strict=True)):
                r["active"].setdefault(name,[]).append(bool(a and cases[j]["reach"][k]))
            modes.setdefault(name,[]).append(out)
    result={}
    for name,(one,two) in modes.items():
        result[name]=list(zip(one,two,strict=True))
    return result,records


def summarize(rows):
    by={}
    for kind in ("repair","safe"):
        group=[r for r in rows if r["kind"]==kind]
        outcomes={}
        for name in MODES:
            data={}
            for k,expected in (("K1_correct",0),("K2_foil",1)):
                data[k]=sum(x["pred"][name][expected]["number"]==x[("target","foil")[expected]] for x in group)
            good=[x for x in group if [y["number"] for y in x["pred"][name]]==[x["target"],x["foil"]]]
            data["joint"]=len(good)
            data["joint_targets"]=len({q["target"] for q in good})
            data["active"]=sum(all(r["active"][name]) for r in group)
            data["new_wrong"]=[sum(x["pred"]["native"][k]["number"]==x["target"] and
                                    x["pred"][name][k]["number"]!=x["target"] for x in group) for k in (0,1)]
            data["disjoint_donor_candidate"]=sum(x["donor_target"] is not None and z["number"] in (x["donor_target"],x["donor_target"]+100)
                                                   for x in group for z in x["pred"][name])
            outcomes[name]=data
        by[kind]=dict(n=len(group),outcomes=outcomes)
    repaired=[x for x in rows if x["kind"]=="repair"]

    def contrast(a,b):
        g={}
        for x in repaired:
            yes=[int([t["number"] for t in x["pred"][m]]==[x["target"],x["foil"]]) for m in (a,b)]
            g[x["target"]]=g.get(x["target"],0)+yes[0]-yes[1]
        edges=(.025,.975) if a=="distant_borrow" else (.0125,.9875)
        return ci(list(g.values()),edges)

    comparisons={m:contrast("distant_borrow",m) for m in ("distant_safe","aligned_safe","geom","other_field")}
    target=by["repair"]["outcomes"]["distant_borrow"]
    by["transport_gate"]=bool(target["joint"]>=8 and target["joint_targets"]>=6 and all(r["interval"][0]>0 for r in comparisons.values()))
    by["transport_comparisons"]=comparisons
    by["drop_vs_geometry"]=contrast("drop_geom","drop")
    by["keep_vs_geometry"]=contrast("keep","keep_geom")
    powers=[p for r in repaired for p in r["subspace_energy"]]
    by["mean_heldout_energy"]=sum(powers)/len(powers) if powers else None
    return by


@torch.inference_mode()
def run(model,root):
    assert load(root/"NATIVE_SUMMARY.json")["gate"]
    m,tok=setup(model)
    f=Forward(m)
    source=root/"NATIVE.json"
    rows=load(source)["rows"]
    pool,_,picks=build_plan(tok,rows)
    matched=sum(p["disjoint"] is not None for p in picks)
    plan=[dict(id=p["id"], disjoint=p["disjoint"], aligned=p["aligned"]) for p in picks]
    gate=matched>=24 and 5*matched>=4*len(picks)
    save(root/"MATCHING.json",dict(n=len(picks),n_disjoint=matched,n_aligned=sum(x["aligned"] is not None for x in picks),
                                   donor_ready=len(pool)//2,gate=gate,plan=plan,native_sha256=sha(source)))
    print("matching",matched,len(picks),"gate",gate,flush=True)
    if not gate:
        return
    qdb,mat=learn(f,list(pool.values()),root)
    by=[(p,c) for p in picks for c in (p["repair"],p["safe"])]
    path=root/"MECHANISM_RAW.json"
    old=load(path)["rows"] if path.exists() else []
    assert [(r["id"],r["kind"]) for r in old]==[(c["id"],c["kind"]) for _,c in by[:len(old)]]
    for i in range(len(old),len(by),8):
        block=by[i:i+8]
        answers,records=interact(f,tok,block,qdb,mat)
        for j,(p,c) in enumerate(block):
            detail={a:v[j] for a,v in answers.items()}
            assert list(detail["native"])==c["native"]
            is_donor=int(p["disjoint"].split("-")[1]) if p["disjoint"] else None
            old.append(dict(id=c["id"],kind=c["kind"],target=c["target"],foil=c["foil"],
                            donor_target=is_donor,reach=c["reach"],pred=detail,**records[j]))
        save(path,dict(rows=old,matching_sha256=sha(root/"MATCHING.json"),subspace_sha256=sha(root/"SUBSPACE.json"),
                       native_sha256=sha(source)))
        print("transport",len(old),len(by),flush=True)
    report=summarize(old)
    save(root/"MECHANISM_SUMMARY.json",report)
    print(report,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)

