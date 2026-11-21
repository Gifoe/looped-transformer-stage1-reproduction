"""Replay actual Ouro blocks and intervene on their newly generated mixed differences."""
import argparse
import contextlib
from pathlib import Path

import torch

from borrow import ci
from interchange import complete, fit_margin, independent_like
from mechanism import matched_plane, same_norm
from native import Forward, load, output, pad, save, setup, sha
from native_deck import all_other_K1_correct, is_repair
from readout import forms
from rule import greedy

LAYERS=(13,14,15,16)
MODES=("native","generated","geometry","propagated","clipped","attention","attention_geom",
       "mlp","mlp_geom","phase2","phase2_geom","paired_generated","paired_geometry")


def mixed(v):
    s=v.reshape(-1,4,*v.shape[1:])
    return s[:,3]-s[:,1]-s[:,2]+s[:,0]


def additive(v):
    s=v.reshape(-1,4,*v.shape[1:])
    return s[:,1]+s[:,2]-s[:,0]


class Trace:
    def __init__(self,f):
        self.f=f
        self.kw={}
        self.saved={}

    def put(self,key,name,x):
        if key[1] in LAYERS and name in ("input","output","a_write","m_write"):
            self.saved[key,name]=x.detach()

    @contextlib.contextmanager
    def phases(self):
        hooks=[]
        try:
            for l in LAYERS:
                def grab(_,args,kw,layer=l):
                    self.kw[kw["current_ut"],layer]=dict(kw)
                hooks.append(self.f.layers[l].register_forward_pre_hook(grab,with_kwargs=True))
            yield
        finally:
            for h in hooks: h.remove()


def replay(block,x,kw,at):
    captured={}
    which=torch.arange(len(at),device=at.device)
    hooks=[]
    for key,mod in (("a",block.input_layernorm_2),("m",block.post_attention_layernorm_2)):
        def capture(_,args,result,k=key):
            captured[k]=result[which,at].detach()
        hooks.append(mod.register_forward_hook(capture))
    try:
        z=block(x,**kw)
    finally:
        for h in hooks: h.remove()
    return z,captured


def decomposition(f,p,tr,k,l):
    get=lambda n:tr.saved[(k,l),n]
    ids=torch.arange(len(p),device=p.device)
    atj=ids[3::4]
    ats=p[atj]
    native=get("input")
    real=get("output")
    got,branches=replay(f.layers[l],native,tr.kw[k,l],p)
    assert torch.equal(got,real)
    for n,letter in (("a_write","a"),("m_write","m")):
        assert torch.equal(branches[letter],get(n)[ids,p])
    nat=real[ids,p]
    quartet=native.reshape(-1,4,*native.shape[1:])
    simple=additive(native)
    norm4=quartet.norm(dim=-1)
    low=norm4.min(dim=1).values
    high=norm4.max(dim=1).values
    size=simple.norm(dim=-1)
    loc=torch.arange(len(ats),device=p.device)
    valid=torch.arange(simple.shape[1],device=p.device)[None,:]<=ats[:,None]
    outside=((size<low)|(size>high))&valid
    radial=(size.clamp(min=low,max=high)/size.clamp_min(1e-12))
    clipped=simple*radial.unsqueeze(-1)
    values={}
    for key,argument in (("full",simple),("clipped",clipped)):
        # Replicate over the original batch dimensions to replay the exact mask/RoPE.
        arr=argument.repeat_interleave(4,dim=0)
        fx,w=replay(f.layers[l],arr,tr.kw[k,l],p)
        quad=nat.view(-1,4,nat.shape[-1])
        fs=fx[atj,ats]
        sinput=(argument-simple)[loc,ats]
        gen=fs-quad[:,1]-quad[:,2]+quad[:,0]-sinput
        prop=quad[:,3]-fs+sinput
        residual=mixed(nat)
        na=nm=None
        if key=="full":
            for letter,n in (("a","a_write"),("m","m_write")):
                h=get(n)[ids,p].view(-1,4,nat.shape[-1])
                g=w[letter][atj]-h[:,1]-h[:,2]+h[:,0]
                if letter=="a": na=g
                else: nm=g
            torch.testing.assert_close(gen,na+nm,rtol=8e-4,atol=2.5e-4)
        torch.testing.assert_close(gen+prop,residual,rtol=8e-4,atol=2.5e-4)
        values[key]=(gen,prop,na,nm)
    info=dict(outside=(outside.sum(dim=1)/valid.sum(dim=1)).tolist(),
              readout_ratio=(size[loc,ats]/quartet[loc,3,ats].norm(dim=-1)).tolist(),
              readout_outside=outside[loc,ats].tolist(),
              generated=values["full"][0].norm(dim=-1).tolist(),
              propagated=values["full"][1].norm(dim=-1).tolist(),
              interaction=mixed(nat).norm(dim=-1).tolist(),
              a=values["full"][2].norm(dim=-1).tolist(),m=values["full"][3].norm(dim=-1).tolist())
    values["writes"]=(get("a_write")[atj,ats],get("m_write")[atj,ats])
    return real[atj,ats],values,info


class Intervene:
    def __init__(self,f,l,at,value):
        self.f=f
        self.l=l
        self.at=at
        self.value=value

    def __enter__(self):
        idx=torch.arange(len(self.at),device=self.at.device)
        def hook(_,args,out):
            new=out.clone()
            new[idx,self.at]=self.value
            return new
        self.hook=self.f.layers[self.l].register_forward_hook(hook)
        return self

    def __exit__(self,*_):
        self.hook.remove()


def quad(f,cases):
    for j in range(max(len(c["prefix"]) for c in cases)):
        active=[(ids+c["prefix"][:j],c["prefix"][j]) for c in cases if len(c["prefix"])>j for ids in c["quad"]]
        prefix,at=pad([q for q,_ in active])
        expected=[q for _,q in active]
        for depth in (1,2):
            assert output(f.model,f.run(prefix,depth)[-1],at).argmax(-1).tolist()==expected
    ids,p=pad([x+c["prefix"] for c in cases for x in c["quad"]])
    tr=Trace(f)
    with tr.phases():
        hh=f.run(ids,2,record=tr)
    ix=torch.arange(3,len(p),4,device=p.device)
    site={l:[decomposition(f,p,tr,k,l) for k in (0,1)] for l in LAYERS}
    return ids[ix],p[ix],(hh[0][ix],hh[1][ix]),site


def geom(a,b,n,cases,l,seed):
    return matched_plane(a,b-a,n,independent_like(n,cases,l,seed))


def arms(data,cases,l):
    a,b=[r[0] for r in data]
    formed=[r[1]["full"][0] for r in data]
    result=[]
    for k,(r,p,_) in enumerate(data):
        n,carry,atn,mlp=p["full"]
        competing,_=same_norm(carry,n)
        result.append(dict(generated=n,geometry=geom(a,b,n,cases,l,721),propagated=competing,clipped=p["clipped"][0],
                           attention=atn,attention_geom=geom(a,b,atn,cases,l,722),
                           mlp=mlp,mlp_geom=geom(a,b,mlp,cases,l,723),
                           phase2=formed[1],phase2_geom=geom(a,b,formed[1],cases,l,724)))
    return result


def replace(f,ids,p,h,l,k,v):
    with Intervene(f,l,p,v):
        return f.run(ids,1,start=h[0] if k else None)[0]


def condition(tok,rows,split):
    lookup={}
    repair=[]
    for g in rows:
        if g["split"]!=split:
            continue
        z=forms(tok,g)
        if is_repair(g):
            repair.append(z)
        if all(z["pair"][1]["reach"]) and not tok.decode(z["pair"][1]["prefix"],skip_special_tokens=True).strip():
            lookup.setdefault(g["target"],[]).append(z)
    return sorted(repair,key=lambda a:a["g"]["id"]),lookup


def margins(f,cases):
    ids,p=pad([c["ids"]+c["prefix"] for c in cases])
    want=torch.tensor([c["labels"] for c in cases],device=ids.device)
    h=f.run(ids,2)
    m=[fit_margin(f,z,p,want).tolist() for z in h]
    return list(zip(*m,strict=True))


def matched(f,tok,root,split):
    r=load(root/"NATIVE.json")["rows"]
    q,s=condition(tok,r,split)
    safe=[z["pair"][1] for t,vals in s.items() for z in vals]
    qms={}
    for k in range(0,len(safe),12):
        grp=safe[k:k+12]
        qms.update({x["id"]:v for x,v in zip(grp,margins(f,grp),strict=True)})
    result=[]
    for j in range(0,len(q),12):
        rec=q[j:j+12]
        b=[g["pair"][0] for g in rec]
        for row,w in zip(rec,margins(f,b),strict=True):
            options=s.get(row["g"]["target"],[])
            def rank(option):
                m=qms[option["g"]["id"]]
                return (option["g"]["pred"][1][4]["number"]!=row["g"]["target"],
                        (abs(m[0])-abs(w[0]))**2+(m[1]-w[1])**2,option["g"]["id"])
            if not options:
                other=row["pair"][1]
                error=[None,None]
            else:
                p=min(options,key=rank)
                other=p["pair"][1]
                v=qms[p["g"]["id"]]
                error=[abs(abs(w[0])-abs(v[0])),abs(w[1]-v[1])]
            result.append((row,other,w,error))
    return result


def ranked(f,tok,root,choice):
    eligible=[r["pair"][0] for r,_,_,_ in choice if r["aligned"]]
    scores={l:[] for l in LAYERS}
    for j in range(0,len(eligible),7):
        b=eligible[j:j+7]
        ids,p,h,by=quad(f,b)
        wanted=torch.tensor([z["labels"] for z in b],device=p.device)
        for l,parts in by.items():
            control=arms(parts,b,l)
            a=[]
            for k in (0,1):
                r=parts[k][0]
                candidates=[]
                for name in ("generated","geometry"):
                    val=r+control[k][name]*(1 if k==0 else -1)
                    z=replace(f,ids,p,h,l,k,val)
                    candidates.append(fit_margin(f,z,p,wanted))
                a.append(((candidates[0]-candidates[1])*(1 if k==0 else -1)).tolist())
            scores[l].extend(dict(id=c["id"],margin_vs_geometry=[a[0][i],a[1][i]]) for i,c in enumerate(b))
    stats={}
    for l,v in scores.items():
        stats[l]=[sum(c["margin_vs_geometry"][k] for c in v)/len(v) if v else 0. for k in (0,1)]
    chosen=max(LAYERS,key=lambda l:stats[l][1])
    record=dict(layer=chosen,means=stats,raw=scores,natives_sha256=sha(root/"NATIVE.json"))
    save(root/"LOCK.json",record)
    print("selected physical layer",chosen+1,stats,flush=True)
    return record


def evaluate(f,tok,cases,main):
    ids,p,h,by=quad(f,cases)
    for k in (0,1):
        r=greedy(f,tok,[c["ids"] for c in cases],k+1)
        assert [o["tokens"] for o in r]==[z["native"][k]["tokens"] for z in cases]
        original=output(f.model,h[k],p).argmax(-1).tolist()
        assert all(not c["reach"][k] or digit==c["native"][k]["tokens"][len(c["prefix"])]
                   for c,digit in zip(cases,original,strict=True))
        identity=replace(f,ids,p,h,main,k,by[main][k][0])
        assert output(f.model,identity,p).argmax(-1).tolist()==original
    part=by[main]
    v=arms(part,cases,main)
    assert len(cases)%2==0 and all(a["kind"]=="repair" and b["kind"]=="safe" for a,b in zip(cases[::2],cases[1::2],strict=True))
    for k in (0,1):
        dose=torch.repeat_interleave(v[k]["generated"][::2],2,dim=0)
        v[k]["paired_generated"]=dose
        v[k]["paired_geometry"]=geom(part[0][0],part[1][0],dose,cases,main,771)
    answer={}
    for name in MODES:
        result=[]
        for k in (0,1):
            if name=="native":
                state=h[k]
            else:
                raw=part[k][0]+v[k][name]*(1 if k==0 else -1)
                state=replace(f,ids,p,h,main,k,raw)
            z=output(f.model,state,p).argmax(-1).tolist()
            result.append(complete(f,tok,cases,z,k))
        answer[name]=list(zip(*result,strict=True))
    phases={}
    for l,x in by.items():
        var=v if l==main else arms(x,cases,l)
        vals={}
        for name in ("generated","geometry"):
            if l==main:
                b=[o[1] for o in answer[name]]
            else:
                v2=x[1][0]-var[1][name]
                sample=replace(f,ids,p,h,l,1,v2)
                b=complete(f,tok,cases,output(f.model,sample,p).argmax(-1).tolist(),1)
            vals[name]=b
        phases[l]=vals
    info=[]
    for j in range(len(cases)):
        info.append(dict(outcomes={m:q[j] for m,q in answer.items()},
                         layers={str(l):{k:v[j] for k,v in d.items()} for l,d in phases.items()},
                         diagnostic={str(l):[{m:v[j] for m,v in t[2].items()} for t in data] for l,data in by.items()}))
    return info


def fit(rows):
    groups={}
    for key in ("repair","safe"):
        part=[x for x in rows if x["kind"]==key]
        read={}
        for m in MODES:
            c=dict(K1=0,K2=0,joint=0,K1_harm=0,K2_harm=0,safe_margin_K2=0)
            targets=set()
            for r in part:
                a,b=[o["number"] for o in r["outcomes"][m]]
                u,v=[o["number"] for o in r["outcomes"]["native"]]
                y,z=r["target"],r["foil"]
                c["K1"]+=a==y
                c["K2"]+=b==z
                c["joint"]+=(a,b)==(y,z)
                c["K1_harm"]+=u==y and a!=y
                c["K2_harm"]+=v==y and b!=y
                c["safe_margin_K2"]+=(b==z and r["margin_matched"])
                if (a,b)==(y,z): targets.add(y)
            c["joint_targets"]=len(targets)
            read[m]=c
        groups[key]=dict(n=len(part),aligned=sum(r["aligned"] for r in part),
                         matched_margin=sum(r["margin_matched"] for r in part),outputs=read)
    samples=[g for g in rows if g["kind"]=="repair"]
    paired={}
    for v,w in (("generated","geometry"),("generated","propagated"),("attention","attention_geom"),
                ("mlp","mlp_geom"),("phase2","phase2_geom")):
        bins={k:{} for k in ("K1","K2","joint")}
        for x in samples:
            z=x["target"]
            a=[]
            for m in (v,w):
                q,p=[o["number"] for o in x["outcomes"][m]]
                a.append((q==z,p==x["foil"],(q,p)==(z,x["foil"])))
            for k,x0,x1 in zip(bins,*a,strict=True):
                t=bins[k]; t[z]=t.get(z,0)+int(x0)-int(x1)
        paired[v+"_vs_"+w]={k:ci(list(g.values()),(.0125,.9875)) for k,g in bins.items()}
    layer_eval={}
    for l in LAYERS:
        here=[0,0]
        bin={}
        for r in samples:
            x=[r["layers"][str(l)][m]["number"]==r["foil"] for m in ("generated","geometry")]
            here=[old+v for old,v in zip(here,x,strict=True)]
            y=r["target"]; bin[y]=bin.get(y,0)+int(x[0])-int(x[1])
        layer_eval[str(l)]=dict(K2_foils=here,clustered=ci(list(bin.values()),(.00625,.99375)))
    m=groups["repair"]["outputs"]["generated"]
    g=paired["generated_vs_geometry"]
    h=paired["generated_vs_propagated"]
    receive={}
    for a,b in zip(rows[::2],rows[1::2],strict=True):
        x=a["target"]
        def excess(g):
            return int(g["outcomes"]["paired_generated"][1]["number"]==g["foil"])-int(g["outcomes"]["paired_geometry"][1]["number"]==g["foil"])
        receive[x]=receive.get(x,0)+excess(a)-excess(b)
    groups.update(paired=paired,per_layer=layer_eval,
                  same_dose_context=ci(list(receive.values()),(.0125,.9875)),
                  dual_gate=m["joint"]>=8 and m["joint_targets"]>=6 and all(x["interval"][0]>0 for x in g.values()),
                  K2_gate=m["K2"]>=8 and g["K2"]["interval"][0]>0 and h["K2"]["interval"][0]>0)
    return groups


@torch.inference_mode()
def run(model,root):
    assert load(root/"NATIVE_SUMMARY.json")["gate"]
    m,tok=setup(model)
    f=Forward(m)
    lock=root/"LOCK.json"
    s=load(lock) if lock.exists() else ranked(f,tok,root,matched(f,tok,root,"discovery"))
    pairs=matched(f,tok,root,"confirmation")
    save(root/"MATCHING.json",dict(pairs=[dict(id=r["g"]["id"],safe=t["id"],margin=w,difference=d)
                                        for r,t,w,d in pairs],native_sha256=sha(root/"NATIVE.json")))
    existing=root/"BLOCK_RAW.json"
    out=load(existing)["rows"] if existing.exists() else []
    all_cases=[(r["g"]["id"],c["kind"]) for r,s,_,_ in pairs for c in (r["pair"][0],s)]
    assert [(v["id"],v["kind"]) for v in out]==all_cases[:len(out)]
    for j in range(len(out)//2,len(pairs),4):
        raw=pairs[j:j+4]
        chunks=[dict(c,id=c["id"]) for r,t,_,_ in raw for c in (r["pair"][0],t)]
        yes=[r["aligned"] and all(t["reach"]) for r,t,_,_ in raw for _ in (0,1)]
        batch=[p for p,ok in zip(chunks,yes,strict=True) if ok]
        measured=iter(evaluate(f,tok,batch,s["layer"]) if batch else [])
        for i,(r,_,_,d) in enumerate(raw):
            for j,c in enumerate(chunks[2*i:2*i+2]):
                valid=yes[2*i+j]
                if valid:
                    meta=next(measured)
                else:
                    refs={k:c["native"] for k in MODES}
                    q={str(l):{m:c["native"][1] for m in ("generated","geometry")} for l in LAYERS}
                    meta=dict(outcomes=refs,layers=q,diagnostic={})
                assert list(meta["outcomes"]["native"])==c["native"]
                out.append(dict(id=r["g"]["id"],source_id=c["id"],kind=c["kind"],target=c["target"],foil=c["foil"],
                                aligned=valid,margin_matched=all(m is not None and m<=2. for m in d),
                                difference=d,three_clean=all_other_K1_correct(r["g"]),prefix=c["prefix"],**meta))
        save(existing,dict(rows=out,native_sha256=sha(root/"NATIVE.json"),selection_sha256=sha(lock)))
        print("block confirmation",len(out)//2,len(pairs),flush=True)
    r=fit(out)
    save(root/"BLOCK_SUMMARY.json",r)
    print(r,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

