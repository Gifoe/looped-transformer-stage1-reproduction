"""Live attention transport between two rows, crossed with a local V transplant."""

import argparse
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence

from edge import short_hash
from interchange import terminated
from native import Forward,load,output,pad,save,setup,sha,timer
from nec_match import enroll,measure
from route_deck import DOMAINS
from route_sites import freeze
from route_stats import ARMS,DIRECTIONS,summarize
from rule import MAX_NEW,text_results
from value_test import shuffle

L = range(24)
ROUTING = {"route":1,"joint":1,"route_control":2,"joint_control":2,"route_full":3,"joint_full":3}
VALUE = {"value":1,"joint":1,"joint_full":1,"value_control":2,"joint_control":2}


@contextmanager
def attention(f,hook):
    original = f.module.eager_attention_forward

    def run(m,q,k,v,mask,scaling,dropout=0.,**kw):
        av,a = original(m,q,k,v,mask,scaling,dropout=dropout,**kw)
        aa,vv = hook(m.layer_idx,a,v)
        if aa is a and vv is v:
            return av,a
        view = f.module.repeat_kv(vv,m.num_key_value_groups)
        return (aa@view).transpose(1,2).contiguous(),aa

    f.module.eager_attention_forward = run
    try:
        yield
    finally:
        f.module.eager_attention_forward = original


def select_a(a,x):
    k=x.flatten(1)[:,None,None,:].expand(a.size(0),a.size(1),a.size(2),-1)
    return a.gather(-1,k).transpose(1,2).unflatten(-1,(4,3))


def select_v(v,x):
    r=torch.arange(len(x),device=x.device)[:,None]
    return v.transpose(1,2)[r,x.flatten(1)].unflatten(1,(4,3))


def native(f,examples,keys):
    bank={}
    for off in range(0,len(keys),18):
        group=keys[off:off+18]
        seq=[examples[i]["ids"]+list(p) for i,p in group]
        ids,at=pad(seq)
        x=torch.tensor([examples[i]["sites"] for i,_ in group],device=ids.device)
        ac,vc,count={},{},defaultdict(int)

        def capture(l,a,v):
            phase=count[l]
            count[l]+=1
            ac[phase,l]=select_a(a,x).sum(-1)
            vc[phase,l]=select_v(v,x)
            return a,v

        with attention(f,capture):
            states=f.run(ids,2)
        assert all(count[l]==2 for l in L)
        check=[output(f.model,z,at).argmax(-1).tolist() for z in states] if any(not p for _,p in group) else None
        for j,key in enumerate(group):
            i,p=key
            n=len(seq[j])
            if not p:
                b=examples[i]["native"]["original"]
                assert [x[j] for x in check]==[y["tokens"][0] for y in b]
            bank[key]=dict(ids=seq[j],start=states[0][j,:n],
                           a=[[ac[k,l][j,:n] for l in L] for k in (0,1)],
                           v=[[vc[k,l][j] for l in L] for k in (0,1)])
    return bank


def route_positions(a,sites,r,c,mode,begin,valid,audit):
    # Compact layout is batch, query, head, row, digit.
    mask=mode!=0
    rows=torch.where(mask)[0]
    a0=a[rows,:,begin:,:]
    sites=sites[rows]
    mode=mode[rows,None,None]
    live=select_a(a0,sites)
    live_mass=live.sum(-1)
    r,c=r[rows,begin:],c[rows,begin:]
    valid=valid[rows,:,None]
    donor_pair=c[...,:2].sum(-1)
    valid=valid & (donor_pair>0)
    prop=c[...,0]/torch.where(donor_pair>0,donor_pair,1.)
    requested=(r[...,:2].sum(-1)*prop-r[...,0])*valid
    other=torch.where(requested>=0,r[...,3],r[...,2])
    planned=torch.sign(requested)*torch.minimum(requested.abs(),other)
    use_other=(mode==2)
    use_full=(mode==3)
    chosen=torch.where(use_other[...,None],live_mass[...,2:],live_mass[...,:2])
    raw=torch.where(use_full,(chosen.sum(-1)*prop-chosen[...,0])*valid,planned)
    actual=torch.maximum(-chosen[...,0],torch.minimum(raw,chosen[...,1]))
    change=torch.stack([actual,-actual],dim=-1)
    edited_pair=(chosen+change).clamp_min(0.)
    next_mass=torch.cat([torch.where(use_other[...,None],live_mass[...,:2],edited_pair),
                         torch.where(use_other[...,None],edited_pair,live_mass[...,2:])],dim=-1)
    mass_ratio=next_mass/torch.where(live_mass>0,live_mass,1.)
    replacement=torch.where((live_mass>0)[...,None],live*mass_ratio[...,None],next_mass[...,None]/3)
    replacement=torch.where((next_mass==live_mass)[...,None],live,replacement)
    indexes=sites.flatten(1)[:,None,None,:].expand(a0.size(0),a0.size(1),a0.size(2),-1)
    changed=a.clone()
    changed[rows,:,begin:,:]=a0.scatter(-1,indexes,replacement.flatten(-2).transpose(1,2))
    masses=torch.stack([x.abs().sum(dim=(1,2)) for x in (requested,planned,actual)],-1)
    meter=a.new_zeros((a.size(0),3))
    meter[rows]=masses
    error=0.
    if audit:
        after=select_a(changed[rows,:,begin:,:],sites)
        source_check=after.sum(-1)
        torch.testing.assert_close(source_check,next_mass,rtol=8e-5,atol=1e-6)
        torch.testing.assert_close(source_check[...,:2].sum(-1),live_mass[...,:2].sum(-1),rtol=8e-5,atol=2e-6)
        torch.testing.assert_close(source_check[...,2:].sum(-1),live_mass[...,2:].sum(-1),rtol=8e-5,atol=2e-6)
        assert torch.equal(after[~valid],live[~valid])
        covered=torch.zeros((a0.size(0),a0.size(-1)),device=a0.device,dtype=torch.bool)
        covered.scatter_(1,sites.flatten(1),True)
        untouched=(~covered)[:,None,None,:].expand_as(a0)
        assert torch.equal(changed[rows,:,begin:,:][untouched],a0[untouched])
        assert torch.equal(a[~mask],changed[~mask])
        error=float((source_check-next_mass).abs().max())
    return changed,meter,error


def v_positions(v,slots,clean,donor,mode,audit):
    new=v.transpose(1,2).clone()
    total=new.new_zeros((2,),dtype=torch.long)
    for kind,region in ((1,slice(0,2)),(2,slice(2,4))):
        i=torch.where(mode==kind)[0]
        if not len(i):
            continue
        k=slots[i,region].flatten(1)
        if kind==1:
            x=donor[i,region]
        else:
            a=clean[i,region]
            move=donor[i,region]-a
            d=donor[i,:2]-clean[i,:2]
            length=move.norm(dim=-1,keepdim=True)
            zero=length<=1e-8
            # Used only at slots with no natural unrelated direction.
            rotated=shuffle(a,d)-a if audit or bool(zero.any()) else d
            change=torch.where(zero,rotated,move*(d.norm(dim=-1,keepdim=True)/length.clamp_min(1e-8)))
            if audit:
                torch.testing.assert_close(change.norm(dim=-1),d.norm(dim=-1),rtol=9e-5,atol=1e-5)
            x=a+change
            total+=torch.stack([zero.sum(),zero.new_tensor(zero.numel(),dtype=torch.long)])
        new[i[:,None],k]=x.flatten(1,2)
        if audit:
            assert torch.equal(new[i[:,None],k],x.flatten(1,2))
            covered=torch.zeros((len(i),new.size(1)),device=new.device,dtype=torch.bool)
            covered.scatter_(1,k,True)
            assert torch.equal(new[i][~covered],v.transpose(1,2)[i][~covered])
    return new.transpose(1,2),total


def step(f,examples,req,audit=False):
    bank=native(f,examples,list(dict.fromkeys((i,p) for i,_,_,p in req)))
    pred,meters={},[]
    fallback=[0,0]
    errors=dict(mass=0.,write=0.,attention=0.,checks=0)
    for phase,d in enumerate(DIRECTIONS):
        tasks=[r for r in req if r[1]==d]
        for off in range(0,len(tasks),24):
            group=tasks[off:off+24]
            refs=[bank[i,p] for i,_,_,p in group]
            ids,at=pad([r["ids"] for r in refs])
            start=pad_sequence([r["start"] for r in refs],batch_first=True) if phase else None
            spans=torch.tensor([examples[i]["sites"] for i,_,_,_ in group],device=ids.device)
            after=spans.amax(dim=(1,2))+1
            begin=int(after.min())
            query=torch.arange(begin,ids.size(1),device=ids.device)[None,:]
            valid=(query>=after[:,None]) & (query<=at[:,None])
            rm=torch.tensor([ROUTING.get(a,0) for _,_,a,_ in group],device=ids.device)
            vm=torch.tensor([VALUE.get(a,0) for _,_,a,_ in group],device=ids.device)
            do_r=any(a in ROUTING for _,_,a,_ in group)
            do_v=any(a in VALUE for _,_,a,_ in group)
            log={}
            control=torch.zeros((2,),device=ids.device,dtype=torch.long)
            size=len(group)

            def edit(l,a,v):
                if do_r:
                    r,c=[pad_sequence([z["a"][k][l] for z in refs],batch_first=True) for k in (phase,1-phase)]
                    a,m,e=route_positions(a,spans,r,c,rm,begin,valid,audit)
                    log[l]=m
                    errors["mass"]=max(errors["mass"],e)
                if do_v:
                    x,y=[torch.stack([z["v"][k][l] for z in refs]) for k in (phase,1-phase)]
                    v,t=v_positions(v,spans,x,y,vm,audit)
                    control.add_(t)
                if audit:
                    row=torch.arange(size,device=ids.device)
                    expanded=f.module.repeat_kv(v,f.layers[l].self_attn.num_key_value_groups)
                    short=(a[row,:,at,:].unsqueeze(-2)@expanded).squeeze(-2)
                    full=(a@expanded).transpose(1,2)[row,at]
                    torch.testing.assert_close(short,full,rtol=7e-5,atol=3e-5)
                    errors["attention"]=max(errors["attention"],float((short-full).abs().max()))
                    errors["checks"]+=size
                return a,v

            with attention(f,edit):
                h=f.run(ids,1,start=start)[0]
            nx=output(f.model,h,at).argmax(-1).tolist()
            pred.update((r[:-1],z) for r,z in zip(group,nx,strict=True))
            fallback=[a+b for a,b in zip(fallback,control.tolist(),strict=True)]
            if log:
                dose=torch.stack([log[j] for j in L],dim=1).tolist()
                for r,z in zip(group,dose,strict=True):
                    i,_,arm,_=r
                    if arm in ROUTING:
                        x=examples[i]
                        meters.append((f"{x['domain']}/{x['kind']}/{d}/{arm}",z))
    return pred,meters,fallback,errors


def sum_meter(total,key,a):
    if key not in total:
        total[key]=a
    else:
        total[key]=[[x+y for x,y in zip(r,s,strict=True)] for r,s in zip(total[key],a,strict=True)]


def decode(f,tok,examples):
    active={(i,d,a):[] for i in range(len(examples)) for d in DIRECTIONS for a in ARMS}
    done,totals={},{}
    fallback=[0,0]
    for _ in range(MAX_NEW):
        if not active:
            break
        nxt,meter,v,_=step(f,examples,[(*k,tuple(p)) for k,p in active.items()])
        fallback=[a+b for a,b in zip(fallback,v,strict=True)]
        for x,u in meter:
            sum_meter(totals,x,u)
        for k,p in list(active.items()):
            p.append(nxt[k])
            if terminated(tok,p):
                done[k]=active.pop(k)
    done.update(active)
    result=dict(zip(done,text_results(tok,list(done.values())),strict=True))
    out=[{d:{a:result[i,d,a] for a in ARMS} for d in DIRECTIONS} for i in range(len(examples))]
    for r,z in zip(examples,out,strict=True):
        assert [z[d]["identity"] for d in DIRECTIONS]==r["native"]["original"]
    return out,totals,fallback


def auditors(examples):
    return {d:{g:sorted((r["id"] for r in examples if (r["domain"],r["kind"])==(d,g)),
                        key=lambda x:short_hash("row routing audit",x))[:4] for g in ("repair","stable")} for d in DOMAINS}


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    started=timer()
    src=root/"NATIVE.json"
    rows=enroll(load(src)["rows"])
    digest=sha(src)
    mf=root/"MARGINS.json"
    if not mf.exists():
        save(mf,dict(native_sha256=digest,rows=measure(f,tok,rows)))
    saved=load(mf)
    assert saved["native_sha256"]==digest
    cases,match,eligible=freeze(tok,rows,saved["rows"])
    audit=auditors(cases)
    inspect={x for u in audit.values() for v in u.values() for x in v}
    lock=root/"COHORT.json"
    plan=dict(native_sha256=digest,margins_sha256=sha(mf),matched=match,eligible=eligible,
              cases=[r["id"] for r in cases],audit=audit)
    if lock.exists():
        assert load(lock)==plan
    else:
        save(lock,plan)
    print("routing frozen pairs",{k:len(v) for k,v in match.items()},flush=True)
    target=root/"ROUTE_RAW.json"
    data=load(target) if target.exists() else dict(rows=[],cohort_sha256=sha(lock),elapsed=0.,dose={},fallback=[0,0])
    out=data["rows"]
    assert data["cohort_sha256"]==sha(lock) and [r["id"] for r in out]==plan["cases"][:len(out)]
    elapsed=data["elapsed"]
    for off in range(len(out),len(cases),10):
        batch=cases[off:off+10]
        result,totals,fallback=decode(f,tok,batch)
        for r,z in zip(batch,result,strict=True):
            q=dict({k:v for k,v in r.items() if k!="ids"},pred=z)
            if r["id"] in inspect:
                _,_,_,q["audit"]=step(f,[r],[(0,d,a,()) for d in DIRECTIONS for a in ARMS],audit=True)
            out.append(q)
        for key,z in totals.items():
            sum_meter(data["dose"],key,z)
        data.update(elapsed=elapsed+timer()-started,peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1),
                    fallback=[x+y for x,y in zip(data["fallback"],fallback,strict=True)])
        save(target,data)
        print("routing replay",len(out),len(cases),flush=True)
    z=summarize(out,match)
    save(root/"ROUTE_SUMMARY.json",z)
    print({k:v for k,v in z.items() if k!="domains"},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)
