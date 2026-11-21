"""Paired full-dose attention edits with exactly matched source debits."""

import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn.functional import linear
from torch.nn.utils.rnn import pad_sequence

from dest_deck import DOMAINS
from dest_stats import ARMS,DIRECTIONS,summarize
from edge import short_hash
from interchange import terminated
from native import Forward,load,output,pad,save,setup,sha,timer
from nec_match import enroll,measure
from route_sites import freeze
from route_test import L,attention,route_positions,select_a,select_v
from rule import MAX_NEW,text_results


def snapshots(f,examples,keys,calibration=False):
    bank={}
    for off in range(0,len(keys),20):
        group=keys[off:off+20]
        seq=[examples[i]["ids"]+list(p) for i,p in group]
        ids,at=pad(seq)
        sites=torch.tensor([examples[i]["sites"] for i,_ in group],device=ids.device)
        c,v,count={},{},defaultdict(int)

        def capture(l,a,x):
            k=count[l]
            count[l]+=1
            c[k,l]=select_a(a,sites)
            if calibration:
                v[k,l]=select_v(x,sites)
            return a,x

        with attention(f,capture):
            hs=f.run(ids,2)
        assert all(count[l]==2 for l in L)
        check=[output(f.model,z,at).argmax(-1).tolist() for z in hs] if any(not p for _,p in group) else None
        for j,key in enumerate(group):
            i,p=key
            n=len(seq[j])
            ref=dict(ids=seq[j],start=hs[0][j,:n],a=[[c[k,l][j,:n] for l in L] for k in (0,1)])
            if calibration:
                ref["v"]=[[v[k,l][j] for l in L] for k in (0,1)]
            if not p:
                orig=examples[i]["native"]["original"]
                assert [x[j] for x in check]==[x["tokens"][0] for x in orig]
            bank[key]=ref
    return bank


def recipe(base,donor,valid,third):
    # Compact layout is batch, query, head, row, digit.
    b=base[...,:2,:]
    weight=b.sum(-1)
    new=donor[...,:2,:].sum(-1)
    total=new.sum(-1)
    share=new[...,0]/torch.where(total>0,total,1.)
    d=(weight.sum(-1)*share-weight[...,0])*(valid[...,None] & (total>0))
    d=torch.maximum(-weight[...,0],torch.minimum(d,weight[...,1]))
    source=(d>=0).long()
    target=1-source
    width=source.shape
    digit_shape=source[...,None,None].expand(*width,1,3)
    source_vec=b.gather(-2,digit_shape).squeeze(-2)
    target_vec=b.gather(-2,target[...,None,None].expand_as(digit_shape)).squeeze(-2)

    def fractions(x):
        y=x.sum(-1,keepdim=True)
        return torch.where(y>0,x/torch.where(y>0,y,1.),torch.full_like(x,1/3))

    minus=-d.abs()[...,None]*fractions(source_vec)
    plus=d.abs()[...,None]*fractions(target_vec)
    debit=torch.zeros_like(base).scatter(-2,digit_shape,minus.unsqueeze(-2))
    natural=debit.scatter(-2,target[...,None,None].expand_as(digit_shape),plus.unsqueeze(-2))
    alt=debit.scatter(-2,third[:,None,None,None,None].expand_as(digit_shape),plus.unsqueeze(-2))
    return natural,alt,minus,source,plus


def projection(f,l,v,a):
    layer=f.layers[l].self_attn
    v=v.repeat_interleave(layer.num_key_value_groups,dim=-2)
    out=torch.einsum("bqhrd,brdhe->bqhe",a,v).flatten(-2)
    return linear(out,layer.o_proj.weight)


@torch.inference_mode()
def calibrate(f,examples):
    records=[]
    for off in range(0,len(examples),14):
        cases=examples[off:off+14]
        q=snapshots(f,cases,[(j,()) for j in range(len(cases))],calibration=True)
        refs=[q[j,()] for j in range(len(cases))]
        device=refs[0]["start"].device
        lengths=[len(x["ids"]) for x in refs]
        start=torch.tensor([max(map(max,x["sites"]))+1 for x in cases],device=device)
        begin=int(start.min())
        at=torch.arange(begin,max(lengths),device=device)[None,:]
        valid=(at>=start[:,None]) & (at<torch.tensor(lengths,device=device)[:,None])
        doses=[]
        for phase in (0,1):
            ss=torch.zeros((len(cases),3),device=device,dtype=torch.float64)
            for l in L:
                r,s=[pad_sequence([x["a"][k][l] for x in refs],batch_first=True)[:,begin:] for k in (phase,1-phase)]
                v=torch.stack([x["v"][phase][l] for x in refs])
                two=torch.full_like(start,2)
                true,alt,_,_,_=recipe(r,s,valid,two)
                _,alt1,_,_,_=recipe(r,s,valid,two+1)
                for k,z in enumerate((true,alt,alt1)):
                    p=projection(f,l,v,z)
                    ss[:,k]+=p.square().sum(dim=(1,2),dtype=torch.float64)
            doses.append(ss.sqrt().tolist())
        for i,r in enumerate(cases):
            a,b=[x[i] for x in doses]
            assert all(x>0 for x in a+b)
            records.append(dict(id=r["id"],dose=[a,b],**selection(a,b)))
        print("destination projected reference",len(records),len(examples),flush=True)
    return records


def selection(a,b):
    import math
    ratios=[[v[k]/v[0] for v in (a,b)] for k in (1,2)]
    pick=min(range(2),key=lambda i:max(abs(math.log(x)) for x in ratios[i]))
    chosen=ratios[pick]
    return dict(third=pick+2,ratio=chosen,projected_match=all(.8<=v<=1.25 for v in chosen))


def anchor(a,sites,b,donor,mode,third,begin,valid,audit):
    natural,alt,debit,source,credit=recipe(b,donor,valid,third)
    delta=torch.where((mode==1)[:,None,None,None,None],natural,
                      torch.where((mode==2)[:,None,None,None,None],alt,0.))
    desired=(b+delta).clamp_min(0.)
    a0=a[:,:,begin:,:]
    live=select_a(a0,sites)
    total=live.sum(dim=(-1,-2))
    ref=b.sum(dim=(-1,-2))
    scale=(1-ref)/(1-total).clamp_min(1e-9)
    rotated=a0*scale.transpose(1,2).unsqueeze(-1)
    ix=sites.flatten(1)[:,None,None,:].expand(a0.size(0),a0.size(1),a0.size(2),-1)
    inject=rotated.scatter(-1,ix,desired.flatten(-2).transpose(1,2))
    mask=(valid & (mode!=3)[:,None])[:,None,:,None]
    changed=a.clone()
    changed[:,:,begin:]=torch.where(mask,inject,a0)
    result=dict(debit=0.,l2=0.,probability=0.)
    if audit:
        gathered=select_a(changed[:,:,begin:],sites)
        active=(valid & (mode!=3)[:,None])
        assert torch.equal(gathered[active],desired[active])
        assert torch.equal(gathered[~active],live[~active])
        src=source[...,None,None].expand(*source.shape,1,3)
        dd1=natural.gather(-2,src).squeeze(-2)
        dd2=alt.gather(-2,src).squeeze(-2)
        assert torch.equal(dd1,debit) and torch.equal(dd2,debit)
        installed=[(b+z).clamp_min(0.)-b for z in (natural,alt)]
        assert torch.equal(installed[0].gather(-2,src),installed[1].gather(-2,src))
        r1,r2=[z.square().sum((-1,-2)) for z in installed]
        torch.testing.assert_close(r1,r2,rtol=8e-6,atol=1e-7)
        t,c=[z.square().sum((-1,-2)) for z in (natural,alt)]
        torch.testing.assert_close(t,c,rtol=8e-6,atol=1e-7)
        for z in (natural,alt):
            torch.testing.assert_close(z.abs().sum((-1,-2)),2*credit.sum(-1),rtol=8e-6,atol=1e-7)
        others=torch.ones((len(sites),a.size(-1)),device=a.device,dtype=torch.bool)
        others.scatter_(-1,sites.flatten(1),False)
        un=others[:,None,None,:] & mask
        assert torch.equal(changed[:,:,begin:][un.expand_as(a0)],rotated[un.expand_as(a0)])
        s=changed.sum(-1)
        torch.testing.assert_close(s,a.sum(-1),rtol=7e-5,atol=3e-6)
        result=dict(debit=float((dd1-dd2).abs().max()),l2=float((r1-r2).abs().max()),
                    probability=float((s-a.sum(-1)).abs().max()))
    return changed,result


def step(f,examples,requests,audit=False):
    bank=snapshots(f,examples,list(dict.fromkeys((i,p) for i,_,_,p in requests)))
    out={}
    notes=dict(debit=0.,l2=0.,probability=0.,attention=0.,checks=0)
    codes={"identity":0,"true":1,"redirect":2,"live_full":3}
    for phase,d in enumerate(DIRECTIONS):
        todo=[q for q in requests if q[1]==d]
        for off in range(0,len(todo),28):
            work=todo[off:off+28]
            refs=[bank[i,p] for i,_,_,p in work]
            ids,at=pad([q["ids"] for q in refs])
            start=pad_sequence([q["start"] for q in refs],batch_first=True) if phase else None
            sites=torch.tensor([examples[i]["sites"] for i,_,_,_ in work],device=ids.device)
            choice=torch.tensor([examples[i]["third"] for i,_,_,_ in work],device=ids.device)
            mode=torch.tensor([codes[a] for _,_,a,_ in work],device=ids.device)
            full_mode=torch.where(mode==3,mode,0)
            do_full=any(a=="live_full" for _,_,a,_ in work)
            do_anchor=any(a!="live_full" for _,_,a,_ in work)
            after=sites.amax(dim=(1,2))+1
            begin=int(after.min())
            q=torch.arange(begin,ids.size(1),device=ids.device)[None,:]
            valid=(q>=after[:,None])&(q<=at[:,None])
            size=len(work)

            def edit(l,a,v):
                base,donor=[pad_sequence([r["a"][k][l] for r in refs],batch_first=True) for k in (phase,1-phase)]
                if do_full:
                    a,_,_=route_positions(a,sites,base.sum(-1),donor.sum(-1),full_mode,begin,valid,audit)
                if do_anchor:
                    a,e=anchor(a,sites,base[:,begin:],donor[:,begin:],mode,choice,begin,valid,audit)
                    if audit:
                        for k,x in e.items():
                            notes[k]=max(notes[k],x)
                if audit:
                    row=torch.arange(size,device=ids.device)
                    expanded=f.module.repeat_kv(v,f.layers[l].self_attn.num_key_value_groups)
                    selected=(a[row,:,at,:].unsqueeze(-2)@expanded).squeeze(-2)
                    full=(a@expanded).transpose(1,2)[row,at]
                    torch.testing.assert_close(selected,full,rtol=7e-5,atol=3e-5)
                    notes["attention"]=max(notes["attention"],float((selected-full).abs().max()))
                    notes["checks"]+=size
                return a,v

            with attention(f,edit):
                h=f.run(ids,1,start=start)[0]
            nxt=output(f.model,h,at).argmax(-1).tolist()
            out.update((r[:-1],v) for r,v in zip(work,nxt,strict=True))
    return out,notes


def decode(f,tok,examples):
    active={(i,d,a):[] for i in range(len(examples)) for d in DIRECTIONS for a in ARMS}
    done={}
    for _ in range(MAX_NEW):
        if not active:
            break
        n,_=step(f,examples,[(*k,tuple(s)) for k,s in active.items()])
        for k,s in list(active.items()):
            s.append(n[k])
            if terminated(tok,s):
                done[k]=active.pop(k)
    done.update(active)
    z=dict(zip(done,text_results(tok,list(done.values())),strict=True))
    answers=[{d:{a:z[i,d,a] for a in ARMS} for d in DIRECTIONS} for i in range(len(examples))]
    for r,a in zip(examples,answers,strict=True):
        assert [a[d]["identity"] for d in DIRECTIONS]==r["native"]["original"]
    return answers


def auditors(rows):
    return {d:{g:sorted((r["id"] for r in rows if (r["domain"],r["kind"])==(d,g)),
                        key=lambda x:short_hash("destination audit",x))[:4] for g in ("repair","stable")} for d in DOMAINS}


def prepare(row,cal):
    assert row["id"]==cal["id"]
    v=row["example"]["original"]
    base=1 if row["domain"]=="standard" else 5
    target,foil=[v["values"].index(row[k])+base for k in ("target","foil")]
    keys=[k for k in v["order"] if k not in (target,foil)]
    third_value=v["values"][keys[cal["third"]-2]-base]
    return dict(row,third=cal["third"],projected_match=cal["projected_match"],third_value=third_value)


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    started=timer()
    src=root/"NATIVE.json"
    digest=sha(src)
    rows=enroll(load(src)["rows"])
    mf=root/"MARGINS.json"
    if not mf.exists():
        save(mf,dict(native_sha256=digest,rows=measure(f,tok,rows)))
    mr=load(mf)
    assert mr["native_sha256"]==digest
    plan,match,eligible=freeze(tok,rows,mr["rows"])
    ref=root/"PROJECTION.json"
    if not ref.exists():
        save(ref,dict(margins_sha256=sha(mf),rows=calibrate(f,plan)))
    calibration=load(ref)
    assert calibration["margins_sha256"]==sha(mf)
    cases=[prepare(r,c) for r,c in zip(plan,calibration["rows"],strict=True)]
    audit=auditors(cases)
    select={i for d in audit.values() for g in d.values() for i in g}
    memo=dict(native_sha256=digest,margins_sha256=sha(mf),projection_sha256=sha(ref),matched=match,eligible=eligible,
              cases=[r["id"] for r in cases],audit=audit)
    lock=root/"COHORT.json"
    if lock.exists():
        assert load(lock)==memo
    else:
        save(lock,memo)
    counts={d:sum(r["domain"]==d and r["kind"]=="repair" and r["projected_match"] for r in cases) for d in DOMAINS}
    print("destination frozen pairs",{d:len(match[d]) for d in DOMAINS},"projected",counts,flush=True)
    dest=root/"DEST_RAW.json"
    data=load(dest) if dest.exists() else dict(rows=[],cohort_sha256=sha(lock),elapsed=0.)
    out=data["rows"]
    assert data["cohort_sha256"]==sha(lock) and [r["id"] for r in out]==memo["cases"][:len(out)]
    elapsed=data["elapsed"]
    for off in range(len(out),len(cases),16):
        batch=cases[off:off+16]
        answer=decode(f,tok,batch)
        for r,p in zip(batch,answer,strict=True):
            q=dict({k:v for k,v in r.items() if k!="ids"},pred=p)
            if r["id"] in select:
                _,q["audit"]=step(f,[r],[(0,d,a,()) for d in DIRECTIONS for a in ARMS],audit=True)
            out.append(q)
        data.update(elapsed=elapsed+timer()-started,peak_allocated_mib=round(torch.cuda.max_memory_allocated()/2**20,1))
        save(dest,data)
        print("destination replay",len(out),len(cases),flush=True)
    z=summarize(out,match)
    save(root/"DEST_SUMMARY.json",z)
    print({k:v for k,v in z.items() if k!="domains"},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)
