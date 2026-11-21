"""Disjoint forward/readout effect transport and out-of-digit-banks projection."""
import argparse
from pathlib import Path

import torch

from borrow import ci
from interchange import complete, independent_like
from mechanism import Edit, Capture, LQ, LR, BANK_RANK, build_plan, labels, matched_plane, prepare, project, same_norm
from native import Forward, load, output, save, setup, sha
from state import rotate_about

MODES=("native","self","distant","distant_safe","aligned_safe","geometry",
       "keep_disjoint","keep_sham","drop_disjoint","drop_sham")


def probe(f,cases):
    seq,last,rec,h,fields=prepare(f,cases)
    perturb=[]
    outputs=[]
    for k in (0,1):
        got=Capture(last)
        result=f.run(seq,1,start=h[0] if k else None,record=got,
                     edit=Edit(last,query={LQ:(None,rec.get(1-k,"q"))}))[0]
        perturb.append(got.get(0,"r")-rec.get(k,"r"))
        outputs.append(result)
    return seq,last,rec,h,perturb,outputs


def matches(h,reference):
    return set((h,h+1)).isdisjoint(reference)


def donors(f,pool,root):
    keyed={}
    rank_rows={}
    for i in range(0,len(pool),12):
        pack=pool[i:i+12]
        *_,difference,__=probe(f,pack)
        for j,z in enumerate(pack):
            keyed[z["id"],z["kind"]]=tuple(v[j] for v in difference)
            if z["kind"]=="repair":
                rank_rows[z["id"]]=(labels(z),[v[j] for v in difference])
    bases={}
    compiled={}
    for digit in range(2,7):
        ident=[name for name,(digits,_) in rank_rows.items() if matches(digit,digits)]
        rows=[p for name in ident for p in rank_rows[name][1]]
        if len(rows)<BANK_RANK:
            compiled[digit]=dict(n=len(rows),eligible=False,ids=ident)
            continue
        _,s,v=torch.linalg.svd(torch.stack(rows),full_matrices=False)
        u=v[:BANK_RANK].contiguous()
        bases[digit]=u
        compiled[digit]=dict(n=len(rows),eligible=True,ids=ident,
                            energy=float(s[:BANK_RANK].square().sum()/s.square().sum()),vectors=u.tolist())
    save(root/"DISJOINT_BASES.json",compiled)
    return keyed,bases


def apply(f,tok,items,book,bases):
    pack=[v["repair"] for v in items]
    seq,last,tr,h,actual,benchmark=probe(f,pack)
    n=len(pack)
    traces=[]
    by={}
    for k in (0,1):
        base=tr.get(k,"r")
        own=actual[k]
        d=independent_like(base,pack,LR,221)
        active={}
        out={}
        for name in ("distant","distant_safe","aligned_safe"):
            region="aligned" if name.startswith("aligned") else "disjoint"
            typ="safe" if name.endswith("safe") else "repair"
            take=[v[region] for v in items]
            v=torch.stack([book[j,typ][k] if j else torch.zeros_like(own[i]) for i,j in enumerate(take)])
            sized,ok=same_norm(v,own)
            mask=ok&torch.tensor([v is not None for v in take],device=base.device)
            out[name]=sized
            active[name]=mask
        out["geometry"]=rotate_about(base,base+out["distant"],
                                     independent_like(base,pack,LR,222))-base
        active["geometry"]=active["distant"]
        p=torch.stack([project(own[j],bases[x["target"]//100]) if x["target"]//100 in bases else torch.zeros_like(own[j])
                       for j,x in enumerate(pack)])
        r=matched_plane(base,own,p,d)
        ok=torch.tensor([x["target"]//100 in bases for x in pack],device=base.device)
        out.update(keep_disjoint=p,keep_sham=r,drop_disjoint=own-p,drop_sham=own-r)
        for a in ("keep_disjoint","keep_sham","drop_disjoint","drop_sham"):
            active[a]=ok
        energies=(p.square().sum(dim=-1)/own.square().sum(dim=-1).clamp_min(1e-15)).tolist()
        traces.append(energies)
        for mode in MODES:
            if mode=="native":
                state=h[k]
            else:
                if mode=="self":
                    delta=own
                else:
                    delta=out[mode]*active[mode][:,None]
                state=f.run(seq,1,start=h[0] if k else None,edit=Edit(last,readout=base+delta))[0]
                if mode=="self":
                    choose=torch.arange(n,device=last.device)
                    torch.testing.assert_close(state[choose,last],benchmark[k][choose,last],rtol=2e-5,atol=2e-5)
            nxt=output(f.model,state,last).argmax(-1).tolist()
            by.setdefault(mode,[]).append(complete(f,tok,pack,nxt,k))
    return {m:list(zip(*x,strict=True)) for m,x in by.items()},list(zip(*traces,strict=True))


def summarize(rows):
    def hit(r,a,k):
        return r["pred"][a][k]["number"]==r[("target","foil")[k]]
    by={}
    for a in MODES:
        by[a]=dict(K1_correct=sum(hit(r,a,0) for r in rows),K2_foil=sum(hit(r,a,1) for r in rows),
                   joint=sum(hit(r,a,0) and hit(r,a,1) for r in rows),
                   donor_candidate=sum(r["donor_target"] is not None and x["number"] in (r["donor_target"],r["donor_target"]+100)
                                       for r in rows for x in r["pred"][a]))
    comp={}
    for a,b in (("distant","distant_safe"),("distant","aligned_safe"),("distant","geometry"),
                ("keep_disjoint","keep_sham"),("drop_sham","drop_disjoint")):
        clusters={}
        for r in rows:
            x=int(hit(r,a,0) and hit(r,a,1))-int(hit(r,b,0) and hit(r,b,1))
            t=r["target"]
            clusters[t]=clusters.get(t,0)+x
        comp[a+"_vs_"+b]=ci(list(clusters.values()),(.0125,.9875))
    return dict(n=len(rows),outcomes=by,comparisons=comp)


@torch.inference_mode()
def run(model,root):
    matching_path=root/"MATCHING.json"
    assert load(matching_path)["gate"]
    m,tok=setup(model)
    f=Forward(m)
    base=load(root/"NATIVE.json")
    pool,_,chosen=build_plan(tok,base["rows"])
    q,bases=donors(f,list(pool.values()),root)
    idx=root/"READOUT_RAW.json"
    out=load(idx)["rows"] if idx.exists() else []
    assert [p["id"] for p in out]==[p["id"] for p in chosen[:len(out)]]
    for i in range(len(out),len(chosen),8):
        pack=chosen[i:i+8]
        results,power=apply(f,tok,pack,q,bases)
        for j,v in enumerate(pack):
            c=v["repair"]
            row={m:x[j] for m,x in results.items()}
            assert list(row["native"])==c["native"]
            d=int(v["disjoint"].split("-")[1]) if v["disjoint"] else None
            out.append(dict(id=v["id"],target=c["target"],foil=c["foil"],donor_target=d,pred=row,energy=power[j],
                            basis_ready=c["target"]//100 in bases))
        save(idx,dict(rows=out,matching_sha256=sha(matching_path),bases_sha256=sha(root/"DISJOINT_BASES.json")))
        print("readout transport",len(out),len(chosen),flush=True)
    result=summarize(out)
    save(root/"READOUT_SUMMARY.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)

