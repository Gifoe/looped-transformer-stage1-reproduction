"""Test the native query-to-readout path with mixed-change and main-effect restoration."""
import argparse
from collections import Counter
from pathlib import Path

import torch

from borrow import ci
from interchange import complete, independent_like
from mechanism import Capture, Edit, LQ, LR, delta_av, graft_av, matched_plane, weights
from native import Forward, load, output, save, setup, sha
from readout import core, mixed, plan
from state import rotate_about

MODES=("native","cut","cut_sham","interaction","interaction_sham","main","main_sham","full")


def cut_quad(f,cases):
    ids,p,tr,h=core(f,cases,Capture)
    fields=[]
    for r in cases:
        for x,_ in r["quad_fields"]:
            fields.append(set(x))
    region=torch.tensor([[i in v for i in range(ids.shape[1])] for v in fields],device=ids.device)
    assert all(bool(t) for t in fields)
    native=weights(f,tr.get(1,"q"),tr.get(1,"k"),p)
    swapped=weights(f,tr.get(0,"q"),tr.get(1,"k"),p)
    torch.testing.assert_close(native,tr.get(1,"a"),rtol=4e-5,atol=5e-6)
    av=tr.get(1,"av")
    shown=delta_av(native,tr.get(1,"v"),torch.arange(ids.shape[1],device=p.device)[None,:]<=p[:,None])
    torch.testing.assert_close(shown,av,rtol=3e-5,atol=2e-5)
    to_av=av+delta_av(swapped-native,tr.get(1,"v"),region)
    rec=Capture(p)
    with graft_av(f,p,to_av):
        generated=f.run(ids,1,start=h[0],record=rec)[0]
    return ids,p,h,tr,av,to_av,rec,generated


def cosine(a,b):
    return ((a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1.e-12)).tolist()


def observe(f,tok,ready):
    pack=[c for s in ready for c in s["pair"]]
    seq,pos,states,tr,av,to_av,cut,native_cut=cut_quad(f,pack)
    atj=torch.arange(3,len(seq),4,device=seq.device)
    x,p=seq[atj],pos[atj]
    original=tr.get(1,"r")
    changed=cut.get(0,"r")
    ni=mixed(original)
    ci_=mixed(changed)
    loss=ni-ci_
    base=changed[atj]
    target=original[atj]
    change=target-base
    contextual=change-loss
    rand=independent_like(loss,pack,LR,414)
    rot=matched_plane(base,change,loss,rand)
    other_rot=matched_plane(base,change,contextual,independent_like(loss,pack,LR,415))
    keep={}
    anchors={"interaction":base+loss,"interaction_sham":base+rot,"main":base+contextual,
             "main_sham":base+other_rot,"full":target}
    for mode in MODES:
        if mode=="native":
            end=states[1][atj]
        elif mode=="cut":
            end=native_cut[atj]
        else:
            inject=to_av[atj]
            read=None
            if mode=="cut_sham":
                inject=rotate_about(av[atj],inject,independent_like(inject,pack,LQ,416))
            else:
                read=anchors[mode]
            with graft_av(f,p,inject):
                end=f.run(x,1,start=states[0][atj],edit=Edit(p,readout=read))[0]
            if mode=="full":
                j=torch.arange(len(p),device=p.device)
                torch.testing.assert_close(end[j,p],states[1][atj][j,p],rtol=2e-5,atol=2e-5)
        z=output(f.model,end,p).argmax(-1).tolist()
        keep[mode]=complete(f,tok,pack,z,1)
    geometry=dict(native_interaction=ni.norm(dim=-1).tolist(),reduced_interaction=ci_.norm(dim=-1).tolist(),
                  loss=loss.norm(dim=-1).tolist(),joint_readout_move=change.norm(dim=-1).tolist(),
                  alignment_to_native=cosine(loss,ni),
                  alignment_to_cross_loop=cosine(loss,ni-mixed(tr.get(0,"r"))))
    return keep,geometry


def clustered(rows,test):
    bins={}
    for r in rows:
        g=r["target"]
        bins[g]=bins.get(g,0)+test(r)
    return ci(list(bins.values()),(.0125,.9875))


def summary(rows):
    total={}
    for key in ("repair","safe"):
        vals=[a for a in rows if a["kind"]==key]
        cells={}
        for arm in MODES:
            c=Counter()
            for r in vals:
                number=r["pred"][arm]["number"]
                c["correct"]+=number==r["target"]
                c["foil"]+=number==r["foil"]
                c["wrong_new"]+=r["pred"]["native"]["number"]==r["target"] and number!=r["target"]
                c["unparsed"]+=number is None
            cells[arm]=dict(c)
        total[key]=dict(n=len(vals),aligned=sum(p["aligned"] for p in vals),outcomes=cells)
    repair=[r for r in rows if r["kind"]=="repair"]
    reversed_=[r for r in repair if r["pred"]["cut"]["number"]==r["foil"]]
    paired=clustered(repair,lambda r:int(r["pred"]["cut"]["number"]==r["foil"])-int(r["pred"]["cut_sham"]["number"]==r["foil"]))
    on_cut={}
    for m in ("interaction","main","full"):
        on_cut[m]=sum(r["pred"][m]["number"]==r["target"] for r in reversed_)
    for m in ("interaction_sham","main_sham"):
        on_cut[m]=sum(r["pred"][m]["number"]==r["target"] for r in reversed_)
    specific=clustered(reversed_,lambda r:int(r["pred"]["interaction"]["number"]==r["target"])-
                       int(r["pred"]["interaction_sham"]["number"]==r["target"]))
    total.update(cut_effect=paired,exact_reversions=len(reversed_),rescues=on_cut,interaction_rescue_excess=specific,
                 chain_gate=bool(len(reversed_)>=5 and paired["interval"][0]>0 and on_cut["interaction"]>=4
                                 and specific["interval"] and specific["interval"][0]>0))
    return total


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    raw=root/"NATIVE.json"
    rows=load(raw)["rows"]
    _,chosen,_=plan(tok,rows)
    pref=root/"MATCHING.json"
    receipt=load(pref)
    assert receipt["targets"]==[z["g"]["id"] for z in chosen]
    out=root/"PATH_RAW.json"
    prev=load(out)["rows"] if out.exists() else []
    assert [(z["id"],z["kind"]) for z in prev]==[(q["id"],q["kind"]) for x in chosen for q in x["pair"]][:len(prev)]
    for i in range(len(prev)//2,len(chosen),4):
        block=chosen[i:i+4]
        valid=[j for j,x in enumerate(block) if x["aligned"]]
        exact={}
        if valid:
            values,stats=observe(f,tok,[block[j] for j in valid])
            for k,j in enumerate(valid):
                exact[j]=[(dict((m,q[2*k+t]) for m,q in values.items()),dict((m,q[2*k+t]) for m,q in stats.items())) for t in (0,1)]
        for j,z in enumerate(block):
            for t,c in enumerate(z["pair"]):
                pred,trace=exact[j][t] if j in exact else ({m:c["native"][1] for m in MODES},{})
                assert pred["native"]==c["native"][1]==pred["full"]
                prev.append(dict(id=c["id"],kind=c["kind"],target=c["target"],foil=c["foil"],pred=pred,trace=trace,
                                 aligned=j in exact))
        save(out,dict(rows=prev,native_sha256=sha(raw),matching_sha256=sha(pref)))
        print("native path",len(prev)//2,len(chosen),flush=True)
    r=summary(prev)
    save(root/"PATH_SUMMARY.json",r)
    print(r,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

