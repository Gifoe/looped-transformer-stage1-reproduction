"""Rollback the A and B query-conditioned value paths in natural correction."""
import argparse
from collections import Counter
from pathlib import Path

import torch

from borrow import ci
from factorial import corrected
from graft import encode
from interchange import complete, fit_margin, independent_like, text_case
from mechanism import Capture, Edit, LQ, delta_av, graft_av, prepare, weights
from native import Forward, load, output, save, setup, sha
from rule import greedy
from state import rotate_about

MODES=("native","full_q","a","a_sham","b","b_sham","ab","ab_sham","restore")


def subcase(tok,g,kind):
    i=int(kind=="borrow")
    old=dict(g,target=g["group"],pairs=[g["pairs"][4],g["pairs"][3]],
             pred=[[row[4],row[3]] for row in g["pred"]])
    z=text_case(tok,old,i)
    ids,fields=encode(tok,g["pairs"][3 if i else 4])
    assert ids==z["ids"]
    a,b=[set(x) for x in fields]
    z["fields"]=[1 if t in a else 2 if t in b else 0 for t in range(len(ids))]+[0]*len(z["prefix"])
    return z


def measure(f,tok,cases):
    ids,p,tr,h,tags=prepare(f,cases)
    n=len(cases)
    answers={}
    margins={}
    labels=torch.tensor([c["labels"] for c in cases],device=p.device)
    for k in (0,1):
        repeats=greedy(f,tok,[c["ids"] for c in cases],k+1)
        assert [x["tokens"] for x in repeats]==[z["native"][k]["tokens"] for z in cases]
        q0,q1=tr.get(k,"q"),tr.get(1-k,"q")
        kw,vw=tr.get(k,"k"),tr.get(k,"v")
        native_v=tr.get(k,"av")
        u=weights(f,q0,kw,p)
        v=weights(f,q1,kw,p)
        torch.testing.assert_close(u,tr.get(k,"a"),rtol=4e-5,atol=5e-6)
        after=Capture(p)
        full=f.run(ids,1,start=h[0] if k else None,record=after,edit=Edit(p,query={LQ:(None,q1)}))[0]
        torch.testing.assert_close(v,after.get(0,"a"),rtol=4e-5,atol=5e-6)
        d=v-u
        a=delta_av(d,vw,tags==1)
        b=delta_av(d,vw,tags==2)
        plan={"a":a,"b":b,"ab":a+b}
        index=torch.arange(n,device=p.device)
        for arm in MODES:
            if arm=="native":
                state=h[k]
            elif arm=="full_q":
                state=full
            else:
                src=native_v
                if arm!="restore":
                    change=plan[arm.removesuffix("_sham")]
                    src=src+change
                    if arm.endswith("_sham"):
                        seed=("a","b","ab").index(arm.removesuffix("_sham"))
                        src=rotate_about(native_v,src,independent_like(src,cases,LQ,261+seed))
                with graft_av(f,p,src):
                    state=f.run(ids,1,start=h[0] if k else None)[0]
                if arm=="restore":
                    assert torch.equal(state[index,p],h[k][index,p])
            tokens=output(f.model,state,p).argmax(-1).tolist()
            result=complete(f,tok,cases,tokens,k)
            answers.setdefault(arm,[]).append(result)
            margins.setdefault(arm,[]).append(fit_margin(f,state,p,labels).tolist())
    packed={m:list(zip(*values,strict=True)) for m,values in answers.items()}
    ms={m:list(zip(*values,strict=True)) for m,values in margins.items()}
    diffs=[[ms["ab"][j][k]-ms["a"][j][k]-ms["b"][j][k]+ms["native"][j][k] for k in (0,1)] for j in range(n)]
    return packed,diffs


def summary(rows):
    by={}
    for condition in ("repair","safe"):
        pack=[g for g in rows if g["kind"]==condition]
        out={}
        for m in MODES:
            c=Counter()
            for g in pack:
                k1,k2=[x["number"] for x in g["pred"][m]]
                y,t=g["target"],g["foil"]
                c["K1_correct"]+=k1==y
                c["K2_foil"]+=k2==t
                c["K2_correct"]+=k2==y
                c["joint"]+=k1==y and k2==t
                c["K2_new_error"]+=g["pred"]["native"][1]["number"]==y and k2!=y
                c["K1_new_error"]+=g["pred"]["native"][0]["number"]==y and k1!=y
                c["K1_unparsed"]+=k1 is None
                c["K2_unparsed"]+=k2 is None
            out[m]=dict(c)
        by[condition]=dict(n=len(pack),both_reach=sum(all(r["reach"]) for r in pack),modes=out)
    subset=[r for r in rows if r["kind"]=="repair"]
    witnesses=[g for g in subset if all(g["pred"][k][1]["number"]==g["foil"] for k in ("a","b"))]
    p={}
    for m in ("a","b","ab"):
        clusters={}
        for r in subset:
            n,s=[r["pred"][q][1]["number"]==r["foil"] for q in (m,m+"_sham")]
            t=r["target"]
            clusters[t]=clusters.get(t,0)+int(n)-int(s)
        p[m]=ci(list(clusters.values()),(.0125,.9875))
    distinct=len({x["target"] for x in witnesses})
    by.update(same_cases_both_cuts=len(witnesses),both_cut_targets=distinct,rollback=p,
              gate=len(witnesses)>=5 and distinct>=4 and all(p[k]["interval"][0]>0 for k in ("a","b")))
    return by


@torch.inference_mode()
def run(model,root):
    assert load(root/"NATIVE_SUMMARY.json")["natural_gate"]
    rows=load(root/"NATIVE.json")["rows"]
    w=sorted((g for g in rows if g["split"]=="confirmation" and corrected(g,3)),key=lambda x:x["id"])
    m,tok=setup(model)
    f=Forward(m)
    all_cases=[subcase(tok,g,t) for g in w for t in ("borrow","safe")]
    path=root/"NATURAL_RAW.json"
    raw=load(path)["rows"] if path.exists() else []
    assert [(x["id"],x["kind"]) for x in raw]==[(r["id"],r["kind"]) for r in all_cases[:len(raw)]]
    for i in range(len(raw),len(all_cases),10):
        part=all_cases[i:i+10]
        ans,interact=measure(f,tok,part)
        for j,r in enumerate(part):
            text={m:v[j] for m,v in ans.items()}
            assert list(text["native"])==r["native"]==list(text["restore"])
            row={k:r[k] for k in ("id","kind","target","foil","reach")}
            raw.append(dict(row,pred=text,two_field_pivot_interaction=interact[j]))
        save(path,dict(rows=raw,native_sha256=sha(root/"NATIVE.json")))
        print("natural field rollback",len(raw),len(all_cases),flush=True)
    result=summary(raw)
    save(root/"NATURAL_SUMMARY.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("root",type=Path)
    a=p.parse_args()
    run(a.model,a.root)

