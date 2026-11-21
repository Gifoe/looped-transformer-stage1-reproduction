"""Compare local attention output edits with native or recomputed MLP."""
import argparse
from collections import Counter
from pathlib import Path

import torch
from block import geom, quad, replace
from borrow import ci
from interchange import Patch, complete
from native import Forward, load, output, save, setup, sha
from readout import forms

TYPES=("natural","early","early_geometry","early_fixed_mlp","late")


def evaluate(f,tok,pack,l):
    ids,p,original,sites=quad(f,pack)
    parts=sites[l]
    pre=[z[1]["writes"] for z in parts]
    late=[z[0] for z in parts]
    n=[x[1]["full"][2] for x in parts]
    matched=[geom(pre[0][0],pre[1][0],v,pack,l,761) for v in n]
    answers={}
    for arm in TYPES:
        for k in (0,1):
            way=1 if k==0 else -1
            if arm=="natural":
                finish=original[k]
            elif arm=="late":
                finish=replace(f,ids,p,original,l,k,late[k]+way*n[k])
            else:
                d=matched[k] if arm=="early_geometry" else n[k]
                patch={(l,"a_write"):pre[k][0]+way*d}
                if arm=="early_fixed_mlp":
                    patch[l,"m_write"]=pre[k][1]
                finish=f.run(ids,1,start=original[0] if k else None,edit=Patch(p,writes=patch))[0]
            z=output(f.model,finish,p).argmax(-1).tolist()
            answers.setdefault(arm,[]).append(complete(f,tok,pack,z,k))
    return {a:list(zip(*v,strict=True)) for a,v in answers.items()}


def summarize(data):
    out={}
    for key in ("repair","safe"):
        samples=[r for r in data if r["kind"]==key]
        res={}
        for arm in TYPES:
            table=Counter()
            for r in samples:
                a,b=[x["number"] for x in r["pred"][arm]]
                u,v=[x["number"] for x in r["pred"]["natural"]]
                table["K1_exact"]+=a==r["target"]
                table["K2_foil"]+=b==r["foil"]
                table["joint"]+=a==r["target"] and b==r["foil"]
                table["K1_harm"]+=u==r["target"] and a!=r["target"]
                table["K2_harm"]+=v==r["target"] and b!=r["target"]
            res[arm]=dict(table)
        out[key]=dict(n=len(samples),outcomes=res)
    contrasts={}
    for v,w in (("early","early_geometry"),("early","early_fixed_mlp")):
        accum=[{},{}]
        for x in data:
            if x["kind"]!="repair":
                continue
            y=x["target"]
            nums=[o["number"] for o in x["pred"][v]]
            ct=[o["number"] for o in x["pred"][w]]
            for k,t in enumerate((y,x["foil"])):
                d=accum[k]; d[y]=d.get(y,0)+int(nums[k]==t)-int(ct[k]==t)
        contrasts[v+"_vs_"+w]=[ci(list(z.values()),(.0125,.9875)) for z in accum]
    out["comparisons"]=contrasts
    return out


@torch.inference_mode()
def run(model,root):
    m,tok=setup(model)
    f=Forward(m)
    raw=root/"BLOCK_RAW.json"
    use=load(raw)["rows"]
    by={r["id"]:r for r in load(root/"NATIVE.json")["rows"]}
    loc=load(root/"LOCK.json")["layer"]
    table=root/"BRANCH_RAW.json"
    rows=load(table)["rows"] if table.exists() else []
    assert [(x["id"],x["kind"]) for x in rows]==[(x["id"],x["kind"]) for x in use[:len(rows)]]
    for t in range(len(rows),len(use),8):
        block=use[t:t+8]
        available=[r for r in block if r["aligned"]]
        cases=[forms(tok,by[r["source_id"]])["pair"][r["kind"]=="safe"] for r in available]
        read=evaluate(f,tok,cases,loc) if cases else {}
        j=0
        for r in block:
            if r["aligned"]:
                z={a:q[j] for a,q in read.items()}
                j+=1
            else:
                z={a:r["outcomes"]["native"] for a in TYPES}
            assert list(z["natural"])==r["outcomes"]["native"]
            assert list(z["late"])==r["outcomes"]["attention"]
            assert [v["tokens"] for v in z["early_fixed_mlp"]]==[v["tokens"] for v in z["late"]]
            rows.append(dict(id=r["id"],kind=r["kind"],target=r["target"],foil=r["foil"],pred=z))
        save(table,dict(rows=rows,block_sha256=sha(raw)))
        print("operator order",len(rows),len(use),flush=True)
    result=summarize(rows)
    save(root/"BRANCH_SUMMARY.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)

