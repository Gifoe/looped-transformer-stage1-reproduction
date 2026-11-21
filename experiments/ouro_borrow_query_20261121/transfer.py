"""Transfer a natural physical branch displacement between disjoint answer digits."""
import argparse
from collections import Counter
from pathlib import Path

import torch

from borrow import ci
from interchange import Patch, batch, complete, independent_like, load_cases, text_case
from native import Forward, load, output, save, setup, sha

MODES=("natural_borrow","natural_safe","random")


def digit_set(c):
    return {c["target"]//100,c["foil"]//100}


def norm_like(x,y):
    return x*(y.norm(dim=-1,keepdim=True)/x.norm(dim=-1,keepdim=True).clamp_min(1e-12))


def summarize(raw):
    result={}
    for m in MODES:
        events=Counter()
        hits={}
        for row in raw:
            if not row["matched"]:
                continue
            a,b=[x["number"] for x in row["pred"][m]]
            target=row["target"]
            foil=row["foil"]
            joint=a==target and b==foil
            events["K1_exact"]+=a==target
            events["K2_foil"]+=b==foil
            events["joint"]+=joint
            events["donor_answer_or_foil"]+=sum(z in (row["donor_target"],row["donor_target"]+100) for z in (a,b))
            hits[row["id"]]=joint
        result[m]=(events,hits)
    cmp={}
    for ctl in MODES[1:]:
        cluster={}
        for r in raw:
            if r["matched"]:
                val=int(result[MODES[0]][1][r["id"]])-int(result[ctl][1][r["id"]])
            else:
                val=0
            target=r["target"]
            cluster[target]=cluster.get(target,0)+val
        cmp[ctl]=ci(list(cluster.values()),(.0125,.9875))
    return dict(n=len(raw),matched=sum(r["matched"] for r in raw),
                modes={m:result[m][0] for m in MODES},comparisons=cmp,
                gate=bool(result[MODES[0]][0]["joint"]>=8 and
                          all(x["interval"] and x["interval"][0]>0 for x in cmp.values())))


@torch.inference_mode()
def run(model,root):
    base=load(root/"QUERY_SUMMARY.json")
    assert base["repair"]["modes"]["write"]["joint"]>=8 and base["comparisons"]["write_vs_write_sham"]["joint"]["interval"][0]>0
    loc=load(root/"SELECTION.json")["write"]
    l,name=loc["layer"],loc["site"]
    m,tok=setup(model)
    f=Forward(m)
    dev,raw_donors=load_cases(root,tok,"discovery")
    paired=[(p,text_case(tok,row,0)) for p,row in zip(raw_donors,dev,strict=True) if all(p["reach"])]
    by_type=[p for duo in paired for p in duo]
    memo={}
    for i in range(0,len(by_type),12):
        pack=by_type[i:i+12]
        _,_,tr,_,_=batch(f,pack)
        diff=tr.at(1,l,name)-tr.at(0,l,name)
        for j,c in enumerate(pack):
            memo[c["id"],c["kind"]]=diff[j]
    _,cases=load_cases(root,tok,"confirmation")
    plan=[(receipt,next((d for d,_ in paired if not digit_set(d).intersection(digit_set(receipt))),None)
                       if all(receipt["reach"]) else None) for receipt in cases]
    path=root/"TRANSFER_RAW.json"
    rows=load(path)["rows"] if path.exists() else []
    assert [r["id"] for r in rows]==[c["id"] for c,_ in plan[:len(rows)]]
    for i in range(len(rows),len(plan),8):
        block=plan[i:i+8]
        notes=[dict(id=c["id"],target=c["target"],foil=c["foil"],matched=d is not None) for c,d in block]
        chosen=[(c,d) for c,d in block if d is not None]
        for z,(_,d) in zip([x for x in notes if x["matched"]],chosen,strict=True):
            z.update(donor=d["id"],donor_target=d["target"])
        if chosen:
            batch_cases=[c for c,_ in chosen]
            seq,at,rec,h1,h2=batch(f,batch_cases)
            delta=rec.at(1,l,name)-rec.at(0,l,name)
            diffs={ 
                "natural_borrow":torch.stack([memo[d["id"],"repair"] for _,d in chosen]),
                "natural_safe":torch.stack([memo[d["id"],"safe"] for _,d in chosen]),
                "random":independent_like(delta,batch_cases,l,66),
            }
            for arm,x in diffs.items():
                v=norm_like(x,delta)
                torch.testing.assert_close(v.norm(dim=-1),delta.norm(dim=-1),rtol=1e-5,atol=1e-5)
                ans=[]
                for k in (0,1):
                    y=rec.at(k,l,name)+v*(1 if k==0 else -1)
                    edit=Patch(at,writes={(l,name):y})
                    h=f.run(seq,1,start=h1 if k else None,edit=edit)[0]
                    p=output(f.model,h,at).argmax(-1).tolist()
                    ans.append(complete(f,tok,batch_cases,p,k))
                for r,z in zip((o for o in notes if o["matched"]),zip(*ans,strict=True),strict=True):
                    r.setdefault("pred",{})[arm]=z
        rows.extend(notes)
        save(path,dict(rows=rows,query_sha256=sha(root/"QUERY_RAW.json"),selection_sha256=sha(root/"SELECTION.json")))
        print("write transfer",len(rows),len(plan),flush=True)
    result=summarize(rows)
    save(root/"TRANSFER_SUMMARY.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.model,a.out)

