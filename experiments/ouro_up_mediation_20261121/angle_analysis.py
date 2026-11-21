"""Evaluate the matched geometry orientation experiment."""
import argparse
from pathlib import Path

from analysis import metric,scores,winner
from native import load,save,sha

CONTROL=("angle_control_0","angle_control_1")


def evaluate(raw,meta):
    data=[]
    proof={"held_k1":0,"max_token_norm_error":0.,"max_per_token_dose_error":0.}
    for r,m in zip(raw,meta,strict=True):
        assert r["id"]==m["id"]
        g=m["gold"]
        natural={k:[scores(x) for x in v] for k,v in r["native"].items()}
        foil=max((j for j in range(len(natural["clean"][1])) if j!=g),key=lambda j:natural["clean"][1][j])
        def val(x):
            return x[g]-x[foil],winner(x)==g
        b={k:[val(q) for q in v] for k,v in natural.items()}
        code="".join("C" if z[1] else "W" for z in b["clean"])
        arms={}
        for world,x in r["arms"].items():
            results={}
            for k,t in x.items():
                assert t["k1_ll"]==r["native"][world][0]
                proof["held_k1"]+=1
                proof["max_token_norm_error"]=max(proof["max_token_norm_error"],t["norm_error"])
                if k!="half":
                    proof["max_per_token_dose_error"]=max(proof["max_per_token_dose_error"],t["per_token_dose_error"])
                results[k]=dict(value=val(scores(t["ll"])),ood=t["output_norm_ratio"]>2,dose=t["dose"])
            arms[world]=results
        data.append(dict(id=m["id"],code=code,baseline=b,arms=arms))
    panels={}
    for p in ("ALL","WC","CC","WW","CW"):
        rows=data if p=="ALL" else [q for q in data if q["code"]==p]
        w={}
        for world,sign in (("drop",1),("clean",-1)):
            x={}
            for cond in data[0]["arms"][world]:
                x[cond]=dict(effect=metric([sign*(r["arms"][world][cond]["value"][0]-r["baseline"][world][1][0]) for r in rows]),
                             n_hit=sum(r["arms"][world][cond]["value"][1] for r in rows),
                             ood2x=sum(r["arms"][world][cond]["ood"] for r in rows),
                             dose=metric([r["arms"][world][cond]["dose"] for r in rows]))
            w[world]=dict(arms=x,versus={k:metric([sign*(r["arms"][world]["direction"]["value"][0]-
                                     r["arms"][world][k]["value"][0]) for r in rows]) for k in CONTROL})
        broken=[r for r in rows if r["baseline"]["clean"][1][1] and not r["baseline"]["drop"][1][1]]
        recovery={k:sum(r["arms"]["drop"][k]["value"][1] for r in broken) for k in data[0]["arms"]["drop"]}
        damage={k:sum(not r["arms"]["clean"][k]["value"][1] for r in broken) for k in data[0]["arms"]["clean"]}
        both=[r for r in broken if r["arms"]["drop"]["direction"]["value"][1] and not r["arms"]["clean"]["direction"]["value"][1]]
        strict=[r["id"] for r in both if all(not r["arms"]["drop"][k]["value"][1] and
                    r["arms"]["clean"][k]["value"][1] for k in CONTROL)]
        panels[p]=dict(n=len(rows),worlds=w,broken=len(broken),recovered=recovery,damaged=damage,
                     both_ids=[r["id"] for r in both],strict_ids=strict)
    return dict(panels=panels,audit=proof,gpu_seconds=sum(x["gpu_seconds"] for x in raw))


def ids(path):
    return {q["id"] for z in load(path)["cohorts"].values() for q in z["rows"]}


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("out",type=Path)
    o=a.parse_args().out
    m=o/"ANGLE_MANIFEST.json"
    assert not ids(m)&(ids(o/"MANIFEST.json")|ids(o/"BODY_MANIFEST.json"))
    p={}
    for f,z in load(m)["cohorts"].items():
        y=load(o/f"ANGLE_{f}.json")
        assert y["manifest_sha256"]==sha(m) and len(y["rows"])==y["expected"]==z["n"]
        p[f]=evaluate(y["rows"],z["rows"])
    tests={}
    for k,m in p.items():
        a=m["panels"]["ALL"]
        w=m["panels"]["WC"]
        checks={}
        for q in ("drop","clean"):
            s=a["worlds"][q]
            checks[q+"_base"]=s["arms"]["direction"]["effect"]["ci"][0]>0
            checks.update({f"{q}_{c}":s["versus"][c]["ci"][0]>0 for c in CONTROL})
        checks["bidirectional_at_least_two"]=len(w["strict_ids"])>=2
        tests[k]=checks
    save(o/"ANGLE_SUMMARY.json",dict(families=p,checks=tests,
                  cross_task_gate=all(all(z.values()) for z in tests.values())))

