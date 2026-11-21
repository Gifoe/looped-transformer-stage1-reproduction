"""Score prompt-position mediation and readout-preserving context transfer."""
import argparse
from pathlib import Path

from analysis import metric,scores,winner
from native import load,save,sha

GROUPS=("ALL","WC","CW","CC","WW")
CTRL=("body_rolled","body_scaled")


def score_raw(base,branches,q):
    gold=q["gold"]
    values={k:[scores(z) for z in zz] for k,zz in base.items()}
    foil=max((i for i in range(len(q["choices"])) if i!=gold),key=lambda i:values["clean"][1][i])
    def score(v):
        return v[gold]-v[foil],winner(v)==gold
    states={k:[score(z) for z in x] for k,x in values.items()}
    pop="".join("C" if x[1] else "W" for x in states["clean"])
    return dict(pop=pop,states=states,branches=branches,score=score)


def support(rows,meta):
    data=[]
    for v,q in zip(rows,meta,strict=True):
        assert v["id"]==q["id"]
        z=score_raw(v["native"],v["arms"],q)
        measure=z["score"]
        arms={}
        for b,m in z["branches"].items():
            vals={}
            for scope,x in m.items():
                clean=measure(scores(x["actual"]))
                ctrl=measure(scores(x["control"]))
                vals[scope]=dict(actual=clean,matched=ctrl,norm=x["product_norm"])
            arms[b]=vals
        data.append(dict(id=q["id"],pop=z["pop"],native=z["states"],arms=arms))
    summaries={}
    for pop in GROUPS:
        r=data if pop=="ALL" else [q for q in data if q["pop"]==pop]
        c={}
        for b,sign in (("drop",1),("clean",-1)):
            c[b]={}
            for support in ("earlier","readout","all"):
                c[b][support]=dict(effect=metric([sign*(x["arms"][b][support]["actual"][0]-x["native"][b][1][0]) for x in r]),
                    versus_matched=metric([sign*(x["arms"][b][support]["actual"][0]-x["arms"][b][support]["matched"][0]) for x in r]),
                    dose=metric([x["arms"][b][support]["norm"] for x in r]),
                    hit=sum(x["arms"][b][support]["actual"][1] for x in r),
                    matched_hit=sum(x["arms"][b][support]["matched"][1] for x in r))
            c[b]["interaction"]=metric([sign*(x["arms"][b]["all"]["actual"][0]-x["arms"][b]["earlier"]["actual"][0]-
                          x["arms"][b]["readout"]["actual"][0]+x["native"][b][1][0]) for x in r])
        summaries[pop]=dict(n=len(r),background=c)
    return summaries


def body(rows,m):
    vals=[]; max_dose=0.; preserve=0
    for x,q in zip(rows,m,strict=True):
        assert q["id"]==x["id"]
        t=score_raw(x["native"],x["arms"],q)
        z={}
        for rec,p in x["arms"].items():
            d=abs(p["body"]["dose"]-p["body_rolled"]["dose"])/max(p["body"]["dose"],1e-12)
            max_dose=max(d,max_dose)
            for k,e in p.items():
                if k.startswith("body"):
                    assert e["k1_readout_bitwise_held"]
                    assert e["k1_margin_input"]==x["native"][rec][0]
                    preserve+=1
            z[rec]={k:dict(result=t["score"](scores(v["ll"])),dose=v["dose"],ood=v["input_norm_max_ratio"]>2 or v["output_norm_max_ratio"]>2)
                    for k,v in p.items()}
        vals.append(dict(id=q["id"],pop=t["pop"],states=t["states"],arms=z))
    y={}
    for pop in GROUPS:
        rs=vals if pop=="ALL" else [x for x in vals if x["pop"]==pop]
        tables={}
        for side,sign in (("drop",1),("clean",-1)):
            tables[side]={k:dict(effect=metric([sign*(r["arms"][side][k]["result"][0]-r["states"][side][1][0]) for r in rs]),
                               dose=metric([r["arms"][side][k]["dose"] for r in rs]),
                               hits=sum(r["arms"][side][k]["result"][1] for r in rs),
                               ood=sum(r["arms"][side][k]["ood"] for r in rs))
                          for k in vals[0]["arms"][side]}
            tables[side]["body_vs_controls"]={
                k:metric([sign*(r["arms"][side]["body"]["result"][0]-r["arms"][side][k]["result"][0]) for r in rs])
                for k in (*CTRL,"boundary")}
        broken=[r for r in rs if r["states"]["clean"][1][1] and not r["states"]["drop"][1][1]]
        true=[]
        strict=[]
        for r in broken:
            a,b=r["arms"]["drop"],r["arms"]["clean"]
            if a["body"]["result"][1] and not b["body"]["result"][1]:
                true.append(r["id"])
                if all(not a[k]["result"][1] and b[k]["result"][1] for k in CTRL):
                    strict.append(r["id"])
        y[pop]=dict(n=len(rs),native={k:[sum(r["states"][k][i][1] for r in rs) for i in range(2)] for k in ("clean","drop")},
             tables=tables,broken=len(broken),rescue={k:sum(r["arms"]["drop"][k]["result"][1] for r in broken)
                             for k in vals[0]["arms"]["drop"]},
             reversal={k:sum(not r["arms"]["clean"][k]["result"][1] for r in broken)
                             for k in vals[0]["arms"]["clean"]},
             dual_ids=true,strict_dual_ids=strict)
    y["audit"]=dict(held_k1_readouts=preserve,relative_dose_max_error=max_dose)
    return y


def read(path,m,expected):
    z=load(path)
    assert len(z["rows"])==expected==z["expected"]
    assert z["manifest_sha256"]==sha(m)
    return z["rows"]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    a=p.parse_args().out
    m=a/"MANIFEST.json"
    b=a/"BODY_MANIFEST.json"
    supports={}
    bodies={}
    for f,v in load(m)["cohorts"].items():
        supports[f]=support(read(a/f"SUPPORT_{f}.json",m,128),v["rows"][:128])
    for f,v in load(b)["cohorts"].items():
        bodies[f]=body(read(a/f"BODY_{f}.json",b,192),v["rows"])
    decisions={}
    for f,z in bodies.items():
        y=z["ALL"]["tables"]
        checks={f"{k}/{t}":y[k]["body_vs_controls"][t]["ci"][0]>0
                for k in ("drop","clean") for t in CTRL}
        checks.update(base_drop=y["drop"]["body"]["effect"]["ci"][0]>0,
                      base_clean=y["clean"]["body"]["effect"]["ci"][0]>0,
                      plenty=z["WC"]["broken"]>=12,reciprocal=len(z["WC"]["strict_dual_ids"])>=2)
        decisions[f]=checks
    save(a/"EXTRA_SUMMARY.json",dict(support=supports,body=bodies,body_decision=dict(checks=decisions,
         pass_all=all(all(list(x.values())) for x in decisions.values()))))


if __name__=="__main__":
    main()

