"""Evaluate the native four corners and earlier-writer localization."""
import argparse
from pathlib import Path

from analysis import metric, scores, winner
from native import load, save, sha


STRATA=("ALL","WC","CW","CC","WW")


def late_foil(v,g):
    return max((i for i in range(len(v)) if i!=g),key=lambda i:v[i])


def base_pair(early,late,g):
    return "".join("C" if winner(y)==g else "W" for y in (early,late))


def factorial(raw,meta):
    work={}
    for row,m in zip(raw,meta,strict=True):
        assert row["id"]==m["id"]
        y0,y1=tuple(map(scores,row["ll_native"]))
        g=m["gold"]
        foe=late_foil(y1,g)
        pop=base_pair(y0,y1,g)
        for locus,v in row["factorials"].items():
            q={k:scores(x["ll"]) for k,x in v.items()}
            margin={k:x[g]-x[foe] for k,x in q.items()}
            hit={k:winner(x)==g for k,x in q.items()}
            work.setdefault(locus,[]).append(dict(pop=pop,correct={k:hit[k] for k in hit},native=winner(y1)==g,
                     margins=margin,interaction=margin["both"]-margin["first"]-margin["second"]+margin["base"],
                     ood={k:x["ratio"]>2 for k,x in v.items()}))
    out={}
    for name,data in work.items():
        out[name]={}
        for pop in STRATA:
            rows=data if pop=="ALL" else [q for q in data if q["pop"]==pop]
            out[name][pop]=dict(n=len(rows),interaction=metric([x["interaction"] for x in rows]),
                 arms={k:dict(effect=metric([q["margins"][k]-q["margins"]["base"] for q in rows]),
                    hits=sum(q["correct"][k] for q in rows),ood2x=sum(q["ood"][k] for q in rows),
                    harm=sum(q["correct"]["base"] and not q["correct"][k] for q in rows),
                    rescue=sum(not q["correct"]["base"] and q["correct"][k] for q in rows))
                    for k in ("base","first","second","both")},
                 residual_rescue_vs_native=sum(q["native"] and not q["correct"]["base"]
                       and q["correct"]["both"] for q in rows),
                 only_joint_rescue=sum(q["native"] and not any(q["correct"][k] for k in ("base","first","second"))
                       and q["correct"]["both"] for q in rows))
    return out


def writer(raw,meta):
    variants={}
    for key in ("standard","first_token_rewrite"):
        cases=[]
        n_total=len(raw)
        for x,m in zip(raw,meta,strict=True):
            assert x["id"]==m["id"]
            row=x[key]
            if row is None:
                continue
            early,late=map(scores,row["native"])
            g=m["gold"]
            foil=late_foil(late,g)
            effects={}
            for k,a in row["arms"].items():
                v=scores(a["ll"])
                effects[k]=dict(mass=row["mass"][1]-a["mass"],factor_loss=a["factor_loss"],
                      margin=(v[g]-v[foil])-(late[g]-late[foil]),
                      hit=winner(v)==g,ood=a["norm_ratio"]>2,write_norm=a["write_norm"])
            cases.append(dict(pop=base_pair(early,late,g),early_mass=row["mass"][0],late_mass=row["mass"][1],
                      native=winner(late)==g,effects=effects))
        if not cases:
            variants[key]=dict(n=0,omissions=n_total)
            continue
        subs={}
        for p in STRATA:
            z=cases if p=="ALL" else [x for x in cases if x["pop"]==p]
            vals={k:{m:metric([x["effects"][k][m] for x in z]) for m in ("mass","factor_loss","margin","write_norm")}
                  | {"damage":sum(x["native"] and not x["effects"][k]["hit"] for x in z),
                     "gain":sum(not x["native"] and x["effects"][k]["hit"] for x in z),
                     "ood":sum(x["effects"][k]["ood"] for x in z)}
                    for k in ("pos0","rest","all","pos0_permuted","rest_permuted")}
            two={}
            for a,b in (("pos0","rest"),("pos0","pos0_permuted"),("rest","rest_permuted")):
                two[f"{a}_vs_{b}"]={m:metric([q["effects"][a][m]-q["effects"][b][m] for q in z])
                                    for m in ("mass","factor_loss","margin")}
            subs[p]=dict(n=len(z),arms=vals,paired=two)
        variants[key]=dict(n=len(cases),early=metric([q["early_mass"] for q in cases]),
            late=metric([q["late_mass"] for q in cases]),panels=subs)
    return variants


def main():
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    a=p.parse_args()
    out=a.out
    src=out/"NATURAL_MANIFEST.json"
    data=load(src)["cohorts"]
    result={}
    for family,cohort in data.items():
        f=load(out/f"FACTORIAL_{family}.json")
        w=load(out/f"WRITER_{family}.json")
        assert f["manifest_sha256"]==w["manifest_sha256"]==sha(src)
        assert len(f["rows"])==f["expected"]==cohort["n"]
        assert len(w["rows"])==w["expected"]==128
        result[family]=dict(factorials=factorial(f["rows"],cohort["rows"]),
                            writer=writer(w["rows"],cohort["rows"][:128]),
                            factorial_gpu_seconds=sum(x["seconds"] for x in f["rows"]),
                            writer_gpu_seconds=sum(x["seconds"] for x in w["rows"]))
    save(out/"MECHANISM_SUMMARY.json",result)


if __name__=="__main__":
    main()

