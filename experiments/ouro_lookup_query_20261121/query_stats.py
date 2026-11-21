"""Whole-value success, paired controls, and independent transfer endpoints."""
from statistics import median

from edge_stats import upper_binomial
from origin_stats import paired, positive

LATTICE={"rescue":("original",0),"reversal":("original",1),"values":("values",0),"crossed":("crossed",0)}
CONTROLS=("geometry","location")
MODES=("native","authentic",*CONTROLS)


def target(row,k):
    if k=="rescue":
        return row["target"]
    if k=="reversal":
        return row["native"]["original"][0]["number"]
    return row["new"]


def outcome(r,k,arm):
    return r["pred"][k][arm]["number"]==target(r,k)


def arm_table(rows,k):
    cohort=[r for r in rows if k in r["test"]]
    arms={}
    for a in MODES:
        old=(LATTICE[k][0],0)
        arms[a]=dict(success=sum(outcome(r,k,a) for r in cohort),
                     source_correct=sum(r["pred"][k][a]["number"]==r["target"] for r in cohort),
                     recipient_natural_wrong=sum(r["pred"][k][a]["number"]==r["native"][old[0]][old[1]]["number"] for r in cohort),
                     unparsed=sum(r["pred"][k][a]["number"] is None for r in cohort))
    compare={a:paired(cohort,[int(outcome(r,k,"authentic"))-int(outcome(r,k,a)) for r in cohort]) for a in CONTROLS}
    return dict(n=len(cohort),arms=arms,paired=compare,overlap_original=sum("rescue" in r["test"] for r in cohort))


def summarize(rows):
    report={k:arm_table(rows,k) for k in LATTICE}
    exact=[r for r in rows if "rescue" in r["test"]]
    def both(r,a):
        return outcome(r,"rescue",a) and outcome(r,"reversal",a)
    counts={a:sum(both(r,a) for r in exact) for a in MODES}
    effects={}
    bounds={}
    for a in CONTROLS:
        z=[int(both(r,"authentic"))-int(both(r,a)) for r in exact]
        effects[a]=paired(exact,z)
        n=len(z)
        k=sum(v>0 for v in z)
        bounds[a]=dict(events=k,n=n,upper_97_5=upper_binomial(k,n))
    report["dual"]=dict(n=len(exact),arms=counts,paired=effects,bounds=bounds)
    report["gates"]=dict(
        bidirectional=counts["authentic"]>=8 and all(map(positive,effects.values())),
        address_transfer=all(report[v]["arms"]["authentic"]["success"]>=8 and all(map(positive,report[v]["paired"].values()))
                             for v in ("values","crossed")),
        exclude_large_dual=bool(exact) and all(x["upper_97_5"]<.10 for x in bounds.values()))
    report["routing"]={}
    for k in LATTICE:
        q=[r for r in rows if k in r["routing"]]
        view={}
        variant=LATTICE[k][0]
        for a in (*MODES,"identity"):
            if not q:
                continue
            right=[r["routing"][k][a][r["key"]-1] for r in q]
            wrong=[r["routing"][k][a][r["example"][variant]["values"].index(r["native"][variant][0]["number"])] for r in q]
            view[a]=dict(correct=median(right),wrong=median(wrong),gap=median(x-y for x,y in zip(right,wrong,strict=True)))
        report["routing"][k]=dict(n=len(q),mass=view)
    return report
