"""Score receiver, old-answer and donor answers as separate complete outcomes."""
from collections import defaultdict
from statistics import median

from edge_stats import upper_binomial
from origin_stats import paired, positive

MODES={"native":("receiver","live",0),"donor_replay":("whole_donor","live",0)}
for label,context in (("original","receiver"),("opposite","opposite"),("same","same")):
    if label!="original":
        MODES[label+"_intact"]=(context,"live",0)
    for suffix,mode,lane in (("full","full",0),("capped","equal",0),("matched","equal",1)):
        MODES[label+"_"+suffix]=(context,mode,lane)
ARMS=tuple(MODES)


def is_answer(row,arm,kind):
    by=row["donor_values"] if kind in ("opposite","same") else row
    return row["pred"][arm]["number"]==by[kind]


def wrong(row,arm):
    return is_answer(row,"native","target") and not is_answer(row,arm,"target")


def measure(rows):
    out={}
    for a in ARMS:
        names={"receiver":"target","old":"foil","opposite":"opposite","same":"same"}
        rates={k:sum(is_answer(r,a,z) for r in rows) for k,z in names.items()}
        rates["donor_natural_k1"]=sum(r["pred"][a]["number"] is not None
                and r["pred"][a]["number"]==r["donor_native"]["opposite"][0]["number"] for r in rows)
        rates["donor_natural_k2"]=sum(r["pred"][a]["number"] is not None
                and r["pred"][a]["number"]==r["donor_native"]["opposite"][1]["number"] for r in rows)
        rates["unparsed"]=sum(r["pred"][a]["number"] is None for r in rows)
        rates["other"]=sum(r["pred"][a]["number"] is not None and not any(is_answer(r,a,b) for b in names.values())
                           for r in rows)
        rates["new_harm"]=sum(wrong(r,a) for r in rows)
        mass=[r["mass"][a] for r in rows if a in r["mass"]]
        if mass:
            rates["median_available_fraction"]=median(x/max(y,1e-15) for x,y in mass)
        out[a]=rates
    native={}
    for a in ("opposite","same"):
        native[a]=[sum(r["donor_native"][a][j]["number"]==r["donor_values"][a] for r in rows) for j in (0,1)]
    return dict(n=len(rows),arms=out,donor_native=native)


def core(repair,safe):
    assert [r["id"] for r in repair]==[s["id"] for s in safe]
    stats={"repair":measure(repair),"safe":measure(safe)}
    def cmp(a,b,which):
        return paired(repair,[int(is_answer(x,a,which))-int(is_answer(x,b,which)) for x in repair])

    def difference(a):
        return paired(repair,[int(wrong(x,a))-int(wrong(y,a)) for x,y in zip(repair,safe,strict=True)])

    p=dict(donor_unrestricted_vs_intact=cmp("opposite_full","opposite_intact","opposite"),
           donor_unrestricted_vs_matched=cmp("opposite_full","opposite_matched","opposite"),
           donor_capped_vs_matched=cmp("opposite_capped","opposite_matched","opposite"),
           receiver_original_vs_opposed_intact=cmp("native","opposite_intact","target"),
           original_capped_vs_matched_old=cmp("original_capped","original_matched","foil"),
           opposite_capped_vs_matched_old=cmp("opposite_capped","opposite_matched","foil"),
           opposite_full_vs_same_full_old=cmp("opposite_full","same_full","foil"),
           original_full_harm_vs_safe=difference("original_full"),
           opposite_full_harm_vs_safe=difference("opposite_full"),
           opposite_capped_harm_vs_safe=difference("opposite_capped"))
    def interaction(cut,clean,which):
        return paired(repair,[int(is_answer(x,"opposite_"+cut,which))-int(is_answer(x,"opposite_"+clean,which))
                             -int(is_answer(x,"original_"+cut,which))
                             +int(is_answer(x,"native" if clean=="intact" else "original_"+clean,which)) for x in repair])

    p["old_factorial_full"]=interaction("full","intact","foil")
    p["old_factorial_equal_mass"]=interaction("capped","matched","foil")
    p["donor_factorial_full"]=interaction("full","intact","opposite")
    p["donor_factorial_equal_mass"]=interaction("capped","matched","opposite")
    p["symmetric_control_exact_donor_increment"]=paired(repair,[
        int(is_answer(x,"opposite_capped","opposite"))-int(is_answer(x,"opposite_matched","opposite"))
        -int(is_answer(x,"same_capped","same"))+int(is_answer(x,"same_matched","same")) for x in repair])
    outcomes=stats["repair"]["arms"]
    gates={
        "donor_context":outcomes["opposite_full"]["opposite"]>=8 and all(positive(p[k]) for k in
            ("donor_unrestricted_vs_intact","donor_capped_vs_matched","opposite_full_harm_vs_safe")),
        "operand_reads":outcomes["original_full"]["old"]>=8 and all(positive(p[k]) for k in
            ("original_capped_vs_matched_old","original_full_harm_vs_safe")),
        "old_joint_interaction":outcomes["opposite_full"]["old"]>=8 and all(positive(p[k]) for k in
            ("old_factorial_full","old_factorial_equal_mass","opposite_full_harm_vs_safe")),
    }
    bounds={}
    for k in ("absolute","excess"):
        clusters=defaultdict(bool)
        for x in repair:
            test=is_answer(x,"opposite_full","foil")
            if k=="excess":
                test=test and not is_answer(x,"opposite_matched","foil")
            clusters[x["target"]]|=test
        n=len(clusters)
        e=sum(clusters.values())
        bounds[k]=dict(n=n,events=e,upper_97_5=upper_binomial(e,n))
    gates["exclude_large_old"]=bool(repair) and max(v["upper_97_5"] for v in bounds.values())<.1
    stats.update(paired=p,bounds=bounds,gates=gates)
    return stats


def summarize(rows):
    repair,safe=([r for r in rows if r["kind"]==k] for k in ("repair","safe"))
    z=core(repair,safe)
    selected={r["id"] for r in repair if all(v is not None and v<=2. for v in r["distance"])}
    z["margin_matched"]=core([x for x in repair if x["id"] in selected],[x for x in safe if x["id"] in selected])
    a=[r["component_diagnostics"] for r in repair if "component_diagnostics" in r]
    z["components"]=dict(n=len(a),median={arm:{k:median(x[arm][k]["shift"] for x in a) for k in vals}
                                                  for arm,vals in a[0].items()} if a else {})
    return z
