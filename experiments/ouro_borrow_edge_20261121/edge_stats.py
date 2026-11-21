"""Separate exact reversions, source restoration and exclusion under clustering."""
from collections import defaultdict
from math import comb
from statistics import median

from origin_stats import paired, positive

ARMS=("native","readcut","matched_read","geometry","restore_p","restore_m","restore_both",
      "restore_geometry","donor_borrow","donor_safe","donor_geometry","self_restore")


def upper_binomial(k,n,alpha=.025):
    if k==n:
        return 1.
    lo,hi=0.,1.
    for _ in range(64):
        x=(lo+hi)/2
        cdf=sum(comb(n,i)*x**i*(1-x)**(n-i) for i in range(k+1))
        if cdf>alpha:
            lo=x
        else:
            hi=x
    return hi


def exact(r,key,endpoint):
    return r["pred"][key]["number"]==r[endpoint]


def summarize(rows):
    result={}
    for part in ("repair","safe"):
        z=[r for r in rows if r["kind"]==part]
        by={}
        for k in ARMS:
            n=dict(correct=0,foil=0,unparsed=0,harm=0,restored=0,donor_exact=0,donor_foil=0,
                   donor_target_digit=0,donor_foil_digit=0)
            for r in z:
                v=r["pred"][k]["number"]
                clean=exact(r,"native","target")
                correct=exact(r,k,"target")
                n["correct"]+=correct
                n["foil"]+=exact(r,k,"foil")
                n["unparsed"]+=v is None
                n["harm"]+=clean and not correct
                n["restored"]+=clean and exact(r,"readcut","foil") and correct
                if r["aligned"] and k.startswith("donor"):
                    n["donor_exact"]+=v==r["donor"]["target"]
                    n["donor_foil"]+=v==r["donor"]["foil"]
                    if v is not None and 100<=v<=999:
                        n["donor_target_digit"]+=v//100==r["donor"]["target"]//100
                        n["donor_foil_digit"]+=v//100==r["donor"]["foil"]//100
            by[k]=n
        a=[r for r in z if r["aligned"]]
        response={}
        if a:
            for k in ("edge_write_norm","control_write_norm","propagation_shift","new_mlp_shift","new_attention_shift",
                      "K2_native_P","K2_native_M","K2_norm_outside"):
                response[k]=median(r[k] for r in a)
            response["true_mass"]=median(r["mass"][-1] for r in a)
            response["matched_mass"]=median(r["matched_mass"][-1] for r in a)
            response["mass_error"]=median(abs(r["mass"][-1]-r["matched_mass"][-1]) for r in a)
            response["control_fraction_within_001"]=sum(abs(r["mass"][-1]-r["matched_mass"][-1])<=.001 for r in a)/len(a)
            response["margin_shift"]={k:median(r["margin"][k]-r["margin"]["native"] for r in a) for k in ARMS}
        result[part]=dict(n=len(z),aligned=len(a),counts=by,median=response)
    data=[r for r in rows if r["kind"]=="repair"]
    strata={w:[r for r in data if r["width"]==w] for w in (3,4)}
    stats={}
    for tag,z in (("pooled",data),("three",strata[3]),("four",strata[4])):
        cmp={}
        counts={}
        for ctrl in ("matched_read","geometry"):
            differences=[int(exact(r,"readcut","foil"))-int(exact(r,ctrl,"foil")) for r in z]
            cmp["cut_vs_"+ctrl]=paired(z,differences)
            b=defaultdict(bool)
            for x,d in zip(z,differences,strict=True):
                b[x["target"]]|=d>0
            counts[ctrl]=dict(positive_targets=sum(b.values()),total_targets=len(b),
                              upper97_5=upper_binomial(sum(b.values()),len(b)))
        for a,b in (("restore_both","restore_geometry"),("donor_borrow","donor_safe"),
                    ("donor_borrow","donor_geometry")):
            gain=[(int(exact(r,a,"target"))-int(exact(r,b,"target"))) if exact(r,"readcut","foil") else 0 for r in z]
            cmp[f"{a}_vs_{b}"]=paired(z,gain)
        stats[tag]=dict(contrasts=cmp,upper_bounds=counts,n=len(z),
                        cut_foil=sum(exact(r,"readcut","foil") for r in z),aligned=sum(r["aligned"] for r in z))
    pooled=stats["pooled"]
    tests=pooled["contrasts"]
    enough=pooled["cut_foil"]>=8 and positive(tests["cut_vs_matched_read"]) and positive(tests["cut_vs_geometry"])
    result["gates"]=dict(read_edge=enough,
                         full_chain=enough and all(positive(tests[k]) for k in
                         ("restore_both_vs_restore_geometry","donor_borrow_vs_donor_safe","donor_borrow_vs_donor_geometry")),
                         exclude_ten_percent=pooled["aligned"]==pooled["n"] and
                         all(x["upper97_5"]<.10 for x in pooled["upper_bounds"].values()))
    result["strata"]=stats
    return result
