"""Recompute prospective group contrasts and native paired corrections."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from native import load, save, sha


def metric(values):
    a=np.asarray(values,dtype=float)
    if len(a)==0:
        return None
    if not np.isfinite(a).all():
        raise ValueError("Non-finite metric")
    rand=np.random.default_rng(21112117)
    bootstrap=a[rand.integers(len(a),size=(4000,len(a)))].mean(1)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),
                ci=np.quantile(bootstrap,[.025,.975]).tolist())


def scores(raw,how="mean"):
    return [float(np.mean(x) if how=="mean" else np.sum(x)) for x in raw]


def winner(s):
    m=max(s)
    return s.index(m) if s.count(m)==1 else None


def robust_margin(s,g,j):
    return s[g]-s[j]


def analyze_natural(raw,meta,how):
    assert [q["id"] for q in raw]==[q["id"] for q in meta]
    cases=[]
    for q,m in zip(raw,meta,strict=True):
        early,late=[scores(x,how) for x in q["native"]]
        true=m["gold"]
        fixed=max((j for j in range(len(late)) if j!=true),key=lambda j:late[j])
        w0,w1=winner(early),winner(late)
        code="".join("C" if p==true else "W" for p in (w0,w1))
        arm={}
        for k,d in q["arms"].items():
            s=scores(d["ll"],how)
            arm[k]=dict(hit=winner(s)==true,score=robust_margin(s,true,fixed),ood=d["norm_ratio"]>2)
        cases.append(dict(id=q["id"],stratum=code,native=[w0==true,w1==true],
                          native_margin=[robust_margin(x,true,fixed) for x in (early,late)],arm=arm))
    out=dict(n=len(cases),correct=[sum(r["native"][i] for r in cases) for i in (0,1)],
             populations={k:sum(q["stratum"]==k for q in cases) for k in ("WC","CW","CC","WW")},
             position0_mass=[metric([r["mean_position0"][k] for r in raw]) for k in range(3)],
             full_pattern_noop_max_error=max(r["same_pattern_parity"] for r in raw),
             seconds=sum(r["seconds"] for r in raw),panels={},cases=cases)
    names=cases[0]["arm"]
    for group in ("ALL","WC","CW","CC","WW"):
        a=cases if group=="ALL" else [x for x in cases if x["stratum"]==group]
        panel={}
        for k in names:
            i=0 if k=="early_add" else 1
            z=[r["arm"][k] for r in a]
            panel[k]=dict(n=len(z),correct=sum(r["hit"] for r in z),ood2x=sum(r["ood"] for r in z),
                         margin_vs_receiving_native=metric([x["score"]-r["native_margin"][i]
                                                           for x,r in zip(z,a,strict=True)]),
                         rescue=sum(x["hit"] and not r["native"][i] for x,r in zip(z,a,strict=True)),
                         damage=sum(not x["hit"] and r["native"][i] for x,r in zip(z,a,strict=True)))
        panel["contrasts"]={f"{x}_vs_{y}":metric([r["arm"][x]["score"]-r["arm"][y]["score"] for r in a])
                            for x,y in (("restore","drop"),("restore","restore_other"),("restore_pos0","drop"),
                             ("restore_rest","drop"),("restore","restore_pos0"),("rollback","random_first"),
                             ("rollback","random_second"),("rollback","other_heads"),("position0","rest"),
                             ("drop","matched_writer"),("drop","half_drop"))}
        panel["transitions"]=dict(
                  native_wc=len(a) if group=="WC" else sum(x["stratum"]=="WC" for x in a),
                  dropped_late_correct=sum(r["native"][1] and not r["arm"]["drop"]["hit"] for r in a),
                  restore_after_drop=sum(r["native"][1] and not r["arm"]["drop"]["hit"] and r["arm"]["restore"]["hit"] for r in a),
                  restore_after_drop_control_specific=sum(r["native"][1] and not r["arm"]["drop"]["hit"] and
                    r["arm"]["restore"]["hit"] and not r["arm"]["restore_other"]["hit"] for r in a),
                  rollback_damages=sum(r["native"][1] and not r["arm"]["rollback"]["hit"] for r in a),
                  exclusive_rollback_damage=sum(r["native"][1] and not r["arm"]["rollback"]["hit"] and
                       all(r["arm"][t]["hit"] for t in ("random_first","random_second","other_heads")) for r in a))
        out["panels"][group]=panel
    return out


def summarize_corners(y,t,other):
    z=[[scores(v) for v in row] for row in y]
    # d[r,c] is the effect of A1 vs A0 on the r-specific answer contrast in content c.
    d=[[((z[c][1][t[r][1]]-z[c][1][t[r][0]])-(z[c][0][t[r][1]]-z[c][0][t[r][0]]))
        for c in range(2)] for r in range(2)]
    i=.5*(d[0][0]+d[1][1]-d[0][1]-d[1][0])
    return dict(interaction=i,matched=[d[0][0],d[1][1]],mismatched=[d[0][1],d[1][0]],
                target_hits=sum(winner(z[c][s])==t[c][s] for s in range(2) for c in range(2)),
                donor_hits=sum(winner(z[c][s])==other[s] for s in range(2) for c in range(2)),
                strict_all_correct=all(winner(z[c][s])==t[c][s] for s in range(2) for c in range(2)),
                donor_margins=[z[c][s][t[c][s]]-z[c][s][other[s]] for s in range(2) for c in range(2)])


def structure(raw,meta):
    assert [q["id"] for q in raw]==[q["id"] for q in meta]
    r={}
    for name in ("all","rules","reverse"):
        entries=[(q,m) for q,m in zip(raw,meta,strict=True) if q["layout"]==name or name=="all"]
        panels={}
        e=[]
        for q,m in entries:
            a={k:summarize_corners(v,m["targets"],m["donor_targets"]) for k,v in q["conditions"].items()}
            a["native_selector_oracle"]=summarize_corners([[y[1] for y in x] for x in q["oracle"]],
                                               m["targets"],m["donor_targets"])
            e.append(a)
        for k in e[0]:
            a=[x[k] for x in e]
            panels[k]=dict(interaction=metric([v["interaction"] for v in a]),
                        matched=[metric([v["matched"][i] for v in a]) for i in range(2)],
                        mismatched=[metric([v["mismatched"][i] for v in a]) for i in range(2)],
                        target_hits=sum(v["target_hits"] for v in a),donor_hits=sum(v["donor_hits"] for v in a),
                        strict_all_correct=sum(v["strict_all_correct"] for v in a),
                        targets_vs_donors=metric([np.mean(v["donor_margins"]) for v in a]))
        for t,s in (("native_values","scramble_values"),("natural_receivers","scramble_receivers"),
                    ("natural_receivers","other_heads_receivers")):
            panels[t+"_improvement_over_"+s]=metric([a[t]["interaction"]-a[s]["interaction"] for a in e])
        r[name]=dict(n=len(e),native_dose=metric([q["dose"] for q,_ in entries]),
            maximal_scramble_distance_error=max(abs(q["dose"]-q["scramble_dose"]) for q,_ in entries),
            other_head_distance_adequate=sum(q["other_dose"]>=.95*q["dose"] for q,_ in entries),
            panels=panels,seconds=sum(q["seconds"] for q,_ in entries))
    return r


def main():
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("--partial",action="store_true")
    a=p.parse_args()
    o=a.out
    m=load(o/"NATURAL_MANIFEST.json")["cohorts"]
    n={}
    for f,items in m.items():
        path=o/f"NATIVE_{f}.json"
        if a.partial and not path.exists():
            continue
        q=load(path)
        assert q["manifest_sha256"]==sha(o/"NATURAL_MANIFEST.json")
        assert a.partial or len(q["rows"])==q["expected"]
        n[f]={h:analyze_natural(q["rows"],items["rows"][:len(q["rows"])],h) for h in ("mean","sum")}
    s={}
    p=o/"STRUCTURE_MANIFEST.json"
    if p.exists():
        rows=load(p)["rows"]
        for f in ("pilot","confirmation"):
            path=o/f"STRUCTURE_{f}.json"
            if a.partial and not path.exists():
                continue
            q=load(path)
            assert q["prediction_sha256"]==sha(o/"PREDICTIONS.json")
            expected=[r for r in rows if r["phase"]==f]
            assert a.partial or len(q["rows"])==len(expected)==q["expected"]
            if a.partial and len(q["rows"])<4:
                continue
            s[f]=structure(q["rows"],expected[:len(q["rows"])])
    decision=None
    if s.get("confirmation"):
        panels=s["confirmation"]["all"]["panels"]
        v=panels["native_values"]
        b=panels["native_values_improvement_over_scramble_values"]
        decision=dict(preregistered_gate=all(m["ci"][0]>0 for m in [*v["matched"],v["interaction"],b]) and
                    b["mean"]>=.02 and panels["natural_receivers"]["interaction"]["mean"]>0)
    save(o/("PARTIAL_SUMMARY.json" if a.partial else "SUMMARY.json"),
         dict(natural=n,structure=s,structured_decision=decision))


if __name__=="__main__":
    main()

