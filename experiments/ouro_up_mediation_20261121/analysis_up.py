"""Paired estimates for the two backgrounds of a proposed native mediator."""
import argparse
from pathlib import Path

import numpy as np

from analysis import metric,scores,winner
from native import load,save,sha

COHORTS=("ALL","WC","CC","CW","WW")
UP_CONTROLS=("up_norm_control","product_norm_control")


def native_cases(raw,m,how):
    assert [x["id"] for x in raw]==[x["id"] for x in m]
    out=[]
    for item,q in zip(raw,m,strict=True):
        truth=q["gold"]
        base={name:[scores(x,how) for x in row] for name,row in item["native"].items()}
        gold_foil=max((j for j in range(len(q["choices"])) if j!=truth),key=lambda j:base["clean"][1][j])
        def info(a):
            return dict(margin=a[truth]-a[gold_foil],hit=winner(a)==truth,pred=winner(a),candidates=a)
        native={k:[info(c) for c in v] for k,v in base.items()}
        arm={k:{a:info(scores(b["ll"],how)) for a,b in v.items()} for k,v in item["arms"].items()}
        code="".join("C" if y["hit"] else "W" for y in native["clean"])
        out.append(dict(id=q["id"],stratum=code,gold=truth,foil=gold_foil,native=native,arms=arm))
    return out


def summarize(cases):
    panels={}
    for pop in COHORTS:
        rows=cases if pop=="ALL" else [q for q in cases if q["stratum"]==pop]
        native={k:dict(correct=[sum(q["native"][k][i]["hit"] for q in rows) for i in range(2)],
                  mean_margin=[metric([q["native"][k][i]["margin"] for q in rows]) for i in range(2)])
                for k in cases[0]["native"]}
        a={}
        for b in ("drop","clean"):
            panel={}
            for path in cases[0]["arms"][b]:
                s=[(q["arms"][b][path],q["native"][b][1]) for q in rows]
                panel[path]=dict(effect=metric([x["margin"]-y["margin"] for x,y in s]),
                    correct=sum(x["hit"] for x,_ in s),harm=sum(y["hit"] and not x["hit"] for x,y in s),
                    rescue=sum(x["hit"] and not y["hit"] for x,y in s))
            a[b]=panel
        def m(q,s,k):
            return q["native"][s][k]["margin"]
        def d(q,b,k):
            return q["arms"][b][k]["margin"]
        advance={}
        for k in ("drop","half","permute_writer"):
            advance[k]={
                "K1_loss":metric([m(q,"clean",0)-m(q,k,0) for q in rows]),
                "K2_loss":metric([m(q,"clean",1)-m(q,k,1) for q in rows]),
                "increment_loss":metric([(m(q,"clean",1)-m(q,k,1))-(m(q,"clean",0)-m(q,k,0)) for q in rows])
            }
        advance["drop_vs_permute_writer"]={f"K{k+1}":metric([m(q,"permute_writer",k)-m(q,"drop",k) for q in rows])
                                           for k in range(2)}
        ctr={}
        for direction,context in ((1,"drop"),(-1,"clean")):
            ctr[context]={k:metric([direction*(d(q,context,"up")-d(q,context,k)) for q in rows])
                          for k in (*UP_CONTROLS,"gate","gate_and_up")}
            ctr[context]["own_base"]=metric([direction*(d(q,context,"up")-m(q,context,1)) for q in rows])
        elig=[q for q in rows if q["native"]["clean"][1]["hit"] and not q["native"]["drop"][1]["hit"]]
        strict_cases=[]
        reciprocal=[]
        for q in elig:
            rv=q["arms"]["clean"]
            fw=q["arms"]["drop"]
            if fw["up"]["hit"] and not rv["up"]["hit"]:
                reciprocal.append(q["id"])
                if all(not fw[x]["hit"] and rv[x]["hit"] for x in UP_CONTROLS):
                    strict_cases.append(q["id"])
        summary=dict(n=len(rows),native_correct_to_drop_wrong=len(elig),
                     restored={k:sum(q["arms"]["drop"][k]["hit"] for q in elig)
                               for k in cases[0]["arms"]["drop"]},
                     reversed={k:sum(not q["arms"]["clean"][k]["hit"] for q in elig)
                               for k in cases[0]["arms"]["clean"]},
                     both_direction_ids=reciprocal,both_directions_vs_all_controls_ids=strict_cases)
        world=metric([m(q,"clean",1)-d(q,"clean","up")-d(q,"drop","up")+m(q,"drop",1) for q in rows])
        panels[pop]=dict(n=len(rows),native=native,arms=a,contrasts=ctr,advance=advance,
                         discrete=summary,up_by_background_interaction=world)
    return panels


def decode(raw,meta):
    assert [q["id"] for q in raw]==[q["id"] for q in meta]
    cases=[]
    for item,m in zip(raw,meta,strict=True):
        truth=m["gold"]
        y={k:x["answer"] for k,x in item["condition"].items()}
        pop="".join("C" if y[k]==truth else "W" for k in ("K1","K2"))
        cases.append(dict(id=item["id"],pop=pop,parsed={k:x is not None for k,x in y.items()},
                          hit={k:x==truth for k,x in y.items()}))
    modes=list(cases[0]["hit"])
    elig=[q for q in cases if q["pop"]=="WC" and not q["hit"]["drop"]]
    return dict(n=len(cases),native_strata={k:sum(x["pop"]==k for x in cases) for k in COHORTS[1:]},
       accuracy={k:sum(q["hit"][k] for q in cases) for k in modes},
       unresolved={k:sum(not q["parsed"][k] for q in cases) for k in modes},
       native_wc_broken=len(elig),
       up_rescue_in_broken_wc=sum(q["hit"]["restore"] for q in elig),
       control_rescue_in_broken_wc=sum(q["hit"]["product_control"] for q in elig),
       strict_both=sum(q["hit"]["restore"] and not q["hit"]["product_control"] and not q["hit"]["reverse"] for q in elig),
       total_seconds=sum(x["block_seconds"] for x in raw))


def audit_numeric(raw):
    q=[a for r in raw for a in r["support"].values()]
    def relative(x,a,b):
        return abs(x[a]-x[b])/max(x[a],1e-10)
    ood={}
    for b in ("clean","drop"):
        ood[b]={k:sum(row["arms"][b][k]["state_norm_ratio"]>2 for row in raw) for k in raw[0]["arms"][b]}
    return dict(noop_max=max(x["noop_error"] for x in q),prompt_lane_max=max(x["native_candidate_lane_alignment_error"] for x in q),
                raw_dose_max_error=max(relative(x,"actual_raw_up","matched_raw_up") for x in q),
                product_dose_max_error=max(relative(x,"actual_product","matched_product") for x in q),
                geometry={k:metric([r["geometry"][k]["ratio"] for r in raw]) for k in ("up","gate","cross")}
                  | {"effective_product_cosine":metric([r["geometry"]["cosine_realized"] for r in raw])},
                carriers={k:metric([r["carrier"][k] for r in raw]) for k in raw[0]["carrier"]},
                activation_ood2x=ood,
                gpu_seconds=sum(r["seconds"] for r in raw))


def gate(primary):
    pools={}
    for f,d in primary.items():
        q=d["mean"]["ALL"]; w=d["mean"]["WC"]
        c=q["contrasts"]
        cases=w["discrete"]
        continuous={f"{b}/{k}":bool(c[b][k]["ci"][0]>0)
                    for b in ("clean","drop") for k in (*UP_CONTROLS,"own_base")}
        checks=dict(continuous=continuous, eligible_wc_at_least_12=cases["native_correct_to_drop_wrong"]>=12,
           restoration_exceeds_both_controls=all(cases["restored"]["up"]>cases["restored"][k] for k in UP_CONTROLS),
           dual_at_least_two=len(cases["both_direction_ids"])>=2,
           incremental_k2_loss=q["advance"]["drop"]["increment_loss"]["ci"][0]>0,
           specific_vs_gate=all(c[b]["gate"]["ci"][0]>0 for b in ("clean","drop")))
        pools[f]=checks
    return dict(families=pools,pass_all=all(all(v["continuous"].values()) and
                all(x for k,x in v.items() if k!="continuous") for v in pools.values()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("--partial",action="store_true")
    a=p.parse_args()
    o=a.out
    manifest=o/"MANIFEST.json"
    m=load(manifest)["cohorts"]
    up={}; gp={}
    for name,data in m.items():
        path=o/f"UP_{name}.json"
        if a.partial and not path.exists():
            continue
        x=load(path)
        raw=x["rows"]
        assert x["manifest_sha256"]==sha(manifest)
        assert a.partial or len(raw)==x["expected"]==data["n"]
        y={h:summarize(native_cases(raw,data["rows"][:len(raw)],h)) for h in ("mean","sum")}
        up[name]=dict(y,**audit_numeric(raw))
        g=o/f"DECODE_{name}.json"
        if a.partial and not g.exists():
            continue
        z=load(g)
        assert z["manifest_sha256"]==sha(manifest)
        assert a.partial or len(z["rows"])==128
        gp[name]=decode(z["rows"],data["rows"][:len(z["rows"])])
    decision=gate(up) if not a.partial else None
    save(o/("PARTIAL_SUMMARY.json" if a.partial else "SUMMARY.json"),
           dict(full_candidate=up,free_decode=gp,primary_decision=decision,
                publication_scope="question-level paired causal contrasts with fixed pre-intervention K2 foil"))


if __name__=="__main__":
    main()

