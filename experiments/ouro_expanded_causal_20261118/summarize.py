"""Analyze conditional behavior, group-level causal effects, and trace landmarks."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from collect import load_lines, write_json, digest
from analyze import interval
from run import predictions

TASKS=("relation","logic","arithmetic")
CONF=1-.05/3
ROOT_JOBS=tuple(f"main-{t}" for t in TASKS)+("A-relation","B-relation","C-logic","C-arithmetic")
COMP_JOBS=("main-logic","main-arithmetic","C-logic","C-arithmetic")
NAT=("calibration","benchmarks","controlled")


def to_csv(file, rows):
    with file.open("w",newline="",encoding="utf-8") as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def by_group(rows):
    out=defaultdict(list)
    for row in rows:
        out[row["id"]].append(row)
    return out


def accuracy(study,prefix):
    rows=[]
    for job in NAT:
        f=study/job/"rows.jsonl"
        if not (study/job/"complete.json").is_file():
            continue
        full=load_lines(f)
        for task in sorted({r["task"] for r in full}):
            tr=[r for r in full if r["task"]==task]
            for family in ["all",*sorted({r["family"] for r in tr if r["family"] is not None})]:
                fr=[r for r in tr if family=="all" or r["family"]==family]
                groups=by_group(fr)
                for k in range(4):
                    items=[r for r in fr if r["world"]=="r"]
                    middles=[r for r in fr if "mid_gold" in r]
                    rows.append(dict(study=prefix,stage=job,task=task,family=family,K=k+1,
                        groups=len(groups),worlds=len(fr),correct=sum(r["pred"][k]==r["gold"] for r in fr),
                        joint=sum(all(r["pred"][k]==r["gold"] for r in g) for g in groups.values()),
                        r_correct=sum(r["pred"][k]==r["gold"] for r in items),
                        ties=sum(r["pred"][k]==-1 for r in fr),intermediate_worlds=len(middles),
                        intermediate_correct=sum(r["mid_pred"][k]==r["mid_gold"] for r in middles),
                        joint_intermediate=sum(all(r["mid_pred"][k]==r["mid_gold"] for r in g) for g in groups.values()) if middles else None))
    return rows


def margin(table,k,goal,original):
    return table[k][goal]-table[k][original]


def native_eligible(group,k=3):
    return all(predictions(r["native"])[k]==predictions(r["donor_native"])[k]==r["original"]
               and predictions(r["input_cf_native"])[k]==r["counterfactual"] for r in group)


def statistics(groups,func,level):
    rates=[]
    for rr in groups.values():
        v=[func(r) for r in rr]
        assert len(v)==2
        rates.append(np.mean(v))
    p,a,b=interval(np.array(rates),level)
    return float(p),float(a),float(b)


def effect_row(label,job,task,region,subset,endpoint,contrast,k,groups,fn,level):
    est,lo,hi=statistics(groups,fn,level)
    return dict(study=label,experiment=job,task=task,region=region,subset=subset,
                endpoint=endpoint,contrast=contrast,K=k+1,groups=len(groups),
                estimate=est,lower=lo,upper=hi,level=level)


def source_rows(label,job,rows):
    task=rows[0]["task"]
    region_names=rows[0]["arms"]
    pop=by_group(rows)
    subsets={"all":pop}
    passed={g:r for g,r in pop.items() if native_eligible(r)}
    if len(passed)>=5:
        subsets["natural_four_correct_K4"]=passed
    for fam in sorted({r["family"] for r in rows}):
        subsets[f"family_{fam}"]={g:r for g,r in pop.items() if r[0]["family"]==fam}
    effects=[]
    decisions=[]
    cases=[]
    for region in region_names:
        for subset,groups in subsets.items():
            for k in range(4):
                for c in ("native","irrelevant","random"):
                    level=CONF if k==3 and subset=="all" and region=="payload" else .95
                    def delta(r):
                        a=r["arms"][region]["arms"]
                        comp=r["native"] if c=="native" else a[c]["scores"]
                        g,o=r["counterfactual"],r["original"]
                        return margin(a["semantic"]["scores"],k,g,o)-margin(comp,k,g,o)
                    effects.append(effect_row(label,job,task,region,subset,"final",f"semantic-minus-{c}",k,groups,delta,level))
                for c in ("native","random"):
                    def probe(r):
                        a=r["arms"][region]["intermediate"]
                        g,o=r["mid_counterfactual"],r["mid_original"]
                        return margin(a["semantic"]["scores"],k,g,o)-margin(a[c]["scores"],k,g,o)
                    effects.append(effect_row(label,job,task,region,subset,"intermediate",f"semantic-minus-{c}",k,groups,probe,.95))
            subset_rows=[r for group in groups.values() for r in group]
            for k in range(4):
                receipt=dict(study=label,experiment=job,task=task,region=region,subset=subset,K=k+1,
                            groups=len(groups),directed=len(subset_rows),
                            natural_expected=sum(predictions(r["native"])[k]==r["original"] for r in subset_rows),
                            natural_counterfactual_correct=sum(predictions(r["input_cf_native"])[k]==r["counterfactual"] for r in subset_rows),
                            flip_semantic=0,flip_irrelevant=0,flip_random=0)
                for arm in ("semantic","irrelevant","random"):
                    receipt[f"flip_{arm}"]=sum(predictions(r["native"])[k]==r["original"] and
                        r["arms"][region]["arms"][arm]["pred"][k]==r["counterfactual"] for r in subset_rows)
                decisions.append(receipt)
        def val(r):
            g,o=r["counterfactual"],r["original"]
            return margin(r["arms"][region]["arms"]["semantic"]["scores"],3,g,o)-margin(r["native"],3,g,o)
        ordered=sorted(rows,key=val)
        cases.append({"study":label,"job":job,"region":region,"selection":"median final semantic effect","row":ordered[len(rows)//2]})
        explaining=[r for r in rows if predictions(r["native"])[3]==predictions(r["donor_native"])[3]==r["original"]
                    and r["arms"][region]["arms"]["semantic"]["pred"][3]==r["counterfactual"]
                    and predictions(r["input_cf_native"])[3]==r["counterfactual"]]
        if explaining:
            cases.append({"study":label,"job":job,"region":region,"selection":"median of naturally correct flips",
                          "row":sorted(explaining,key=val)[len(explaining)//2]})
    return effects,decisions,cases


def b_pair(r,arm,k):
    g,o=r["counterfactual"],r["original"]
    v=r["arms"][arm]
    return margin(v["semantic"]["scores"],k,g,o)-margin(v["none"]["scores"],k,g,o)


def b_implicit(r,k,arm="injected"):
    g,o=r["counterfactual"],r["original"]
    candidate=r[arm] if arm=="injected" else r["arms"][arm]["scores"]
    return margin(candidate,k,g,o)-margin(r["native"],k,g,o)


def b_rows(label,job,rows):
    task=rows[0]["task"]
    groups=by_group(rows)
    arms=("explicit_open","k2_map","k2_direct","k2_both","k3_both","k2k3_both",
          "k2_both_restore_map","k2_both_restore_query")
    implicit=("injected","random","irrelevant","erase_k1_source","erase_k1_map","erase_k1_query","erase_k1_suffix")
    effects=[]
    for k in range(4):
        for arm in arms:
            effects.append(effect_row(label,job,task,"routes","all","final",arm,k,groups,lambda r:b_pair(r,arm,k),.95))
            if arm not in ("explicit_open","k2_both_restore_map","k2_both_restore_query"):
                level=CONF if k==3 and arm in ("k2_map","k2_direct","k2_both") else .95
                effects.append(effect_row(label,job,task,"routes","all","final",f"open-minus-{arm}",
                                          k,groups,lambda r:b_pair(r,"explicit_open",k)-b_pair(r,arm,k),level))
            elif arm.endswith("restore_map") or arm.endswith("restore_query"):
                effects.append(effect_row(label,job,task,"routes","all","final",f"{arm}-minus-k2_both",
                                          k,groups,lambda r:b_pair(r,arm,k)-b_pair(r,"k2_both",k),.95))
        for arm in implicit:
            effects.append(effect_row(label,job,task,"boundary","all","final",arm,k,groups,
                                      lambda r:b_implicit(r,k,arm),.95))
            if arm!="injected":
                effects.append(effect_row(label,job,task,"boundary","all","final",f"injected-minus-{arm}",
                                          k,groups,lambda r:b_implicit(r,k)-b_implicit(r,k,arm),.95))
        for c in ("native","random"):
            def mid(r):
                g,o=r["mid_counterfactual"],r["mid_original"]
                return margin(r["mid"]["semantic"],k,g,o)-margin(r["mid"][c],k,g,o)
            effects.append(effect_row(label,job,task,"payload","all","intermediate",f"semantic-minus-{c}",k,groups,mid,.95))
    mask_error=max(abs(a-b) for r in rows for k in range(4)
                   for a,b in zip(r["native"][k],r["arms"]["explicit_open"]["none"]["scores"][k]))
    cases=[]
    ordered=sorted(rows,key=lambda r:b_implicit(r,3))
    cases.append({"study":label,"job":job,"region":"routes","selection":"median injected K4 effect","row":ordered[len(rows)//2]})
    return effects,cases,mask_error


def matrix(study, effects, counts):
    rows=[]
    selected=("payload","binding","bridge","prefix")
    for r in effects:
        if (r["region"] not in selected or r["subset"]!="all" or r["endpoint"]!="final"
                or r["contrast"]!="semantic-minus-random" or r["K"]!=4):
            continue
        check=next((x for x in counts if x["study"]==r["study"] and
                    f'{x["job"]}-{x["task"]}'==r["experiment"]),None)
        if not check:
            continue
        contrasts=[x for x in effects if all(x[k]==r[k] for k in
                    ("study","experiment","task","region","subset","endpoint","K"))]
        mid=next(x for x in effects if all(x[k]==r[k] for k in
                 ("study","experiment","task","region","subset","K","contrast"))
                 and x["endpoint"]=="intermediate")
        item={"dataset":r["study"],"investigation":r["experiment"],"task":r["task"],
              "edited_region":r["region"],"groups":r["groups"],
              "naturally_correct_all_4_worlds_K4":check["naturally_capable_groups_K4"],
              "final_nats_vs_random":r["estimate"],"final_lower":r["lower"],"final_upper":r["upper"],
              "intermediate_nats_vs_random":mid["estimate"],"intermediate_lower":mid["lower"],
              "intermediate_upper":mid["upper"],"all_three_final_bounds_positive":all(x["lower"]>0 for x in contrasts),
              "interval_level":r["level"]}
        rows.append(item)
    if rows:
        to_csv(study/"cross_task_causal_matrix.csv",rows)


def summarize(study, partial=False):
    behavior=[]
    effects=[]
    decisions=[]
    cases=[]
    checks={}
    traces=[]
    done=[]
    screened=study/"screened"
    screened_jobs=()
    if (screened/"screen_complete.json").exists():
        chosen=json.loads((screened/"screen_complete.json").read_text())
        screened_jobs=tuple(f"C-{t}" for t,n in chosen["retained"].items() if n)
    datasets=(("expanded",study,ROOT_JOBS),("competence",study/"competence",COMP_JOBS),
              ("boolean_pilot",study/"boolean",()),("screened",screened,screened_jobs))
    for label,root,jobs in datasets:
        if not root.is_dir():
            continue
        behavior.extend(accuracy(root,label))
        for job in jobs:
            d=root/job
            if not (d/"complete.json").is_file():
                if partial:
                    continue
                raise RuntimeError(f"Missing completion receipt: {d}")
            rows=load_lines(d/"rows.jsonl")
            meta=json.loads((d/"complete.json").read_text())
            groups=by_group(rows)
            assert len(rows)==2*len(groups)==meta["rows"] and len(groups)==meta["groups"]
            assert all({r["receive"] for r in group}=={"r","d"} for group in groups.values())
            done.append({"study":label,**meta,"naturally_capable_groups_K4":sum(native_eligible(g) for g in groups.values())})
            if job=="B-relation":
                a,c,mask=b_rows(label,job,rows)
                cases.extend(c)
                effects.extend(a)
                checks["max_native_vs_neutral_explicit_mask_candidate_logp_error"]=mask
                selected={v:[r["trace"][v] for r in rows] for v in ("explicit_open","k2_both","implicit")}
            else:
                a,b,c=source_rows(label,job,rows)
                effects.extend(a)
                decisions.extend(b)
                cases.extend(c)
                selected={"target_payload":[r["trace"] for r in rows]}
            for route,bank in selected.items():
                for field,events in bank[0].items():
                    for event in events:
                        vals=np.array([x[field][event] for x in bank])
                        traces.append(dict(study=label,experiment=job,route=route,witness=field,
                                event=event,median=float(np.median(vals)),mean=float(vals.mean()),
                                directions=len(vals)))
    if behavior:
        to_csv(study/"natural_behavior.csv",behavior)
    if effects:
        to_csv(study/"causal_effects.csv",effects)
        matrix(study,effects,done)
    if decisions:
        to_csv(study/"decisions.csv",decisions)
    if traces:
        to_csv(study/"trajectory_witnesses.csv",traces)
    write_json(study/"case_witnesses.json",cases)
    write_json(study/"execution_summary.json",{"jobs":done,"checks":checks})
    print("Analyzed",len(done),"causal jobs,",len(effects),"contrasts",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("study",type=Path)
    p.add_argument("--partial",action="store_true")
    a=p.parse_args()
    summarize(a.study,a.partial)
