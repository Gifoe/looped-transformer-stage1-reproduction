"""Summarize the frozen structural target test and the paired deeper panel."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


TYPES=("float32","bfloat16")
FAMILIES=("tableau","twohop_open","twohop_options")
METHODS=("native","chart_anderson","chart","full_loop","branch_flow","prefix_ridge","ridge","persistence")
LABELS={"native":"Native","chart_anderson":"Anderson","chart":"Picard","full_loop":"Full-loop",
        "branch_flow":"Branch-flow","prefix_ridge":"Prefix ridge","ridge":"Token ridge","persistence":"Persistence"}
ARMS={"source_rule":(1,2),"random":(3,4),"unrelated":(5,6),"same_y":(7,8)}


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def stat(values):
    x=np.asarray(values)
    if not x.size:
        return None
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)))


def winner(z):
    k=int(np.argmax(z))
    return k if sum(x==z[k] for x in z)==1 else None


def bootstrap(values):
    a=np.array([np.mean(x) for x in values.values() if x])
    if not len(a):
        return None
    rng=np.random.default_rng(9511)
    draw=a[rng.integers(len(a),size=(5000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci=np.quantile(draw,[.025,.975]).tolist(),groups=len(a))


def ratio_by_group(events):
    if not events:
        return None
    vals=np.array(list(events.values()))
    rng=np.random.default_rng(733)
    s=vals[rng.integers(len(vals),size=(5000,len(vals)))].sum(1)
    return dict(ratio=float(vals[:,0].sum()/vals[:,1].sum()),
                interval=np.quantile(s[:,0]/s[:,1],[.025,.975]).tolist(),eligible_groups=len(vals))


def margins(z,y):
    return z[y]-max(t for j,t in enumerate(z) if j!=y)


def cells(rows,arm):
    for row in rows:
        native=row["methods"]["native"]
        for lane in ARMS[arm]:
            clean,change=winner(native["scores"][0]),winner(native["scores"][lane])
            desired=row["targets"][lane]
            match_clean=clean==row["true_clean"]
            if clean is None or change is None:
                outcome="tie"
            elif clean==change:
                outcome="unchanged"
            elif change==desired:
                outcome="to_target"
            else:
                outcome="to_other"
            yield row,lane,clean,change,desired,match_clean,outcome


def causal_summary(rows,arm):
    all_cases=list(cells(rows,arm))
    counts=Counter(q[-1] for q in all_cases)
    golden=sum(q[-1]=="to_target" and q[-2] for q in all_cases)
    native_output_doses=[q[0]["methods"]["native"]["native_dose"][q[1]] for q in all_cases]
    stats={}
    for name in METHODS:
        hits=Counter()
        delta=[]
        cos=[]
        input_doses=[]
        logit=[]
        signed=[]
        grouped_targets=defaultdict(lambda:[0,0])
        for row,i,clean,change,desired,match_clean,outcome in all_cases:
            m=row["methods"][name]
            a,b=winner(m["scores"][0]),winner(m["scores"][i])
            if a is None or b is None:
                hits["pred_tie"]+=1
            exact=a is not None and b is not None and a==clean and b==change
            if outcome in ("to_target","to_other"):
                hits["matched_any_native_flip"]+=exact
            if outcome=="to_target":
                hits["matched_to_target"]+=exact
                grouped_targets[row["group"]][0]+=exact
                grouped_targets[row["group"]][1]+=1
                if match_clean:
                    hits["matched_clean_correct_to_target"]+=exact
            if outcome=="unchanged":
                hits["spurious_flip"]+=(a is not None and b is not None and a!=b)
            hits["final_correct"]+=b==desired
            hits["clean_correct"]+=a==row["true_clean"]
            logit.append(m["logit_response_rmse"][i])
            signed.append(m["target_margin_response"][i])
            if m["native_dose"][i]>1e-10:
                delta.append(m["response_error"][i])
                cos.append(m["response_cos"][i])
            input_doses.append(row["input_dose"][i])
        stats[name]=dict(counts=dict(hits),ratio_to_target=ratio_by_group(grouped_targets),
                         logit_rmse=stat(logit),direction_cos=stat(cos),state_response_error=stat(delta),
                         target_margin_delta=stat(signed),dose=stat(input_doses))
    contrast={}
    for name in METHODS[2:]:
        values=defaultdict(list)
        for row,i,*_ in all_cases:
            m=row["methods"]
            values[row["group"]].append(m[name]["logit_response_rmse"][i]-m["chart_anderson"]["logit_response_rmse"][i])
        contrast[name]=bootstrap(values)
    return dict(cases=len(all_cases),native_counts=dict(counts),clean_correct_and_to_target=golden,
                methods=stats,paired_chart_improvement=contrast,native_output_effect=stat(native_output_doses))


def natural_summary(batch):
    out={}
    for method in METHODS:
        rows=[q["methods"][method] for q in batch]
        logits=[(a,b,g,w) for r,d in zip(batch,rows,strict=True)
                for a,b,g,w in zip(d["logits"],d["ref_logits"],r["gold"],r["initial_logits"],strict=True)]
        natural_flips=[(a,b) for a,b,_,w in logits if winner(w) is not None and winner(b) is not None
                       and winner(w)!=winner(b)]
        out[method]=dict(endpoint=stat([v for x in rows for v in x["error"]["full"]]),
                token_error=stat([v for x in rows for r in x["error"]["tokens"] for v in r]),
                direction=stat([v for x in rows for v in x["update_cos"]]),
                max_logit_error=stat([v for x in rows for v in x["max_logit_error"]]),
                agree=sum(winner(a) is not None and winner(a)==winner(b) for a,b,_,_ in logits),
                correct=sum(winner(a)==g for a,_,g,_ in logits),n=len(logits),
                natural_flips=len(natural_flips),natural_flips_hit=sum(winner(a)==winner(b) for a,b in natural_flips))
    return out


def intact_summary(context, rows):
    by_group={q["group"]:q for q in rows}
    pooled={}
    for group in context:
        observed=by_group[group["group"]]
        native=observed["methods"]["native"]["scores"]
        counters=pooled.setdefault(group["cohort"],Counter())
        clean=winner(group["native_unpack"][0])
        for i in (1,2):
            counters["n"]+=1
            y=group["targets"][i]
            unedited=winner(group["native_unpack"][i])
            edited=winner(native[i])
            counters["donor_correct"]+=unedited==y
            if clean==group["targets"][0] and unedited==y:
                counters["both_intact_prompts_correct"]+=1
                counters["patched_recipient_reaches_target"]+=edited==y
    return {f:dict(v) for f,v in pooled.items()}


def summarize(result):
    freeze=read(result/"FREEZE.json")
    s={}
    for dt in TYPES:
        stages={}
        context_path=result/f"DONOR_CONTEXT_{dt}.json"
        context=read(context_path) if context_path.exists() else {}
        timing=defaultdict(list)
        for depth in (2,3,4):
            raw={f:read(result/f"causal_{dt}_K{depth}_{f}.json") for f in FAMILIES}
            nat={f:read(result/f"natural_{dt}_K{depth}_{f}.json") for f in (*FAMILIES,"arc")}
            rows={f:d["rows"] for f,d in raw.items()}
            combined=sum(rows.values(),[])
            rows["all_structure"]=combined
            if depth==2:
                selected={g for f in FAMILIES for g in freeze["paired_depth_panel"][f]}
                rows["paired_depth_panel"]=[r for r in combined if r["group"] in selected]
            ca={k:{a:causal_summary(r,a) for a in ARMS} for k,r in rows.items()}
            same_groups=[r for r in combined if max(q["scaling"] for q in r["source_same_y"])<=4
                         and min(q["original_dose"] for q in r["source_same_y"])>1e-9]
            if same_groups:
                ca["same_y_scale_le4"]={"same_y":causal_summary(same_groups,"same_y"),
                                        "source_rule":causal_summary(same_groups,"source_rule")}
            stages[f"K{depth}"]=dict(causal=ca,natural={f:natural_summary(d["batches"]) for f,d in nat.items()},
                                    same_y_scale=stat([q["scaling"] for r in combined for q in r["source_same_y"]]))
            actual=[abs(r["input_dose"][7+j]-r["input_dose"][1+j])/max(r["input_dose"][1+j],1e-10)
                    for r in combined for j in range(2)]
            stages[f"K{depth}"]["same_y_dose_error"]=stat(actual)
            if f"K{depth}" in context:
                stages[f"K{depth}"]["intact_prompts"]=intact_summary(context[f"K{depth}"],combined)
            for v in (*raw.values(),*nat.values()):
                for q in v["timing"]:
                    timing[q["run"]].append(q)
        s[dt]=dict(loops=stages,timing={k:{p:stat([d[p] for d in v])
                      for p in ("wall_s","allocated_peak_bytes","attention_calls","mlp_calls","loop_field_calls")}
                      for k,v in timing.items()})
    return s


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("results",type=Path)
    a=p.parse_args()
    s=summarize(a.results)
    (a.results/"SUMMARY.json").write_text(json.dumps(s,indent=2,allow_nan=False)+"\n",encoding="utf-8")
